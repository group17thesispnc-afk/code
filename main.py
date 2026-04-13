"""
main.py

Smart Road Alert — Raspberry Pi Host Controller

Entry point for the Smart Road Alert host application.

Responsibilities:
    - Initialise and start the SerialManager (ESP32 ↔ RPi USB link).
    - Initialise and start the HC12Manager (RPi 1 ↔ RPi 2 peer-to-peer wireless).
    - Main thread: process inbound ESP32 and HC-12 messages, forward telemetry,
      and send periodic commands.
    - Graceful shutdown on SIGINT / SIGTERM.

Architecture (Fully Symmetric Peer-to-Peer):
    Both RPis run identical code with identical hardware. Each RPi:
    ✓ Has a local ESP32 (USB).
    ✓ Has an HC-12 radio module (UART /dev/ttyS0).
    ✓ Communicates bidirectionally with the peer RPi via HC-12.
    ✓ Processes and forwards telemetry from both local and remote sources.

    RPi 1              HC-12 433 MHz              RPi 2
    ├─ ESP32 [USB]  ────────────────────────  ESP32 [USB]
    └─ HC-12 Radio  ════════════════════════  HC-12 Radio

All serial communication is accessed exclusively through each manager's API.
This file does NOT import the 'serial' package directly.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Optional

import socket

try:
    from serial_config import SerialManager
except ImportError:  # pragma: no cover
    SerialManager = None  # type: ignore[assignment]

# ─── Logging Configuration ────────────────────────────────────────────────────


# ─── Logging to Console and File ─────────────────────────────────────────────
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "smart_road_alert.log.txt")
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger("SmartRoadAlert")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── Optional Camera Inference Imports ────────────────────────────────────────

try:
    import cv2
    import depthai as dai
    from ultralytics import YOLO as _YOLO
    _CAMERA_INFERENCE_AVAILABLE = True
except ImportError:
    cv2 = None        # type: ignore[assignment]
    dai = None        # type: ignore[assignment]
    _YOLO = None      # type: ignore[assignment]
    _CAMERA_INFERENCE_AVAILABLE = False

try:
    from gtts import gTTS as _gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _gTTS = None
    _GTTS_AVAILABLE = False

try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False

try:
    import customtkinter as ctk
    _CTK_AVAILABLE = True
except ImportError:
    ctk = None  # type: ignore[assignment]
    _CTK_AVAILABLE = False

# ─── Application Timing Constants ─────────────────────────────────────────────

# Whether to attempt connection to a local ESP32 device via USB.
# Both RPis have their own ESP32 attached, so this should be True on both.
ESP32_ENABLED: bool = SerialManager is not None

# Whether to start YOLOv8n camera inference (requires ultralytics + opencv).
CAMERA_INFERENCE_ENABLED: bool = True

# UI animations (fade/pulse) — disable by default to minimise distraction.
ANIMATIONS_ENABLED: bool = False

# DepthAI RGB preview size used by the normal runtime path.
CAMERA_PREVIEW_SIZE: tuple[int, int] = (640, 480)

# YOLO inference tuning for the normal runtime path.
YOLO_CONF_THRESHOLD: float = 0.45
YOLO_MAX_DET: int = 5

# How often to send a PING command to the ESP32 (seconds).
PING_INTERVAL_S: float   = 5.0

# How often to request a status report from the ESP32 (seconds).
STATUS_INTERVAL_S: float = 10.0

# Main-loop message-poll interval (seconds).  Short to minimise latency.
POLL_INTERVAL_S: float   = 0.05

# How often to send an HC-12 heartbeat ping to the remote RPi (seconds).
HC12_PING_INTERVAL_S: float = 5.0

# ─── Cross-Post Decision Timing ──────────────────────────────────────────────

# How long the winner side stays in a "GO hold" after a decision (seconds).
GO_HOLD_S: float = 15.0

# How long both sides display STOP before an arbitrated decision (seconds).
ARBITRATION_PAUSE_S: float = 3.0

# Idle announcement repeat interval when both sides detect no vehicle (seconds).
NO_VEHICLE_REPEAT_S: float = 20.0

# Freshness timeout for local/remote vehicle_state packets (seconds).
STATE_TIMEOUT_S: float = 1.5

# Telemetry cadence for vehicle_state packets (seconds).
VEHICLE_STATE_SEND_INTERVAL_S: float = 0.2  # 5 Hz

# Leader resends the latest decision at this interval during active states.
DECISION_RESEND_INTERVAL_S: float = 1.0

# How long the leader waits for remote acknowledgement before failing safe.
DECISION_ACK_TIMEOUT_S: float = 2.0

# Scenario 2 (one-side) loser shows remote info briefly, then clears display.
ONE_SIDE_LOSER_INFO_S: float = 2.0

# Node identifier used in peer-to-peer HC-12 messages (hostname-based).
NODE_ID: str = socket.gethostname()

# Dashboard GUI sizes itself based on the attached monitor resolution at runtime.

MODEL_PATH_ENV_VAR: str = "SMART_ROAD_MODEL_DIR"
DISTANCE_PRIORITY_ENABLED: bool = True
BBOX_DISTANCE_CONFIDENCE: int = 1

# ─── Vehicle Tracking & Speed Estimation Constants ────────────────────────────

# All vehicle classes the YOLO model can detect.
VEHICLE_CLASSES: list[str] = [
    "ambulance", "bicycle", "bus", "car", "ev_large", "ev_small",
    "fire_truck", "jeepney", "kalesa", "kariton", "motorcycle",
    "pedicab", "police_car", "tricycle", "truck", "tuktuk", "van",
]

# Emergency vehicle classes — trigger priority alerts.
_EMERGENCY_CLASSES: frozenset = frozenset({"ambulance", "fire_truck", "police_car"})

# Conversion factor: pixels (linear dimension) → metres.
# Tune based on camera mounting height and road width visible in frame.
METERS_PER_PIXEL: float = 0.03

# Minimum inter-frame time gap before speed is computed (seconds).
_MIN_TIME_DELTA: float = 0.2

# Minimum linear size change (pixels) to consider vehicle moving (noise floor).
# Raised from 2.0 to filter YOLO bbox jitter on stationary objects.
_MIN_SIZE_CHANGE: float = 6.0

# Minimum speed (km/h) to treat a detection as an "approaching vehicle".
# Helps ignore parked/stationary vehicles that YOLO can still detect.
_MIN_APPROACH_SPEED_KMH: float = 3.0

# Minimum bounding-box area (pixels²) — filters tiny / noisy detections.
_MIN_BBOX_AREA: int = 500

# Maximum centroid distance (pixels) to associate a detection with an existing track.
_MAX_CENTROID_DISTANCE: float = 120.0

# Minimum number of frames a track must be alive before telemetry is sent.
_MIN_STABLE_FRAMES: int = 3

# Seconds without a detection before a track is discarded.
_TRACK_TIMEOUT_S: float = 2.0

# Maximum history entries per track (deque maxlen).
_TRACK_HISTORY_LEN: int = 10

# ─── Speed Smoothing ──────────────────────────────────────────────────────────

# Exponential moving average factor.  Lower = smoother; higher = responsive.
_SPEED_EMA_ALPHA: float = 0.3

# Max speed measurements kept per track for variance / acceleration.
_SPEED_HISTORY_LEN: int = 20

# ─── Distance Estimation (Bbox Fallback) ──────────────────────────────────────

# Reference bbox area (px²) at 1 m.  distance ≈ sqrt(REF / area) metres.
# Tune per camera mount height and lens field-of-view.
_REF_BBOX_AREA_AT_1M: float = 50000.0

# ─── Emergency Detection Thresholds ──────────────────────────────────────────

# Relative speed margin over average traffic for emergency inference.
_EMERGENCY_RELATIVE_MARGIN_KMH: float = 15.0

# Minimum speed (km/h) fallback when traffic average is unknown.
_EMERGENCY_SPEED_FALLBACK_THRESHOLD: float = 40.0

# Minimum longitudinal acceleration (m/s²) for emergency inference.
_EMERGENCY_ACCEL_THRESHOLD: float = 1.5

# Speed standard-deviation (km/h) threshold for erratic-driving detection.
_EMERGENCY_VARIANCE_THRESHOLD: float = 4.0

# Minimum sustained time (s) above accel threshold before flagging emergency.
_EMERGENCY_ACCEL_SUSTAIN_S: float = 2.0

# Sliding window (s) for computing average road speed from non-emergency traffic.
_ROAD_SPEED_AVG_WINDOW_S: float = 10.0

# ─── Vehicle Size Classification ─────────────────────────────────────────────

_LARGE_VEHICLES: frozenset = frozenset({
    "bus", "truck", "van", "jeepney",
    "fire_truck", "ambulance",
})
_MEDIUM_VEHICLES: frozenset = frozenset({
    "pedicab",
    "tricycle", "ev_large", "tuktuk", "car", "kariton", "police_car", "kalesa",
})
_SMALL_VEHICLES: frozenset = frozenset({"bicycle", "motorcycle", "ev_small"})

# All other classes (including "none") are treated as NONE/UNKNOWN.


def category_rank(label: str) -> int:
    """Return 3/2/1/0 for LARGE/MEDIUM/SMALL/NONE based on the vehicle label."""
    if not label or label in ("none", "clear"):
        return 0
    if label in _LARGE_VEHICLES:
        return 3
    if label in _MEDIUM_VEHICLES:
        return 2
    if label in _SMALL_VEHICLES:
        return 1
    return 0


def category_name(label: str) -> str:
    """Return 'LARGE'/'MEDIUM'/'SMALL'/'NONE' for the given vehicle label."""
    r = category_rank(label)
    if r == 3:
        return "LARGE"
    if r == 2:
        return "MEDIUM"
    if r == 1:
        return "SMALL"
    return "NONE"

# ─── Alert Signal Thresholds ─────────────────────────────────────────────────

# Speed (km/h) above which a large incoming vehicle triggers STOP.
_STOP_SPEED_THRESHOLD: float = 60.0

# Speed (km/h) above which a medium incoming vehicle triggers GO SLOW.
_SLOW_SPEED_THRESHOLD: float = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# TTSManager — Non-blocking text-to-speech
# ─────────────────────────────────────────────────────────────────────────────

class TTSManager:
    """Non-blocking text-to-speech manager using gTTS.

    - Background worker thread processes a message queue.
    - Deduplicates: same message won't repeat within cooldown.
    - Rate-limits: minimum interval between any playback.
    """

    _MIN_INTERVAL_S: float = 1.0
    _SAME_MSG_COOLDOWN_S: float = 10.0

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue(maxsize=5)
        self._last_played: dict[str, float] = {}
        self._last_any: float = 0.0
        self._running: bool = True
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()
        logger.info("TTSManager: worker thread started.")

    def speak(self, text: str, force: bool = False) -> None:
        """Queue a TTS message (non-blocking, drops if queue full or throttled)."""
        now = time.monotonic()
        if not force:
            if now - self._last_any < self._MIN_INTERVAL_S:
                return
            last = self._last_played.get(text)
            if last is not None and now - last < self._SAME_MSG_COOLDOWN_S:
                return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3)
        logger.info("TTSManager: stopped.")

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            now = time.monotonic()
            self._last_played[text] = now
            self._last_any = now
            tmp_path: Optional[str] = None
            try:
                tts = _gTTS(text=text, lang='en')
                fd, tmp_path = tempfile.mkstemp(suffix='.mp3')
                os.close(fd)
                tts.save(tmp_path)
                subprocess.run(
                    ['mpg123', '-q', tmp_path],
                    timeout=15, capture_output=True, check=False,
                )
                logger.debug("TTS played: %s", text)
            except FileNotFoundError:
                logger.warning(
                    "TTS: 'mpg123' not found — install it for audio playback.")
            except Exception as exc:
                logger.warning("TTS playback error: %s", exc)
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass


class TTSManager:
    """Non-blocking text-to-speech manager with offline-first backends."""

    _MIN_INTERVAL_S: float = 1.0
    _SAME_MSG_COOLDOWN_S: float = 10.0

    @classmethod
    def detect_backend(cls) -> Optional[dict]:
        """Return the best available TTS backend for the current host."""
        for cmd in ("espeak-ng", "espeak", "spd-say"):
            path = shutil.which(cmd)
            if path:
                return {"kind": "offline_cli", "name": cmd, "path": path}

        if sys.platform == "darwin":
            path = shutil.which("say")
            if path:
                return {"kind": "offline_cli", "name": "say", "path": path}

        if os.name == "nt":
            path = shutil.which("powershell") or shutil.which("pwsh")
            if path:
                return {"kind": "powershell", "name": "powershell", "path": path}

        if _GTTS_AVAILABLE:
            for player in ("mpg123", "ffplay", "mpv", "cvlc", "vlc"):
                path = shutil.which(player)
                if path:
                    return {
                        "kind": "gtts",
                        "name": "gtts",
                        "path": path,
                        "player": player,
                    }
        return None

    @classmethod
    def is_available(cls) -> bool:
        return cls.detect_backend() is not None

    def __init__(self) -> None:
        self._backend = self.detect_backend()
        if self._backend is None:
            raise RuntimeError("No usable TTS backend detected.")

        self._queue: queue.Queue[str] = queue.Queue(maxsize=5)
        self._last_played: dict[str, float] = {}
        self._last_any: float = 0.0
        self._running: bool = True
        self._thread = threading.Thread(
            target=self._worker, name="tts-worker", daemon=True)
        self._thread.start()
        logger.info(
            "TTSManager: worker thread started using backend=%s.",
            self._backend["name"],
        )

    def speak(self, text: str, force: bool = False) -> None:
        """Queue a TTS message (non-blocking, drops if queue full or throttled)."""
        now = time.monotonic()
        if not force:
            if now - self._last_any < self._MIN_INTERVAL_S:
                return
            last = self._last_played.get(text)
            if last is not None and now - last < self._SAME_MSG_COOLDOWN_S:
                return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=3)
        logger.info("TTSManager: stopped.")

    def _run_player(self, player: str, media_path: str) -> None:
        if player == "mpg123":
            cmd = [player, "-q", media_path]
        elif player == "ffplay":
            cmd = [player, "-nodisp", "-autoexit", "-loglevel", "error", media_path]
        elif player == "mpv":
            cmd = [player, "--no-video", "--really-quiet", media_path]
        else:
            cmd = [player, "--play-and-exit", "--quiet", media_path]
        subprocess.run(cmd, timeout=20, capture_output=True, check=False)

    def _speak_offline_cli(self, text: str) -> None:
        backend = self._backend or {}
        cmd_name = backend.get("name", "")
        cmd_path = str(backend.get("path", ""))
        if cmd_name == "spd-say":
            cmd = [cmd_path, "-t", "female1", text]
        elif cmd_name == "say":
            cmd = [cmd_path, text]
        else:
            cmd = [cmd_path, "-s", "165", "-v", "en", text]
        subprocess.run(cmd, timeout=15, capture_output=True, check=False)

    def _speak_powershell(self, text: str) -> None:
        backend = self._backend or {}
        cmd_path = str(backend.get("path", "powershell"))
        escaped = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{escaped}')"
        )
        subprocess.run(
            [cmd_path, "-NoProfile", "-Command", script],
            timeout=20,
            capture_output=True,
            check=False,
        )

    def _speak_gtts(self, text: str) -> None:
        if not _GTTS_AVAILABLE:
            raise RuntimeError("gTTS is not available on this host.")

        backend = self._backend or {}
        player = str(backend.get("player", ""))
        if not player:
            raise RuntimeError("No audio player detected for gTTS.")

        tmp_path: Optional[str] = None
        try:
            tts = _gTTS(text=text, lang="en")
            fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            tts.save(tmp_path)
            self._run_player(player, tmp_path)
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _speak_once(self, text: str) -> None:
        backend = self._backend or {}
        kind = backend.get("kind", "")
        if kind == "offline_cli":
            self._speak_offline_cli(text)
            return
        if kind == "powershell":
            self._speak_powershell(text)
            return
        if kind == "gtts":
            self._speak_gtts(text)
            return
        raise RuntimeError(f"Unsupported TTS backend: {kind!r}")

    def _worker(self) -> None:
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            now = time.monotonic()
            self._last_played[text] = now
            self._last_any = now

            try:
                self._speak_once(text)
                logger.debug("TTS played: %s", text)
            except FileNotFoundError as exc:
                logger.warning("TTS playback dependency missing: %s", exc)
            except Exception as exc:
                logger.warning("TTS playback error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# DashboardGUI — Fullscreen roadside traffic display
# ─────────────────────────────────────────────────────────────────────────────

class DashboardGUI:
    """Fullscreen traffic display built with CustomTkinter.

    Design goals:
    - Looks like a real-world intelligent roadside LCD sign.
    - Massive colour-coded status dominates the screen (GO / SLOW / STOP).
    - Vehicle info shown below in large, high-contrast text.
    - Responsive: all sizes derived from screen dimensions, not fixed px.
    - Instant updates (no GUI animations) for clarity.
    """

    # ── Colour palette ──
    _GREEN  = "#00E050"
    _YELLOW = "#FFB800"
    _RED    = "#FF2222"
    _BG     = "#08090C"
    _FG     = "#E6EDF3"
    _DIM    = "#555E6A"
    _PANEL  = "#11151C"
    _BORDER = "#1C2333"

    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("Smart Road Alert")
        self.root.configure(fg_color=self._BG)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        # Fallback: size the window to the full screen even if fullscreen is unsupported.
        self.root.geometry(f"{sw}x{sh}+0+0")

        # ── Fullscreen, borderless, always-on-top ──
        self.root.attributes("-fullscreen", True)
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        self.root.bind("<Escape>", lambda _e: self.root.attributes(
            "-fullscreen", False))
        self.root.bind("<F11>", lambda _e: self.root.attributes(
            "-fullscreen",
            not self.root.attributes("-fullscreen")))

        # ── Derive sizes from screen resolution ──
        self._s = min(sw, sh)                  # reference dimension
        # Make GO/SLOW/STOP/NO VEHICLE as large as possible for roadside clarity.
        self._status_font_size  = max(int(self._s * 0.24), 92)
        self._vehicle_font_size = max(int(self._s * 0.07),  36)
        self._info_font_size    = max(int(self._s * 0.045), 24)
        self._header_font_size  = max(int(self._s * 0.025), 14)
        self._emerg_font_size   = max(int(self._s * 0.055), 30)
        self._params_font_size  = max(int(self._s * 0.020), 12)
        self._pad = max(int(self._s * 0.02), 10)

        # ── Image cache and source directory ──
        self._image_cache: dict = {}
        self._IMAGES_DIR = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "images")
        # Target bounding-box for vehicle thumbnails
        self._image_size = (
            max(int(sw * 0.15), 120),
            max(int(sh * 0.20), 100),
        )

        # ── Root grid:
        #     row 0 (weight 2) → top_frame    : image | label | speed
        #     row 1 (weight 0) → separator
        #     row 2 (weight 5) → middle_frame : STATUS (dominant)
        #     row 3 (weight 0) → separator
        #     row 4 (weight 1) → bottom_frame : EMERGENCY (hidden by default)
        self.root.grid_rowconfigure(0, weight=2)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=5)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ══════════════════════════════════════════════════════════════════
        # TOP FRAME — Row 1: vehicle image | label | speed+direction
        # ══════════════════════════════════════════════════════════════════
        top_frame = ctk.CTkFrame(self.root, fg_color=self._PANEL, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(0, weight=2)   # image cell
        top_frame.grid_columnconfigure(1, weight=3)   # label cell
        top_frame.grid_columnconfigure(2, weight=2)   # speed cell
        top_frame.grid_rowconfigure(0, weight=1)

        # Col 0 — vehicle image (or placeholder text when unavailable)
        self._vehicle_image_label = ctk.CTkLabel(
            top_frame, text="—",
            font=ctk.CTkFont(size=self._info_font_size),
            text_color=self._DIM)
        self._vehicle_image_label.grid(
            row=0, column=0, sticky="nsew",
            padx=self._pad, pady=self._pad)

        # Col 1 — vehicle type label (large bold)
        self._vehicle_label = ctk.CTkLabel(
            top_frame, text="—",
            font=ctk.CTkFont(size=self._vehicle_font_size, weight="bold"),
            text_color=self._FG)
        self._vehicle_label.grid(
            row=0, column=1, sticky="nsew",
            padx=self._pad, pady=self._pad)

        # Col 2 — speed (top) + direction (bottom) stacked
        speed_col = ctk.CTkFrame(top_frame, fg_color=self._PANEL, corner_radius=0)
        speed_col.grid(row=0, column=2, sticky="nsew",
                       padx=self._pad, pady=self._pad)
        speed_col.grid_rowconfigure(0, weight=2)
        speed_col.grid_rowconfigure(1, weight=1)
        speed_col.grid_columnconfigure(0, weight=1)

        self._speed_label = ctk.CTkLabel(
            speed_col, text="0  km/h",
            font=ctk.CTkFont(size=self._info_font_size, weight="bold"),
            text_color=self._DIM)
        self._speed_label.grid(row=0, column=0, sticky="nsew")

        self._direction_label = ctk.CTkLabel(
            speed_col, text="—",
            font=ctk.CTkFont(size=self._header_font_size, weight="bold"),
            text_color=self._DIM)
        self._direction_label.grid(row=1, column=0, sticky="nsew")

        # ── Thin horizontal divider ──
        ctk.CTkFrame(
            self.root, fg_color=self._BORDER, height=2, corner_radius=0
        ).grid(row=1, column=0, sticky="ew")

        # ══════════════════════════════════════════════════════════════════
        # MIDDLE FRAME — Row 2: GO / SLOW / STOP  (dominant)
        # ══════════════════════════════════════════════════════════════════
        self._status_frame = ctk.CTkFrame(
            self.root, fg_color=self._BG, corner_radius=0)
        self._status_frame.grid(row=2, column=0, sticky="nsew")
        self._status_frame.grid_rowconfigure(0, weight=1)
        self._status_frame.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            self._status_frame, text="GO",
            font=ctk.CTkFont(size=self._status_font_size, weight="bold"),
            text_color=self._GREEN)
        self._status_label.grid(row=0, column=0, sticky="nsew")

        # ── Thin horizontal divider ──
        ctk.CTkFrame(
            self.root, fg_color=self._BORDER, height=2, corner_radius=0
        ).grid(row=3, column=0, sticky="ew")

        # ══════════════════════════════════════════════════════════════════
        # BOTTOM FRAME — Row 3: EMERGENCY indicator (hidden by default)
        # ══════════════════════════════════════════════════════════════════
        self._emergency_frame = ctk.CTkFrame(
            self.root, fg_color="#1A0000", corner_radius=0)
        self._emergency_frame.grid(row=4, column=0, sticky="nsew")
        self._emergency_frame.grid_rowconfigure(0, weight=1)
        self._emergency_frame.grid_columnconfigure(0, weight=1)

        self._emergency_label = ctk.CTkLabel(
            self._emergency_frame,
            text="EMERGENCY MODE ACTIVE",
            font=ctk.CTkFont(size=self._emerg_font_size, weight="bold"),
            text_color=self._RED)
        self._emergency_label.grid(row=0, column=0, sticky="nsew")
        # Hidden until an emergency is signalled
        self._emergency_frame.grid_remove()

        # ── Mode banner (shown briefly after toggling NARROW/WIDE) ──
        self._mode_banner_after_id: Optional[str] = None
        self._mode_banner_frame = ctk.CTkFrame(
            self.root, fg_color=self._PANEL, corner_radius=14
        )
        self._mode_banner_label = ctk.CTkLabel(
            self._mode_banner_frame,
            text="",
            font=ctk.CTkFont(size=max(int(self._s * 0.05), 22), weight="bold"),
            text_color=self._FG,
        )
        self._mode_banner_label.pack(
            padx=int(self._pad * 1.2),
            pady=int(self._pad * 0.5),
        )
        self._mode_banner_frame.place(relx=0.5, rely=0.02, anchor="n")
        self._mode_banner_frame.place_forget()

        # ── Lower-right parameters panel (system logs / metrics) ──
        self._params_frame = ctk.CTkFrame(self.root, fg_color=self._PANEL, corner_radius=14)
        self._params_title = ctk.CTkLabel(
            self._params_frame,
            text="SYSTEM LOGS",
            font=ctk.CTkFont(size=self._header_font_size, weight="bold"),
            text_color=self._DIM,
        )
        self._params_title.pack(anchor="w", padx=self._pad, pady=(self._pad, int(self._pad * 0.2)))
        self._params_label = ctk.CTkLabel(
            self._params_frame,
            text="",
            font=ctk.CTkFont(size=self._params_font_size),
            text_color=self._FG,
            justify="left",
        )
        self._params_label.pack(anchor="w", padx=self._pad, pady=(0, self._pad))
        self._params_frame.place(relx=1.0, rely=1.0, anchor="se", x=-self._pad, y=-self._pad)

        # ── UI state ──
        self._cur_status: str = "GO"

        # Cached widget values to skip redundant redraws
        self._cache_vehicle: str = ""
        self._cache_speed: str = ""
        self._cache_direction: str = ""
        self._cache_emergency: Optional[bool] = None
        self._cache_params_text: str = ""

        logger.info("DashboardGUI: fullscreen display initialised (%dx%d).", sw, sh)

    # ──────────────────────────────────────────────────────────────────────
    # Public update API (called from _gui_poll on the main thread)
    # ──────────────────────────────────────────────────────────────────────

    def update_status(self, signal_text: str, emergency: bool = False) -> None:
        """Update the central status indicator (supports GO/SLOW/STOP/blank/idle)."""
        # Emergency row is handled separately; do not override the main signal.
        _ = emergency

        signal_text = signal_text or ""
        if signal_text == "":
            status_key = "BLANK"
            display_text = ""
            color = self._DIM
        elif signal_text == "NO VEHICLE":
            status_key = "IDLE"
            display_text = "NO VEHICLE"
            color = self._DIM
        elif signal_text == "STOP":
            status_key = "STOP"
            display_text = "STOP"
            color = self._RED
        elif signal_text in ("GO SLOW", "SLOW"):
            status_key = "SLOW"
            display_text = "SLOW"
            color = self._YELLOW
        else:
            status_key = "GO"
            display_text = "GO"
            color = self._GREEN

        if status_key == self._cur_status:
            return  # no change — skip redraw

        self._cur_status = status_key

        # Instant update (no animations).
        self._status_label.configure(text=display_text, text_color=color)

    def update_vehicle_info(self, label: str, speed: float,
                            direction: str) -> None:
        """Update the vehicle info panel (only redraws on change)."""
        v_text = label.upper().replace("_", " ") if label and label != "none" else "—"
        s_text = f"{speed:.0f}  km/h" if speed > 0 else "0  km/h"
        d_text = direction.upper() if direction and direction != "NONE" else "—"

        if v_text != self._cache_vehicle:
            self._cache_vehicle = v_text
            self._vehicle_label.configure(text=v_text)
            # Load and display vehicle thumbnail (cached, no I/O on repeat calls)
            clean_label = label.lower() if label and label not in ("none", "clear") else ""
            img = self._load_vehicle_image(clean_label) if clean_label else None
            if img is not None:
                self._vehicle_image_label.configure(image=img, text="")
            else:
                self._vehicle_image_label.configure(
                    image=None, text="\U0001F697" if v_text != "—" else "—")
        if s_text != self._cache_speed:
            self._cache_speed = s_text
            self._speed_label.configure(
                text=s_text,
                text_color=self._FG if speed > 0 else self._DIM)
        if d_text != self._cache_direction:
            self._cache_direction = d_text
            self._direction_label.configure(
                text=d_text,
                text_color=self._FG if d_text != "—" else self._DIM)

    # Backward-compatible aliases used by _gui_poll
    def update_local(self, label: str, speed: float, direction: str) -> None:
        pass  # display is unified — handled via update_vehicle_info

    def update_remote(self, label: str, speed: float, direction: str) -> None:
        pass  # display is unified — handled via update_vehicle_info

    def update_display(self, status: str, label: str, speed: float,
                       direction: str, emergency: bool = False) -> None:
        """Central GUI update — single entry point from SmartRoadAlertHost.

        Args:
            status:    GO / SLOW / STOP
            label:     vehicle class string (e.g. 'truck', 'car')
            speed:     speed in km/h
            direction: LEFT / RIGHT / FRONT / NONE
            emergency: True to trigger emergency pulse
        """
        self.update_status(status, emergency=emergency)
        self.update_vehicle_info(label, speed, direction)
        self.update_emergency(emergency)

    def update_emergency(self, emergency: bool) -> None:
        """Show or hide the emergency indicator row (no blinking/animation)."""
        if emergency == self._cache_emergency:
            return
        self._cache_emergency = emergency
        if emergency:
            self._emergency_frame.grid()
            self._emergency_label.configure(text_color=self._RED)
        else:
            self._emergency_frame.grid_remove()

    def show_mode_banner(self, mode: str) -> None:
        """Show MODE banner for 5 seconds (no animation)."""
        mode = (mode or "").upper()
        if mode not in ("NARROW", "WIDE"):
            return
        self._mode_banner_label.configure(text=f"MODE: {mode}")
        self._mode_banner_frame.place(relx=0.5, rely=0.02, anchor="n")
        if self._mode_banner_after_id is not None:
            try:
                self.root.after_cancel(self._mode_banner_after_id)
            except Exception:
                pass
            self._mode_banner_after_id = None

        def _hide() -> None:
            self._mode_banner_after_id = None
            try:
                self._mode_banner_frame.place_forget()
            except Exception:
                pass

        self._mode_banner_after_id = self.root.after(5000, _hide)

    def update_params_text(self, text: str) -> None:
        """Update the lower-right parameters panel (skips redraws on no change)."""
        text = text or ""
        if text == self._cache_params_text:
            return
        self._cache_params_text = text
        self._params_label.configure(text=text)

    def _load_vehicle_image(self, label: str) -> Optional[object]:
        """Return a cached ``ctk.CTkImage`` for *label*, or ``None``.

        Images are loaded once from ``images/<label>.jpg`` (or .jpeg/.png)
        and then stored in ``_image_cache``.  A ``None`` sentinel is stored
        on failure, preventing repeated failed disk lookups.
        """
        if not _PIL_AVAILABLE or not label:
            return None
        if label in self._image_cache:
            return self._image_cache[label]
        img_path: Optional[str] = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(self._IMAGES_DIR, label + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break
        if img_path is None:
            # Try generic fallback
            for ext in (".png", ".jpg"):
                candidate = os.path.join(self._IMAGES_DIR, "unknown" + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break
        if img_path is None:
            self._image_cache[label] = None
            return None
        try:
            pil_img = _PILImage.open(img_path).convert("RGBA")  # type: ignore[union-attr]
            # Proportional resize to fit thumbnail cell
            resample = getattr(_PILImage, "LANCZOS",
                               getattr(_PILImage, "ANTIALIAS", 1))
            pil_img.thumbnail(self._image_size, resample)
            ctk_img = ctk.CTkImage(
                light_image=pil_img,
                dark_image=pil_img,
                size=(pil_img.width, pil_img.height),
            )
            self._image_cache[label] = ctk_img
            return ctk_img
        except Exception:
            self._image_cache[label] = None
            return None

    def run(self) -> None:
        """Start the tkinter mainloop (blocks)."""
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# SmartRoadAlertHost
# ─────────────────────────────────────────────────────────────────────────────

class SmartRoadAlertHost:
    """
    Main application controller for the Smart Road Alert Raspberry Pi host.

    Each RPi independently:
    1. Reads telemetry from its local ESP32 via USB (SerialManager).
    2. Sends/receives telemetry to/from the peer RPi via HC-12 radio (HC12Manager).
    3. Processes alerts and forwards data bidirectionally.
    """

    def __init__(self) -> None:
        self._serial: Optional[SerialManager] = (
            SerialManager() if (ESP32_ENABLED and SerialManager is not None) else None
        )
        self._meters_per_pixel: float = METERS_PER_PIXEL
        self._ref_bbox_area_at_1m: float = _REF_BBOX_AREA_AT_1M
        self._emergency_relative_margin_kmh: float = _EMERGENCY_RELATIVE_MARGIN_KMH
        self._emergency_speed_fallback_threshold: float = _EMERGENCY_SPEED_FALLBACK_THRESHOLD
        self._distance_priority_enabled: bool = DISTANCE_PRIORITY_ENABLED
        self._model_path: Optional[str] = self._resolve_model_path()
        self._running: bool      = False
        self._t_last_ping        = 0.0
        self._t_last_status      = 0.0
        self._t_last_hc12_ping   = 0.0
        self._camera_thread: Optional[threading.Thread] = None
        self._camera_running: bool = False
        # ── Vehicle tracker state (accessed only from the camera-inference thread) ──
        self._tracks: dict = {}        # {track_id: track_dict}
        self._next_track_id: int = 0
        # ── Telemetry overlay state (written by camera thread, read by main thread) ──
        self._last_telemetry: dict = {
            "label":     "none",
            "speed":     0.0,
            "distance":  0.0,
            "direction": "none",
            "priority":  "LOW",
            "emergency": False,
            "last_seen": 0.0,
        }
        # ── TTS & GUI state ──────────────────────────────────────────────────
        self._tts: Optional[TTSManager] = None
        self._gui: Optional[DashboardGUI] = None
        self._display_lock = threading.Lock()
        self._local_display: dict = {
            "label": "none", "speed": 0.0, "direction": "NONE",
            "signal": "GO", "emergency": False,
        }
        self._remote_display: dict = {
            "label": "none", "speed": 0.0, "direction": "NONE",
            "signal": "GO", "emergency": False,
        }
        self._tts_last_signal: str = "GO"
        self._tts_last_remote_signal: str = ""
        # Thesis audio state (decision-driven, de-spammed).
        self._idle_started_mono: Optional[float] = None
        self._last_idle_spoken_mono: float = 0.0
        self._last_spoken_decision_id: int = 0
        self._last_announced_local_uid: int = -1
        self._last_surface_log_key: Optional[tuple] = None
        self._last_surface_log_mono: float = 0.0
        # ── GUI smoothing / cross-RPI timing ────────────────────────────────────
        # Monotonic timestamp of the last *real* (non-empty) remote vehicle packet.
        self._last_received_time: float = 0.0
        # How long to hold the last remote display state after telemetry stops.
        self._REMOTE_HOLD_S: float = 3.0
        # Monotonic timestamp of the last frame where local camera had an active alert.
        self._last_local_active_time: float = 0.0
        # How long to consider local detection "active" after the last confirmed frame.
        self._LOCAL_ACTIVE_HOLD_S: float = 2.0

        # ── Cross-post shared state (decision engine inputs) ──
        self._state_lock = threading.Lock()
        self._road_mode: str = "NARROW"  # toggled with 'S' and synced over HC-12
        self._mode_seq: int = 0

        self._road_speed_samples: deque = deque()  # [(speed_kmh, t_monotonic), ...]

        self._vehicle_state_seq: int = 0
        self._last_vehicle_state_sent: float = 0.0
        self._local_vehicle_state: dict = {
            "present": False,
            "label": "none",
            "category_rank": 0,
            "speed": 0.0,
            "distance": 0.0,
            "distance_confidence": 0,
            "emergency_active": False,
            "vehicle_uid": -1,
            "detected_at_ms": 0,
            "h_direction": "N",
            "seq": 0,
            "node": NODE_ID,
            "updated_mono": 0.0,
        }
        self._remote_vehicle_state: dict = {
            "present": False,
            "label": "none",
            "category_rank": 0,
            "speed": 0.0,
            "distance": 0.0,
            "distance_confidence": 0,
            "emergency_active": False,
            "vehicle_uid": -1,
            "detected_at_ms": 0,
            "det_to_tx_ms": None,
            "h_direction": "N",
            "seq": 0,
            "node": "",
            "received_mono": 0.0,
            "received_epoch_ms": 0,
            "ts_valid": False,
        }

        # ── Agreement / decision state (leader broadcasts, follower applies) ──
        self._decision_id_counter: int = 0
        self._current_decision: dict = {
            "decision_id": 0,
            "kind": "IDLE",
            "node_a": NODE_ID,
            "node_b": "",
            "signal_a": "",
            "signal_b": "",
            "emergency_mode": False,
            "traveling_node": "",
            "traveling_category_rank": 0,
            "traveling_emergency": False,
            "hold_until_mono": 0.0,
            "received_mono": 0.0,
            "acked_by_remote": False,
            "acked_remote_node": "",
            "acked_mono": 0.0,
        }
        self._last_decision_sent_mono: float = 0.0
        self._arbitration_start_mono: Optional[float] = None
        self._arbitration_context: Optional[tuple] = None

        # ── Lightweight dashboard metrics (shown in the GUI lower-right panel) ──
        self._metrics_lock = threading.Lock()
        self._inference_times_mono: deque[float] = deque(maxlen=30)
        self._inference_fps: float = 0.0
        self._last_inference_ms: float = 0.0

        self._hc12_ping_seq: int = 0
        self._hc12_pings_sent: int = 0
        self._hc12_pongs_recv: int = 0
        self._hc12_ping_pending: dict[int, float] = {}
        self._hc12_rtt_samples_ms: deque[float] = deque(maxlen=20)
        self._hc12_last_rtt_ms: Optional[float] = None
        self._hc12_last_rssi_dbm: Optional[float] = None

        self._timing_last_status: str = ""
        self._last_rx_to_alert_ms: Optional[float] = None
        self._last_alert_epoch_ms: int = 0
        self._last_params_update_mono: float = 0.0

        # Pi system-health snapshot (best-effort; shows N/A on non-Pi hosts).
        self._sys_last_check_mono: float = 0.0
        self._sys_temp_c: Optional[float] = None
        self._sys_throttled: Optional[int] = None

    def _resolve_model_path(self) -> Optional[str]:
        """Resolve the YOLO model path from env or the default local folder."""
        candidates = [
            os.environ.get(MODEL_PATH_ENV_VAR, ""),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_ncnn_model"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            resolved = candidate
            if not os.path.isabs(resolved):
                resolved = os.path.join(os.path.dirname(os.path.abspath(__file__)), resolved)
            if os.path.exists(resolved):
                return resolved
        return None

    def _log_runtime_health(self) -> None:
        """Log deployment-critical runtime readiness state."""
        logger.info(
            "Runtime health: camera=%s gui=%s serial=%s tts=%s",
            _CAMERA_INFERENCE_AVAILABLE,
            _CTK_AVAILABLE,
            self._serial is not None,
            TTSManager.is_available(),
        )
        logger.info(
            "Runtime health: camera defaults preview=%dx%d conf=%.2f max_det=%d distance=bbox-only.",
            CAMERA_PREVIEW_SIZE[0],
            CAMERA_PREVIEW_SIZE[1],
            YOLO_CONF_THRESHOLD,
            YOLO_MAX_DET,
        )
        if self._model_path is not None:
            logger.info("Runtime health: model path resolved to %s.", self._model_path)
        elif CAMERA_INFERENCE_ENABLED:
            logger.error(
                "Runtime health: model path not found. Set %s or provide best_ncnn_model.",
                MODEL_PATH_ENV_VAR,
            )

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise serial connections and enter the main processing loop."""
        logger.info("Smart Road Alert Host starting.")
        self._log_runtime_health()
        if self._serial is not None:
            self._serial.start()
        self._running = True
        if TTSManager.is_available():
            self._tts = TTSManager()
        else:
            logger.warning("TTS unavailable: no supported speech backend detected.")
        self.start_camera_inference()
        if _CTK_AVAILABLE:
            self._gui = DashboardGUI()
            self._gui.root.protocol("WM_DELETE_WINDOW", self._on_gui_close)
            # Mode toggle (per thesis spec).
            self._gui.root.bind("<KeyPress-s>", lambda _e: self._toggle_road_mode())
            self._gui.root.bind("<KeyPress-S>", lambda _e: self._toggle_road_mode())
            self._gui_poll()
            self._gui.run()  # blocks on mainloop
        else:
            self._run_loop()

    def stop(self) -> None:
        """Graceful shutdown — drain queues, stop threads, close ports."""
        logger.info("Smart Road Alert Host shutting down...")
        self._running = False
        self._camera_running = False
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=2)
        if self._serial is not None:
            self._serial.stop()
        if self._tts is not None:
            self._tts.stop()
        if _CAMERA_INFERENCE_AVAILABLE and cv2 is not None:
            cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

    # ─── Camera Inference ─────────────────────────────────────────────────────

    def start_camera_inference(self) -> None:
        """Start YOLOv8n camera inference in a non-blocking daemon thread.

        Does nothing if ``CAMERA_INFERENCE_ENABLED`` is ``False`` or if
        the required packages (``ultralytics``, ``opencv-python-headless``)
        are not installed.
        """
        if not CAMERA_INFERENCE_ENABLED:
            logger.info("Camera inference disabled by configuration.")
            return
        if not _CAMERA_INFERENCE_AVAILABLE:
            logger.warning(
                "Camera inference unavailable: install 'ultralytics' and "
                "'opencv-python-headless' to enable."
            )
            return
        if self._model_path is None:
            logger.error(
                "Camera inference unavailable: model path not found. "
                "Set %s or provide best_ncnn_model next to main.py.",
                MODEL_PATH_ENV_VAR,
            )
            return

        def camera_loop() -> None:
            # ── 1. Build RGB-only DepthAI v3 pipeline ──────────────────────
            try:
                oak_pipeline = dai.Pipeline()
                oak_cam = oak_pipeline.create(dai.node.Camera).build()
                videoOut = oak_cam.requestOutput(
                    CAMERA_PREVIEW_SIZE,
                    type=dai.ImgFrame.Type.BGR888p,
                )
                q = videoOut.createOutputQueue()
                oak_pipeline.start()
                logger.info(
                    "Camera inference: OAK RGB pipeline started at %dx%d.",
                    CAMERA_PREVIEW_SIZE[0],
                    CAMERA_PREVIEW_SIZE[1],
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Camera inference: failed to start OAK pipeline: %s", exc)
                return

            # ── 2. Auto-load ncnn model (relative to this script) ──────────
            try:
                model = _YOLO(self._model_path, task="detect")
                labels = model.names
                logger.info("Camera inference: ncnn model loaded from %s.", self._model_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Camera inference: failed to load ncnn model: %s", exc)
                oak_pipeline.stop()
                return

            # ── 3. Inference loop ──────────────────────────────────────────
            while self._camera_running and oak_pipeline.isRunning():
                # Pull the latest RGB frame from DepthAI.
                try:
                    pkt = q.tryGet() if hasattr(q, "tryGet") else q.get()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Camera inference: frame queue error: %s", exc)
                    time.sleep(0.1)
                    continue
                if pkt is None:
                    time.sleep(0.001)
                    continue
                frame = pkt.getCvFrame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                # ── 4. Run inference ───────────────────────────────────────
                try:
                    t_infer0 = time.monotonic()
                    try:
                        results = model(
                            frame,
                            verbose=False,
                            conf=YOLO_CONF_THRESHOLD,
                            max_det=YOLO_MAX_DET,
                        )
                    except TypeError:
                        results = model(frame, verbose=False)
                    t_infer1 = time.monotonic()
                except Exception as exc:  # noqa: BLE001
                    logger.error("Camera inference: inference error: %s", exc)
                    time.sleep(0.1)
                    continue

                infer_ms = (t_infer1 - t_infer0) * 1000.0
                with self._metrics_lock:
                    self._last_inference_ms = float(infer_ms)
                    self._inference_times_mono.append(t_infer1)
                    if len(self._inference_times_mono) >= 2:
                        span = self._inference_times_mono[-1] - self._inference_times_mono[0]
                        if span > 0.0:
                            self._inference_fps = (len(self._inference_times_mono) - 1) / span

                # ── 5. Collect detections ──────────────────────────────────
                detections: list[dict] = []
                raw_boxes = results[0].boxes
                for i in range(len(raw_boxes)):
                    conf = raw_boxes[i].conf.item()
                    if conf < YOLO_CONF_THRESHOLD:
                        continue
                    xyxy  = raw_boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy
                    cls_id    = int(raw_boxes[i].cls.item())
                    classname = labels[cls_id]
                    detections.append({
                        "label": classname,
                        "confidence": conf,
                        "bbox": (xmin, ymin, xmax, ymax),
                    })

                # ── 6. Track, compute kinematics, emit vehicle_state ───────
                self._on_inference_detections(detections)

                # ── Minimal sleep to prevent CPU saturation ────────────────
                time.sleep(0.001)

            oak_pipeline.stop()
            logger.info("Camera inference thread stopped.")

        self._camera_running = True
        self._camera_thread = threading.Thread(
            target=camera_loop, name="camera-inference", daemon=True
        )
        self._camera_thread.start()
        logger.info("Camera inference thread started.")

    def _on_inference_detections(
        self,
        detections: list,
    ) -> None:
        """Process per-frame detections: track vehicles, estimate direction and
        speed, and send structured telemetry over HC-12.

        Called from the camera-inference thread; all state it touches
        (``_tracks``, ``_next_track_id``) is private to that thread.

        Parameters
        ----------
        detections:
            List of dicts with keys ``"label"`` (str), ``"confidence"``
            (float), and ``"bbox"`` (xmin, ymin, xmax, ymax) for every
            bounding box detected in the current frame.
        """
        now = time.monotonic()

        # ── 1. Build candidate objects from raw detections ─────────────────
        candidates: list[dict] = []
        for det in detections:
            bbox = det.get("bbox")
            if bbox is None:
                continue
            xmin, ymin, xmax, ymax = bbox
            w = max(xmax - xmin, 1)
            h = max(ymax - ymin, 1)
            area = w * h
            if area < _MIN_BBOX_AREA:
                continue
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            candidates.append({
                "label":      det["label"],
                "confidence": det["confidence"],
                "bbox":       bbox,
                "cx":         cx,
                "cy":         cy,
                "area":       area,
                "size":       math.sqrt(area),  # linear proxy for speed
            })

        # ── 2. Nearest-centroid matching: candidates → existing tracks ──────
        matched_track_ids: set = set()
        unmatched: list[dict]  = []

        for obj in candidates:
            best_id:   Optional[int]   = None
            best_dist: float           = _MAX_CENTROID_DISTANCE

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                lx, ly = track["last_pos"]
                dist = math.sqrt((obj["cx"] - lx) ** 2 + (obj["cy"] - ly) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_id   = tid

            if best_id is not None:
                matched_track_ids.add(best_id)
                t = self._tracks[best_id]
                t["history"].append((obj["cx"], obj["cy"], obj["area"], obj["size"], now))
                t["last_seen"]   = now
                t["last_pos"]    = (obj["cx"], obj["cy"])
                t["confidence"]  = obj["confidence"]
                t["bbox"]        = obj["bbox"]
                t["frames"]      = t.get("frames", 0) + 1
            else:
                unmatched.append(obj)

        # ── 3. Spawn new tracks for unmatched candidates ───────────────────
        for obj in unmatched:
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = {
                "label":         obj["label"],
                "history":       deque(
                    [(obj["cx"], obj["cy"], obj["area"], obj["size"], now)],
                    maxlen=_TRACK_HISTORY_LEN,
                ),
                "speed_history": deque(maxlen=_SPEED_HISTORY_LEN),
                "last_seen":     now,
                "last_pos":      (obj["cx"], obj["cy"]),
                "confidence":    obj["confidence"],
                "bbox":          obj["bbox"],
                "smooth_speed":  0.0,
                "frames":        1,
                "first_seen_mono": now,
                "detected_at_ms":  int(time.time() * 1000),
                "accel_high_since": None,
            }

        # ── 4. Prune stale tracks ──────────────────────────────────────────
        stale = [tid for tid, t in self._tracks.items()
                 if now - t["last_seen"] > _TRACK_TIMEOUT_S]
        for tid in stale:
            del self._tracks[tid]

        # Reset speed state of alive-but-unmatched tracks so that if they are
        # re-matched in a later frame they do not inherit a stale smooth_speed.
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track["smooth_speed"] = 0.0
                track["speed_history"] = deque(maxlen=_SPEED_HISTORY_LEN)

        # ── 4b. No-detection telemetry (rate-limited to once per second) ──
        # Triggered as soon as NO detection matches this frame — do NOT wait
        # for the 2-second track-prune timeout.  Without this, the remote HUD
        # would keep showing a stale speed for up to 2 extra seconds after the
        # vehicle leaves YOLO's detection window.
        # ── Thesis-aligned vehicle_state selection + telemetry (v2 logic) ──
        # Select ONE "active approaching vehicle" (incoming only) using:
        # emergency_active > size > closest distance > highest speed > earliest first_seen.
        #
        # This replaces the legacy per-track "vehicle" telemetry and is the
        # single source of truth for the cross-post decision engine.

        # Prune average road-speed samples used for emergency inference.
        while self._road_speed_samples and now - self._road_speed_samples[0][1] > _ROAD_SPEED_AVG_WINDOW_S:
            self._road_speed_samples.popleft()

        best_state: Optional[dict] = None
        best_key: Optional[tuple] = None

        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                continue

            label = track["label"]
            hist  = track["history"]

            if len(hist) < _MIN_STABLE_FRAMES:
                continue

            kin = self._compute_track_kinematics(track)
            direction  = kin["direction"]
            speed_kmh  = float(kin["speed"])
            distance_m = float(kin["distance"])
            h_direction = self._compute_horizontal_direction(track["last_pos"][0])
            h_code = {"LEFT": "L", "RIGHT": "R", "FRONT": "F"}.get(h_direction, "N")

            # Update average-road-speed samples (incoming, non-emergency classes only).
            if direction == "incoming" and label not in _EMERGENCY_CLASSES and speed_kmh > 0.0:
                self._road_speed_samples.append((speed_kmh, now))
                while (
                    self._road_speed_samples
                    and now - self._road_speed_samples[0][1] > _ROAD_SPEED_AVG_WINDOW_S
                ):
                    self._road_speed_samples.popleft()

            avg_speed_kmh: Optional[float] = None
            if self._road_speed_samples:
                avg_speed_kmh = sum(s for s, _t in self._road_speed_samples) / len(self._road_speed_samples)

            emergency_active = self._is_emergency_active(track, kin, avg_speed_kmh, now)
            cat_rank = category_rank(label)

            if direction != "incoming":
                continue
            if speed_kmh < _MIN_APPROACH_SPEED_KMH and not emergency_active:
                # Ignore parked/stationary vehicles (common YOLO false positives).
                continue

            detected_at_ms = int(track.get("detected_at_ms", 0))
            distance_confidence = int(kin.get("distance_confidence", 0))
            key = (
                1 if emergency_active else 0,
                int(cat_rank),
                distance_confidence,
                -distance_m if distance_confidence > 0 else 0.0,
                speed_kmh,
                -detected_at_ms,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = {
                    "present": True,
                    "label": label,
                    "category_rank": int(cat_rank),
                    "speed": speed_kmh,
                    "distance": distance_m,
                    "distance_confidence": distance_confidence,
                    "emergency_active": bool(emergency_active),
                    "vehicle_uid": int(tid),
                    "detected_at_ms": detected_at_ms,
                    "h_direction": h_code,
                }

        if best_state is None:
            best_state = {
                "present": False,
                "label": "none",
                "category_rank": 0,
                "speed": 0.0,
                "distance": 0.0,
                "distance_confidence": 0,
                "emergency_active": False,
                "vehicle_uid": -1,
                "detected_at_ms": 0,
                "h_direction": "N",
            }

        with self._state_lock:
            self._local_vehicle_state.update({
                "present": best_state["present"],
                "label": best_state["label"],
                "category_rank": best_state["category_rank"],
                "speed": best_state["speed"],
                "distance": best_state["distance"],
                "distance_confidence": best_state["distance_confidence"],
                "emergency_active": best_state["emergency_active"],
                "vehicle_uid": best_state["vehicle_uid"],
                "detected_at_ms": best_state["detected_at_ms"],
                "h_direction": best_state["h_direction"],
                "updated_mono": now,
            })

        # Broadcast at a fixed cadence.
        if now - self._last_vehicle_state_sent >= VEHICLE_STATE_SEND_INTERVAL_S:
            self._last_vehicle_state_sent = now
            self._vehicle_state_seq += 1
            self.send_via_hc12({
                "type": "vehicle_state",
                "p": 1 if best_state["present"] else 0,
                "l": best_state["label"],
                "cr": int(best_state["category_rank"]),
                "s": round(float(best_state["speed"]), 1),
                "d": round(float(best_state["distance"]), 1),
                "dc": int(best_state["distance_confidence"]),
                "e": 1 if best_state["emergency_active"] else 0,
                "id": int(best_state["vehicle_uid"]),
                "t": int(best_state["detected_at_ms"]),
                "nm": int(time.time() * 1000),
                "q": int(self._vehicle_state_seq),
                "n": NODE_ID,
                "h": best_state["h_direction"],
            })

        return

    # ─── Kinematics Helpers ───────────────────────────────────────────────────

    def _compute_track_kinematics(self, track: dict) -> dict:
        """Return direction, smoothed speed, acceleration, variance, distance."""
        hist_list = list(track["history"])

        # Direction (area trend)
        first_areas = [e[2] for e in hist_list[:2]]
        last_areas  = [e[2] for e in hist_list[-2:]]
        direction = ("incoming"
                     if sum(last_areas) / len(last_areas)
                        > sum(first_areas) / len(first_areas)
                     else "outgoing")

        # Raw speed (linear-size change rate)
        f, l = hist_list[0], hist_list[-1]
        dt    = l[4] - f[4]
        dsize = abs(l[3] - f[3])
        if dt >= _MIN_TIME_DELTA and dsize >= _MIN_SIZE_CHANGE:
            raw_speed = (dsize / dt) * self._meters_per_pixel * 3.6
        else:
            raw_speed = 0.0

        # EMA smoothing.
        # IMPORTANT: if no movement was measured this frame (raw_speed == 0),
        # immediately zero the EMA instead of letting it decay slowly.  A
        # decaying-but-nonzero smooth_speed while no bbox growth is measured
        # is always a ghost artefact, not a real velocity reading.
        prev = track.get("smooth_speed", 0.0)
        if raw_speed == 0.0:
            smooth = 0.0
        elif prev == 0.0:
            smooth = raw_speed
        else:
            smooth = _SPEED_EMA_ALPHA * raw_speed + (1 - _SPEED_EMA_ALPHA) * prev
        track["smooth_speed"] = smooth

        # Speed history for acceleration / variance
        now = time.monotonic()
        sh = track.get("speed_history", deque(maxlen=_SPEED_HISTORY_LEN))
        sh.append((smooth, now))
        track["speed_history"] = sh

        # Acceleration (m/s²)
        accel = 0.0
        if len(sh) >= 2:
            s1, t1 = sh[0]
            s2, t2 = sh[-1]
            dt_h = t2 - t1
            if dt_h > 0.3:
                accel = ((s2 - s1) / 3.6) / dt_h

        # Speed variance (std dev km/h)
        variance = 0.0
        if len(sh) >= 3:
            speeds = [s for s, _ in sh]
            mean_s = sum(speeds) / len(speeds)
            variance = math.sqrt(sum((s - mean_s) ** 2 for s in speeds) / len(speeds))

        latest_area = hist_list[-1][2]
        distance = self._estimate_distance(latest_area)
        distance_confidence = BBOX_DISTANCE_CONFIDENCE

        return {
            "direction":      direction,
            "speed":          round(smooth, 1),
            "acceleration":   round(accel, 2),
            "speed_variance": round(variance, 1),
            "distance":       round(distance, 1),
            "distance_confidence": distance_confidence,
        }

    def _estimate_distance(self, bbox_area: float) -> float:
        """Approximate distance (m) from bounding-box area."""
        if bbox_area <= 0:
            return 999.0
        return math.sqrt(self._ref_bbox_area_at_1m / bbox_area)

    def _is_emergency_active(
        self,
        track: dict,
        kin: dict,
        avg_speed_kmh: Optional[float],
        now_mono: float,
    ) -> bool:
        """Infer an 'active emergency' using kinematic thresholds.

        The visual classifier cannot confirm siren/lights, so we infer emergency
        response only for emergency-labeled vehicles using:
          - Speed relative to traffic average (+15 km/h) with absolute fallback
          - Sustained acceleration ≥ 1.5 m/s² for ≥ 2 seconds
          - Speed variance (std dev) above threshold
        """
        label = track.get("label", "")
        if label not in _EMERGENCY_CLASSES:
            # Reset any emergency-specific state when the label is not emergency.
            track["accel_high_since"] = None
            return False

        speed_kmh = float(kin.get("speed", 0.0))
        accel_mps2 = float(kin.get("acceleration", 0.0))
        variance_kmh = float(kin.get("speed_variance", 0.0))

        # 1) Speed relative to traffic (fallback if avg is unknown/unreliable).
        if avg_speed_kmh is not None and avg_speed_kmh > 0.0:
            speed_crit = speed_kmh >= (avg_speed_kmh + self._emergency_relative_margin_kmh)
        else:
            speed_crit = speed_kmh >= self._emergency_speed_fallback_threshold

        # 2) Sustained acceleration.
        accel_crit = False
        since = track.get("accel_high_since")
        if accel_mps2 >= _EMERGENCY_ACCEL_THRESHOLD:
            if since is None:
                track["accel_high_since"] = now_mono
            else:
                accel_crit = (now_mono - float(since)) >= _EMERGENCY_ACCEL_SUSTAIN_S
        else:
            track["accel_high_since"] = None

        # 3) Speed variance.
        var_crit = variance_kmh >= _EMERGENCY_VARIANCE_THRESHOLD

        return bool(speed_crit or accel_crit or var_crit)

    @staticmethod
    def _get_priority(label: str, emergency_active: bool) -> str:
        if emergency_active:
            return "HIGH"
        if label in _LARGE_VEHICLES:
            return "HIGH"
        if label in _MEDIUM_VEHICLES:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _get_alert_signal(priority: str, direction: str,
                          speed: float, emergency_active: bool) -> str:
        if direction != "incoming":
            return "GO"
        if emergency_active:
            return "STOP"
        if priority == "HIGH":
            return "STOP" if speed >= _SLOW_SPEED_THRESHOLD else "GO SLOW"
        if priority == "MEDIUM":
            return "GO SLOW" if speed >= _SLOW_SPEED_THRESHOLD else "GO"
        return "GO"

    @staticmethod
    def _compute_horizontal_direction(cx: float, frame_width: float = 640.0) -> str:
        """Derive LEFT / RIGHT / FRONT from the bounding-box centroid x-position."""
        third = frame_width / 3.0
        if cx < third:
            return "LEFT"
        if cx > 2 * third:
            return "RIGHT"
        return "FRONT"

    def _tts_local_alert(self, label: str, speed: float,
                         h_dir: str, signal: str, emergency: bool) -> None:
        """Speak a local-detection alert when the overall signal changes."""
        if self._tts is None:
            return
        new_key = "EMERGENCY" if emergency else signal
        old_key = self._tts_last_signal
        if new_key == old_key:
            return
        self._tts_last_signal = new_key
        name = label.replace("_", " ")
        if emergency:
            self._tts.speak(f"Emergency! {name} approaching", force=True)
        elif signal == "STOP":
            self._tts.speak("Stop. Large vehicle approaching at high speed")
        elif signal in ("GO SLOW", "SLOW"):
            self._tts.speak(f"Slow down. {name} detected from {h_dir.lower()}")
        elif signal == "GO" and old_key in ("STOP", "GO SLOW", "SLOW", "EMERGENCY"):
            self._tts.speak("Road is clear")

    def _tts_remote_alert(self, label: str, speed: float,
                          signal: str, emergency: bool) -> None:
        """Speak a remote-telemetry alert when the remote signal changes."""
        if self._tts is None:
            return
        new_key = "EMERGENCY" if emergency else signal
        old_key = self._tts_last_remote_signal
        if new_key == old_key:
            return
        self._tts_last_remote_signal = new_key
        name = label.replace("_", " ")
        if emergency:
            self._tts.speak(
                f"Warning! Emergency {name} approaching from opposite side",
                force=True)
        elif signal == "STOP":
            self._tts.speak(f"Incoming {name} at high speed from opposite side")
        elif signal in ("GO SLOW", "SLOW"):
            self._tts.speak("Vehicle approaching from opposite side")
        elif signal == "GO" and old_key in ("STOP", "GO SLOW", "SLOW", "EMERGENCY"):
            self._tts.speak("Opposite side clear")

    def _send_display_command(self, label: str, signal: str, speed: float,
                              priority: str, emergency: bool) -> None:
        """Send a display-update command to the local ESP32."""
        if self._serial is None or not self._serial.is_connected():
            return
        cmd = json.dumps({
            "cmd": "display", "label": label, "signal": signal,
            "speed": round(speed, 1), "priority": priority,
            "emergency": emergency,
        }, separators=(",", ":"))
        self._serial.send(cmd)

    def _service_tick(self, now: float) -> None:
        """Shared runtime tick used by both GUI and headless execution."""
        if self._serial is not None:
            while True:
                msg = self._serial.receive()
                if msg is None:
                    break
                self._handle_esp32_message(msg)

        if self._serial is not None and self._serial.is_connected():
            if now - self._t_last_ping >= PING_INTERVAL_S:
                self._t_last_ping = now
                self._serial.send('{"cmd":"ping"}')
                logger.debug("Sent PING to ESP32.")

            if now - self._t_last_status >= STATUS_INTERVAL_S:
                self._t_last_status = now
                self._serial.send('{"cmd":"status"}')
                logger.debug("Sent STATUS request to ESP32.")

            if now - self._t_last_hc12_ping >= HC12_PING_INTERVAL_S:
                self._t_last_hc12_ping = now
                with self._metrics_lock:
                    self._hc12_ping_seq += 1
                    seq = int(self._hc12_ping_seq)
                    self._hc12_pings_sent += 1
                    self._hc12_ping_pending[seq] = now
                    # Prevent unbounded growth if the link is down.
                    stale = [k for k, t0 in self._hc12_ping_pending.items() if (now - float(t0)) > 30.0]
                    for k in stale:
                        self._hc12_ping_pending.pop(k, None)
                self.send_via_hc12({"type": "RPI_PING", "node": NODE_ID, "q": seq})
                logger.debug("Sent RPI_PING via HC-12 (node=%s q=%s).", NODE_ID, seq)

        self._leader_update_decision(now)

    @staticmethod
    def _is_decision_leader(decision: dict) -> bool:
        node_a = str(decision.get("node_a", ""))
        node_b = str(decision.get("node_b", ""))
        if not node_a:
            return False
        if not node_b:
            return NODE_ID == node_a
        return NODE_ID == min(node_a, node_b)

    @staticmethod
    def _decision_requires_ack(kind: str) -> bool:
        return kind in ("ONE_SIDE", "BOTH_GO", "BOTH_SLOW", "ARBITRATED", "EMERGENCY")

    def _agreement_pending(self, decision: dict, now_mono: float) -> tuple[bool, bool]:
        kind = str(decision.get("kind", "IDLE"))
        if not self._decision_requires_ack(kind):
            return False, False
        if not self._is_decision_leader(decision):
            return False, False
        if bool(decision.get("acked_by_remote", False)):
            return False, False
        pending_for = now_mono - float(decision.get("received_mono", 0.0))
        return True, pending_for >= DECISION_ACK_TIMEOUT_S

    def _agreement_safe_status(self, decision: dict, timed_out: bool) -> str:
        kind = str(decision.get("kind", "IDLE"))
        if timed_out:
            return "STOP"
        if kind in ("ARBITRATED", "EMERGENCY", "ARBITRATING"):
            return "STOP"
        return ""

    def _log_surface_state(self, disp: dict, now_mono: float) -> None:
        key = (
            disp.get("status", ""),
            disp.get("label", ""),
            round(float(disp.get("speed", 0.0)), 1),
            disp.get("direction", ""),
            bool(disp.get("emergency", False)),
        )
        if key == self._last_surface_log_key and now_mono - self._last_surface_log_mono < 1.0:
            return
        self._last_surface_log_key = key
        self._last_surface_log_mono = now_mono
        logger.info(
            "Surface state: status=%s label=%s speed=%.1f direction=%s emergency=%s",
            disp.get("status", ""),
            disp.get("label", ""),
            float(disp.get("speed", 0.0)),
            disp.get("direction", ""),
            bool(disp.get("emergency", False)),
        )

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Main thread processing loop.

        1. Drain the inbound message queue from SerialManager.
        2. Send periodic PING commands.
        3. Send periodic STATUS requests.
        """
        logger.info("Main loop running. Press Ctrl+C to stop.")

        # Pre-create the display window in the main thread before the loop so
        # that the first cv2.imshow() call does not have to initialise the
        # window mid-loop (avoids black-frame on some X11 / Wayland compositors).
        if False and CAMERA_INFERENCE_ENABLED and _CAMERA_INFERENCE_AVAILABLE and cv2 is not None:
            cv2.namedWindow("Smart Road Alert — YOLO", cv2.WINDOW_NORMAL)

        while self._running:
            now = time.monotonic()
            self._service_tick(now)
            disp = self._thesis_display_state(now)
            self._audio_tick(now, disp)
            self._log_surface_state(disp, now)

            # ── Process all queued inbound messages from the local ESP32 (if enabled) ──

            # ── Periodic commands to ESP32 (only while connected) ──
            if False and self._serial is not None and self._serial.is_connected():

                if now - self._t_last_ping >= PING_INTERVAL_S:
                    self._t_last_ping = now
                    self._serial.send('{"cmd":"ping"}')
                    logger.debug("Sent PING to ESP32.")

                if now - self._t_last_status >= STATUS_INTERVAL_S:
                    self._t_last_status = now
                    self._serial.send('{"cmd":"status"}')
                    logger.debug("Sent STATUS request to ESP32.")

                # ── HC-12 heartbeat ping to remote RPi ──────────────────────
                if now - self._t_last_hc12_ping >= HC12_PING_INTERVAL_S:
                    self._t_last_hc12_ping = now
                    self.send_via_hc12({"type": "RPI_PING", "node": NODE_ID})
                    logger.debug("Sent RPI_PING via HC-12 (node=%s).", NODE_ID)

            # ── Display latest camera frame on the main thread (GUI calls must
            #    not run from a daemon thread — black frames are a threading issue)
            if False and CAMERA_INFERENCE_ENABLED and _CAMERA_INFERENCE_AVAILABLE and cv2 is not None:
                # Take a copy under the lock so the camera thread is free to
                # write the next frame without blocking the renderer.
                with self._frame_lock:
                    frame = (
                        self._latest_frame.copy()
                        if self._latest_frame is not None
                        else None
                    )


                if frame is not None:
                    frame_display = cv2.resize(
                        frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                    # ── Telemetry overlay (data received from remote via HC-12) ──
                    # Auto-reset HUD after 5 s of no HC-12 receive
                    if time.time() - self._last_telemetry["last_seen"] > 5.0:
                        self._last_telemetry.update({
                            "label": "none", "speed": 0.0,
                            "distance": 0.0, "direction": "none",
                            "priority": "LOW", "emergency": False,
                        })
                    tel = self._last_telemetry
                    em  = tel["emergency"]
                    col = (0, 0, 255) if em else (0, 255, 255)
                    cv2.rectangle(frame_display, (5, 5), (440, 160),
                                  (0, 0, 0), cv2.FILLED)
                    cv2.rectangle(frame_display, (5, 5), (440, 160),
                                  col, 1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    lbl_text = tel["label"]
                    if em:
                        lbl_text = "!! " + lbl_text + " EMERGENCY"
                    cv2.putText(frame_display,
                                f"Label   : {lbl_text}",
                                (12, 35), font, 0.75, col, 2)
                    cv2.putText(frame_display,
                                f"Speed   : {tel['speed']:.1f} km/h",
                                (12, 70), font, 0.75, col, 2)
                    cv2.putText(frame_display,
                                f"Distance: {tel['distance']:.1f} m",
                                (12, 105), font, 0.75, col, 2)
                    cv2.putText(frame_display,
                                f"Direction: {tel['direction']}",
                                (12, 140), font, 0.75, col, 2)
                    surface_color = (255, 255, 255)
                    if disp["status"] == "STOP":
                        surface_color = (0, 0, 255)
                    elif disp["status"] in ("SLOW", "GO SLOW"):
                        surface_color = (0, 215, 255)
                    elif disp["status"] == "GO":
                        surface_color = (0, 255, 0)
                    cv2.putText(
                        frame_display,
                        f"SURFACE: {disp['status'] or 'WAIT'}",
                        (12, 185),
                        font,
                        0.85,
                        surface_color,
                        2,
                    )
                    cv2.putText(
                        frame_display,
                        f"REMOTE: {disp['label']} {disp['speed']:.1f} km/h",
                        (12, 220),
                        font,
                        0.7,
                        surface_color,
                        2,
                    )
                    # ──────────────────────────────────────────────────────

                    cv2.imshow("Smart Road Alert — YOLO", frame_display)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    self._camera_running = False
                    self._running = False
            else:
                time.sleep(POLL_INTERVAL_S)

    # ─── GUI Polling (CustomTkinter mainloop callbacks) ───────────────────────

    @staticmethod
    def _fmt_epoch_ms(epoch_ms: int) -> str:
        """Format epoch milliseconds as HH:MM:SS.mmm (local time)."""
        if not epoch_ms:
            return "N/A"
        try:
            t = time.localtime(epoch_ms / 1000.0)
            return time.strftime("%H:%M:%S", t) + f".{int(epoch_ms) % 1000:03d}"
        except Exception:
            return str(epoch_ms)

    def _update_alert_timing(self, now_mono: float, disp: dict) -> None:
        """Capture RX->Alert timing on the rising edge of STOP/SLOW."""
        status = str(disp.get("status", ""))
        alert_statuses = {"STOP", "SLOW", "GO SLOW"}

        with self._metrics_lock:
            prev_status = self._timing_last_status
            self._timing_last_status = status

        is_alert = status in alert_statuses
        was_alert = prev_status in alert_statuses
        if not (is_alert and not was_alert):
            return

        _local, remote, _road_mode, _decision = self._snapshot_vehicle_states(now_mono)
        rx_to_alert_ms: Optional[float] = None
        if bool(remote.get("present", False)):
            rx_mono = float(remote.get("received_mono", 0.0) or 0.0)
            if rx_mono > 0.0:
                rx_to_alert_ms = max(0.0, (now_mono - rx_mono) * 1000.0)

        with self._metrics_lock:
            self._last_rx_to_alert_ms = rx_to_alert_ms
            self._last_alert_epoch_ms = int(time.time() * 1000)

    def _update_system_stats(self, now_mono: float) -> None:
        """Best-effort Raspberry Pi health stats (temp + undervoltage flags).

        These values are only available on Raspberry Pi OS; on other hosts they
        remain as N/A without breaking runtime.
        """
        if now_mono - float(self._sys_last_check_mono) < 2.0:
            return
        self._sys_last_check_mono = now_mono

        # CPU temperature (millidegrees C → degrees C)
        temp_c: Optional[float] = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r", encoding="ascii") as fh:
                raw = fh.read().strip()
            if raw:
                temp_c = int(raw) / 1000.0
        except Exception:
            temp_c = None
        self._sys_temp_c = temp_c

        # Undervoltage/throttle flags.
        throttled: Optional[int] = None
        try:
            out = subprocess.check_output(
                ["vcgencmd", "get_throttled"],
                timeout=1.0,
                stderr=subprocess.DEVNULL,
            )
            s = out.decode("utf-8", errors="ignore").strip()
            if "throttled=" in s:
                val = s.split("throttled=", 1)[1].strip()
                throttled = int(val, 16)
        except Exception:
            throttled = None
        self._sys_throttled = throttled

    def _build_dashboard_params_text(self, now_mono: float) -> str:
        """Build the lower-right parameters text shown on the GUI dashboard."""
        self._update_system_stats(now_mono)
        local, remote, road_mode, _decision = self._snapshot_vehicle_states(now_mono)

        with self._metrics_lock:
            fps = float(self._inference_fps)
            infer_ms = float(self._last_inference_ms)
            pings_sent = int(self._hc12_pings_sent)
            pongs_recv = int(self._hc12_pongs_recv)
            last_rtt = self._hc12_last_rtt_ms
            rssi_dbm = self._hc12_last_rssi_dbm
            rtt_samples = list(self._hc12_rtt_samples_ms)
            rx_to_alert_ms = self._last_rx_to_alert_ms
            alert_epoch_ms = int(self._last_alert_epoch_ms)

        pdr_pct: Optional[float] = None
        if pings_sent > 0:
            pdr_pct = (pongs_recv / pings_sent) * 100.0

        jitter_ms: Optional[float] = None
        if len(rtt_samples) >= 2:
            diffs = [abs(rtt_samples[i] - rtt_samples[i - 1]) for i in range(1, len(rtt_samples))]
            if diffs:
                jitter_ms = sum(diffs) / len(diffs)

        det_to_tx_ms = remote.get("det_to_tx_ms")
        det_to_tx_str = "N/A"
        if isinstance(det_to_tx_ms, (int, float)) and float(det_to_tx_ms) >= 0.0:
            det_to_tx_str = f"{float(det_to_tx_ms):.0f}ms"

        rtt_str = f"{float(last_rtt):.0f}ms" if isinstance(last_rtt, (int, float)) else "N/A"
        jitter_str = f"{float(jitter_ms):.0f}ms" if isinstance(jitter_ms, (int, float)) else "N/A"
        pdr_str = f"{float(pdr_pct):.0f}%" if isinstance(pdr_pct, (int, float)) else "N/A"
        rssi_str = f"{float(rssi_dbm):.0f} dBm" if isinstance(rssi_dbm, (int, float)) else "N/A"
        rx_alert_str = f"{float(rx_to_alert_ms):.0f}ms" if isinstance(rx_to_alert_ms, (int, float)) else "N/A"

        # Detection metrics require ground-truth labels; keep accurate by showing N/A.
        det_metrics = "P/R/F1/mAP: N/A"

        # Speed estimation values are directly from the algorithm (vehicle_state).
        local_speed = float(local.get("speed", 0.0) or 0.0)
        remote_speed = float(remote.get("speed", 0.0) or 0.0)

        temp_str = f"{float(self._sys_temp_c):.1f}C" if isinstance(self._sys_temp_c, (int, float)) else "N/A"
        uv_now = None
        uv_past = None
        if isinstance(self._sys_throttled, int):
            uv_now = bool(self._sys_throttled & 0x1)
            uv_past = bool(self._sys_throttled & 0x10000)
        uv_now_str = "YES" if uv_now is True else ("NO" if uv_now is False else "N/A")
        uv_past_str = "YES" if uv_past is True else ("NO" if uv_past is False else "N/A")

        lines = [
            f"MODE: {road_mode}",
            f"FPS: {fps:.1f} | Infer: {infer_ms:.0f}ms",
            f"HC12 RTT: {rtt_str} | Jitter: {jitter_str} | PDR: {pdr_str}",
            f"RSSI: {rssi_str}",
            f"Pi Temp: {temp_str} | UV now/past: {uv_now_str}/{uv_past_str}",
            f"Timing RX->Alert: {rx_alert_str} | Alert@: {self._fmt_epoch_ms(alert_epoch_ms)}",
            f"Remote Det->TX: {det_to_tx_str}",
            f"Speed (L/R): {local_speed:.0f}/{remote_speed:.0f} km/h",
            det_metrics,
        ]
        return "\n".join(lines)

    def _gui_poll(self) -> None:
        """Periodic callback inside the tkinter mainloop; replaces _run_loop."""
        if not self._running:
            self._gui.root.destroy()
            return

        now = time.monotonic()
        self._service_tick(now)

        # ── Process ESP32 messages ──
        if False and self._serial is not None:
            while True:
                msg = self._serial.receive()
                if msg is None:
                    break
                self._handle_esp32_message(msg)

        # ── Periodic commands to ESP32 ──
        if False and self._serial is not None and self._serial.is_connected():
            if now - self._t_last_ping >= PING_INTERVAL_S:
                self._t_last_ping = now
                self._serial.send('{"cmd":"ping"}')
            if now - self._t_last_status >= STATUS_INTERVAL_S:
                self._t_last_status = now
                self._serial.send('{"cmd":"status"}')
            if now - self._t_last_hc12_ping >= HC12_PING_INTERVAL_S:
                self._t_last_hc12_ping = now
                self.send_via_hc12({"type": "RPI_PING", "node": NODE_ID})

        # ── Agreement + decision engine (leader broadcasts, follower applies) ──
        if False:
            self._leader_update_decision(now)

        # ── Compute thesis-aligned display state then update GUI ──
        disp = self._thesis_display_state(now)
        self._gui.update_display(
            disp["status"], disp["label"], disp["speed"],
            disp["direction"], disp.get("emergency", False))
        self._update_alert_timing(now, disp)
        if now - float(self._last_params_update_mono) >= 0.25:
            self._last_params_update_mono = now
            self._gui.update_params_text(self._build_dashboard_params_text(now))
        self._audio_tick(now, disp)
        self._log_surface_state(disp, now)

        self._gui.root.after(50, self._gui_poll)

    def _on_gui_close(self) -> None:
        """Handle CustomTkinter window close event."""
        self._running = False
        self._camera_running = False
        self._gui.root.destroy()

    def _toggle_road_mode(self) -> None:
        """Toggle NARROW/WIDE mode locally and broadcast to the opposite post."""
        with self._state_lock:
            self._road_mode = "WIDE" if self._road_mode == "NARROW" else "NARROW"
            self._mode_seq += 1
            mode = self._road_mode
            seq = self._mode_seq
        logger.info("Road mode toggled: %s", mode)
        self.send_via_hc12({"type": "mode", "m": mode, "q": seq, "n": NODE_ID})
        if self._gui is not None:
            self._gui.show_mode_banner(mode)

    def _toggle_camera_preview(self) -> None:
        """Toggle the optional OpenCV camera preview window (debug)."""
        # Live Feed has been removed to reduce lag and power draw on the Pi.
        logger.info("Camera preview removed (Live Feed toggle disabled).")
        return
        if not (CAMERA_INFERENCE_ENABLED and _CAMERA_INFERENCE_AVAILABLE and cv2 is not None):
            logger.info("Camera preview unavailable (missing packages or disabled).")
            return
        with self._state_lock:
            self._show_camera_preview = not self._show_camera_preview
            enabled = self._show_camera_preview
        if enabled:
            try:
                cv2.namedWindow("Smart Road Alert — Camera Feed", cv2.WINDOW_NORMAL)
            except Exception:
                pass
            logger.info("Camera preview enabled (press D to hide).")
        else:
            try:
                cv2.destroyWindow("Smart Road Alert — Camera Feed")
            except Exception:
                pass
            logger.info("Camera preview disabled (press D to show).")

    # ─── Thesis Decision Engine (Leader + Agreement + Hold) ──────────────────

    def _snapshot_vehicle_states(self, now_mono: float) -> tuple[dict, dict, str, dict]:
        """Return (local_state, remote_state, road_mode, decision_snapshot)."""
        with self._state_lock:
            local = dict(self._local_vehicle_state)
            remote = dict(self._remote_vehicle_state)
            road_mode = str(self._road_mode)
            decision = dict(self._current_decision)

        # Freshness gating.
        if now_mono - float(local.get("updated_mono", 0.0)) > STATE_TIMEOUT_S:
            local.update({
                "present": False,
                "label": "none",
                "category_rank": 0,
                "speed": 0.0,
                "distance": 0.0,
                "distance_confidence": 0,
                "emergency_active": False,
                "vehicle_uid": -1,
                "detected_at_ms": 0,
                "h_direction": "N",
            })
        if now_mono - float(remote.get("received_mono", 0.0)) > STATE_TIMEOUT_S:
            remote.update({
                "present": False,
                "label": "none",
                "category_rank": 0,
                "speed": 0.0,
                "distance": 0.0,
                "distance_confidence": 0,
                "emergency_active": False,
                "vehicle_uid": -1,
                "detected_at_ms": 0,
                "h_direction": "N",
            })

        local["node"] = NODE_ID
        return local, remote, road_mode, decision

    @staticmethod
    def _dir_text_from_code(code: str) -> str:
        return {"L": "LEFT", "R": "RIGHT", "F": "FRONT"}.get(str(code).upper(), "NONE")

    @staticmethod
    def _is_leader_node(remote_node: str) -> bool:
        """Deterministic leader election: lexicographically smallest NODE_ID."""
        if not remote_node:
            return True
        return NODE_ID <= remote_node

    @staticmethod
    def _hold_override(
        traveling_cat_rank: int,
        traveling_emergency: bool,
        new_cat_rank: int,
        new_emergency: bool,
        road_mode: str,
        traveling_label: str = "",
        new_label: str = "",
    ) -> tuple[str, str, bool]:
        """During-hold override table (returns travel_signal, new_signal, needs_arbitration)."""
        # Thesis-specific exception: two police cars in WIDE mode remain SLOW/SLOW,
        # even if either one currently meets the emergency kinematic thresholds.
        if (
            road_mode == "WIDE"
            and traveling_label == "police_car"
            and new_label == "police_car"
        ):
            return "SLOW", "SLOW", False

        # Traveling emergency always keeps GO; other side always STOP.
        if traveling_emergency:
            return "GO", "STOP", False

        # Traveling LARGE always keeps GO; other side always STOP (even emergency).
        if traveling_cat_rank >= 3:
            return "GO", "STOP", False

        # Treat any new-side emergency as "large enough to force a stop" unless
        # it is blocked by traveling large/emergency (handled above).
        if new_emergency:
            return "STOP", "STOP", True

        # Traveling SMALL
        if traveling_cat_rank == 1:
            if new_cat_rank == 1:
                return "GO", "GO", False
            if new_cat_rank == 2:
                return "SLOW", "SLOW", False
            return "STOP", "STOP", True  # new LARGE

        # Traveling MEDIUM
        if traveling_cat_rank == 2:
            if new_cat_rank == 1:
                return "GO", "SLOW", False
            if new_cat_rank == 2:
                if road_mode == "WIDE":
                    return "SLOW", "SLOW", False
                return "STOP", "STOP", True
            return "STOP", "STOP", True  # new LARGE

        return "GO", "GO", False

    @staticmethod
    def _stable_tie_winner(node_a: str, node_b: str, decision_id: int) -> str:
        """Deterministic tie-break without clock sync (alternates by decision_id)."""
        lo, hi = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
        seed = f"{decision_id}:{lo}:{hi}".encode("utf-8")
        bit = sum(seed) & 1
        return lo if bit == 0 else hi

    def _priority_winner_v2(self, local: dict, remote: dict, decision_id: int) -> str:
        """Emergency > size > distance > speed > time > deterministic tie."""
        local_node = NODE_ID
        remote_node = str(remote.get("node", ""))

        local_em = bool(local.get("emergency_active", False))
        remote_em = bool(remote.get("emergency_active", False))
        if local_em and not remote_em:
            return local_node
        if remote_em and not local_em:
            return remote_node

        lc = int(local.get("category_rank", 0))
        rc = int(remote.get("category_rank", 0))
        if lc != rc:
            return local_node if lc > rc else remote_node

        ld = float(local.get("distance", 999.0))
        rd = float(remote.get("distance", 999.0))
        ldc = int(local.get("distance_confidence", 0))
        rdc = int(remote.get("distance_confidence", 0))
        if self._distance_priority_enabled and min(ldc, rdc) > 0:
            dist_margin = 0.5 if min(ldc, rdc) >= 2 else 1.5
            if abs(ld - rd) > dist_margin:
                return local_node if ld < rd else remote_node

        ls = float(local.get("speed", 0.0))
        rs = float(remote.get("speed", 0.0))
        if abs(ls - rs) > 0.5:
            return local_node if ls > rs else remote_node

        # Timestamp tie-break only when we believe clocks are comparable.
        if bool(remote.get("ts_valid", False)):
            lt = int(local.get("detected_at_ms", 0))
            rt = int(remote.get("detected_at_ms", 0))
            if lt and rt and lt != rt:
                return local_node if lt < rt else remote_node

        return self._stable_tie_winner(local_node, remote_node, decision_id)

    def _send_decision(self, decision: dict) -> None:
        """Broadcast the current decision over HC-12 (leader only)."""
        node_a = str(decision.get("node_a", ""))
        node_b = str(decision.get("node_b", ""))
        if not node_a or not node_b or node_a == node_b:
            return
        payload = {
            "type": "decision",
            "id": int(decision["decision_id"]),
            "k": str(decision["kind"]),
            "a": node_a,
            "b": node_b,
            "sa": str(decision["signal_a"]),
            "sb": str(decision["signal_b"]),
            "em": 1 if decision.get("emergency_mode") else 0,
            "tn": str(decision.get("traveling_node", "")),
            "tc": int(decision.get("traveling_category_rank", 0)),
            "te": 1 if decision.get("traveling_emergency") else 0,
            "hs": float(decision.get("hold_s", 0.0) or 0.0),
            "hu": int(decision.get("hold_until_ms", 0) or 0),
        }
        self.send_via_hc12(payload)

    def _leader_update_decision(self, now_mono: float) -> None:
        """Leader computes + broadcasts decisions; followers only apply received ones."""
        local, remote, road_mode, cur = self._snapshot_vehicle_states(now_mono)
        remote_node = str(remote.get("node", ""))
        if not self._is_leader_node(remote_node):
            return

        desired = self._compute_leader_decision(now_mono, local, remote, road_mode, cur)

        def _sig(d: dict) -> tuple:
            return (
                d.get("kind"),
                d.get("node_a"), d.get("node_b"),
                d.get("signal_a"), d.get("signal_b"),
                d.get("emergency_mode"),
                d.get("traveling_node"),
                d.get("traveling_category_rank"),
                d.get("traveling_emergency"),
                int(d.get("hold_until_mono", 0.0) > now_mono),
            )

        with self._state_lock:
            cur2 = dict(self._current_decision)
        changed = _sig(desired) != _sig(cur2)

        # Assign decision_id on change.
        if changed:
            self._decision_id_counter += 1
            desired["decision_id"] = self._decision_id_counter
            desired["received_mono"] = now_mono
            desired["acked_by_remote"] = False
            desired["acked_remote_node"] = ""
            desired["acked_mono"] = 0.0
            with self._state_lock:
                self._current_decision.update(desired)
            self._last_decision_sent_mono = 0.0  # force immediate send
        else:
            desired["decision_id"] = int(cur2.get("decision_id", 0))
            desired["acked_by_remote"] = bool(cur2.get("acked_by_remote", False))
            desired["acked_remote_node"] = str(cur2.get("acked_remote_node", ""))
            desired["acked_mono"] = float(cur2.get("acked_mono", 0.0))

        # Resend while active to reduce packet-loss impact.
        active = str(desired.get("kind", "IDLE")) != "IDLE"
        if (changed or (active and now_mono - self._last_decision_sent_mono >= DECISION_RESEND_INTERVAL_S)):
            self._last_decision_sent_mono = now_mono
            self._send_decision(desired)

    def _compute_leader_decision(
        self,
        now_mono: float,
        local: dict,
        remote: dict,
        road_mode: str,
        cur: dict,
    ) -> dict:
        """Leader-only decision computation with arbitration pause + hold."""
        local_node = NODE_ID
        remote_node = str(remote.get("node", ""))
        emergency_mode = bool(local.get("emergency_active") or remote.get("emergency_active"))

        hold_until = float(cur.get("hold_until_mono", 0.0))
        hold_active = hold_until > 0.0 and now_mono < hold_until
        if hold_active and cur.get("traveling_node"):
            traveling_node = str(cur.get("traveling_node", ""))
            traveling_cat = int(cur.get("traveling_category_rank", 0))
            traveling_em = bool(cur.get("traveling_emergency", False))

            # Determine which side is "new" during hold.
            travel_is_local = traveling_node == local_node
            travel_state = local if travel_is_local else remote
            new_state = remote if travel_is_local else local

            if bool(new_state.get("present", False)):
                travel_sig, new_sig, needs_arb = self._hold_override(
                    traveling_cat,
                    traveling_em,
                    int(new_state.get("category_rank", 0)),
                    bool(new_state.get("emergency_active", False)),
                    road_mode,
                    str(travel_state.get("label", "")),
                    str(new_state.get("label", "")),
                )
                if needs_arb:
                    # Interrupt hold and enter arbitration.
                    self._arbitration_start_mono = None
                    self._arbitration_context = None
                    hold_active = False
                else:
                    # If a hold override changes the surface signals, broadcast it and
                    # reset the hold timer for the currently moving direction.
                    if travel_sig == "GO" and new_sig != "GO":
                        hold_until = now_mono + GO_HOLD_S
                        traveling_cat = int(travel_state.get("category_rank", traveling_cat))
                        traveling_em = bool(travel_state.get("emergency_active", traveling_em))
                    else:
                        hold_until = 0.0
                        traveling_node = ""
                        traveling_cat = 0
                        traveling_em = False

                    node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
                    sig_a = travel_sig if node_a == traveling_node else new_sig
                    sig_b = new_sig if node_b != traveling_node else travel_sig
                    if traveling_node and node_a == (remote_node if travel_is_local else local_node):
                        # swap safety, but keep as computed below by node mapping
                        pass

                    return {
                        "kind": "ARBITRATED" if hold_until else ("BOTH_GO" if travel_sig == "GO" and new_sig == "GO" else "BOTH_SLOW"),
                        "node_a": node_a,
                        "node_b": node_b,
                        "signal_a": sig_a,
                        "signal_b": sig_b,
                        "emergency_mode": emergency_mode,
                        "traveling_node": traveling_node,
                        "traveling_category_rank": traveling_cat,
                        "traveling_emergency": traveling_em,
                        "hold_until_mono": hold_until,
                        "hold_s": GO_HOLD_S if hold_until else 0.0,
                        "hold_until_ms": int(time.time() * 1000 + (GO_HOLD_S * 1000)) if hold_until else 0,
                    }

            # No new vehicle during hold → keep current decision, but keep emergency flag updated.
            return {
                "kind": str(cur.get("kind", "IDLE")),
                "node_a": str(cur.get("node_a", local_node)),
                "node_b": str(cur.get("node_b", remote_node)),
                "signal_a": str(cur.get("signal_a", "")),
                "signal_b": str(cur.get("signal_b", "")),
                "emergency_mode": emergency_mode,
                "traveling_node": traveling_node,
                "traveling_category_rank": traveling_cat,
                "traveling_emergency": traveling_em,
                "hold_until_mono": hold_until,
                "hold_s": GO_HOLD_S,
                "hold_until_ms": int(time.time() * 1000 + (hold_until - now_mono) * 1000),
            }

        # No hold active.
        local_present = bool(local.get("present", False))
        remote_present = bool(remote.get("present", False))

        # Clear arbitration when not needed.
        if not (local_present and remote_present):
            self._arbitration_start_mono = None
            self._arbitration_context = None

        if not local_present and not remote_present:
            return {
                "kind": "IDLE",
                "node_a": local_node,
                "node_b": remote_node,
                "signal_a": "",
                "signal_b": "",
                "emergency_mode": False,
                "traveling_node": "",
                "traveling_category_rank": 0,
                "traveling_emergency": False,
                "hold_until_mono": 0.0,
                "hold_s": 0.0,
                "hold_until_ms": 0,
            }

        # One-side detection → winner GO, loser blank, with hold.
        if local_present != remote_present:
            winner = local_node if local_present else remote_node
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            traveling_state = local if winner == local_node else remote
            is_emergency = bool(traveling_state.get("emergency_active", False))
            if is_emergency:
                sig_a = "GO" if node_a == winner else "STOP"
                sig_b = "GO" if node_b == winner else "STOP"
                kind = "EMERGENCY"
            else:
                sig_a = "GO" if node_a == winner else ""
                sig_b = "GO" if node_b == winner else ""
                kind = "ONE_SIDE"
            return {
                "kind": kind,
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": sig_a,
                "signal_b": sig_b,
                "emergency_mode": is_emergency,
                "traveling_node": winner,
                "traveling_category_rank": int(traveling_state.get("category_rank", 0)),
                "traveling_emergency": is_emergency,
                "hold_until_mono": now_mono + GO_HOLD_S,
                "hold_s": GO_HOLD_S,
                "hold_until_ms": int(time.time() * 1000 + GO_HOLD_S * 1000),
            }

        # Both sides present.
        local_em = bool(local.get("emergency_active", False))
        remote_em = bool(remote.get("emergency_active", False))
        lc = int(local.get("category_rank", 0))
        rc = int(remote.get("category_rank", 0))

        # Emergency involvement is immediate (no arbitration pause).
        if local_em or remote_em:
            self._arbitration_start_mono = None
            self._arbitration_context = None
            winner = self._priority_winner_v2(local, remote, int(cur.get("decision_id", 0)) + 1)
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            sig_a = "GO" if node_a == winner else "STOP"
            sig_b = "GO" if node_b == winner else "STOP"
            traveling_state = local if winner == local_node else remote
            return {
                "kind": "EMERGENCY",
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": sig_a,
                "signal_b": sig_b,
                "emergency_mode": True,
                "traveling_node": winner,
                "traveling_category_rank": int(traveling_state.get("category_rank", 0)),
                "traveling_emergency": bool(traveling_state.get("emergency_active", False)),
                "hold_until_mono": now_mono + GO_HOLD_S,
                "hold_s": GO_HOLD_S,
                "hold_until_ms": int(time.time() * 1000 + GO_HOLD_S * 1000),
            }

        # Non-emergency, size-based rules.
        if lc == 1 and rc == 1:
            self._arbitration_start_mono = None
            self._arbitration_context = None
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            return {
                "kind": "BOTH_GO",
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": "GO",
                "signal_b": "GO",
                "emergency_mode": False,
                "traveling_node": "",
                "traveling_category_rank": 0,
                "traveling_emergency": False,
                "hold_until_mono": 0.0,
                "hold_s": 0.0,
                "hold_until_ms": 0,
            }

        if (lc, rc) in ((1, 2), (2, 1)):
            self._arbitration_start_mono = None
            self._arbitration_context = None
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            return {
                "kind": "BOTH_SLOW",
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": "SLOW",
                "signal_b": "SLOW",
                "emergency_mode": False,
                "traveling_node": "",
                "traveling_category_rank": 0,
                "traveling_emergency": False,
                "hold_until_mono": 0.0,
                "hold_s": 0.0,
                "hold_until_ms": 0,
            }

        if lc == 2 and rc == 2 and road_mode == "WIDE":
            self._arbitration_start_mono = None
            self._arbitration_context = None
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            return {
                "kind": "BOTH_SLOW",
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": "SLOW",
                "signal_b": "SLOW",
                "emergency_mode": False,
                "traveling_node": "",
                "traveling_category_rank": 0,
                "traveling_emergency": False,
                "hold_until_mono": 0.0,
                "hold_s": 0.0,
                "hold_until_ms": 0,
            }

        # Arbitration-required cases: MEDIUM/MEDIUM in NARROW, or any LARGE involvement.
        context = (
            bool(local_present),
            bool(remote_present),
            str(local.get("label", "")),
            str(remote.get("label", "")),
            int(lc),
            int(rc),
            road_mode,
        )
        if self._arbitration_start_mono is None or self._arbitration_context != context:
            self._arbitration_start_mono = now_mono
            self._arbitration_context = context

        if now_mono - float(self._arbitration_start_mono) < ARBITRATION_PAUSE_S:
            node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
            return {
                "kind": "ARBITRATING",
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": "STOP",
                "signal_b": "STOP",
                "emergency_mode": False,
                "traveling_node": "",
                "traveling_category_rank": 0,
                "traveling_emergency": False,
                "hold_until_mono": 0.0,
                "hold_s": 0.0,
                "hold_until_ms": 0,
            }

        # Arbitration resolved.
        self._arbitration_start_mono = None
        self._arbitration_context = None
        winner = self._priority_winner_v2(local, remote, int(cur.get("decision_id", 0)) + 1)
        node_a, node_b = (local_node, remote_node) if (not remote_node or local_node <= remote_node) else (remote_node, local_node)
        sig_a = "GO" if node_a == winner else "STOP"
        sig_b = "GO" if node_b == winner else "STOP"
        traveling_state = local if winner == local_node else remote
        return {
            "kind": "ARBITRATED",
            "node_a": node_a,
            "node_b": node_b,
            "signal_a": sig_a,
            "signal_b": sig_b,
            "emergency_mode": False,
            "traveling_node": winner,
            "traveling_category_rank": int(traveling_state.get("category_rank", 0)),
            "traveling_emergency": bool(traveling_state.get("emergency_active", False)),
            "hold_until_mono": now_mono + GO_HOLD_S,
            "hold_s": GO_HOLD_S,
            "hold_until_ms": int(time.time() * 1000 + GO_HOLD_S * 1000),
        }

    def _thesis_display_state(self, now_mono: float) -> dict:
        """Compute the GUI state for this node from the latest decision + remote state."""
        local, remote, _road_mode, decision = self._snapshot_vehicle_states(now_mono)

        # Prefer leader/follower decision when present; otherwise fall back to local-only.
        node_a = str(decision.get("node_a", NODE_ID))
        node_b = str(decision.get("node_b", ""))
        signal = ""
        if NODE_ID == node_a:
            signal = str(decision.get("signal_a", ""))
        elif NODE_ID == node_b:
            signal = str(decision.get("signal_b", ""))

        hold_active = now_mono < float(decision.get("hold_until_mono", 0.0))
        kind = str(decision.get("kind", "IDLE"))
        awaiting_agreement, agreement_timed_out = self._agreement_pending(decision, now_mono)

        # Idle scenario (both sides none) → dedicated message.
        if not bool(local.get("present", False)) and not bool(remote.get("present", False)) and not hold_active:
            status_text = "NO VEHICLE"
        else:
            status_text = signal
        if awaiting_agreement:
            status_text = self._agreement_safe_status(decision, agreement_timed_out)

        # Vehicle info always shows the OPPOSITE side's approaching vehicle.
        show_remote = bool(remote.get("present", False))
        label = str(remote.get("label", "none")) if show_remote else "none"
        speed = float(remote.get("speed", 0.0)) if show_remote else 0.0
        direction = self._dir_text_from_code(str(remote.get("h_direction", "N"))) if show_remote else "NONE"

        # Scenario 2 loser: blank status; show remote info briefly, then clear during hold.
        # One-sided emergency cases use kind=EMERGENCY, so STOP stays visible.
        if kind == "ONE_SIDE" and hold_active and status_text == "":
            since = now_mono - float(decision.get("received_mono", 0.0))
            if since > ONE_SIDE_LOSER_INFO_S:
                label, speed, direction = "none", 0.0, "NONE"

        emergency_mode = bool(decision.get("emergency_mode", False))
        if awaiting_agreement and kind != "EMERGENCY":
            emergency_mode = bool(local.get("emergency_active", False) or remote.get("emergency_active", False))
        return {
            "status": status_text,
            "label": label,
            "speed": speed,
            "direction": direction,
            "emergency": emergency_mode,
        }

    def _audio_tick(self, now_mono: float, disp: dict) -> None:
        """Drive thesis-specified TTS prompts (called from the GUI thread)."""
        if self._tts is None:
            return

        local, remote, _road_mode, decision = self._snapshot_vehicle_states(now_mono)

        # ── Scenario 1: Idle announcement every 20s (after initial 20s) ──
        if disp.get("status") == "NO VEHICLE":
            if self._idle_started_mono is None:
                self._idle_started_mono = now_mono
                self._last_idle_spoken_mono = 0.0
            if (
                now_mono - float(self._idle_started_mono) >= NO_VEHICLE_REPEAT_S
                and now_mono - float(self._last_idle_spoken_mono) >= NO_VEHICLE_REPEAT_S
            ):
                self._tts.speak("THERE'S NO VEHICLE ON THE OPPOSITE SIDE.")
                self._last_idle_spoken_mono = now_mono
        else:
            self._idle_started_mono = None

        # ── Local detection alert (rising edge / new track) ──
        if bool(local.get("present", False)):
            uid = int(local.get("vehicle_uid", -1))
            if uid != int(self._last_announced_local_uid):
                label = str(local.get("label", "none"))
                if label and label not in ("none", "clear"):
                    name = label.replace("_", " ").upper()
                    self._tts.speak(f"APPROACHING {name} DETECTED")
                self._last_announced_local_uid = uid
        else:
            self._last_announced_local_uid = -1

        # ── Decision-driven prompts (agreement events) ──
        decision_id = int(decision.get("decision_id", 0))
        if decision_id <= int(self._last_spoken_decision_id):
            return
        awaiting_agreement, agreement_timed_out = self._agreement_pending(decision, now_mono)
        if awaiting_agreement and not agreement_timed_out:
            return
        self._last_spoken_decision_id = decision_id

        node_a = str(decision.get("node_a", NODE_ID))
        node_b = str(decision.get("node_b", ""))
        our_signal = ""
        if NODE_ID == node_a:
            our_signal = str(decision.get("signal_a", ""))
        elif NODE_ID == node_b:
            our_signal = str(decision.get("signal_b", ""))

        kind = str(decision.get("kind", "IDLE"))
        remote_cat_rank = int(remote.get("category_rank", 0))
        remote_cat = {3: "LARGE", 2: "MEDIUM", 1: "SMALL"}.get(remote_cat_rank, "VEHICLE")
        remote_present = bool(remote.get("present", False))

        msg: Optional[str] = None
        force = False

        if agreement_timed_out:
            self._tts.speak("STOP! AGREEMENT NOT CONFIRMED. PLEASE WAIT.", force=True)
            return

        if kind == "ONE_SIDE":
            force = True
            if our_signal == "GO":
                msg = "GO! THERE'S NO VEHICLE ON THE OPPOSITE SIDE."
            else:
                msg = "VEHICLE APPROACHING FROM THE OPPOSITE SIDE."

        elif kind == "BOTH_GO":
            msg = f"GO! {remote_cat} VEHICLE DETECTED FROM THE OPPOSITE SIDE."

        elif kind == "BOTH_SLOW":
            msg = f"SLOW DOWN! {remote_cat} VEHICLE DETECTED FROM THE OPPOSITE SIDE."

        elif kind == "ARBITRATING":
            force = True
            msg = f"STOP! {remote_cat} VEHICLE FROM THE OPPOSITE SIDE."

        elif kind == "ARBITRATED":
            force = True
            if our_signal == "GO":
                msg = f"GO! {remote_cat} VEHICLE FROM THE OPPOSITE SIDE IS ON STANDBY."
            else:
                msg = "STOP! PLEASE STANDBY."

        elif kind == "EMERGENCY":
            force = True
            if our_signal == "STOP":
                msg = "STOP! EMERGENCY VEHICLE APPROACHING FROM THE OPPOSITE SIDE, PLEASE WAIT!"
            else:
                msg = (
                    "GO! THE VEHICLE DETECTED FROM THE OPPOSITE SIDE IS ON STANDBY."
                    if remote_present
                    else "GO! THERE'S NO VEHICLE ON THE OPPOSITE SIDE."
                )

        if msg:
            self._tts.speak(msg, force=force)

    # ─── Cross-RPI Display Decision ───────────────────────────────────────────

    def _decide_display_state(self, now: float) -> dict:
        """Legacy fallback GUI-state helper.

        This path is retained only as a non-thesis fallback / reference.
        The thesis runtime uses `_gui_poll()` -> `_leader_update_decision()` ->
        `_thesis_display_state()`.

        Rules (symmetric peer-to-peer):
        ─ Case 1/2 (one side detecting):
          • Detecting side   → GO (no incoming threat).
          • Receiving side   → SLOW / STOP per remote telemetry.
        ─ Case 3 (both sides detecting simultaneously):
          • Priority winner  → GO.
          • Loser            → SLOW / STOP.
        ─ Neither active     → GO (road clear).

        Smoothing:
          Remote display is held for ``_REMOTE_HOLD_S`` seconds after the last
          received non-empty packet, preventing flickering when frames are
          briefly missed.  Local activity uses ``_LOCAL_ACTIVE_HOLD_S`` for
          the same reason.
        """
        remote_fresh = (now - self._last_received_time) < self._REMOTE_HOLD_S
        local_active = (now - self._last_local_active_time) < self._LOCAL_ACTIVE_HOLD_S

        _go: dict = {
            "status": "GO", "label": "none",
            "speed": 0.0, "direction": "NONE", "emergency": False,
        }

        if not remote_fresh:
            # No recent remote telemetry → road clear from our perspective.
            return _go

        # Remote is fresh: read latest snapshot.
        with self._display_lock:
            rd = dict(self._remote_display)

        if not local_active:
            # Case 1/2: only remote is active → show their signal.
            return {
                "status":    rd["signal"],
                "label":     rd["label"],
                "speed":     rd["speed"],
                "direction": rd["direction"],
                "emergency": rd["emergency"],
            }

        # Case 3: both sides detecting → run priority decision.
        with self._display_lock:
            ld = dict(self._local_display)

        winner = self._priority_winner(ld, rd)
        if winner == "local":
            # We have right-of-way → show GO, still display remote vehicle info
            # so our driver knows what is on the other side.
            return {
                "status":    "GO",
                "label":     rd["label"],
                "speed":     rd["speed"],
                "direction": rd["direction"],
                "emergency": False,  # priority won, no emergency pulse
            }
        else:
            # Remote has right-of-way → show SLOW / STOP.
            return {
                "status":    rd["signal"],
                "label":     rd["label"],
                "speed":     rd["speed"],
                "direction": rd["direction"],
                "emergency": rd["emergency"],
            }

    @staticmethod
    def _priority_winner(local: dict, remote: dict) -> str:
        """Legacy fallback priority helper used by `_decide_display_state()`.

        Returns ``"local"`` if the local vehicle should proceed (show GO),
        or ``"remote"`` if the remote vehicle has priority (show SLOW/STOP).

        Decision order:
        1. Emergency class always beats non-emergency.
        2. Larger vehicle class wins (bus/truck > car > motorcycle).
        3. Higher speed wins (more kinetic energy, harder to stop).
        4. Tie → remote wins (conservative — slow down by default).
        """
        local_em  = local.get("emergency", False)
        remote_em = remote.get("emergency", False)

        if local_em and not remote_em:
            return "local"
        if remote_em and not local_em:
            return "remote"

        def _size_rank(label: str) -> int:
            if label in _LARGE_VEHICLES:
                return 3
            if label in _MEDIUM_VEHICLES:
                return 2
            return 1

        local_rank  = _size_rank(local.get("label", ""))
        remote_rank = _size_rank(remote.get("label", ""))
        if local_rank != remote_rank:
            return "local" if local_rank > remote_rank else "remote"

        local_speed  = local.get("speed", 0.0)
        remote_speed = remote.get("speed", 0.0)
        if abs(local_speed - remote_speed) > 5.0:
            return "local" if local_speed > remote_speed else "remote"

        # Tie → conservative default: yield to remote.
        return "remote"

    # ─── Message Dispatcher ───────────────────────────────────────────────────

    def _handle_esp32_message(self, raw: str) -> None:
        """
        Parse a JSON line received from the local ESP32 and dispatch to the
        appropriate handler.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ESP32: received non-JSON message: %r", raw)
            return

        msg_type: str = data.get("type", "")

        if msg_type == "vehicle":
            # Local telemetry from ESP32 — process and forward to remote RPi.
            self._on_local_vehicle_data(data)

        elif msg_type == "pong":
            logger.debug("ESP32 PONG received.")

        elif msg_type == "status":
            state = data.get("state", "unknown")
            logger.debug("ESP32 status: %s", state)

        elif msg_type == "heartbeat":
            logger.debug("ESP32 heartbeat received.")

        elif msg_type == "HC12_RECV":
            # The ESP32 received a message from HC-12 and relayed it here.
            # Unwrap the inner payload and pass it to the wireless dispatcher.
            inner_str = data.get("payload", "")
            self._handle_wireless_message(inner_str)

        elif msg_type == "error":
            logger.warning("ESP32 error: %s", data.get("msg", "(no detail)"))

        else:
            logger.warning("ESP32: unknown message type %r: %s", msg_type, raw)

    # ─── Message Handlers ─────────────────────────────────────────────────────

    def _on_local_vehicle_data(self, data: dict) -> None:
        """
        Process telemetry from the local ESP32 device.

        NOTE: No physical speed/distance sensor is currently connected to the
        ESP32.  Vehicle telemetry is provided exclusively by the RPi YOLO
        camera pipeline.  Any packet arriving here is ignored until a real
        sensor is wired up and the ESP32 firmware is updated to send real data.
        """
        logger.debug("ESP32 sensor packet ignored (no hardware sensor): %s", data)

    # ─── HC-12 Send Helper ────────────────────────────────────────────────────

    def send_via_hc12(self, payload_dict: dict) -> None:
        """
        Serialise *payload_dict* to JSON and send it to the remote RPi via the
        local ESP32's HC-12 radio bridge.

        The ESP32 expects messages in the following envelope format:
            {"type":"HC12_SEND","payload":"<escaped-json-string>"}

        The ESP32 extracts the inner payload string and transmits it over the
        HC-12 UART; the remote ESP32 receives it, wraps it in HC12_RECV, and
        feeds it to the remote RPi via USB.

        This method is a no-op when the ESP32 serial link is unavailable.
        """
        if self._serial is None or not self._serial.is_connected():
            logger.debug("send_via_hc12: ESP32 not connected — dropped: %s", payload_dict)
            return
        inner_json = json.dumps(payload_dict, separators=(",", ":"))
        wrapper    = json.dumps(
            {"type": "HC12_SEND", "payload": inner_json},
            separators=(",", ":"),
        )
        self._serial.send(wrapper)
        logger.debug("HC12_SEND dispatched: %s", inner_json)

    def _on_remote_vehicle_state(self, data: dict) -> None:
        """Process compact vehicle_state packets received from the remote post."""
        recv_mono = time.monotonic()
        recv_epoch_ms = int(time.time() * 1000)

        # Compact keys (see sender in _on_inference_detections v2 logic).
        present = bool(int(data.get("p", 0)))
        label = str(data.get("l", "none"))
        cat_rank = int(data.get("cr", 0))
        speed = float(data.get("s", 0.0))
        distance = float(data.get("d", 0.0))
        distance_confidence = int(data.get("dc", 0))
        emergency = bool(int(data.get("e", 0)))
        vehicle_uid = int(data.get("id", -1))
        detected_at_ms = int(data.get("t", 0))
        remote_now_ms = data.get("nm")
        seq = int(data.get("q", 0))
        node = str(data.get("n", ""))
        h_code = str(data.get("h", "N"))

        det_to_tx_ms: Optional[int] = None
        if remote_now_ms is not None and detected_at_ms > 0:
            try:
                det_to_tx_ms = max(0, int(remote_now_ms) - int(detected_at_ms))
            except Exception:
                det_to_tx_ms = None

        ts_valid = False
        if remote_now_ms is not None:
            try:
                ts_valid = abs(int(remote_now_ms) - recv_epoch_ms) <= 5000
            except Exception:
                ts_valid = False

        with self._state_lock:
            self._remote_vehicle_state.update({
                "present": present,
                "label": label,
                "category_rank": cat_rank,
                "speed": speed,
                "distance": distance,
                "distance_confidence": distance_confidence,
                "emergency_active": emergency,
                "vehicle_uid": vehicle_uid,
                "detected_at_ms": detected_at_ms,
                "det_to_tx_ms": det_to_tx_ms,
                "h_direction": h_code,
                "seq": seq,
                "node": node,
                "received_mono": recv_mono,
                "received_epoch_ms": recv_epoch_ms,
                "ts_valid": ts_valid,
            })

    def _on_remote_mode(self, data: dict) -> None:
        """Sync road mode from the opposite post (NARROW/WIDE)."""
        mode = str(data.get("m", "")).upper()
        if mode not in ("NARROW", "WIDE"):
            return
        with self._state_lock:
            self._road_mode = mode
        logger.info("Road mode synced from remote: %s", mode)
        if self._gui is not None:
            self._gui.show_mode_banner(mode)

    def _on_remote_decision(self, data: dict) -> None:
        """Apply an authoritative decision broadcast by the deterministic leader."""
        recv_mono = time.monotonic()

        try:
            decision_id = int(data.get("id", 0))
        except Exception:
            return

        with self._state_lock:
            if decision_id <= int(self._current_decision.get("decision_id", 0)):
                return

        kind = str(data.get("k", "IDLE"))
        node_a = str(data.get("a", ""))
        node_b = str(data.get("b", ""))
        signal_a = str(data.get("sa", ""))
        signal_b = str(data.get("sb", ""))
        emergency_mode = bool(int(data.get("em", 0)))
        traveling_node = str(data.get("tn", ""))
        traveling_category_rank = int(data.get("tc", 0))
        traveling_emergency = bool(int(data.get("te", 0)))
        hold_s = float(data.get("hs", 0.0) or 0.0)

        hold_until_mono = 0.0
        if hold_s > 0.0 and kind in ("ONE_SIDE", "ARBITRATED", "EMERGENCY"):
            hold_until_mono = recv_mono + hold_s

        with self._state_lock:
            self._current_decision.update({
                "decision_id": decision_id,
                "kind": kind,
                "node_a": node_a,
                "node_b": node_b,
                "signal_a": signal_a,
                "signal_b": signal_b,
                "emergency_mode": emergency_mode,
                "traveling_node": traveling_node,
                "traveling_category_rank": traveling_category_rank,
                "traveling_emergency": traveling_emergency,
                "hold_until_mono": hold_until_mono,
                "received_mono": recv_mono,
                "acked_by_remote": False,
                "acked_remote_node": "",
                "acked_mono": 0.0,
            })

        if NODE_ID in (node_a, node_b):
            self.send_via_hc12({
                "type": "ack",
                "for": "decision",
                "id": decision_id,
                "node": NODE_ID,
            })

    def _on_remote_ack(self, data: dict) -> None:
        """Track agreement acknowledgments from the opposite post."""
        if str(data.get("for", "")).lower() != "decision":
            logger.debug("HC-12 ACK from remote RPi: %s", data.get("msg", ""))
            return

        try:
            decision_id = int(data.get("id", 0))
        except Exception:
            return
        ack_node = str(data.get("node", ""))
        if not ack_node:
            return

        with self._state_lock:
            if decision_id != int(self._current_decision.get("decision_id", 0)):
                return
            self._current_decision.update({
                "acked_by_remote": True,
                "acked_remote_node": ack_node,
                "acked_mono": time.monotonic(),
            })
        logger.info("Decision %d acknowledged by %s.", decision_id, ack_node)

    def _on_remote_vehicle_data(self, data: dict) -> None:
        """Process vehicle telemetry received from the remote RPi via HC-12.

        Handles both legacy ``{speed, distance}`` packets from ESP32 sensors
        and enriched camera-based telemetry with label, priority, direction.
        """
        label     = data.get("label", "vehicle")
        speed     = data.get("speed")
        distance  = data.get("distance", 0.0)
        direction = data.get("direction", "unknown")
        priority  = data.get("priority", "LOW")
        emergency = data.get("emergency_active", False)
        node      = data.get("node", "unknown")

        if speed is None:
            logger.warning("Incomplete vehicle packet from HC-12: %s", data)
            return

        speed    = float(speed)
        distance = float(distance) if distance is not None else 0.0

        # ── No-vehicle reset packet: speed==0 && direction=="none" ──────────
        # The remote RPi sends this every second when its YOLO has no
        # detections.  Update the cv2 HUD overlay only; the GUI smoothing
        # timer (_last_received_time) handles the 3-second hold before clearing
        # the signal (GO/SLOW/STOP).  We DO clear the vehicle info fields
        # immediately so the image/label disappears as soon as RPi2 sees nothing.
        if speed == 0.0 and direction in ("none", "unknown") and not emergency:
            self._last_telemetry.update({
                "label":     "none",
                "speed":     0.0,
                "distance":  0.0,
                "direction": "none",
                "priority":  "LOW",
                "emergency": False,
                "last_seen": time.time(),
            })
            # Clear vehicle info in the display snapshot so the image is removed
            # immediately without waiting for the hold timer to expire.
            with self._display_lock:
                self._remote_display.update({
                    "label":     "none",
                    "speed":     0.0,
                    "direction": "NONE",
                })
            self._tts_remote_alert("none", 0.0, "GO", False)
            logger.debug("REMOTE → no vehicle | source=%s", node)
            return

        logger.info(
            "TELEMETRY → %-12s | %5.1f km/h | %-8s | dist=%5.1fm | pri=%-6s | em=%s | source=%s",
            label, speed, direction, distance, priority, emergency, node,
        )

        # ── Update HUD overlay with received HC-12 data ──
        self._last_received_time = time.monotonic()   # mark remote as active
        self._last_telemetry.update({
            "label":     label,
            "speed":     speed,
            "distance":  distance,
            "direction": direction,
            "priority":  priority,
            "emergency": emergency,
            "last_seen": time.time(),
        })

        # Any active remote vehicle = local side must be warned (SLOW / STOP).
        # The camera-relative "incoming"/"outgoing" direction is NOT used to
        # gate the alert — either direction means a vehicle is occupying the
        # road and the local side needs to react.
        signal = self._get_alert_signal(priority, "incoming", speed, emergency)
        # _get_alert_signal returns "GO" for LOW-priority or slow MEDIUM vehicles
        # (thresholds designed for LOCAL detection).  For REMOTE telemetry any
        # detected vehicle always warrants at minimum SLOW on the receiving side.
        if signal == "GO":
            signal = "SLOW"
        h_dir  = data.get("h_direction", "FRONT")

        self._send_display_command(label, signal, speed, priority, emergency)
        with self._display_lock:
            self._remote_display.update({
                "label":     label,
                "speed":     speed,
                "direction": h_dir,
                "signal":    signal,
                "emergency": emergency,
            })
        self._tts_remote_alert(label, speed, signal, emergency)

        if emergency:
            logger.warning(
                "REMOTE EMERGENCY: active %s at %.1f km/h from %s",
                label, speed, node,
            )
        elif signal in ("STOP", "GO SLOW", "SLOW"):
            logger.warning(
                "REMOTE ALERT: %s at %.1f km/h from %s — %s",
                label, speed, node, signal,
            )

    def _process_vehicle_telemetry(self, speed: float, distance: float, source: str) -> None:
        """
        Common alert logic for vehicle telemetry from any source.
        
        Parameters
        ----------
        speed   : float — vehicle speed in km/h
        distance : float — forward distance in metres
        source   : str   — where the data came from ("local_esp32" or "remote_rpi")
        """
        # ── Alert thresholds ──
        if speed > 80.0:
            logger.warning(
                "ALERT: High-speed vehicle detected (%.1f km/h) [source: %s].",
                speed, source,
            )
            if source == "local_esp32":
                self.send_via_hc12(
                    {"type": "ALERT", "speed": speed, "distance": distance, "node": NODE_ID}
                )

        if distance < 5.0:
            logger.warning(
                "ALERT: Vehicle proximity warning (%.1f m) [source: %s].",
                distance, source,
            )
            if source == "local_esp32":
                self.send_via_hc12(
                    {"type": "ALERT", "speed": speed, "distance": distance, "node": NODE_ID}
                )


    # ─── HC-12 Wireless Message Handler ─────────────────────────────────────

    def _handle_wireless_message(self, raw: str) -> None:
        """
        Parse and dispatch a JSON message received over the HC-12 radio link
        from the remote Raspberry Pi (true peer-to-peer communication).
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("HC-12: received non-JSON message: %r", raw)
            return

        msg_type: str = data.get("type", "")

        if msg_type == "vehicle_state":
            self._on_remote_vehicle_state(data)

        elif msg_type == "mode":
            self._on_remote_mode(data)

        elif msg_type == "decision":
            self._on_remote_decision(data)

        elif msg_type == "vehicle":
            # Legacy remote telemetry (kept for backward compatibility).
            self._on_remote_vehicle_data(data)

        elif msg_type == "ALERT":
            remote_node = data.get("node", "unknown")
            speed    = data.get("speed")
            distance = data.get("distance")
            logger.warning(
                "HC-12 ALERT from %s: speed=%.1f km/h, distance=%.1f m.",
                remote_node, speed or 0.0, distance or 0.0,
            )
            if speed is not None and distance is not None:
                self._process_vehicle_telemetry(
                    float(speed), float(distance), source=f"remote_{remote_node}"
                )

        elif msg_type == "RPI_PING":
            remote_node = data.get("node", "unknown")
            logger.debug("HC-12 RPI_PING from %s — sending PONG.", remote_node)
            self.send_via_hc12({"type": "RPI_PONG", "node": NODE_ID, "q": int(data.get("q", 0) or 0)})

        elif msg_type == "RPI_PONG":
            remote_node = data.get("node", "unknown")
            recv_mono = time.monotonic()
            try:
                seq = int(data.get("q", 0) or 0)
            except Exception:
                seq = 0
            rssi_val = data.get("rssi", data.get("RSSI"))
            rtt_ms: Optional[float] = None
            with self._metrics_lock:
                self._hc12_pongs_recv += 1
                if rssi_val is not None:
                    try:
                        self._hc12_last_rssi_dbm = float(rssi_val)
                    except Exception:
                        pass
                sent_mono = self._hc12_ping_pending.pop(seq, None) if seq else None
                if sent_mono is not None:
                    rtt_ms = (recv_mono - float(sent_mono)) * 1000.0
                    self._hc12_last_rtt_ms = float(rtt_ms)
                    self._hc12_rtt_samples_ms.append(float(rtt_ms))
            logger.debug("HC-12 RPI_PONG received from %s.", remote_node)

        elif msg_type == "ack":
            self._on_remote_ack(data)

        elif msg_type == "alert":
            logger.warning("HC-12 ALERT from remote RPi: %s", data.get("msg", "(no detail)"))

        elif msg_type == "status":
            status = data.get("status", "(no detail)")
            logger.debug("HC-12 status from remote RPi: %s", status)

        else:
            logger.warning("HC-12: unknown message type %r: %s", msg_type, raw)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def _run_decision_tests() -> None:
    """Pure-logic decision harness (no hardware required)."""
    host = SmartRoadAlertHost()

    class _FakeTTS:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def speak(self, text: str, force: bool = False) -> None:
            self.calls.append((text, force))

    def _mk_state(
        node: str,
        present: bool,
        label: str,
        cat_rank: int,
        speed: float,
        distance: float,
        emergency: bool = False,
        detected_at_ms: int = 1000,
        ts_valid: bool = True,
        vehicle_uid: int = 1,
        h_direction: str = "F",
    ) -> dict:
        return {
            "node": node,
            "present": present,
            "label": label,
            "category_rank": cat_rank,
            "speed": speed,
            "distance": distance,
            "distance_confidence": 2,
            "emergency_active": emergency,
            "detected_at_ms": detected_at_ms,
            "ts_valid": ts_valid,
            "vehicle_uid": vehicle_uid,
            "h_direction": h_direction,
        }

    def _sig_for(dec: dict, node: str) -> str:
        if node == str(dec.get("node_a", "")):
            return str(dec.get("signal_a", ""))
        if node == str(dec.get("node_b", "")):
            return str(dec.get("signal_b", ""))
        return ""

    peer = "PEER_NODE"

    def _blank_local_state(updated_mono: float = 0.0) -> dict:
        return {
            "present": False,
            "label": "none",
            "category_rank": 0,
            "speed": 0.0,
            "distance": 0.0,
            "distance_confidence": 0,
            "emergency_active": False,
            "vehicle_uid": -1,
            "detected_at_ms": 0,
            "h_direction": "N",
            "seq": 0,
            "node": NODE_ID,
            "updated_mono": updated_mono,
        }

    def _blank_remote_state(received_mono: float = 0.0, node: str = peer) -> dict:
        return {
            "present": False,
            "label": "none",
            "category_rank": 0,
            "speed": 0.0,
            "distance": 0.0,
            "distance_confidence": 0,
            "emergency_active": False,
            "vehicle_uid": -1,
            "detected_at_ms": 0,
            "h_direction": "N",
            "seq": 0,
            "node": node,
            "received_mono": received_mono,
            "received_epoch_ms": 0,
            "ts_valid": False,
        }

    def _blank_decision(node_b: str = peer) -> dict:
        return {
            "decision_id": 0,
            "kind": "IDLE",
            "node_a": NODE_ID,
            "node_b": node_b,
            "signal_a": "",
            "signal_b": "",
            "emergency_mode": False,
            "traveling_node": "",
            "traveling_category_rank": 0,
            "traveling_emergency": False,
            "hold_until_mono": 0.0,
            "received_mono": 0.0,
            "acked_by_remote": False,
            "acked_remote_node": "",
            "acked_mono": 0.0,
        }

    def _set_snapshot(
        target_host: SmartRoadAlertHost,
        *,
        local_state: Optional[dict] = None,
        remote_state: Optional[dict] = None,
        decision: Optional[dict] = None,
        road_mode: str = "NARROW",
    ) -> None:
        with target_host._state_lock:
            target_host._road_mode = road_mode
            target_host._local_vehicle_state = _blank_local_state()
            if local_state is not None:
                target_host._local_vehicle_state.update(local_state)
            target_host._remote_vehicle_state = _blank_remote_state()
            if remote_state is not None:
                target_host._remote_vehicle_state.update(remote_state)
            target_host._current_decision = _blank_decision(
                str((remote_state or {}).get("node", peer))
            )
            if decision is not None:
                target_host._current_decision.update(decision)

    local0 = _mk_state(NODE_ID, False, "none", 0, 0.0, 0.0)
    remote0 = _mk_state(peer, False, "none", 0, 0.0, 0.0)
    cur = dict(host._current_decision)

    # Scenario 1: idle
    d = host._compute_leader_decision(0.0, local0, remote0, "NARROW", cur)
    assert d["kind"] == "IDLE"

    # Scenario 2: one-side
    local1 = _mk_state(NODE_ID, True, "motorcycle", 1, 30.0, 8.0)
    d = host._compute_leader_decision(0.0, local1, remote0, "NARROW", cur)
    assert d["kind"] == "ONE_SIDE"
    assert _sig_for(d, NODE_ID) == "GO"
    assert _sig_for(d, peer) == ""

    # Scenario 2b: one-side active emergency → EMERGENCY + STOP on loser
    local_one_em = _mk_state(NODE_ID, True, "ambulance", 3, 60.0, 8.0, emergency=True)
    d = host._compute_leader_decision(0.0, local_one_em, remote0, "NARROW", cur)
    assert d["kind"] == "EMERGENCY"
    assert d["emergency_mode"] is True
    assert _sig_for(d, NODE_ID) == "GO"
    assert _sig_for(d, peer) == "STOP"

    # Scenario 3: small vs small → both GO
    remote1 = _mk_state(peer, True, "bicycle", 1, 20.0, 9.0)
    d = host._compute_leader_decision(0.0, local1, remote1, "NARROW", cur)
    assert d["kind"] == "BOTH_GO"
    assert _sig_for(d, NODE_ID) == "GO" and _sig_for(d, peer) == "GO"

    # Small vs medium → both SLOW
    remote_med = _mk_state(peer, True, "car", 2, 25.0, 7.0)
    d = host._compute_leader_decision(0.0, local1, remote_med, "NARROW", cur)
    assert d["kind"] == "BOTH_SLOW"
    assert _sig_for(d, NODE_ID) == "SLOW" and _sig_for(d, peer) == "SLOW"

    # Medium vs medium wide → both SLOW
    local_med = _mk_state(NODE_ID, True, "car", 2, 25.0, 7.0)
    d = host._compute_leader_decision(0.0, local_med, remote_med, "WIDE", cur)
    assert d["kind"] == "BOTH_SLOW"

    # Medium vs medium narrow → STOP then ARBITRATED after pause
    host._arbitration_start_mono = None
    host._arbitration_context = None
    d0 = host._compute_leader_decision(0.0, local_med, remote_med, "NARROW", cur)
    assert d0["kind"] == "ARBITRATING"
    d1 = host._compute_leader_decision(ARBITRATION_PAUSE_S + 0.1, local_med, remote_med, "NARROW", cur)
    assert d1["kind"] == "ARBITRATED"

    # Large involvement → arbitration → large wins
    local_large = _mk_state(NODE_ID, True, "truck", 3, 20.0, 12.0)
    remote_med2 = _mk_state(peer, True, "car", 2, 20.0, 2.0)
    host._arbitration_start_mono = None
    host._arbitration_context = None
    d0 = host._compute_leader_decision(0.0, local_large, remote_med2, "NARROW", cur)
    assert d0["kind"] == "ARBITRATING"
    d1 = host._compute_leader_decision(ARBITRATION_PAUSE_S + 0.1, local_large, remote_med2, "NARROW", cur)
    assert d1["kind"] == "ARBITRATED"
    assert _sig_for(d1, NODE_ID) == "GO"

    # Large vs large → arbitration then closest/faster/timestamp winner
    local_large2 = _mk_state(NODE_ID, True, "truck", 3, 20.0, 7.0, detected_at_ms=1000)
    remote_large = _mk_state(peer, True, "bus", 3, 20.0, 11.0, detected_at_ms=1500)
    host._arbitration_start_mono = None
    host._arbitration_context = None
    d0 = host._compute_leader_decision(0.0, local_large2, remote_large, "NARROW", cur)
    assert d0["kind"] == "ARBITRATING"
    d1 = host._compute_leader_decision(ARBITRATION_PAUSE_S + 0.1, local_large2, remote_large, "NARROW", cur)
    assert d1["kind"] == "ARBITRATED"
    assert _sig_for(d1, NODE_ID) == "GO"

    # Emergency involvement vs non-emergency → immediate EMERGENCY
    local_em = _mk_state(NODE_ID, True, "ambulance", 3, 60.0, 10.0, emergency=True)
    d = host._compute_leader_decision(0.0, local_em, remote_med2, "NARROW", cur)
    assert d["kind"] == "EMERGENCY"
    assert _sig_for(d, NODE_ID) == "GO"
    assert _sig_for(d, peer) == "STOP"

    # Both emergencies → earlier detection timestamp wins
    remote_em_late = _mk_state(peer, True, "fire_truck", 3, 60.0, 10.0, emergency=True, detected_at_ms=2000)
    d = host._compute_leader_decision(0.0, local_em, remote_em_late, "NARROW", cur)
    assert d["kind"] == "EMERGENCY"
    assert _sig_for(d, NODE_ID) == "GO"

    # Both emergencies with invalid timestamp comparison → deterministic fallback
    local_em_tie = _mk_state(NODE_ID, True, "ambulance", 3, 60.0, 10.0, emergency=True, detected_at_ms=1000)
    remote_em_tie = _mk_state(peer, True, "fire_truck", 3, 60.0, 10.0, emergency=True, detected_at_ms=1000, ts_valid=False)
    d = host._compute_leader_decision(0.0, local_em_tie, remote_em_tie, "NARROW", cur)
    expected = host._stable_tie_winner(NODE_ID, peer, 1)
    assert d["kind"] == "EMERGENCY"
    assert _sig_for(d, expected) == "GO"

    # Hold override table coverage
    ts, ns, arb = host._hold_override(1, False, 1, False, "NARROW")
    assert (ts, ns, arb) == ("GO", "GO", False)
    ts, ns, arb = host._hold_override(1, False, 2, False, "NARROW")
    assert (ts, ns, arb) == ("SLOW", "SLOW", False)
    ts, ns, arb = host._hold_override(1, False, 3, False, "NARROW")
    assert (ts, ns, arb) == ("STOP", "STOP", True)
    ts, ns, arb = host._hold_override(1, False, 2, True, "NARROW")
    assert (ts, ns, arb) == ("STOP", "STOP", True)
    ts, ns, arb = host._hold_override(2, False, 1, False, "NARROW")
    assert (ts, ns, arb) == ("GO", "SLOW", False)
    ts, ns, arb = host._hold_override(2, False, 2, False, "WIDE")
    assert (ts, ns, arb) == ("SLOW", "SLOW", False)
    ts, ns, arb = host._hold_override(2, False, 2, False, "NARROW")
    assert (ts, ns, arb) == ("STOP", "STOP", True)
    ts, ns, arb = host._hold_override(2, False, 3, False, "NARROW")
    assert (ts, ns, arb) == ("STOP", "STOP", True)
    ts, ns, arb = host._hold_override(3, False, 1, False, "NARROW")
    assert (ts, ns, arb) == ("GO", "STOP", False)
    ts, ns, arb = host._hold_override(3, False, 1, True, "NARROW")
    assert (ts, ns, arb) == ("GO", "STOP", False)
    ts, ns, arb = host._hold_override(2, True, 3, True, "NARROW")
    assert (ts, ns, arb) == ("GO", "STOP", False)
    ts, ns, arb = host._hold_override(2, True, 2, True, "WIDE", "police_car", "police_car")
    assert (ts, ns, arb) == ("SLOW", "SLOW", False)

    # Thesis display: idle state shows NO VEHICLE
    display_host = SmartRoadAlertHost()
    _set_snapshot(
        display_host,
        local_state=_blank_local_state(updated_mono=100.0),
        remote_state=_blank_remote_state(received_mono=100.0),
        decision=_blank_decision(),
    )
    disp = display_host._thesis_display_state(100.0)
    assert disp["status"] == "NO VEHICLE"
    assert disp["label"] == "none"
    assert disp["emergency"] is False

    # Thesis display: one-side loser briefly shows remote info, then clears it during hold
    loser_host = SmartRoadAlertHost()
    _set_snapshot(
        loser_host,
        local_state=_blank_local_state(updated_mono=101.0),
        remote_state={
            **_blank_remote_state(received_mono=101.0),
            **_mk_state(peer, True, "car", 2, 28.0, 9.0, vehicle_uid=9),
            "received_mono": 101.0,
        },
        decision={
            **_blank_decision(peer),
            "decision_id": 1,
            "kind": "ONE_SIDE",
            "node_a": NODE_ID,
            "node_b": peer,
            "signal_a": "",
            "signal_b": "GO",
            "hold_until_mono": 115.0,
            "received_mono": 100.0,
        },
    )
    disp = loser_host._thesis_display_state(101.4)
    assert disp["status"] == ""
    assert disp["label"] == "car"
    disp = loser_host._thesis_display_state(102.2)
    assert disp["status"] == "STOP"
    assert disp["label"] == "car"
    assert disp["speed"] == 28.0

    # Thesis display: one-side emergency loser shows STOP with emergency banner
    emergency_display_host = SmartRoadAlertHost()
    _set_snapshot(
        emergency_display_host,
        local_state=_blank_local_state(updated_mono=100.0),
        remote_state={
            **_blank_remote_state(received_mono=100.0),
            **_mk_state(peer, True, "ambulance", 3, 60.0, 8.0, emergency=True, vehicle_uid=5),
            "received_mono": 100.0,
        },
        decision={
            **_blank_decision(peer),
            "decision_id": 1,
            "kind": "EMERGENCY",
            "node_a": NODE_ID,
            "node_b": peer,
            "signal_a": "STOP",
            "signal_b": "GO",
            "emergency_mode": True,
            "traveling_node": peer,
            "traveling_category_rank": 3,
            "traveling_emergency": True,
            "hold_until_mono": 115.0,
            "received_mono": 100.0,
            "acked_by_remote": True,
        },
    )
    disp = emergency_display_host._thesis_display_state(100.5)
    assert disp["status"] == "STOP"
    assert disp["label"] == "ambulance"
    assert disp["emergency"] is True

    # Agreement gating: leader does not surface GO before remote ACK.
    pre_ack_host = SmartRoadAlertHost()
    _set_snapshot(
        pre_ack_host,
        local_state={
            **_blank_local_state(updated_mono=100.0),
            **_mk_state(NODE_ID, True, "car", 2, 20.0, 8.0, vehicle_uid=21),
            "updated_mono": 100.0,
        },
        remote_state={
            **_blank_remote_state(received_mono=100.0),
            "node": peer,
            "received_mono": 100.0,
        },
        decision={
            **_blank_decision(peer),
            "decision_id": 4,
            "kind": "ONE_SIDE",
            "node_a": NODE_ID,
            "node_b": peer,
            "signal_a": "GO",
            "signal_b": "",
            "traveling_node": NODE_ID,
            "traveling_category_rank": 2,
            "traveling_emergency": False,
            "hold_until_mono": 115.0,
            "received_mono": 100.0,
            "acked_by_remote": False,
        },
    )
    disp = pre_ack_host._thesis_display_state(100.2)
    assert disp["status"] == ""
    disp = pre_ack_host._thesis_display_state(102.5)
    assert disp["status"] == "STOP"

    # Low-confidence fallback distance is ignored for winner tie-breaks.
    local_low_conf = _mk_state(NODE_ID, True, "truck", 3, 18.0, 4.0)
    local_low_conf["distance_confidence"] = 0
    remote_low_conf = _mk_state(peer, True, "bus", 3, 22.0, 10.0)
    remote_low_conf["distance_confidence"] = 0
    assert host._priority_winner_v2(local_low_conf, remote_low_conf, 9) == peer

    # Thesis audio: idle reminder repeats after 10 seconds, not immediately
    idle_audio_host = SmartRoadAlertHost()
    idle_audio_host._tts = _FakeTTS()
    _set_snapshot(
        idle_audio_host,
        local_state=_blank_local_state(updated_mono=0.0),
        remote_state=_blank_remote_state(received_mono=0.0),
        decision=_blank_decision(),
    )
    idle_audio_host._audio_tick(0.0, {"status": "NO VEHICLE"})
    assert idle_audio_host._tts.calls == []
    idle_audio_host._audio_tick(NO_VEHICLE_REPEAT_S + 0.1, {"status": "NO VEHICLE"})
    assert idle_audio_host._tts.calls == [("THERE'S NO VEHICLE ON THE OPPOSITE SIDE.", False)]

    # Thesis audio: emergency loser prompt
    emergency_stop_audio_host = SmartRoadAlertHost()
    emergency_stop_audio_host._tts = _FakeTTS()
    _set_snapshot(
        emergency_stop_audio_host,
        local_state=_blank_local_state(updated_mono=100.0),
        remote_state={
            **_blank_remote_state(received_mono=100.0),
            **_mk_state(peer, True, "ambulance", 3, 60.0, 8.0, emergency=True),
            "received_mono": 100.0,
        },
        decision={
            **_blank_decision(peer),
            "decision_id": 1,
            "kind": "EMERGENCY",
            "node_a": NODE_ID,
            "node_b": peer,
            "signal_a": "STOP",
            "signal_b": "GO",
            "emergency_mode": True,
            "traveling_node": peer,
            "traveling_category_rank": 3,
            "traveling_emergency": True,
            "hold_until_mono": 115.0,
            "received_mono": 100.0,
            "acked_by_remote": True,
        },
    )
    emergency_stop_audio_host._audio_tick(100.0, {"status": "STOP"})
    assert emergency_stop_audio_host._tts.calls == [
        ("STOP! EMERGENCY VEHICLE APPROACHING FROM THE OPPOSITE SIDE, PLEASE WAIT!", True)
    ]

    # Thesis audio: emergency winner prompt after local detection already announced
    emergency_go_audio_host = SmartRoadAlertHost()
    emergency_go_audio_host._tts = _FakeTTS()
    emergency_go_audio_host._last_announced_local_uid = 42
    _set_snapshot(
        emergency_go_audio_host,
        local_state={
            **_blank_local_state(updated_mono=100.0),
            **_mk_state(NODE_ID, True, "ambulance", 3, 60.0, 8.0, emergency=True, vehicle_uid=42),
            "updated_mono": 100.0,
        },
        remote_state={
            **_blank_remote_state(received_mono=100.0),
            **_mk_state(peer, True, "car", 2, 25.0, 9.0),
            "received_mono": 100.0,
        },
        decision={
            **_blank_decision(peer),
            "decision_id": 1,
            "kind": "EMERGENCY",
            "node_a": NODE_ID,
            "node_b": peer,
            "signal_a": "GO",
            "signal_b": "STOP",
            "emergency_mode": True,
            "traveling_node": NODE_ID,
            "traveling_category_rank": 3,
            "traveling_emergency": True,
            "hold_until_mono": 115.0,
            "received_mono": 100.0,
            "acked_by_remote": True,
        },
    )
    emergency_go_audio_host._audio_tick(100.0, {"status": "GO"})
    assert emergency_go_audio_host._tts.calls == [
        ("GO! THE VEHICLE DETECTED FROM THE OPPOSITE SIDE IS ON STANDBY.", True)
    ]

    # Thesis agreement: remote decision ACK marks agreement complete on the leader.
    ack_host = SmartRoadAlertHost()
    with ack_host._state_lock:
        ack_host._current_decision.update({
            "decision_id": 7,
            "node_a": NODE_ID,
            "node_b": peer,
            "acked_by_remote": False,
            "acked_remote_node": "",
            "acked_mono": 0.0,
        })
    ack_host._on_remote_ack({"type": "ack", "for": "decision", "id": 7, "node": peer})
    with ack_host._state_lock:
        assert ack_host._current_decision["acked_by_remote"] is True
        assert ack_host._current_decision["acked_remote_node"] == peer
        assert float(ack_host._current_decision["acked_mono"]) > 0.0

    print("Decision tests passed.")


def main() -> None:
    if "--decision-tests" in sys.argv:
        _run_decision_tests()
        return
    host = SmartRoadAlertHost()

    def _shutdown(sig: int, _frame) -> None:
        logger.info("Signal %d received — initiating shutdown.", sig)
        host.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    host.start()


if __name__ == "__main__":
    main()

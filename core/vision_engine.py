"""Camera capture + MediaPipe HandLandmarker running in a dedicated thread.

The engine owns the camera and the landmarker (MediaPipe graphs must stay on the
thread that created them). Each processed frame is delivered through `on_frame`
on the vision thread. When debug rendering is on, an annotated BGR frame is
published to a single-slot buffer; the *main* thread is responsible for showing
it, because OpenCV HighGUI is not thread-safe (and must be on the main thread on macOS).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision

from core.config import VisionConfig
from core.smoothing import OneEuroFilter

logger = logging.getLogger(__name__)

HAND_CONNECTIONS = tuple(
    (c.start, c.end) for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS
)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class HandObservation:
    landmarks: np.ndarray  # (21, 3) smoothed; x, y normalized to [0, 1], y down
    raw_landmarks: np.ndarray  # (21, 3) unfiltered model output
    handedness: str
    score: float
    aspect: float  # frame width / height, for isotropic distance math


@dataclass(frozen=True)
class VisionFrame:
    hands: Tuple[HandObservation, ...]
    timestamp: float  # time.monotonic() at capture
    fps: float

    @property
    def hand(self) -> Optional[HandObservation]:
        return self.hands[0] if self.hands else None


def ensure_model(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    logger.info("Downloading hand landmarker model to %s", path)
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
    return path


def open_camera(config: VisionConfig) -> cv2.VideoCapture:
    if sys.platform == "darwin":
        backend = cv2.CAP_AVFOUNDATION
    elif sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_V4L2
    cap = cv2.VideoCapture(config.camera_index, backend)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Cannot open camera {config.camera_index}. Check the index and that this "
            "terminal has camera permission (macOS: System Settings > Privacy & Security > Camera)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    cap.set(cv2.CAP_PROP_FPS, config.max_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def create_landmarker(config: VisionConfig) -> mp_vision.HandLandmarker:
    model_path = ensure_model(config.model_path, config.model_url)
    delegate = BaseOptions.Delegate.GPU if config.use_gpu else BaseOptions.Delegate.CPU
    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path), delegate=delegate),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_hand_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_hand_overlay(image: np.ndarray, hands: Optional[Sequence[HandObservation]]) -> None:
    if not hands:
        return
    h, w = image.shape[:2]
    for hand in hands:
        # Right hand moves the cursor (green). Left hand clicks (orange).
        color = (0, 180, 255) if hand.handedness == "Left" else (0, 200, 0)
        pts = (hand.landmarks[:, :2] * (w, h)).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(image, tuple(p), 3, (0, 0, 255), -1, cv2.LINE_AA)


class _FpsMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last: Optional[float] = None
        self.fps = 0.0

    def tick(self, now: float) -> float:
        if self._last is not None:
            instant = 1.0 / max(now - self._last, 1e-6)
            self.fps = instant if self.fps == 0.0 else (
                self._smoothing * self.fps + (1.0 - self._smoothing) * instant
            )
        self._last = now
        return self.fps


class VisionEngine:
    def __init__(
        self,
        config: VisionConfig,
        on_frame: Callable[[VisionFrame], None],
        render_debug: bool = False,
    ) -> None:
        self._config = config
        self._on_frame = on_frame
        self._render_debug = render_debug
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)
        self._filters: Dict[str, OneEuroFilter] = {}
        self._debug_lock = threading.Lock()
        self._debug_image: Optional[np.ndarray] = None
        self.error: Optional[BaseException] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the camera and model are initialized (or the thread failed)."""
        return self._ready.wait(timeout)

    def latest_debug_image(self) -> Optional[np.ndarray]:
        with self._debug_lock:
            image, self._debug_image = self._debug_image, None
        return image

    def _run(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        try:
            landmarker = create_landmarker(self._config)
            cap = open_camera(self._config)
            self._ready.set()
            with landmarker:
                cap = self._loop(cap, landmarker)
        except Exception as exc:
            self.error = exc
            logger.exception("Vision engine stopped")
        finally:
            self._ready.set()
            if cap is not None:
                cap.release()

    def _loop(
        self, cap: cv2.VideoCapture, landmarker: mp_vision.HandLandmarker
    ) -> cv2.VideoCapture:
        min_interval = 1.0 / self._config.max_fps
        # The Metal delegate only accepts 4-channel images.
        if self._config.use_gpu:
            color_code, image_format = cv2.COLOR_BGR2RGBA, mp.ImageFormat.SRGBA
        else:
            color_code, image_format = cv2.COLOR_BGR2RGB, mp.ImageFormat.SRGB
        meter = _FpsMeter()
        failed_reads = 0
        last_ts_ms = -1

        while not self._stop.is_set():
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= self._config.reopen_after_failed_reads:
                    logger.warning("Camera stopped delivering frames; reopening")
                    cap.release()
                    cap = open_camera(self._config)
                    failed_reads = 0
                time.sleep(0.01)
                continue
            failed_reads = 0

            if self._config.mirror:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # VIDEO mode requires strictly increasing timestamps.
            ts_ms = max(int(started * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            converted = cv2.cvtColor(frame, color_code)
            result = landmarker.detect_for_video(
                mp.Image(image_format=image_format, data=converted), ts_ms
            )

            hands = self._extract_hands(result, started, w / h)
            fps = meter.tick(started)

            if self._render_debug:
                draw_hand_overlay(frame, hands)
                cv2.putText(frame, f"{fps:5.1f} FPS", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                with self._debug_lock:
                    self._debug_image = frame

            self._on_frame(VisionFrame(hands=hands, timestamp=started, fps=fps))

            remaining = min_interval - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)
        return cap

    def _filter_for(self, key: str) -> OneEuroFilter:
        filt = self._filters.get(key)
        if filt is None:
            cfg = self._config
            filt = OneEuroFilter(cfg.smoothing_min_cutoff, cfg.smoothing_beta, cfg.smoothing_d_cutoff)
            self._filters[key] = filt
        return filt

    def _extract_hands(
        self, result: mp_vision.HandLandmarkerResult, timestamp: float, aspect: float
    ) -> Tuple[HandObservation, ...]:
        if not result.hand_landmarks:
            for filt in self._filters.values():
                filt.reset()
            return ()
        hands = []
        seen = set()
        for index, landmarks in enumerate(result.hand_landmarks):
            raw = np.array([(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float64)
            category = None
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
            name = category.category_name if category else "Unknown"
            # The frame is mirrored into a selfie view, which flips chirality.
            if self._config.mirror and name == "Left":
                name = "Right"
            elif self._config.mirror and name == "Right":
                name = "Left"
            seen.add(name)
            hands.append(HandObservation(
                landmarks=self._filter_for(name)(raw, timestamp),
                raw_landmarks=raw,
                handedness=name,
                score=float(category.score) if category else 0.0,
                aspect=aspect,
            ))
        for key, filt in self._filters.items():
            if key not in seen:
                filt.reset()
        return tuple(hands)

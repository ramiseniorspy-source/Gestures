"""Sanity check the vision stack before running the full system.

Stages:
    1. Camera: open the device and read frames (resolution, raw FPS).
    2. Model: download/load the MediaPipe hand landmarker.
    3. Live detection: landmarks, FPS, detection rate, plus the per-frame pose and
       pinch ratio the gesture detector sees (use this to tune GestureConfig).

Nothing is sent to the OS; no mouse or keyboard events are generated.

Usage:
    python test_camera.py                       # window, until q/Esc
    python test_camera.py --headless --seconds 10
Exit codes: 0 ok, 1 camera/model failure, 2 camera ok but no hand ever detected.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import threading
import time
from typing import Optional

import cv2

from core.config import AppConfig, GestureConfig, VisionConfig
from core.gesture_detector import HandFeatures, classify, measure
from core.overlay import WINDOW_NAME, draw_hud
from core.vision_engine import CameraError, VisionEngine, VisionFrame, create_landmarker, open_camera


def check_camera(config: VisionConfig, frames: int = 30) -> bool:
    print(f"[1/3] Opening camera {config.camera_index} ...")
    try:
        cap = open_camera(config)
    except CameraError as exc:
        print(f"  FAIL: {exc}")
        return False
    try:
        ok_frames, started, shape = 0, time.monotonic(), None
        for _ in range(frames):
            ok, frame = cap.read()
            if ok:
                ok_frames += 1
                shape = frame.shape
        elapsed = time.monotonic() - started
    finally:
        cap.release()
    if ok_frames == 0 or shape is None:
        print("  FAIL: camera opened but returned no frames (permission or device busy?)")
        return False
    print(f"  OK: {shape[1]}x{shape[0]}, {ok_frames}/{frames} frames, ~{ok_frames / elapsed:.1f} FPS raw")
    return True


def check_model(config: VisionConfig) -> bool:
    print("[2/3] Loading hand landmarker model ...")
    try:
        create_landmarker(config).close()
    except Exception as exc:
        print(f"  FAIL: {exc}")
        return False
    print(f"  OK: {config.model_path}")
    return True


class _Stats:
    def __init__(self, gesture: GestureConfig) -> None:
        self._gesture = gesture
        self._lock = threading.Lock()
        self.frames = 0
        self.with_hand = 0
        self.fps = 0.0
        self.features: Optional[HandFeatures] = None
        self.pose_name = "-"

    def on_frame(self, frame: VisionFrame) -> None:
        features = measure(frame.hand, self._gesture) if frame.hand else None
        pose = classify(features, self._gesture.pinch_enter, self._gesture) if features else None
        with self._lock:
            self.frames += 1
            self.with_hand += frame.hand is not None
            self.fps = frame.fps
            self.features = features
            self.pose_name = pose.name if pose else "no hand"


def check_detection(config: AppConfig, seconds: float, headless: bool) -> int:
    print("[3/3] Live detection" + ("" if headless else " (show your hand; q/Esc to finish)"))
    stats = _Stats(config.gesture)
    engine = VisionEngine(config.vision, on_frame=stats.on_frame, render_debug=not headless)
    engine.start()
    engine.wait_ready()
    deadline = time.monotonic() + seconds if seconds > 0 else float("inf")
    try:
        if headless:
            while engine.is_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            while engine.is_alive() and time.monotonic() < deadline:
                image = engine.latest_debug_image()
                if image is not None:
                    draw_hud(image, config.gesture, stats.features, [f"pose {stats.pose_name}"])
                    cv2.imshow(WINDOW_NAME, image)
                if (cv2.waitKey(10) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        engine.stop()
        engine.join(timeout=3.0)
        if not headless:
            cv2.destroyAllWindows()

    if engine.error is not None:
        print(f"  FAIL: {engine.error}")
        return 1
    rate = 100.0 * stats.with_hand / stats.frames if stats.frames else 0.0
    print(f"  {stats.frames} frames, {stats.fps:.1f} FPS pipeline, hand in {rate:.0f}% of frames")
    if stats.with_hand == 0:
        print("  WARN: no hand detected. Check lighting and keep your hand 30-80 cm from the camera.")
        return 2
    print("  OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Webcam + hand landmark sanity check")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="stop after N seconds (default: until q, or 10 s when headless)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    base = AppConfig()
    config = dataclasses.replace(
        base, vision=dataclasses.replace(base.vision, camera_index=args.camera)
    )
    if not check_camera(config.vision) or not check_model(config.vision):
        return 1
    seconds = args.seconds or (10.0 if args.headless else 0.0)
    return check_detection(config, seconds, args.headless)


if __name__ == "__main__":
    sys.exit(main())

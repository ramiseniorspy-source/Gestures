"""Debug HUD drawing shared by main.py and test_camera.py (main thread only)."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from core.config import GestureConfig
from core.gesture_detector import HandFeatures

WINDOW_NAME = "Gestures (q / Esc to quit)"


def draw_hud(
    image: np.ndarray,
    config: GestureConfig,
    features: Optional[HandFeatures],
    lines: Sequence[str],
) -> None:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = config.active_region
    cv2.rectangle(image, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                  (255, 180, 0), 1, cv2.LINE_AA)

    text = list(lines)
    if features is not None:
        fingers = "".join(n if e else "-" for n, e in zip("IMRP", features.extended))
        text.append(f"fingers {fingers}  pinch {features.pinch_ratio:.2f}  "
                    f"reach {features.index_reach:.2f}")
    for i, line in enumerate(text):
        y = 24 + i * 22
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)

"""One Euro filter: an adaptive exponential moving average for noisy real-time signals.

At low speed the cutoff frequency stays near `min_cutoff` (heavy smoothing, no
jitter); as speed rises the cutoff grows by `beta * |velocity|` (light smoothing,
low lag). Casiez et al., CHI 2012. Vectorized so all 21x3 landmarks filter at once.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev = 0.0

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self._x_prev is None or self._dx_prev is None:
            self._x_prev = x.astype(np.float64, copy=True)
            self._dx_prev = np.zeros_like(self._x_prev)
            self._t_prev = t
            return self._x_prev.copy()

        dt = max(t - self._t_prev, 1e-4)
        dx = (x - self._x_prev) / dt
        a_d = _alpha(np.full_like(dx, self._d_cutoff), dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self._min_cutoff + self._beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat.copy()

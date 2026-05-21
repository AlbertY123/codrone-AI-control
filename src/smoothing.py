"""Signal smoothing utilities.

OneEuroFilter — adaptive low-pass that suppresses jitter while passing fast
motion (Casiez et al., CHI 2012). Great for landmark coords and stick outputs.

ScalarEMA — plain exponential moving average for things like FPS counters.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class OneEuroFilter:
    min_cutoff: float = 1.0   # Hz — lower = smoother but more lag
    beta: float = 0.02        # cutoff slope w.r.t. derivative
    d_cutoff: float = 1.0
    _x_prev: Optional[float] = None
    _dx_prev: float = 0.0
    _t_prev: Optional[float] = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.monotonic()
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x
        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


@dataclass
class ScalarEMA:
    alpha: float = 0.2
    _value: Optional[float] = None

    def __call__(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value if self._value is not None else 0.0

    def reset(self) -> None:
        self._value = None

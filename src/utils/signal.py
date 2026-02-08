"""
Signal processing utilities
"""

import math
import time


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        # Parameters
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff

        # Previous values
        self.x_prev: float | None = None
        self.dx_prev: float | None = None
        self.t_prev: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t

            return x

        dt = t - self.t_prev

        # Prevent division by zero
        if dt <= 0:
            return self.x_prev

        # Compute filtered derivative of the signal
        dx = (x - self.x_prev) / dt
        dx_hat = self._exp_smooth(
            x=dx, x_prev=self.dx_prev,
            dt=dt, cutoff=self.d_cutoff
        )

        # Compute dynamic cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter signal
        x_hat = self._exp_smooth(
            x=x, x_prev=self.x_prev,
            dt=dt, cutoff=cutoff
        )

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    @staticmethod
    def _exp_smooth(x: float, x_prev: float, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)

        return alpha * x + (1.0 - alpha) * x_prev

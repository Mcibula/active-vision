"""
Signal processing utilities
"""

import math
import time


class OneEuroFilter:
    """
    Simple Online 1€ Filter implementation for filtering trajectory signals.

    Adapted from:
        https://github.com/jaantollander/OneEuroFilter
        and https://github.com/casiez/OneEuroFilter

    See also: https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        """
        Initialize a 1€ Filter instance

        :param min_cutoff: Minimum cutoff frequency in Hz; lower values remove more jitter
        :param beta: Speed coefficient
        :param d_cutoff: Derivative cutoff frequency in Hz
        """

        # Parameters
        self.min_cutoff: float = min_cutoff
        self.beta: float = beta
        self.d_cutoff: float = d_cutoff

        # Previous values
        self.x_prev: float | None = None
        self.dx_prev: float | None = None
        self.t_prev: float | None = None

    def __call__(self, x: float, t: float | None = None) -> float:
        """
        Compute the filtered signal for the given point `x` at time `t`

        :param x: Signal point value
        :param t: Timestamp in seconds
        :return: Filtered signal point value
        """

        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t

            return x

        self.x_prev: float
        self.dx_prev: float
        self.t_prev: float

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
        """
        Perform exponential smoothing on `x`

        :param x: Current point value
        :param x_prev: Point value in the previous time step
        :param dt: Time difference between `x` and `x_prev` in seconds
        :param cutoff: Cutoff frequency in Hz
        :return: Smoothed `x`
        """

        # Compute the smoothing factor
        tau = 1.0 / (2.0 * math.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)

        return alpha * x + (1.0 - alpha) * x_prev

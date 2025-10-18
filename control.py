# control.py
import numpy as np

class PDController:
    """
    Simple discrete PD controller.
    compute(error) returns control u = kp*error + kd*(error - last_error)
    """

    def __init__(self, kp: float = 0.15, kd: float = 0.6, u_min=None, u_max=None):
        self.kp = kp
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        self._last_error = None

    def reset(self):
        """Reset stored error history (for fresh runs)."""
        self._last_error = None

    def compute(self, error: float) -> float:
        """Compute the control action for the given error."""
        d = 0.0
        if self._last_error is not None:
            d = error - self._last_error
        u = self.kp * error + self.kd * d
        self._last_error = float(error)
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)
        return float(u)
"""Hard feasibility filtering and learned soft constraints.

Parent proposal, section 9: known damage-relevant constraints are enforced
exactly and are never learned by failure; only non-damaging empirical
constraints (measurement-quality regions) are modelled statistically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ControlConfig
from .control import ControlMap
from .gp import GaussianProcess


@dataclass
class HardConstraints:
    """Exact feasibility domain U_hard (Step 1 design, section 6).

    Checks, for a candidate physical correction u_B:
      1. actuator travel limits (the control box),
      2. minimum disk/mirror gaps (collision avoidance),
      3. maximum safe step from the current achieved geometry.

    A conservative safety margin shrinks the gap limit rather than
    letting uncertainty be explored experimentally.
    """

    control_map: ControlMap
    cfg: ControlConfig
    gap_safety_margin: float = 0.0

    def feasible(
        self,
        u_B: np.ndarray,
        current_achieved: np.ndarray | None = None,
    ) -> bool:
        u_B = np.asarray(u_B, dtype=float)
        if not self.control_map.within_travel_limits(u_B):
            return False
        if self.control_map.min_gap(u_B) < self.cfg.min_gap + self.gap_safety_margin:
            return False
        if current_achieved is not None:
            x_new = self.control_map.to_normalized(u_B)
            x_cur = self.control_map.to_normalized(np.asarray(current_achieved))
            if np.any(np.abs(x_new - x_cur) > self.cfg.max_step_normalized + 1e-12):
                return False
        return True

    def filter(
        self,
        candidates: np.ndarray,
        current_achieved: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boolean mask of feasible candidates (physical units)."""
        return np.array(
            [self.feasible(u, current_achieved) for u in np.atleast_2d(candidates)]
        )


@dataclass
class SoftConstraintModel:
    """Learned probability of a *non-damaging* measurement success.

    P_soft-safe(u) (Step 1 design, section 15).  Failures (bad coupling,
    unreliable gradient-method determination, parasitic modes) are recorded
    as 0/1 outcomes in normalized control space and smoothed with a GP;
    the predicted success probability multiplies the acquisition.
    """

    lengthscale: float = 0.3
    prior_success: float = 0.9
    _x: list = field(default_factory=list)
    _y: list = field(default_factory=list)

    def observe(self, x_normalized: np.ndarray, success: bool) -> None:
        self._x.append(np.asarray(x_normalized, dtype=float))
        self._y.append(1.0 if success else 0.0)

    @property
    def n_observations(self) -> int:
        return len(self._y)

    def probability_feasible(self, x_normalized: np.ndarray) -> np.ndarray:
        x_star = np.atleast_2d(np.asarray(x_normalized, dtype=float))
        if len(self._y) < 3:
            return np.full(len(x_star), self.prior_success)
        x = np.stack(self._x)
        y = np.array(self._y) - self.prior_success
        gp = GaussianProcess(amplitude=0.5, lengthscales=np.full(x.shape[1], self.lengthscale))
        gp.fit(x, y, noise_sd=np.full(len(y), 0.25))
        mean, _ = gp.predict(x_star)
        return np.clip(self.prior_success + mean, 0.02, 1.0)

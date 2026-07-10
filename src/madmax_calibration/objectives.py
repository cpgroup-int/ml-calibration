"""Scalar calibration objectives J derived from the boost-factor curve.

The parent proposal (section 6) requires the objective to be a physics
figure of merit tied to the MADMAX area-law trade-off, chosen with the
physics team.  Three candidates are implemented; the active one is picked
in :class:`~madmax_calibration.config.CalibrationConfig` (default:
scan-rate proxy).  Curve-to-objective uncertainty propagation is Monte
Carlo (Step 4 design, section 9.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def scan_rate_proxy(beta2: np.ndarray) -> float:
    """Scan-rate proxy over W: mean of beta^4 (scan rate ~ beta^4).

    Returned in units of 1e-4 * <beta^4> so typical values are O(1-100).
    """
    b = np.asarray(beta2, dtype=float)
    return float(np.mean(b**2) * 1e-4)


def smooth_min(beta2: np.ndarray, tau: float = 10.0) -> float:
    """Soft minimum of beta^2 over W (broadband robustness objective)."""
    b = np.asarray(beta2, dtype=float)
    return float(-tau * np.log(np.mean(np.exp(-b / tau))))


def peak_boost(beta2: np.ndarray) -> float:
    """Peak of beta^2 over W."""
    return float(np.max(beta2))


_OBJECTIVES = {
    "scan_rate": scan_rate_proxy,
    "smooth_min": smooth_min,
    "peak": peak_boost,
}


@dataclass
class Objective:
    """Named scalar objective J with MC uncertainty propagation."""

    name: str = "scan_rate"

    def __post_init__(self) -> None:
        if self.name not in _OBJECTIVES:
            raise ValueError(f"unknown objective '{self.name}'; options: {list(_OBJECTIVES)}")
        self._fn = _OBJECTIVES[self.name]

    @property
    def objective_id(self) -> str:
        return f"J_{self.name}"

    def __call__(self, beta2: np.ndarray) -> float:
        return self._fn(beta2)

    def with_uncertainty(
        self,
        beta2: np.ndarray,
        beta2_sigma: np.ndarray,
        rng: np.random.Generator | None = None,
        n_samples: int = 256,
        correlated_fraction: float = 0.5,
    ) -> tuple[float, float]:
        """Propagate curve uncertainty to (J, sigma_J).

        ``correlated_fraction`` splits the per-bin sigma into a shared
        (normalization-like, fully correlated) component and an
        independent per-bin component (Step 4 design, section 9.4).
        """
        rng = rng or np.random.default_rng(0)
        beta2 = np.asarray(beta2, dtype=float)
        sig = np.broadcast_to(np.asarray(beta2_sigma, dtype=float), beta2.shape)
        shared = np.sqrt(correlated_fraction) * sig
        indep = np.sqrt(1.0 - correlated_fraction) * sig
        js = np.empty(n_samples)
        for s in range(n_samples):
            curve = (
                beta2
                + shared * rng.standard_normal()
                + indep * rng.standard_normal(beta2.shape)
            )
            js[s] = self._fn(np.clip(curve, 0.0, None))
        return float(self._fn(beta2)), float(np.std(js, ddof=1))

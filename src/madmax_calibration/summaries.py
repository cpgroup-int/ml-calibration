"""Curve summaries for curve-summary-level inference (roadmap Phase 1.1).

An HF measurement returns a full boost curve beta^2(nu).  At scalar level
(objective J only) a frequency shift, an amplitude loss and a bandwidth
change are indistinguishable — the source of the near-degenerate
detector-state directions seen in scalar-mode inference.  This module
compresses the curve into a small vector of *physically meaningful,
smooth* summaries (Step 5 design note, observation level 4.2):

======== ===========================================================
``J``          the configured scalar objective (component 0)
``log_peak``   log of a smooth power-mean peak proxy
``centroid``   beta^4-weighted band centroid, in window half-widths
``bandwidth``  beta^4-weighted RMS width, in window half-widths
``flatness``   coefficient of variation of beta^2 over the window
======== ===========================================================

All summaries are smooth functions of the curve (no argmax), so the
joint-MAP optimizer in Step 5 gets usable finite-difference gradients.
Measurement uncertainty is propagated by Monte Carlo with the same
shared-normalization + independent per-bin split used for the scalar
objective (Step 4 design, section 9.4); per-component standard
deviations are returned (cross-summary correlations are neglected at
inference Level A and documented as such).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .objectives import Objective

SUMMARY_NAMES = ("J", "log_peak", "centroid", "bandwidth", "flatness")


@dataclass
class CurveSummarizer:
    """Maps a boost curve on a fixed frequency grid to the summary vector."""

    objective: Objective
    freqs: np.ndarray
    peak_power: int = 8      # power-mean order of the smooth peak proxy
    weight_power: int = 2    # centroid/bandwidth weights: (beta^2)^2 = beta^4
    _center: float = field(init=False, default=0.0)
    _half_width: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float)
        self._center = 0.5 * (self.freqs[0] + self.freqs[-1])
        self._half_width = max(0.5 * (self.freqs[-1] - self.freqs[0]), 1.0)

    @property
    def names(self) -> tuple:
        return SUMMARY_NAMES

    @property
    def dim(self) -> int:
        return len(SUMMARY_NAMES)

    def __call__(self, beta2: np.ndarray) -> np.ndarray:
        b = np.clip(np.asarray(beta2, dtype=float), 1e-12, None)
        j = self.objective(b)
        # Smooth peak proxy: power mean of order p approaches max(b).
        peak = float(np.mean(b**self.peak_power) ** (1.0 / self.peak_power))
        log_peak = float(np.log(peak))
        # beta^4-weighted first and second frequency moments.
        w = b**self.weight_power
        w_sum = float(np.sum(w))
        centroid_f = float(np.sum(self.freqs * w) / w_sum)
        centroid = (centroid_f - self._center) / self._half_width
        var = float(np.sum(w * (self.freqs - centroid_f) ** 2) / w_sum)
        bandwidth = float(np.sqrt(max(var, 0.0)) / self._half_width)
        flatness = float(np.std(b) / np.mean(b))
        return np.array([j, log_peak, centroid, bandwidth, flatness])

    def with_uncertainty(
        self,
        beta2: np.ndarray,
        beta2_sigma: np.ndarray,
        rng: np.random.Generator | None = None,
        n_samples: int = 256,
        correlated_fraction: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo propagation of curve uncertainty to (z, sigma_z).

        Component 0 of ``z`` is the scalar objective, so callers can use
        ``z[0], sigma_z[0]`` in place of the scalar-only propagation.
        """
        rng = rng or np.random.default_rng(0)
        beta2 = np.asarray(beta2, dtype=float)
        sig = np.broadcast_to(np.asarray(beta2_sigma, dtype=float), beta2.shape)
        shared = np.sqrt(correlated_fraction) * sig
        indep = np.sqrt(1.0 - correlated_fraction) * sig
        zs = np.empty((n_samples, self.dim))
        for s in range(n_samples):
            curve = (
                beta2
                + shared * rng.standard_normal()
                + indep * rng.standard_normal(beta2.shape)
            )
            zs[s] = self(np.clip(curve, 0.0, None))
        return self(beta2), np.std(zs, axis=0, ddof=1)


REFLECTIVITY_SUMMARY_NAMES = ("refl_mean", "refl_slope", "gd_centroid", "gd_mean_ns")


@dataclass
class ReflectivitySummarizer:
    """Summaries of the low-fidelity reflectivity measurement.

    Compresses the power reflectivity |Gamma|^2(nu) and the group delay
    tau_g(nu) into a small vector (roadmap Phase 1.2):

    ============= =====================================================
    ``refl_mean``   mean power reflectivity over W (dielectric loss)
    ``refl_slope``  linear slope of |Gamma|^2 across the window
    ``gd_centroid`` tau_g^2-weighted band centroid, in half-widths
                    (resonance position -> geometry)
    ``gd_mean_ns``  mean group delay in nanoseconds (photon storage)
    ============= =====================================================

    All components are smooth in the underlying geometry, and Monte-Carlo
    uncertainty propagation mirrors :class:`CurveSummarizer`.
    """

    freqs: np.ndarray
    _center: float = field(init=False, default=0.0)
    _half_width: float = field(init=False, default=1.0)
    _xi: np.ndarray = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.freqs = np.asarray(self.freqs, dtype=float)
        self._center = 0.5 * (self.freqs[0] + self.freqs[-1])
        self._half_width = max(0.5 * (self.freqs[-1] - self.freqs[0]), 1.0)
        self._xi = (self.freqs - self._center) / self._half_width   # [-1, 1]

    @property
    def names(self) -> tuple:
        return REFLECTIVITY_SUMMARY_NAMES

    @property
    def dim(self) -> int:
        return len(REFLECTIVITY_SUMMARY_NAMES)

    def __call__(self, refl: np.ndarray, gd: np.ndarray) -> np.ndarray:
        refl = np.asarray(refl, dtype=float)
        gd = np.asarray(gd, dtype=float)
        refl_mean = float(np.mean(refl))
        # Least-squares slope against the normalized frequency coordinate.
        xi = self._xi
        refl_slope = float(np.sum((xi - xi.mean()) * refl) / np.sum((xi - xi.mean()) ** 2))
        w = gd**2
        w_sum = float(np.sum(w)) + 1e-30
        gd_centroid = float(np.sum(xi * w) / w_sum)
        gd_mean_ns = float(np.mean(gd) * 1e9)
        return np.array([refl_mean, refl_slope, gd_centroid, gd_mean_ns])

    def with_uncertainty(
        self,
        refl: np.ndarray,
        refl_sigma: np.ndarray,
        gd: np.ndarray,
        gd_sigma: np.ndarray,
        rng: np.random.Generator | None = None,
        n_samples: int = 256,
        correlated_fraction: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng(0)
        refl = np.asarray(refl, dtype=float)
        gd = np.asarray(gd, dtype=float)
        r_sig = np.broadcast_to(np.asarray(refl_sigma, dtype=float), refl.shape)
        g_sig = np.broadcast_to(np.asarray(gd_sigma, dtype=float), gd.shape)
        r_shared = np.sqrt(correlated_fraction) * r_sig
        r_indep = np.sqrt(1.0 - correlated_fraction) * r_sig
        g_shared = np.sqrt(correlated_fraction) * g_sig
        g_indep = np.sqrt(1.0 - correlated_fraction) * g_sig
        zs = np.empty((n_samples, self.dim))
        for s in range(n_samples):
            r = refl + r_shared * rng.standard_normal() + r_indep * rng.standard_normal(refl.shape)
            g = gd + g_shared * rng.standard_normal() + g_indep * rng.standard_normal(gd.shape)
            zs[s] = self(np.clip(r, 0.0, None), g)
        return self(refl, gd), np.std(zs, axis=0, ddof=1)

"""Simulation-based calibration for the amortized posterior (Phase 2.2).

Simulation-based calibration (SBC) is the standard correctness check for
an amortized/approximate posterior: if the posterior is calibrated, then
for data simulated from ``theta* ~ prior`` the rank of ``theta*`` among
posterior samples is **uniformly distributed**.  This module runs SBC and
the associated empirical-coverage check for
:class:`~madmax_calibration.amortized.AmortizedPosterior`, including
**under injected discrepancy** (the misspecification the posterior was
trained to be robust to) — the acceptance instrument the roadmap
specifies in place of the Laplace-specific residual tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amortized import AmortizedPosterior, build_conditioning
from .objectives import Objective
from .simulator import DetectorState
from .summaries import CurveSummarizer, ReflectivitySummarizer

THETA_NAMES = ("z_offset", "compression", "log_loss")


@dataclass
class SBCResult:
    """Ranks and coverage of an SBC run."""

    ranks: np.ndarray                  # (n_trials, 3) integer ranks in [0, n_post]
    n_posterior: int
    coverage_1sigma: np.ndarray        # (3,)
    coverage_2sigma: np.ndarray        # (3,)
    mean_abs_error: np.ndarray         # (3,) physical units
    ks_uniform_p: np.ndarray           # (3,) KS p-value vs uniform ranks
    discrepancy_injection: float

    def summary(self) -> dict:
        return {
            "n_trials": len(self.ranks),
            "n_posterior": self.n_posterior,
            "coverage_1sigma": dict(zip(THETA_NAMES, np.round(self.coverage_1sigma, 3))),
            "coverage_2sigma": dict(zip(THETA_NAMES, np.round(self.coverage_2sigma, 3))),
            "mean_abs_error": dict(zip(THETA_NAMES, self.mean_abs_error)),
            "ks_uniform_p": dict(zip(THETA_NAMES, np.round(self.ks_uniform_p, 3))),
            "discrepancy_injection": self.discrepancy_injection,
        }

    def well_calibrated(self, min_ks_p: float = 0.05,
                        coverage_tol: float = 0.1) -> bool:
        """True if ranks are plausibly uniform and 2-sigma coverage is
        within tolerance of 0.95 for every parameter."""
        return bool(
            np.all(self.ks_uniform_p > min_ks_p)
            and np.all(np.abs(self.coverage_2sigma - 0.954) < coverage_tol)
        )


def _ks_uniform_pvalue(ranks: np.ndarray, n_post: int) -> float:
    """Two-sided KS p-value of integer ranks against discrete uniform."""
    from scipy.stats import kstest

    u = (ranks + 0.5) / (n_post + 1)
    return float(kstest(u, "uniform").pvalue)


def run_sbc(
    posterior: AmortizedPosterior,
    simulator,
    control_map,
    config,
    n_trials: int = 300,
    n_posterior: int = 200,
    hf_range: tuple = (6, 11),
    lf_range: tuple = (3, 7),
    discrepancy_injection: float = 0.05,
    hf_noise_rel: float = 0.02,
    lf_noise_rel: float = 0.03,
    seed: int = 0,
) -> SBCResult:
    """Run SBC for the amortized posterior on simulated episodes.

    Each trial draws ``theta* ~ prior``, simulates a measurement set (with
    measurement noise and a shared systematic bias of amplitude
    ``discrepancy_injection``), infers the posterior, and records the rank
    of each true parameter among posterior samples plus 1/2-sigma
    coverage. ``discrepancy_injection = 0`` tests the well-specified case.

    The measurement designs are drawn from the same distribution as
    training (domain-wide uniform): SBC is only meaningful against the
    joint distribution the posterior is amortized for.
    """
    rng = np.random.default_rng(seed)
    objective = Objective(config.objective)
    summ = CurveSummarizer(objective, simulator.freqs)
    refl = ReflectivitySummarizer(simulator.freqs)
    prior_sd = posterior.prior_sd
    hf_scale = posterior.featurizer.hf_component_scale
    lf_scale = posterior.featurizer.lf_component_scale
    limits = control_map.cfg.limits()

    ranks = np.zeros((n_trials, 3), dtype=int)
    within1 = np.zeros((n_trials, 3), dtype=bool)
    within2 = np.zeros((n_trials, 3), dtype=bool)
    abserr = np.zeros((n_trials, 3))

    for t in range(n_trials):
        theta_std = np.clip(rng.standard_normal(3), -3.5, 3.5)
        theta = DetectorState.from_vector(theta_std * prior_sd)
        theta_vec = theta.to_vector()
        n_hf = int(rng.integers(hf_range[0], hf_range[1] + 1))
        n_lf = int(rng.integers(lf_range[0], lf_range[1] + 1))

        hf_u = np.stack([rng.uniform(-0.5, 0.5, control_map.dim) * limits for _ in range(n_hf)])
        hf_bias = discrepancy_injection * hf_scale * rng.standard_normal(len(hf_scale))
        hf_z = np.stack([simulator.predict_summaries(u, theta, summ) for u in hf_u])
        hf_z = hf_z + hf_bias + hf_noise_rel * hf_scale * rng.standard_normal(hf_z.shape)
        if n_lf > 0:
            lf_u = np.stack([rng.uniform(-0.5, 0.5, control_map.dim) * limits for _ in range(n_lf)])
            lf_bias = discrepancy_injection * lf_scale * rng.standard_normal(len(lf_scale))
            lf_z = np.stack([simulator.predict_reflectivity_summaries(u, theta, refl) for u in lf_u])
            lf_z = lf_z + lf_bias + lf_noise_rel * lf_scale * rng.standard_normal(lf_z.shape)
        else:
            lf_u = np.zeros((0, control_map.dim))
            lf_z = np.zeros((0, len(lf_scale)))

        c = build_conditioning(
            posterior.featurizer, hf_u, hf_z, lf_u, lf_z,
            control_map, simulator, objective, summ, refl,
        )
        mean, cov, sampler = posterior.infer(c, rng)
        samples = np.array([s.to_vector() for s in sampler(n_posterior, rng)])
        ranks[t] = np.sum(samples < theta_vec[None, :], axis=0)
        sd = np.sqrt(np.diag(cov))
        err = np.abs(mean.to_vector() - theta_vec)
        abserr[t] = err
        within1[t] = err <= sd
        within2[t] = err <= 2 * sd

    ks = np.array([_ks_uniform_pvalue(ranks[:, i], n_posterior) for i in range(3)])
    return SBCResult(
        ranks=ranks,
        n_posterior=n_posterior,
        coverage_1sigma=within1.mean(axis=0),
        coverage_2sigma=within2.mean(axis=0),
        mean_abs_error=abserr.mean(axis=0),
        ks_uniform_p=ks,
        discrepancy_injection=discrepancy_injection,
    )

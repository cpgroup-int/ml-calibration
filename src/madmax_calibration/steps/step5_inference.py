"""Step 5: jointly update detector-state, discrepancy, noise, drift inference.

Implements inference Level A of the Step 5 design note (section 13.1) at
either of two observation levels (section 4, selected by
``Step5Config.observation_level``):

- ``"curve_summary"`` (default, roadmap Phase 1.1): each HF measurement
  contributes a vector of physically meaningful curve summaries
  z = (J, log peak, band centroid, bandwidth, flatness) — frequency
  shifts, amplitude losses and bandwidth changes are then distinguishable,
  which breaks the detector-state degeneracies that scalar-level
  inference cannot resolve.
- ``"scalar"``: each HF measurement contributes only (J, sigma_J) — the
  pre-Phase-1.1 behaviour, kept as the 1-component special case for A/B
  benchmarking.

The joint model, per observation component k::

    z_{i,k} = z_sim_k(u~_i, theta) + r_k(u~_i) + [k = J] d (t_i - t_ref)
              + eps_{i,k},

    eps_{i,k} ~ N(0, sigma_{i,k}^2 + (s * scale_k)^2),

where each component has its own discrepancy GP r_k (marginalized
analytically inside the objective — the discrepancy channel is present
*while* theta is inferred, never fitted afterwards), the linear drift
term acts on the objective component, and one shared dimensionless
noise-inflation factor ``s`` scales with each component's typical
measurement sigma.  Cross-component measurement correlations are
neglected at Level A (documented approximation).

Uncertainty for (theta, drift) comes from a Laplace approximation around
the joint MAP; a prior-sensitivity refit flags weakly identifiable
parameters (section 15.2); inferred parameters are classified
correctable vs diagnostic-only against the online control basis
(section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..config import CalibrationConfig
from ..control import ControlMap
from ..gp import GaussianProcess, fit_gp_hyperparameters
from ..objectives import Objective
from ..records import CalibrationDataset
from ..simulator import BoostSimulator, DetectorState
from ..summaries import CurveSummarizer


@dataclass
class LFLinkModel:
    """Learned affine relation between the LF proxy and the HF objective."""

    alpha: float = 1.0
    beta: float = 0.0
    resid_sd: float = np.inf
    n_points: int = 0
    validated: bool = False

    def predict_proxy(self, j_hf: float) -> float:
        return self.alpha * j_hf + self.beta

    def invert(self, proxy: float) -> float:
        return (proxy - self.beta) / self.alpha if abs(self.alpha) > 1e-9 else np.nan


@dataclass
class Step5Result:
    """Joint model state M_t handed to Step 6 (design, section 21).

    ``gp`` is the discrepancy GP of the objective component (what Step 6
    consumes for J-prediction); the per-summary discrepancy GPs live in
    ``summary_gps``.
    """

    theta_map: DetectorState
    theta_cov: np.ndarray               # Laplace covariance (3x3)
    drift_rate: float                   # J units / hour
    drift_rate_sd: float
    noise_inflation: float              # extra J-noise sd (absolute)
    gp: GaussianProcess                 # J-component discrepancy GP
    t_ref: float
    x_train: np.ndarray
    residual_noise_sd: np.ndarray       # J-component effective noise
    observation_level: str = "scalar"
    summary_names: tuple = ("J",)
    summary_gps: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)   # name -> "correctable"|"diagnostic"
    identifiability: dict = field(default_factory=dict)  # name -> "ok"|"weak"
    lf_link: LFLinkModel = field(default_factory=LFLinkModel)
    diagnostics: dict = field(default_factory=dict)
    step6_ready: bool = True

    def theta_samples(self, n: int, rng: np.random.Generator) -> list[DetectorState]:
        chol = np.linalg.cholesky(self.theta_cov + 1e-18 * np.eye(3))
        mu = self.theta_map.to_vector()
        return [DetectorState.from_vector(mu + chol @ rng.standard_normal(3)) for _ in range(n)]


def _prepare_hf_data(
    dataset: CalibrationDataset,
    control_map: ControlMap,
    want_summaries: bool,
):
    """Assemble the (n x K) observation matrix and its uncertainties.

    Falls back to the scalar level (K = 1) when any HF record lacks
    summaries; the fallback is reported through the returned level.
    """
    recs = dataset.hf_records()
    u = np.stack([r.u_B_achieved for r in recs])
    x = np.stack([control_map.to_normalized(r.u_B_achieved) for r in recs])
    t = np.array([r.time for r in recs])

    if want_summaries and all(r.summaries is not None and r.summaries_sigma is not None for r in recs):
        from ..summaries import SUMMARY_NAMES

        z = np.stack([r.summaries for r in recs])
        sig = np.stack([np.clip(r.summaries_sigma, 1e-12, None) for r in recs])
        names = tuple(SUMMARY_NAMES)
        level = "curve_summary"
    else:
        z = np.array([[r.J] for r in recs])
        sig = np.array([[max(r.sigma_J, 1e-9)] for r in recs])
        names = ("J",)
        level = "scalar"
    return recs, u, x, z, sig, t, names, level


def run_step5(
    dataset: CalibrationDataset,
    simulator: BoostSimulator,
    control_map: ControlMap,
    config: CalibrationConfig,
    objective: Objective,
    previous: Step5Result | None = None,
    rng: np.random.Generator | None = None,
) -> Step5Result:
    """Update the joint calibration model from the accumulated data."""
    rng = rng or np.random.default_rng(0)
    cfg5 = config.step5
    want_summaries = cfg5.observation_level == "curve_summary"
    recs, u_data, x_data, z_data, sigma_data, t_data, names, level = _prepare_hf_data(
        dataset, control_map, want_summaries
    )
    n, n_comp = z_data.shape
    if n < cfg5.min_hf_points_for_inference:
        raise ValueError(
            f"need >= {cfg5.min_hf_points_for_inference} valid HF points, have {n}"
        )
    j_idx = names.index("J")

    # Per-component scales: typical measurement sigma (noise-inflation
    # scale) and response scale (discrepancy-amplitude prior scale).
    meas_scale = np.median(sigma_data, axis=0)                        # (K,)
    response_scale = np.maximum(np.mean(np.abs(z_data), axis=0), 5.0 * meas_scale)
    j_scale = float(response_scale[j_idx])
    t_ref = float(np.min(t_data))
    dt = t_data - t_ref

    prior_sd = np.array(
        [cfg5.prior_sd_z_offset, cfg5.prior_sd_compression, cfg5.prior_sd_log_loss]
    )
    # Amplitude priors: a fraction of the response scale, floored at a few
    # measurement sigmas so systematics can live in the discrepancy channel
    # rather than biasing theta (see Step5Config.discrepancy_sigma_floor).
    amp_priors = np.maximum(
        cfg5.discrepancy_amplitude_prior * response_scale,
        cfg5.discrepancy_sigma_floor * meas_scale,
    )
    ls_lo, ls_hi = cfg5.discrepancy_lengthscale_bounds

    summarizer = CurveSummarizer(objective, simulator.freqs) if level == "curve_summary" else None

    def sim_z(theta_std: np.ndarray) -> np.ndarray:
        """(n x K) simulator prediction at standardized theta."""
        theta = DetectorState.from_vector(theta_std * prior_sd)
        if summarizer is not None:
            return np.stack(
                [simulator.predict_summaries(ui, theta, summarizer) for ui in u_data]
            )
        return np.array([[simulator.predict_J(ui, theta, objective)] for ui in u_data])

    # Parameter vector (standardized): [theta (3), drift, log noise-inflation].
    def unpack(p: np.ndarray):
        theta_std = p[:3]
        drift = p[3] * cfg5.drift_rate_prior
        s = np.exp(p[4]) * cfg5.noise_inflation_prior   # dimensionless factor
        return theta_std, drift, s

    def neg_log_post(p: np.ndarray, gp_hypers: list) -> float:
        theta_std, drift, s = unpack(p)
        try:
            pred = sim_z(theta_std)
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1e12
        total = 0.5 * float(np.sum(theta_std**2)) + 0.5 * p[3] ** 2 + 0.5 * np.exp(2 * p[4])
        for k in range(n_comp):
            resid = z_data[:, k] - pred[:, k]
            if k == j_idx:
                resid = resid - drift * dt
            noise = np.sqrt(sigma_data[:, k] ** 2 + (s * meas_scale[k]) ** 2)
            amp, ls = gp_hypers[k]
            gp = GaussianProcess(amplitude=amp, lengthscales=np.full(x_data.shape[1], ls))
            try:
                gp.fit(x_data, resid, noise)
                total -= gp.log_marginal_likelihood()
            except np.linalg.LinAlgError:
                return 1e12
        return total

    # Warm start from the previous posterior state (component-matched).
    default_hyper = lambda k: (0.5 * amp_priors[k], 0.5 * (ls_lo + ls_hi))  # noqa: E731
    if previous is not None:
        p0 = np.concatenate(
            [
                previous.theta_map.to_vector() / prior_sd,
                [previous.drift_rate / cfg5.drift_rate_prior],
                [np.log(max(previous.noise_inflation / max(cfg5.noise_inflation_prior * meas_scale[j_idx], 1e-15), 1e-3))],
            ]
        )
        prev_gps = dict(previous.summary_gps)
        prev_gps.setdefault("J", previous.gp)
        gp_hypers = [
            (prev_gps[nm].amplitude, float(prev_gps[nm].lengthscales[0]))
            if nm in prev_gps and previous.summary_names == names
            else default_hyper(k)
            for k, nm in enumerate(names)
        ]
    else:
        p0 = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        gp_hypers = [default_hyper(k) for k in range(n_comp)]

    bounds = [(-5, 5)] * 3 + [(-5, 5), (-4, 2)]

    # Alternate: (a) MAP over (theta, drift, noise) given GP hypers,
    # (b) penalized ML-II refit of each component's GP hypers on its
    # residuals.  Two rounds suffice with the warm start.
    res = None
    for _ in range(2):
        res = minimize(
            neg_log_post, p0, args=(gp_hypers,), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 60},
        )
        p0 = res.x
        theta_std, drift, s = unpack(p0)
        pred = sim_z(theta_std)
        new_hypers = []
        for k in range(n_comp):
            resid = z_data[:, k] - pred[:, k]
            if k == j_idx:
                resid = resid - drift * dt
            noise = np.sqrt(sigma_data[:, k] ** 2 + (s * meas_scale[k]) ** 2)
            gp_fitted = fit_gp_hyperparameters(
                x_data, resid, noise,
                amplitude_bounds=(1e-4 * response_scale[k], 10 * amp_priors[k]),
                lengthscale_bounds=(ls_lo, ls_hi),
                amplitude_prior_sd=amp_priors[k],
                seed=config.seed + k,
            )
            new_hypers.append((gp_fitted.amplitude, float(gp_fitted.lengthscales[0])))
        gp_hypers = new_hypers

    theta_std, drift, s = unpack(res.x)
    theta_map = DetectorState.from_vector(theta_std * prior_sd)
    s_extra_j = float(s * meas_scale[j_idx])   # absolute, J units

    # Laplace covariance for (theta, drift) by finite-difference Hessian.
    def nlp_sub(q: np.ndarray) -> float:
        p = res.x.copy()
        p[:4] = q
        return neg_log_post(p, gp_hypers)

    h = 1e-3
    q0 = res.x[:4].copy()
    hess = np.zeros((4, 4))
    for i in range(4):
        for k in range(i, 4):
            ei = np.zeros(4)
            ek = np.zeros(4)
            ei[i] = h
            ek[k] = h
            fpp = nlp_sub(q0 + ei + ek)
            fpm = nlp_sub(q0 + ei - ek)
            fmp = nlp_sub(q0 - ei + ek)
            fmm = nlp_sub(q0 - ei - ek)
            hess[i, k] = hess[k, i] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    hess = 0.5 * (hess + hess.T)
    eigval, eigvec = np.linalg.eigh(hess)
    eigval = np.clip(eigval, 1.0, None)  # prior curvature floor
    cov_std = eigvec @ np.diag(1.0 / eigval) @ eigvec.T
    scale = np.concatenate([prior_sd, [cfg5.drift_rate_prior]])
    cov_full = cov_std * np.outer(scale, scale)
    theta_cov = cov_full[:3, :3]
    drift_sd = float(np.sqrt(cov_full[3, 3]))

    # Final per-component discrepancy GPs conditioned at the MAP.
    pred = sim_z(theta_std)
    summary_gps: dict[str, GaussianProcess] = {}
    noise_j = None
    for k, nm in enumerate(names):
        resid = z_data[:, k] - pred[:, k]
        if k == j_idx:
            resid = resid - drift * dt
        noise = np.sqrt(sigma_data[:, k] ** 2 + (s * meas_scale[k]) ** 2)
        amp, ls = gp_hypers[k]
        gp_k = GaussianProcess(amplitude=amp, lengthscales=np.full(x_data.shape[1], ls))
        gp_k.fit(x_data, resid, noise)
        summary_gps[nm] = gp_k
        if k == j_idx:
            noise_j = noise
    gp = summary_gps["J"]

    # ---- classification: correctable vs diagnostic (section 8) ----------
    classification = {
        name: ("correctable" if DetectorState.CORRECTABLE[name] else "diagnostic")
        for name in DetectorState.NAMES
    }

    # ---- identifiability: prior-sensitivity check (section 15.2) --------
    identifiability = {name: "ok" for name in DetectorState.NAMES}
    if cfg5.prior_sensitivity_check:
        wide_hypers = [(3.0 * amp, ls) for amp, ls in gp_hypers]
        res_wide = minimize(
            neg_log_post, res.x, args=(wide_hypers,), method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 40},
        )
        theta_sd = np.sqrt(np.diag(theta_cov))
        shift = np.abs(res_wide.x[:3] - res.x[:3]) * prior_sd
        for i, name in enumerate(DetectorState.NAMES):
            if shift[i] > 1.0 * max(theta_sd[i], 1e-12):
                identifiability[name] = "weak"

    # ---- LF proxy link model (section 12) --------------------------------
    lf_link = LFLinkModel()
    lf_recs = dataset.lf_records()
    if len(lf_recs) >= 3:
        mu_at_lf = []
        y_lf = []
        for r in lf_recs:
            x_r = control_map.to_normalized(r.u_B_achieved)
            m, _ = gp.predict(x_r[None, :])
            mu_at_lf.append(
                simulator.predict_J(r.u_B_achieved, theta_map, objective) + float(m[0])
            )
            y_lf.append(r.proxy_value)
        mu_at_lf = np.array(mu_at_lf)
        y_lf = np.array(y_lf)
        A = np.stack([mu_at_lf, np.ones_like(mu_at_lf)], axis=1)
        coef, *_ = np.linalg.lstsq(A, y_lf, rcond=None)
        pred_lf = A @ coef
        rsd = float(np.std(y_lf - pred_lf, ddof=1)) if len(y_lf) > 2 else np.inf
        spread = float(np.std(mu_at_lf))
        lf_link = LFLinkModel(
            alpha=float(coef[0]),
            beta=float(coef[1]),
            resid_sd=rsd,
            n_points=len(y_lf),
            validated=bool(coef[0] > 0 and rsd < max(abs(coef[0]) * spread, 1e-12)),
        )

    # ---- diagnostics -----------------------------------------------------
    resid_j = z_data[:, j_idx] - pred[:, j_idx] - drift * dt
    z_std = resid_j / noise_j
    gp_mean_train, _ = gp.predict(x_data)
    z_after = (resid_j - gp_mean_train) / noise_j
    discrepancy_dominant = bool(
        gp.amplitude > 2.0 * amp_priors[j_idx] and np.std(resid_j) > 2.0 * np.mean(noise_j)
    )
    baseline = dataset.baseline_records()
    baseline_drift = None
    if len(baseline) >= 2:
        bt = np.array([r.time for r in baseline])
        bj = np.array([r.J for r in baseline])
        if bt.max() > bt.min():
            baseline_drift = float(np.polyfit(bt, bj, 1)[0])

    diagnostics = {
        "observation_level": level,
        "n_hf": n,
        "n_lf": len(lf_recs),
        "n_components": n_comp,
        "j_scale": j_scale,
        "max_abs_standardized_residual": float(np.max(np.abs(z_std))),
        "rms_standardized_residual_after_gp": float(np.sqrt(np.mean(z_after**2))),
        "discrepancy_amplitude": gp.amplitude,
        "discrepancy_amplitude_prior": float(amp_priors[j_idx]),
        "discrepancy_amplitudes": {nm: summary_gps[nm].amplitude for nm in names},
        "discrepancy_dominant": discrepancy_dominant,
        "noise_inflation": s_extra_j,
        "noise_inflation_factor": float(s),
        "drift_rate": float(drift),
        "baseline_drift_estimate": baseline_drift,
        "converged": bool(res.success),
    }
    if want_summaries and level == "scalar":
        diagnostics["warning"] = (
            "curve_summary level requested but some HF records lack summaries; "
            "fell back to scalar-level inference"
        )

    return Step5Result(
        theta_map=theta_map,
        theta_cov=theta_cov,
        drift_rate=float(drift),
        drift_rate_sd=drift_sd,
        noise_inflation=s_extra_j,
        gp=gp,
        t_ref=t_ref,
        x_train=x_data,
        residual_noise_sd=noise_j,
        observation_level=level,
        summary_names=names,
        summary_gps=summary_gps,
        classification=classification,
        identifiability=identifiability,
        lf_link=lf_link,
        diagnostics=diagnostics,
        step6_ready=True,
    )

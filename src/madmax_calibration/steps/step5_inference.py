"""Step 5: jointly update detector-state, discrepancy, noise, drift inference.

Implements inference Level A of the Step 5 design note (section 13.1):
a *joint* MAP fit of

    J_i = J_sim(u~_i, theta) + r(u~_i) + d * (t_i - t_ref) + eps_i,
    eps_i ~ N(0, sigma_i^2 + s_extra^2),

where the discrepancy GP ``r`` is marginalized analytically (its
contribution enters through the GP marginal likelihood) and theta, the
drift rate ``d`` and the noise inflation ``s_extra`` carry Gaussian /
half-normal priors.  The discrepancy term is present *while* theta is
inferred — never fitted afterwards on a leftover residual (section 3).

Uncertainty for theta comes from a Laplace approximation around the MAP;
a prior-sensitivity refit flags weakly identifiable parameters
(section 15.2).  Inferred parameters are classified correctable vs
diagnostic-only against the online control basis (section 8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..config import CalibrationConfig
from ..control import ControlMap
from ..gp import GaussianProcess, fit_gp_hyperparameters
from ..objectives import Objective
from ..records import CalibrationDataset, MeasurementRecord
from ..simulator import BoostSimulator, DetectorState


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
    """Joint model state M_t handed to Step 6 (design, section 21)."""

    theta_map: DetectorState
    theta_cov: np.ndarray               # Laplace covariance (3x3)
    drift_rate: float                   # J units / hour
    drift_rate_sd: float
    noise_inflation: float              # extra J-noise sd
    gp: GaussianProcess                 # discrepancy GP on normalized u_B
    t_ref: float
    x_train: np.ndarray
    residual_noise_sd: np.ndarray
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
):
    recs = dataset.hf_records()
    u = np.stack([r.u_B_achieved for r in recs])
    x = np.stack([control_map.to_normalized(r.u_B_achieved) for r in recs])
    j = np.array([r.J for r in recs])
    sig = np.array([max(r.sigma_J, 1e-9) for r in recs])
    t = np.array([r.time for r in recs])
    return recs, u, x, j, sig, t


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
    recs, u_data, x_data, j_data, sigma_data, t_data = _prepare_hf_data(dataset, control_map)
    n = len(recs)
    if n < cfg5.min_hf_points_for_inference:
        raise ValueError(
            f"need >= {cfg5.min_hf_points_for_inference} valid HF points, have {n}"
        )

    j_scale = max(float(np.mean(np.abs(j_data))), 1e-9)
    t_ref = float(np.min(t_data))
    dt = t_data - t_ref

    prior_sd = np.array(
        [cfg5.prior_sd_z_offset, cfg5.prior_sd_compression, cfg5.prior_sd_log_loss]
    )
    amp_prior = cfg5.discrepancy_amplitude_prior * j_scale
    ls_lo, ls_hi = cfg5.discrepancy_lengthscale_bounds

    def sim_j(theta_vec: np.ndarray) -> np.ndarray:
        theta = DetectorState.from_vector(theta_vec * prior_sd_full[:3])
        return np.array([simulator.predict_J(ui, theta, objective) for ui in u_data])

    # Parameter vector (all standardized by prior scales):
    # [theta_z, theta_c, theta_l, drift, log_noise_inflation]
    prior_sd_full = np.concatenate([prior_sd, [cfg5.drift_rate_prior]])

    def unpack(p: np.ndarray):
        theta_std = p[:3]
        drift = p[3] * cfg5.drift_rate_prior
        s_extra = np.exp(p[4]) * cfg5.noise_inflation_prior * j_scale
        return theta_std, drift, s_extra

    def neg_log_post(p: np.ndarray, gp_amp: float, gp_ls: float) -> float:
        theta_std, drift, s_extra = unpack(p)
        try:
            resid = j_data - sim_j(theta_std) - drift * dt
        except (np.linalg.LinAlgError, FloatingPointError):
            return 1e12
        noise = np.sqrt(sigma_data**2 + s_extra**2)
        gp = GaussianProcess(amplitude=gp_amp, lengthscales=np.full(x_data.shape[1], gp_ls))
        try:
            gp.fit(x_data, resid, noise)
            lml = gp.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            return 1e12
        # Priors: standardized Gaussians for theta and drift; half-normal
        # (in log-param form) for the noise inflation.
        prior = 0.5 * float(np.sum(theta_std**2)) + 0.5 * p[3] ** 2 + 0.5 * np.exp(2 * p[4])
        return -lml + prior

    # Warm start from the previous posterior state.
    if previous is not None:
        p0 = np.concatenate(
            [
                previous.theta_map.to_vector() / prior_sd,
                [previous.drift_rate / cfg5.drift_rate_prior],
                [np.log(max(previous.noise_inflation / (cfg5.noise_inflation_prior * j_scale), 1e-3))],
            ]
        )
        gp_amp, gp_ls = previous.gp.amplitude, float(previous.gp.lengthscales[0])
    else:
        p0 = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        gp_amp, gp_ls = 0.5 * amp_prior, 0.5 * (ls_lo + ls_hi)

    # Alternate: (a) MAP over (theta, drift, noise) given GP hypers,
    # (b) refit GP hypers on the residuals.  Two rounds suffice with the
    # warm start; this is the joint fit at Level A fidelity.
    res = None
    for _ in range(2):
        res = minimize(
            neg_log_post,
            p0,
            args=(gp_amp, gp_ls),
            method="L-BFGS-B",
            bounds=[(-5, 5)] * 3 + [(-5, 5), (-4, 2)],
            options={"maxiter": 60},
        )
        p0 = res.x
        theta_std, drift, s_extra = unpack(p0)
        resid = j_data - sim_j(theta_std) - drift * dt
        noise = np.sqrt(sigma_data**2 + s_extra**2)
        gp_fitted = fit_gp_hyperparameters(
            x_data,
            resid,
            noise,
            amplitude_bounds=(1e-4 * j_scale, 10 * amp_prior),
            lengthscale_bounds=(ls_lo, ls_hi),
            amplitude_prior_sd=amp_prior,
            seed=config.seed,
        )
        gp_amp, gp_ls = gp_fitted.amplitude, float(gp_fitted.lengthscales[0])

    theta_std, drift, s_extra = unpack(res.x)
    theta_map = DetectorState.from_vector(theta_std * prior_sd)

    # Laplace covariance for (theta, drift) by finite-difference Hessian.
    def nlp_sub(q: np.ndarray) -> float:
        p = res.x.copy()
        p[:4] = q
        return neg_log_post(p, gp_amp, gp_ls)

    h = 1e-3
    q0 = res.x[:4].copy()
    hess = np.zeros((4, 4))
    f0 = nlp_sub(q0)
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
    # Regularize to at least the prior curvature (identity in std units).
    hess = 0.5 * (hess + hess.T)
    eigval, eigvec = np.linalg.eigh(hess)
    eigval = np.clip(eigval, 1.0, None)  # prior curvature floor
    cov_std = eigvec @ np.diag(1.0 / eigval) @ eigvec.T
    scale = np.concatenate([prior_sd, [cfg5.drift_rate_prior]])
    cov_full = cov_std * np.outer(scale, scale)
    theta_cov = cov_full[:3, :3]
    drift_sd = float(np.sqrt(cov_full[3, 3]))

    # Final discrepancy GP conditioned at the MAP.
    resid = j_data - sim_j(theta_std) - drift * dt
    noise = np.sqrt(sigma_data**2 + s_extra**2)
    gp = GaussianProcess(amplitude=gp_amp, lengthscales=np.full(x_data.shape[1], gp_ls))
    gp.fit(x_data, resid, noise)

    # ---- classification: correctable vs diagnostic (section 8) ----------
    classification = {
        name: ("correctable" if DetectorState.CORRECTABLE[name] else "diagnostic")
        for name in DetectorState.NAMES
    }

    # ---- identifiability: prior-sensitivity check (section 15.2) --------
    identifiability = {name: "ok" for name in DetectorState.NAMES}
    if cfg5.prior_sensitivity_check:
        res_wide = minimize(
            neg_log_post,
            res.x,
            args=(3.0 * gp_amp, gp_ls),
            method="L-BFGS-B",
            bounds=[(-5, 5)] * 3 + [(-5, 5), (-4, 2)],
            options={"maxiter": 40},
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
        pred = A @ coef
        rsd = float(np.std(y_lf - pred, ddof=1)) if len(y_lf) > 2 else np.inf
        spread = float(np.std(mu_at_lf))
        lf_link = LFLinkModel(
            alpha=float(coef[0]),
            beta=float(coef[1]),
            resid_sd=rsd,
            n_points=len(y_lf),
            # The proxy is only trusted if it tracks the objective: positive
            # slope and residuals small relative to the explained spread.
            validated=bool(coef[0] > 0 and rsd < max(abs(coef[0]) * spread, 1e-12)),
        )

    # ---- diagnostics -----------------------------------------------------
    z = resid / noise
    gp_mean_train, _ = gp.predict(x_data)
    z_after = (resid - gp_mean_train) / noise
    discrepancy_dominant = bool(
        gp.amplitude > 2.0 * amp_prior and np.std(resid) > 2.0 * np.mean(noise)
    )
    baseline = dataset.baseline_records()
    baseline_drift = None
    if len(baseline) >= 2:
        bt = np.array([r.time for r in baseline])
        bj = np.array([r.J for r in baseline])
        if bt.max() > bt.min():
            baseline_drift = float(np.polyfit(bt, bj, 1)[0])

    diagnostics = {
        "n_hf": n,
        "n_lf": len(lf_recs),
        "j_scale": j_scale,
        "max_abs_standardized_residual": float(np.max(np.abs(z))),
        "rms_standardized_residual_after_gp": float(np.sqrt(np.mean(z_after**2))),
        "discrepancy_amplitude": gp.amplitude,
        "discrepancy_amplitude_prior": amp_prior,
        "discrepancy_dominant": discrepancy_dominant,
        "noise_inflation": s_extra,
        "drift_rate": drift,
        "baseline_drift_estimate": baseline_drift,
        "converged": bool(res.success),
    }

    return Step5Result(
        theta_map=theta_map,
        theta_cov=theta_cov,
        drift_rate=float(drift),
        drift_rate_sd=drift_sd,
        noise_inflation=float(s_extra),
        gp=gp,
        t_ref=t_ref,
        x_train=x_data,
        residual_noise_sd=noise,
        classification=classification,
        identifiability=identifiability,
        lf_link=lf_link,
        diagnostics=diagnostics,
        step6_ready=True,
    )

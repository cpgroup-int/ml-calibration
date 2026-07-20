"""Step 5: jointly update detector-state, discrepancy, noise, drift inference.

Implements inference Level A of the Step 5 design note (section 13.1) over
two observation *channels*:

- the **high-fidelity channel** at either observation level (section 4,
  selected by ``Step5Config.observation_level``): the curve-summary
  vector z = (J, log peak, band centroid, bandwidth, flatness) per HF
  measurement (default, roadmap Phase 1.1), or only (J, sigma_J) with
  ``"scalar"`` (the pre-1.1 behaviour, kept for A/B benchmarking);
- the **low-fidelity physics channel** (roadmap Phase 1.2, enabled by
  ``Step5Config.lf_channel = "physics"``): reflectivity/group-delay
  summaries of cheap RF measurements, modelled through the simulator as
  ``y_l = S_l(u, theta) + r_l + eps`` — exactly the structure of the
  design note's section 12 — so LF data constrain theta *jointly* with
  HF data instead of through a statistical link.  The affine LF->J link
  remains as a fallback for proxies without a simulator counterpart.

The joint model, per observation component k of either channel::

    z_{i,k} = z_sim_k(u~_i, theta) + r_k(u~_i) + [k = J] d (t_i - t_ref)
              + eps_{i,k},

    eps_{i,k} ~ N(0, sigma_{i,k}^2 + (s * scale_k)^2),

where each component has its own discrepancy GP r_k (marginalized
analytically inside the objective — the discrepancy channel is present
*while* theta is inferred, never fitted afterwards), the linear drift
term acts on the objective component, and one shared dimensionless
noise-inflation factor ``s`` scales with each component's typical
measurement sigma.  Discrepancy amplitude priors are floored at a few
measurement sigmas (``discrepancy_sigma_floor``) so unmodelled
systematics (curve tilt, reflectivity mis-calibration, cable-delay
offset) stay in the discrepancy channel rather than biasing theta.
Cross-component measurement correlations are neglected at Level A
(documented approximation).

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
from ..summaries import (
    REFLECTIVITY_SUMMARY_NAMES,
    SUMMARY_NAMES,
    CurveSummarizer,
    ReflectivitySummarizer,
)


_POSTERIOR_CACHE: dict = {}


def _default_weights_path() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[3] / "weights" / "npe_prototype.pt")


def _load_amortized_posterior(path: str | None):
    """Load (and cache) a trained :class:`AmortizedPosterior`, or None."""
    from pathlib import Path

    resolved = path or _default_weights_path()
    if resolved in _POSTERIOR_CACHE:
        return _POSTERIOR_CACHE[resolved]
    post = None
    if Path(resolved).exists():
        try:
            from ..amortized import AmortizedPosterior

            post = AmortizedPosterior.load(resolved)
        except Exception:
            post = None
    _POSTERIOR_CACHE[resolved] = post
    return post


@dataclass
class LFLinkModel:
    """Learned affine relation between a scalar LF proxy and the HF objective.

    Retained as the fallback for proxies without a simulator counterpart;
    when the physics LF channel is active (``Step5Result.lf_channel ==
    "physics"``) this link is not used for decisions.
    """

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
class _Channel:
    """One observation channel of the joint fit."""

    label: str                      # "hf" | "lf"
    names: tuple                    # component names (globally unique)
    x: np.ndarray                   # (n, d) normalized inputs
    z: np.ndarray                   # (n, K) observations
    sigma: np.ndarray               # (n, K) measurement sigmas
    dt: np.ndarray                  # (n,) hours since t_ref (drift term)
    drift_component: str | None     # component name carrying the drift term


@dataclass
class Step5Result:
    """Joint model state M_t handed to Step 6 (design, section 21).

    ``gp`` is the discrepancy GP of the objective component (what Step 6
    consumes for J-prediction); all per-component discrepancy GPs —
    including the LF reflectivity ones — live in ``summary_gps``.
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
    lf_channel: str = "none"            # "physics" | "affine" | "none"
    n_lf_physics: int = 0
    summary_names: tuple = ("J",)
    summary_gps: dict = field(default_factory=dict)
    classification: dict = field(default_factory=dict)   # name -> "correctable"|"diagnostic"
    identifiability: dict = field(default_factory=dict)  # name -> "ok"|"weak"
    lf_link: LFLinkModel = field(default_factory=LFLinkModel)
    diagnostics: dict = field(default_factory=dict)
    step6_ready: bool = True
    inference_engine: str = "joint_map"
    # Exact posterior sampler (set by the amortized-NPE engine); when
    # present, theta_samples draws from it instead of the Laplace Gaussian.
    sampler: object = field(default=None, repr=False)

    def theta_samples(self, n: int, rng: np.random.Generator) -> list[DetectorState]:
        if self.sampler is not None:
            return self.sampler(n, rng)
        chol = np.linalg.cholesky(self.theta_cov + 1e-18 * np.eye(3))
        mu = self.theta_map.to_vector()
        return [DetectorState.from_vector(mu + chol @ rng.standard_normal(3)) for _ in range(n)]


def _prepare_hf_data(
    dataset: CalibrationDataset,
    control_map: ControlMap,
    want_summaries: bool,
):
    """Assemble the HF (n x K) observation matrix and its uncertainties.

    Falls back to the scalar level (K = 1) when any HF record lacks
    summaries; the fallback is reported through the returned level.
    """
    recs = dataset.hf_records()
    u = np.stack([r.u_B_achieved for r in recs])
    x = np.stack([control_map.to_normalized(r.u_B_achieved) for r in recs])
    t = np.array([r.time for r in recs])

    if want_summaries and all(r.summaries is not None and r.summaries_sigma is not None for r in recs):
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
    recs, u_hf, x_hf, z_hf, sig_hf, t_hf, hf_names, level = _prepare_hf_data(
        dataset, control_map, want_summaries
    )
    n_hf = len(recs)
    if n_hf < cfg5.min_hf_points_for_inference:
        raise ValueError(
            f"need >= {cfg5.min_hf_points_for_inference} valid HF points, have {n_hf}"
        )
    t_ref = float(np.min(t_hf))

    channels = [
        _Channel(
            label="hf", names=hf_names, x=x_hf, z=z_hf, sigma=sig_hf,
            dt=t_hf - t_ref, drift_component="J",
        )
    ]

    # ---- LF physics channel (roadmap Phase 1.2) --------------------------
    lf_phys_recs = []
    if cfg5.lf_channel == "physics":
        lf_phys_recs = [
            r
            for r in dataset.lf_records()
            if r.summaries is not None
            and r.summaries_sigma is not None
            and r.observable_id == "reflectivity"
        ]
    refl_summarizer = ReflectivitySummarizer(simulator.freqs) if lf_phys_recs else None
    if lf_phys_recs:
        u_lf = np.stack([r.u_B_achieved for r in lf_phys_recs])
        channels.append(
            _Channel(
                label="lf",
                names=tuple(REFLECTIVITY_SUMMARY_NAMES),
                x=np.stack([control_map.to_normalized(r.u_B_achieved) for r in lf_phys_recs]),
                z=np.stack([r.summaries for r in lf_phys_recs]),
                sigma=np.stack([np.clip(r.summaries_sigma, 1e-12, None) for r in lf_phys_recs]),
                dt=np.array([r.time for r in lf_phys_recs]) - t_ref,
                drift_component=None,
            )
        )

    # ---- per-component scales and priors ---------------------------------
    prior_sd = np.array(
        [cfg5.prior_sd_z_offset, cfg5.prior_sd_compression, cfg5.prior_sd_log_loss]
    )
    meas_scale: dict[str, float] = {}
    response_scale: dict[str, float] = {}
    amp_prior: dict[str, float] = {}
    for ch in channels:
        for k, nm in enumerate(ch.names):
            ms = float(np.median(ch.sigma[:, k]))
            rs = max(float(np.mean(np.abs(ch.z[:, k]))), 5.0 * ms)
            meas_scale[nm] = ms
            response_scale[nm] = rs
            # Fraction of the response scale, floored at a few measurement
            # sigmas (Step5Config.discrepancy_sigma_floor).
            amp_prior[nm] = max(
                cfg5.discrepancy_amplitude_prior * rs,
                cfg5.discrepancy_sigma_floor * ms,
            )
    j_scale = response_scale["J"]
    ls_lo, ls_hi = cfg5.discrepancy_lengthscale_bounds

    summarizer = CurveSummarizer(objective, simulator.freqs) if level == "curve_summary" else None

    def sim_channel(ch: _Channel, theta: DetectorState) -> np.ndarray:
        if ch.label == "hf":
            if summarizer is not None:
                return np.stack(
                    [simulator.predict_summaries(ui, theta, summarizer) for ui in u_hf]
                )
            return np.array([[simulator.predict_J(ui, theta, objective)] for ui in u_hf])
        return np.stack(
            [
                simulator.predict_reflectivity_summaries(ui, theta, refl_summarizer)
                for ui in u_lf
            ]
        )

    # Parameter vector (standardized): [theta (3), drift, log noise-inflation].
    def unpack(p: np.ndarray):
        theta_std = p[:3]
        drift = p[3] * cfg5.drift_rate_prior
        s = np.exp(p[4]) * cfg5.noise_inflation_prior   # dimensionless factor
        return theta_std, drift, s

    def residuals(ch: _Channel, pred: np.ndarray, k: int, drift: float) -> np.ndarray:
        resid = ch.z[:, k] - pred[:, k]
        if ch.drift_component is not None and ch.names[k] == ch.drift_component:
            resid = resid - drift * ch.dt
        return resid

    def neg_log_post(p: np.ndarray, gp_hypers: dict) -> float:
        theta_std, drift, s = unpack(p)
        theta = DetectorState.from_vector(theta_std * prior_sd)
        total = 0.5 * float(np.sum(theta_std**2)) + 0.5 * p[3] ** 2 + 0.5 * np.exp(2 * p[4])
        for ch in channels:
            try:
                pred = sim_channel(ch, theta)
            except (np.linalg.LinAlgError, FloatingPointError):
                return 1e12
            for k, nm in enumerate(ch.names):
                resid = residuals(ch, pred, k, drift)
                noise = np.sqrt(ch.sigma[:, k] ** 2 + (s * meas_scale[nm]) ** 2)
                amp, ls = gp_hypers[nm]
                gp = GaussianProcess(amplitude=amp, lengthscales=np.full(ch.x.shape[1], ls))
                try:
                    gp.fit(ch.x, resid, noise)
                    total -= gp.log_marginal_likelihood()
                except np.linalg.LinAlgError:
                    return 1e12
        return total

    # Warm start from the previous posterior state (matched by name).
    def default_hyper(nm: str):
        return (0.5 * amp_prior[nm], 0.5 * (ls_lo + ls_hi))

    all_names = [nm for ch in channels for nm in ch.names]
    if previous is not None:
        p0 = np.concatenate(
            [
                previous.theta_map.to_vector() / prior_sd,
                [previous.drift_rate / cfg5.drift_rate_prior],
                [np.log(max(previous.noise_inflation / max(cfg5.noise_inflation_prior * meas_scale["J"], 1e-15), 1e-3))],
            ]
        )
        prev_gps = dict(previous.summary_gps)
        prev_gps.setdefault("J", previous.gp)
        gp_hypers = {
            nm: (
                (prev_gps[nm].amplitude, float(prev_gps[nm].lengthscales[0]))
                if nm in prev_gps
                else default_hyper(nm)
            )
            for nm in all_names
        }
    else:
        p0 = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        gp_hypers = {nm: default_hyper(nm) for nm in all_names}

    bounds = [(-5, 5)] * 3 + [(-5, 5), (-4, 2)]

    def refit_hypers(theta_std, drift, s, gp_hypers):
        """Penalized ML-II refit of each component's GP hypers on its
        residuals (amplitude capped at 4 prior sds: weak Occam factor at
        small n cannot otherwise rule out runaway discrepancies)."""
        theta = DetectorState.from_vector(theta_std * prior_sd)
        new_hypers = {}
        seed_offset = 0
        for ch in channels:
            pred = sim_channel(ch, theta)
            for k, nm in enumerate(ch.names):
                resid = residuals(ch, pred, k, drift)
                noise = np.sqrt(ch.sigma[:, k] ** 2 + (s * meas_scale[nm]) ** 2)
                gp_fitted = fit_gp_hyperparameters(
                    ch.x, resid, noise,
                    amplitude_bounds=(1e-4 * response_scale[nm], 4 * amp_prior[nm]),
                    lengthscale_bounds=(ls_lo, ls_hi),
                    amplitude_prior_sd=amp_prior[nm],
                    seed=config.seed + seed_offset,
                )
                new_hypers[nm] = (gp_fitted.amplitude, float(gp_fitted.lengthscales[0]))
                seed_offset += 1
        return new_hypers

    sampler = None
    identifiability = {name: "ok" for name in DetectorState.NAMES}

    # Engine selection: amortized NPE needs curve-summary HF data and a
    # loadable weights file whose conditioning dimension matches the
    # current control basis (the network is basis/window-specific — stale
    # weights safely fall back to joint MAP).
    posterior = None
    if cfg5.inference_engine == "amortized_npe" and level == "curve_summary":
        candidate = _load_amortized_posterior(cfg5.npe_weights_path)
        if candidate is not None and candidate.featurizer.control_dim == control_map.dim:
            posterior = candidate
    engine = "amortized_npe" if posterior is not None else "joint_map"

    if engine == "amortized_npe":
        # Amortized posterior for theta from the residual-projection
        # conditioning of the current dataset (roadmap Phase 2.1).
        from ..amortized import build_conditioning

        refl_summ = ReflectivitySummarizer(simulator.freqs)
        lf_u_arr = np.stack([r.u_B_achieved for r in lf_phys_recs]) if lf_phys_recs else np.zeros((0, u_hf.shape[1]))
        lf_z_arr = np.stack([r.summaries for r in lf_phys_recs]) if lf_phys_recs else np.zeros((0, len(REFLECTIVITY_SUMMARY_NAMES)))
        c = build_conditioning(
            posterior.featurizer, u_hf, z_hf, lf_u_arr, lf_z_arr,
            control_map, simulator, objective, summarizer, refl_summ,
        )
        theta_mean_phys, theta_cov, sampler = posterior.infer(c, rng)
        theta_std = theta_mean_phys.to_vector() / prior_sd
        # Focused fit of (drift, log noise-inflation) with theta fixed,
        # alternating with a hyper refit (theta stays at the NPE estimate).
        dn = np.array([0.0, -1.0])
        for _ in range(2):
            def nlp_dn(q):
                return neg_log_post(np.concatenate([theta_std, q]), gp_hypers)
            r_dn = minimize(nlp_dn, dn, method="L-BFGS-B",
                            bounds=[(-5, 5), (-4, 2)], options={"maxiter": 40})
            dn = r_dn.x
            _, drift, s = unpack(np.concatenate([theta_std, dn]))
            gp_hypers = refit_hypers(theta_std, drift, s, gp_hypers)
        _, drift, s = unpack(np.concatenate([theta_std, dn]))
        # Crude drift sd from the 1-D curvature of the drift objective.
        hd = 0.25
        base = np.concatenate([theta_std, dn])
        f0 = neg_log_post(base, gp_hypers)
        fp = neg_log_post(base + np.array([0, 0, 0, hd, 0]), gp_hypers)
        fm = neg_log_post(base + np.array([0, 0, 0, -hd, 0]), gp_hypers)
        curv = max((fp - 2 * f0 + fm) / hd**2, 1.0)
        drift_sd = float(np.sqrt(1.0 / curv) * cfg5.drift_rate_prior)
        # Identifiability from the amortized marginal posterior width.
        theta_sd = np.sqrt(np.diag(theta_cov))
        for i, name in enumerate(DetectorState.NAMES):
            if theta_sd[i] > 0.6 * prior_sd[i]:
                identifiability[name] = "weak"
    else:
        # Alternate: (a) multi-start MAP over (theta, drift, noise) given
        # GP hypers, (b) refit hypers on residuals.  Two rounds suffice
        # with the warm start.  Multi-starting every round (current point
        # + prior mean) guards the spurious loss-runaway mode seen in
        # Phase-1.2 validation.
        prior_mean_start = np.array([0.0, 0.0, 0.0, 0.0, -1.0])
        res = None
        for _ in range(2):
            starts = [p0]
            if float(np.max(np.abs(p0 - prior_mean_start))) > 1e-9:
                starts.append(prior_mean_start)
            best = None
            for start in starts:
                cand = minimize(
                    neg_log_post, start, args=(gp_hypers,), method="L-BFGS-B",
                    bounds=bounds, options={"maxiter": 60},
                )
                if best is None or cand.fun < best.fun:
                    best = cand
            res = best
            p0 = res.x
            theta_std, drift, s = unpack(p0)
            gp_hypers = refit_hypers(theta_std, drift, s, gp_hypers)

        theta_std, drift, s = unpack(res.x)

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

        # Prior-sensitivity identifiability check (design §15.2).
        if cfg5.prior_sensitivity_check:
            wide_hypers = {nm: (3.0 * amp, ls) for nm, (amp, ls) in gp_hypers.items()}
            res_wide = minimize(
                neg_log_post, res.x, args=(wide_hypers,), method="L-BFGS-B",
                bounds=bounds, options={"maxiter": 40},
            )
            theta_sd = np.sqrt(np.diag(theta_cov))
            shift = np.abs(res_wide.x[:3] - res.x[:3]) * prior_sd
            for i, name in enumerate(DetectorState.NAMES):
                if shift[i] > 1.0 * max(theta_sd[i], 1e-12):
                    identifiability[name] = "weak"

    theta_map = DetectorState.from_vector(theta_std * prior_sd)
    s_extra_j = float(s * meas_scale["J"])   # absolute, J units

    # Final per-component discrepancy GPs conditioned at the MAP.
    summary_gps: dict[str, GaussianProcess] = {}
    noise_j = None
    resid_j = None
    for ch in channels:
        pred = sim_channel(ch, theta_map)
        for k, nm in enumerate(ch.names):
            resid = residuals(ch, pred, k, drift)
            noise = np.sqrt(ch.sigma[:, k] ** 2 + (s * meas_scale[nm]) ** 2)
            amp, ls = gp_hypers[nm]
            gp_k = GaussianProcess(amplitude=amp, lengthscales=np.full(ch.x.shape[1], ls))
            gp_k.fit(ch.x, resid, noise)
            summary_gps[nm] = gp_k
            if nm == "J":
                noise_j = noise
                resid_j = resid
    gp = summary_gps["J"]

    # ---- classification: correctable vs diagnostic (section 8) ----------
    classification = {
        name: ("correctable" if DetectorState.CORRECTABLE[name] else "diagnostic")
        for name in DetectorState.NAMES
    }
    # ``identifiability`` was set by the estimator branch above.

    # ---- LF link model: affine fallback only (section 12) ----------------
    lf_channel = "physics" if lf_phys_recs else "none"
    lf_link = LFLinkModel(n_points=len(lf_phys_recs))
    lf_scalar_recs = [
        r for r in dataset.lf_records() if r.proxy_value is not None
    ]
    if not lf_phys_recs and cfg5.lf_channel != "off" and len(lf_scalar_recs) >= 3:
        mu_at_lf = []
        y_lf = []
        for r in lf_scalar_recs:
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
        lf_channel = "affine"

    # ---- diagnostics -----------------------------------------------------
    z_std = resid_j / noise_j
    gp_mean_train, _ = gp.predict(x_hf)
    z_after = (resid_j - gp_mean_train) / noise_j
    discrepancy_dominant = bool(
        gp.amplitude > 2.0 * amp_prior["J"] and np.std(resid_j) > 2.0 * np.mean(noise_j)
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
        "lf_channel": lf_channel,
        "n_hf": n_hf,
        "n_lf": len(dataset.lf_records()),
        "n_lf_physics": len(lf_phys_recs),
        "n_components": len(all_names),
        "j_scale": j_scale,
        "max_abs_standardized_residual": float(np.max(np.abs(z_std))),
        "rms_standardized_residual_after_gp": float(np.sqrt(np.mean(z_after**2))),
        "discrepancy_amplitude": gp.amplitude,
        "discrepancy_amplitude_prior": float(amp_prior["J"]),
        "discrepancy_amplitudes": {nm: summary_gps[nm].amplitude for nm in all_names},
        "discrepancy_dominant": discrepancy_dominant,
        "noise_inflation": s_extra_j,
        "noise_inflation_factor": float(s),
        "drift_rate": float(drift),
        "baseline_drift_estimate": baseline_drift,
        "engine": engine,
    }
    if want_summaries and level == "scalar":
        diagnostics["warning"] = (
            "curve_summary level requested but some HF records lack summaries; "
            "fell back to scalar-level inference"
        )
    if cfg5.inference_engine == "amortized_npe" and engine != "amortized_npe":
        diagnostics["warning"] = (
            "amortized_npe requested but unavailable (missing weights or "
            "non-curve-summary level); fell back to joint_map"
        )

    return Step5Result(
        theta_map=theta_map,
        theta_cov=theta_cov,
        drift_rate=float(drift),
        drift_rate_sd=drift_sd,
        noise_inflation=s_extra_j,
        gp=gp,
        t_ref=t_ref,
        x_train=x_hf,
        residual_noise_sd=noise_j,
        observation_level=level,
        lf_channel=lf_channel,
        n_lf_physics=len(lf_phys_recs),
        summary_names=tuple(all_names),
        summary_gps=summary_gps,
        classification=classification,
        identifiability=identifiability,
        lf_link=lf_link,
        diagnostics=diagnostics,
        step6_ready=True,
        inference_engine=engine,
        sampler=sampler,
    )

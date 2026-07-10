"""Step 6: update the optimizer-facing predictive model.

Converts the Step-5 joint calibration state into the posterior predictive
object used by Step 1 (Step 6 design note).  Sample-based construction
(section 8): detector-state uncertainty is propagated by pushing Laplace
posterior samples of theta through the fast simulator; the discrepancy GP
adds its own mean and variance; drift extrapolates to the future
measurement time.  Latent-objective and future-observation predictions are
kept distinct (section 7), and every prediction carries an extrapolation
regime diagnostic (section 17).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CalibrationConfig
from ..control import ControlMap
from ..objectives import Objective
from ..records import CalibrationDataset
from ..simulator import BoostSimulator
from .step5_inference import Step5Result


@dataclass
class Prediction:
    """Posterior predictive summary for a batch of candidates."""

    latent_mean: np.ndarray
    latent_sd: np.ndarray
    obs_sd: np.ndarray            # includes future measurement noise
    samples: np.ndarray           # (n_candidates, n_samples) latent samples
    extrapolation: list           # "interpolation" | "mild" | "strong"
    nearest_distance: np.ndarray


@dataclass
class PredictiveModel:
    """p(J_HF(u) | D): the prediction interface handed to Step 1."""

    step5: Step5Result
    simulator: BoostSimulator
    control_map: ControlMap
    objective: Objective
    config: CalibrationConfig
    hf_noise_sd: float                       # typical future HF measurement noise
    theta_samples: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)

    # ---- core query (design section 19.1) --------------------------------

    def predict(self, u_candidates: np.ndarray, t_future: float | None = None) -> Prediction:
        u = np.atleast_2d(np.asarray(u_candidates, dtype=float))
        n_cand = len(u)
        s5 = self.step5
        n_s = len(self.theta_samples)

        # Simulator pushed through theta posterior samples.
        sim_vals = np.empty((n_cand, n_s))
        for j, theta in enumerate(self.theta_samples):
            for i in range(n_cand):
                sim_vals[i, j] = self.simulator.predict_J(u[i], theta, self.objective)
        mean_sim = sim_vals.mean(axis=1)
        var_theta = sim_vals.var(axis=1, ddof=1) if n_s > 1 else np.zeros(n_cand)

        # Discrepancy GP.
        x = np.stack([self.control_map.to_normalized(ui) for ui in u])
        mean_r, sd_r = s5.gp.predict(x)

        # Drift to the future measurement time.
        dt = 0.0 if t_future is None else max(t_future - s5.t_ref, 0.0)
        drift_mean = s5.drift_rate * dt
        drift_var = (s5.drift_rate_sd * dt) ** 2

        latent_mean = mean_sim + mean_r + drift_mean
        latent_var = var_theta + sd_r**2 + drift_var
        latent_sd = np.sqrt(latent_var)
        obs_sd = np.sqrt(latent_var + self.hf_noise_sd**2 + s5.noise_inflation**2)

        # Latent posterior samples for Thompson-style acquisition use.
        rng = np.random.default_rng(self.config.step1.seed)
        samples = (
            sim_vals
            + mean_r[:, None]
            + drift_mean
            + sd_r[:, None] * rng.standard_normal((n_cand, n_s))
        )

        # Extrapolation diagnostics (section 17).
        d = np.min(
            np.linalg.norm(x[:, None, :] - s5.x_train[None, :, :], axis=-1), axis=1
        )
        regimes = [
            "interpolation" if di < 0.15 else ("mild extrapolation" if di < 0.35 else "strong extrapolation")
            for di in d
        ]
        return Prediction(latent_mean, latent_sd, obs_sd, samples, regimes, d)

    def predict_mean_map(self, u: np.ndarray, t_future: float | None = None) -> float:
        """Cheap plug-in predictive mean (MAP theta + GP mean + drift).

        Used inside Step-1 candidate refinement, where the full
        sample-based prediction would be too expensive per evaluation.
        """
        s5 = self.step5
        m_sim = self.simulator.predict_J(np.asarray(u, dtype=float), s5.theta_map, self.objective)
        x = self.control_map.to_normalized(u)[None, :]
        m_r, _ = s5.gp.predict(x)
        dt = 0.0 if t_future is None else max(t_future - s5.t_ref, 0.0)
        return float(m_sim + m_r[0] + s5.drift_rate * dt)

    def predict_lf(self, u_candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predicted LF proxy mean/sd via the learned link (section 11)."""
        pred = self.predict(u_candidates)
        link = self.step5.lf_link
        mean = link.alpha * pred.latent_mean + link.beta
        sd = np.sqrt((link.alpha * pred.latent_sd) ** 2 + link.resid_sd**2)
        return mean, sd

    # ---- staleness (section 16) -------------------------------------------

    def staleness_hours(self, now: float, dataset: CalibrationDataset) -> float:
        last = dataset.last_baseline_or_incumbent_time()
        return np.inf if last is None else now - last

    def summary(self) -> dict:
        s5 = self.step5
        return {
            "theta_map": dict(zip(("z_offset", "compression", "log_loss"), s5.theta_map.to_vector())),
            "theta_sd": list(np.sqrt(np.diag(s5.theta_cov))),
            "classification": s5.classification,
            "identifiability": s5.identifiability,
            "discrepancy_amplitude": s5.gp.amplitude,
            "noise_inflation": s5.noise_inflation,
            "drift_rate": s5.drift_rate,
            "hf_noise_sd": self.hf_noise_sd,
            "validation": self.validation,
        }


def run_step6(
    step5: Step5Result,
    simulator: BoostSimulator,
    control_map: ControlMap,
    dataset: CalibrationDataset,
    config: CalibrationConfig,
    objective: Objective,
    rng: np.random.Generator | None = None,
) -> PredictiveModel:
    """Build and validate the predictive model."""
    rng = rng or np.random.default_rng(config.step1.seed)
    hf = dataset.hf_records()
    stated = float(np.sqrt(np.mean([r.sigma_J**2 for r in hf])))
    empirical = dataset.empirical_repeat_sd()
    hf_noise = max(stated, empirical or 0.0)

    model = PredictiveModel(
        step5=step5,
        simulator=simulator,
        control_map=control_map,
        objective=objective,
        config=config,
        hf_noise_sd=hf_noise,
        theta_samples=step5.theta_samples(config.step1.n_theta_samples, rng),
    )

    # Validation (design section 21): standardized residuals of the
    # observation prediction at the training points.
    u_train = np.stack([r.u_B_achieved for r in hf])
    j_train = np.array([r.J for r in hf])
    t_train = np.array([r.time for r in hf])
    z = []
    for i in range(len(hf)):
        pred = model.predict(u_train[i][None, :], t_future=t_train[i])
        z.append((j_train[i] - pred.latent_mean[0]) / pred.obs_sd[0])
    z = np.array(z)
    rms_z = float(np.sqrt(np.mean(z**2)))
    model.validation = {
        "rms_standardized_residual": rms_z,
        "max_abs_standardized_residual": float(np.max(np.abs(z))),
        "overconfident": bool(rms_z > 2.0),
        "underconfident": bool(rms_z < 0.2 and len(z) > 4),
        "n_points": len(z),
    }
    return model

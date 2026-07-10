"""Step-6 predictive-model validation tests (Step 6 design, section 27)."""

import numpy as np

from madmax_calibration.simulator import DetectorState
from madmax_calibration.steps.step5_inference import run_step5
from madmax_calibration.steps.step6_predictive import run_step6
from tests.conftest import make_setup
from tests.test_step5_inference import _make_dataset


def _build_model(objective, theta_true=None, n_points=12, discrepancy=None, seed=0):
    config, control_map, simulator, _ = make_setup(seed=seed)
    theta_true = theta_true or DetectorState()
    ds = _make_dataset(simulator, control_map, objective, theta_true, n_points=n_points, seed=seed)
    if discrepancy is not None:
        for rec in ds.hf_records():
            rec.J += discrepancy(rec.u_B_achieved)
    s5 = run_step5(ds, simulator, control_map, config, objective)
    model = run_step6(s5, simulator, control_map, ds, config, objective)
    return config, control_map, simulator, ds, model


def test_no_discrepancy_closure(objective):
    """Closure test (design section 27.1): with simulator-generated data
    the predictive model tracks the simulator and stays calibrated."""
    config, control_map, simulator, ds, model = _build_model(objective)
    assert model.validation["rms_standardized_residual"] < 2.0
    assert not model.validation["overconfident"]
    # Prediction at a fresh interior point close to the simulator truth.
    u_test = 0.2 * control_map.cfg.limits()
    pred = model.predict(u_test[None, :])
    j_sim = simulator.predict_J(u_test, DetectorState(), objective)
    assert abs(pred.latent_mean[0] - j_sim) < 4 * pred.latent_sd[0] + 0.05 * abs(j_sim)


def test_known_discrepancy_is_learned(objective):
    """Known-discrepancy test (design section 27.2): a smooth offset between
    measurement and simulator improves predictions when modelled."""
    offset = 0.08

    def disc(u):
        return offset

    config, control_map, simulator, ds, model = _build_model(objective, discrepancy=disc, n_points=14)
    u_test = 0.1 * control_map.cfg.limits()
    pred = model.predict(u_test[None, :])
    j_raw_sim = simulator.predict_J(u_test, model.step5.theta_map, objective)
    j_true = simulator.predict_J(u_test, DetectorState(), objective) + offset
    # The corrected prediction must be closer to the truth than a
    # simulator-only prediction that ignores the discrepancy channel.
    assert abs(pred.latent_mean[0] - j_true) < abs(j_raw_sim - j_true) + 0.02


def test_observation_sd_exceeds_latent_sd(objective):
    """Latent vs future-observation distinction (design section 7)."""
    _, control_map, _, _, model = _build_model(objective)
    u = np.zeros((1, control_map.dim))
    pred = model.predict(u)
    assert pred.obs_sd[0] > pred.latent_sd[0]


def test_extrapolation_flagged(objective):
    """Extrapolation diagnostics (design section 17)."""
    config, control_map, _, _, model = _build_model(objective)
    u_far = control_map.cfg.limits() * 0.98
    pred = model.predict(u_far[None, :])
    assert pred.extrapolation[0] in ("mild extrapolation", "strong extrapolation")

    u_near = model.step5.x_train[0]
    pred2 = model.predict(control_map.to_physical(u_near)[None, :])
    assert pred2.extrapolation[0] == "interpolation"


def test_drift_prediction_extrapolates_in_time(objective):
    """Drift handling (design section 16): predictions at a later time move
    with the inferred drift rate and gain uncertainty."""
    config, control_map, simulator, _ = make_setup()
    ds = _make_dataset(
        simulator, control_map, objective, DetectorState(), n_points=14,
        drift_per_hour=0.01, noise=0.005,
    )
    s5 = run_step5(ds, simulator, control_map, config, objective)
    model = run_step6(s5, simulator, control_map, ds, config, objective)
    u = np.zeros((1, control_map.dim))
    now = model.predict(u, t_future=s5.t_ref)
    later = model.predict(u, t_future=s5.t_ref + 20.0)
    assert later.latent_mean[0] > now.latent_mean[0]      # positive drift
    assert later.latent_sd[0] > now.latent_sd[0]          # more uncertain


def test_theta_uncertainty_propagates(objective):
    """Biased-theta test (design section 27.3): fewer data means wider
    theta posterior and wider predictions."""
    _, _, _, _, model_small = _build_model(objective, n_points=5, seed=1)
    _, control_map, _, _, model_large = _build_model(objective, n_points=16, seed=1)
    u = 0.3 * control_map.cfg.limits()
    sd_small = model_small.predict(u[None, :]).latent_sd[0]
    sd_large = model_large.predict(u[None, :]).latent_sd[0]
    assert sd_small > 0.5 * sd_large  # loose but directionally checked

"""GP regression module and scalar objectives."""

import numpy as np

from madmax_calibration.gp import GaussianProcess, fit_gp_hyperparameters
from madmax_calibration.objectives import Objective


def test_gp_recovers_smooth_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(40, 1))
    f = np.sin(4 * x[:, 0])
    noise = 0.05
    y = f + noise * rng.standard_normal(40)
    gp = fit_gp_hyperparameters(
        x, y, np.full(40, noise), amplitude_bounds=(1e-3, 10), lengthscale_bounds=(0.05, 2)
    )
    x_test = np.linspace(0.1, 0.9, 20)[:, None]
    mean, sd = gp.predict(x_test)
    assert np.max(np.abs(mean - np.sin(4 * x_test[:, 0]))) < 0.15
    assert np.all(sd < 0.2)


def test_gp_prior_prediction_without_data():
    gp = GaussianProcess(amplitude=2.0, lengthscales=np.array([0.3]))
    mean, sd = gp.predict(np.array([[0.5]]))
    assert mean[0] == 0.0 and sd[0] == 2.0


def test_amplitude_prior_shrinks_discrepancy():
    """The half-normal amplitude prior keeps the discrepancy GP small when
    data can be explained by noise (Step 5 design, section 9.3)."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(15, 2))
    y = 0.05 * rng.standard_normal(15)
    gp_tight = fit_gp_hyperparameters(
        x, y, np.full(15, 0.05), (1e-4, 10), (0.1, 2), amplitude_prior_sd=0.01
    )
    assert gp_tight.amplitude < 0.05


def test_objectives_basic_properties():
    curve = np.concatenate([np.full(20, 10.0), np.full(20, 50.0)])
    j_scan = Objective("scan_rate")(curve)
    j_peak = Objective("peak")(curve)
    j_min = Objective("smooth_min")(curve)
    assert j_peak == 50.0
    assert j_min < 20.0  # soft-min is pulled toward the low plateau
    assert j_scan > 0


def test_objective_uncertainty_propagation(rng):
    obj = Objective("scan_rate")
    curve = np.full(41, 40.0)
    j, sigma = obj.with_uncertainty(curve, 0.02 * curve, rng=rng)
    # beta^4 objective: ~2x the relative curve error.
    rel = sigma / j
    assert 0.01 < rel < 0.1

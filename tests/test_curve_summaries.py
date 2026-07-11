"""Curve-summary likelihood tests (roadmap Phase 1.1).

Covers the summarizer itself, the Step-4 record plumbing, and the core
acceptance property: on identical data, curve-summary inference recovers
the correctable detector-state parameters with materially smaller
posterior error than scalar inference, and makes the loss parameter
identifiable at all.
"""

import numpy as np
import pytest

from madmax_calibration.objectives import Objective
from madmax_calibration.records import (
    ActionType,
    CalibrationDataset,
    Fidelity,
    MeasurementRecord,
)
from madmax_calibration.simulator import DetectorState
from madmax_calibration.steps.step5_inference import run_step5
from madmax_calibration.summaries import SUMMARY_NAMES, CurveSummarizer
from tests.conftest import make_setup


@pytest.fixture(scope="module")
def setup_and_summarizer():
    config, control_map, simulator, hardware = make_setup()
    objective = Objective(config.objective)
    return config, control_map, simulator, CurveSummarizer(objective, simulator.freqs)


# ---------------------------------------------------------------------------
# The summarizer
# ---------------------------------------------------------------------------

def test_summary_vector_layout(setup_and_summarizer):
    _, _, simulator, summ = setup_and_summarizer
    curve = simulator.beta2(np.zeros(simulator.control_map.dim), DetectorState())
    z = summ(curve)
    assert z.shape == (len(SUMMARY_NAMES),)
    assert SUMMARY_NAMES[0] == "J"
    assert z[0] == pytest.approx(summ.objective(curve))
    assert np.exp(z[1]) <= curve.max() + 1e-9      # smooth peak <= true peak
    assert abs(z[2]) < 1.0                          # centroid inside the window


def test_summaries_distinguish_shift_from_amplitude(setup_and_summarizer):
    """The whole point of Phase 1.1: a frequency shift and an amplitude
    loss with the same scalar J are separable at summary level."""
    _, _, simulator, summ = setup_and_summarizer
    freqs = simulator.freqs
    half = 0.5 * (freqs[-1] - freqs[0])
    base = 40.0 * np.exp(-0.5 * ((freqs - freqs.mean()) / (0.4 * half)) ** 2)
    shifted = 40.0 * np.exp(-0.5 * ((freqs - freqs.mean() - 0.3 * half) / (0.4 * half)) ** 2)
    # Scale the shifted curve so both have (nearly) the same objective.
    obj = summ.objective
    shifted *= np.sqrt(obj(base) / obj(shifted))
    z_base, z_shift = summ(base), summ(shifted)
    assert abs(z_base[0] - z_shift[0]) < 0.02 * abs(z_base[0])   # same J...
    assert abs(z_base[2] - z_shift[2]) > 0.1                     # ...different centroid


def test_summary_uncertainty_propagation(setup_and_summarizer, rng):
    _, _, simulator, summ = setup_and_summarizer
    curve = simulator.beta2(np.zeros(simulator.control_map.dim), DetectorState())
    z, sigma_z = summ.with_uncertainty(curve, 0.02 * curve, rng=rng)
    assert np.all(sigma_z > 0)
    # Component 0 matches the scalar objective's own propagation scale.
    j, sigma_j = summ.objective.with_uncertainty(curve, 0.02 * curve, rng=rng)
    assert z[0] == pytest.approx(j)
    assert 0.3 * sigma_j < sigma_z[0] < 3.0 * sigma_j


def test_summaries_are_smooth_in_theta(setup_and_summarizer):
    """No argmax jumps: summaries respond continuously to small geometry
    changes (needed for finite-difference MAP optimization)."""
    _, _, simulator, summ = setup_and_summarizer
    u0 = np.zeros(simulator.control_map.dim)
    z1 = simulator.predict_summaries(u0, DetectorState(z_offset=100e-6), summ)
    z2 = simulator.predict_summaries(u0, DetectorState(z_offset=101e-6), summ)
    rel = np.abs(z2 - z1) / (np.abs(z1) + 1e-9)
    assert np.all(rel < 0.05)


# ---------------------------------------------------------------------------
# Step-5 A/B on identical data
# ---------------------------------------------------------------------------

def _make_summary_dataset(simulator, control_map, summ, theta_true, n_points=10, seed=0):
    rng = np.random.default_rng(seed)
    ds = CalibrationDataset()
    limits = control_map.cfg.limits()
    for i in range(n_points):
        u = rng.uniform(-0.5, 0.5, control_map.dim) * limits
        curve = simulator.beta2(u, theta_true) * simulator.aligned_coupling(u, theta_true)
        meas = curve * (1 + 0.01 * rng.standard_normal()) * (
            1 + 0.01 * rng.standard_normal(len(curve))
        )
        sig = meas * np.sqrt(2) * 0.01
        z, sz = summ.with_uncertainty(meas, sig, rng=rng, n_samples=128)
        ds.append(
            MeasurementRecord(
                candidate_id=f"c{i}", iteration=i, fidelity=Fidelity.HF,
                action=ActionType.NEW_CANDIDATE, time_start=float(i), time_end=float(i),
                u_B_cmd=u, u_B_achieved=u,
                J=float(z[0]), sigma_J=float(sz[0]), summaries=z, summaries_sigma=sz,
            )
        )
    return ds


@pytest.fixture(scope="module")
def ab_results(setup_and_summarizer):
    config, control_map, simulator, summ = setup_and_summarizer
    theta_true = DetectorState(z_offset=0.8e-3, compression=0.4e-3, log_loss=0.3)
    ds = _make_summary_dataset(simulator, control_map, summ, theta_true)
    objective = summ.objective
    results = {}
    for level in ("scalar", "curve_summary"):
        config.step5.observation_level = level
        results[level] = run_step5(ds, simulator, control_map, config, objective)
    config.step5.observation_level = "curve_summary"
    return theta_true, results


def test_summary_level_is_used(ab_results):
    _, results = ab_results
    assert results["scalar"].observation_level == "scalar"
    assert results["curve_summary"].observation_level == "curve_summary"
    assert set(results["curve_summary"].summary_gps) == set(SUMMARY_NAMES)


def test_summary_inference_tightens_correctable_parameters(ab_results):
    """Acceptance core: materially smaller posterior error on identical
    data (roadmap Phase 1.1)."""
    theta_true, results = ab_results
    sd_scalar = np.sqrt(np.diag(results["scalar"].theta_cov))
    sd_summary = np.sqrt(np.diag(results["curve_summary"].theta_cov))
    # Posterior sd shrinks by at least 2x for both correctable parameters.
    assert sd_summary[0] < 0.5 * sd_scalar[0]
    assert sd_summary[1] < 0.5 * sd_scalar[1]
    # And the estimates are accurate.
    est = results["curve_summary"].theta_map
    assert abs(est.z_offset - theta_true.z_offset) < 3 * sd_summary[0] + 20e-6
    assert abs(est.compression - theta_true.compression) < 3 * sd_summary[1] + 20e-6


def test_summary_inference_identifies_loss(ab_results):
    """The diagnostic loss parameter becomes identifiable: scalar-level
    J barely constrains it, the amplitude-carrying summaries do."""
    theta_true, results = ab_results
    sd_scalar = np.sqrt(results["scalar"].theta_cov[2, 2])
    sd_summary = np.sqrt(results["curve_summary"].theta_cov[2, 2])
    assert sd_summary < 0.3 * sd_scalar
    est = results["curve_summary"].theta_map.log_loss
    assert abs(est - theta_true.log_loss) < 4 * sd_summary + 0.05


def test_fallback_to_scalar_without_summaries(setup_and_summarizer):
    """Records without summaries trigger the documented scalar fallback."""
    config, control_map, simulator, summ = setup_and_summarizer
    from tests.test_step5_inference import _make_dataset

    objective = summ.objective
    ds = _make_dataset(simulator, control_map, objective, DetectorState(), n_points=6)
    config.step5.observation_level = "curve_summary"
    res = run_step5(ds, simulator, control_map, config, objective)
    assert res.observation_level == "scalar"
    assert "warning" in res.diagnostics

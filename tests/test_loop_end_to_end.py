"""End-to-end closed-loop calibration on the simulated detector.

This is the integration-level validation: starting from the degraded
baseline (hidden detector-state errors), the loop must find a validated
improvement without ever proposing an unsafe configuration or exceeding
the budget.
"""

import numpy as np
import pytest

from madmax_calibration.constraints import HardConstraints
from madmax_calibration.hardware import MockHardware
from madmax_calibration.loop import CalibrationLoop
from madmax_calibration.simulator import BoostSimulator, nominal_half_wave_geometry
from madmax_calibration.control import ControlMap
from tests.conftest import make_light_config


@pytest.fixture(scope="module")
def result_and_loop():
    config = make_light_config()
    config.budget.max_hf_measurements = 14
    config.budget.max_total_hours = 40.0
    gaps, thick = nominal_half_wave_geometry(config.simulator)
    control_map = ControlMap(config.control, config.simulator, gaps, thick)
    simulator = BoostSimulator(config.simulator, control_map)
    hardware = MockHardware(simulator, config, seed=1000)
    loop = CalibrationLoop(hardware, simulator, config)
    result = loop.run(max_iterations=8)
    return result, loop


def test_loop_finds_validated_improvement(result_and_loop):
    result, loop = result_and_loop
    assert result.J_best > result.J_baseline, (
        f"no improvement: J_best={result.J_best:.4f} vs J0={result.J_baseline:.4f}"
    )
    # The correction direction must match the hidden truth: the truth has a
    # positive stack z-offset, so the correction must move z_global down.
    n = loop.config.control.n_disk_modes
    assert result.u_B_star[n] < 0


def test_loop_never_violates_hard_constraints(result_and_loop):
    result, loop = result_and_loop
    hard = HardConstraints(loop.control_map, loop.config.control)
    for rec in result.dataset.records:
        if rec.u_B_cmd is not None:
            assert hard.feasible(rec.u_B_cmd), f"unsafe command in record {rec.candidate_id}"


def test_loop_respects_budget(result_and_loop):
    result, loop = result_and_loop
    counts = result.dataset.counts()
    assert counts["hf"] <= loop.config.budget.max_hf_measurements
    assert counts["lf"] <= loop.config.budget.max_lf_measurements
    assert result.feasibility_report["total_cost_hours"] <= loop.config.budget.max_total_hours + 2.0


def test_feasibility_report_complete(result_and_loop):
    result, _ = result_and_loop
    report = result.feasibility_report
    for key in (
        "J_baseline",
        "sigma_J_baseline",
        "J_best_validated",
        "sigma_J_best",
        "improvement",
        "improvement_over_noise",
        "n_hf_measurements",
        "n_lf_measurements",
        "total_cost_hours",
        "stop_reason",
        "theta_map",
        "classification",
    ):
        assert key in report
    assert report["classification"]["log_loss"] == "diagnostic"


def test_achieved_geometry_recorded_everywhere(result_and_loop):
    """Parent proposal section 2.5: the dataset uses achieved readback."""
    result, _ = result_and_loop
    for rec in result.dataset.hf_records():
        assert rec.u_B_achieved is not None
        assert rec.u_A_achieved is not None


def test_theta_estimate_close_to_truth(result_and_loop):
    result, loop = result_and_loop
    truth = loop.hardware.truth.theta
    est = result.step5.theta_map
    sd = np.sqrt(np.diag(result.step5.theta_cov))
    assert abs(est.z_offset - truth.z_offset) < 4 * sd[0] + 0.2e-3
    assert abs(est.compression - truth.compression) < 4 * sd[1] + 0.1e-3

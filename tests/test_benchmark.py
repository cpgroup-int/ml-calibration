"""Benchmark harness (roadmap Phase 0.2): one run end-to-end + reporting."""

import numpy as np
import pytest

from madmax_calibration.benchmark import (
    BenchmarkSummary,
    achievable_objective,
    format_table,
    run_one,
)
from madmax_calibration.loop import build_loop_from_settings
from madmax_calibration.settings import load_settings, write_settings_file


@pytest.fixture(scope="module")
def light_settings(tmp_path_factory):
    """A small, fast benchmark configuration."""
    path = write_settings_file(
        tmp_path_factory.mktemp("bench") / "bench.toml",
        f_min=21e9, f_max=23e9, n_windows=2, n_disks=3,
    )
    text = path.read_text()
    text = text.replace("n_freq = 81", "n_freq = 41")
    text = text.replace("n_candidates = 128", "n_candidates = 64")
    text = text.replace("n_theta_samples = 6", "n_theta_samples = 4")
    text = text.replace("prior_sensitivity_check = true", "prior_sensitivity_check = false")
    text = text.replace("max_hf_measurements = 25", "max_hf_measurements = 12")
    path.write_text(text)
    return path


@pytest.fixture(scope="module")
def run_result(light_settings):
    settings = load_settings(light_settings)
    return run_one(settings, label="bench", seed=0, max_iterations=6)


def test_achievable_objective_beats_degraded_baseline(light_settings):
    settings = load_settings(light_settings)
    loop = build_loop_from_settings(settings, seed=0)
    j_ach = achievable_objective(loop)
    # The achievable optimum must exceed the degraded nominal state.
    from madmax_calibration.simulator import DetectorState

    j_degraded = loop.simulator.predict_J(
        np.zeros(loop.control_map.dim), loop.hardware.truth.theta, loop.objective
    )
    assert j_ach > j_degraded


def test_run_one_produces_complete_metrics(run_result):
    r = run_result
    assert r.safety_ok, "benchmark run violated hard constraints"
    assert r.budget_ok, "benchmark run violated the budget"
    assert r.n_hf >= 3
    assert r.hours > 0
    assert np.isfinite(r.J_baseline) and np.isfinite(r.J_best)
    assert np.isfinite(r.J_achievable)
    assert r.coverage_2sigma is None or 0.0 <= r.coverage_2sigma <= 1.0
    assert r.stop_reason


def test_summary_table_formats(run_result):
    summary = BenchmarkSummary(label="bench", runs=[run_result])
    table = format_table([summary])
    for column in ("label", "improvement", "hf_to_converge", "safety", "budget"):
        assert column in table
    assert "bench" in table
    assert "1/1" in table  # safety and budget compliance columns

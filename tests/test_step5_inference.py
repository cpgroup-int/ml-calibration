"""Step-5 joint inference validation tests (Step 5 design, section 20)."""

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
from tests.conftest import make_setup


def _make_dataset(
    simulator,
    control_map,
    objective,
    theta_true,
    n_points=12,
    noise=0.01,
    drift_per_hour=0.0,
    lf_link=None,
    seed=0,
):
    rng = np.random.default_rng(seed)
    ds = CalibrationDataset()
    limits = control_map.cfg.limits()
    for i in range(n_points):
        u = rng.uniform(-0.5, 0.5, control_map.dim) * limits
        t = float(i)  # one measurement per hour
        j_true = simulator.predict_J(u, theta_true, objective) + drift_per_hour * t
        j = j_true + noise * rng.standard_normal()
        rec = MeasurementRecord(
            candidate_id=f"c{i}",
            iteration=i,
            fidelity=Fidelity.HF,
            action=ActionType.NEW_CANDIDATE,
            time_start=t,
            time_end=t,
            u_B_cmd=u,
            u_B_achieved=u,
            J=j,
            sigma_J=noise,
            baseline_or_incumbent="baseline" if i % 4 == 0 else None,
        )
        ds.append(rec)
        if lf_link is not None:
            a, b = lf_link
            lf = MeasurementRecord(
                candidate_id=f"lf{i}",
                iteration=i,
                fidelity=Fidelity.LF_PROXY,
                action=ActionType.LF_PROBE,
                time_start=t,
                time_end=t,
                u_B_cmd=u,
                u_B_achieved=u,
                proxy_value=a * j_true + b + 0.5 * noise * rng.standard_normal(),
                proxy_sigma=0.5 * noise,
            )
            ds.append(lf)
    return ds


def test_synthetic_recovery_of_correctable_parameters(objective):
    """Synthetic recovery test (design section 20.1)."""
    config, control_map, simulator, _ = make_setup()
    theta_true = DetectorState(z_offset=0.4e-3, compression=-0.15e-3, log_loss=0.0)
    ds = _make_dataset(simulator, control_map, objective, theta_true, n_points=14)
    res = run_step5(ds, simulator, control_map, config, objective)
    sd = np.sqrt(np.diag(res.theta_cov))
    assert abs(res.theta_map.z_offset - theta_true.z_offset) < 3 * sd[0] + 0.1e-3
    assert abs(res.theta_map.compression - theta_true.compression) < 3 * sd[1] + 0.05e-3
    assert res.step6_ready


def test_classification_labels_follow_control_basis(objective):
    """Correctable-versus-diagnostic test (design section 20.5)."""
    config, control_map, simulator, _ = make_setup()
    ds = _make_dataset(simulator, control_map, objective, DetectorState(), n_points=8)
    res = run_step5(ds, simulator, control_map, config, objective)
    assert res.classification["z_offset"] == "correctable"
    assert res.classification["compression"] == "correctable"
    assert res.classification["log_loss"] == "diagnostic"


def test_drift_detected(objective):
    """Drift test (design section 20.4): a linear objective drift is
    picked up by the drift term."""
    config, control_map, simulator, _ = make_setup()
    drift = 0.01  # J units per hour, ~1% of J per hour
    ds = _make_dataset(
        simulator, control_map, objective, DetectorState(), n_points=14,
        drift_per_hour=drift, noise=0.005,
    )
    res = run_step5(ds, simulator, control_map, config, objective)
    assert res.drift_rate > 0.3 * drift


def test_lf_link_learned_but_not_pooled(objective):
    """Multi-fidelity consistency test (design section 20.6): the affine
    proxy relation is learned; LF data are never treated as HF."""
    config, control_map, simulator, _ = make_setup()
    ds = _make_dataset(
        simulator, control_map, objective, DetectorState(), n_points=10,
        lf_link=(0.85, 0.05),
    )
    res = run_step5(ds, simulator, control_map, config, objective)
    assert res.lf_link.n_points == 10
    assert res.lf_link.validated
    assert abs(res.lf_link.alpha - 0.85) < 0.2
    # LF records contributed no J values.
    assert all(r.J is None for r in ds.lf_records())


def test_prior_sensitivity_flags_weak_identifiability(objective):
    """Confounding test (design section 20.2): with few points and large
    discrepancy freedom the check runs and produces labels."""
    config, control_map, simulator, _ = make_setup()
    config.step5.prior_sensitivity_check = True
    ds = _make_dataset(simulator, control_map, objective, DetectorState(z_offset=0.3e-3), n_points=5)
    res = run_step5(ds, simulator, control_map, config, objective)
    assert set(res.identifiability.keys()) == {"z_offset", "compression", "log_loss"}
    assert all(v in ("ok", "weak") for v in res.identifiability.values())


def test_requires_minimum_hf_points(objective):
    config, control_map, simulator, _ = make_setup()
    ds = _make_dataset(simulator, control_map, objective, DetectorState(), n_points=2)
    with pytest.raises(ValueError):
        run_step5(ds, simulator, control_map, config, objective)

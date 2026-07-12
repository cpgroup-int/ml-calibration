"""Step-4 measurement wrapper tests (Step 4 design, section 21)."""

import numpy as np

from madmax_calibration.records import ActionType, Fidelity, QualityFlag
from madmax_calibration.steps.step4_measure import run_step4
from tests.conftest import make_setup


def _measure(fidelity, pre_flags=None, seed=0):
    config, control_map, simulator, hardware = make_setup(seed=seed)
    from madmax_calibration.objectives import Objective

    obj = Objective(config.objective)
    u0 = np.zeros(control_map.dim)
    hardware.move_booster(u0)
    hardware.move_antenna(hardware.simulator.control_map and np.zeros(2))
    return run_step4(
        hardware,
        config,
        obj,
        candidate_id="c1",
        iteration=1,
        fidelity=fidelity,
        action=ActionType.NEW_CANDIDATE,
        u_B_cmd=u0,
        u_B_achieved=hardware.booster_readback(),
        u_A_cmd=np.zeros(2),
        u_A_achieved=hardware.antenna_readback(),
        pre_flags=pre_flags,
        rng=np.random.default_rng(seed),
    )


def test_hf_record_is_complete():
    """High-fidelity pipeline test (design section 21.2)."""
    rec = _measure(Fidelity.HF)
    assert rec.valid
    assert rec.beta2_curve is not None and rec.beta2_sigma is not None
    assert rec.J is not None and rec.sigma_J is not None and rec.sigma_J > 0
    assert QualityFlag.VALID_HIGH_FIDELITY in rec.quality_flags
    assert rec.cost_hours > 0
    assert rec.u_B_cmd is not None and rec.u_B_achieved is not None
    assert not np.allclose(rec.u_B_cmd, rec.u_B_achieved)  # achieved != commanded


def test_lf_record_never_pretends_to_be_hf():
    """Lower-fidelity pipeline test (design section 21.3): J_HF is
    explicitly not measured; the record carries the reflectivity and
    group-delay curves (roadmap Phase 1.2)."""
    rec = _measure(Fidelity.LF_PROXY)
    assert rec.valid
    assert rec.J is None and rec.sigma_J is None
    assert rec.proxy_value is not None and rec.proxy_sigma is not None
    assert rec.proxy_curves is not None
    for key in ("reflectivity", "reflectivity_sigma", "group_delay", "group_delay_sigma"):
        assert key in rec.proxy_curves
    assert QualityFlag.VALID_LOW_FIDELITY in rec.quality_flags
    assert "not measured" in rec.comments


def test_failed_pre_check_returns_invalid_record_without_fabricating_J():
    """Failure mode 20.1: no J is fabricated when the state check fails."""
    rec = _measure(Fidelity.HF, pre_flags=[QualityFlag.GEOMETRY_OUT_OF_TOLERANCE])
    assert not rec.valid
    assert rec.J is None
    assert QualityFlag.MEASUREMENT_FAILED in rec.quality_flags


def test_objective_consistency_across_repeats():
    """Objective-consistency test (design section 21.6): the same curve
    always maps to the same J."""
    from madmax_calibration.objectives import Objective

    obj = Objective("scan_rate")
    curve = np.linspace(10, 60, 41)
    assert obj(curve) == obj(curve.copy())

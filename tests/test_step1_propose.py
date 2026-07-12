"""Step-1 decision-logic validation tests (Step 1 design, section 25)."""

import numpy as np

from madmax_calibration.config import TrustRegionConfig
from madmax_calibration.constraints import HardConstraints, SoftConstraintModel
from madmax_calibration.records import ActionType, Fidelity
from madmax_calibration.simulator import DetectorState
from madmax_calibration.steps.step1_propose import BudgetState, Step1Proposer, TrustRegion
from tests.conftest import make_setup
from tests.test_step6_predictive import _build_model


def _proposer_setup(objective, theta_true=None, seed=0):
    config, control_map, simulator, ds, model = _build_model(
        objective, theta_true=theta_true or DetectorState(z_offset=0.3e-3), seed=seed
    )
    # The legacy decision-logic tests exercise the HF path; the
    # identification-first LF rule has its own test below.
    config.step5.lf_channel = "off"
    hard = HardConstraints(control_map, config.control)
    soft = SoftConstraintModel()
    proposer = Step1Proposer(config, hard, soft, rng=np.random.default_rng(seed))
    x0 = control_map.to_normalized(np.zeros(control_map.dim))
    tr = TrustRegion(center=x0, size=config.trust_region.initial_size)
    budget = BudgetState(hf_remaining=10, lf_remaining=10, moves_remaining=10, hours_remaining=40.0)
    current = np.zeros(control_map.dim)
    now = float(max(r.time_end for r in ds.records))
    return config, control_map, ds, model, hard, proposer, tr, budget, current, now


def test_proposals_are_always_hard_feasible(objective):
    """Constraint filtering (design section 25, check 1)."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    for seed in range(3):
        proposer.rng = np.random.default_rng(seed)
        prop = proposer.propose(model, ds, tr, budget, current, now)
        if prop.u_B is not None and prop.action == ActionType.NEW_CANDIDATE:
            assert hard.feasible(prop.u_B, current)


def test_noise_response_no_hf_when_improvement_unresolvable(objective):
    """Noise response (design section 25, check 2): when measurement noise
    dwarfs any possible improvement, do not spend a new HF measurement on
    a new candidate."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    model.hf_noise_sd = 10.0  # absurdly noisy instrument
    prop = proposer.propose(model, ds, tr, budget, current, now)
    assert prop.action != ActionType.NEW_CANDIDATE


def test_budget_response_no_hf_without_budget(objective):
    """Budget response (design section 25, check 3)."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    budget.hf_remaining = 0
    prop = proposer.propose(model, ds, tr, budget, current, now)
    assert prop.fidelity != Fidelity.HF
    assert prop.action in (ActionType.LF_PROBE, ActionType.STOP)


def test_drift_response_rebaseline_when_stale(objective):
    """Drift response (design section 25, check 6)."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    stale_now = now + config.step1.rebaseline_after_hours + 1.0
    prop = proposer.propose(model, ds, tr, budget, current, stale_now)
    assert prop.action == ActionType.REBASELINE
    assert prop.fidelity == Fidelity.HF


def test_stop_when_nothing_meaningful_remains(objective):
    """Fallback ladder terminates in a stop recommendation (section 22.5)."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    model.hf_noise_sd = 10.0
    budget.lf_remaining = 0
    prop = proposer.propose(model, ds, tr, budget, current, now)
    # With huge noise the incumbent-sd fallback cannot trigger either
    # (sd threshold scales with noise), so the ladder ends at STOP.
    assert prop.action == ActionType.STOP
    assert prop.reason


def test_identification_first_lf_probes(objective):
    """Roadmap Phase 1.2: with the physics LF channel active and no
    reflectivity data yet, the planner spends cheap identification probes
    before new HF candidates."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    config.step5.lf_channel = "physics"
    prop = proposer.propose(model, ds, tr, budget, current, now)
    assert prop.action == ActionType.LF_PROBE
    assert prop.fallback == "lf_identification"
    assert hard.feasible(prop.u_B, current)
    # Without LF budget the rule cannot fire.
    budget.lf_remaining = 0
    prop2 = proposer.propose(model, ds, tr, budget, current, now)
    assert prop2.fallback != "lf_identification"


def test_trust_region_expands_and_shrinks():
    """Trust-region behaviour (design section 25, check 4)."""
    cfg = TrustRegionConfig(success_tolerance=2, failure_tolerance=2)
    tr = TrustRegion(center=np.full(6, 0.5), size=0.2)
    tr.update(True, cfg)
    tr.update(True, cfg)
    assert tr.size > 0.2
    size_after_expand = tr.size
    tr.update(False, cfg)
    tr.update(False, cfg)
    assert tr.size < size_after_expand
    # Bounds always stay inside [0,1].
    lo, hi = tr.bounds()
    assert np.all(lo >= 0) and np.all(hi <= 1)


def test_soft_constraint_veto(objective):
    """Soft-constraint handling (design section 15): candidates in a
    learned failure region are vetoed."""
    config, control_map, ds, model, hard, proposer, tr, budget, current, now = _proposer_setup(objective)
    # Teach the soft model that the whole trust region fails.
    rng = np.random.default_rng(0)
    lo, hi = tr.bounds()
    for _ in range(10):
        proposer.soft.observe(rng.uniform(lo, hi), False)
    prop = proposer.propose(model, ds, tr, budget, current, now)
    assert prop.action != ActionType.NEW_CANDIDATE

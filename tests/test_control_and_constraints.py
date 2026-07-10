"""Control map, hard-constraint filtering and learned soft constraints."""

import numpy as np

from madmax_calibration.constraints import HardConstraints, SoftConstraintModel
from tests.conftest import make_setup


def test_normalized_physical_round_trip():
    _, control_map, _, _ = make_setup()
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, size=control_map.dim)
    u = control_map.to_physical(x)
    assert np.allclose(control_map.to_normalized(u), x)
    assert control_map.within_travel_limits(u)


def test_travel_limits_rejected():
    config, control_map, _, _ = make_setup()
    hard = HardConstraints(control_map, config.control)
    u_bad = np.zeros(control_map.dim)
    u_bad[config.control.n_disk_modes] = 2 * config.control.z_global_limit
    assert not hard.feasible(u_bad)
    assert hard.feasible(np.zeros(control_map.dim))


def test_min_gap_enforced():
    """A configuration that closes a gap below the safety minimum is
    rejected even inside travel limits."""
    config, control_map, _, _ = make_setup()
    config.control.min_gap = float(control_map.nominal_gaps.min()) - 100e-6
    hard = HardConstraints(control_map, config.control)
    n = config.control.n_disk_modes
    u = np.zeros(control_map.dim)
    u[n + 1] = 0.4e-3  # mirror toward the stack shrinks gap 0 by 0.4 mm
    assert not hard.feasible(u)


def test_max_step_from_current_geometry():
    config, control_map, _, _ = make_setup()
    hard = HardConstraints(control_map, config.control)
    current = np.zeros(control_map.dim)
    u = np.zeros(control_map.dim)
    # A full-limit jump in one coordinate is 0.5 in normalized units,
    # above the 0.35 max step.
    u[config.control.n_disk_modes] = config.control.z_global_limit
    assert hard.feasible(u)                       # inside the domain...
    assert not hard.feasible(u, current)          # ...but too far in one move


def test_soft_constraint_learns_failure_region():
    model = SoftConstraintModel()
    x_bad = np.full(6, 0.9)
    x_good = np.full(6, 0.3)
    for _ in range(5):
        model.observe(x_bad + 0.01 * np.random.default_rng(1).standard_normal(6), False)
        model.observe(x_good, True)
    p_bad = model.probability_feasible(x_bad[None, :])[0]
    p_good = model.probability_feasible(x_good[None, :])[0]
    assert p_bad < 0.5 < p_good

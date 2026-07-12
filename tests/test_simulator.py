"""Physics sanity tests for the 1D transfer-matrix boost simulator."""

import numpy as np

from madmax_calibration.config import SimulatorConfig
from madmax_calibration.control import ControlMap
from madmax_calibration.simulator import (
    BoostSimulator,
    DetectorState,
    _beta2_curves,
    nominal_half_wave_geometry,
)
from tests.conftest import make_setup


def test_mirror_only_boost_is_unity():
    cfg = SimulatorConfig()
    freqs = cfg.frequency_grid()
    beta2 = _beta2_curves(freqs, np.array([5e-3]), np.array([]), cfg.disk_index)
    assert np.allclose(beta2, 1.0, atol=1e-9)


def test_transparent_mode_boost_matches_analytic():
    """Lossless half-wave stack: beta = 1 + 2N(1 - 1/n^2) at the design
    frequency (transparent-mode amplitude addition, Millar et al. 2017)."""
    cfg = SimulatorConfig(n_disks=3, disk_loss_tan=0.0)
    gaps, thick = nominal_half_wave_geometry(cfg)
    beta2 = _beta2_curves(
        np.array([cfg.target_frequency]), gaps, thick, cfg.disk_index
    )
    beta_analytic = 1.0 + 2 * cfg.n_disks * (1.0 - 1.0 / cfg.disk_index**2)
    assert abs(beta2[0] - beta_analytic**2) < 1e-6


def test_detector_state_errors_degrade_objective(objective):
    _, _, simulator, _ = make_setup()
    u0 = np.zeros(simulator.control_map.dim)
    j_nominal = simulator.predict_J(u0, DetectorState(), objective)
    j_err = simulator.predict_J(
        u0, DetectorState(z_offset=0.6e-3, compression=0.25e-3), objective
    )
    assert j_err < 0.9 * j_nominal


def test_control_basis_cancels_correctable_errors(objective):
    """The online control basis can compensate z-offset + compression
    (control-basis consistency, Step 1 design section 25, check 5)."""
    config, control_map, simulator, _ = make_setup()
    theta = DetectorState(z_offset=0.4e-3, compression=-0.2e-3)
    n = config.control.n_disk_modes
    u_fix = np.zeros(control_map.dim)
    # theta displaces disk i by z + c*i; mode 0 displaces by a*(i+1) and
    # z_global by z_g, so a = -c, z_g = -z + c cancels exactly.
    u_fix[0] = -theta.compression
    u_fix[n] = -theta.z_offset + theta.compression

    j_nominal = simulator.predict_J(np.zeros(control_map.dim), DetectorState(), objective)
    j_fixed = simulator.predict_J(u_fix, theta, objective)
    assert abs(j_fixed - j_nominal) < 0.01 * j_nominal


def test_reflection_unitary_when_lossless():
    """Energy conservation: a lossless stack in front of a perfect mirror
    reflects everything, |Gamma|^2 = 1 at every frequency."""
    from madmax_calibration.simulator import _reflection_curves

    cfg = SimulatorConfig(n_disks=3, disk_loss_tan=0.0)
    gaps, thick = nominal_half_wave_geometry(cfg)
    gamma = _reflection_curves(cfg.frequency_grid(), gaps, thick, cfg.disk_index + 0j)
    assert np.allclose(np.abs(gamma) ** 2, 1.0, atol=1e-10)


def test_reflection_loss_absorbs_and_group_delay_positive():
    """With dielectric loss the stack absorbs (|Gamma|^2 < 1) and the
    reflection group delay is the positive round-trip storage time."""
    from madmax_calibration.simulator import group_delay

    _, control_map, simulator, _ = make_setup()
    u0 = np.zeros(control_map.dim)
    refl, gd = simulator.reflectivity_observables(u0, DetectorState(log_loss=0.3))
    assert np.all(refl < 1.0)
    assert np.all(gd > 0.0)
    # Loss reduces the boost objective (physical absorption).
    from madmax_calibration.objectives import Objective

    obj = Objective("scan_rate")
    j0 = obj(simulator.beta2(u0, DetectorState()))
    j_lossy = obj(simulator.beta2(u0, DetectorState(log_loss=1.0)))
    assert j_lossy < j0


def test_reflectivity_responds_to_geometry():
    """The LF channel carries geometry information (roadmap Phase 1.2)."""
    _, control_map, simulator, _ = make_setup()
    u0 = np.zeros(control_map.dim)
    r0, g0 = simulator.reflectivity_observables(u0, DetectorState())
    r1, g1 = simulator.reflectivity_observables(u0, DetectorState(z_offset=500e-6))
    assert np.sqrt(np.mean((r1 - r0) ** 2)) > 0.005   # above LF noise floor
    assert np.max(np.abs(g1 - g0)) > 5e-12            # group delay moves > 5 ps


def test_coupling_decreases_off_beam_and_off_focus():
    _, _, simulator, _ = make_setup()
    on = simulator.coupling(np.zeros(2), 0.0)
    off_beam = simulator.coupling(np.array([5e-3, 0.0]), 0.0)
    off_focus = simulator.coupling(np.zeros(2), 1.5e-3)
    assert on == 1.0
    assert off_beam < 0.6 * on
    assert off_focus < 0.8 * on

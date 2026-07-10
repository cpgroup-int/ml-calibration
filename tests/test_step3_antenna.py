"""Step-3 antenna alignment validation tests (Step 3 design, section 20)."""

import numpy as np
import pytest

from madmax_calibration.config import AntennaConfig
from madmax_calibration.steps.step3_antenna import run_step3


class FakeAlignmentHardware:
    """2D response surface + optional achieved-position error."""

    def __init__(self, fn, seed=0, noise=0.005, repeatability=0.0):
        self.fn = fn
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.repeatability = repeatability
        self.pos = np.zeros(2)
        self.n_measurements = 0

    def move_antenna(self, u_A):
        self.pos = np.asarray(u_A, dtype=float) + self.repeatability * self.rng.standard_normal(2)
        return self.pos.copy()

    def measure_alignment_proxy(self):
        self.n_measurements += 1
        val = self.fn(self.pos) + self.noise * self.rng.standard_normal()
        return float(val), self.noise


def gaussian_beam(center, width=3e-3):
    def fn(p):
        return np.exp(-np.sum((p - center) ** 2) / (2 * width**2))
    return fn


def test_recovers_gaussian_beam_center():
    """Synthetic Gaussian beam test (design section 20.1)."""
    center = np.array([2.0e-3, -1.0e-3])
    hw = FakeAlignmentHardware(gaussian_beam(center))
    res = run_step3(hw, AntennaConfig(), start=np.zeros(2), rng=np.random.default_rng(1))
    assert np.linalg.norm(res.u_A_achieved - center) < 1.5e-3
    assert res.score > 0.8


def test_incumbent_reused_when_still_good():
    """No full scan if the previous alignment validates (design section 9)."""
    center = np.zeros(2)
    hw = FakeAlignmentHardware(gaussian_beam(center))
    res = run_step3(
        hw, AntennaConfig(), start=np.zeros(2), expected_score=0.95,
        rng=np.random.default_rng(2),
    )
    assert res.method == "reused_incumbent"
    assert hw.n_measurements == 1


def test_distorted_surface_falls_back_to_gp_bo():
    """Distorted beam test (design section 20.2): a multimodal surface must
    not break the alignment; the GP-BO fallback still finds a good point."""
    def multimodal(p):
        g1 = np.exp(-np.sum((p - np.array([3e-3, 3e-3])) ** 2) / (2 * (2e-3) ** 2))
        g2 = 0.6 * np.exp(-np.sum((p + np.array([4e-3, 2e-3])) ** 2) / (2 * (1.5e-3) ** 2))
        return g1 + g2

    hw = FakeAlignmentHardware(multimodal, noise=0.01)
    res = run_step3(hw, AntennaConfig(max_evaluations=25), start=np.zeros(2), rng=np.random.default_rng(3))
    assert res.score > 0.7
    assert res.n_evaluations <= 25 + 1


def test_achieved_positions_recorded():
    """Hysteresis/readback test (design section 20.5): the returned and
    logged positions are achieved readbacks, not commands."""
    center = np.array([1e-3, 1e-3])
    hw = FakeAlignmentHardware(gaussian_beam(center), repeatability=100e-6)
    res = run_step3(hw, AntennaConfig(), start=np.zeros(2), rng=np.random.default_rng(4))
    # achieved differs from commanded due to repeatability, and it is the
    # achieved value that is reported
    assert not np.allclose(res.u_A_achieved, res.u_A_cmd)
    assert len(res.data) == res.n_evaluations


def test_respects_measurement_budget():
    """Budget test (design section 20.7)."""
    def flat_noisy(p):
        return 0.5

    cfg = AntennaConfig(max_evaluations=12)
    hw = FakeAlignmentHardware(flat_noisy, noise=0.05)
    res = run_step3(hw, cfg, start=np.zeros(2), rng=np.random.default_rng(5))
    assert hw.n_measurements <= cfg.max_evaluations + 1

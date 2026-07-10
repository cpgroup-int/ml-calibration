"""Shared fixtures: a light configuration so tests stay fast."""

import numpy as np
import pytest

from madmax_calibration.config import CalibrationConfig
from madmax_calibration.control import ControlMap
from madmax_calibration.hardware import MockHardware
from madmax_calibration.objectives import Objective
from madmax_calibration.simulator import BoostSimulator, nominal_half_wave_geometry


def make_light_config() -> CalibrationConfig:
    cfg = CalibrationConfig()
    cfg.simulator.n_freq = 41
    cfg.step1.n_candidates = 64
    cfg.step1.n_theta_samples = 4
    cfg.step5.prior_sensitivity_check = False
    cfg.n_baseline_replicates = 3
    return cfg


def make_setup(seed: int = 0, config: CalibrationConfig | None = None):
    """Config + control map + simulator + mock hardware."""
    config = config or make_light_config()
    gaps, thick = nominal_half_wave_geometry(config.simulator)
    control_map = ControlMap(config.control, config.simulator, gaps, thick)
    simulator = BoostSimulator(config.simulator, control_map)
    hardware = MockHardware(simulator, config, seed=seed + 1000)
    return config, control_map, simulator, hardware


@pytest.fixture
def light_config():
    return make_light_config()


@pytest.fixture
def setup():
    return make_setup()


@pytest.fixture
def objective(light_config):
    return Objective(light_config.objective)


@pytest.fixture
def rng():
    return np.random.default_rng(0)

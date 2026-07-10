"""Hardware interface and a simulated MADMAX detector for offline tests.

The calibration loop never touches physics directly; it goes through
:class:`HardwareInterface`.  On the real experiment this wraps detector
control and the existing gradient-method boost-factor determination
(Step 4 design, section 6: the gradient method is a measurement oracle,
not something this project reimplements).

:class:`MockHardware` is the simulated detector required by the
pre-hardware validation sections of every design note.  It hides:

- true detector-state errors theta* (stack offset, compression, loss),
- a systematic simulator-measurement discrepancy (smooth curve tilt),
- a mis-centred antenna beam and focus optimum,
- actuator hysteresis / repeatability and readback noise
  (achieved != commanded, parent proposal section 2.5),
- slow drift of the stack offset with time,
- measurement noise (shared normalization jitter + per-bin noise),
- a soft measurement-failure region at very poor coupling.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from .config import CalibrationConfig
from .simulator import BoostSimulator, DetectorState


class HardwareInterface(abc.ABC):
    """Minimal detector-control surface used by the calibration loop."""

    @abc.abstractmethod
    def move_booster(self, u_B_cmd: np.ndarray) -> np.ndarray:
        """Move to the commanded booster correction; return achieved readback."""

    @abc.abstractmethod
    def move_antenna(self, u_A_cmd: np.ndarray) -> np.ndarray:
        """Move the antenna; return achieved readback."""

    @abc.abstractmethod
    def booster_readback(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def antenna_readback(self) -> np.ndarray:
        ...

    @abc.abstractmethod
    def measure_alignment_proxy(self) -> tuple[float, float]:
        """Cheap coupling proxy at the current state: (value, sigma)."""

    @abc.abstractmethod
    def measure_boost_factor(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """High-fidelity gradient-method measurement at the current state.

        Returns (beta2_curve, per-bin sigma estimate, success flag).
        """

    @abc.abstractmethod
    def measure_lf_proxy(self) -> tuple[float, float, bool]:
        """Lower-fidelity scalar RF proxy: (value, sigma estimate, success)."""

    @abc.abstractmethod
    def advance_time(self, hours: float) -> None:
        ...

    @property
    @abc.abstractmethod
    def now(self) -> float:
        """Current run time in hours."""


@dataclass
class MockTruth:
    """Hidden ground truth of the simulated detector."""

    theta: DetectorState = field(
        default_factory=lambda: DetectorState(z_offset=0.6e-3, compression=0.25e-3, log_loss=0.3)
    )
    beam_center: np.ndarray = field(default_factory=lambda: np.array([2.5e-3, -1.5e-3]))
    focus_optimum: float = 0.6e-3
    # Systematic multiplicative discrepancy across the window: 1 + tilt * xi,
    # xi in [-1, 1] (unmodelled 3D/receiver-chain effects).
    discrepancy_tilt: float = 0.04
    drift_rate_z: float = 2e-6        # stack z-offset drift [m / hour]


@dataclass
class NoiseModel:
    """Measurement-noise settings of the simulated instrument."""

    hf_norm_jitter: float = 0.01      # shared relative normalization noise
    hf_bin_noise: float = 0.01        # independent relative per-bin noise
    lf_rel_noise: float = 0.05        # relative noise of the LF proxy
    actuator_repeatability: float = 5e-6     # [m] per booster coordinate
    actuator_hysteresis: float = 10e-6       # [m] direction-dependent bias
    readback_noise: float = 2e-6             # [m]
    antenna_repeatability: float = 50e-6     # [m]


class MockHardware(HardwareInterface):
    """Simulated detector: the simulator physics + hidden imperfections."""

    def __init__(
        self,
        simulator: BoostSimulator,
        config: CalibrationConfig,
        truth: MockTruth | None = None,
        noise: NoiseModel | None = None,
        seed: int = 1234,
    ):
        self.simulator = simulator
        self.config = config
        self.truth = truth or MockTruth()
        self.noise = noise or NoiseModel()
        self.rng = np.random.default_rng(seed)
        self._time = 0.0
        dim = simulator.control_map.dim
        self._u_B_cmd = np.zeros(dim)
        self._u_B_achieved = np.zeros(dim)
        self._u_A_cmd = np.zeros(2)
        self._u_A_achieved = np.zeros(2)
        # LF proxy: affine function of the true objective (bias + scale).
        self.lf_scale = 0.85
        self.lf_offset_frac = 0.05
        self.n_hf_calls = 0
        self.n_lf_calls = 0

    # ---- time ------------------------------------------------------------

    @property
    def now(self) -> float:
        return self._time

    def advance_time(self, hours: float) -> None:
        self._time += hours

    def _theta_now(self) -> DetectorState:
        t = self.truth.theta
        return DetectorState(
            z_offset=t.z_offset + self.truth.drift_rate_z * self._time,
            compression=t.compression,
            log_loss=t.log_loss,
        )

    # ---- actuators ---------------------------------------------------------

    def move_booster(self, u_B_cmd: np.ndarray) -> np.ndarray:
        u_B_cmd = np.asarray(u_B_cmd, dtype=float)
        delta = u_B_cmd - self._u_B_cmd
        hyst = self.noise.actuator_hysteresis * np.sign(delta)
        self._u_B_achieved = (
            u_B_cmd
            + hyst
            + self.noise.actuator_repeatability * self.rng.standard_normal(len(u_B_cmd))
        )
        self._u_B_cmd = u_B_cmd.copy()
        self.advance_time(self.config.cost.move_base)
        return self.booster_readback()

    def move_antenna(self, u_A_cmd: np.ndarray) -> np.ndarray:
        u_A_cmd = np.asarray(u_A_cmd, dtype=float)
        self._u_A_achieved = (
            u_A_cmd + self.noise.antenna_repeatability * self.rng.standard_normal(2)
        )
        self._u_A_cmd = u_A_cmd.copy()
        return self.antenna_readback()

    def booster_readback(self) -> np.ndarray:
        return self._u_B_achieved + self.noise.readback_noise * self.rng.standard_normal(
            len(self._u_B_achieved)
        )

    def antenna_readback(self) -> np.ndarray:
        return self._u_A_achieved + self.noise.readback_noise * self.rng.standard_normal(2)

    # ---- true response -------------------------------------------------------

    def _true_curve(self) -> np.ndarray:
        sim = self.simulator
        theta = self._theta_now()
        beta2 = sim.beta2(self._u_B_achieved, theta)
        z_focus = float(self._u_B_achieved[-1])
        eta = sim.coupling(
            self._u_A_achieved,
            z_focus,
            beam_center=self.truth.beam_center,
            focus_optimum=self.truth.focus_optimum,
        )
        xi = np.linspace(-1.0, 1.0, len(beta2))
        systematic = 1.0 + self.truth.discrepancy_tilt * xi
        return beta2 * eta * systematic

    def _true_coupling(self) -> float:
        z_focus = float(self._u_B_achieved[-1])
        return self.simulator.coupling(
            self._u_A_achieved,
            z_focus,
            beam_center=self.truth.beam_center,
            focus_optimum=self.truth.focus_optimum,
        )

    # ---- measurements ----------------------------------------------------------

    def measure_alignment_proxy(self) -> tuple[float, float]:
        """Cheap coupling observable used by Step 3 (noisy true coupling)."""
        eta = self._true_coupling()
        sigma = 0.01 + 0.02 * eta
        value = eta + sigma * self.rng.standard_normal()
        self.advance_time(0.01)
        return float(value), float(sigma)

    def _soft_failure(self) -> bool:
        """Non-damaging measurement failure at very poor coupling."""
        eta = self._true_coupling()
        p_fail = 0.9 if eta < 0.05 else 0.0
        return bool(self.rng.random() < p_fail)

    def measure_boost_factor(self) -> tuple[np.ndarray, np.ndarray, bool]:
        self.n_hf_calls += 1
        self.advance_time(self.config.cost.hf_measurement)
        curve = self._true_curve()
        if self._soft_failure():
            return curve * 0.0, curve * 0.0 + 1.0, False
        norm = 1.0 + self.noise.hf_norm_jitter * self.rng.standard_normal()
        bins = 1.0 + self.noise.hf_bin_noise * self.rng.standard_normal(len(curve))
        measured = np.clip(curve * norm * bins, 0.0, None)
        sigma = measured * np.sqrt(self.noise.hf_norm_jitter**2 + self.noise.hf_bin_noise**2)
        return measured, np.clip(sigma, 1e-12, None), True

    def measure_lf_proxy(self) -> tuple[float, float, bool]:
        """Scalar RF proxy: affine in the true objective, cheap and noisy."""
        from .objectives import Objective

        self.n_lf_calls += 1
        self.advance_time(self.config.cost.lf_measurement)
        if self._soft_failure():
            return 0.0, 1.0, False
        obj = Objective(self.config.objective)
        j_true = obj(self._true_curve())
        value = self.lf_scale * j_true + self.lf_offset_frac * abs(j_true)
        sigma = max(self.noise.lf_rel_noise * abs(j_true), 1e-6)
        return float(value + sigma * self.rng.standard_normal()), float(sigma), True

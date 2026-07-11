"""Configuration objects for the closed-loop calibration.

Every numerical choice that the design notes flag as "to be fixed with the
MADMAX team" (Step 1 design, section 26) lives here with an explicit,
documented default so it can be changed in one place.  See
``docs/DESIGN_DECISIONS.md`` for the rationale behind each default.

Units: lengths in metres, frequencies in Hz, times/costs in hours.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


@dataclass
class SimulatorConfig:
    """Fast 1D physics simulator settings (parent proposal, section 2.3)."""

    n_disks: int = 3                 # MADMAX prototype: 3 disks + mirror
    disk_index: float = 5.0          # LaAlO3-like refractive index
    disk_loss_tan: float = 2e-3      # nominal dielectric loss tangent
    target_frequency: float = 22e9   # centre of the target window W [Hz]
    window_half_width: float = 0.25e9  # W = target +/- half width [Hz]
    n_freq: int = 81                 # frequency-grid points across W

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.target_frequency

    def frequency_grid(self) -> np.ndarray:
        return np.linspace(
            self.target_frequency - self.window_half_width,
            self.target_frequency + self.window_half_width,
            self.n_freq,
        )


@dataclass
class ControlConfig:
    """Online booster control basis (parent proposal, section 4.1).

    ``u_B = (a_disk[0..n_disk_modes-1], z_global, z_mirror, z_focus)``.

    Disk-correction modes (columns of the basis B, Step 1 design section 2):
      mode 0: uniform inter-disk gap change (stack expansion/compression)
      mode 1: linear gradient in inter-disk gaps
      mode 2: quadratic gap profile

    Only ``n_disk_modes`` of these are active.
    """

    n_disk_modes: int = 2
    # Half-widths of the allowed correction box around the nominal
    # configuration, in metres (hard actuator travel limits).
    disk_mode_limit: float = 0.5e-3
    z_global_limit: float = 1.0e-3
    z_mirror_limit: float = 0.5e-3
    z_focus_limit: float = 2e-3
    # Maximum safe single move from the current achieved geometry, in
    # normalized [0,1] units per coordinate (Step 1 design, section 6).
    max_step_normalized: float = 0.35
    # Minimum allowed physical gap between any two adjacent elements
    # (disk collision avoidance; enforced exactly, never learned).
    min_gap: float = 3.0e-3

    @property
    def dim(self) -> int:
        return self.n_disk_modes + 3

    def limits(self) -> np.ndarray:
        """Per-coordinate half-widths of the hard control box (physical)."""
        return np.array(
            [self.disk_mode_limit] * self.n_disk_modes
            + [self.z_global_limit, self.z_mirror_limit, self.z_focus_limit]
        )


@dataclass
class AntennaConfig:
    """Antenna x/y alignment settings (Step 3 design)."""

    travel_limit: float = 10e-3        # |x|,|y| <= limit [m], hard constraint
    initial_scan_step: float = 1.5e-3  # local plus-pattern scan step [m]
    max_evaluations: int = 20          # Step-3 local measurement budget B_A
    noise_repeats: int = 1             # repeats per proxy measurement
    # Improvement must exceed kappa * sigma to accept a new position.
    kappa: float = 2.0


@dataclass
class BudgetConfig:
    """Calibration budget (parent proposal, section 7)."""

    max_hf_measurements: int = 25
    max_lf_measurements: int = 60
    max_booster_moves: int = 60
    max_total_hours: float = 60.0


@dataclass
class CostConfig:
    """Measurement/movement cost model C(u, l) in hours (Step 1, sec. 4.6)."""

    hf_measurement: float = 1.0
    lf_measurement: float = 0.10
    antenna_alignment: float = 0.20
    move_base: float = 0.05
    move_per_normalized_distance: float = 0.10


@dataclass
class TrustRegionConfig:
    """Trust-region policy (Step 1 design, section 14)."""

    initial_size: float = 0.25   # edge half-width in normalized [0,1] space
    min_size: float = 0.02
    max_size: float = 0.6
    expand_factor: float = 1.6
    shrink_factor: float = 0.5
    success_tolerance: int = 2   # consecutive successes before expanding
    failure_tolerance: int = 3   # consecutive failures before shrinking


@dataclass
class Step1Config:
    """Acquisition settings (Step 1 design, sections 9-13)."""

    n_candidates: int = 128          # candidate-pool size per proposal
    lambda_info: float = 0.15        # weight of the information term
    soft_feasibility_threshold: float = 0.5
    # A new HF measurement is only worth it if EI > ei_noise_factor * sigma_J.
    ei_noise_factor: float = 0.3
    # Replicate the incumbent when its posterior sd exceeds this multiple of
    # the HF measurement noise.
    incumbent_sd_factor: float = 1.5
    # Re-baseline when this much time (hours) has passed since the last
    # baseline/incumbent HF measurement (drift-aware rule, section 17).
    rebaseline_after_hours: float = 12.0
    n_theta_samples: int = 8         # posterior samples used for prediction
    seed: int = 0


@dataclass
class Step5Config:
    """Joint inference settings (Step 5 design, section 13.1: Level A)."""

    # Gaussian prior standard deviations for detector-state parameters.
    prior_sd_z_offset: float = 0.5e-3      # global stack z-offset [m]
    prior_sd_compression: float = 0.25e-3  # uniform inter-disk gap error [m]
    prior_sd_log_loss: float = 0.7         # log of loss-scale factor
    # Discrepancy GP prior (scalar objective level).
    discrepancy_amplitude_prior: float = 0.05   # relative to |J0|
    discrepancy_lengthscale_bounds: tuple = (0.1, 2.0)  # normalized units
    # Half-normal prior sd for the extra noise-inflation term (units of J).
    noise_inflation_prior: float = 0.02
    # Prior sd for the linear drift rate (units of J per hour).
    drift_rate_prior: float = 0.002
    min_hf_points_for_inference: int = 3
    prior_sensitivity_check: bool = True


@dataclass
class Step7Config:
    """Stopping policy (parent proposal, Step 7)."""

    # Stop when the best predicted improvement in the trust region is below
    # this multiple of the HF measurement noise for `patience` iterations.
    improvement_noise_factor: float = 0.5
    patience: int = 3
    # Optional absolute target: stop once J_best >= target (None = disabled).
    target_objective: float | None = None


@dataclass
class CalibrationConfig:
    """Top-level configuration aggregating all sub-configs."""

    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    trust_region: TrustRegionConfig = field(default_factory=TrustRegionConfig)
    step1: Step1Config = field(default_factory=Step1Config)
    step5: Step5Config = field(default_factory=Step5Config)
    step7: Step7Config = field(default_factory=Step7Config)
    # Scalar physics objective (parent proposal, section 6). One of
    # "scan_rate", "smooth_min", "peak". Chosen with the physics team;
    # default is the scan-rate proxy over W.
    objective: str = "scan_rate"
    # Number of baseline replicates in Step 0 (noise estimate).
    n_baseline_replicates: int = 3
    seed: int = 0

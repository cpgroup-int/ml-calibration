"""Step 4: measure the selected observable and, when required, the boost factor.

Wraps the hardware measurement oracles in the reproducible protocol of the
Step 4 design note: pre-measurement state checks, execution, uncertainty
estimation, quality control, cost accounting, and a standardized
:class:`~madmax_calibration.records.MeasurementRecord`.
"""

from __future__ import annotations

import numpy as np

from ..config import CalibrationConfig
from ..hardware import HardwareInterface
from ..objectives import Objective
from ..records import ActionType, Fidelity, MeasurementRecord, QualityFlag


def run_step4(
    hardware: HardwareInterface,
    config: CalibrationConfig,
    objective: Objective,
    candidate_id: str,
    iteration: int,
    fidelity: Fidelity,
    action: ActionType,
    u_B_cmd: np.ndarray,
    u_B_achieved: np.ndarray,
    u_A_cmd: np.ndarray | None,
    u_A_achieved: np.ndarray | None,
    pre_flags: list | None = None,
    replicate_group: str | None = None,
    baseline_or_incumbent: str | None = None,
    rng: np.random.Generator | None = None,
) -> MeasurementRecord:
    """Execute the measurement selected by Step 1 and return the record."""
    rng = rng or np.random.default_rng(0)
    flags = list(pre_flags or [])
    t_start = hardware.now
    cost0 = t_start

    record = MeasurementRecord(
        candidate_id=candidate_id,
        iteration=iteration,
        fidelity=fidelity,
        action=action,
        time_start=t_start,
        time_end=t_start,
        u_B_cmd=np.asarray(u_B_cmd, dtype=float).copy(),
        u_B_achieved=np.asarray(u_B_achieved, dtype=float).copy(),
        u_A_cmd=None if u_A_cmd is None else np.asarray(u_A_cmd, dtype=float).copy(),
        u_A_achieved=None if u_A_achieved is None else np.asarray(u_A_achieved, dtype=float).copy(),
        objective_id=objective.objective_id,
        quality_flags=flags,
        replicate_group=replicate_group,
        baseline_or_incumbent=baseline_or_incumbent,
    )

    # Pre-measurement state check (Step 4 design, section 6.1): a candidate
    # already marked invalid is not measured; return a failed record.
    if QualityFlag.GEOMETRY_OUT_OF_TOLERANCE in flags:
        record.valid = False
        record.quality_flags.append(QualityFlag.MEASUREMENT_FAILED)
        record.comments = "pre-measurement geometry check failed; no data taken"
        record.time_end = hardware.now
        return record

    if fidelity in (Fidelity.HF, Fidelity.HF_VALIDATION):
        curve, curve_sigma, success = hardware.measure_boost_factor()
        record.time_end = hardware.now
        record.cost_hours = hardware.now - cost0
        if not success:
            record.valid = False
            record.quality_flags.append(QualityFlag.MEASUREMENT_FAILED)
            record.comments = "gradient-method boost-factor determination failed"
            return record
        record.beta2_curve = curve
        record.beta2_sigma = curve_sigma
        j, sigma_j = objective.with_uncertainty(curve, curve_sigma, rng=rng)
        record.J = j
        record.sigma_J = sigma_j
        # Signal-quality check (section 10.3).
        if np.max(curve) < 1.0:
            record.quality_flags.append(QualityFlag.INSUFFICIENT_SNR)
        if sigma_j > 0.5 * abs(j):
            record.quality_flags.append(QualityFlag.OBJECTIVE_NOT_RESOLVABLE)
        record.quality_flags.append(QualityFlag.VALID_HIGH_FIDELITY)
        # Post-measurement geometry drift check (section 6.3).
        post = hardware.booster_readback()
        if np.max(np.abs(post - record.u_B_achieved)) > 50e-6:
            record.quality_flags.append(QualityFlag.DRIFT_SUSPECTED)
    else:
        value, sigma, success = hardware.measure_lf_proxy()
        record.time_end = hardware.now
        record.cost_hours = hardware.now - cost0
        if not success:
            record.valid = False
            record.quality_flags.append(QualityFlag.MEASUREMENT_FAILED)
            record.comments = "LF proxy measurement failed"
            return record
        record.proxy_value = value
        record.proxy_sigma = sigma
        record.quality_flags.append(QualityFlag.VALID_LOW_FIDELITY)
        # J_HF is explicitly NOT measured in this iteration (section 7.3):
        # record.J stays None.
        record.comments = "J_HF not measured in this iteration"

    return record

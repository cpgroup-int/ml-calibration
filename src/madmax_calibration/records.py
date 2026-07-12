"""Data records exchanged between the calibration steps.

Implements the standardized measurement record of the Step 4 design
(section 13), the Step 1 proposal package (Step 1 design, section 21) and
the accumulated calibration dataset ``D_{1:t}`` (parent proposal,
section 10).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Fidelity(str, Enum):
    """Measurement fidelities / action types (Step 4 design, section 5)."""

    HF = "HF"                        # gradient-method boost-factor measurement
    HF_VALIDATION = "HF_validation"  # conservative final/incumbent validation
    LF_PROXY = "LF_proxy"            # cheap RF/reflectivity proxy observable


class ActionType(str, Enum):
    """Outer-loop action selected by Step 1 (Step 1 design, section 8)."""

    NEW_CANDIDATE = "new_candidate"
    REPLICATE_INCUMBENT = "replicate_incumbent"
    REBASELINE = "rebaseline"
    LF_PROBE = "lf_probe"
    STOP = "stop"


class QualityFlag(str, Enum):
    """Measurement quality flags (Step 4 design, section 4.5)."""

    VALID_HIGH_FIDELITY = "valid_high_fidelity_measurement"
    VALID_LOW_FIDELITY = "valid_low_fidelity_measurement"
    MEASUREMENT_FAILED = "measurement_failed"
    GEOMETRY_OUT_OF_TOLERANCE = "geometry_out_of_tolerance"
    ANTENNA_ALIGNMENT_SUSPECT = "antenna_alignment_suspect"
    DRIFT_SUSPECTED = "drift_suspected"
    INSUFFICIENT_SNR = "insufficient_signal_to_noise"
    PARASITIC_MODE_SUSPECTED = "parasitic_mode_suspected"
    REPEAT_REQUESTED = "repeat_requested"
    OBJECTIVE_NOT_RESOLVABLE = "objective_not_resolvable_above_noise"


_candidate_counter = itertools.count()


def new_candidate_id() -> str:
    return f"cand-{next(_candidate_counter):05d}"


@dataclass
class Proposal:
    """Step-1 output package (Step 1 design, section 21)."""

    action: ActionType
    fidelity: Fidelity | None
    u_B: np.ndarray | None                     # physical booster correction
    candidate_id: str = field(default_factory=new_candidate_id)
    predicted_mean: float | None = None
    predicted_sd: float | None = None
    expected_improvement: float | None = None
    acquisition_value: float | None = None
    expected_cost: float | None = None
    soft_feasibility: float | None = None
    hard_feasible: bool = True
    trust_region_size: float | None = None
    reason: str = ""
    fallback: str = ""

    def diagnostics(self) -> dict[str, Any]:
        """Step-1 diagnostics for the experimental team (section 24)."""
        return {
            "candidate_id": self.candidate_id,
            "action": self.action.value,
            "fidelity": self.fidelity.value if self.fidelity else None,
            "u_B": None if self.u_B is None else list(map(float, self.u_B)),
            "predicted_mean": self.predicted_mean,
            "predicted_sd": self.predicted_sd,
            "expected_improvement": self.expected_improvement,
            "acquisition_value": self.acquisition_value,
            "expected_cost": self.expected_cost,
            "soft_feasibility": self.soft_feasibility,
            "hard_feasible": self.hard_feasible,
            "trust_region_size": self.trust_region_size,
            "reason": self.reason,
            "fallback": self.fallback,
        }


@dataclass
class MeasurementRecord:
    """Standardized Step-4 measurement record (Step 4 design, section 13).

    Every objective value is traceable to geometry + protocol + uncertainty.
    ``J`` is only set for high-fidelity data; for lower-fidelity records it
    stays ``None`` ("J_HF not measured in this iteration").
    """

    candidate_id: str
    iteration: int
    fidelity: Fidelity
    action: ActionType
    time_start: float
    time_end: float
    # Geometry: commanded and achieved (parent proposal, section 2.5).
    u_B_cmd: np.ndarray | None = None
    u_B_achieved: np.ndarray | None = None
    u_A_cmd: np.ndarray | None = None
    u_A_achieved: np.ndarray | None = None
    # Processed observable.
    beta2_curve: np.ndarray | None = None      # HF boost-factor curve
    beta2_sigma: np.ndarray | None = None      # per-bin uncertainty
    proxy_value: float | None = None           # LF scalar proxy
    proxy_sigma: float | None = None
    # Scalar objective (HF only).
    J: float | None = None
    sigma_J: float | None = None
    objective_id: str = ""
    # Curve summaries.  For HF records: boost-curve summaries (component 0
    # is J; roadmap Phase 1.1).  For LF records: reflectivity summaries
    # (roadmap Phase 1.2).  ``observable_id`` says which.
    summaries: np.ndarray | None = None
    summaries_sigma: np.ndarray | None = None
    observable_id: str = ""
    # Raw LF curves (e.g. {"reflectivity": ..., "group_delay": ...}).
    proxy_curves: dict | None = None
    # Bookkeeping.
    quality_flags: list[QualityFlag] = field(default_factory=list)
    valid: bool = True
    replicate_group: str | None = None
    baseline_or_incumbent: str | None = None   # "baseline" | "incumbent" | None
    cost_hours: float = 0.0
    comments: str = ""

    @property
    def time(self) -> float:
        return 0.5 * (self.time_start + self.time_end)

    @property
    def is_hf(self) -> bool:
        return self.fidelity in (Fidelity.HF, Fidelity.HF_VALIDATION)

    def usable_for_inference(self) -> bool:
        return self.valid and QualityFlag.MEASUREMENT_FAILED not in self.quality_flags


@dataclass
class CalibrationDataset:
    """Accumulated calibration data ``D_{1:t}`` with query helpers."""

    records: list[MeasurementRecord] = field(default_factory=list)
    excluded: list[tuple[MeasurementRecord, str]] = field(default_factory=list)

    def append(self, record: MeasurementRecord) -> None:
        self.records.append(record)

    def exclude(self, record: MeasurementRecord, reason: str) -> None:
        """Excluded records are kept with a reason (Step 5, section 14.1)."""
        self.excluded.append((record, reason))

    # ---- queries -------------------------------------------------------

    def hf_records(self, valid_only: bool = True) -> list[MeasurementRecord]:
        out = [r for r in self.records if r.is_hf and r.J is not None]
        if valid_only:
            out = [r for r in out if r.usable_for_inference()]
        return out

    def lf_records(self, valid_only: bool = True) -> list[MeasurementRecord]:
        out = [r for r in self.records if r.fidelity == Fidelity.LF_PROXY]
        if valid_only:
            out = [r for r in out if r.usable_for_inference()]
        return out

    def best_validated(self) -> MeasurementRecord | None:
        """Best validated high-fidelity measurement so far (J_best)."""
        hf = self.hf_records()
        if not hf:
            return None
        return max(hf, key=lambda r: r.J)

    def baseline_records(self) -> list[MeasurementRecord]:
        return [
            r
            for r in self.hf_records()
            if r.baseline_or_incumbent == "baseline"
        ]

    def replicate_groups(self) -> dict[str, list[MeasurementRecord]]:
        groups: dict[str, list[MeasurementRecord]] = {}
        for r in self.hf_records():
            if r.replicate_group:
                groups.setdefault(r.replicate_group, []).append(r)
        return groups

    def empirical_repeat_sd(self) -> float | None:
        """Pooled empirical scatter of replicated HF measurements."""
        sds = []
        for grp in self.replicate_groups().values():
            if len(grp) >= 2:
                vals = np.array([r.J for r in grp])
                sds.append(np.std(vals, ddof=1))
        if not sds:
            return None
        return float(np.sqrt(np.mean(np.square(sds))))

    def last_hf_time(self) -> float | None:
        hf = self.hf_records()
        return max(r.time_end for r in hf) if hf else None

    def last_baseline_or_incumbent_time(self) -> float | None:
        recs = [
            r
            for r in self.hf_records()
            if r.baseline_or_incumbent in ("baseline", "incumbent")
        ]
        return max(r.time_end for r in recs) if recs else None

    def total_cost(self) -> float:
        return sum(r.cost_hours for r in self.records)

    def counts(self) -> dict[str, int]:
        return {
            "hf": len([r for r in self.records if r.is_hf]),
            "lf": len([r for r in self.records if r.fidelity == Fidelity.LF_PROXY]),
            "total": len(self.records),
        }

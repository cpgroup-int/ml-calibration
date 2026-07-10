"""Closed-loop calibration for the MADMAX detector.

This package implements the seven-step closed-loop calibration algorithm
described in ``docs/madmax_closed_loop_calibration_proposal.md`` (version 3)
and its step-specific technical design notes:

- Step 0: initialize, validate baseline, set feasibility limits
- Step 1: propose the next booster-state correction and measurement action
- Step 2: set the booster geometry and record achieved geometry
- Step 3: align the antenna for the fixed booster state
- Step 4: measure the selected observable / boost factor
- Step 5: jointly update detector-state, discrepancy, noise, drift inference
- Step 6: update the optimizer-facing predictive model
- Step 7: stop or repeat

The package is hardware-agnostic: the loop talks to a
:class:`~madmax_calibration.hardware.HardwareInterface`.  A
:class:`~madmax_calibration.hardware.MockHardware` implementation (a
simulated detector with hidden detector-state errors, hysteresis, drift,
and measurement noise) is provided for offline validation, as required by
the pre-hardware validation sections of the design notes.
"""

from .config import CalibrationConfig
from .control import BoosterControl, ControlMap
from .hardware import HardwareInterface, MockHardware
from .loop import CalibrationLoop, CalibrationResult
from .records import (
    ActionType,
    CalibrationDataset,
    Fidelity,
    MeasurementRecord,
    Proposal,
    QualityFlag,
)
from .simulator import BoostSimulator, DetectorState

__all__ = [
    "ActionType",
    "BoosterControl",
    "BoostSimulator",
    "CalibrationConfig",
    "CalibrationDataset",
    "CalibrationLoop",
    "CalibrationResult",
    "ControlMap",
    "DetectorState",
    "Fidelity",
    "HardwareInterface",
    "MeasurementRecord",
    "MockHardware",
    "Proposal",
    "QualityFlag",
]

__version__ = "0.1.0"

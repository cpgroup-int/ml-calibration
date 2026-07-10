"""Step 2: set the booster geometry and record achieved geometry.

Parent proposal, Step 2: move the detector to the proposed booster state
(disk modes, global z, reflecting mirror, focusing mirror are one complete
booster geometry) and record the achieved geometry, not only the commanded
one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..hardware import HardwareInterface
from ..records import QualityFlag


@dataclass
class Step2Result:
    u_B_cmd: np.ndarray
    u_B_achieved: np.ndarray
    within_tolerance: bool
    quality_flags: list


def run_step2(
    hardware: HardwareInterface,
    u_B_cmd: np.ndarray,
    tolerance: float = 20e-6,
    max_retries: int = 1,
) -> Step2Result:
    """Move the booster; verify readback; retry once if out of tolerance.

    ``tolerance`` is the per-coordinate command-vs-achieved acceptance in
    metres.  If the geometry stays out of tolerance the result is flagged
    (never silently treated as achieved == commanded; Step 4 design,
    section 10.1).
    """
    u_B_cmd = np.asarray(u_B_cmd, dtype=float)
    flags: list = []
    achieved = hardware.move_booster(u_B_cmd)
    for _ in range(max_retries):
        if np.all(np.abs(achieved - u_B_cmd) <= tolerance):
            break
        achieved = hardware.move_booster(u_B_cmd)
    within = bool(np.all(np.abs(achieved - u_B_cmd) <= tolerance))
    if not within:
        flags.append(QualityFlag.GEOMETRY_OUT_OF_TOLERANCE)
    return Step2Result(
        u_B_cmd=u_B_cmd,
        u_B_achieved=np.asarray(achieved, dtype=float),
        within_tolerance=within,
        quality_flags=flags,
    )

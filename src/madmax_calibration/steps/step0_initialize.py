"""Step 0: initialize, validate baseline, and set feasibility limits.

Parent proposal, Step 0: start from the nominal configuration q0(W), align
the antenna, measure the baseline boost factor with replication, estimate
the baseline objective J0 and its measurement uncertainty sigma_J0, and
freeze the hard feasibility domain and budget before the loop starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CalibrationConfig
from ..constraints import HardConstraints
from ..hardware import HardwareInterface
from ..objectives import Objective
from ..records import ActionType, CalibrationDataset, Fidelity, new_candidate_id
from .step2_set_geometry import run_step2
from .step3_antenna import run_step3
from .step4_measure import run_step4


@dataclass
class Step0Result:
    J0: float
    sigma_J0: float
    u_B_achieved: np.ndarray
    u_A_achieved: np.ndarray
    n_replicates: int
    noise_consistent: bool     # empirical repeat scatter vs stated sigma
    resolvable: bool           # is sigma_J0 small enough to calibrate at all?


def run_step0(
    hardware: HardwareInterface,
    config: CalibrationConfig,
    objective: Objective,
    hard_constraints: HardConstraints,
    dataset: CalibrationDataset,
    rng: np.random.Generator,
    summarizer=None,
) -> Step0Result:
    """Measure and validate the baseline at u_B = 0."""
    dim = hard_constraints.control_map.dim
    u0 = np.zeros(dim)
    assert hard_constraints.feasible(u0), "nominal configuration must be hard-feasible"

    step2 = run_step2(hardware, u0)
    step3 = run_step3(hardware, config.antenna, rng=rng)
    hardware.advance_time(config.cost.antenna_alignment)

    cid = new_candidate_id()
    group = f"baseline-{cid}"
    js = []
    sigmas = []
    for i in range(max(1, config.n_baseline_replicates)):
        rec = run_step4(
            hardware,
            config,
            objective,
            candidate_id=cid,
            iteration=0,
            fidelity=Fidelity.HF,
            action=ActionType.REBASELINE,
            u_B_cmd=u0,
            u_B_achieved=step2.u_B_achieved,
            u_A_cmd=step3.u_A_cmd,
            u_A_achieved=step3.u_A_achieved,
            replicate_group=group,
            baseline_or_incumbent="baseline",
            rng=rng,
            summarizer=summarizer,
        )
        dataset.append(rec)
        if rec.usable_for_inference() and rec.J is not None:
            js.append(rec.J)
            sigmas.append(rec.sigma_J)

    if not js:
        raise RuntimeError("baseline measurement failed; cannot start calibration")

    j0 = float(np.mean(js))
    stated = float(np.sqrt(np.mean(np.square(sigmas))))
    if len(js) >= 2:
        empirical = float(np.std(js, ddof=1))
        sigma_j0 = max(stated, empirical)
        noise_consistent = empirical <= 2.0 * stated
    else:
        sigma_j0 = stated
        noise_consistent = True

    # Feasibility condition (parent proposal, section 7): expected
    # improvements must be resolvable above the measurement noise.  We use
    # a generous screen here; Step 7 applies the running version.
    resolvable = sigma_j0 < 0.5 * abs(j0)

    return Step0Result(
        J0=j0,
        sigma_J0=sigma_j0,
        u_B_achieved=step2.u_B_achieved,
        u_A_achieved=step3.u_A_achieved,
        n_replicates=len(js),
        noise_consistent=noise_consistent,
        resolvable=resolvable,
    )

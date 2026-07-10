"""Step 7: stop or repeat.

Stopping rules from the parent proposal (Step 7): budget exhaustion,
expected improvement not resolvable above measurement uncertainty for
several consecutive iterations, an optional absolute target, or a Step-1
stop recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CalibrationConfig
from ..records import ActionType, Proposal
from .step1_propose import BudgetState


@dataclass
class Step7Decision:
    stop: bool
    reason: str


def run_step7(
    config: CalibrationConfig,
    proposal: Proposal,
    budget: BudgetState,
    j_best: float | None,
    hf_noise: float,
    consecutive_unresolvable: int,
) -> Step7Decision:
    cfg7 = config.step7

    if proposal.action == ActionType.STOP:
        return Step7Decision(True, f"Step 1 recommended stop: {proposal.reason}")

    if budget.hours_remaining <= 0:
        return Step7Decision(True, "calibration time budget exhausted")
    if budget.hf_remaining <= 0:
        return Step7Decision(True, "high-fidelity measurement budget exhausted")
    if budget.moves_remaining <= 0:
        return Step7Decision(True, "booster movement budget exhausted")

    if cfg7.target_objective is not None and j_best is not None and j_best >= cfg7.target_objective:
        return Step7Decision(True, f"target objective reached (J_best={j_best:.4g})")

    if consecutive_unresolvable >= cfg7.patience:
        return Step7Decision(
            True,
            f"expected improvement below {cfg7.improvement_noise_factor} x HF noise "
            f"({hf_noise:.3g}) for {consecutive_unresolvable} consecutive iterations",
        )

    return Step7Decision(False, "continue")

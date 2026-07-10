"""The outer closed-loop calibration orchestrator.

Runs Step 0 once, then iterates Steps 1-7 (parent proposal, section 11)
against a :class:`~madmax_calibration.hardware.HardwareInterface`,
accumulating the calibration dataset, updating the joint model and the
predictive model, and producing the final validated configuration plus
the feasibility report required by the parent proposal (section 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import CalibrationConfig
from .constraints import HardConstraints, SoftConstraintModel
from .control import ControlMap
from .hardware import HardwareInterface
from .objectives import Objective
from .records import (
    ActionType,
    CalibrationDataset,
    Fidelity,
    MeasurementRecord,
    Proposal,
    QualityFlag,
)
from .simulator import BoostSimulator
from .steps.step0_initialize import Step0Result, run_step0
from .steps.step1_propose import BudgetState, Step1Proposer, TrustRegion
from .steps.step2_set_geometry import run_step2
from .steps.step3_antenna import run_step3
from .steps.step4_measure import run_step4
from .steps.step5_inference import Step5Result, run_step5
from .steps.step6_predictive import PredictiveModel, run_step6
from .steps.step7_stopping import run_step7


@dataclass
class CalibrationResult:
    """Final calibration output (parent proposal, section 3)."""

    u_B_star: np.ndarray                  # best validated booster correction
    u_A_star: np.ndarray                  # antenna position at the best state
    J_best: float
    sigma_J_best: float
    J_baseline: float
    sigma_J_baseline: float
    beta2_curve: np.ndarray | None
    improvement_significant: bool
    stop_reason: str
    dataset: CalibrationDataset = field(repr=False, default=None)
    step5: Step5Result | None = field(repr=False, default=None)
    feasibility_report: dict = field(default_factory=dict)
    history: list = field(default_factory=list, repr=False)


class CalibrationLoop:
    """Seven-step closed-loop calibration driver."""

    def __init__(
        self,
        hardware: HardwareInterface,
        simulator: BoostSimulator,
        config: CalibrationConfig | None = None,
    ):
        self.hardware = hardware
        self.simulator = simulator
        self.config = config or CalibrationConfig()
        self.control_map: ControlMap = simulator.control_map
        self.objective = Objective(self.config.objective)
        self.dataset = CalibrationDataset()
        self.hard = HardConstraints(self.control_map, self.config.control)
        self.soft = SoftConstraintModel()
        self.rng = np.random.default_rng(self.config.seed)
        self.proposer = Step1Proposer(self.config, self.hard, self.soft, rng=self.rng)
        self.history: list[dict] = []

    # ------------------------------------------------------------------

    def _budget(self) -> BudgetState:
        counts = self.dataset.counts()
        b = self.config.budget
        moves_used = sum(
            1 for r in self.dataset.records if r.action in (ActionType.NEW_CANDIDATE, ActionType.REBASELINE)
        )
        return BudgetState(
            hf_remaining=b.max_hf_measurements - counts["hf"],
            lf_remaining=b.max_lf_measurements - counts["lf"],
            moves_remaining=b.max_booster_moves - moves_used,
            hours_remaining=b.max_total_hours - self.hardware.now,
        )

    def _move_with_step_limit(self, u_target: np.ndarray, current: np.ndarray):
        """Split large commanded moves so each hardware move respects the
        max safe step size (Step 1 design, section 6)."""
        max_step = self.config.control.max_step_normalized
        cur = np.asarray(current, dtype=float)
        result = None
        for _ in range(10):
            x_cur = self.control_map.to_normalized(cur)
            x_tgt = self.control_map.to_normalized(u_target)
            delta = x_tgt - x_cur
            scale = np.max(np.abs(delta))
            if scale <= max_step:
                result = run_step2(self.hardware, u_target)
                break
            frac = max_step / scale
            u_mid = self.control_map.to_physical(x_cur + frac * delta)
            result = run_step2(self.hardware, u_mid)
            cur = result.u_B_achieved
        return result

    # ------------------------------------------------------------------

    def _execute_proposal(
        self,
        proposal: Proposal,
        iteration: int,
        current_achieved: np.ndarray,
        last_antenna: np.ndarray | None,
        last_alignment_score: float | None,
    ) -> tuple[MeasurementRecord, np.ndarray, np.ndarray, float]:
        """Steps 2-4 for one proposal.  Returns (record, u_B_achieved,
        u_A_achieved, alignment_score)."""
        # Step 2: set booster geometry.
        step2 = self._move_with_step_limit(proposal.u_B, current_achieved)

        # Step 3: align the antenna for this booster state.
        step3 = run_step3(
            self.hardware,
            self.config.antenna,
            start=last_antenna,
            expected_score=last_alignment_score,
            rng=self.rng,
        )
        self.hardware.advance_time(self.config.cost.antenna_alignment)
        pre_flags = list(step2.quality_flags)
        if step3.quality_flag in ("budget_limited", "noise_limited"):
            pre_flags.append(QualityFlag.ANTENNA_ALIGNMENT_SUSPECT)

        baseline_tag = None
        if proposal.action == ActionType.REBASELINE:
            baseline_tag = "baseline" if np.allclose(proposal.u_B, 0.0) else "incumbent"
        elif proposal.action == ActionType.REPLICATE_INCUMBENT:
            baseline_tag = "incumbent"

        record = run_step4(
            self.hardware,
            self.config,
            self.objective,
            candidate_id=proposal.candidate_id,
            iteration=iteration,
            fidelity=proposal.fidelity,
            action=proposal.action,
            u_B_cmd=proposal.u_B,
            u_B_achieved=step2.u_B_achieved,
            u_A_cmd=step3.u_A_cmd,
            u_A_achieved=step3.u_A_achieved,
            pre_flags=pre_flags,
            replicate_group=proposal.candidate_id if baseline_tag else None,
            baseline_or_incumbent=baseline_tag,
            rng=self.rng,
        )
        self.dataset.append(record)

        # Learned soft constraints see only non-damaging failures.
        x_norm = self.control_map.to_normalized(step2.u_B_achieved)
        self.soft.observe(x_norm, record.usable_for_inference())
        return record, step2.u_B_achieved, step3.u_A_achieved, step3.score

    # ------------------------------------------------------------------

    def run(self, max_iterations: int = 30, verbose: bool = False) -> CalibrationResult:
        cfg = self.config

        # ---- Step 0 ----------------------------------------------------
        step0: Step0Result = run_step0(
            self.hardware, cfg, self.objective, self.hard, self.dataset, self.rng
        )
        if verbose:
            print(f"[step0] J0 = {step0.J0:.4g} +/- {step0.sigma_J0:.4g}")

        x0 = self.control_map.to_normalized(np.zeros(self.control_map.dim))
        trust_region = TrustRegion(center=x0, size=cfg.trust_region.initial_size)
        current_achieved = step0.u_B_achieved
        last_antenna = step0.u_A_achieved
        last_score: float | None = None
        consecutive_unresolvable = 0
        stop_reason = f"reached max_iterations={max_iterations}"
        step5_state: Step5Result | None = None
        model: PredictiveModel | None = None

        # ---- outer loop --------------------------------------------------
        for t in range(1, max_iterations + 1):
            # Steps 5 + 6 (model updates use all data so far).
            step5_state = run_step5(
                self.dataset, self.simulator, self.control_map, cfg, self.objective,
                previous=step5_state, rng=self.rng,
            )
            model = run_step6(
                step5_state, self.simulator, self.control_map, self.dataset, cfg,
                self.objective, rng=self.rng,
            )

            budget = self._budget()
            best_rec = self.dataset.best_validated()
            j_best_before = best_rec.J if best_rec else -np.inf

            # Step 1: proposal.
            proposal = self.proposer.propose(
                model, self.dataset, trust_region, budget, current_achieved, self.hardware.now
            )
            if verbose:
                print(f"[iter {t}] action={proposal.action.value} reason={proposal.reason}")

            # Step 7 (pre-execution).
            decision = run_step7(
                cfg, proposal, budget, j_best_before, model.hf_noise_sd, consecutive_unresolvable
            )
            if decision.stop:
                stop_reason = decision.reason
                break

            # Steps 2-4.
            record, current_achieved, last_antenna, last_score = self._execute_proposal(
                proposal, t, current_achieved, last_antenna, last_score
            )

            # Trust-region + noise bookkeeping.
            improved = (
                record.is_hf
                and record.usable_for_inference()
                and record.J is not None
                and record.J > j_best_before + 0.5 * model.hf_noise_sd
            )
            if proposal.action == ActionType.NEW_CANDIDATE:
                trust_region.update(bool(improved), cfg.trust_region)
                if improved:
                    trust_region.recenter(self.control_map.to_normalized(record.u_B_achieved))
            if improved or (
                proposal.action == ActionType.NEW_CANDIDATE
                and proposal.expected_improvement is not None
                and proposal.expected_improvement
                > cfg.step7.improvement_noise_factor * model.hf_noise_sd
            ):
                consecutive_unresolvable = 0
            else:
                consecutive_unresolvable += 1

            self.history.append(
                {
                    "iteration": t,
                    "proposal": proposal.diagnostics(),
                    "J": record.J,
                    "sigma_J": record.sigma_J,
                    "valid": record.usable_for_inference(),
                    "J_best": (self.dataset.best_validated().J if self.dataset.best_validated() else None),
                    "trust_region_size": trust_region.size,
                    "time_hours": self.hardware.now,
                }
            )

            # Step 7 (post-execution budget check).
            decision = run_step7(
                cfg, proposal, self._budget(), j_best_before, model.hf_noise_sd, consecutive_unresolvable
            )
            if decision.stop:
                stop_reason = decision.reason
                break

        # ---- final HF validation at the best configuration ---------------
        best = self.dataset.best_validated()
        u_star = best.u_B_achieved if best else current_achieved
        final_record = None
        if self._budget().can_afford_hf() and best is not None:
            proposal = Proposal(
                action=ActionType.REPLICATE_INCUMBENT,
                fidelity=Fidelity.HF_VALIDATION,
                u_B=best.u_B_cmd if best.u_B_cmd is not None else u_star,
                reason="final high-fidelity validation of the accepted configuration",
            )
            final_record, current_achieved, last_antenna, _ = self._execute_proposal(
                proposal, max_iterations + 1, current_achieved, last_antenna, last_score
            )
            if not final_record.usable_for_inference():
                final_record = None

        # ---- result + feasibility report ----------------------------------
        best = self.dataset.best_validated()
        j_best = best.J if best else step0.J0
        sigma_best = best.sigma_J if best else step0.sigma_J0
        if final_record is not None and final_record.J is not None:
            j_best = float(np.mean([best.J, final_record.J]))
            sigma_best = float(max(best.sigma_J, final_record.sigma_J) / np.sqrt(2))

        improvement = j_best - step0.J0
        sig = float(np.hypot(sigma_best, step0.sigma_J0))
        counts = self.dataset.counts()
        report = {
            "J_baseline": step0.J0,
            "sigma_J_baseline": step0.sigma_J0,
            "J_best_validated": j_best,
            "sigma_J_best": sigma_best,
            "improvement": improvement,
            "improvement_over_noise": improvement / sig if sig > 0 else np.inf,
            "improvement_significant": bool(improvement > 2.0 * sig),
            "n_hf_measurements": counts["hf"],
            "n_lf_measurements": counts["lf"],
            "total_cost_hours": self.hardware.now,
            "stop_reason": stop_reason,
            "theta_map": (
                dict(zip(("z_offset", "compression", "log_loss"), step5_state.theta_map.to_vector()))
                if step5_state
                else None
            ),
            "identifiability": step5_state.identifiability if step5_state else None,
            "classification": step5_state.classification if step5_state else None,
        }

        return CalibrationResult(
            u_B_star=u_star,
            u_A_star=last_antenna,
            J_best=j_best,
            sigma_J_best=sigma_best,
            J_baseline=step0.J0,
            sigma_J_baseline=step0.sigma_J0,
            beta2_curve=(final_record.beta2_curve if final_record is not None else (best.beta2_curve if best else None)),
            improvement_significant=report["improvement_significant"],
            stop_reason=stop_reason,
            dataset=self.dataset,
            step5=step5_state,
            feasibility_report=report,
            history=self.history,
        )


def build_default_loop(seed: int = 0, config: CalibrationConfig | None = None):
    """Convenience factory: nominal geometry + simulator + mock hardware."""
    from .hardware import MockHardware
    from .simulator import nominal_half_wave_geometry

    config = config or CalibrationConfig()
    gaps, thick = nominal_half_wave_geometry(config.simulator)
    control_map = ControlMap(config.control, config.simulator, gaps, thick)
    simulator = BoostSimulator(config.simulator, control_map)
    hardware = MockHardware(simulator, config, seed=seed + 1000)
    return CalibrationLoop(hardware, simulator, config)

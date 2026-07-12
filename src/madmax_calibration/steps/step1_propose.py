"""Step 1: propose the next booster-state correction and measurement action.

Implements the "recommended" Step 1 of the design note (section 19.2):
a physics-informed, trust-region, constrained, budget-aware Bayesian
decision step.  Candidate generation follows the five-stage procedure of
section 20; the acquisition combines noisy expected improvement, an
information term, soft-constraint feasibility and expected cost
(sections 9-13); noise-aware and drift-aware fallbacks return
replication, re-baseline, LF-probe or stop actions (sections 16-17, 22).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc

from ..config import CalibrationConfig
from ..constraints import HardConstraints, SoftConstraintModel
from ..records import ActionType, CalibrationDataset, Fidelity, Proposal
from .step6_predictive import PredictiveModel


@dataclass
class TrustRegion:
    """Local search region T_t in normalized [0,1]^d (design section 14)."""

    center: np.ndarray            # normalized coordinates
    size: float                   # per-coordinate half-width
    n_success: int = 0
    n_failure: int = 0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.clip(self.center - self.size, 0.0, 1.0)
        hi = np.clip(self.center + self.size, 0.0, 1.0)
        return lo, hi

    def update(self, success: bool, cfg) -> None:
        if success:
            self.n_success += 1
            self.n_failure = 0
            if self.n_success >= cfg.success_tolerance:
                self.size = min(self.size * cfg.expand_factor, cfg.max_size)
                self.n_success = 0
        else:
            self.n_failure += 1
            self.n_success = 0
            if self.n_failure >= cfg.failure_tolerance:
                self.size = max(self.size * cfg.shrink_factor, cfg.min_size)
                self.n_failure = 0

    def recenter(self, center: np.ndarray) -> None:
        self.center = np.clip(np.asarray(center, dtype=float), 0.0, 1.0)


@dataclass
class BudgetState:
    """Remaining calibration budget B_t (design section 4.9)."""

    hf_remaining: int
    lf_remaining: int
    moves_remaining: int
    hours_remaining: float

    def can_afford_hf(self) -> bool:
        return self.hf_remaining > 0 and self.hours_remaining > 0 and self.moves_remaining > 0

    def can_afford_lf(self) -> bool:
        return self.lf_remaining > 0 and self.hours_remaining > 0 and self.moves_remaining > 0


def _expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    """EI for maximization with noisy incumbent (uses latent posterior)."""
    sd = np.clip(sd, 1e-12, None)
    z = (mean - best) / sd
    return sd * (z * norm.cdf(z) + norm.pdf(z))


class Step1Proposer:
    """Outer-loop planner: builds and optimizes the acquisition."""

    def __init__(
        self,
        config: CalibrationConfig,
        hard_constraints: HardConstraints,
        soft_constraints: SoftConstraintModel,
        rng: np.random.Generator | None = None,
    ):
        self.config = config
        self.hard = hard_constraints
        self.soft = soft_constraints
        self.control_map = hard_constraints.control_map
        self.rng = rng or np.random.default_rng(config.step1.seed)
        self._sobol = qmc.Sobol(d=self.control_map.dim, scramble=True, seed=config.step1.seed)

    # ---- candidate generation (design section 20, stages 1-3) ------------

    def _candidates(
        self, trust_region: TrustRegion, current_achieved: np.ndarray
    ) -> np.ndarray:
        lo, hi = trust_region.bounds()
        n = self.config.step1.n_candidates
        raw = self._sobol.random(n)
        x = lo + raw * (hi - lo)
        u = np.stack([self.control_map.to_physical(xi) for xi in x])
        mask = self.hard.filter(u, current_achieved)
        return u[mask]

    def _refine_exploit(
        self,
        model: PredictiveModel,
        u_start: np.ndarray,
        trust_region: TrustRegion,
        current_achieved: np.ndarray,
        t_future: float,
    ) -> np.ndarray | None:
        """Physics-informed exploit candidate: local maximization of the
        plug-in predictive mean inside T_t âˆ© U_hard.  This is the
        'simulator-guided' part of the proposal (design section 4.3): the
        Sobol pool explores, this candidate exploits the calibrated
        simulator."""
        from scipy.optimize import minimize

        lo, hi = trust_region.bounds()
        x0 = np.clip(self.control_map.to_normalized(u_start), lo, hi)

        def neg_mean(x: np.ndarray) -> float:
            x = np.clip(x, lo, hi)
            u = self.control_map.to_physical(x)
            if not self.hard.feasible(u, current_achieved):
                return 1e6
            return -model.predict_mean_map(u, t_future)

        res = minimize(
            neg_mean, x0, method="Nelder-Mead",
            options={"maxfev": 150, "xatol": 1e-3, "fatol": 1e-5},
        )
        x = np.clip(res.x, lo, hi)
        u = self.control_map.to_physical(x)
        return u if self.hard.feasible(u, current_achieved) else None

    # ---- cost model (design section 12) ----------------------------------

    def expected_cost(self, u_B: np.ndarray, fidelity: Fidelity, current_achieved: np.ndarray) -> float:
        c = self.config.cost
        x_new = self.control_map.to_normalized(u_B)
        x_cur = self.control_map.to_normalized(current_achieved)
        dist = float(np.linalg.norm(x_new - x_cur))
        move = c.move_base + c.move_per_normalized_distance * dist
        meas = c.hf_measurement if fidelity in (Fidelity.HF, Fidelity.HF_VALIDATION) else c.lf_measurement
        return move + c.antenna_alignment + meas

    # ---- main proposal (design section 20, stages 4-5) --------------------

    def propose(
        self,
        model: PredictiveModel,
        dataset: CalibrationDataset,
        trust_region: TrustRegion,
        budget: BudgetState,
        current_achieved: np.ndarray,
        now: float,
    ) -> Proposal:
        cfg1 = self.config.step1
        best_rec = dataset.best_validated()
        j_best = best_rec.J if best_rec else -np.inf
        hf_noise = model.hf_noise_sd

        # Drift-aware gate (section 17): re-baseline before proposing new
        # moves if the model state is stale.
        staleness = model.staleness_hours(now, dataset)
        drift_scale = abs(model.step5.drift_rate) * max(staleness, 0.0)
        if budget.can_afford_hf() and (
            staleness > cfg1.rebaseline_after_hours or drift_scale > 2.0 * hf_noise
        ):
            # Re-command the recorded *command*, not the achieved readback:
            # readback noise/hysteresis can sit marginally outside the hard
            # travel box even though the command never did.
            u_inc = (
                best_rec.u_B_cmd
                if best_rec is not None and best_rec.u_B_cmd is not None
                else np.zeros(self.control_map.dim)
            )
            return Proposal(
                action=ActionType.REBASELINE,
                fidelity=Fidelity.HF,
                u_B=u_inc,
                expected_cost=self.expected_cost(u_inc, Fidelity.HF, current_achieved),
                trust_region_size=trust_region.size,
                reason=(
                    f"stale model state ({staleness:.1f} h since last baseline/incumbent HF; "
                    f"drift scale {drift_scale:.3g} vs noise {hf_noise:.3g})"
                ),
            )

        # Candidate pool inside T_t âˆ© U_hard.
        cands = self._candidates(trust_region, current_achieved)
        if len(cands) == 0:
            return self._fallback(model, dataset, trust_region, budget, current_achieved, now,
                                  why="no hard-feasible candidates in trust region")

        t_future = now + self.config.cost.hf_measurement
        pred = model.predict(cands, t_future=t_future)

        # Refine the best-mean pool candidate into a physics-informed
        # exploit candidate and merge it into the pool.
        u_exploit = self._refine_exploit(
            model, cands[int(np.argmax(pred.latent_mean))], trust_region, current_achieved, t_future
        )
        if u_exploit is not None:
            pred_e = model.predict(u_exploit[None, :], t_future=t_future)
            cands = np.vstack([cands, u_exploit[None, :]])
            pred.latent_mean = np.concatenate([pred.latent_mean, pred_e.latent_mean])
            pred.latent_sd = np.concatenate([pred.latent_sd, pred_e.latent_sd])
            pred.obs_sd = np.concatenate([pred.obs_sd, pred_e.obs_sd])
            pred.extrapolation = pred.extrapolation + pred_e.extrapolation

        ei = _expected_improvement(pred.latent_mean, pred.latent_sd, j_best)
        info = pred.latent_sd                      # information value proxy
        x_norm = np.stack([self.control_map.to_normalized(u) for u in cands])
        p_soft = self.soft.probability_feasible(x_norm)
        costs = np.array([self.expected_cost(u, Fidelity.HF, current_achieved) for u in cands])

        # Identification-first rule (roadmap Phase 1.2; a minimal version of
        # the Phase-3.4 design): while the physics LF channel has fewer than
        # `min_lf_identification` measurements, spend cheap reflectivity
        # probes before new HF candidates — they constrain theta at ~1/10
        # of the HF cost.
        if self.config.step5.lf_channel == "physics" and budget.can_afford_lf():
            n_lf_phys = sum(
                1
                for r in dataset.lf_records()
                if r.observable_id == "reflectivity"
            )
            if n_lf_phys < cfg1.min_lf_identification:
                i_info = int(np.argmax(info * p_soft / costs))
                return Proposal(
                    action=ActionType.LF_PROBE,
                    fidelity=Fidelity.LF_PROXY,
                    u_B=cands[i_info],
                    predicted_mean=float(pred.latent_mean[i_info]),
                    predicted_sd=float(pred.latent_sd[i_info]),
                    expected_cost=self.expected_cost(cands[i_info], Fidelity.LF_PROXY, current_achieved),
                    soft_feasibility=float(p_soft[i_info]),
                    trust_region_size=trust_region.size,
                    reason=(
                        f"identification: {n_lf_phys} < {cfg1.min_lf_identification} "
                        "physics-channel LF measurements; probing reflectivity at the "
                        "most informative feasible candidate"
                    ),
                    fallback="lf_identification",
                )

        utility = ei + cfg1.lambda_info * info
        acq = utility * p_soft / costs
        acq[p_soft < cfg1.soft_feasibility_threshold] = -np.inf

        order = np.argsort(acq)[::-1]
        i_best = int(order[0])

        # Final gates (stage 4): improvement must be resolvable and budget
        # must allow a HF measurement.
        ei_best = float(ei[i_best])
        if budget.can_afford_hf() and np.isfinite(acq[i_best]) and ei_best > cfg1.ei_noise_factor * hf_noise:
            u_sel = cands[i_best]
            return Proposal(
                action=ActionType.NEW_CANDIDATE,
                fidelity=Fidelity.HF,
                u_B=u_sel,
                predicted_mean=float(pred.latent_mean[i_best]),
                predicted_sd=float(pred.latent_sd[i_best]),
                expected_improvement=ei_best,
                acquisition_value=float(acq[i_best]),
                expected_cost=float(costs[i_best]),
                soft_feasibility=float(p_soft[i_best]),
                trust_region_size=trust_region.size,
                reason=(
                    f"EI {ei_best:.3g} > {cfg1.ei_noise_factor} x HF noise {hf_noise:.3g}; "
                    f"regime: {pred.extrapolation[i_best]}"
                ),
            )

        # LF probe (section 13): information is cheap and the LF channel is
        # informative — always with the physics channel (measurements
        # constrain theta directly); with the affine fallback only while
        # the link is validated or still needs validation data.
        link = model.step5.lf_link
        lf_cfg = self.config.step5.lf_channel
        lf_informative = lf_cfg == "physics" or (
            lf_cfg == "affine" and (link.validated or link.n_points < 3)
        )
        if budget.can_afford_lf() and lf_informative:
            i_info = int(np.argmax(info * p_soft / costs))
            u_sel = cands[i_info]
            return Proposal(
                action=ActionType.LF_PROBE,
                fidelity=Fidelity.LF_PROXY,
                u_B=u_sel,
                predicted_mean=float(pred.latent_mean[i_info]),
                predicted_sd=float(pred.latent_sd[i_info]),
                expected_improvement=float(ei[i_info]),
                expected_cost=self.expected_cost(u_sel, Fidelity.LF_PROXY, current_achieved),
                soft_feasibility=float(p_soft[i_info]),
                trust_region_size=trust_region.size,
                reason=(
                    "expected HF improvement below noise threshold; gathering cheap "
                    "LF information at the most uncertain feasible candidate"
                ),
            )

        return self._fallback(model, dataset, trust_region, budget, current_achieved, now,
                              why="no candidate passes the EI-vs-noise gate")

    # ---- fallback ladder (design section 22) ------------------------------

    def _fallback(
        self,
        model: PredictiveModel,
        dataset: CalibrationDataset,
        trust_region: TrustRegion,
        budget: BudgetState,
        current_achieved: np.ndarray,
        now: float,
        why: str,
    ) -> Proposal:
        cfg1 = self.config.step1
        best_rec = dataset.best_validated()

        # 22.1 Repeat the incumbent when its uncertainty dominates.
        if best_rec is not None and budget.can_afford_hf():
            u_inc = best_rec.u_B_cmd if best_rec.u_B_cmd is not None else best_rec.u_B_achieved
            pred = model.predict(best_rec.u_B_achieved[None, :], t_future=now)
            if float(pred.latent_sd[0]) > cfg1.incumbent_sd_factor * model.hf_noise_sd:
                return Proposal(
                    action=ActionType.REPLICATE_INCUMBENT,
                    fidelity=Fidelity.HF,
                    u_B=u_inc,
                    predicted_sd=float(pred.latent_sd[0]),
                    expected_cost=self.expected_cost(u_inc, Fidelity.HF, current_achieved),
                    trust_region_size=trust_region.size,
                    reason=f"{why}; incumbent posterior sd {float(pred.latent_sd[0]):.3g} "
                    f"> {cfg1.incumbent_sd_factor} x HF noise",
                    fallback="replicate_incumbent",
                )

        # 22.5 Stop: nothing meaningful remains.
        return Proposal(
            action=ActionType.STOP,
            fidelity=None,
            u_B=None,
            trust_region_size=trust_region.size,
            reason=f"{why}; no fallback action has positive expected value",
            fallback="stop",
        )

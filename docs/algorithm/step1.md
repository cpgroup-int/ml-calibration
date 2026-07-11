# Step 1 — Propose the next correction and measurement action

**Module:** {mod}`madmax_calibration.steps.step1_propose` ·
**Design:** {doc}`../design/madmax_step1_booster_proposal_technical_design`

Step 1 is the outer-loop planner. Its output is never a measurement — it
is a structured {class}`~madmax_calibration.records.Proposal`: a booster
correction $u_B$, a measurement action/fidelity $\ell$, and the metadata
that makes the decision auditable. The implementation is the
**recommended** variant of the design note (§19.2): trust-region
constrained, hard-feasibility filtered, noise-, cost- and budget-aware,
with replication / re-baseline / LF-probe / stop fallbacks.

## Decision flow

```text
propose(model, dataset, trust_region, budget, current, now)
│
├─ 1. Drift gate: stale model or drift_rate × staleness > 2 × HF noise?
│        └─► REBASELINE (HF at incumbent/baseline)
│
├─ 2. Candidate pool: Sobol in  T_t ∩ U_hard  (max-step filtered)
│        └─ empty? ─► fallback ladder
│
├─ 3. Full posterior prediction on the pool (Step-6 model)
│     + physics-informed exploit candidate: Nelder-Mead maximization of
│       the plug-in predictive mean, merged into the pool
│
├─ 4. Acquisition:  A(u) = [EI(u) + λ_info · σ_lat(u)] · P_soft(u) / C(u, HF)
│       with a hard veto for P_soft < threshold
│
├─ 5. Gates:  budget allows HF?   EI_best > ei_noise_factor × σ_J,HF ?
│        ├─ yes ─► NEW_CANDIDATE (HF) with full prediction metadata
│        └─ no  ─► LF probe affordable & link validated/unproven?
│                    ├─ yes ─► LF_PROBE at the most informative candidate
│                    └─ no  ─► fallback ladder
│
└─ fallback ladder:  incumbent posterior sd > factor × HF noise?
                        ├─ yes ─► REPLICATE_INCUMBENT (HF)
                        └─ no  ─► STOP recommendation
```

## The acquisition in detail

- **Improvement value** (design §10): noisy expected improvement on the
  *latent* posterior, $\mathrm{EI}(u) = \mathbb E[\max(0, J(u) -
  J_{\mathrm{best}})]$ against the best *validated* HF measurement.
- **Information value** (§11): the latent posterior sd as a tractable
  information proxy, weighted by `lambda_info`. It matters most early,
  when the $\theta$-posterior dominates the uncertainty.
- **Cost awareness** (§12): division by the expected action cost
  $C = $ move + alignment + measurement (hours), so a marginally better
  distant candidate loses to an equally good nearby one.
- **Soft constraints** (§15): the learned success probability both
  multiplies the utility and vetoes below
  `soft_feasibility_threshold`.
- **The exploit candidate**: Sobol exploration alone rarely lands exactly
  on the correcting configuration; the planner therefore refines the
  best-predicted-mean pool point by local optimization of the cheap
  plug-in mean ({meth}`~madmax_calibration.steps.step6_predictive.PredictiveModel.predict_mean_map`)
  inside $\mathcal T_t \cap \mathcal U_{\mathrm{hard}}$ — this is how the
  calibrated simulator's knowledge ("cancel the inferred stack offset")
  enters the proposal directly.

## Noise- and drift-aware behaviour

Two gates keep the optimizer from chasing noise (design §16–17):

- a new HF measurement requires
  $\mathrm{EI} > \texttt{ei\_noise\_factor} \times \sigma_{J,\mathrm{HF}}$;
- a re-baseline is forced when the model state is stale
  (`rebaseline_after_hours`) or the inferred drift over the staleness
  window exceeds twice the measurement noise.

When the EI gate fails, the budget decides between *cheap information*
(LF probe at the most uncertain feasible candidate — also how the LF↔HF
link gets its calibration data), *replication* (incumbent posterior sd too
large), or a *stop recommendation* that Step 7 will act on.

## Trust region

{class}`~madmax_calibration.steps.step1_propose.TrustRegion` implements
the §14 policy: centred on the best validated achieved configuration,
expanded ×1.6 after 2 consecutive validated improvements, shrunk ×0.5
after 3 consecutive failures, clamped to `[min_size, max_size]` and to
the unit box. The loop recenters it whenever a new validated best appears.

## Budget state

{class}`~madmax_calibration.steps.step1_propose.BudgetState` carries the
remaining HF / LF / move / hour budgets (design §4.9); both the HF and LF
paths check affordability before proposing.

## Configuration

All knobs in {class}`~madmax_calibration.config.Step1Config` and
{class}`~madmax_calibration.config.TrustRegionConfig` — see
{doc}`../user_guide/configuration` for the full table and tuning guidance.

## Tests

`tests/test_step1_propose.py` implements the design's §25 checklist:

| Design check | Test |
|---|---|
| 1. Constraint filtering | `test_proposals_are_always_hard_feasible` |
| 2. Noise response | `test_noise_response_no_hf_when_improvement_unresolvable` |
| 3. Budget response | `test_budget_response_no_hf_without_budget` |
| 4. Trust-region behaviour | `test_trust_region_expands_and_shrinks` |
| 6. Drift response | `test_drift_response_rebaseline_when_stale` |
| soft-constraint handling | `test_soft_constraint_veto` |
| fallback termination | `test_stop_when_nothing_meaningful_remains` |

(Check 5, control-basis consistency, is a physics-level test:
`test_control_basis_cancels_correctable_errors`.)

# Architecture

## The seven-step loop

The package is a direct implementation of the seven-step structure locked
in the parent proposal ({doc}`../design/madmax_closed_loop_calibration_proposal`,
section 11). One module per step, one orchestrator:

```text
                       ┌─────────────────────────────────────────────┐
                       │            CalibrationLoop (loop.py)         │
                       └─────────────────────────────────────────────┘
                                          │
   Step 0 (once)   steps/step0_initialize.py     baseline + noise + limits
                                          │
   ┌──────────────────────── outer iteration t ─────────────────────────┐
   │ Step 5  steps/step5_inference.py   p(θ, r, σ, drift | D)           │
   │ Step 6  steps/step6_predictive.py  p(J_HF(u) | D)  (+ diagnostics) │
   │ Step 1  steps/step1_propose.py     (u_B, ℓ) or fallback action     │
   │ Step 7  steps/step7_stopping.py    stop? ──────────────────────────┼──►
   │ Step 2  steps/step2_set_geometry.py  move, record achieved geometry│
   │ Step 3  steps/step3_antenna.py       align antenna, record achieved│
   │ Step 4  steps/step4_measure.py       measure, QC, uncertainty, cost│
   └────────────────────────────────────────────────────────────────────┘
                                          │
              final HF validation + feasibility report (loop.py)
```

Note the ordering inside an iteration: the *model updates* (Steps 5 and 6)
run at the **top** of each iteration so that Step 1 always decides on the
freshest posterior — this matches the information flow of the parent
proposal (section 13), where measurement → joint update → predictive
update → next proposal closes the loop.

## Information flow

$$
\text{Step 4: measurement}
\;\longrightarrow\;
\text{Step 5: } p(\theta, r, \sigma_J, \text{drift} \mid D)
\;\longrightarrow\;
\text{Step 6: } p(J_{\mathrm{HF}}(u) \mid D)
\;\longrightarrow\;
\text{Step 1: next } (u_B, \ell)
$$

Three data objects carry this flow (all in
{mod}`madmax_calibration.records`):

{class}`~madmax_calibration.records.MeasurementRecord`
: The standardized Step-4 output (Step 4 design, section 13): commanded
  **and** achieved geometry, fidelity label, processed observable,
  scalar objective + uncertainty (HF only), quality flags, replication
  tags, cost. Lower-fidelity records never carry a `J` value — the
  "J_HF not measured in this iteration" rule is enforced by type.

{class}`~madmax_calibration.records.CalibrationDataset`
: The accumulated $D_{1:t}$ with query helpers (best validated HF result,
  baseline repeats, empirical repeat scatter, cost totals). Records
  excluded from inference are kept with a reason, never deleted.

{class}`~madmax_calibration.records.Proposal`
: The structured Step-1 output package (Step 1 design, section 21):
  action type, fidelity, physical correction, prediction metadata,
  expected cost, feasibility certificates and a human-readable reason —
  the audit trail that makes the optimizer inspectable.

## Module map

| Module | Implements | Design doc anchor |
|---|---|---|
| {mod}`madmax_calibration.config` | every tunable default | Step 1 §26 open choices |
| {mod}`madmax_calibration.records` | records, proposals, dataset | Step 4 §13, Step 1 §21 |
| {mod}`madmax_calibration.control` | basis $B$, $u_B = T(x_B)$, control→geometry | proposal §4, Step 1 §5 |
| {mod}`madmax_calibration.simulator` | $\beta^2_{\mathrm{sim}}(\nu; q, \theta)$ stand-in | proposal §2.3 |
| {mod}`madmax_calibration.objectives` | scalar $J$ + MC uncertainty | proposal §6, Step 4 §9.5 |
| {mod}`madmax_calibration.gp` | exact GP (RBF, fixed noise) | — |
| {mod}`madmax_calibration.constraints` | hard filtering + learned soft constraints | proposal §9 |
| {mod}`madmax_calibration.hardware` | hardware interface + simulated detector | proposal §2.5, §10 |
| {mod}`madmax_calibration.steps` | Steps 0–7 | one design note each |
| {mod}`madmax_calibration.loop` | orchestrator + feasibility report | proposal §11–12 |

## One iteration in sequence

1. `run_step5(dataset, …)` → {class}`~madmax_calibration.steps.step5_inference.Step5Result`
   — joint MAP + Laplace over $(\theta, \text{drift}, \text{noise
   inflation})$ with the discrepancy GP marginalized inside the objective;
   correctable/diagnostic classification; identifiability flags; LF link.
2. `run_step6(step5, …)` → {class}`~madmax_calibration.steps.step6_predictive.PredictiveModel`
   — sample-based posterior predictive with latent/observation
   distinction, extrapolation and staleness diagnostics, validated
   against the training data before hand-off.
3. `proposer.propose(model, …)` → {class}`~madmax_calibration.records.Proposal`
   — Sobol pool in $\mathcal{T}_t \cap \mathcal{U}_{\mathrm{hard}}$, a
   refined exploit candidate, acquisition gates, fallback ladder.
4. `run_step7(…)` — stop checks **before** spending hardware time.
5. `run_step2(…)` — move (auto-split into safe sub-steps), verify readback.
6. `run_step3(…)` — incumbent check → local scan + fit → GP-BO fallback.
7. `run_step4(…)` — measure, reduce, estimate uncertainty, flag, cost.
8. Trust-region update, soft-constraint update, history entry, budget
   re-check.

## Design boundaries respected in code

The design notes draw hard responsibility boundaries, and the module
signatures mirror them:

- Step 4 never proposes, never re-aligns, never fits models — it returns a
  record.
- Step 5 never predicts future candidates — that is Step 6's job (the
  `Step5Result` ↔ `PredictiveModel` split).
- Step 6 never chooses an acquisition — it hands Step 1 distributions,
  samples and diagnostics.
- Step 3 never touches booster variables; Step 1 never touches antenna
  variables (the nested objective $F(u_B) = \max_{u_A} J$).
- Hard constraints live outside every learned model
  ({class}`~madmax_calibration.constraints.HardConstraints` is consulted
  before candidates are scored *and* before hardware executes).

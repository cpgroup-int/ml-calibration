# Step 3 — Align the antenna

**Module:** {mod}`madmax_calibration.steps.step3_antenna` ·
**Design:** {doc}`../design/madmax_step3_antenna_alignment_technical_design`

For the booster geometry fixed by Step 2, Step 3 solves the
two-dimensional inner alignment problem

$$
u_A^\star = \arg\max_{u_A \in\, \mathcal U_{A,\mathrm{hard}}}
A_\ell(u_A \mid \tilde u_B, W)
$$

using the cheap coupling proxy $A_\ell$ provided by the hardware
(`measure_alignment_proxy`). It never touches booster variables — the
nested-objective separation $F(u_B) = \max_{u_A} J$ is preserved.

## The hybrid strategy

The implementation follows the design's recommended structure (§7):

```text
incumbent validation → local plus-pattern scan → quadratic fit →
confirmation → 2D GP-BO fallback → return best achieved position
```

**Stage 1 — warm start.** The previous best antenna position (or the safe
origin) is the starting point; across iterations the loop passes the last
alignment's position and score so small booster changes don't trigger full
scans.

**Stage 2 — incumbent validation (§9).** One measurement at the start
position; if it is within $\kappa \sigma$ of the expected score, the
alignment is *reused* (`reused_incumbent`) and Step 3 costs a single
proxy measurement.

**Stage 3 — local scan + quadratic fit (§10).** A 9-point plus/diagonal
pattern at `initial_scan_step`, then a least-squares quadratic surface.
The fitted optimum is accepted only if the Hessian is negative-definite,
the optimum lies inside the hard domain and within reach of the scan, and
a confirmation measurement agrees with the fit within noise.

**Stage 4 — GP-BO fallback (§11).** If the fit is unreliable (distorted,
multi-modal or noisy surface), a 2D Gaussian process is fitted to all
collected data and a noise-aware UCB acquisition selects further
measurements until the predicted improvement drops below the noise floor
or the local budget `max_evaluations` is exhausted.

The hardware is finally moved to the best *achieved* position, which is
what Step 4 records.

## Output

{class}`~madmax_calibration.steps.step3_antenna.Step3Result`: commanded
and achieved position, score ± sigma, the method/quality flag
(`reused_incumbent`, `local_fit_confirmed`, `gp_bo_confirmed`,
`budget_limited`, …), the evaluation count and the **full local dataset**
— per the design (§16), alignment data are forwarded rather than thrown
away, and a `budget_limited`/`noise_limited` flag propagates into the
Step-4 record as `antenna_alignment_suspect`.

## Hard constraints

The antenna travel box ($|x|, |y| \leq$ `travel_limit`) is enforced by
clipping every candidate before it is commanded — candidates outside the
domain are never proposed, measured, or treated as learnable failures
(§12.1).

## Configuration

{class}`~madmax_calibration.config.AntennaConfig`: `travel_limit` ⚠,
`initial_scan_step` (match to the expected beam width),
`max_evaluations` (the Step-3 budget $B_A$), `kappa` (validation and
confirmation tolerance).

## Tests

`tests/test_step3_antenna.py` implements the design §20 checklist:

| Design check | Test |
|---|---|
| 20.1 synthetic Gaussian beam | `test_recovers_gaussian_beam_center` |
| 20.2 distorted/multimodal beam | `test_distorted_surface_falls_back_to_gp_bo` |
| 20.5 hysteresis/readback | `test_achieved_positions_recorded` |
| 20.7 budget | `test_respects_measurement_budget` |
| §9 incumbent reuse | `test_incumbent_reused_when_still_good` |

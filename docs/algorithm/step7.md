# Step 7 — Stop or repeat

**Module:** {mod}`madmax_calibration.steps.step7_stopping` ·
**Design:** parent proposal, Step 7

Step 7 decides whether the calibration is finished. It is evaluated
**twice per iteration** — before executing a proposal (so a stop
recommendation or an exhausted budget never costs hardware time) and after
(so a budget that was consumed by the just-executed action ends the loop
immediately).

## Stopping rules

{func}`~madmax_calibration.steps.step7_stopping.run_step7` stops when any
of the following holds:

| Rule | Source |
|---|---|
| Step 1 returned a `STOP` recommendation | fallback ladder exhausted (Step 1 design §22.5) |
| time budget exhausted | `budget.max_total_hours` |
| HF measurement budget exhausted | `budget.max_hf_measurements` |
| booster move budget exhausted | `budget.max_booster_moves` |
| optional absolute target reached | `step7.target_objective` |
| expected improvement below `improvement_noise_factor` × HF noise for `patience` consecutive iterations | the resolvability condition (parent proposal §7) |

The consecutive-unresolvable counter is maintained by the loop: it resets
on a validated improvement (or a new-candidate proposal whose EI clears
the resolvability threshold) and increments otherwise — LF probes,
replications and re-baselines all count toward patience, so the loop
cannot idle forever on cheap actions.

## What happens after stopping

Stopping is not the end of the run — the loop then performs the **final
high-fidelity validation** (Step 4 design §5.3): the best configuration is
re-measured with the conservative `HF_validation` action, the result is
combined with the incumbent measurement, and the feasibility report is
assembled (baseline vs best, improvement in units of the combined noise,
measurement counts, cost, stop reason, final detector-state summary with
classification and identifiability labels). See
{class}`~madmax_calibration.loop.CalibrationResult`.

## Configuration

{class}`~madmax_calibration.config.Step7Config`:
`improvement_noise_factor` (default 0.5), `patience` (default 3),
`target_objective` (default off).

## Tests

Stopping behaviour is covered end-to-end: `test_loop_respects_budget`
(budget rules), the smoke/example runs (noise-patience rule fires and
stops the loop on its own), and `test_stop_when_nothing_meaningful_remains`
(the Step-1 stop recommendation that Step 7 acts on).

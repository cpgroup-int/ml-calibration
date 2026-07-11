# Step 0 — Initialize and validate the baseline

**Module:** {mod}`madmax_calibration.steps.step0_initialize` ·
**Design:** parent proposal, Step 0

Step 0 runs once, before the loop. It gives the optimizer a *measured*
reference point and prevents the calibration from starting without a
noise/budget sanity check.

## What it does

1. **Assert the nominal configuration is feasible** — a hard failure here
   means the configured limits are inconsistent, not a hardware question.
2. **Move to $u_B = 0$** (Step 2 machinery, achieved geometry recorded).
3. **Align the antenna** (full Step-3 procedure — there is no incumbent to
   reuse yet).
4. **Measure the baseline boost factor with replication** —
   `n_baseline_replicates` high-fidelity measurements at the same
   commanded state, tagged as one replicate group and as `baseline`
   records.
5. **Estimate the baseline objective and its uncertainty:**

   $$
   J_0 = \overline{J^{(k)}}, \qquad
   \sigma_{J,0} = \max\bigl(\text{stated}, \text{empirical}\bigr)
   $$

   where *stated* is the RMS of the per-measurement propagated
   uncertainties and *empirical* is the scatter of the replicates. Taking
   the max protects against optimistic instrument-level uncertainty
   models; the comparison itself is reported as `noise_consistent`.
6. **Feasibility screen** — `resolvable` is false when
   $\sigma_{J,0} \geq 0.5\,|J_0|$: with noise that large no calibration
   improvement could be demonstrated, which is the core feasibility
   condition of the parent proposal (§7).

## Output

{class}`~madmax_calibration.steps.step0_initialize.Step0Result`:
$J_0$, $\sigma_{J,0}$, the achieved baseline geometry (used as the initial
trust-region anchor and "current achieved" state), replicate count and the
two consistency flags. The baseline records themselves live in the shared
{class}`~madmax_calibration.records.CalibrationDataset` — they are the
first three points the Step-5 inference will fit, and the reference
against which drift is later detected.

## Configuration

- `CalibrationConfig.n_baseline_replicates` — 3 by default; 2 is the
  minimum for an empirical scatter estimate.
- Everything measured here uses the same Step-2/3/4 machinery and hence
  the same tolerances and cost model as the loop proper.

## Tests

Step 0 is exercised by every end-to-end test; its noise logic is
additionally covered through the dataset helpers
(`CalibrationDataset.empirical_repeat_sd`) used in
`tests/test_loop_end_to_end.py`.

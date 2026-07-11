# Step 2 — Set the booster geometry

**Module:** {mod}`madmax_calibration.steps.step2_set_geometry` ·
**Design:** parent proposal, Step 2

Step 2 executes the geometry part of a proposal: it moves the detector to
the proposed booster state — disk-correction modes, global z, reflecting
mirror and focusing mirror are commanded as **one complete booster
geometry**, not as disconnected sub-steps — and records what was actually
achieved.

## What it does

{func}`~madmax_calibration.steps.step2_set_geometry.run_step2`:

1. command `move_booster(u_B_cmd)`;
2. compare the achieved readback against the command, per coordinate,
   against `tolerance` (default 20 µm);
3. retry once if out of tolerance;
4. return {class}`~madmax_calibration.steps.step2_set_geometry.Step2Result`
   with `u_B_cmd`, `u_B_achieved`, `within_tolerance` and — when the
   retry did not help — the `geometry_out_of_tolerance` quality flag.

The flag matters downstream: Step 4 treats it as a failed pre-measurement
state check and returns an invalid record instead of measuring — the
dataset never pretends a commanded geometry was achieved when readback
says otherwise (Step 4 design §10.1).

## Safe-step splitting

The maximum safe single move (`max_step_normalized`) is a *hard*
constraint, but re-baseline and validation actions may legitimately target
a configuration far from the current one. The loop therefore wraps Step 2
in {meth}`CalibrationLoop._move_with_step_limit
<madmax_calibration.loop.CalibrationLoop>`: a commanded move larger than
the safe step is split into a sequence of intermediate feasible moves,
each individually within the limit and each with its own readback.

## Achieved geometry is the currency

Everything downstream uses `u_B_achieved`, not `u_B_cmd`:

- Step 4 stores both in the measurement record;
- Step 5 fits the model on achieved positions (hysteresis/creep would
  otherwise smear the learned response map — parent proposal §2.5);
- the trust region is recentred on achieved positions;
- the max-step constraint for the *next* proposal is evaluated from the
  achieved state.

The mock hardware injects direction-dependent hysteresis (10 µm),
repeatability noise (5 µm) and readback noise (2 µm) precisely so tests
exercise this distinction (`test_hf_record_is_complete` asserts commanded
≠ achieved).

## Tests

Step 2 is covered through the end-to-end suite
(`test_achieved_geometry_recorded_everywhere`,
`test_loop_never_violates_hard_constraints`) and the Step-4 pre-check test
(`test_failed_pre_check_returns_invalid_record_without_fabricating_J`).

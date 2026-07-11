# Safety model

The single most important rule of the design (parent proposal §9, Step 1
design §6):

> Damage-relevant constraints are not black-box constraints to be learned
> by failure. They are exact or conservative feasibility filters.

The implementation splits "hardware constraints" into two entirely
separate mechanisms with different code paths.

## Hard constraints — exact, never learned

{class}`~madmax_calibration.constraints.HardConstraints` implements the
exact feasibility domain $\mathcal U_{\mathrm{hard}}$. A candidate $u_B$
passes only if **all** of the following hold:

1. **Actuator travel box** — every component of $u_B$ within its
   configured limit ({class}`~madmax_calibration.config.ControlConfig`).
2. **Collision avoidance** — the resulting physical geometry keeps every
   gap above `min_gap` (+ an optional conservative `gap_safety_margin`;
   if a damage-relevant limit is uncertain, the domain is *shrunk*, not
   explored).
3. **Maximum safe step** — the move from the *current achieved* geometry
   stays below `max_step_normalized` per coordinate.

Where the filter is applied — deliberately more than once:

- inside Step-1 candidate generation (infeasible Sobol candidates are
  dropped before scoring);
- on the physics-informed exploit candidate after refinement;
- in Step 0 (the nominal configuration itself is asserted feasible);
- at execution time: {meth}`CalibrationLoop._move_with_step_limit
  <madmax_calibration.loop.CalibrationLoop>` splits any commanded move
  larger than the safe step into a sequence of feasible sub-moves, so even
  a re-baseline jump from the far side of the domain respects the
  per-move limit.

The end-to-end test `test_loop_never_violates_hard_constraints` asserts
that **no record in the entire dataset** was ever commanded outside
$\mathcal U_{\mathrm{hard}}$.

## Soft constraints — learned, non-damaging only

{class}`~madmax_calibration.constraints.SoftConstraintModel` learns a
probability of *measurement success* $P_{\text{soft-safe}}(u)$ from
observed non-damaging failures (failed gradient-method determinations,
bad-coupling regions, parasitic-mode regions — in the mock detector,
measurements fail when the coupling is very poor). Mechanics:

- every executed measurement contributes a success/failure observation at
  its achieved normalized position;
- a GP smooths the 0/1 outcomes around a prior success probability of 0.9;
- Step 1 multiplies the acquisition utility by
  $P_{\text{soft-safe}}(u)$ **and** vetoes candidates below
  `soft_feasibility_threshold` (both mechanisms of Step 1 design §15).

Nothing that could damage hardware is ever routed through this model — a
learned constraint can only make the optimizer *avoid wasting
measurements*, never grant access to a region the hard filter forbids.

## Related safeguards

- **Geometry verification** (Step 2): after every move the achieved
  readback is compared against the command; out-of-tolerance states are
  flagged `geometry_out_of_tolerance` and the measurement is aborted
  rather than silently attributed to the commanded point.
- **Pre-measurement state check** (Step 4 §6.1): a candidate already
  flagged invalid is never measured; a failed check returns an invalid
  record with no fabricated objective value.
- **Budget gates** (Step 1/Step 7): the loop cannot exceed the configured
  HF/LF/move/time budgets — checked both before proposing and after
  executing.
- **Noise gates**: improvements smaller than the measurement noise never
  trigger exploitation (see {doc}`../algorithm/step1`), which also
  protects against drift-chasing behaviour that would rack up pointless
  hardware moves.

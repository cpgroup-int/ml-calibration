# Hardware integration guide

The calibration loop never touches physics or motors directly — it goes
through two interfaces. To run against the real detector you implement
one class and substitute one object:

1. implement {class}`~madmax_calibration.hardware.HardwareInterface`
   around the real detector control + the existing gradient-method
   boost-factor routine;
2. substitute the real MADMAX simulation behind the
   {class}`~madmax_calibration.simulator.BoostSimulator` interface.

Nothing else in the package knows about the mock.

## The `HardwareInterface` contract

```python
class MyMadmaxHardware(HardwareInterface):
    ...
```

| Method | Semantics | Units / conventions |
|---|---|---|
| `move_booster(u_B_cmd) -> u_B_achieved` | Command the complete booster correction (disk modes, global z, mirror, focus) **relative to the nominal configuration** and return the best available achieved readback. | metres; same ordering as `ControlMap` |
| `move_antenna(u_A_cmd) -> u_A_achieved` | Command the antenna (x, y) and return achieved readback. | metres |
| `booster_readback()` / `antenna_readback()` | Fresh readback without moving — used for post-measurement drift checks. | metres |
| `measure_alignment_proxy() -> (value, sigma)` | The cheap coupling observable used by Step-3 alignment at the *current* state. Higher = better coupling. Must be fast — it is called ~10–20× per alignment. | arbitrary but consistent scale |
| `measure_boost_factor() -> (curve, sigma, success)` | The existing gradient-method boost-factor determination at the current state: $\widehat{\beta^2}(\nu)$ on the configured frequency grid, a per-bin 1σ estimate, and a success flag. Return `success=False` rather than fabricated data on failure. | curve on `SimulatorConfig.frequency_grid()` |
| `measure_lf_proxy() -> (refl, refl_sigma, gd, gd_sigma, success)` | The lower-fidelity RF measurement: power reflectivity $\lvert\Gamma\rvert^2(\nu)$ and group delay $\tau_g(\nu)$ [s] on the configured frequency grid, with per-bin 1σ estimates. A *physics observable* the simulator can predict — this is what lets cheap RF data calibrate the detector state (roadmap Phase 1.2). | curves on `SimulatorConfig.frequency_grid()` |
| `advance_time(hours)` / `now` | The loop's clock for drift modelling and budget accounting. On real hardware, back `now` with wall-clock time and make `advance_time` a no-op. | hours |

### Contract details that matter

- **Achieved ≠ commanded.** The loop learns from readback (hysteresis,
  creep, finite repeatability are expected); always return the truest
  available position, and never echo the command back as "achieved" if a
  real readback exists.
- **Failures are data.** A failed measurement must return
  `success=False`; Step 4 records it as invalid (no objective value is
  fabricated) and the learned soft-constraint model steers the optimizer
  away from the region.
- **Relative corrections.** `u_B = 0` must mean "the nominal
  configuration $q_0(W)$". The mapping from mode amplitudes to individual
  actuator setpoints is your responsibility; keep it consistent with the
  basis in {func}`~madmax_calibration.control.disk_mode_basis` or replace
  that basis with the real one (see below).
- **Costs.** If measurement/movement durations differ significantly from
  the defaults, update {class}`~madmax_calibration.config.CostConfig` so
  the cost-aware acquisition works with real numbers.

Use {class}`~madmax_calibration.hardware.MockHardware` as the reference
implementation — it exercises every part of the contract, including
failure injection and readback noise.

## Substituting the real simulator

{class}`~madmax_calibration.simulator.BoostSimulator` is consumed through
three methods only:

- `beta2(u_B, theta)` — boost curve on the frequency grid for a control
  correction and a detector-state hypothesis;
- `predict_J(u_B, theta, objective)` — antenna-aligned scalar objective
  (this is what Steps 1, 5 and 6 call in their inner loops);
- `coupling(...)` / `aligned_coupling(...)` — receiver-coupling model.

Wrap the real MADMAX simulation in a class with these signatures.
Performance target: `predict_J` is called ~$10^3$ times per outer
iteration (candidate pool × posterior samples + inference), so a single
evaluation should stay in the few-millisecond range — batch or cache
internally if needed.

**Detector-state parameterization.** The three-parameter
{class}`~madmax_calibration.simulator.DetectorState` (stack offset, gap
compression, log loss-scale) must be extended to whatever the real
simulation exposes. When you change it:

1. update `DetectorState.NAMES` / `CORRECTABLE` — the
   correctable/diagnostic classification is derived from it;
2. update the Step-5 priors in
   {class}`~madmax_calibration.config.Step5Config` (informative priors
   are load-bearing, see {doc}`statistics`);
3. re-check that every *correctable* parameter really is spanned by the
   control basis (the control-basis consistency rule, parent proposal
   §5) — `test_control_basis_cancels_correctable_errors` shows how to
   test this.

## Replacing the control basis

The disk-correction modes in
{func}`~madmax_calibration.control.disk_mode_basis` (uniform / linear /
quadratic gap profiles) are structural placeholders. To use the real
actuator layout, replace the basis matrix — the rest of the stack
(normalization, hard constraints, acquisition) works with any
$n_{\mathrm{disks}} \times n_{\mathrm{modes}}$ basis. Keep the columns
scaled so a unit amplitude produces order-unity gap changes, and update
the travel limits in {class}`~madmax_calibration.config.ControlConfig`.

## Pre-hardware checklist

Before connecting the loop to the real detector:

1. **Fix the ⚠ items** in {doc}`../design/DESIGN_DECISIONS` with the
   experimental team (hard limits, objective, fidelities, budgets,
   priors).
2. **Run the validation suite against your hardware class in
   simulation-backed mode** if possible — the tests in
   `tests/test_step4_measure.py` and `tests/test_loop_end_to_end.py` are
   written against the interface, not the mock internals.
3. **Baseline repeatability** (Step 4 design §21.1): measure the nominal
   configuration several times; if $\sigma_{J,0}$ is comparable to the
   expected calibration gains, the loop will (correctly) refuse to run —
   fix the measurement first.
4. **Geometry readback test** (§21.4): command known positions, verify
   commanded and achieved are both recorded and differ by the expected
   actuator repeatability.
5. **Dry-run Step 3 alone** on the real antenna with a generous
   `travel_limit` margin and confirm the proxy actually correlates with
   coupling before trusting it inside the loop (Step 3 design §5).
6. **Start with a small trust region and a small budget** —
   `trust_region.initial_size = 0.1`, a handful of HF measurements — and
   inspect `calibration_history.json` before scaling up.

# Step 4 — Measure the selected observable

**Module:** {mod}`madmax_calibration.steps.step4_measure` ·
**Design:** {doc}`../design/madmax_step4_measurement_technical_design`

Step 4 is the bridge between hardware and statistics: it executes the
measurement selected by Step 1 at the achieved detector state and returns
a quality-controlled, uncertainty-aware
{class}`~madmax_calibration.records.MeasurementRecord`. It never proposes,
never re-aligns, never fits models.

## Execution protocol

{func}`~madmax_calibration.steps.step4_measure.run_step4` follows the
design's §19 logic:

1. **Pre-measurement state check (§6.1).** A candidate arriving with a
   `geometry_out_of_tolerance` flag is *not measured*: the function
   returns an invalid record with `measurement_failed` and no objective
   value — failure is recorded, never silently corrected, and no $J$ is
   ever fabricated (§20.1).
2. **Execute the selected fidelity.**
   - `HF` / `HF_validation` → the gradient-method boost-factor oracle
     (`measure_boost_factor`), treated exactly as the design requires: an
     existing measurement routine wrapped, not reimplemented.
   - `LF_proxy` → the scalar proxy oracle (`measure_lf_proxy`).
3. **Data reduction.** For HF data: store the curve
   $\widehat{\beta^2}(\nu)$ with its per-bin uncertainty, then compute
   the scalar objective with Monte-Carlo uncertainty propagation
   ({meth}`~madmax_calibration.objectives.Objective.with_uncertainty`),
   which distinguishes a shared normalization-like component from
   independent per-bin noise (§9.4–9.5).
4. **Quality control (§10).** Signal-quality (`insufficient_snr`),
   resolvability (`objective_not_resolvable_above_noise`), and a
   post-measurement geometry re-read that flags `drift_suspected` when
   the state moved more than 50 µm during the measurement (§6.3).
5. **Cost accounting (§12).** Wall-clock hours consumed are stored on the
   record and feed the budget state and the cost-aware acquisition.

## The fidelity firewall

The record schema enforces the design's central data rule:

- HF records carry `J` and `sigma_J` (plus the curve);
- LF records carry `proxy_value`/`proxy_sigma`, **`J` stays `None`**, and
  the comment field states "J_HF not measured in this iteration" (§7.3).

Nothing downstream can accidentally treat proxy data as a validated
boost-factor objective — Step 5 consumes LF records only through the
explicit link model, and `CalibrationDataset.best_validated()` only ever
looks at HF records.

## Replication and tagging

Records carry `replicate_group` and `baseline_or_incumbent` tags set by
the caller (Step 0 replication, re-baseline and incumbent-repeat actions).
These enable the empirical-vs-stated noise cross-check
({meth}`~madmax_calibration.records.CalibrationDataset.empirical_repeat_sd`)
and the drift diagnostics in Step 5.

## Failure-mode handling

| Design failure mode (§20) | Behaviour |
|---|---|
| 20.1 measurement fails completely | invalid record, metadata preserved, no fabricated $J$ |
| 20.2 geometry differs from command | achieved geometry in the record; flag; measured state is the truth |
| 20.3 too noisy to rank | value + large σ returned; the noise gates upstream decide |
| 20.5 drift during measurement | pre/post readback compared, `drift_suspected` flag |

## Tests

`tests/test_step4_measure.py` implements the §21 checklist:

| Design check | Test |
|---|---|
| 21.2 HF pipeline | `test_hf_record_is_complete` |
| 21.3 LF pipeline | `test_lf_record_never_pretends_to_be_hf` |
| 21.6 objective consistency | `test_objective_consistency_across_repeats` |
| §20.1 no fabricated J | `test_failed_pre_check_returns_invalid_record_without_fabricating_J` |

(21.1 baseline repeatability is Step 0's job; 21.4 geometry readback is
covered by the Step-2/end-to-end tests; 21.7 the end-to-end handoff by
the Step-5/6 suites.)

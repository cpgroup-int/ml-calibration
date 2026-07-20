# Validation matrix

Each design note ends with a pre-hardware validation checklist. The test
suite implements those checklists — this page is the traceability matrix
from design requirement to test. Run everything with:

```bash
pytest            # ~2-3 minutes
```

The tests use a lightened configuration (`tests/conftest.py`: fewer
frequency bins, smaller candidate pools, prior-sensitivity check off) so
the suite stays fast while exercising identical code paths.

## Physics / simulator

| Requirement | Test |
|---|---|
| mirror-only boost is exactly 1 | `test_simulator.py::test_mirror_only_boost_is_unity` |
| transparent mode matches $\beta = 1 + 2N(1 - 1/n^2)$ analytically | `test_simulator.py::test_transparent_mode_boost_matches_analytic` |
| detector-state errors degrade the objective | `test_simulator.py::test_detector_state_errors_degrade_objective` |
| control basis spans the correctable errors (proposal §5) | `test_simulator.py::test_control_basis_cancels_correctable_errors` |
| coupling decreases off-beam / off-focus | `test_simulator.py::test_coupling_decreases_off_beam_and_off_focus` |

## Control map and constraints (proposal §9)

| Requirement | Test |
|---|---|
| normalized ↔ physical bijection | `test_control_and_constraints.py::test_normalized_physical_round_trip` |
| travel limits enforced exactly | `…::test_travel_limits_rejected` |
| minimum-gap (collision) limit enforced exactly | `…::test_min_gap_enforced` |
| max safe step from achieved geometry | `…::test_max_step_from_current_geometry` |
| soft model learns non-damaging failure regions | `…::test_soft_constraint_learns_failure_region` |

## GP and objectives

| Requirement | Test |
|---|---|
| GP recovers a smooth function within noise | `test_gp_and_objectives.py::test_gp_recovers_smooth_function` |
| prior prediction without data | `…::test_gp_prior_prediction_without_data` |
| discrepancy-amplitude prior prevents absorption (Step 5 §9.3) | `…::test_amplitude_prior_shrinks_discrepancy` |
| objective definitions | `…::test_objectives_basic_properties` |
| curve→J MC uncertainty propagation (Step 4 §9.5) | `…::test_objective_uncertainty_propagation` |

## Step 1 (design §25)

| Design check | Test |
|---|---|
| 1 constraint filtering — unsafe candidates never proposed | `test_step1_propose.py::test_proposals_are_always_hard_feasible` |
| 2 noise response — no HF when improvement < noise | `…::test_noise_response_no_hf_when_improvement_unresolvable` |
| 3 budget response | `…::test_budget_response_no_hf_without_budget` |
| 4 trust-region expand/shrink | `…::test_trust_region_expands_and_shrinks` |
| 5 control-basis consistency | `test_simulator.py::test_control_basis_cancels_correctable_errors` |
| 6 drift response — re-baseline when stale | `…::test_drift_response_rebaseline_when_stale` |
| soft-constraint veto (§15) | `…::test_soft_constraint_veto` |
| fallback ladder ends in STOP (§22) | `…::test_stop_when_nothing_meaningful_remains` |

## Step 3 (design §20)

| Design check | Test |
|---|---|
| 20.1 synthetic Gaussian beam | `test_step3_antenna.py::test_recovers_gaussian_beam_center` |
| 20.2 distorted/multimodal surface → GP-BO fallback | `…::test_distorted_surface_falls_back_to_gp_bo` |
| 20.5 hysteresis/readback | `…::test_achieved_positions_recorded` |
| 20.7 budget | `…::test_respects_measurement_budget` |
| §9 incumbent revalidation | `…::test_incumbent_reused_when_still_good` |

## Step 4 (design §21)

| Design check | Test |
|---|---|
| 21.2 HF pipeline completeness | `test_step4_measure.py::test_hf_record_is_complete` |
| 21.3 LF pipeline / fidelity firewall | `…::test_lf_record_never_pretends_to_be_hf` |
| 21.6 objective consistency | `…::test_objective_consistency_across_repeats` |
| §20.1 failed check ⇒ no fabricated J | `…::test_failed_pre_check_returns_invalid_record_without_fabricating_J` |

## Step 5 (design §20)

| Design check | Test |
|---|---|
| 20.1 synthetic recovery with honest uncertainty | `test_step5_inference.py::test_synthetic_recovery_of_correctable_parameters` |
| 20.2 confounding ⇒ weak-identifiability flags | `…::test_prior_sensitivity_flags_weak_identifiability` |
| 20.4 drift detection | `…::test_drift_detected` |
| 20.5 correctable vs diagnostic labels | `…::test_classification_labels_follow_control_basis` |
| 20.6 multi-fidelity: link learned, never pooled | `…::test_lf_link_learned_but_not_pooled` |
| minimum-data guard | `…::test_requires_minimum_hf_points` |

## Step 6 (design §27)

| Design check | Test |
|---|---|
| 27.1 no-discrepancy closure | `test_step6_predictive.py::test_no_discrepancy_closure` |
| 27.2 known discrepancy learned | `…::test_known_discrepancy_is_learned` |
| 27.3 theta uncertainty propagates | `…::test_theta_uncertainty_propagates` |
| 27.5 drift-aware prediction | `…::test_drift_prediction_extrapolates_in_time` |
| §7 latent vs observation | `…::test_observation_sd_exceeds_latent_sd` |
| §17 extrapolation flags | `…::test_extrapolation_flagged` |

## Amortized NPE engine (roadmap Phase 2)

`tests/test_amortized.py`:

| Requirement | Test |
|---|---|
| flow learns a conditional mean | `test_flow_learns_conditional_mean` |
| sampling matches posterior moments | `test_flow_sampling_matches_moments` |
| sampling is seed-reproducible | `test_flow_sampling_is_seed_reproducible` |
| conditioning is permutation-invariant, right dim | `test_featurizer_dimension_and_permutation_invariance` |
| weights save/load round-trip | `test_posterior_save_load_round_trip` |
| training recovers the detector state | `test_training_recovers_theta_direction` |
| Step 5 uses NPE with shipped weights (+ exact sampling) | `test_step5_uses_npe_engine_with_shipped_weights` |
| missing/mismatched weights → joint_map fallback | `test_step5_falls_back_when_weights_missing` |
| **SBC: well-specified posterior is calibrated** | `test_sbc_well_specified_is_calibrated` |
| SBC degrades gracefully under misspecification | `test_sbc_degrades_gracefully_under_misspecification` |

## End-to-end (integration)

`tests/test_loop_end_to_end.py` runs the complete loop against the
simulated detector once (module-scoped fixture) and asserts:

| Requirement | Test |
|---|---|
| validated improvement found, correction direction matches truth | `test_loop_finds_validated_improvement` |
| **no record was ever commanded outside the hard domain** | `test_loop_never_violates_hard_constraints` |
| HF/LF/time budgets respected | `test_loop_respects_budget` |
| feasibility report complete (proposal §3) | `test_feasibility_report_complete` |
| achieved geometry recorded everywhere (proposal §2.5) | `test_achieved_geometry_recorded_everywhere` |
| detector-state estimate consistent with hidden truth | `test_theta_estimate_close_to_truth` |

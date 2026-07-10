# MADMAX Closed-Loop Calibration
# Step 4 Technical Design: Measure the Selected Observable and, When Required, the Boost Factor

**Status:** Step-specific technical design note  
**Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3  
**Scope:** This document expands only **Step 4** of the seven-step closed-loop calibration algorithm. It describes the more technical structure of the measurement step after the booster geometry has been set and the antenna has been aligned. It does **not** specify detector-control software, final DAQ interfaces, final numerical thresholds, or the internal physics derivation of the MADMAX gradient-method boost-factor determination.

---

## 1. Role of Step 4 in the full calibration loop

Step 4 is the experimental measurement step of the outer calibration loop.

Before Step 4 begins, the loop has already done three things:

1. **Step 1** proposed a booster-state correction and, if multiple fidelities are available, selected a measurement action or fidelity.
2. **Step 2** moved the detector to the proposed booster geometry and recorded the achieved geometry.
3. **Step 3** aligned the antenna for that fixed booster geometry and recorded the achieved antenna position.

Step 4 then performs the measurement selected for this configuration.

At iteration \(t+1\), the complete achieved detector state is:

\[
q^{(t+1)}
=
q(q_0, \tilde{u}_B^{(t+1)}, \tilde{u}_A^{(t+1)}),
\]

where:

- \(q_0\) is the nominal configuration,
- \(\tilde{u}_B^{(t+1)}\) is the achieved booster-state correction,
- \(\tilde{u}_A^{(t+1)}\) is the achieved antenna position,
- and the tilde indicates achieved/read-back geometry rather than merely commanded geometry.

Step 4 must return a reliable data record for the model-update steps.

In words, Step 4 answers:

> For the current achieved detector configuration, what measured observable do we obtain, how uncertain is it, and is the measurement good enough to be used by the calibration model?

If the selected action is high-fidelity, Step 4 returns a measured boost-factor curve and a scalar objective value. If the selected action is lower-fidelity, Step 4 returns the selected proxy observable and its uncertainty.

---

## 2. What Step 4 is and is not

Step 4 is a **measurement and data-reduction step**.

It is responsible for:

- executing the measurement plan selected in Step 1,
- using the achieved geometry from Steps 2 and 3,
- applying the existing MADMAX boost-factor determination routine when high-fidelity measurement is requested,
- reducing raw measurement data to calibrated observables,
- estimating measurement uncertainty,
- assigning quality flags,
- and producing a data record for Steps 5 and 6.

Step 4 is not responsible for:

- proposing the next booster correction,
- changing the outer-loop acquisition strategy,
- re-optimizing the antenna,
- inferring detector-state parameters,
- fitting the simulation--measurement discrepancy model,
- or deciding the final physics objective.

Those responsibilities belong to other steps of the loop.

---

## 3. Inputs to Step 4

At iteration \(t+1\), Step 4 should receive a complete measurement packet from the previous steps.

### 3.1 Candidate and iteration metadata

Step 4 should know:

\[
t+1,
\]

as well as a unique candidate identifier. This identifier should connect the Step-4 measurement to:

- the Step-1 proposal,
- the Step-2 geometry move,
- the Step-3 antenna alignment,
- and the later Step-5/6 model update.

The identifier should remain attached to all raw and processed data products.

### 3.2 Achieved booster geometry

Step 4 should receive:

\[
\tilde{u}_B^{(t+1)}.
\]

This includes the achieved version of the proposed booster-state variables:

\[
\tilde{u}_B
=
\left(
\tilde{a}_{\mathrm{disk}},
\tilde{z}_{\mathrm{global}},
\tilde{z}_{\mathrm{reflecting\ mirror}},
\tilde{z}_{\mathrm{focusing\ mirror}}
\right).
\]

If readback exists at a finer level, Step 4 should also receive the achieved physical positions of the relevant disks, mirror, and focusing element, not only the low-dimensional control amplitudes.

The measurement record should preserve both:

\[
u_{B,\mathrm{cmd}}^{(t+1)}
\]

and:

\[
\tilde{u}_B^{(t+1)}.
\]

This is important because actuator hysteresis, creep, and finite positioning repeatability can make the achieved geometry differ from the commanded one.

### 3.3 Achieved antenna geometry

Step 4 should receive the antenna state after Step 3:

\[
\tilde{u}_A^{(t+1)}
=
(\tilde{x}_{\mathrm{ant}}, \tilde{y}_{\mathrm{ant}}).
\]

If the antenna-alignment step used multiple trial positions, Step 4 should also know which antenna position was selected as the final one for this candidate.

Step 4 should not repeat the full antenna-alignment loop unless a measurement-quality check shows that the antenna alignment has failed or drifted significantly.

### 3.4 Measurement action or fidelity

Step 4 should receive the measurement action selected by Step 1:

\[
\ell^{(t+1)}.
\]

The action \(\ell\) may represent, for example:

- a low-fidelity RF-response measurement,
- an antenna-coupling or receiver-chain proxy measurement,
- a reflectivity/group-delay measurement,
- a high-fidelity gradient-method boost-factor determination,
- or a high-fidelity validation repeat at the current best configuration.

The exact list of supported fidelities should be agreed with the experimental team.

### 3.5 Frequency window and objective definition

Step 4 should receive the target frequency window:

\[
W,
\]

and the already chosen scalar objective function:

\[
J.
\]

Step 4 does not choose the physics objective. It only evaluates the objective that was already defined for the calibration run.

For high-fidelity measurements, Step 4 will compute:

\[
J_{\mathrm{meas},t+1}
=
J\left(\widehat{\beta^2}_{t+1}(\nu), \nu \in W\right).
\]

For lower-fidelity measurements, Step 4 may not compute the final high-fidelity objective. Instead, it returns the proxy observable and its uncertainty.

### 3.6 Noise, budget, and replication instructions

Step 4 should receive the current measurement policy, for example:

- whether this measurement is exploratory or confirmatory,
- whether replication is required,
- the maximum allowed measurement time,
- whether the current candidate is the incumbent best and should be remeasured,
- and whether the measurement should include before/after baseline checks.

This policy is generated by the broader calibration strategy, but Step 4 must execute it and record what was actually done.

---

## 4. Outputs of Step 4

Step 4 should produce a standardized measurement record.

At minimum, the output should contain:

\[
D_{t+1}^{(4)}.
\]

The superscript indicates that this is the Step-4 measurement contribution to the full calibration data set.

### 4.1 Raw data products

The record should include references to the raw data products, for example:

- raw VNA traces,
- raw reflectivity data,
- raw receiver or power-spectrum data,
- raw bead-pull or gradient-method data, if applicable,
- actuator readback logs,
- antenna-position logs,
- timestamps,
- environmental or cryogenic monitoring data,
- and any instrument status logs relevant to the measurement.

The raw data should not be overwritten by later processing.

### 4.2 Processed observable

The record should include the processed observable corresponding to the chosen fidelity \(\ell\).

For a high-fidelity measurement, this is:

\[
\widehat{\beta^2}_{\mathrm{meas},t+1}(\nu).
\]

For a lower-fidelity measurement, this might be a proxy observable such as:

\[
y_{\ell,t+1}(\nu),
\]

where \(y_\ell\) may represent reflectivity, group delay, a coupling proxy, a receiver-chain observable, or another validated lower-fidelity measurement.

### 4.3 Scalar objective value

If the measurement is high-fidelity, Step 4 should compute:

\[
J_{\mathrm{meas},t+1}
=
J\left(\widehat{\beta^2}_{\mathrm{meas},t+1}(\nu)\right).
\]

If the measurement is lower-fidelity, the record should clearly indicate whether a final high-fidelity objective value is **not available** for this iteration.

The distinction is important:

\[
\text{low-fidelity proxy data}
\neq
\text{validated high-fidelity boost-factor objective}.
\]

### 4.4 Measurement uncertainty

The record should include uncertainty estimates:

\[
\sigma_{\beta^2,t+1}(\nu)
\]

for a high-fidelity boost-factor curve, and:

\[
\sigma_{J,t+1}
\]

for the scalar objective.

If the uncertainties are frequency-correlated, the record should ideally include a covariance representation:

\[
\Sigma_{\beta^2,t+1}(\nu_i,\nu_j),
\]

or at least a clear statement that the uncertainties are correlated and not independent point-by-point errors.

For a lower-fidelity measurement, the record should include the corresponding uncertainty:

\[
\sigma_{\ell,t+1}(\nu)
\]

or scalar uncertainty if the proxy is scalar.

### 4.5 Quality flags

Step 4 should return quality flags, for example:

- `valid_high_fidelity_measurement`,
- `valid_low_fidelity_measurement`,
- `repeat_requested`,
- `measurement_failed`,
- `geometry_out_of_tolerance`,
- `antenna_alignment_suspect`,
- `drift_suspected`,
- `instrument_status_bad`,
- `parasitic_mode_suspected`,
- `insufficient_signal_to_noise`,
- `objective_not_resolvable_above_noise`.

These flags are not just bookkeeping. They determine how Steps 5 and 6 should use the data and whether Step 1 should later treat the region as promising, uncertain, or problematic.

---

## 5. Measurement fidelities in Step 4

The parent proposal allows Step 1 to choose a measurement action or fidelity. Step 4 is where that choice is executed.

The useful distinction is:

\[
\text{cheap information-gathering measurement}
\]

versus:

\[
\text{expensive high-fidelity boost-factor determination}.
\]

### 5.1 High-fidelity measurement

The high-fidelity Step-4 action is the existing MADMAX gradient-method boost-factor determination.

For the calibration algorithm, this method is treated as a measurement oracle for:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu; q).
\]

The internal physics of the gradient method is not redesigned by this project. However, Step 4 must wrap that existing method in a reproducible measurement protocol:

1. define the achieved detector configuration,
2. execute the gradient-method measurement routine,
3. record all geometry readbacks and instrument settings,
4. obtain the measured boost-factor curve,
5. propagate uncertainties into \(\widehat{\beta^2}(\nu)\),
6. compute \(J\) if required,
7. and return a quality-controlled data record.

The high-fidelity measurement is the only measurement type that can validate the final accepted configuration.

### 5.2 Lower-fidelity measurements

Lower-fidelity measurements may include faster observables that are informative about the final boost-factor objective but are not identical to it.

Possible examples include:

- reflectivity measurements,
- group-delay measurements,
- RF-response measurements,
- antenna-coupling proxies,
- receiver-chain or power-spectrum diagnostics,
- and repeated geometry/readback stability checks.

These measurements are useful because they may be much cheaper than a full boost-factor determination. However, they should be explicitly labeled as lower fidelity:

\[
\ell \neq \mathrm{HF}.
\]

The data model should preserve the fidelity label so that Step 5 can learn how each lower-fidelity observable relates to the high-fidelity objective.

### 5.3 Final validation measurement

The best configuration returned by the complete calibration loop should be validated with the high-fidelity boost-factor measurement.

Therefore, Step 4 should support a special measurement action:

\[
\ell = \mathrm{HF\_validation}.
\]

This is not conceptually different from the high-fidelity measurement, but it should usually be treated more conservatively:

- repeat the measurement if possible,
- compare against the baseline or previous incumbent,
- record drift-sensitive diagnostics,
- and produce the final uncertainty statement.

---

## 6. High-fidelity gradient-method measurement wrapper

The project assumes the gradient method already exists. Step 4 should not replace it.

However, the calibration loop needs a wrapper around it.

The wrapper should have the following conceptual stages.

### 6.1 Pre-measurement state check

Before the gradient-method measurement begins, Step 4 should verify:

- the achieved booster geometry is within the allowed tolerance,
- the achieved antenna position is within the allowed tolerance,
- the system is in a stable measurement state,
- the required instrument configuration is available,
- no hard safety interlock has been triggered,
- and no previous step has marked the candidate as invalid.

If this check fails, Step 4 should not silently return a bad objective value. It should return a failed or invalid measurement record.

### 6.2 Execute the existing boost-factor determination routine

For high-fidelity measurement, Step 4 calls the existing gradient-method routine.

Conceptually, the routine returns:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu),
\]

and possibly intermediate electromagnetic response data.

Because the gradient method involves small controlled disk movements, Step 4 should record:

- the commanded small movements,
- the achieved small movements,
- the order in which the movements were performed,
- the raw response measurements used by the gradient-method routine,
- and whether the detector returned to the intended nominal candidate state after the sequence.

This is essential because the outer optimizer should learn from the achieved geometry, not from an idealized geometry that was only commanded.

### 6.3 Post-measurement state check

After the high-fidelity measurement, Step 4 should verify:

- whether the achieved geometry has drifted,
- whether the antenna position is still valid,
- whether the instrument state remained stable,
- and whether repeated or reference measurements agree within the expected uncertainty.

If the high-fidelity measurement itself perturbs the detector state, that perturbation must be recorded and passed to the later model-update steps.

### 6.4 Return a high-fidelity data product

The high-fidelity data product should contain:

\[
\left(
\tilde{u}_B,
\tilde{u}_A,
t,
\ell=\mathrm{HF},
\widehat{\beta^2}_{\mathrm{meas}}(\nu),
\sigma_{\beta^2}(\nu),
J_{\mathrm{meas}},
\sigma_J,
\text{quality flags}
\right).
\]

This is the main experimental object used by Steps 5 and 6.

---

## 7. Lower-fidelity measurement wrapper

When Step 1 selects a lower-fidelity action, Step 4 should execute that measurement without pretending that the final boost-factor objective has been measured.

### 7.1 Pre-measurement state check

The lower-fidelity measurement should still verify:

- achieved geometry,
- antenna position,
- instrument readiness,
- and basic signal quality.

Lower fidelity does not mean lower discipline in data recording.

### 7.2 Execute the selected proxy measurement

The output is a lower-fidelity observable:

\[
y_{\ell,t+1}(\nu)
\]

or:

\[
y_{\ell,t+1}
\]

if the proxy is scalar.

Step 4 should store the proxy in a form that allows Step 5 to model its relationship to the high-fidelity objective.

For example, if the observable is reflectivity or group delay, the record should include the frequency grid, calibration state, and processing assumptions.

### 7.3 Return a lower-fidelity data product

The lower-fidelity data product should contain:

\[
\left(
\tilde{u}_B,
\tilde{u}_A,
t,
\ell,
y_{\ell}(\nu),
\sigma_{\ell}(\nu),
\text{quality flags}
\right).
\]

It should explicitly state:

\[
J_{\mathrm{HF}} \; \text{not measured in this iteration}.
\]

This prevents lower-fidelity data from being accidentally treated as a validated boost-factor objective.

---

## 8. Data reduction for the measured boost-factor curve

If Step 4 performs a high-fidelity measurement, the main processed output is:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu).
\]

The data reduction should be standardized so that different iterations are comparable.

### 8.1 Frequency grid handling

The measured curve should be represented on a defined frequency grid:

\[
\nu_1, \nu_2, \dots, \nu_m.
\]

The record should specify:

- the measured frequency range,
- the target window \(W\),
- the frequency resolution,
- any interpolation or rebinning,
- and any masked or excluded frequency regions.

If the scalar objective \(J\) uses a specific frequency grid, Step 4 should evaluate all high-fidelity measurements on that grid or document how they were transformed.

### 8.2 Conversion from curve to objective

Once \(\widehat{\beta^2}_{\mathrm{meas}}(\nu)\) is available, Step 4 computes:

\[
J_{\mathrm{meas}}
=
J\left(\widehat{\beta^2}_{\mathrm{meas}}(\nu), \nu \in W\right).
\]

Step 4 should not choose between peak boost, average boost, smooth-min boost, scan-rate proxy, or multi-objective quantities. That decision belongs to the objective-definition stage of the full proposal.

Step 4 only evaluates the chosen objective consistently.

### 8.3 Handling multi-objective outputs

If the calibration run uses a multi-objective formulation, Step 4 should return a vector rather than a single scalar:

\[
\mathbf{J}_{\mathrm{meas}}
=
\left(
J_1,
J_2,
\dots,
J_K
\right).
\]

Examples might include:

- peak boost,
- bandwidth or band flatness,
- scan-rate proxy,
- measurement cost,
- or robustness indicators.

The output record should clearly identify each objective component and its uncertainty.

---

## 9. Uncertainty estimation in Step 4

Uncertainty estimation is one of the central responsibilities of Step 4.

The outer Bayesian optimizer should not receive only a point value:

\[
J_{\mathrm{meas}}.
\]

It should receive:

\[
J_{\mathrm{meas}} \pm \sigma_J,
\]

or, more generally, a likelihood for the measured observable.

### 9.1 Type-A uncertainty components

Type-A components come from statistical variation in repeated measurements.

Examples include:

- repeated RF-response sweeps,
- repeated gradient-method determinations at the same achieved geometry,
- repeated measurements of the baseline configuration,
- short-term receiver or VNA noise,
- and variation across repeated antenna-coupling checks.

If Step 4 repeats a measurement, it should return the repeat statistics rather than only the average.

### 9.2 Type-B uncertainty components

Type-B components come from known or estimated systematic effects.

Examples include:

- geometry readback uncertainty,
- actuator reproducibility,
- antenna-position uncertainty,
- calibration uncertainty of the measurement chain,
- uncertainty from the finite perturbation steps used inside the gradient method,
- model dependence inside the boost-factor determination routine,
- drift during the measurement,
- and known systematics in the receiver chain or RF setup.

The exact uncertainty model should be finalized with the experimental team, but Step 4 must at least preserve the information needed to build such a budget.

### 9.3 Frequency-dependent uncertainty

The boost-factor uncertainty may depend on frequency:

\[
\sigma_{\beta^2}(\nu_i) \neq \sigma_{\beta^2}(\nu_j).
\]

Therefore Step 4 should not assume a single constant uncertainty across the entire measured curve unless this has been validated.

The outer model should be able to receive heteroscedastic uncertainty information if available.

### 9.4 Correlated uncertainty across frequency

Uncertainties in the boost-factor curve may be correlated across frequency bins.

For example, a global normalization uncertainty affects many frequencies coherently, while local measurement noise may affect individual bins more independently.

Therefore, when possible, Step 4 should preserve the distinction between:

\[
\text{pointwise noise}
\]

and:

\[
\text{shared systematic uncertainty}.
\]

If a full covariance matrix is too expensive or not available, Step 4 should still record which uncertainty components are likely correlated.

### 9.5 Propagating curve uncertainty to objective uncertainty

The scalar objective uncertainty should be derived from the uncertainty on the measured curve:

\[
\widehat{\beta^2}(\nu) \rightarrow J.
\]

Conceptually, this can be done by propagating uncertainty through the objective function.

For example, if \(J\) is a smooth function of the curve values, standard uncertainty propagation may be sufficient. If \(J\) is a non-linear or non-smooth objective, a Monte-Carlo propagation over sampled boost-factor curves may be more appropriate.

The exact numerical implementation can be chosen later, but the Step-4 output must include:

\[
\sigma_J.
\]

Without \(\sigma_J\), the optimizer cannot know whether an apparent improvement is meaningful.

---

## 10. Measurement-quality control

Step 4 should apply quality control before passing data to the statistical model.

The purpose is not to discard inconvenient data, but to label data honestly.

### 10.1 Geometry consistency check

Verify that:

\[
\tilde{u}_B^{(t+1)}
\]

and:

\[
\tilde{u}_A^{(t+1)}
\]

are close enough to the intended measurement state.

If not, Step 4 should either:

- use the achieved geometry in the data record,
- request a repeat after repositioning,
- or return a quality flag indicating that the measurement was not taken at the intended state.

The calibration data should never pretend that the commanded geometry was achieved when readback says otherwise.

### 10.2 Instrument-state check

The record should include whether the relevant measurement instruments were in a valid state.

Examples include:

- VNA settings,
- receiver-chain configuration,
- integration time,
- frequency range,
- calibration state,
- gain or noise calibration state,
- and whether any interlock or instrument warning occurred.

The exact instrumentation details depend on the experimental setup and should be supplied by the MADMAX team.

### 10.3 Signal-quality check

Step 4 should flag measurements where the signal quality is too poor to support a useful objective value.

Possible indicators include:

- low signal-to-noise ratio,
- unstable repeated sweeps,
- saturated or clipped traces,
- unexpected discontinuities,
- missing frequency regions,
- or disagreement between repeated measurements beyond the expected uncertainty.

### 10.4 Mode and parasitic-response check

MADMAX prototype analyses have shown that parasitic modes, higher-order modes, antenna reflections, and receiver-chain effects can affect the interpreted boost-factor response.

Therefore Step 4 should include quality flags for situations such as:

- suspected parasitic mode contamination,
- unexpected extra peaks,
- mode identification ambiguity,
- antenna-reflection ripples,
- receiver-chain mismatch effects,
- or a mismatch between a proxy observable and the expected response.

These flags should not necessarily invalidate the measurement automatically. Instead, they tell Step 5 that the data may require a larger discrepancy or noise model.

### 10.5 Drift check

Step 4 should record timing information:

\[
t_{\mathrm{start}}, \quad t_{\mathrm{end}}.
\]

If the measurement takes long enough that drift is plausible, Step 4 should support one or more of the following:

- compare pre- and post-measurement geometry readback,
- repeat a quick reference measurement,
- remeasure the baseline or incumbent best configuration periodically,
- or flag the measurement as potentially drift-affected.

This is especially important for high-fidelity measurements that take long enough for mechanical or thermal drift to matter.

---

## 11. Replication policy

Step 4 should support replication, because the calibration loop can only improve the detector if improvements are resolvable above measurement noise.

### 11.1 Baseline replication

The nominal baseline configuration should be measured more than once at the beginning or during the run when possible:

\[
q_0 \rightarrow J_0^{(1)}, J_0^{(2)}, \dots
\]

This gives an empirical estimate of measurement repeatability.

### 11.2 Candidate replication

Candidate replication should be used when:

- a candidate appears to improve the objective but only marginally,
- the predicted improvement is comparable to \(\sigma_J\),
- the quality flags are ambiguous,
- the candidate may become the new incumbent best,
- or Step 1 requests replication for information-gain reasons.

### 11.3 Incumbent remeasurement

The current best configuration should be remeasured periodically if drift is plausible.

This prevents the optimizer from comparing a fresh candidate against a stale incumbent measured under different conditions.

### 11.4 Replication output

If a measurement is repeated, Step 4 should return:

- all individual repeated values,
- the average or combined estimate,
- the empirical scatter,
- and any systematic difference between early and late repeats.

The data record should not hide the individual repeats.

---

## 12. Measurement-cost accounting

Step 4 should record the cost of each measurement.

At minimum, the record should include:

\[
C_{t+1}
\]

where \(C\) may include:

- total wall-clock time,
- number of high-fidelity boost-factor determinations,
- number of lower-fidelity measurements,
- number of small disk moves used internally by the gradient method,
- antenna checks or realignments triggered during measurement,
- and whether the measurement consumed special cryogenic or magnet time.

This cost record is passed back to Step 1 so that later acquisition decisions can be cost-aware.

The cost record is also needed for the final feasibility report.

---

## 13. Data schema for Step 4

The following schema is a conceptual design, not a software-interface requirement.

A Step-4 measurement record should contain:

```text
MeasurementRecord:
    candidate_id
    iteration_index
    timestamp_start
    timestamp_end

    commanded_booster_state
    achieved_booster_state
    commanded_antenna_state
    achieved_antenna_state

    measurement_fidelity
    measurement_protocol_id
    instrument_configuration_id

    raw_data_references
    processed_observable
    processed_observable_uncertainty
    processed_observable_covariance_or_correlation_notes

    scalar_objective_value
    scalar_objective_uncertainty
    objective_definition_id

    quality_flags
    validity_status
    repeat_index_or_replication_group

    measurement_cost
    comments
```

The important point is that every objective value must be traceable back to:

\[
\text{geometry} + \text{measurement protocol} + \text{raw data} + \text{uncertainty model}.
\]

---

## 14. Interaction with Step 1

Step 4 does not propose the next move, but it provides the information Step 1 will later need.

From Step 4, Step 1 eventually receives, through Steps 5 and 6:

- whether the selected measurement was high- or lower-fidelity,
- what objective or proxy value was observed,
- how uncertain the observation was,
- how expensive the measurement was,
- whether the measurement was valid,
- whether the candidate region is promising,
- whether the region is noisy or unstable,
- and whether future measurements should be replicated or avoided.

Therefore Step 4 should be designed so that the data it produces can support:

\[
\text{expected improvement},
\quad
\text{information gain},
\quad
\text{cost awareness},
\quad
\text{noise awareness},
\quad
\text{safe exploration}.
\]

---

## 15. Interaction with Step 5

Step 5 jointly updates the detector-state, discrepancy, noise, and drift model.

Therefore Step 4 must provide data in a form suitable for that inference task.

The Step-5 input should include:

\[
D_{t+1}
=
\left(
\tilde{u}_B^{(t+1)},
\tilde{u}_A^{(t+1)},
t_{t+1},
\ell_{t+1},
\text{measured observable}_{t+1},
\sigma_{t+1},
\text{quality flags}_{t+1}
\right).
\]

For high-fidelity measurements, the measured observable includes:

\[
\widehat{\beta^2}_{\mathrm{meas},t+1}(\nu)
\]

and:

\[
J_{\mathrm{meas},t+1}.
\]

For lower-fidelity measurements, the measured observable is the proxy observable and its uncertainty.

Step 4 should not simplify all data into a single scalar if richer data are available. The full curve or proxy trace may contain information useful for detector-state and discrepancy inference.

---

## 16. Interaction with Step 6

Step 6 updates the optimizer-facing posterior predictive model.

For that to work, Step 4 must preserve enough information to construct a likelihood or observation model.

For example, Step 6 needs to know whether a data point represents:

\[
J_{\mathrm{HF}}(u)
\]

or:

\[
y_{\ell}(u), \quad \ell \neq \mathrm{HF}.
\]

It also needs the corresponding uncertainty.

Therefore Step 4 should avoid ambiguous outputs such as:

```text
score = 0.82
```

without saying whether the score is a high-fidelity objective, a proxy objective, a normalized alignment metric, or a diagnostic quantity.

---

## 17. Minimal first implementation of Step 4

A first implementation of Step 4 can be deliberately simple.

The minimal version should support:

1. one high-fidelity measurement mode using the existing gradient-method boost-factor determination,
2. one lower-fidelity RF or proxy measurement mode if available,
3. achieved geometry recording,
4. scalar objective calculation for high-fidelity data,
5. basic uncertainty estimation,
6. basic quality flags,
7. and a standardized measurement record.

The minimal high-fidelity output should be:

\[
\left(
\tilde{u}_B,
\tilde{u}_A,
\widehat{\beta^2}_{\mathrm{meas}}(\nu),
J_{\mathrm{meas}},
\sigma_J,
\text{validity flag}
\right).
\]

This is enough to allow Steps 5 and 6 to update the calibration model and allow Step 1 to propose the next move.

---

## 18. More complete implementation of Step 4

A more complete implementation should add:

- repeated measurements at selected configurations,
- frequency-dependent uncertainty on \(\widehat{\beta^2}(\nu)\),
- covariance or correlation information across frequency,
- explicit Type-A and Type-B uncertainty components,
- lower-fidelity proxy measurements with fidelity labels,
- cost accounting,
- baseline/incumbent remeasurement support,
- drift flags,
- and full raw-data provenance.

This version is more useful for a real calibration campaign because it allows the statistical model to distinguish:

\[
\text{real detector improvement}
\]

from:

\[
\text{measurement noise, drift, or proxy mismatch}.
\]

---

## 19. Recommended Step-4 execution logic

The Step-4 logic can be summarized as follows.

```text
Input:
    candidate_id
    achieved booster state from Step 2
    achieved antenna state from Step 3
    selected measurement fidelity/action from Step 1
    target frequency window W
    objective definition J
    measurement budget and replication instructions

Step 4 procedure:

    1. Check pre-measurement validity:
       geometry, antenna position, instrument state, safety status.

    2. Execute the selected measurement:
       if fidelity is high:
           run the existing gradient-method boost-factor determination.
       if fidelity is lower:
           run the selected proxy measurement.

    3. Record raw data and metadata:
       raw traces, readbacks, timestamps, instrument settings, protocol ID.

    4. Reduce data:
       produce beta^2_meas(nu) for high-fidelity measurements,
       or produce y_l(nu) for lower-fidelity measurements.

    5. Estimate uncertainty:
       produce curve uncertainty and objective uncertainty where applicable.

    6. Compute objective when valid:
       for high-fidelity data, compute J_meas from beta^2_meas(nu).
       for lower-fidelity data, mark J_HF as not measured.

    7. Apply quality control:
       assign validity status and quality flags.

    8. Record measurement cost:
       wall-clock time, repetitions, high-fidelity count, proxy count.

    9. Return standardized MeasurementRecord:
       pass to Step 5 and Step 6.
```

---

## 20. Failure modes and recommended responses

Step 4 should explicitly handle failure modes.

### 20.1 Measurement fails completely

Examples:

- instrument failure,
- missing raw data,
- interlock event,
- hardware state invalid,
- or measurement aborted.

Recommended response:

- return an invalid measurement record,
- preserve available metadata,
- do not fabricate \(J\),
- and let Step 1 or the operator decide whether to repeat or avoid the region.

### 20.2 Measurement succeeds but geometry differs from command

Recommended response:

- use achieved geometry in the record,
- flag the command--achievement discrepancy,
- and avoid treating the candidate as if it had been measured at the commanded point.

### 20.3 Measurement is too noisy to rank candidates

Recommended response:

- return the measured value and large uncertainty,
- consider replication,
- do not declare an improvement unless it is statistically meaningful,
- and pass the uncertainty to the model.

### 20.4 Lower-fidelity proxy disagrees with expectation

Recommended response:

- return the proxy value and quality flags,
- do not convert it into a high-fidelity objective by hand,
- and let the joint calibration/discrepancy model decide how to interpret the discrepancy.

### 20.5 Drift suspected during measurement

Recommended response:

- record pre/post readback differences,
- add a drift flag,
- possibly repeat a reference measurement,
- and pass timing information to the drift component of the model.

### 20.6 Parasitic or ambiguous mode behavior appears

Recommended response:

- preserve the raw curve,
- add a parasitic-mode or ambiguity flag,
- avoid overconfident scalar summaries,
- and allow Step 5 to model the observation as noisy or discrepant.

---

## 21. Validation tests for Step 4 before full closed-loop operation

Before using Step 4 inside the full calibration loop, it should be validated independently.

### 21.1 Baseline repeatability test

Measure the nominal configuration multiple times and estimate:

\[
\sigma_{J,0}.
\]

This tests whether the expected calibration improvements are resolvable above measurement noise.

### 21.2 High-fidelity pipeline test

Run the full high-fidelity measurement wrapper at a known configuration and verify that:

- raw data are saved,
- the boost-factor curve is produced,
- the objective is computed,
- uncertainty is attached,
- and quality flags are meaningful.

### 21.3 Lower-fidelity pipeline test

Run each lower-fidelity measurement mode and verify that:

- the proxy observable is recorded,
- its fidelity label is preserved,
- uncertainty is attached,
- and it is not accidentally treated as a high-fidelity objective.

### 21.4 Geometry-readback test

Command a set of known positions and verify that the Step-4 record correctly stores both commanded and achieved geometry.

This test is important for avoiding hysteresis-smeared learning.

### 21.5 Drift test

Repeat a reference measurement before and after a longer measurement block.

Check whether the measured change is consistent with the assumed drift model or whether the loop needs more frequent re-baselining.

### 21.6 Objective-consistency test

Take the same measured boost-factor curve and evaluate the selected objective \(J\).

Confirm that the objective calculation is reproducible and uses the same frequency window, masks, and scalarization every time.

### 21.7 End-to-end handoff test

Pass a Step-4 measurement record into the Step-5/6 model update without moving hardware.

Verify that the model can distinguish:

- high-fidelity observations,
- lower-fidelity observations,
- missing high-fidelity objectives,
- invalid measurements,
- large uncertainties,
- and quality flags.

---

## 22. Practical design principles for Step 4

The most important design principles are:

1. **Measure achieved geometry, not ideal geometry.**  
   The calibration loop should learn the real map from achieved detector state to measured response.

2. **Do not hide uncertainty.**  
   A boost-factor or objective value without uncertainty is not sufficient for noisy Bayesian optimization.

3. **Separate high-fidelity objective data from lower-fidelity proxy data.**  
   Proxy measurements are useful, but they do not replace final boost-factor validation.

4. **Preserve raw data and metadata.**  
   Later model updates may need information that was not part of the first scalar summary.

5. **Flag problems instead of silently correcting them.**  
   Ambiguous modes, drift, poor signal quality, and geometry mismatch should be passed forward explicitly.

6. **Make measurements comparable across iterations.**  
   The same objective, frequency grid, masks, and uncertainty conventions should be used consistently.

7. **Record measurement cost.**  
   The optimizer cannot be budget-aware unless Step 4 reports what each measurement actually cost.

---

## 23. Step-4 output in compact mathematical form

For a high-fidelity measurement, Step 4 should return:

\[
D_{t+1}^{(4)}
=
\left(
\tilde{u}_B^{(t+1)},
\tilde{u}_A^{(t+1)},
t_{t+1},
\ell_{t+1}=\mathrm{HF},
\widehat{\beta^2}_{\mathrm{meas},t+1}(\nu),
\sigma_{\beta^2,t+1}(\nu),
J_{\mathrm{meas},t+1},
\sigma_{J,t+1},
Q_{t+1},
C_{t+1}
\right),
\]

where:

- \(Q_{t+1}\) denotes quality flags,
- and \(C_{t+1}\) denotes measurement cost.

For a lower-fidelity measurement, Step 4 should return:

\[
D_{t+1}^{(4)}
=
\left(
\tilde{u}_B^{(t+1)},
\tilde{u}_A^{(t+1)},
t_{t+1},
\ell_{t+1},
y_{\ell,t+1}(\nu),
\sigma_{\ell,t+1}(\nu),
Q_{t+1},
C_{t+1}
\right),
\]

with an explicit statement that:

\[
J_{\mathrm{HF},t+1}
\]

was not measured in that iteration.

---

## 24. Final Step-4 summary

Step 4 is the bridge between hardware and statistics.

It takes the achieved detector state and returns a quality-controlled, uncertainty-aware measurement record.

The high-fidelity version of Step 4 uses the existing MADMAX gradient method to obtain:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu),
\]

then evaluates the chosen calibration objective:

\[
J_{\mathrm{meas}}.
\]

The lower-fidelity version of Step 4 returns cheaper proxy observables that can inform the joint calibration and Bayesian-optimization model, but cannot by themselves validate the final configuration.

The essential output is not only a number. It is:

\[
\text{measurement} + \text{uncertainty} + \text{geometry} + \text{fidelity} + \text{quality flags} + \text{cost}.
\]

That full record is what allows the later inference and optimization steps to remain statistically honest.

---

## 25. Source anchors

This Step-4 design follows the parent proposal and uses the following external references as background anchors.

1. **Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3.  
   The parent proposal defines Step 4 as measuring the selected observable and, when required, the boost factor; it also requires high-fidelity validation, lower-fidelity proxy handling, achieved-geometry readback, noise modeling, drift handling, and cost-aware multi-fidelity calibration.

2. **MADMAX overview:**  
   The 2024 MADMAX overview describes in-situ boost-factor measurement from electromagnetic response using a bead-pull method and reports a prototype boost-factor peak around 600 with about 15% uncertainty.  
   <https://arxiv.org/html/2409.20169v1>

3. **MADMAX proof-of-principle booster:**  
   The proof-of-principle paper discusses the relationship between reflectivity, group delay, and boost factor; it also discusses tuning from measured electromagnetic response, systematic effects from antenna reflections, frequency stability, and uncertainty extraction.  
   <https://link.springer.com/article/10.1140/epjc/s10052-020-7985-8>

4. **MADMAX prototype axion search:**  
   The prototype axion-search paper discusses boost-factor determination from measurements, field-shape identification, receiver-chain effects, drift monitoring, and boost-factor uncertainties at the 13%--17% level for the prototype runs.  
   <https://bib-pubdb1.desy.de/record/639228/files/c749-419q.pdf?subformat=pdfa>

5. **Experimental determination of axion signal power:**  
   This work describes reciprocity and bead-pull ideas for directly determining expected signal power from measurements in open broadband haloscope setups.  
   <https://arxiv.org/html/2311.13359v2>

6. **Measurement uncertainty guidance:**  
   NIST Technical Note 1297 gives the standard measurement-uncertainty language used here, including Type-A and Type-B uncertainty components, combined uncertainty, expanded uncertainty, and reporting of uncertainty.  
   <https://www.nist.gov/pml/nist-technical-note-1297>

7. **BoTorch noise-modeling context:**  
   BoTorch documentation distinguishes fixed-noise and heteroscedastic-noise settings, which motivates preserving measurement uncertainties rather than passing only point values to the optimizer.  
   <https://botorch.org/docs/models>

8. **BoTorch multi-fidelity context:**  
   BoTorch documentation describes multi-fidelity Bayesian optimization with knowledge-gradient acquisition, motivating the separation between cheaper proxy measurements and the target high-fidelity objective.  
   <https://botorch.org/docs/v0.16.0/tutorials/multi_fidelity_bo>

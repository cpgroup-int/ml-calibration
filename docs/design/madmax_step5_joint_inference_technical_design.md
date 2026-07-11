# MADMAX Closed-Loop Calibration
# Step 5 Technical Design: Jointly Update Detector-State, Discrepancy, Noise, and Drift Inference

**Status:** Step-specific technical design note  
**Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3  
**Scope:** This document expands only **Step 5** of the seven-step closed-loop calibration algorithm. It describes how the accumulated calibration data should be used to update the joint statistical model of detector state, simulator discrepancy, measurement noise, and drift. It does **not** specify the final detector-control software, final priors, exact numerical thresholds, or a final implementation language.

---

## 1. Role of Step 5 in the full calibration loop

Step 5 is the main **statistical inference update** of the closed-loop calibration algorithm.

Before Step 5 begins, the loop has already done the following:

1. **Step 1** proposed a safe booster-state correction and selected a measurement action or fidelity.
2. **Step 2** moved the booster geometry and recorded the achieved booster state.
3. **Step 3** aligned the antenna for that fixed booster state and recorded the achieved antenna position.
4. **Step 4** measured the selected observable, and when required, measured the boost factor using the high-fidelity gradient method.

Step 5 then updates the statistical belief about the real detector.

At iteration \(t+1\), Step 5 receives the accumulated calibration data:

\[
D_{1:t+1}
=
\left\{
 \tilde{u}_B^{(i)},
 \tilde{u}_A^{(i)},
 t_i,
 \ell_i,
 \text{measured observable}_i,
 J_i,
 \sigma_{J,i},
 \text{quality flags}_i
\right\}_{i=0}^{t+1}.
\]

The target of Step 5 is the joint calibration state:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}).
\]

Here:

- \(\theta_t\) denotes physically interpretable detector-state or nuisance parameters;
- \(r_\ell\) denotes simulation--measurement discrepancy, possibly dependent on measurement fidelity \(\ell\);
- \(\sigma_J\) denotes measurement-noise behavior for the scalar objective or other reduced observables;
- and `drift` denotes slow time dependence not explained by commanded or achieved geometry.

In words, Step 5 answers:

> Given all calibration data so far, what detector-state parameters, simulator discrepancy, measurement noise, and drift behavior are consistent with the observations?

This step is deliberately **joint**. It should not first fit \(\theta\) and only afterward fit a leftover residual. Detector-state parameters and model discrepancy can be confounded, so the discrepancy model must be present while \(\theta\) is inferred.

---

## 2. What Step 5 is and is not

Step 5 is an **inference and model-calibration step**.

It is responsible for:

- collecting the Step-4 measurement records into the calibration data set;
- checking which data are usable for inference;
- updating the joint model of detector state, discrepancy, noise, and drift;
- classifying inferred detector-state parameters as correctable or diagnostic-only;
- quantifying uncertainty and parameter correlations;
- detecting identifiability or confounding problems;
- and producing a model-state object for Step 6.

Step 5 is not responsible for:

- proposing the next booster-state correction;
- computing the final Step-1 acquisition function;
- moving detector hardware;
- repeating antenna alignment;
- performing the high-fidelity gradient-method measurement;
- or declaring the final best configuration.

Those responsibilities belong to other steps.

The boundary with Step 6 is especially important:

- **Step 5** updates the joint posterior model.
- **Step 6** converts that joint posterior into the optimizer-facing posterior predictive model \(p(J_{\mathrm{HF}}(u) \mid D)\).

Step 5 may prepare everything needed for posterior prediction, but Step 6 is where the prediction object for future candidate configurations is constructed.

---

## 3. Core modeling principle

The central model should have the following structure:

\[
y_{i,\ell}
=
S_\ell(\tilde{u}_i, \theta_{t_i})
+
r_\ell(\tilde{u}_i, t_i)
+
\epsilon_{i,\ell}.
\]

Here:

- \(y_{i,\ell}\) is the measured data product from Step 4;
- \(\ell\) is the measurement fidelity or measurement type;
- \(\tilde{u}_i = (\tilde{u}_B^{(i)}, \tilde{u}_A^{(i)})\) is the achieved geometry;
- \(S_\ell\) is the simulator or simulator-derived prediction for that measurement type;
- \(\theta_{t_i}\) are detector-state parameters, possibly slowly time-dependent;
- \(r_\ell\) is a discrepancy term capturing simulator--measurement mismatch;
- and \(\epsilon_{i,\ell}\) is measurement noise.

For high-fidelity boost-factor measurements, \(y_{i,\ell}\) can be either the measured boost-factor curve:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu; \tilde{u}_i),
\]

or a reduced scalar objective:

\[
J_i = J\left(\widehat{\beta^2}_{\mathrm{meas}}(\nu; \tilde{u}_i), \nu \in W\right).
\]

The curve-level form is statistically richer. The scalar-objective form is simpler. The first implementation can use the scalar form if the curve-level model is not yet ready, but the document should record that this sacrifices information about frequency-dependent structure.

---

## 4. Recommended observation levels

Step 5 should support three observation levels, ordered from simplest to richest.

### 4.1 Scalar-objective level

At the simplest level, each high-fidelity measurement contributes:

\[
(J_i, \sigma_{J,i}).
\]

The model becomes:

\[
J_i
=
J_{\mathrm{sim}}(\tilde{u}_i, \theta_{t_i})
+
r_J(\tilde{u}_i, t_i)
+
\epsilon_i.
\]

This is the easiest form to connect to Step 1 and Step 6.

Use this level for the first working prototype of Step 5.

### 4.2 Curve-summary level

A slightly richer level compresses the measured boost-factor curve into a small number of physically meaningful summaries, for example:

\[
z_i =
(\text{peak height},\ \text{peak frequency},\ \text{bandwidth},\ \text{band flatness},\ J_i).
\]

The model becomes multivariate:

\[
z_i
=
z_{\mathrm{sim}}(\tilde{u}_i, \theta_{t_i})
+
r_z(\tilde{u}_i, t_i)
+
\epsilon_i.
\]

This level helps diagnose whether a bad \(J\) came from a frequency shift, a loss of peak amplitude, a bandwidth change, or poor band flatness.

### 4.3 Curve-level inference

The richest level uses the measured curve directly:

\[
\widehat{\beta^2}_i(\nu)
=
\beta^2_{\mathrm{sim}}(\nu; \tilde{u}_i, \theta_{t_i})
+
r_\beta(\nu, \tilde{u}_i, t_i)
+
\epsilon_i(\nu).
\]

This is the most informative but also the most demanding, because the discrepancy and noise may be frequency-dependent.

A practical implementation can start with scalar-level inference and later add curve summaries or curve-level inference when the data volume and modeling effort justify it.

---

## 5. Inputs to Step 5

Step 5 should receive a standardized data packet from Step 4 and the previous posterior state.

### 5.1 Measurement records

Each measurement record should include:

\[
D_i =
(\tilde{u}_i, t_i, \ell_i, y_i, \sigma_i, \text{quality flags}_i).
\]

At minimum, the record should contain:

- iteration index;
- candidate identifier;
- commanded booster variables;
- achieved booster variables;
- commanded antenna variables;
- achieved antenna variables;
- measurement time or run index;
- measurement fidelity \(\ell\);
- measured observable;
- scalar objective if available;
- uncertainty estimate;
- replication information;
- baseline/incumbent-repeat tag if applicable;
- and measurement-quality flags.

The achieved geometry is the primary input to the statistical model. Commanded geometry should be stored for actuator diagnostics, but the inference model should prefer achieved geometry whenever available.

### 5.2 Simulator interface

Step 5 needs access to the fast physics simulator in a form that can evaluate:

\[
S_\ell(\tilde{u}_i, \theta)
\]

for the measurement type \(\ell\).

For high-fidelity boost-factor data, this may mean:

\[
\beta^2_{\mathrm{sim}}(\nu; q_0, \tilde{u}_B, \tilde{u}_A, \theta),
\]

or the corresponding scalar objective:

\[
J_{\mathrm{sim}}(\tilde{u}, \theta)
=
J\left(\beta^2_{\mathrm{sim}}(\nu; q_0, \tilde{u}, \theta), \nu \in W\right).
\]

For lower-fidelity data, the simulator interface should return the simulated version of the lower-fidelity observable, not pretend that the lower-fidelity observable is identical to the high-fidelity boost factor.

### 5.3 Prior model

Step 5 needs prior information for all inferred quantities:

\[
p(\theta, r_\ell, \sigma_J, \text{drift}).
\]

The priors should encode experimental knowledge such as:

- mechanical tolerances;
- readback precision;
- expected disk-position reproducibility;
- plausible mirror or focusing offsets;
- plausible antenna offsets;
- expected loss ranges;
- expected measurement noise;
- allowed drift rates;
- and expected discrepancy amplitude and smoothness.

The discrepancy prior is especially important. If the discrepancy GP is too flexible, it can absorb real geometry effects and make \(\theta\) meaningless. If the discrepancy prior is too rigid, the model can force simulator error into \(\theta\), again biasing the detector-state inference.

### 5.4 Control-basis map

Step 5 also needs the map between inferred detector-state variables and controllable variables.

The control basis defines:

\[
q_{\mathrm{disk}}
=
q_{0,\mathrm{disk}} + B a_{\mathrm{disk}}.
\]

Step 5 should know which components of \(\theta\) are spanned by the online correction basis and which are not.

This supports the classification:

\[
\theta = (\theta_{\mathrm{corr}}, \theta_{\mathrm{diag}}).
\]

### 5.5 Previous inference state

At iteration \(t+1\), Step 5 should receive the previous inference state from iteration \(t\):

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t}).
\]

A first implementation may store this as a joint MAP estimate plus approximate covariance. A more Bayesian implementation may store posterior samples, variational approximations, or a fitted hierarchical model state.

---

## 6. Outputs of Step 5

Step 5 should produce a standardized model-update record:

\[
M_{t+1}^{(5)}.
\]

This record is passed to Step 6.

### 6.1 Updated joint posterior or approximation

The main output is:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}),
\]

or an approximation to it.

Acceptable approximation levels include:

- joint MAP estimate plus uncertainty approximation;
- Laplace approximation around the joint MAP;
- variational approximation;
- posterior samples from MCMC or HMC/NUTS;
- or a hybrid approximation, such as MAP for some nuisance hyperparameters and posterior samples for the physically important variables.

The output should make clear which approximation level was used.

### 6.2 Detector-state parameter summaries

For each detector-state parameter, Step 5 should return:

- posterior mean or MAP value;
- posterior uncertainty;
- prior-to-posterior shift;
- physical units;
- whether the value is near a prior or hardware bound;
- whether the parameter is correctable or diagnostic-only;
- and any strong posterior correlations with discrepancy or other parameters.

The summaries must not overstate identifiability. If a parameter is strongly confounded with the discrepancy term, Step 5 should label it as weakly identifiable.

### 6.3 Discrepancy-model state

Step 5 should return the updated discrepancy model:

\[
r_\ell(\tilde{u}, t).
\]

For a GP discrepancy model, the output should include:

- the fitted or inferred discrepancy amplitude;
- length-scale information;
- measurement-fidelity dependence;
- uncertainty in the discrepancy;
- and any warnings that the discrepancy model is absorbing too much structure.

The discrepancy model should not be interpreted as purely random noise. It represents systematic simulator--measurement mismatch.

### 6.4 Noise-model state

Step 5 should return the updated noise model:

\[
\sigma_J(u, \ell, t)
\]

or its approximation.

The noise model should state whether it is:

- fixed from Step-4 measurement uncertainties;
- homoskedastic;
- heteroscedastic in configuration or frequency;
- fidelity-dependent;
- or inflated due to poor measurement quality.

Replicated measurements should be used to check whether the Step-4 uncertainty estimates are realistic.

### 6.5 Drift state

Step 5 should return a drift summary, for example:

- no evidence of drift;
- constant offset between baseline repeats;
- slow trend over time;
- abrupt shift after a hardware move;
- or uncertain drift requiring re-baselining.

A simple drift model is acceptable at first, but Step 5 should not silently assume that all data remain equally current if baseline repeats indicate time dependence.

### 6.6 Correctable versus diagnostic classification

For each inferred error mode, Step 5 should output:

\[
\theta_j \in \theta_{\mathrm{corr}}
\]

or:

\[
\theta_j \in \theta_{\mathrm{diag}}.
\]

This classification should be based on whether the current online control basis can actually compensate that error.

For example:

- a global z-offset may be correctable;
- a smooth stack-compression mode may be correctable if included in \(B\);
- a localized single-disk error may be diagnostic-only if the online basis cannot move that disk independently;
- a material-loss parameter may be diagnostic-only if it cannot be changed during calibration;
- a parasitic-mode indicator may be diagnostic-only unless there is a known actuator move that avoids it.

### 6.7 Diagnostics and warnings

Step 5 should output diagnostics such as:

- poor fit to the latest measurement;
- posterior multimodality;
- strong \(\theta\)-discrepancy confounding;
- excessive discrepancy amplitude;
- noise larger than expected;
- suspected drift;
- low information content in recent measurements;
- posterior sensitivity to priors;
- and uncorrectable errors dominating the predicted performance loss.

These diagnostics are essential because the downstream optimizer should not treat a poorly identified model as high-confidence truth.

---

## 7. Detector-state parameter model

The detector-state vector should be divided into physically meaningful groups.

A possible high-level structure is:

\[
\theta =
(\theta_{\mathrm{geom}},
 \theta_{\mathrm{antenna}},
 \theta_{\mathrm{mirror}},
 \theta_{\mathrm{loss}},
 \theta_{\mathrm{readback}},
 \theta_{\mathrm{misc}}).
\]

The exact content should be agreed with the experimental team. The purpose of the structure is to avoid a single undifferentiated nuisance vector.

### 7.1 Geometry-related parameters

Geometry-related parameters may include:

- global booster z-offset;
- global stack expansion or compression;
- smooth disk-position modes;
- relative mirror offset;
- relative focusing-mirror offset;
- and low-dimensional geometry modes not captured by the commanded variables.

These parameters are likely to be the most directly relevant to calibration.

### 7.2 Antenna-related parameters

Antenna-related parameters may include:

- residual antenna x-offset after Step 3;
- residual antenna y-offset after Step 3;
- uncertainty in the achieved antenna position;
- coupling-efficiency scale factors;
- or orientation-related nuisance parameters if needed.

Because Step 3 already aligns the antenna, Step 5 should usually infer only residual or diagnostic antenna effects, not re-run the antenna optimizer.

### 7.3 Mirror and focusing parameters

Mirror and focusing parameters may include:

- reflecting-mirror z-offset;
- focusing-mirror z-offset;
- effective focal-position error;
- tilt-like effective parameters;
- or other low-dimensional optical coupling errors.

Some of these may be correctable if they correspond to available actuators. Others may only inform the model.

### 7.4 Loss and material parameters

Loss and material parameters may include:

- effective dielectric losses;
- effective reflectivity or receiver-chain losses;
- frequency-dependent attenuation;
- or global scale factors affecting the measured boost-factor amplitude.

These parameters may explain performance degradation but are not necessarily correctable online.

### 7.5 Readback and actuator parameters

Readback and actuator parameters may include:

- achieved-versus-commanded offsets;
- hysteresis-related bias;
- creep-like relaxation after moves;
- or uncertainty in the position readback.

These parameters are important because Step 5 should model the map from achieved geometry to response, not only the map from commands to response.

---

## 8. Correctable and diagnostic variables

The parent proposal requires consistency between the inference model and the control basis.

Step 5 should therefore maintain a parameter table with at least four columns:

| Parameter group | Example | Correctable by online control? | How it affects the loop |
|---|---|---:|---|
| Global z-offset | Booster shifted along beam axis | Yes, if global z actuator exists | Can inform future \(u_B\) proposals |
| Stack compression | Smooth expansion/compression of disk stack | Yes, if included in \(B\) | Can inform disk-mode correction |
| Smooth disk mode | Low-order mode in \(B\) | Yes | Can inform disk-mode correction |
| Single localized disk offset | One disk position anomalous | Only if individual correction is allowed | Otherwise diagnostic-only |
| Mirror/focusing offset | Mirror or focusing element displaced | Yes, if actuator exists | Can inform future booster-state correction |
| Disk tilt or deformation | Non-1D imperfection | Usually no in the first version | Explains loss or coupling degradation |
| Effective loss | Increased loss or attenuation | Usually no | Explains lower amplitude; affects uncertainty |
| Parasitic-mode indicator | Unstable or unwanted mode | Maybe avoidable, not directly correctable | Can inform learned constraints or risk flags |

The rule is:

> The inference model may diagnose more errors than the controller can correct, but Step 5 must label uncorrectable errors explicitly.

This prevents the optimizer from chasing a correction that is outside the allowed control space.

---

## 9. Discrepancy model

The discrepancy term captures systematic mismatch between simulator and measured detector response:

\[
r_\ell(\tilde{u}, t).
\]

It should be present while \(\theta\) is inferred.

### 9.1 Why discrepancy is needed

The fast simulator may not capture all effects of the real detector, for example:

- small three-dimensional effects;
- unmodeled losses;
- antenna-coupling imperfections;
- disk imperfections;
- mirror or focusing imperfections;
- parasitic modes;
- or residual receiver-chain effects.

If discrepancy is ignored, the inference model may force these effects into \(\theta\), making the inferred detector state physically misleading.

### 9.2 GP discrepancy

A natural first discrepancy model is a Gaussian-process model:

\[
r_\ell(\tilde{u}, t) \sim \mathcal{GP}(m_r, k_r).
\]

At the scalar-objective level:

\[
r_J(\tilde{u}, t) \sim \mathcal{GP}.
\]

At the curve-summary level, each summary can have its own discrepancy model or a shared multi-output discrepancy model.

At the curve level, the discrepancy may depend on frequency:

\[
r_\beta(\nu, \tilde{u}, t).
\]

The first implementation should keep the discrepancy model simple enough to be identifiable with the available data.

### 9.3 Informative discrepancy priors

The discrepancy prior should be informative enough to prevent the discrepancy from absorbing all physical parameter effects.

Useful prior restrictions may include:

- amplitude scale based on expected simulator inadequacy;
- length scales that prevent extremely rapid variation unless justified;
- fidelity-specific discrepancy scales;
- smoothness assumptions over small geometry changes;
- and physical sign or scale expectations where justified.

These are not arbitrary regularization choices. They are necessary because \(\theta\) and \(r\) are not automatically identifiable from the data.

### 9.4 Discrepancy warning signs

Step 5 should warn the user if:

- the discrepancy amplitude is much larger than expected;
- the discrepancy length scale becomes unrealistically short;
- \(\theta\) changes strongly when the discrepancy prior changes;
- the discrepancy explains nearly all variation in \(J\);
- or the discrepancy is strongly correlated with physically important \(\theta\)-parameters.

These warnings should be propagated to Step 6 so the optimizer can avoid overconfident predictions.

---

## 10. Measurement-noise model

Step 5 should update the measurement-noise model rather than treat all observations as equally precise.

At the scalar-objective level, the likelihood may be written as:

\[
J_i \mid \theta, r, \sigma_i
\sim
\mathcal{N}\left(
J_{\mathrm{sim}}(\tilde{u}_i, \theta) + r_J(\tilde{u}_i, t_i),
\sigma_i^2
\right).
\]

If Step 4 provides reliable uncertainties, \(\sigma_i\) can be treated as known or fixed-noise input.

If the uncertainty is not reliable, Step 5 should infer an additional noise scale or noise-inflation factor:

\[
\sigma_{i,\mathrm{eff}}^2
=
\sigma_{i,\mathrm{Step4}}^2
+
\sigma_{\mathrm{extra}}^2.
\]

### 10.1 Heteroscedasticity

The noise may depend on:

- frequency;
- measurement fidelity;
- detector configuration;
- coupling quality;
- drift state;
- or whether the measurement was exploratory or confirmatory.

Therefore the model should allow for heteroscedasticity if replicated measurements show that the noise level is not constant.

### 10.2 Replication and baseline checks

Replicated measurements are valuable because they separate measurement noise from real detector response changes.

Step 5 should use repeats of:

- the baseline configuration;
- the incumbent best configuration;
- and selected new candidates

in order to check whether the stated \(\sigma_i\) values are realistic.

### 10.3 Robustness to outliers

A first implementation can use Gaussian measurement noise. If Step 4 quality flags or repeated measurements show occasional outliers, Step 5 can use a robust likelihood, for example a Student-t-like likelihood.

This should be optional and justified by data, not added automatically.

---

## 11. Drift model

The detector state may slowly change over a multi-hour calibration run.

Step 5 should therefore include at least a simple drift representation.

### 11.1 Minimal drift handling

The minimal approach is:

1. record measurement time or run index;
2. compare repeated baseline or incumbent measurements;
3. inflate uncertainty if drift is visible;
4. mark stale data if the detector state has changed appreciably.

This minimal approach is sufficient for the first implementation.

### 11.2 Simple parametric drift

A slightly richer model is:

\[
\theta_t = \theta_0 + b t,
\]

or:

\[
J_i = J_{\mathrm{sim}}(\tilde{u}_i, \theta) + r(\tilde{u}_i) + d(t_i) + \epsilon_i,
\]

where \(d(t)\) is a slow time-dependent offset.

### 11.3 Random-walk or GP drift

If the data justify it, drift can be represented as:

\[
\theta_{t+1} = \theta_t + \eta_t,
\]

or as a smooth GP in time.

This should not be the default unless baseline repeats show that drift is large enough to matter.

### 11.4 Drift warnings

Step 5 should issue a warning if:

- the baseline has shifted beyond measurement uncertainty;
- the incumbent best configuration no longer reproduces its earlier value;
- the inferred drift is comparable to expected calibration improvements;
- or the model cannot distinguish drift from geometry-dependent discrepancy.

Such warnings should affect Step 7 stopping decisions and Step 1 future acquisition behavior.

---

## 12. Multi-fidelity data handling inside Step 5

Step 5 should support multiple measurement fidelities, but it should not blindly pool them.

The high-fidelity target is the gradient-method boost-factor measurement. Lower-fidelity observables may include RF-response proxies, reflectivity-based quantities, antenna-coupling proxies, or other cheaper measurements.

The model should keep the fidelity label:

\[
\ell_i.
\]

A simple multi-fidelity observation model is:

\[
y_{i,\ell}
=
S_\ell(\tilde{u}_i, \theta)
+
r_\ell(\tilde{u}_i)
+
\epsilon_{i,\ell}.
\]

This allows different fidelities to have:

- different simulator outputs;
- different discrepancies;
- different noise levels;
- and different costs.

The key rule is:

> Lower-fidelity data may inform the joint detector-state and discrepancy model, but it must not be treated as identical to high-fidelity boost-factor data.

If the relationship between a proxy observable and the high-fidelity objective is weak or unvalidated, Step 5 should mark that proxy as low-confidence for inference.

---

## 13. Recommended inference levels

Step 5 can be implemented in increasing levels of sophistication.

### 13.1 Level A: joint MAP with uncertainty approximation

This is the recommended first implementation.

The model estimates:

\[
(\hat{\theta}, \hat{r}, \hat{\sigma}, \widehat{\text{drift}})
=
\arg\max
p(\theta, r, \sigma, \text{drift} \mid D).
\]

Important: this is still a **joint** fit. The discrepancy term is included while \(\theta\) is estimated.

After the joint MAP fit, approximate uncertainty can be obtained from:

- local curvature;
- bootstrap over measurements;
- repeated baseline measurements;
- or a simplified Laplace approximation.

This level is practical for online calibration.

### 13.2 Level B: partial Bayesian inference

The next level keeps posterior uncertainty for the most important quantities.

For example:

- sample \(\theta_{\mathrm{corr}}\) and key discrepancy hyperparameters;
- keep minor nuisance parameters at MAP values;
- use fixed measurement variances from Step 4;
- and propagate uncertainty into Step 6.

This level is a compromise between computational cost and statistical honesty.

### 13.3 Level C: full Bayesian inference

The most complete level samples from:

\[
p(\theta, r, \sigma, \text{drift} \mid D).
\]

This may use MCMC, HMC/NUTS, or another Bayesian sampling method.

This level is most appropriate for:

- offline validation;
- final uncertainty reporting;
- prior-sensitivity studies;
- and testing whether the simpler online approximation is reliable.

It may be too expensive for every online iteration, depending on the simulator and model complexity.

---

## 14. Practical Step-5 update routine

A practical Step-5 routine can be organized as follows.

### 14.1 Validate and ingest the latest measurement

For the new measurement from Step 4:

1. check that achieved geometry is present;
2. check that the measurement fidelity is recorded;
3. check that uncertainty information is available;
4. check quality flags;
5. decide whether the measurement is usable, usable with inflated uncertainty, or excluded from inference;
6. append the accepted record to the calibration data set.

Excluded records should not be deleted. They should remain in the log with a clear reason for exclusion.

### 14.2 Select the inference resolution

Decide whether the current update uses:

- scalar-objective inference;
- curve-summary inference;
- or curve-level inference.

The default should be scalar-objective inference unless there is enough reliable curve data and a validated curve-level model.

### 14.3 Update the joint model

Fit or sample the joint model:

\[
p(\theta, r, \sigma, \text{drift} \mid D_{1:t+1}).
\]

The update must include the discrepancy term during \(\theta\)-inference.

### 14.4 Run posterior checks

Check whether the model explains the data:

- compare predicted and measured values at observed configurations;
- examine residuals versus configuration, time, and measurement fidelity;
- check whether residuals are consistent with the noise model;
- check whether baseline repeats are consistent;
- and check whether new data are surprising under the previous posterior.

### 14.5 Classify inferred parameters

Update the classification:

\[
\theta = (\theta_{\mathrm{corr}}, \theta_{\mathrm{diag}}).
\]

If a parameter is physically meaningful but not controllable with the current basis, mark it diagnostic-only.

### 14.6 Record uncertainty and identifiability diagnostics

For each important parameter, record:

- posterior uncertainty;
- posterior correlation with discrepancy;
- sensitivity to discrepancy prior;
- sensitivity to included data fidelities;
- and whether the parameter should be trusted for physical interpretation.

### 14.7 Produce Step-6 handoff package

The final Step-5 output should include everything Step 6 needs to construct:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t+1}).
\]

This handoff should include posterior samples or an approximation, not only a point estimate.

---

## 15. Priors and identifiability safeguards

Step 5 should be conservative about physical interpretation.

### 15.1 Informative priors are necessary

Because \(\theta\) and \(r\) can be confounded, informative priors are not optional. They encode experimental knowledge needed to separate physical detector-state effects from simulator inadequacy.

Priors should be attached to:

- geometry offsets;
- actuator/readback error;
- loss parameters;
- discrepancy amplitude;
- discrepancy smoothness;
- drift amplitude;
- and measurement-noise inflation.

### 15.2 Prior-sensitivity checks

Step 5 should periodically test whether important conclusions depend strongly on prior choices.

For example:

- If \(\theta_{\mathrm{corr}}\) changes drastically when the discrepancy amplitude prior is widened, \(\theta_{\mathrm{corr}}\) is weakly identifiable.
- If the discrepancy term absorbs almost all variation, the simulator may be insufficiently informative in the current search region.
- If the posterior remains close to the prior after many measurements, the data may not be informative for that parameter.

### 15.3 Multiple data types improve identifiability

Where available, Step 5 should use multiple observables because they constrain \(\theta\) and discrepancy differently.

For example, a boost-factor scalar alone may not distinguish a frequency shift from an amplitude loss. Curve summaries, reflectivity proxies, antenna-coupling proxies, and repeated geometry measurements can help break such degeneracies.

This does not mean every observable must be used immediately. It means the data model should be designed so that additional observables can be incorporated without changing the conceptual structure.

---

## 16. Diagnostics that Step 5 should expose

Step 5 should not only output estimates. It should also output model-health diagnostics.

Recommended diagnostics include:

### 16.1 Fit diagnostics

- predicted versus measured \(J\) at observed configurations;
- standardized residuals;
- residuals versus time;
- residuals versus fidelity;
- residuals versus achieved geometry;
- and baseline-repeat consistency.

### 16.2 Posterior diagnostics

- posterior uncertainty for each important parameter;
- posterior correlations between \(\theta\) and \(r\);
- multimodality indicators;
- convergence diagnostics if MCMC is used;
- effective sample-size and chain-mixing summaries if sampling is used;
- and warnings for parameters pinned to prior or hardware bounds.

### 16.3 Discrepancy diagnostics

- discrepancy amplitude relative to expected simulator error;
- discrepancy amplitude relative to observed improvement in \(J\);
- discrepancy length scale relative to control-space scale;
- discrepancy changes across fidelities;
- and regions where discrepancy dominates simulator prediction.

### 16.4 Noise diagnostics

- empirical repeat variance versus Step-4 uncertainty estimates;
- configuration-dependent noise evidence;
- fidelity-dependent noise evidence;
- outlier rate;
- and whether expected improvements are resolvable above noise.

### 16.5 Drift diagnostics

- baseline shift over time;
- incumbent-best reproducibility;
- drift magnitude relative to expected improvement;
- and whether old data should be downweighted or rechecked.

---

## 17. Step-5 failure modes and responses

Step 5 should detect failure modes explicitly.

### 17.1 Failure mode: \(\theta\)-discrepancy confounding

**Symptom:** The inferred physical parameters change strongly when discrepancy priors are changed, or \(\theta\) is highly correlated with the discrepancy function.

**Response:** Mark the affected parameters as weakly identifiable. Do not use them as strong physical claims. Pass inflated uncertainty to Step 6.

### 17.2 Failure mode: discrepancy absorbs everything

**Symptom:** The discrepancy model explains most of the observed variation, while the simulator contributes little.

**Response:** Simplify the search region, add more informative measurements, tighten physically justified discrepancy priors, or reduce physical interpretation of \(\theta\).

### 17.3 Failure mode: noise larger than improvement

**Symptom:** Replicated measurements show that \(\sigma_J\) is comparable to or larger than expected calibration gains.

**Response:** Recommend replication, use cheaper information-gathering measurements, reduce dimensionality, or trigger a Step-7 stopping condition.

### 17.4 Failure mode: uncorrectable error dominates

**Symptom:** The model explains performance loss through \(\theta_{\mathrm{diag}}\), not \(\theta_{\mathrm{corr}}\).

**Response:** Inform Step 6 and Step 7 that further online correction may be limited. The correct action may be diagnostic reporting or hardware intervention outside the online calibration basis.

### 17.5 Failure mode: drift invalidates older data

**Symptom:** Baseline or incumbent repeats shift beyond expected noise.

**Response:** Add drift uncertainty, downweight stale data if justified, request re-baselining, or pause exploitation until the detector state is stable enough.

### 17.6 Failure mode: measurement-quality region is unreliable

**Symptom:** Certain configurations produce failed or low-quality Step-4 measurements.

**Response:** Mark these regions as measurement-quality constraints for Step 1, but keep damage-relevant constraints separate from learned constraints.

---

## 18. Suggested first implementation

The first implementation of Step 5 should be deliberately simple but statistically correct.

A good first version is:

```text
Input:
    scalar high-fidelity objective measurements J_i
    Step-4 uncertainty estimates sigma_i
    achieved geometry u_tilde_i
    measurement fidelity labels ell_i
    baseline and incumbent repeat tags
    fast simulator J_sim(u_tilde_i, theta)
    physically informed priors

Model:
    J_i = J_sim(u_tilde_i, theta) + r_ell(u_tilde_i) + d(t_i) + epsilon_i
    epsilon_i ~ Normal(0, sigma_eff_i^2)
    r_ell is a simple GP discrepancy or low-dimensional smooth discrepancy term
    d(t_i) is absent or a simple slow drift term in the first version

Inference:
    joint MAP for theta, discrepancy hyperparameters, noise inflation, and optional drift
    uncertainty approximation from local curvature, bootstrap, or posterior samples for selected variables

Output:
    joint model state
    theta_corr / theta_diag classification
    uncertainty and identifiability diagnostics
    Step-6 handoff package
```

The key feature is not the exact inference engine. The key feature is that \(\theta\), discrepancy, noise, and drift are updated together.

---

## 19. Suggested later upgrades

After the first implementation works, Step 5 can be upgraded in stages.

### 19.1 Add curve-summary inference

Instead of using only \(J_i\), include summaries such as:

- peak boost;
- peak frequency;
- bandwidth;
- band flatness;
- and scan-rate proxy.

This can help distinguish different physical causes of performance changes.

### 19.2 Add richer multi-fidelity modeling

Lower-fidelity measurements can be modeled as related but biased observations of the high-fidelity target. The relationship should be learned or physically specified, not assumed perfect.

### 19.3 Add partial posterior sampling

Use posterior samples for the most important uncertain quantities, especially \(\theta_{\mathrm{corr}}\), discrepancy amplitude, noise inflation, and drift.

### 19.4 Add time-dependent detector state

If baseline repeats show relevant drift, replace the static \(\theta\) assumption with a slowly time-dependent model.

### 19.5 Add stronger identifiability checks

Use prior sensitivity, synthetic injection tests, and held-out configurations to test whether inferred physical parameters are trustworthy.

---

## 20. Validation tests for Step 5

Before using Step 5 in a real closed-loop calibration, validate it on controlled data.

### 20.1 Synthetic recovery test

Generate synthetic measurements with known \(\theta\), discrepancy, noise, and drift. Check whether Step 5 can recover the correct posterior uncertainty and avoid overconfident wrong estimates.

### 20.2 Confounding test

Generate data where a physical parameter and discrepancy can explain the same effect. Check whether Step 5 reports weak identifiability rather than making an overconfident physical claim.

### 20.3 Noise test

Generate repeated measurements with known heteroscedastic noise. Check whether Step 5 estimates or respects the noise levels correctly.

### 20.4 Drift test

Generate a slowly drifting baseline. Check whether Step 5 detects the drift and prevents stale measurements from producing overconfident predictions.

### 20.5 Correctable-versus-diagnostic test

Generate an error that is inferable but not spanned by the control basis. Check whether Step 5 labels it diagnostic-only.

### 20.6 Multi-fidelity consistency test

Generate low- and high-fidelity measurements with a known bias between them. Check whether Step 5 learns the relationship without treating the proxy as exact high-fidelity data.

---

## 21. Step-5 handoff to Step 6

Step 5 should hand off a complete model-state object to Step 6.

The handoff should contain:

1. accepted data records;
2. excluded data records with reasons;
3. posterior approximation or samples;
4. detector-state summaries;
5. discrepancy-model state;
6. noise-model state;
7. drift-model state;
8. correctable/diagnostic labels;
9. identifiability warnings;
10. model-health diagnostics;
11. and recommended uncertainty inflation if the model is weakly identified.

Step 6 will use this handoff to construct:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t+1}).
\]

Step 5 should not reduce everything to a single point estimate. If a point estimate is used for computational reasons, uncertainty and identifiability warnings must still be carried forward.

---

## 22. Minimal data schema

A practical Step-5 data schema should contain one table of measurement records and one table of model-state summaries.

### 22.1 Measurement-record fields

| Field | Meaning |
|---|---|
| `iteration` | Outer-loop iteration index |
| `candidate_id` | Unique candidate identifier |
| `time` | Measurement time or run index |
| `fidelity` | Measurement type \(\ell\) |
| `u_B_cmd` | Commanded booster correction |
| `u_B_achieved` | Achieved booster correction/readback |
| `u_A_cmd` | Commanded antenna position |
| `u_A_achieved` | Achieved antenna position/readback |
| `observable_raw_ref` | Reference to raw Step-4 data |
| `observable_processed` | Processed measurement or summaries |
| `J` | Scalar objective if available |
| `sigma_J` | Objective uncertainty if available |
| `quality_flags` | Measurement-quality flags |
| `replicate_group` | Identifier for repeated measurements |
| `baseline_or_incumbent_tag` | Whether this is a baseline/incumbent repeat |

### 22.2 Model-state summary fields

| Field | Meaning |
|---|---|
| `posterior_type` | MAP, Laplace, variational, MCMC, hybrid |
| `theta_summary` | Detector-state parameter summaries |
| `theta_corr` | Correctable inferred parameters |
| `theta_diag` | Diagnostic-only inferred parameters |
| `discrepancy_state` | Discrepancy model and hyperparameters |
| `noise_state` | Noise model and uncertainty inflation |
| `drift_state` | Drift model or drift flags |
| `identifiability_flags` | Confounding and prior-sensitivity warnings |
| `fit_diagnostics` | Residual and posterior diagnostics |
| `step6_ready` | Whether model is valid for Step-6 prediction |

---

## 23. Acceptance criteria for Step 5

Step 5 should be considered successful at an iteration if it can produce the following:

1. an accepted and traceable measurement data set;
2. a joint inference update that includes discrepancy while inferring \(\theta\);
3. an uncertainty estimate for the important inferred quantities;
4. an updated noise model or confirmation that Step-4 uncertainties are adequate;
5. a drift assessment;
6. correctable-versus-diagnostic classification;
7. identifiability warnings where needed;
8. and a complete handoff package for Step 6.

If any of these are missing, Step 5 should still produce a record, but Step 6 and Step 1 should treat the resulting predictive model as degraded.

---

## 24. What should not be overclaimed

Step 5 should not claim that:

- \(\theta\) is uniquely identified without informative priors and diagnostics;
- discrepancy is merely random noise;
- a joint MAP estimate is equivalent to a full posterior;
- low-fidelity proxy data are equivalent to high-fidelity boost-factor measurements;
- commanded geometry is sufficient when achieved readback is available;
- uncorrectable inferred errors can be corrected by the online optimizer;
- or old measurements remain valid if baseline repeats show drift.

The correct interpretation is more careful:

> Step 5 produces the best current joint statistical explanation of the calibration data, with explicit uncertainty and diagnostics, so that Step 6 can build an honest posterior predictive model for the next optimization decision.

---

## 25. Source anchors

This Step-5 design is based on the locked parent proposal and the following source anchors:

1. **Parent closed-loop calibration proposal:** Defines Step 5 as the joint update of detector-state, discrepancy, noise, and drift inference.  
   `madmax_closed_loop_calibration_proposal.md`

2. **Kennedy and O'Hagan Bayesian calibration:** Introduces Bayesian calibration of computer models with parameter uncertainty and a discrepancy term.  
   <https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/KennedyOHagan_2002.pdf>

3. **Brynjarsdóttir and O'Hagan on model discrepancy:** Shows that ignoring model discrepancy can lead to biased and overconfident physical-parameter inference.  
   <https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf>

4. **Tuo and Wu on calibration of imperfect computer models:** Discusses theoretical issues and calibration behavior for imperfect simulators.  
   <https://projecteuclid.org/journals/annals-of-statistics/volume-43/issue-6/Efficient-calibration-for-imperfect-computer-models/10.1214/15-AOS1314.pdf>

5. **Plumlee on calibration consistency:** Discusses problems caused by discrepancy-prior effects and parameter consistency in computer-model calibration.  
   <https://academic.oup.com/jrsssb/article/81/3/519/7048342>

6. **BoTorch model documentation:** Provides current terminology for homoskedastic, fixed-noise, and heteroskedastic noise modeling in GP/BO contexts.  
   <https://botorch.org/docs/models>

7. **PyMC NUTS documentation:** Example of a modern Bayesian inference engine for posterior sampling of continuous variables.  
   <https://www.pymc.io/projects/docs/en/v5.9.1/api/generated/pymc.NUTS.html>

8. **ArviZ diagnostics:** Provides standard posterior-sampling diagnostics such as R-hat and effective sample size.  
   <https://python.arviz.org/en/latest/api/generated/arviz.rhat.html>

9. **MADMAX prototype and positioning context:** MADMAX-related sources motivate the need to use achieved geometry, account for mechanical stability, and treat boost-factor uncertainty carefully.  
   <https://scoap3-prod-backend.s3.cern.ch/media/files/54293/10.1140/epjc/s10052-020-7985-8_a.pdf>  
   <https://arxiv.org/abs/2305.12808>

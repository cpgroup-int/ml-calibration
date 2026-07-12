# MADMAX Closed-Loop Calibration
# Step 6 Technical Design: Update the Optimizer-Facing Predictive Model

**Status:** Step-specific technical design note  
**Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3  
**Scope:** This document expands only **Step 6** of the seven-step closed-loop calibration algorithm. It describes how the output of Step 5 should be converted into the predictive object used by Step 1. It does **not** choose the next detector move, define the final Step-1 acquisition function, move hardware, perform antenna alignment, or redo the Step-5 inference.

---

## 1. Role of Step 6 in the full calibration loop

Step 6 is the **prediction-interface step** of the closed-loop calibration algorithm.

Before Step 6 begins, the loop has already done the following:

1. **Step 1** proposed a safe booster-state correction and selected a measurement action or fidelity.
2. **Step 2** set the booster geometry and recorded the achieved geometry.
3. **Step 3** aligned the antenna for that booster state and recorded the achieved antenna position.
4. **Step 4** measured the selected observable and, when required, measured the boost factor using the high-fidelity gradient method.
5. **Step 5** updated the joint calibration model:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}).
\]

Step 6 takes this joint calibration result and converts it into the optimizer-facing posterior predictive model:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t+1}).
\]

This posterior predictive model is the object used by the next iteration of Step 1.

In words, Step 6 answers:

> Given the current joint detector-state, discrepancy, noise, and drift model, what should the optimizer believe about the high-fidelity objective at future candidate configurations?

The essential point is that Step 6 is not merely a storage or plotting step. It is the step that makes the result of the statistical calibration usable by the Bayesian optimizer.

---

## 2. What Step 6 is and is not

Step 6 is responsible for:

- constructing predictions for future candidate detector states;
- propagating uncertainty from detector-state parameters, discrepancy, noise, and drift;
- distinguishing latent objective uncertainty from future measurement noise;
- exposing predictions for high-fidelity and lower-fidelity measurements if a measurement hierarchy is used;
- exposing learned soft-constraint predictions if they exist;
- preserving the distinction between correctable and diagnostic-only inferred errors;
- packaging the prediction object for Step 1;
- and validating whether the predictive model is numerically and statistically usable.

Step 6 is not responsible for:

- proposing the next booster correction;
- optimizing the acquisition function;
- moving disks, mirrors, or antenna hardware;
- measuring the boost factor;
- re-fitting the full Step-5 joint posterior from raw data;
- deciding the final operating point on the peak--bandwidth trade-off;
- or declaring the calibration finished.

The clean boundary is:

\[
\boxed{\text{Step 5: infer the joint calibration state}}
\]

\[
\boxed{\text{Step 6: convert that state into posterior predictions for Step 1}}
\]

\[
\boxed{\text{Step 1: use those predictions to choose the next action}}
\]

---

## 3. Main input to Step 6

The main input is the Step-5 joint calibration state:

\[
\mathcal{M}_{t+1}
=
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}).
\]

Here:

- \(\theta_t\) are detector-state or nuisance parameters;
- \(r_\ell\) is the simulation--measurement discrepancy for measurement fidelity \(\ell\);
- \(\sigma_J\) is the measurement-noise model for the scalar objective or reduced observables;
- and `drift` represents slow time dependence, baseline drift, or achieved-geometry evolution.

Step 6 also receives:

- the fast physics simulator

\[
\beta^2_{\mathrm{sim}}(\nu; q, \theta),
\]

- the current calibration data set \(D_{1:t+1}\), including achieved geometries;
- the definition of the objective \(J\) or objective vector \(\mathbf{J}\);
- the available measurement fidelities \(\ell\);
- the selected online control basis for booster corrections;
- known hard feasibility rules, already represented as a candidate-domain filter;
- learned soft-constraint models, if Step 5 maintains them;
- and the antenna-alignment convention from Step 3.

The last item matters because Step 1 proposes a booster state, but the objective is evaluated after Step 3 has aligned the antenna. Therefore, Step 6 should make clear whether its prediction is for:

\[
J(q_0,u_B,u_A)
\]

at a specified antenna position, or for the antenna-aligned booster-level objective:

\[
F(u_B)
=
\max_{u_A} J(q_0,u_B,u_A).
\]

For the outer-loop optimizer, the preferred Step-6 target is usually the antenna-aligned quantity \(F(u_B)\), possibly with uncertainty from the antenna-alignment step included.

---

## 4. Main output of Step 6

The output of Step 6 should be an **optimizer-facing predictive model object**.

Conceptually, this object should provide the following query:

\[
(u,\ell,t_{\mathrm{future}})
\longmapsto
p(Y_\ell(u,t_{\mathrm{future}}) \mid D_{1:t+1}),
\]

where:

- \(u\) is a future candidate state or correction;
- \(\ell\) is the requested measurement fidelity;
- \(t_{\mathrm{future}}\) is the expected time or iteration of the future measurement;
- and \(Y_\ell\) is the predicted observable at that fidelity.

For high-fidelity boost-factor validation, the key output is:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t+1}).
\]

The predictive model should expose at least:

\[
\mu_{\mathrm{pred}}(u),
\qquad
\sigma_{\mathrm{pred}}(u),
\qquad
\text{credible interval}(u),
\qquad
\text{posterior samples if available}.
\]

If the calibration is formulated as a multi-objective problem, Step 6 should instead expose:

\[
p(\mathbf{J}_{\mathrm{HF}}(u) \mid D_{1:t+1}),
\]

where \(\mathbf{J}\) may include, for example:

\[
\mathbf{J}
=
(\text{peak boost},\ \text{bandwidth},\ \text{scan-rate proxy},\ \text{risk/cost penalty}).
\]

The predictive model should then provide a mean vector, covariance matrix, and posterior samples of the objective vector.

---

## 5. Candidate state representation

Step 6 should use a candidate representation compatible with the outer-loop optimizer.

A candidate should include, at minimum:

\[
u
=
(u_B,\ell,t_{\mathrm{future}}),
\]

where:

\[
u_B
=
(a_{\mathrm{disk}}, z_{\mathrm{global}}, z_{\mathrm{reflecting\ mirror}}, z_{\mathrm{focusing\ mirror}}).
\]

If Step 6 predicts the full detector configuration directly, then the candidate may also include:

\[
u_A=(x_{\mathrm{ant}},y_{\mathrm{ant}}).
\]

However, for the outer-loop Step-1 proposal, it is usually cleaner to predict the antenna-aligned objective:

\[
F_{\mathrm{HF}}(u_B)
=
J_{\mathrm{HF}}(q_0,u_B,u_A^\star(u_B)).
\]

In that case, Step 6 should document which approximation is being used for \(u_A^\star(u_B)\):

1. **Plug-in antenna optimum:** use the best antenna position predicted by the current antenna-alignment model.
2. **Marginalized antenna alignment:** integrate over uncertainty in the antenna-alignment result.
3. **Penalty-aware alignment:** subtract an alignment-quality penalty if the antenna optimum is uncertain or near a hardware boundary.
4. **Specified antenna position:** only for diagnostics or special tests, predict at a fixed \(u_A\).

A practical first version can use a plug-in antenna optimum, but it should record that the resulting uncertainty does not fully include antenna-alignment uncertainty. A stronger version should propagate antenna-alignment uncertainty into the predicted high-fidelity objective.

---

## 6. Achieved geometry versus commanded geometry

The parent proposal requires the optimizer to use achieved geometry whenever readback is available, rather than only commanded motor positions. Step 6 must preserve this distinction.

Existing data should be represented as:

\[
D_i =
( u_{\mathrm{cmd},i},\ \tilde{u}_{\mathrm{achieved},i},\ t_i,\ \ell_i,\ y_i,\ \sigma_i,\ \text{quality flags}_i ).
\]

For future candidates, Step 1 usually proposes a commanded state:

\[
u_{\mathrm{cmd}}.
\]

But the detector will actually realize an achieved state:

\[
\tilde{u}_{\mathrm{achieved}}.
\]

Therefore, Step 6 should define whether the predictive model is conditional on:

\[
\tilde{u}_{\mathrm{achieved}} = u_{\mathrm{cmd}},
\]

or whether it includes an achieved-geometry model:

\[
p(\tilde{u}_{\mathrm{achieved}} \mid u_{\mathrm{cmd}}, t, D).
\]

The more statistically honest prediction is:

\[
p(J_{\mathrm{HF}}(u_{\mathrm{cmd}}) \mid D)
=
\int
p(J_{\mathrm{HF}}(\tilde{u}) \mid \tilde{u},D)
\ p(\tilde{u} \mid u_{\mathrm{cmd}},t,D)
\ d\tilde{u}.
\]

In a first implementation, this may be approximated by using the expected achieved geometry:

\[
\tilde{u}=\mathbb{E}[\tilde{u}_{\mathrm{achieved}} \mid u_{\mathrm{cmd}},t,D].
\]

But the file should make clear whether achieved-geometry uncertainty is included or ignored.

---

## 7. Core posterior-predictive construction

The central predictive relation is:

\[
Y_\ell(u,t)
=
J_{\ell,\mathrm{sim}}(u,\theta_t)
+
r_\ell(u,t)
+
\epsilon_\ell.
\]

For the high-fidelity objective:

\[
Y_{\mathrm{HF}}(u,t)
=
J_{\mathrm{HF},\mathrm{sim}}(u,\theta_t)
+
r_{\mathrm{HF}}(u,t)
+
\epsilon_{\mathrm{HF}}.
\]

The latent high-fidelity objective is:

\[
J_{\mathrm{HF}}^{\mathrm{latent}}(u,t)
=
J_{\mathrm{HF},\mathrm{sim}}(u,\theta_t)
+
r_{\mathrm{HF}}(u,t).
\]

The future measured high-fidelity observation is:

\[
Y_{\mathrm{HF}}(u,t)
=
J_{\mathrm{HF}}^{\mathrm{latent}}(u,t)
+
\epsilon_{\mathrm{HF}}.
\]

This distinction is important:

- the **latent objective** tells the optimizer what the real detector performance is expected to be;
- the **future observation** tells the optimizer how noisy a future measurement will be.

Step 6 should expose both when possible:

\[
p(J_{\mathrm{HF}}^{\mathrm{latent}}(u) \mid D)
\]

and

\[
p(Y_{\mathrm{HF}}(u) \mid D).
\]

The acquisition function in Step 1 may use either one depending on whether it is optimizing true detector performance, deciding whether to replicate a measurement, or deciding which fidelity to use.

---

## 8. Sample-based construction from the Step-5 posterior

A robust way to construct the posterior predictive is sample-based.

Assume Step 5 provides posterior samples:

\[
\left\{
\theta^{(s)},
 r_\ell^{(s)},
 \sigma_J^{(s)},
 \text{drift}^{(s)}
\right\}_{s=1}^{S}.
\]

For each future candidate \(u\):

1. Convert the candidate into the relevant detector geometry:

\[
q^{\mathrm{cand}} = q(q_0,u_B,u_A).
\]

2. If using achieved-geometry uncertainty, sample or approximate:

\[
\tilde{u}^{(s)} \sim p(\tilde{u}_{\mathrm{achieved}} \mid u_{\mathrm{cmd}},t,D).
\]

3. Evaluate the simulator under the sampled detector state:

\[
\beta^{2,(s)}_{\mathrm{sim}}(\nu)
=
\beta^2_{\mathrm{sim}}(\nu;q^{\mathrm{cand}},\theta^{(s)}).
\]

4. Convert the simulated curve to the relevant objective:

\[
J^{(s)}_{\mathrm{sim}}
=
J(\beta^{2,(s)}_{\mathrm{sim}}(\nu)).
\]

5. Add the sampled discrepancy:

\[
J^{(s)}_{\mathrm{latent}}
=
J^{(s)}_{\mathrm{sim}}
+
r^{(s)}_{\mathrm{HF}}(u,t).
\]

6. If predicting the future measured observation, also add measurement noise:

\[
Y^{(s)}_{\mathrm{HF}}
=
J^{(s)}_{\mathrm{latent}}
+
\epsilon^{(s)}_{\mathrm{HF}},
\qquad
\epsilon^{(s)}_{\mathrm{HF}}\sim p(\epsilon\mid \sigma_J^{(s)},u,t).
\]

7. Summarize the sample cloud:

\[
\mu_{\mathrm{pred}}(u)
=
\frac{1}{S}\sum_s J^{(s)}_{\mathrm{latent}}(u),
\]

\[
\sigma^2_{\mathrm{pred}}(u)
=
\frac{1}{S-1}\sum_s
\left(J^{(s)}_{\mathrm{latent}}(u)-\mu_{\mathrm{pred}}(u)\right)^2.
\]

This sample-based procedure avoids replacing the Step-5 posterior by a single point estimate. It also makes it natural to propagate non-Gaussian uncertainty, skewed uncertainty, multi-objective uncertainty, and drift uncertainty.

---

## 9. Curve-level versus scalar-level prediction

Step 6 should decide whether it predicts:

1. the boost-factor curve first, then reduces it to an objective; or
2. the scalar objective directly.

### 9.1 Curve-level prediction

Curve-level prediction models:

\[
p(\beta^2_{\mathrm{HF}}(\nu;u) \mid D).
\]

Then the objective is computed from each posterior sample:

\[
J^{(s)}(u)=J(\beta^{2,(s)}(\nu;u)).
\]

This is preferable when:

- the objective may change later;
- the area-law trade-off is being studied;
- multi-objective calibration is being used;
- peak position, bandwidth, and flatness all matter;
- or the physics team wants diagnostic plots of the predicted boost-factor curve.

### 9.2 Scalar-level prediction

Scalar-level prediction models:

\[
p(J_{\mathrm{HF}}(u) \mid D).
\]

This is simpler and may be sufficient when:

- the scalar objective has already been fixed;
- the measurement budget is small;
- the calibration problem is not multi-objective;
- or Step 5 only outputs scalar reduced data.

### 9.3 Recommended first version

The recommended first version is:

\[
\boxed{\text{keep the curve-level simulator output, but train the optimizer-facing surrogate on scalar }J.}
\]

That means the simulator and diagnostics retain the full boost-factor curve, while Step 1 receives a compact scalar prediction. If the project later moves to multi-objective or area-law studies, Step 6 can expose a vector objective without changing the entire loop.

---

## 10. Handling the MADMAX area-law trade-off

The parent proposal emphasizes that the scalar objective is physically consequential because MADMAX has a peak--bandwidth trade-off. Step 6 should therefore avoid hiding the objective choice.

If a scalar objective has been selected, Step 6 should label it clearly, for example:

\[
J_{\mathrm{scan}}(u),
\qquad
J_{\mathrm{broadband}}(u),
\qquad
J_{\mathrm{narrow}}(u).
\]

If no single objective is fixed, Step 6 should expose a multi-objective predictive distribution:

\[
p(\mathbf{J}_{\mathrm{HF}}(u)\mid D),
\]

where possible components are:

- peak boost;
- effective bandwidth;
- band flatness;
- scan-rate proxy;
- robustness to mechanical errors;
- measurement cost;
- risk or failure probability.

Step 6 does not choose the operating point. It only provides the predicted trade-off surface or Pareto-relevant objective samples for Step 1 and the physics team.

---

## 11. Multi-fidelity prediction

The parent proposal allows Step 1 to choose not only a booster correction, but also a measurement action or fidelity. Therefore, Step 6 should support predictions for multiple fidelities when available.

Let \(\ell\) denote the measurement type or fidelity, for example:

\[
\ell \in
\{\text{simulator-only},\ \text{cheap RF proxy},\ \text{reflectivity-like observable},\ \text{gradient-method HF boost factor}\}.
\]

The high-fidelity target remains:

\[
J_{\mathrm{HF}}(u).
\]

Lower-fidelity observables should be modeled because they can reduce uncertainty about \(J_{\mathrm{HF}}\):

\[
p(Y_\ell(u),J_{\mathrm{HF}}(u)\mid D).
\]

Step 6 should expose, for each candidate and fidelity:

- predicted low-fidelity mean and uncertainty;
- predicted high-fidelity mean and uncertainty;
- correlation or information link between the low-fidelity observable and high-fidelity objective;
- expected measurement noise at that fidelity;
- and cost metadata for that measurement.

This is necessary for Step 1 to perform cost-aware or multi-fidelity acquisition. A lower-fidelity measurement is valuable only if it meaningfully updates beliefs about the high-fidelity objective.

---

## 12. Learned soft constraints and feasibility predictions

Known hard constraints are not learned in Step 6. They are enforced before a candidate is proposed to hardware.

However, Step 6 may expose predictions for learned soft constraints, for example:

- probability that the gradient-method boost-factor determination succeeds;
- probability of acceptable antenna coupling;
- probability of avoiding parasitic-mode contamination;
- probability that the measurement quality flags pass;
- probability that the local antenna alignment is well-conditioned;
- probability that drift will not invalidate the measurement during the estimated measurement duration.

For each learned constraint \(c_k\), Step 6 should expose:

\[
p(c_k(u) \leq 0 \mid D)
\]

or, more generally:

\[
P_{\mathrm{feas},k}(u).
\]

The combined learned feasibility may be summarized as:

\[
P_{\mathrm{feas}}(u)
=
P(c_1(u)\leq0,\dots,c_K(u)\leq0\mid D).
\]

Step 6 should also label which constraints are:

- **hard and exact**, handled outside the predictive model;
- **soft and learned**, handled through posterior feasibility;
- or **diagnostic flags**, reported but not directly used for candidate filtering.

This keeps the safety logic consistent with the parent proposal: damage-relevant constraints are never learned by trying unsafe points.

---

## 13. Noise prediction and uncertainty decomposition

Step 6 should report not only a total uncertainty, but also an uncertainty decomposition when possible.

For a candidate \(u\), the total predictive variance for a future observation can be decomposed conceptually as:

\[
\mathrm{Var}(Y_{\mathrm{HF}}(u)\mid D)
=
\mathrm{Var}_{\theta}[\cdot]
+
\mathrm{Var}_{r}[\cdot]
+
\mathrm{Var}_{\mathrm{drift}}[\cdot]
+
\mathrm{Var}_{\mathrm{achieved\ geometry}}[\cdot]
+
\mathrm{Var}_{\epsilon}[\cdot].
\]

The exact decomposition may not be unique if the terms are correlated, but even an approximate decomposition is useful diagnostically.

Step 6 should distinguish:

### 13.1 Epistemic uncertainty

This is uncertainty about the latent detector response due to limited calibration data:

\[
\mathrm{Var}(J_{\mathrm{HF}}^{\mathrm{latent}}(u)\mid D).
\]

This can be reduced by better measurements.

### 13.2 Aleatoric measurement noise

This is future measurement noise:

\[
\mathrm{Var}(\epsilon_{\mathrm{HF}}).
\]

This may be reduced by replication or longer measurement, but not by changing the statistical model alone.

### 13.3 Drift uncertainty

This is uncertainty from time dependence:

\[
\mathrm{Var}_{\mathrm{drift}}(u,t).
\]

This may be reduced by re-baselining, faster calibration, or time-aware modeling.

### 13.4 Control/achieved-geometry uncertainty

This is uncertainty from actuator hysteresis, creep, or finite readback precision:

\[
\mathrm{Var}_{\tilde{u}}(J(u)).
\]

This may be reduced by better readback, repeated moves, or using achieved geometry in the model.

Step 1 can use this decomposition to decide whether to exploit, explore, replicate, or re-baseline.

---

## 14. Preventing double counting of noise and discrepancy

Step 6 must be careful not to count the same uncertainty twice.

The recommended distinction is:

\[
\text{simulator uncertainty / detector-state uncertainty}
\rightarrow
\theta
\]

\[
\text{systematic simulator--measurement mismatch}
\rightarrow
r_\ell
\]

\[
\text{repeat-measurement scatter}
\rightarrow
\epsilon_\ell
\]

\[
\text{slow time dependence}
\rightarrow
\text{drift}
\]

The latent objective prediction should include \(\theta\), discrepancy, achieved-geometry uncertainty, and drift, but not necessarily the future measurement noise if the optimizer wants the true expected detector performance:

\[
p(J_{\mathrm{latent}}(u)\mid D).
\]

The observation prediction should additionally include measurement noise:

\[
p(Y(u)\mid D).
\]

If the same measured residual is used both to inflate discrepancy and to inflate noise, the posterior predictive uncertainty may become too conservative. If discrepancy is ignored and the residual is treated only as noise, the model may become overconfident in the wrong mean response.

---

## 15. Treatment of correctable and diagnostic-only errors

Step 5 classifies inferred detector-state variables into:

\[
\theta_{\mathrm{corr}}
\quad \text{and} \quad
\theta_{\mathrm{diag}}.
\]

Step 6 should preserve this classification.

Correctable parameters can influence future predictions in a way that the optimizer may exploit. For example, if the model infers a global stack offset and the booster-control variables include a global z correction, then Step 6 can predict the effect of compensating that offset.

Diagnostic-only parameters can influence predictions, but Step 6 should not represent them as directly correctable unless the control basis spans them.

For example, if Step 5 infers a localized disk error but the online disk-correction basis contains only smooth global modes, then Step 6 should propagate the resulting performance limitation into:

\[
p(J_{\mathrm{HF}}(u)\mid D),
\]

but should not create a fake control direction that cancels that local error.

A useful Step-6 output is therefore:

\[
\text{uncorrectable-loss estimate}(u),
\]

or a diagnostic note that part of the predicted performance limit is caused by inferred errors outside the control basis.

---

## 16. Time dependence and stale predictions

Because calibration may take long enough for mechanical or thermal drift to matter, Step 6 should timestamp predictions.

A prediction should be associated with an intended future time:

\[
t_{\mathrm{future}}.
\]

If no time-dependent model is available, Step 6 should at least provide a staleness flag based on elapsed time since:

- the most recent high-fidelity measurement;
- the most recent baseline measurement;
- the most recent antenna alignment;
- the most recent achieved-geometry readback;
- and the most recent Step-5 posterior update.

If a drift model is available, Step 6 should produce:

\[
p(J_{\mathrm{HF}}(u,t_{\mathrm{future}})\mid D).
\]

If drift uncertainty becomes large, the predictive model should signal that Step 1 should consider a re-baseline or incumbent remeasurement rather than a new exploratory move.

---

## 17. Candidate-domain and extrapolation diagnostics

Step 6 should mark predictions that are made outside the well-supported region of calibration data.

For every candidate, the predictive model should report an extrapolation diagnostic, for example:

- distance from the nearest measured achieved geometry;
- distance from the trust-region center;
- distance in normalized control space;
- whether the candidate lies outside the convex hull of previous high-fidelity measurements;
- whether the candidate is supported only by low-fidelity data;
- whether the candidate is near a hard constraint boundary;
- and whether the candidate requires extrapolating a discrepancy model.

This does not necessarily forbid such candidates, but Step 1 should know when a prediction is mostly extrapolation.

A simple categorical output can be used:

\[
\text{prediction regime}
\in
\{\text{interpolation},\ \text{mild extrapolation},\ \text{strong extrapolation}\}.
\]

---

## 18. Model forms suitable for Step 6

Step 6 can be implemented using different levels of sophistication. The document should not lock in one final software architecture, but it should define acceptable model families.

### 18.1 Minimal model

The minimal model uses:

- a fast simulator mean;
- a Gaussian-process discrepancy model for the scalar objective;
- fixed measurement noise from Step 4/Step 5;
- no explicit drift model beyond timestamp warnings;
- and plug-in antenna alignment.

The predictive mean is:

\[
\mu_{\mathrm{pred}}(u)
=
J_{\mathrm{sim}}(u,\hat{\theta})
+
\mu_r(u).
\]

The predictive variance is:

\[
\sigma^2_{\mathrm{pred}}(u)
=
\sigma^2_r(u)
+
\sigma^2_{\theta}(u)
+
\sigma^2_{\mathrm{noise}}(u),
\]

where \(\sigma^2_{\theta}\) may come from a Laplace approximation, posterior samples, or a conservative uncertainty inflation.

This version is useful for a first software prototype, but the file should state clearly when it uses plug-in approximations.

### 18.2 Posterior-sample model

The stronger model uses posterior samples from Step 5 and constructs predictions by sampling through the simulator and discrepancy model.

This is preferable because it naturally propagates uncertainty in:

- \(\theta\),
- discrepancy,
- noise,
- drift,
- and achieved geometry.

### 18.3 Multi-output or multi-task model

If the project uses multiple fidelities or multiple objectives, the predictive model can be represented as a multi-output or multi-task model.

Possible outputs include:

\[
(Y_{\mathrm{proxy}},\ Y_{\mathrm{RF}},\ J_{\mathrm{HF}}),
\]

or:

\[
(\text{peak boost},\ \text{bandwidth},\ \text{scan-rate proxy},\ \text{quality flag}).
\]

The key requirement is that the model must preserve the statistical relationship between the lower-fidelity measurements and the high-fidelity objective. Independent models are acceptable only if cross-fidelity information is not being used.

---

## 19. Recommended conceptual API

A practical Step-6 output can be described by a conceptual API. This is not final software, but it clarifies what Step 1 needs.

### 19.1 Prediction query

```text
predict(candidates, fidelity="HF", time=None, objective="selected")
```

Returns:

```text
mean
variance
credible_intervals
posterior_samples, if available
latent_objective_prediction
future_observation_prediction
uncertainty_breakdown
prediction_quality_flags
```

### 19.2 Multi-fidelity query

```text
predict_fidelities(candidates, fidelities, time=None)
```

Returns:

```text
mean and variance for each fidelity
cross-fidelity covariance or correlation
predicted information about J_HF
measurement-cost metadata
```

### 19.3 Constraint query

```text
predict_feasibility(candidates)
```

Returns:

```text
hard_constraint_status
learned_probability_of_feasibility
soft-constraint uncertainty
quality-flag probabilities
```

### 19.4 Diagnostic query

```text
diagnostics(candidates)
```

Returns:

```text
nearest-data distance
trust-region distance
extrapolation regime
staleness flag
drift warning
uncorrectable-error warning
noise-dominance warning
```

Step 1 then uses these outputs to compute its acquisition function.

---

## 20. Step-6 update sequence

A high-level Step-6 update should proceed as follows.

```text
Input:
    Step-5 joint calibration state
    current data set
    simulator interface
    objective definition
    measurement-fidelity definitions
    hard-domain and learned-constraint metadata
    antenna-alignment convention

Step 6 procedure:

    1. Freeze or snapshot the current Step-5 posterior state.

    2. Define the candidate representation expected by Step 1.

    3. Define whether predictions are made for commanded geometry,
       achieved geometry, or achieved-geometry distributions.

    4. Construct the latent high-fidelity predictive distribution:
           p(J_HF^latent(u) | D).

    5. Construct the future-observation predictive distribution:
           p(Y_HF(u) | D).

    6. If lower-fidelity measurements exist, construct:
           p(Y_l(u), J_HF(u) | D)
       for each useful fidelity l.

    7. If multi-objective calibration is active, construct:
           p(J_vector(u) | D).

    8. Attach learned soft-constraint and quality-flag predictions.

    9. Attach cost, drift, and extrapolation diagnostics.

    10. Validate the predictive model on recent and held-out data.

    11. Export the prediction object to Step 1.
```

---

## 21. Validation checks before handing off to Step 1

Step 6 should perform lightweight validation before Step 1 uses the model.

### 21.1 Shape and domain checks

Verify that:

- the candidate dimension matches the expected control basis;
- units and normalizations are consistent;
- all hard-domain transformations are correct;
- and predictions are only requested for variables represented in the model.

### 21.2 Posterior predictive checks

Compare posterior predictions to already measured data.

For measured points \(u_i\), check whether:

\[
J_i
\]

falls inside the model's credible interval at a reasonable frequency.

The goal is not perfect coverage after a few data points, but obvious underconfidence or overconfidence should be flagged.

### 21.3 Standardized residuals

Compute standardized residuals:

\[
z_i
=
\frac{J_i-\mu_{\mathrm{pred}}(u_i)}{\sigma_{\mathrm{pred}}(u_i)}.
\]

If \(|z_i|\) is systematically too large, the predictive uncertainty is too small or the mean model is biased. If \(|z_i|\) is systematically too small, the predictive uncertainty may be too conservative.

### 21.4 Leave-one-out or recent-point checks

For a small data set, leave-one-out checks can be expensive but informative. A simpler first version is a recent-point check:

- fit or update using all but the most recent point;
- predict the most recent point;
- check whether the prediction was calibrated.

### 21.5 Fidelity-consistency checks

If multi-fidelity prediction is used, verify whether low-fidelity observations actually improve high-fidelity predictions. If the estimated correlation between proxy and high-fidelity objective is weak or unstable, Step 6 should report that lower-fidelity data should be used cautiously.

### 21.6 Drift checks

If the baseline or incumbent has been remeasured, compare old and new predictions. If the observed change is larger than predicted by the drift model, Step 6 should inflate uncertainty or flag the model as stale.

---

## 22. Diagnostics to report after each Step-6 update

Step 6 should produce a short diagnostic summary for the operator or log file.

Recommended diagnostics:

- number of high-fidelity observations used;
- number of lower-fidelity observations used;
- current best measured configuration and objective;
- predicted objective at the current incumbent;
- predicted objective uncertainty at the incumbent;
- posterior predictive uncertainty in the active trust region;
- fraction of candidate space dominated by hard constraints;
- learned probability of feasible measurement near likely candidate regions;
- noise-dominance indicator;
- drift/staleness indicator;
- extrapolation warnings;
- whether uncorrectable diagnostic errors are limiting predicted performance;
- and whether multi-fidelity correlation is strong enough to justify proxy measurements.

A useful compact log entry is:

```text
Step 6 predictive-model status:
    HF data count:
    proxy data count:
    current incumbent J:
    predicted incumbent J ± uncertainty:
    max predicted improvement in trust region:
    dominant uncertainty source:
    feasibility warning:
    drift warning:
    extrapolation warning:
    ready for Step 1: yes/no
```

---

## 23. Failure modes and mitigations

### 23.1 Overconfident prediction

**Failure mode:** The predictive uncertainty is smaller than observed residuals.

**Mitigation:** Inflate noise, widen discrepancy priors, use posterior samples instead of plug-in estimates, or require replication before exploitation.

### 23.2 Discrepancy absorbs physical structure

**Failure mode:** The discrepancy model becomes so flexible that it hides physically meaningful detector-state information.

**Mitigation:** Use informative priors, restrict discrepancy smoothness/amplitude, and preserve the Step-5 distinction between correctable and diagnostic variables.

### 23.3 Physical parameters absorb model error

**Failure mode:** The predictive model effectively ignores discrepancy and over-interprets \(\theta\).

**Mitigation:** Ensure Step 6 uses the joint Step-5 posterior, not a point estimate of \(\theta\) alone.

### 23.4 Low-fidelity model is misleading

**Failure mode:** Cheap RF/proxy data are weakly correlated with the high-fidelity gradient-method objective in the region of interest.

**Mitigation:** Report cross-fidelity uncertainty and force high-fidelity validation before accepting a configuration.

### 23.5 Objective mismatch

**Failure mode:** Step 6 predicts a scalar objective that is not the physics figure of merit the team actually wants.

**Mitigation:** Preserve curve-level or multi-objective predictions until the scalar objective is fixed.

### 23.6 Extrapolation beyond data support

**Failure mode:** Step 1 trusts predictions far outside the measured region.

**Mitigation:** Report extrapolation flags, increase uncertainty, and restrict predictions to the trust region.

### 23.7 Drift invalidates the model

**Failure mode:** The detector changes between the data used for inference and the next proposed measurement.

**Mitigation:** Use time-aware prediction, re-baseline, or inflate uncertainty for stale predictions.

### 23.8 Commanded-position model hides hysteresis

**Failure mode:** The model learns the map from commanded positions to measured objective, but actuator hysteresis causes the same command to produce different achieved geometries.

**Mitigation:** Use achieved geometry whenever available and include achieved-geometry uncertainty in prediction.

---

## 24. Recommended minimal viable Step 6

A first implementation should not try to solve every modeling problem at once. A good minimal Step 6 is:

1. Use achieved geometry for all measured data where available.
2. Use the fast simulator as the physics mean function.
3. Use the Step-5 joint MAP or posterior samples, but do not ignore discrepancy.
4. Fit or carry forward a scalar-objective discrepancy GP.
5. Include fixed measurement noise estimates from repeated measurements or Step 4 uncertainty propagation.
6. Produce both latent-objective and future-observation predictions.
7. Return mean, variance, credible interval, and posterior samples for \(J_{\mathrm{HF}}\).
8. Report extrapolation, drift, and noise-dominance flags.
9. Preserve the option to upgrade to multi-fidelity or multi-objective prediction.

In formula form, the minimal predictive model can be written as:

\[
J_{\mathrm{HF}}^{\mathrm{latent}}(u)
=
J_{\mathrm{sim}}(u,\theta)
+
r_{\mathrm{HF}}(u),
\]

\[
Y_{\mathrm{HF}}(u)
=
J_{\mathrm{HF}}^{\mathrm{latent}}(u)
+
\epsilon_{\mathrm{HF}}.
\]

The key requirement is that the uncertainty in \(\theta\), \(r\), and \(\epsilon\) is represented in the prediction sent to Step 1.

---

## 25. Stronger Step 6 version

A stronger version should add:

- posterior-sample propagation through the simulator;
- achieved-geometry uncertainty;
- drift prediction to a future measurement time;
- multi-fidelity predictive covariance;
- multi-objective prediction for peak--bandwidth trade-off studies;
- learned soft-constraint probabilities;
- antenna-alignment uncertainty from Step 3;
- and explicit uncertainty decomposition.

This stronger version is more appropriate once the initial software loop works and the measurement budget justifies richer modeling.

---

## 26. Handoff to Step 1

Step 6 should hand Step 1 a prediction object, not a single best candidate.

The handoff should include:

\[
p(J_{\mathrm{HF}}(u)\mid D),
\]

or, in the multi-objective case,

\[
p(\mathbf{J}_{\mathrm{HF}}(u)\mid D).
\]

It should also include:

- posterior predictive samples;
- mean and uncertainty functions;
- latent versus observation prediction distinction;
- learned feasibility probabilities;
- hard-domain metadata;
- measurement-fidelity predictions;
- measurement-cost metadata;
- drift/staleness diagnostics;
- extrapolation diagnostics;
- and a clear statement of which objective definition is being predicted.

Step 1 then uses this model to compute an acquisition function such as constrained expected improvement, noisy expected improvement, knowledge gradient, multi-fidelity knowledge gradient, or multi-objective acquisition, depending on the active calibration mode.

Step 6 must not hard-code the acquisition function. It should provide the calibrated predictive distribution that makes several acquisition choices possible.

---

## 27. Suggested offline validation tests

Before using Step 6 in a live detector loop, test it offline.

### 27.1 No-discrepancy closure test

Generate synthetic data from the simulator with no discrepancy. Step 6 should recover predictions consistent with the simulator and should not invent a large residual model.

### 27.2 Known-discrepancy test

Generate synthetic data with a known smooth discrepancy. Step 6 should learn the discrepancy and improve high-fidelity predictions relative to the raw simulator.

### 27.3 Biased-theta test

Generate data where simulator parameter changes and discrepancy are partially confounded. Step 6 should propagate the resulting uncertainty rather than collapse to an overconfident point estimate.

### 27.4 Noise-scaling test

Generate repeated noisy observations with known noise. Step 6 should predict future observation scatter correctly.

### 27.5 Drift test

Generate synthetic data with slow time drift. Step 6 should either track the drift or flag stale predictions.

### 27.6 Multi-fidelity test

Generate low- and high-fidelity data with controlled correlation. Step 6 should use low-fidelity data only when it improves high-fidelity prediction.

### 27.7 Multi-objective test

Generate boost curves with a peak--bandwidth trade-off. Step 6 should expose the objective vector or selected scalar objective without hiding the trade-off.

### 27.8 Achieved-geometry test

Generate data where commanded and achieved positions differ due to hysteresis. Step 6 should perform better when using achieved geometry than when using commanded geometry alone.

---

## 28. Success criteria for Step 6

Step 6 is successful if:

- it produces calibrated posterior predictions for future high-fidelity objective values;
- it does not collapse the Step-5 posterior into an overconfident point estimate;
- it preserves the distinction between simulator prediction, discrepancy, measurement noise, and drift;
- it exposes enough uncertainty information for Step 1 to decide between exploitation, exploration, replication, re-baselining, and lower-fidelity measurement;
- it uses achieved geometry where available;
- it reports when predictions are extrapolations;
- and it allows final configurations to be validated against high-fidelity boost-factor measurements.

A simple operational success condition is:

\[
\text{measured } J_{\mathrm{HF}}
\text{ at proposed candidates should fall inside Step-6 predictive intervals at approximately the advertised rate.}
\]

If this is not true, Step 6 should not be trusted for aggressive optimization until the model is corrected or uncertainty is inflated.

---

## 29. Compact Step-6 summary

```text
Input:
    Step-5 joint posterior over detector state, discrepancy, noise, and drift
    fast simulator
    current data set with achieved geometries
    objective definition
    measurement-fidelity definitions
    learned soft-constraint models
    antenna-alignment convention

Core action:
    Build the posterior predictive distribution for future candidate states.

Main prediction:
    p(J_HF(u) | data)

Must include:
    simulator prediction
    uncertainty in detector-state parameters
    discrepancy uncertainty
    measurement-noise model
    drift/staleness handling
    achieved-geometry uncertainty when available
    optional multi-fidelity and multi-objective outputs
    learned soft-constraint predictions

Output to Step 1:
    optimizer-facing predictive model object
    mean / variance / samples / credible intervals
    feasibility and quality predictions
    cost and fidelity metadata
    diagnostics and warnings

Not responsible for:
    choosing the next candidate
    optimizing the acquisition function
    moving hardware
    measuring the boost factor
    re-running full inference
```

---

## 30. Source anchors

This Step-6 design is based on the locked version-3 parent proposal and the step-specific designs already developed for Steps 1, 3, 4, and 5.

External background anchors used for this Step-6 design:

1. **BoTorch acquisition functions:** BoTorch acquisition functions use posterior summaries or posterior samples to compute quantities such as expected improvement, which motivates Step 6 as the provider of the posterior predictive object used by Step 1.  
   <https://botorch.org/docs/acquisition>

2. **BoTorch model types and noise handling:** BoTorch distinguishes homoskedastic, fixed, and heteroskedastic noise handling, and supports single-task, multi-output, and multi-task model structures.  
   <https://botorch.org/docs/models>

3. **GPyTorch posterior predictive distinction:** GPyTorch distinguishes the latent model posterior from the posterior predictive distribution after including likelihood noise. This motivates the Step-6 distinction between latent detector performance and future noisy measurement.  
   <https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html>

4. **SCBO / trust-region constrained Bayesian optimization:** Step 6 provides the prediction object that Step 1 can use in the constrained BO loop.  
   <https://botorch.org/docs/tutorials/scalable_constrained_bo>

5. **BoTorch constraints:** BoTorch distinguishes parameter constraints from outcome constraints. This supports the proposal's split between exact hard-domain filtering and learned soft-constraint predictions.  
   <https://botorch.org/docs/constraints>

6. **Multi-fidelity Bayesian optimization:** BoTorch's multi-fidelity tutorial uses lower-fidelity evaluations to optimize a target high-fidelity objective, motivating Step 6's multi-fidelity predictive output.  
   <https://botorch.org/docs/v0.16.0/tutorials/multi_fidelity_bo>

7. **Knowledge Gradient:** Knowledge-gradient acquisition is a look-ahead method that values the expected benefit of additional observations, motivating Step 6's need to expose uncertainty and fidelity information, not only the current predictive mean.  
   <https://archive.botorch.org/tutorials/one_shot_kg>

8. **Multi-objective Bayesian optimization:** BoTorch describes multi-objective BO as learning the Pareto front of trade-offs, motivating Step 6's optional vector-output prediction for peak--bandwidth trade-off studies.  
   <https://botorch.org/docs/multi_objective>

9. **Bayesian calibration and discrepancy modeling:** Kennedy and O'Hagan introduced Bayesian calibration with a discrepancy term, motivating the Step-5/Step-6 use of joint calibration and discrepancy rather than a simulator-only prediction.  
   <https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/KennedyOHagan_2002.pdf>

10. **Bayesian calibration with adaptive model discrepancy:** Recent calibration literature continues to emphasize joint treatment of model parameters and discrepancy.  
    <https://inria.hal.science/hal-03827922/document>

---

# Appendix A — Implementation status (Step 6)

*Added by the implementation; the design text above is unchanged.
Module: `madmax_calibration.steps.step6_predictive`.*

## A.1 What was built

The **posterior-sample** construction (§8): Laplace posterior samples of
θ from Step 5 are pushed through the fast simulator per candidate; the
discrepancy GP adds its mean and variance; drift extrapolates to the
intended future measurement time. Latent-objective and future-observation
predictions are kept distinct (§7) — `latent_sd` answers "is this
configuration better", `obs_sd` adds the measurement-noise floor for
"what will a measurement look like". Posterior samples of the latent
objective are exposed for Thompson-style use.

Every prediction carries the diagnostics the note requires: an
extrapolation regime from distance to the nearest HF training point
(§17: interpolation / mild / strong), a staleness measure (§16), and,
before hand-off to Step 1, a validation pass — standardized residuals
over the training data with a 2σ coverage figure and over/under-confidence
flags (§21). Correctable-vs-diagnostic labels and the hard-domain filter
are preserved.

## A.2 Unchanged interface across Phases 1.1–1.2

The Step-5→Step-6 boundary held: the richer Step-5 inference (curve
summaries, reflectivity channel) flows through **without any Step-6
interface change** — better θ posteriors simply yield sharper, better-
calibrated predictions of J. The J-component discrepancy GP is the one
Step 6 consumes; the per-summary and reflectivity GPs live alongside it
in the Step-5 result and are available but not yet used for multi-output
prediction.

## A.3 Open items

- **Plug-in antenna optimum** (§5, option 1): predictions are for the
  antenna-aligned F(u_B) but do **not** yet propagate antenna-alignment
  uncertainty (roadmap Phase 4.3). This is the documented approximation
  of §5/§6.
- **Multi-fidelity predictive output** (§11): Step 6 predicts J; it does
  not yet expose a joint predictive over the reflectivity observables
  (the LF channel is consumed inside Step 5 instead). A multi-output
  predictive is the natural next step if LF-informed look-ahead
  acquisition is added (roadmap Phase 3).
- **Achieved-geometry distribution** (§6): predictions condition on
  commanded ≈ expected achieved geometry; the achieved-geometry
  *distribution* is not integrated (achieved readback is used for all
  training data).
- Multi-objective predictive output (§10) is not implemented (scalar
  objective).

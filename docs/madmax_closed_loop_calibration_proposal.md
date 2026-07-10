# Closed-Loop Calibration Proposal for the MADMAX Detector

**Status:** Revised high-level project outline, version 3  
**Revision note:** This version keeps the locked seven-step calibration structure, but adds the main feasibility and modeling corrections raised during review: measurement-noise budget, control-basis consistency, hard versus learned constraints, the MADMAX area-law trade-off, drift/readback, multi-fidelity measurements, posterior uncertainty propagation, and an acquisition strategy that can value information as well as immediate improvement.

**Scope:** This document remains at the conceptual and statistical-method level. It intentionally avoids lower-level implementation details beyond the points discussed in the project conversation.

---

## 1. Project idea in one sentence

The project is to build a **closed-loop calibration algorithm** for the MADMAX detector that starts from an already optimized nominal disk configuration, uses existing boost-factor determination as the high-fidelity experimental objective, and iteratively finds the best calibrated real-detector configuration using physics-informed, safe, budget-aware Bayesian optimization together with a joint detector-state/discrepancy calibration model.

The key point is that the project does **not** replace existing MADMAX disk-spacing optimization. Instead, it works **around** an already available nominal configuration and calibrates the real detector.

---

## 2. Physical and methodological assumptions

We assume the following are already available.

### 2.1 Nominal disk configuration

For a target frequency window \(W\), an initial nominal disk/mirror configuration is already known:

\[
q_0(W).
\]

This nominal configuration comes from an existing offline MADMAX disk-spacing optimization procedure. Therefore, the project does **not** need to solve the initial disk-spacing problem from scratch.

### 2.2 High-fidelity boost-factor determination

For any given detector configuration \(q\), the boost factor can be determined experimentally using the existing MADMAX **gradient method**:

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu; q).
\]

Here, “gradient method” refers to the MADMAX boost-factor determination method based on small controlled disk movements and electromagnetic response measurements. It does **not** refer to stochastic gradient descent or machine-learning gradient descent.

In this proposal, the gradient-method boost-factor determination is treated as the **high-fidelity** measurement of the calibration objective.

### 2.3 Fast physics simulation

A fast physics simulation is available:

\[
\beta^2_{\mathrm{sim}}(\nu; q, \theta),
\]

where \(\theta\) represents detector-state or nuisance parameters, such as offsets, losses, misalignments, or effective geometry corrections.

This simulator is used to guide the next proposed calibration move, but the final objective is still anchored to the measured boost factor.

### 2.4 Lower-fidelity or proxy measurements

The loop may also use faster, cheaper observables if available, for example RF-response or reflectivity-based measurements, antenna-coupling proxies, or simulator-only predictions.

These cheaper observables are **not** replacements for the final boost-factor measurement. Instead, they can be used to explore more efficiently and reduce the number of expensive high-fidelity boost-factor determinations.

Thus the measurement hierarchy is:

\[
\text{simulation / cheap RF proxies}
\rightarrow
\text{gradient-method boost-factor measurement}
\rightarrow
\text{final high-fidelity validation.}
\]

The exact set of available measurement fidelities should be decided with the experimental team.

### 2.5 Achieved geometry, not only commanded geometry

The optimizer should use the **achieved** detector geometry whenever readback is available, not only the commanded motor positions.

Therefore the calibration data should distinguish:

\[
u_{\mathrm{cmd}} \quad \text{and} \quad \tilde{u}_{\mathrm{achieved}},
\]

where \(\tilde{u}_{\mathrm{achieved}}\) denotes the geometry inferred from position readback, actuator feedback, or other monitoring information.

This matters because piezo actuators can show hysteresis, creep, and finite positioning reproducibility. If the optimizer learns only the commanded map, it may learn a smeared version of the true detector response.

---

## 3. Goal of the calibration algorithm

The goal is to find a calibrated configuration near the nominal one:

\[
q^\star = q_0 + \Delta q^\star
\]

that is best according to a physics-relevant objective derived from the **measured** boost-factor curve.

The main outputs should be:

\[
q^\star,
\]

\[
\widehat{\beta^2}_{\mathrm{meas}}(\nu; q^\star),
\]

and an uncertainty statement for the final measured boost-factor curve.

The algorithm should also output a calibrated statistical state:

\[
p(\theta, r, \text{noise}, \text{drift} \mid D),
\]

or a carefully qualified summary of it.

Finally, the project should output a **feasibility report**, including:

- the estimated measurement uncertainty in the scalar objective \(J\),
- the number and cost of high-fidelity measurements used,
- the number and cost of lower-fidelity measurements used,
- and whether the achieved improvement is statistically meaningful compared with the measurement noise.

---

## 4. Main control variables

At the high level, the control variables are divided into two groups.

---

### 4.1 Booster-state variables

The booster-state variables are collected as:

\[
u_B =
\left(
 a_{\mathrm{disk}},
 z_{\mathrm{global}},
 z_{\mathrm{reflecting\ mirror}},
 z_{\mathrm{focusing\ mirror}}
\right).
\]

These variables describe the state of the booster geometry.

The term \(a_{\mathrm{disk}}\) should **not** be interpreted as full independent re-optimization of all disk positions. Instead, it represents a small number of physically meaningful disk-correction modes around the nominal configuration \(q_0\).

Conceptually:

\[
q_{\mathrm{disk}}
=
q_{0,\mathrm{disk}} + B a_{\mathrm{disk}},
\]

where \(B\) is a low-dimensional basis of allowed disk-correction modes.

The global z-positioning of the booster is part of this same booster-state block. At this high-level stage, disk correction, global z-positioning, reflecting mirror correction, and focusing mirror correction are treated together as one proposed booster geometry.

---

### 4.2 Antenna variables

The antenna variables are:

\[
u_A =
\left(
 x_{\mathrm{ant}},
 y_{\mathrm{ant}}
\right).
\]

These describe the receiver antenna position in the two transverse directions.

The full detector configuration is therefore written as:

\[
q = q(q_0, u_B, u_A).
\]

---

## 5. Control basis versus inferred error model

The correction basis and the detector-state/error model must be consistent.

The proposal should distinguish between two kinds of inferred detector-state variables:

\[
\theta = (\theta_{\mathrm{corr}}, \theta_{\mathrm{diag}}).
\]

### 5.1 Correctable detector-state variables

\(\theta_{\mathrm{corr}}\) contains errors that the chosen control variables can actually correct.

For example, if the online control basis \(B\) includes a global stack shift, stack compression, and a small number of smooth disk modes, then \(\theta_{\mathrm{corr}}\) may contain corresponding global or smooth geometry errors.

These inferred errors can feed directly into future control proposals.

### 5.2 Diagnostic but uncorrectable detector-state variables

\(\theta_{\mathrm{diag}}\) contains errors that may be inferable but not directly correctable with the chosen online control basis.

For example, if the inference model can detect a localized per-disk offset but the online correction basis does not contain a mode capable of compensating that local error, then that inferred quantity is diagnostic. It can explain performance loss and improve predictions, but it should not be treated as something the optimizer can cancel.

The proposal therefore adopts the following rule:

> Every inferred error parameter should be labeled as either correctable by the chosen control basis or diagnostic-only.

This avoids a silent mismatch in which the model diagnoses errors that the controller cannot physically fix.

---

## 6. Calibration objective and the MADMAX area-law trade-off

For a target frequency window \(W\), define a scalar calibration objective:

\[
J(q)
=
J\left(
\widehat{\beta^2}_{\mathrm{meas}}(\nu; q),
\nu \in W
\right).
\]

However, this scalar objective must not be treated as an arbitrary curve summary. MADMAX obeys an area-law-type trade-off: for a fixed disk system, changing disk spacings trades peak boost against bandwidth. Therefore, maximizing peak boost, maximizing bandwidth, and maximizing a smooth minimum over a band are physically different choices.

The objective should therefore be tied to the actual physics figure of merit, for example:

- expected scan rate over the target window,
- expected sensitivity over the target window,
- robust broadband performance over \(W\),
- or narrow-band confirmation performance near a target frequency.

At this high-level stage, the exact scalarization can remain open, but the proposal should explicitly state:

> The scalar objective \(J\) must be chosen together with the physics team because it fixes the operating point on the peak--bandwidth trade-off.

If the operating point is not known in advance, the better high-level formulation is **multi-objective calibration**. For example, the optimizer can expose a Pareto front between:

\[
\text{peak boost},
\qquad
\text{bandwidth or band flatness},
\qquad
\text{scan-rate proxy},
\qquad
\text{calibration cost or risk}.
\]

Then the physics team can choose the preferred operating point from the Pareto front rather than having one scalarization baked in prematurely.

Operationally, the nested objective remains:

\[
u_B^\star
=
\arg\max_{u_B} F(u_B),
\]

with

\[
F(u_B)
=
\max_{u_A} J(q_0, u_B, u_A),
\]

but \(J\) should be a physics-motivated figure of merit, not just a generic average or peak of the boost curve.

---

## 7. Measurement-noise and calibration-budget requirement

The calibration loop is only useful if the expected improvements are resolvable above measurement noise and achievable within the available experimental budget.

Before running a full outer-loop calibration, the project should estimate:

\[
\sigma_J(u, \ell),
\]

where \(\sigma_J\) is the uncertainty of the scalar objective and \(\ell\) denotes the measurement fidelity.

The proposal should allow for:

- replicated measurements at the baseline and at selected candidate configurations,
- a noise model for \(J\),
- possible heteroscedasticity, meaning that the measurement uncertainty may depend on frequency, configuration, or measurement type,
- and a calibration budget measured in high-fidelity boost-factor determinations, lower-fidelity measurements, and total experimental time.

A candidate improvement should be considered meaningful only if it is large compared with the relevant uncertainty. Conceptually:

\[
\Delta J_{\mathrm{expected}}
\gtrsim
\text{measurement uncertainty threshold}.
\]

If expected improvements are smaller than the boost-factor determination noise, the optimizer should not continue blindly. It should either:

- replicate the measurement,
- switch to cheaper information-gathering measurements,
- reduce the search space,
- use the simulator and posterior model without moving hardware,
- or stop and report that the remaining improvement is not experimentally resolvable.

This budget/noise check is not a low-level implementation detail. It is a core feasibility condition for the project.

---

## 8. Joint detector-state, discrepancy, noise, and drift model

The statistical calibration model should not first point-estimate detector-state parameters and then fit a leftover residual. Detector-state parameters and simulator discrepancy can be confounded.

The corrected high-level model is:

\[
J_{\ell,\mathrm{meas}}(\tilde{u}, t)
=
J_{\ell,\mathrm{sim}}(\tilde{u}, \theta_t)
+
r_\ell(\tilde{u}, t)
+
\epsilon_\ell,
\]

where:

- \(\tilde{u}\) is the achieved geometry, not merely the commanded geometry,
- \(\ell\) is the measurement fidelity,
- \(\theta_t\) are detector-state variables, possibly slowly time-dependent,
- \(r_\ell\) is a fidelity-dependent simulation--measurement discrepancy,
- and \(\epsilon_\ell\) is measurement noise.

The Bayesian calibration target is therefore:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D).
\]

The optimizer-facing prediction is the posterior predictive distribution for the high-fidelity objective:

\[
p(J_{\mathrm{HF}}(u) \mid D),
\]

where the uncertainty in \(\theta\), discrepancy, noise, and drift is marginalized over rather than replaced by a single point estimate.

A compact expression for the predictive mean is:

\[
\mu_{\mathrm{pred}}(u)
=
\mathbb{E}\left[
J_{\mathrm{sim}}(u, \theta) + r(u)
\mid D
\right],
\]

but the optimizer should also use the corresponding predictive uncertainty.

---

## 9. Constraint handling: hard constraints versus learned constraints

The phrase “hardware constraints” must be split into distinct categories.

### 9.1 Known hard constraints

Known geometric and mechanical constraints must be enforced exactly before any candidate is sent to the hardware.

Examples include:

- disk collision avoidance,
- minimum disk gaps,
- piezo travel limits,
- CAD-defined geometry limits,
- actuator stroke limits,
- and known safe envelopes.

These constraints are not learned by trying unsafe points. They define the allowed proposal set:

\[
u \in \mathcal{U}_{\mathrm{hard}}.
\]

A candidate outside \(\mathcal{U}_{\mathrm{hard}}\) is never proposed.

### 9.2 Unknown or soft constraints

Other constraints may be unknown or empirical, for example:

- measurement failure regions,
- bad coupling regions,
- parasitic-mode regions,
- unstable RF-response regions,
- or regions where the gradient-method boost determination becomes unreliable.

These may be modeled statistically and learned during calibration.

For such constraints, constrained BO or safe-BO methods are appropriate. But for damage-relevant constraints, the algorithm should use a conservative safe set and never rely on “learning by failing.”

---

## 10. Drift and time dependence

The calibration loop should not assume that the detector state is perfectly static over a multi-hour procedure.

MADMAX-like hardware may experience:

- mechanical vibration,
- thermal contraction or relaxation,
- piezo hysteresis,
- piezo creep,
- slow alignment drift,
- and slow changes in RF response.

The high-level mitigation is simple:

1. Use achieved geometry/readback whenever available.
2. Re-measure the baseline or incumbent best configuration periodically.
3. Include time or run index in the calibration data.
4. Treat stale measurements carefully if the detector state has drifted.

Thus the data should be represented as:

\[
D_i =
(\tilde{u}_i, t_i, \ell_i, \widehat{\beta^2}_i(\nu), J_i, \sigma_{J,i}),
\]

not only as:

\[
D_i = (u_i, J_i).
\]

This does not require a complex drift model in the first version, but the proposal should explicitly recognize drift as part of the calibration uncertainty.

---

# 11. Revised seven-step calibration loop

## Step 0 — Initialize, validate baseline, and set feasibility limits

This step happens before the repeated loop starts.

Start from the nominal MADMAX configuration:

\[
q_0(W).
\]

Set the correction variables initially to zero, or to the best known previous calibration:

\[
u_B^{(0)} = 0,
\qquad
u_A^{(0)} = 0.
\]

Measure the initial boost-factor curve using the gradient method:

\[
\widehat{\beta^2}_0(\nu).
\]

Compute the baseline objective:

\[
J_0.
\]

Also estimate the baseline measurement uncertainty:

\[
\sigma_{J,0}.
\]

At this stage the project should also define:

- the physics objective or objective family,
- the hard feasibility domain,
- the available calibration time or evaluation budget,
- the available measurement fidelities,
- and the initial safe region for online exploration.

This step gives the optimizer a real measured reference point and prevents the calibration from starting without a noise/budget sanity check.

---

## Step 1 — Propose the next booster-state correction and measurement action

This is the main outer-loop decision step.

The algorithm proposes the next booster-state correction:

\[
u_B^{(t+1)},
\]

and, if a measurement hierarchy is available, the measurement action or fidelity:

\[
\ell^{(t+1)}.
\]

The proposal uses **physics-informed, constrained, budget-aware Bayesian optimization**.

The proposal is based on the current posterior-predictive model:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t}),
\]

which combines:

\[
J_{\mathrm{sim}}(u, \theta)
\]

with the discrepancy model:

\[
r(u),
\]

while marginalizing over the current joint posterior:

\[
p(\theta, r, \sigma_J, \text{drift} \mid D_{1:t}).
\]

The proposal step should explicitly use:

1. the fast physics simulation,
2. the current joint detector-state/discrepancy posterior,
3. the optimizer-facing posterior-predictive model,
4. previous measured boost-factor and proxy-measurement data,
5. the measurement-noise model,
6. the measurement-cost/budget model,
7. known hard constraints,
8. and learned soft/safety constraints.

The acquisition criterion should not be pure exploitation of the current mean predicted boost factor. It should balance:

\[
\text{expected improvement},
\qquad
\text{uncertainty reduction},
\qquad
\text{measurement cost},
\qquad
\text{safety}. 
\]

In words, Step 1 answers:

> Given the simulator, the joint calibration/discrepancy model, current uncertainty, measurement costs, and hardware constraints, which safe booster correction and measurement action should be tried next?

The recommended high-level method remains trust-region constrained Bayesian optimization, but augmented with:

- hard feasibility filtering,
- noise awareness,
- cost awareness,
- optional multi-fidelity acquisition,
- and optional information-gain terms for calibration.

For the first iteration, before a meaningful calibration/discrepancy posterior exists, the optimizer can fall back to the nominal simulator prediction and the baseline measured data.

---

## Step 2 — Set the booster geometry and record achieved geometry

Move the detector to the proposed booster state:

\[
q_B^{(t+1)}
=
q_{0,B}
+
\Delta q_B(u_B^{(t+1)}).
\]

At this high level, this one step includes:

- disk correction modes,
- global z-positioning,
- reflecting mirror correction,
- focusing mirror correction.

After moving, record the achieved geometry as well as possible:

\[
\tilde{u}_B^{(t+1)}.
\]

The important conceptual point is that the algorithm proposes and sets a **complete booster geometry**, rather than treating these sub-components as disconnected outer-loop steps.

---

## Step 3 — Align the antenna for this booster state

With the booster geometry fixed, optimize the antenna position:

\[
u_A =
(x_{\mathrm{ant}}, y_{\mathrm{ant}}).
\]

The high-level antenna-alignment problem is:

\[
u_A^{(t+1)}
=
\arg\max_{u_A}
J(q_0, u_B^{(t+1)}, u_A).
\]

The recommended high-level method is either:

\[
\boxed{\text{2D Bayesian optimization}}
\]

or:

\[
\boxed{\text{2D scan plus local fit}}.
\]

This step answers:

> For the current booster geometry, where should the antenna be placed to couple best to the boosted signal?

As with the booster geometry, the achieved antenna position should be recorded if readback is available.

---

## Step 4 — Measure the selected observable and, when required, the boost factor

After the booster geometry is set and the antenna is aligned, perform the measurement selected in Step 1.

If the selected measurement is high-fidelity, use the existing MADMAX gradient method to determine:

\[
\widehat{\beta^2}_{t+1}(\nu)
\]

for the complete current configuration:

\[
q^{(t+1)}
=
q(q_0, \tilde{u}_B^{(t+1)}, \tilde{u}_A^{(t+1)}).
\]

Then convert the measured curve into the scalar calibration score:

\[
J_{\mathrm{meas},t+1}
=
J\left(
\widehat{\beta^2}_{t+1}(\nu)
\right).
\]

If the selected measurement is lower-fidelity, record the corresponding proxy observable and its uncertainty.

The final accepted best configuration must be validated with the high-fidelity boost-factor determination. Lower-fidelity measurements are allowed to guide exploration, but not to replace final validation.

---

## Step 5 — Jointly update detector-state, discrepancy, noise, and drift inference

This is the main statistical model-update step.

It is **not** a hardware movement step.

Use the accumulated calibration data:

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
 \sigma_{J,i}
\right\}_{i=0}^{t+1}
\]

to update the joint calibration model:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}).
\]

Here \(\theta\) contains physically interpretable detector-state or nuisance variables, for example:

- disk offsets,
- global z-offset,
- antenna offset,
- mirror or focusing misalignment,
- loss parameters,
- effective geometry errors.

The discrepancy \(r_\ell\) captures simulation--measurement mismatch not fully explained by the simulator and the physically interpretable parameters.

This step also classifies inferred detector-state variables as:

\[
\theta_{\mathrm{corr}}
\quad \text{or} \quad
\theta_{\mathrm{diag}}.
\]

That is, it separates errors that are correctable by the online control basis from errors that are diagnostic-only.

This step answers:

> Given all measured data so far, what joint combination of detector-state parameters, discrepancy, measurement noise, and drift best explains the observations, within the chosen priors and constraints?

The recommended high-level method is:

\[
\boxed{\text{Bayesian calibration with a joint } (\theta, r, \text{noise}, \text{drift}) \text{ model}.}}
\]

A simplified version may use a joint MAP approximation, but the key requirement is that discrepancy is included while \(\theta\) is inferred.

---

## Step 6 — Update the optimizer-facing predictive model

This is the second model-update step.

It converts the joint calibration result into the predictive object needed by the Bayesian optimizer.

Using:

\[
p(\theta_t, r_\ell, \sigma_J, \text{drift} \mid D_{1:t+1}),
\]

construct the posterior-predictive model for candidate future calibration states:

\[
p(J_{\mathrm{HF}}(u) \mid D_{1:t+1}).
\]

This prediction should marginalize over uncertainty in:

- detector-state variables,
- discrepancy,
- measurement noise,
- and drift.

The optimizer-facing mean can be written as:

\[
\mu_{\mathrm{pred},t+1}(u)
=
\mathbb{E}\left[
J_{\mathrm{sim}}(u, \theta) + r(u)
\mid D_{1:t+1}
\right],
\]

but the predictive uncertainty is just as important:

\[
\sigma_{\mathrm{pred},t+1}(u).
\]

This model is used directly in the next iteration of Step 1.

The corrected split remains:

\[
\boxed{\text{Step 5: jointly infer detector state, discrepancy, noise, and drift}}
\]

\[
\boxed{\text{Step 6: update the prediction model used by the optimizer}.}
\]

---

## Step 7 — Stop or repeat

Finally, decide whether the calibration should stop.

Stop if one of the following holds:

- the best measured configuration is good enough,
- the expected improvement is smaller than the measurement uncertainty,
- the expected improvement per unit measurement cost is too small,
- the calibration budget is exhausted,
- the posterior uncertainty is dominated by uncorrectable diagnostic errors,
- drift makes further exploitation unreliable without re-baselining,
- or the physics team selects an operating point from the current scalar objective or Pareto front.

Otherwise, return to Step 1.

The loop closes as:

\[
\text{measurement}
\rightarrow
\text{joint calibration/discrepancy/noise/drift update}
\rightarrow
\text{predictive-model update}
\rightarrow
\text{better next proposal}.
\]

---

# 12. Clean algorithm summary

```text
Given:
    nominal configuration q0(W)
    fast physics simulator beta_sim^2(nu; q, theta)
    gradient-method boost-factor measurement as high-fidelity objective
    optional lower-fidelity RF/proxy measurements
    achieved-geometry readback when available
    hard mechanical/geometric constraints
    learned soft/safety constraints
    allowed booster and antenna control variables
    calibration time/evaluation budget

Goal:
    find q* near q0 that optimizes a physics-relevant measured boost-factor objective J
    while respecting noise, cost, safety, and the peak-bandwidth trade-off

Step 0:
    Start from q0.
    Measure the baseline boost factor and objective J0.
    Estimate measurement uncertainty and set budget, objective, and hard feasibility domain.

Outer calibration loop:

    1. Propose next booster-state correction u_B and measurement action/fidelity
       using physics-informed, constrained, budget-aware Bayesian optimization.
       The acquisition uses the posterior predictive p(J_HF(u) | data), not a plug-in theta estimate.

    2. Set the booster geometry:
       disk correction modes,
       global z-position,
       reflecting mirror correction,
       focusing mirror correction.
       Record achieved geometry, not only commanded geometry.

    3. Align the antenna:
       optimize x_ant and y_ant for the current booster state.
       Record achieved antenna position if available.

    4. Measure the selected observable:
       use lower-fidelity measurements for cheap exploration when appropriate;
       use the gradient method for high-fidelity boost-factor measurement and final validation.

    5. Jointly update detector-state, discrepancy, noise, and drift inference:
       update p(theta, discrepancy, noise, drift | data),
       and classify inferred errors as correctable or diagnostic-only.

    6. Update the optimizer-facing predictive model:
       produce the corrected posterior predictive distribution for future high-fidelity J,
       marginalizing over detector-state, discrepancy, noise, and drift uncertainty.

    7. Stop or repeat:
       if no meaningful improvement remains relative to uncertainty and cost,
       return the best validated configuration;
       otherwise go back to Step 1.
```

---

# 13. Information flow

The central information flow is:

\[
\boxed{\text{Step 4: measured observable / boost factor}}
\]

feeds:

\[
\boxed{\text{Step 5: joint detector-state/discrepancy/noise/drift update}}
\]

which feeds:

\[
\boxed{\text{Step 6: optimizer-facing posterior predictive model}}
\]

which feeds back into:

\[
\boxed{\text{Step 1: next booster-state and measurement proposal}.}
\]

Thus, Step 1 uses:

\[
\boxed{
\text{fast simulation}
+
\text{joint detector-state/discrepancy posterior}
+
\text{noise and drift model}
+
\text{measurement-cost model}
+
\text{previous data}
+
\text{hard and learned constraints}
}.
\]

The discrepancy model is therefore not an after-the-fact diagnostic. It is part of the predictive model that decides the next calibration move.

---

# 14. Recommended high-level statistical methods

| Calibration part | Objective | Recommended method |
|---|---|---|
| Booster-state correction | Choose disk/global-z/mirror/focus corrections that improve the high-fidelity measured objective | Physics-informed constrained Bayesian optimization |
| Hardware safety | Avoid impossible or damaging moves | Exact hard-constraint filtering; safe BO only for uncertain but non-damaging constraints |
| Measurement budget | Spend expensive measurements only where useful | Budget-aware / cost-aware acquisition |
| Measurement hierarchy | Use cheap information before expensive validation | Multi-fidelity Bayesian optimization |
| Antenna alignment | Maximize coupling for fixed booster state | 2D Bayesian optimization or 2D scan plus local fit |
| Boost-factor evaluation | Obtain experimental objective value | Existing MADMAX gradient method as high-fidelity measurement |
| Detector-state and discrepancy calibration | Infer detector-state parameters and simulation--measurement discrepancy together | Joint Bayesian calibration model, or joint MAP approximation |
| Noise modeling | Avoid chasing improvements smaller than measurement uncertainty | Replication plus fixed-noise or heteroscedastic-noise model |
| Drift handling | Avoid stale calibration state | Achieved-geometry readback, periodic re-baselining, optional time-dependent model |
| Objective selection | Respect peak-bandwidth trade-off | Physics figure of merit; optionally multi-objective BO / Pareto front |
| Experiment design | Learn useful calibration information, not only exploit current best J | Acquisition balancing improvement and information gain |
| Final decision | Decide whether calibration is finished | Expected improvement versus uncertainty, cost, safety, and drift |

The central statistical components are therefore:

\[
\boxed{\text{safe, constrained, budget-aware Bayesian optimization}}
\]

and:

\[
\boxed{\text{joint Bayesian calibration with discrepancy, noise, and drift modeling}.}
\]

---

# 15. What changed relative to the previous version

This version keeps the same high-level seven-step structure, but makes the following conceptual corrections.

## 15.1 Noise and budget are now explicit

The proposal no longer assumes that the optimizer can resolve small improvements automatically. The loop must estimate measurement uncertainty, allow replication, model noise, and stop or switch strategy when expected improvement is not distinguishable from measurement error.

## 15.2 The control basis and error model are reconciled

The proposal now separates inferred errors into correctable and diagnostic-only variables. A low-dimensional disk-correction basis is still preferred for online BO efficiency, but the inference model must not silently imply that arbitrary local errors can be corrected if the control basis cannot span them.

## 15.3 Hardware constraints are split

Known geometric and mechanical constraints are enforced exactly and are never learned by failure. Only unknown, soft, or measurement-quality constraints are modeled statistically.

## 15.4 The objective is tied to physics

The proposal now explicitly flags the MADMAX area-law trade-off. The scalar objective \(J\) should be chosen as a scan-rate, sensitivity, broadband robustness, or confirmation objective, not as a generic boost-curve summary. If the operating point is not known, multi-objective BO can expose the Pareto front.

## 15.5 Drift and achieved geometry are included

The proposal now treats achieved geometry/readback and periodic baseline remeasurement as part of the high-level calibration design.

## 15.6 Step 1 uses full posterior uncertainty

The proposal no longer plugs only a point estimate \(\hat{\theta}\) into the predictor. The acquisition should use the posterior predictive distribution, marginalizing over detector-state, discrepancy, noise, and drift uncertainty.

## 15.7 Multi-fidelity and information-gain ideas are included

The proposal now allows the outer loop to choose not only the next configuration, but also the measurement action or fidelity. It also frames calibration as partly an experiment-design problem: sometimes the best next step is the one that reduces model uncertainty, not the one with the highest immediate expected boost factor.

---

# 16. What the project is not doing

This project does **not** aim to:

- replace the existing offline MADMAX disk-spacing optimization,
- learn the boost factor from scratch with a neural network,
- use reinforcement learning as the main hardware controller,
- independently optimize every disk position online,
- replace the gradient-method boost-factor measurement,
- discover damaging hardware constraints by trial and error,
- claim that detector-state parameters \(\theta\) can always be uniquely separated from simulator discrepancy \(r\),
- or claim that arbitrary inferred local disk errors can be corrected if the online control basis cannot span them.

Instead, the project uses existing MADMAX tools and adds a safe, sample-efficient, statistically honest closed-loop calibration layer around them.

---

# 17. Final revised picture

The calibration algorithm is best summarized as:

\[
\boxed{\text{Nominal MADMAX configuration}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Baseline measurement, noise estimate, budget, and hard feasibility domain}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Propose safe booster correction and measurement action}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Set booster geometry and record achieved geometry}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Align antenna}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Measure selected observable / high-fidelity boost factor}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Jointly update detector state, discrepancy, noise, and drift}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Update optimizer-facing posterior predictive model}}
\]

\[
\downarrow
\]

\[
\boxed{\text{Repeat until improvement is no longer meaningful relative to uncertainty, cost, and physics goal}.}
\]

This is the revised locked high-level proposal.

---

# 18. Source anchors

The proposal above is based on the project structure discussed in this conversation, with the following external sources used as background anchors:

1. **MADMAX proof-of-principle booster setup:** The boost factor depends on disk spacing and can be tuned by changing distances between disks; the same paper discusses reflectivity/group-delay tuning, MHz-scale mechanical stability, and the area-law peak--bandwidth relationship.  
   <https://link.springer.com/article/10.1140/epjc/s10052-020-7985-8>

2. **MADMAX / dielectric-haloscope simulation with pre-optimized seeds:** This work uses pre-optimized reference configurations, notes that small disk-position changes significantly deform boost-factor curves, and explicitly invokes the area law.  
   <https://link.springer.com/article/10.1007/s41781-022-00091-5>

3. **MADMAX 3D requirements:** This work analyzes 3D effects, beam shape, antenna coupling, tilt, planarity, and surface requirements.  
   <https://arxiv.org/abs/2104.06553>

4. **MADMAX prototype boost-factor determination:** The dark-photon prototype paper discusses boost-factor determination from bead-pull/VNA measurements, receiver-chain corrections, antenna--booster distance effects, and boost-factor uncertainties.  
   <https://arxiv.org/html/2408.02368v1>

5. **MADMAX prototype axion search:** The prototype axion-search paper discusses booster-mode tuning, parasitic modes, mirror tilt sensitivity, and boost-factor uncertainty.  
   <https://arxiv.org/html/2409.11777v2>

6. **MADMAX piezo actuator qualification:** MADMAX piezo stages were tested at cryogenic temperatures and high magnetic fields.  
   <https://arxiv.org/abs/2305.12808>

7. **MADMAX disk-drive / laser-interferometer feedback context:** Public MADMAX prototype descriptions mention piezo-electric actuation and laser-interferometer feedback.  
   <https://www.physik.uni-hamburg.de/en/iexp/gruppe-garutti/news-ag-garutti/202203-madmax-cern.html>  
   <https://www.physik.uni-hamburg.de/en/iexp/gruppe-garutti/news-ag-garutti/202204-madmax-cern.html>

8. **SCBO / trust-region constrained Bayesian optimization:** BoTorch describes SCBO as a closed-loop constrained BO method and as a constrained version of TuRBO.  
   <https://botorch.org/docs/tutorials/scalable_constrained_bo/>

9. **Noise handling in BoTorch models:** BoTorch distinguishes homoskedastic, fixed-noise, and heteroskedastic-noise settings.  
   <https://botorch.org/docs/models>

10. **Multi-fidelity Bayesian optimization:** BoTorch provides multi-fidelity BO using knowledge-gradient acquisition, targeting the high-fidelity objective while using cheaper lower-fidelity evaluations.  
    <https://botorch.org/docs/v0.16.0/tutorials/multi_fidelity_bo>

11. **BO constraints:** BoTorch describes outcome constraints as black-box quantities modeled by surrogate models, motivating the distinction between learned constraints and known hard constraints.  
    <https://botorch.org/docs/constraints>

12. **Safe Bayesian optimization:** SafeOpt-style methods address optimization on physical systems where unsafe evaluations can damage hardware, using safety constraints separate from the objective.  
    <https://link.springer.com/article/10.1007/s10994-021-06019-1>

13. **Multi-objective Bayesian optimization:** BoTorch describes multi-objective BO as learning the Pareto front of optimal trade-offs.  
    <https://botorch.org/docs/multi_objective>

14. **Bayesian calibration and discrepancy modeling:** Kennedy and O’Hagan introduced Bayesian calibration with a model-discrepancy term.  
    <https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/KennedyOHagan_2002.pdf>

15. **Importance of discrepancy for physical parameter learning:** Brynjarsdóttir and O’Hagan emphasize that ignoring model discrepancy can bias physical-parameter inference and produce overconfident predictions.  
    <https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf>

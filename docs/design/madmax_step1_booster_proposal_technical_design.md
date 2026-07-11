# MADMAX Closed-Loop Calibration  
# Step 1 Technical Design: Propose the Next Booster-State Correction and Measurement Action

**Status:** Step-specific technical design note  
**Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3  
**Scope:** This document expands only **Step 1** of the seven-step closed-loop calibration algorithm. It gives a more technical, implementation-oriented structure for how the next booster-state correction and measurement action should be proposed. It does **not** specify detailed code, software classes, detector-control interfaces, or final numerical hyperparameters.

---

## 1. Role of Step 1 in the full calibration loop

Step 1 is the decision-making step of the outer calibration loop.

At iteration \(t\), the algorithm has accumulated calibration data:

\[
D_{1:t}
\]

from previous booster configurations, antenna alignments, measured observables, boost-factor determinations, geometry readbacks, and model updates.

Step 1 must use this information to decide what to try next.

The output of Step 1 is not a measured boost factor. It is a **proposal**:

\[
u_B^{(t+1)}
\]

and, if multiple measurement fidelities are available, a **measurement action**:

\[
\ell^{(t+1)}.
\]

In words, Step 1 answers:

> Given the simulator, the current posterior model, previous data, noise, budget, and constraints, which safe booster correction and which measurement action should be tried next?

The proposal is then passed to later steps of the loop:

- Step 2 sets the booster geometry.
- Step 3 aligns the antenna for that geometry.
- Step 4 performs the selected measurement.
- Steps 5 and 6 update the calibration and predictive models.

Step 1 therefore acts as the **outer-loop planner**.

---

## 2. What Step 1 is allowed to control

Step 1 proposes only the **booster-state correction**, not the full detector state.

The high-level booster-state control vector is:

\[
u_B =
\left(
 a_{\mathrm{disk}},
 z_{\mathrm{global}},
 z_{\mathrm{reflecting\ mirror}},
 z_{\mathrm{focusing\ mirror}}
\right).
\]

Here:

- \(a_{\mathrm{disk}}\) denotes a low-dimensional vector of disk-correction mode amplitudes.
- \(z_{\mathrm{global}}\) denotes the global translation of the relevant booster geometry.
- \(z_{\mathrm{reflecting\ mirror}}\) denotes the reflecting-mirror correction.
- \(z_{\mathrm{focusing\ mirror}}\) denotes the focusing-mirror correction.

The disk correction is represented as:

\[
q_{\mathrm{disk}}
=
q_{0,\mathrm{disk}} + B a_{\mathrm{disk}},
\]

where:

- \(q_{0,\mathrm{disk}}\) is the nominal disk configuration from the existing offline MADMAX optimization,
- \(B\) is the chosen low-dimensional correction basis,
- \(a_{\mathrm{disk}}\) are the online correction amplitudes.

The key rule is:

> Step 1 must propose only corrections that are actually available in the chosen online control basis.

If a detector-state inference model later identifies an error that cannot be spanned by this basis, that error may influence prediction and uncertainty, but it must not be treated as directly correctable by Step 1.

---

## 3. What Step 1 should not control

Step 1 should not directly optimize the antenna position:

\[
u_A = (x_{\mathrm{ant}}, y_{\mathrm{ant}}).
\]

Antenna alignment is handled later, in Step 3, for the booster state proposed by Step 1.

Therefore, Step 1 should think in terms of the nested objective:

\[
F(u_B)
=
\max_{u_A} J(q_0, u_B, u_A).
\]

In practice, Step 1 does not solve the inner antenna problem itself. Instead, it proposes \(u_B\) under the assumption that the antenna will subsequently be aligned for that booster geometry.

This means that the expected cost and uncertainty of the antenna-alignment step should be included in the planning logic, but antenna coordinates should not be mixed into the outer-loop booster proposal unless the project later chooses to collapse the outer and inner loops.

---

## 4. Inputs to Step 1

At iteration \(t\), Step 1 should receive the following inputs.

### 4.1 Nominal configuration and control map

The nominal configuration:

\[
q_0(W)
\]

and the mapping from control variables to physical geometry:

\[
q_B = q_{0,B} + \Delta q_B(u_B).
\]

This mapping includes the disk-correction basis \(B\), global z-positioning, reflecting-mirror correction, and focusing-mirror correction.

### 4.2 Current calibration data

The accumulated data should contain, at minimum:

\[
D_i =
(\tilde{u}_{B,i},
 \tilde{u}_{A,i},
 t_i,
 \ell_i,
 y_i,
 \sigma_i,
 \text{cost}_i,
 \text{validity flags}_i).
\]

Here:

- \(\tilde{u}_{B,i}\) is the achieved booster geometry or best available geometry readback.
- \(\tilde{u}_{A,i}\) is the achieved antenna position or best available antenna readback.
- \(t_i\) is the time or run index.
- \(\ell_i\) is the measurement fidelity or action type.
- \(y_i\) is the measured scalar outcome or proxy outcome.
- \(\sigma_i\) is the measurement uncertainty.
- \(\text{cost}_i\) is the measurement and movement cost.
- \(\text{validity flags}_i\) record whether the measurement was usable, marginal, failed, or affected by known problems.

The high-fidelity outcome is the scalar objective derived from the gradient-method boost-factor measurement:

\[
y_i = J_{\mathrm{HF}}(u_i)
\]

when \(\ell_i\) is the high-fidelity boost-factor determination.

### 4.3 Fast physics simulator

Step 1 should have access to the simulator prediction:

\[
\beta^2_{\mathrm{sim}}(\nu; q, \theta)
\]

and to the corresponding simulated scalar objective:

\[
J_{\mathrm{sim}}(u_B, \theta)
=
J\left(\beta^2_{\mathrm{sim}}(\nu; q_0, u_B, \theta)\right).
\]

The simulator should be treated as a guide, not as the final objective.

### 4.4 Posterior-predictive model from Step 6

Step 1 should use the current optimizer-facing posterior-predictive model:

\[
p(J_{\mathrm{HF}}(u_B) \mid D_{1:t}).
\]

This distribution should already account for uncertainty in:

- detector-state parameters,
- simulator discrepancy,
- measurement noise,
- drift,
- and, where possible, lower-fidelity measurement relationships.

A compact mean representation is:

\[
\mu_{\mathrm{pred},t}(u_B)
=
\mathbb{E}\left[
J_{\mathrm{sim}}(u_B, \theta) + r(u_B)
\mid D_{1:t}
\right],
\]

but the predictive variance is equally important:

\[
\sigma^2_{\mathrm{pred},t}(u_B).
\]

Step 1 should not use only a plug-in point estimate \(\hat{\theta}\) unless the project is explicitly running a simplified first version and clearly labels it as such.

### 4.5 Measurement-noise model

Step 1 should know the estimated uncertainty of each available measurement type:

\[
\sigma_J(u_B, \ell).
\]

The uncertainty may be:

- approximately constant,
- fidelity-dependent,
- frequency-dependent,
- configuration-dependent,
- or time-dependent.

If the noise is strongly configuration- or frequency-dependent, the model should be treated as heteroscedastic at the conceptual level.

### 4.6 Measurement-cost and movement-cost model

Step 1 should know the expected cost of a candidate action:

\[
C(u_B, \ell).
\]

This cost should include, conceptually:

- booster movement cost,
- expected antenna-alignment cost,
- measurement cost for fidelity \(\ell\),
- possible repeat-measurement cost,
- and any extra cost caused by large moves or re-baselining.

The cost model does not need to be perfect initially, but Step 1 should not treat all candidate actions as equally expensive if the real experiment does not.

### 4.7 Hard constraints

Step 1 must receive the known hard feasibility domain:

\[
\mathcal{U}_{\mathrm{hard}}.
\]

This includes constraints such as:

- disk collision avoidance,
- minimum disk gaps,
- piezo travel limits,
- CAD-defined geometry limits,
- actuator stroke limits,
- allowed global translations,
- maximum safe step size from the current achieved geometry,
- and any known safe operating envelopes.

Candidates outside this domain are never proposed.

### 4.8 Learned soft or unknown constraints

Step 1 may also receive statistical models for soft or unknown constraints:

\[
p(c_k(u_B, \ell) \leq 0 \mid D_{1:t}).
\]

Examples include:

- measurement-quality failure regions,
- unreliable gradient-method determination regions,
- bad coupling regions,
- parasitic-mode regions,
- unstable RF-response regions.

These are not damage-relevant hard constraints. They are constraints that can be learned statistically because evaluating them does not require intentionally risking the hardware.

### 4.9 Budget state

Step 1 should receive the remaining calibration budget:

\[
B_t.
\]

This budget may be expressed as:

- remaining high-fidelity boost-factor determinations,
- remaining lower-fidelity measurements,
- remaining cryogenic runtime,
- maximum allowed number of booster moves,
- maximum allowed total movement distance,
- or a combined budget score.

---

## 5. Internal representation used by Step 1

For numerical stability and comparability across physical variables, Step 1 should work internally with a normalized parameter vector:

\[
x_B \in [0,1]^d,
\]

where \(x_B\) maps bijectively to the physical booster correction vector \(u_B\).

The normalized representation is useful because the components of \(u_B\) have different physical units and scales. For example, disk-mode amplitudes, global z-shifts, and mirror corrections may have very different natural length scales.

The mapping should be explicit:

\[
u_B = T(x_B).
\]

Step 1 should output the physical command vector \(u_B\), not only the normalized \(x_B\).

---

## 6. Hard feasibility filtering

Before the Bayesian optimizer is allowed to score or propose a point, the candidate must satisfy hard constraints:

\[
u_B \in \mathcal{U}_{\mathrm{hard}}.
\]

This hard filter should be applied before hardware execution and preferably also during acquisition optimization.

The feasibility check should answer:

1. Is the proposed geometry inside all known mechanical limits?
2. Are all disk gaps safe?
3. Are all actuator strokes inside safe ranges?
4. Is the requested move from the current achieved geometry safe?
5. Does the proposal respect any known CAD or cryogenic operating envelope?

The important rule is:

> Damage-relevant constraints are not black-box constraints to be learned by failure. They are exact or conservative feasibility filters.

If uncertainty exists in a damage-relevant constraint, the safe domain should be shrunk by a conservative safety margin rather than explored experimentally.

---

## 7. The prediction model used by Step 1

Step 1 should use a prediction model for the high-fidelity objective:

\[
p(J_{\mathrm{HF}}(u_B) \mid D_{1:t}).
\]

At a practical level, this prediction combines:

\[
J_{\mathrm{sim}}(u_B, \theta)
\]

with a discrepancy term:

\[
r(u_B),
\]

and marginalizes over uncertainty:

\[
p(\theta, r, \sigma_J, \text{drift} \mid D_{1:t}).
\]

So the predictive distribution is conceptually:

\[
J_{\mathrm{HF}}(u_B)
\sim
J_{\mathrm{sim}}(u_B, \theta) + r(u_B) + \epsilon.
\]

This model gives Step 1 two essential quantities:

\[
\mu_{\mathrm{pred},t}(u_B)
\]

and:

\[
\sigma_{\mathrm{pred},t}(u_B).
\]

The mean tells Step 1 where the objective is expected to be high. The uncertainty tells Step 1 where the model is uncertain and where additional measurements may be valuable.

---

## 8. Candidate actions and measurement fidelities

Step 1 may choose not only a booster correction \(u_B\), but also a measurement action \(\ell\).

Possible action types include:

### 8.1 High-fidelity action

Use the full gradient-method boost-factor determination after setting the booster geometry and aligning the antenna.

This gives the most direct objective information:

\[
J_{\mathrm{HF}}(u_B).
\]

It is also expected to be expensive.

### 8.2 Lower-fidelity RF or proxy action

Use a faster measurement that is informative about the boost-factor objective but is not itself the final objective.

Examples may include reflectivity or RF-response proxies, depending on what the experimental setup makes available.

This action is useful when the optimizer needs information but the expected value of a full high-fidelity measurement is not yet high enough.

### 8.3 Replication action

Repeat a measurement at the current best point, baseline point, or a selected previous point.

This is useful when:

- the apparent improvement is comparable to measurement noise,
- the noise model is uncertain,
- drift is suspected,
- or the current best configuration needs validation.

### 8.4 Re-baseline action

Return to the nominal configuration or incumbent best configuration and remeasure it.

This is useful when:

- drift may have occurred,
- previous measurements are stale,
- the posterior model is becoming overconfident,
- or the baseline uncertainty dominates the decision.

### 8.5 Simulator-only planning action

The algorithm may evaluate many simulated candidates internally without moving hardware. This is not a calibration measurement, but it can be used inside Step 1 for candidate screening and acquisition optimization.

---

## 9. Acquisition objective for Step 1

The acquisition function is the quantity Step 1 maximizes to select the next action.

A useful conceptual form is:

\[
A_t(u_B, \ell)
=
\frac{
\text{utility}_t(u_B, \ell)
}{
C(u_B, \ell)
}
\times
P_{\mathrm{soft-safe}}(u_B, \ell),
\]

subject to:

\[
u_B \in \mathcal{U}_{\mathrm{hard}}.
\]

The utility should combine several terms:

\[
\text{utility}_t
=
\text{improvement value}
+
\lambda_{\mathrm{info}}\,\text{information value}
-
\lambda_{\mathrm{risk}}\,\text{risk penalty}.
\]

In words, the acquisition should balance:

1. expected improvement of the high-fidelity objective,
2. reduction of model uncertainty,
3. measurement cost,
4. soft-constraint feasibility,
5. and safety/risk.

The simplest version can ignore some terms, but the full conceptual design should keep them visible.

---

## 10. Improvement value

The improvement term asks:

> How likely is this candidate to improve the best currently validated high-fidelity objective?

Let:

\[
J_{\mathrm{best}}
\]

be the best validated high-fidelity score so far.

A conceptual expected-improvement term is:

\[
\mathrm{EI}_t(u_B)
=
\mathbb{E}\left[
\max(0, J_{\mathrm{HF}}(u_B) - J_{\mathrm{best}})
\mid D_{1:t}
\right].
\]

Because the measurements are noisy, a noisy expected-improvement or posterior-sampling variant is preferable to a noise-free acquisition.

If the expected improvement is smaller than the measurement uncertainty, Step 1 should not automatically spend a high-fidelity measurement. It should consider replication, lower-fidelity measurement, re-baselining, or stopping.

---

## 11. Information value

Because this is a calibration task, Step 1 should not be pure exploitation.

A candidate can be valuable if it reduces uncertainty in:

- the high-fidelity objective near the optimum,
- the detector-state/discrepancy model,
- the relation between lower-fidelity and high-fidelity measurements,
- the noise model,
- or the drift model.

Conceptually, define an information term:

\[
\mathrm{IG}_t(u_B, \ell)
\]

that measures the expected reduction in posterior uncertainty after making the measurement.

The acquisition may use:

\[
A_t(u_B, \ell)
\propto
\mathrm{EI}_t(u_B)
+
\lambda_{\mathrm{info}}\mathrm{IG}_t(u_B, \ell).
\]

The information weight \(\lambda_{\mathrm{info}}\) can be higher early in calibration and lower near final exploitation.

This turns Step 1 from simple maximization into an experiment-design step.

---

## 12. Cost awareness

A high-fidelity boost-factor determination may be much more expensive than a lower-fidelity proxy measurement.

Therefore, Step 1 should compare candidates by value per cost, not just absolute value:

\[
A_t(u_B, \ell)
\sim
\frac{\text{expected value of action}}{\text{expected cost of action}}.
\]

The cost should include not only the selected measurement but also the expected downstream cost of:

- moving the booster,
- aligning the antenna,
- performing the measurement,
- and possibly repeating the measurement if uncertainty is too large.

This is especially important if Step 1 is allowed to choose between cheap proxy measurements and expensive high-fidelity boost-factor determinations.

---

## 13. Multi-fidelity logic

If lower-fidelity measurements are available, Step 1 should use a multi-fidelity view.

The optimizer should model outcomes as:

\[
y_\ell(u_B)
\]

where \(\ell\) indexes the fidelity or measurement type.

The high-fidelity target remains:

\[
J_{\mathrm{HF}}(u_B).
\]

Lower-fidelity data are useful only insofar as they improve the prediction of the high-fidelity objective or reduce uncertainty in the detector model.

A suitable high-level strategy is:

- use low-fidelity measurements for broad exploration and model correction,
- reserve high-fidelity gradient-method boost-factor determinations for candidates with high expected value, validation, and final selection,
- use high-fidelity replication when the apparent improvement is near the noise level.

The measurement action selected by Step 1 is therefore:

\[
(u_B^{(t+1)}, \ell^{(t+1)}).
\]

---

## 14. Trust-region logic

The search should not explore the full allowed space blindly at every iteration.

Step 1 should operate inside a trust region:

\[
\mathcal{T}_t \subseteq \mathcal{U}_{\mathrm{hard}}.
\]

The trust region is centered near a good known configuration, typically the current best validated achieved booster state.

The trust region should:

- expand when recent proposals produce validated improvement,
- shrink when proposals fail, produce no improvement, or become too uncertain,
- respect maximum movement limits from the current achieved geometry,
- and never exceed hard feasibility constraints.

The purpose of the trust region is to keep the search local and sample-efficient around the nominal MADMAX configuration and around the currently best calibrated state.

---

## 15. Soft-constraint handling

For learned soft constraints, Step 1 should estimate a probability of feasibility:

\[
P_{\mathrm{soft-safe}}(u_B, \ell)
=
P(c_k(u_B, \ell) \leq 0\ \forall k \mid D_{1:t}).
\]

The acquisition can then either:

1. multiply the utility by the feasibility probability,
2. restrict proposals to points with feasibility probability above a threshold,
3. or both.

A conceptual constrained acquisition is:

\[
A_t(u_B, \ell)
=
\mathrm{Utility}_t(u_B, \ell)
\cdot
P_{\mathrm{soft-safe}}(u_B, \ell)
\cdot
\frac{1}{C(u_B, \ell)}.
\]

This is only for constraints whose violation is non-damaging. Hard safety constraints remain outside this learned model and are enforced directly.

---

## 16. Noise-aware decision rules

Step 1 should explicitly compare predicted improvements to uncertainty.

Let:

\[
\Delta\mu_t(u_B)
=
\mu_{\mathrm{pred},t}(u_B)-J_{\mathrm{best}}.
\]

If:

\[
\Delta\mu_t(u_B)
\lesssim
\sigma_{\mathrm{relevant}},
\]

then the candidate is not clearly better than the current best configuration.

In that case, Step 1 should consider:

- repeating the current best measurement,
- repeating the baseline measurement,
- choosing a lower-fidelity information-gathering action,
- reducing the trust region,
- or stopping if the remaining expected improvement is not experimentally resolvable.

This prevents the optimizer from chasing noise.

---

## 17. Drift-aware decision rules

If the calibration loop runs long enough for drift to matter, Step 1 should include drift indicators.

Examples:

- time since last baseline measurement,
- posterior uncertainty attributed to drift,
- disagreement between repeated measurements,
- discrepancy between commanded and achieved geometry,
- unexplained degradation at a known good point.

If drift indicators are high, Step 1 should be allowed to propose a re-baseline or incumbent-recheck action rather than a new booster correction.

This keeps the posterior model from becoming stale.

---

## 18. Multi-objective option

If the scalar objective \(J\) is not fixed by the physics team, Step 1 should not prematurely collapse the problem into one arbitrary scalar.

Instead, Step 1 may treat the objective as multi-objective, for example:

\[
\left(
\text{peak boost},
\text{bandwidth or band flatness},
\text{scan-rate proxy},
\text{cost or risk}
\right).
\]

Then Step 1 proposes candidates that improve the estimated Pareto front.

This is only needed if the operating point on the peak--bandwidth trade-off is not fixed before calibration.

For a first implementation, it is acceptable to use a physics-approved scalar objective \(J\). But the design should preserve the option to switch to a Pareto-front view later.

---

## 19. Recommended algorithmic variants

Step 1 can be implemented in increasing levels of sophistication.

### 19.1 Minimal viable Step 1

Use this version first if the project needs a clean baseline.

- Use one fixed scalar objective \(J\).
- Use only high-fidelity boost-factor measurements.
- Use hard feasibility filtering.
- Use a trust-region constrained Bayesian optimizer.
- Use a noise-aware acquisition function.
- Use fixed or empirically estimated observation noise.
- Output one safe booster correction per outer iteration.

This version tests whether closed-loop booster correction is feasible at all.

### 19.2 Recommended Step 1

Use this as the main project target.

- Use a physics-informed posterior-predictive model.
- Use trust-region constrained Bayesian optimization.
- Use hard feasibility filtering.
- Use learned soft-constraint models where appropriate.
- Use measured noise estimates.
- Include measurement cost.
- Allow replication and re-baselining actions.
- Output both the booster correction and the measurement action.

This version matches the revised high-level proposal.

### 19.3 Extended Step 1

Use this if the measurement hierarchy and budget make it worthwhile.

- Add multi-fidelity acquisition.
- Add explicit information-gain terms for calibration.
- Add multi-objective Pareto-front logic if the physics objective is not fixed.
- Add stronger drift-aware action selection.

This version is more powerful, but it should only be attempted after the minimal and recommended versions are understood.

---

## 20. Candidate-generation procedure

A practical Step 1 proposal procedure should follow this order.

### Stage 1: Define the current search domain

Construct the current trust region:

\[
\mathcal{T}_t
\]

and intersect it with the hard feasibility domain:

\[
\mathcal{S}_t
=
\mathcal{T}_t \cap \mathcal{U}_{\mathrm{hard}}.
\]

Only candidates in \(\mathcal{S}_t\) are considered.

### Stage 2: Build the acquisition function

Construct an acquisition function using:

- posterior predictive mean,
- posterior predictive uncertainty,
- measurement noise,
- measurement cost,
- soft-constraint feasibility,
- and optional information value.

### Stage 3: Optimize or search the acquisition

Find candidate actions:

\[
(u_B, \ell)
\]

with high acquisition value.

The acquisition optimization can use continuous optimization, a candidate pool, or a hybrid of both. The method is a software choice, not a physics requirement.

### Stage 4: Apply safety and budget gates

Before returning a proposal, check:

- hard feasibility,
- maximum safe movement from current achieved geometry,
- remaining budget,
- expected measurement cost,
- expected improvement relative to noise,
- and soft-constraint probability.

### Stage 5: Select the final action

Return one of the following:

1. a new booster correction and measurement action,
2. a repeat measurement at the incumbent best point,
3. a baseline remeasurement,
4. a lower-fidelity information-gathering action,
5. or a recommendation to stop because no meaningful action remains.

---

## 21. Step 1 output package

Step 1 should return a structured proposal, not just a vector of numbers.

The output should include:

\[
\text{Proposal}_{t+1}
=
\left(
 u_B^{(t+1)},
 \ell^{(t+1)},
 \text{metadata}
\right).
\]

The metadata should include:

- predicted high-fidelity objective mean,
- predicted high-fidelity objective uncertainty,
- expected improvement or acquisition value,
- measurement fidelity/action type,
- expected cost,
- hard-feasibility certificate,
- soft-constraint feasibility probability if applicable,
- trust-region state,
- reason for action selection,
- and fallback recommendation if the proposal cannot be executed.

This metadata is important for later debugging, scientific interpretation, and deciding whether the calibration process is actually useful.

---

## 22. Fallback behavior

Step 1 should have defined behavior when no good candidate exists.

Possible fallback actions are:

### 22.1 Repeat the incumbent

Use when the current best configuration appears good but its uncertainty is too large.

### 22.2 Re-measure the baseline

Use when drift or stale calibration state is suspected.

### 22.3 Query a cheaper proxy

Use when high-fidelity improvement is uncertain but a lower-fidelity measurement could reduce uncertainty cheaply.

### 22.4 Shrink the trust region

Use when proposed moves repeatedly fail to improve the objective or violate soft feasibility.

### 22.5 Stop

Use when expected improvement is small compared with uncertainty and cost.

---

## 23. Conceptual pseudocode

```text
Input:
    D_t                         # accumulated calibration data
    q0(W)                       # nominal configuration
    control_map                 # maps u_B to booster geometry
    predictive_model_t          # p(J_HF(u_B) | D_t)
    hard_constraints            # exact known feasibility domain
    soft_constraint_models      # optional learned feasibility models
    noise_model                 # sigma_J(u_B, fidelity)
    cost_model                  # C(u_B, fidelity)
    trust_region_state          # current local search region
    budget_state                # remaining calibration budget

Output:
    proposal_{t+1} = (u_B, fidelity/action, metadata)

Procedure:

    1. Build current search set:
           S_t = trust_region ∩ hard_feasible_domain

    2. If drift or stale baseline is suspected:
           consider baseline or incumbent remeasurement action

    3. Construct acquisition A_t(u_B, fidelity):
           value from posterior predictive improvement
           plus optional information gain
           weighted by soft-constraint feasibility
           divided or penalized by expected cost

    4. Search for high-acquisition candidates in S_t.

    5. Apply final gates:
           hard feasibility
           maximum safe step size
           budget availability
           expected improvement versus noise
           soft feasibility threshold

    6. If a candidate passes:
           return booster correction u_B and measurement action fidelity

       Else if uncertainty at incumbent is too large:
           return replication action

       Else if drift is suspected:
           return re-baseline action

       Else if cheap information could help:
           return lower-fidelity information action

       Else:
           return stop recommendation
```

---

## 24. Diagnostics for Step 1

Each Step 1 call should produce diagnostics that can be inspected by the experimental team.

Useful diagnostics include:

- proposed \(u_B\) in physical units,
- distance from nominal configuration,
- distance from current achieved configuration,
- predicted \(J_{\mathrm{HF}}\) mean and uncertainty,
- expected improvement,
- predicted measurement cost,
- probability of soft feasibility,
- active hard constraints or safety margins,
- trust-region size,
- reason for choosing the measurement fidelity,
- whether the action is exploitative, exploratory, replication-based, or re-baselining,
- and whether expected improvement is large compared with measurement noise.

These diagnostics are part of the scientific output of the calibration system. They make the optimizer auditable.

---

## 25. Minimal validation tests for Step 1 before hardware use

Before Step 1 is connected to real hardware, it should be tested in simulation.

At the Step-1 level, the validation should check:

1. **Constraint filtering:** unsafe candidates are never proposed.
2. **Noise response:** the optimizer replicates or stops when improvement is smaller than noise.
3. **Budget response:** expensive measurements are not used when cheap ones give comparable information.
4. **Trust-region behavior:** the region expands after success and shrinks after failure.
5. **Control-basis consistency:** uncorrectable inferred errors do not produce impossible control proposals.
6. **Drift response:** re-baseline actions are proposed when simulated drift makes old data stale.
7. **Acquisition behavior:** candidates are chosen for plausible improvement or information value, not only because uncertainty is large in irrelevant regions.

These tests do not prove detector performance, but they reduce the risk that the decision logic itself is flawed.

---

## 26. Main design choices still to be fixed with the MADMAX team

The following choices should be fixed before implementation:

1. **Dimension and content of \(u_B\):** which disk-correction modes and z/mirror/focusing variables are included.
2. **Hard feasibility domain:** exact geometry, travel, gap, and safety constraints.
3. **Scalar versus multi-objective formulation:** whether \(J\) is fixed or whether a Pareto front is needed.
4. **Available measurement fidelities:** which proxy measurements are available and how they relate to the high-fidelity gradient-method boost-factor determination.
5. **Measurement-noise model:** whether fixed-noise, replicated-noise, or heteroscedastic-noise modeling is needed.
6. **Budget model:** how many high-fidelity and lower-fidelity measurements are realistic.
7. **Allowed risk level for learned soft constraints:** what probability of soft feasibility is acceptable.
8. **Drift policy:** how often to re-baseline and how stale old measurements may become.
9. **Trust-region policy:** how far the calibration may move away from \(q_0\) and from the current achieved geometry.

---

## 27. Recommended Step 1 design in one paragraph

Step 1 should be implemented as a **physics-informed, trust-region, constrained, budget-aware Bayesian decision step**. It proposes a safe booster-state correction \(u_B\) and a measurement action \(\ell\) by maximizing an acquisition function built from the posterior predictive high-fidelity objective \(p(J_{\mathrm{HF}}(u_B) \mid D)\). The acquisition should use the fast simulator, the joint detector-state/discrepancy/noise/drift posterior, measurement uncertainty, measurement cost, hard feasibility constraints, and learned soft constraints. It should be able to choose a new high-fidelity measurement, a lower-fidelity information-gathering measurement, a repeat measurement, a re-baseline action, or a stop recommendation. The most important safety rule is that known damaging constraints are enforced exactly before proposal and are never learned by trial and error.

---

## 28. Source anchors

This Step-1 design note is based on the current MADMAX closed-loop calibration proposal and the method anchors below.

1. **Current parent proposal:** `madmax_closed_loop_calibration_proposal.md`.

2. **SCBO / trust-region constrained Bayesian optimization:** BoTorch describes SCBO as a closed-loop constrained Bayesian optimization method and as a constrained version of TuRBO.  
   <https://botorch.org/docs/v0.17.1/tutorials/scalable_constrained_bo>

3. **BoTorch outcome constraints:** BoTorch distinguishes black-box outcome constraints, which are modeled by surrogates, from constraints that should be imposed directly on the candidate domain.  
   <https://botorch.org/docs/constraints>

4. **SafeOpt / safe Bayesian optimization:** Safe BO restricts optimization to a safe set when unsafe evaluations are unacceptable for physical systems.  
   <https://link.springer.com/article/10.1007/s10994-021-06019-1>

5. **Multi-fidelity Bayesian optimization:** BoTorch describes multi-fidelity BO with knowledge-gradient acquisition, where cheaper evaluations help optimize the target high-fidelity objective.  
   <https://botorch.org/docs/v0.16.0/tutorials/multi_fidelity_bo>

6. **BoTorch model/noise documentation:** BoTorch distinguishes fixed-noise and heteroscedastic-noise settings at the modeling level.  
   <https://botorch.org/docs/models>

7. **Multi-objective Bayesian optimization:** BoTorch describes multi-objective BO as learning a Pareto front of optimal trade-offs.  
   <https://botorch.org/docs/multi_objective>

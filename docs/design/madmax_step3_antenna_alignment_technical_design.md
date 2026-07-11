# MADMAX Closed-Loop Calibration
# Step 3 Technical Design: Align the Antenna for a Fixed Booster State

**Status:** Step-specific technical design note  
**Parent proposal:** `madmax_closed_loop_calibration_proposal.md`, version 3  
**Scope:** This document expands only **Step 3** of the seven-step closed-loop calibration algorithm. It describes the more technical structure of the antenna-alignment subproblem after the booster geometry has already been proposed and set. It does **not** specify detector-control software, final motor-control parameters, final numerical hyperparameters, or experiment-specific DAQ interfaces.

---

## 1. Role of Step 3 in the full calibration loop

Step 3 happens after the outer loop has already selected a booster-state correction and after the detector has moved to that booster geometry.

At this point, the booster state is fixed:

\[
\tilde{u}_B^{(t+1)}
\]

where the tilde indicates the achieved booster geometry, not merely the commanded geometry.

Step 3 then optimizes the antenna position:

\[
u_A = (x_{\mathrm{ant}}, y_{\mathrm{ant}}).
\]

The purpose of Step 3 is not to retune the disks, the reflecting mirror, or the focusing mirror. Those variables belong to the booster-state proposal from Step 1 and the booster-setting operation from Step 2.

The purpose of Step 3 is narrower:

> For the fixed booster geometry, find the antenna position that gives the best coupling to the boosted signal, or to a validated proxy for that coupling.

The high-level parent proposal defines Step 3 as optimizing \(x_{\mathrm{ant}}\) and \(y_{\mathrm{ant}}\) for the current booster state, using either two-dimensional Bayesian optimization or a two-dimensional scan plus local fit.

Step 3 is therefore an **inner alignment subproblem** inside the larger calibration loop.

---

## 2. Why Step 3 should remain separate from the booster optimization

The booster variables and antenna variables affect the detector in different ways.

The booster geometry determines the electromagnetic response of the dielectric stack and mirror system. The antenna position mainly controls how efficiently the receiver system couples to the emitted or reflected field.

This separation is physically motivated. A dielectric haloscope such as MADMAX uses dielectric disks and a mirror to boost the axion-induced signal; a focusing mirror couples the field to an antenna connected to the receiver chain. The transverse behavior of the field is explicitly important for the complete signal-power description, and antenna/focusing geometry determines the beam coupled into the receiver system.

Therefore Step 3 should be treated as:

\[
\text{fixed booster state} \quad \rightarrow \quad \text{optimize receiver coupling}.
\]

It should not be treated as:

\[
\text{jointly change disks, mirror, focusing mirror, and antenna at once}.
\]

Keeping antenna alignment separate has three advantages:

1. The antenna problem is only two-dimensional in the current proposal.
2. It can usually be solved with a cheaper inner procedure than the full outer-loop calibration.
3. It prevents the outer-loop optimizer from confusing booster-response changes with receiver-coupling changes.

---

## 3. Inputs to Step 3

At iteration \(t+1\), Step 3 should receive the following inputs.

### 3.1 Fixed achieved booster state

The booster state after Step 2:

\[
\tilde{u}_B^{(t+1)}.
\]

This includes the achieved disk-correction modes, global z-position, reflecting-mirror correction, and focusing-mirror correction.

Step 3 should use the achieved geometry whenever available, rather than the commanded geometry alone.

### 3.2 Current antenna state

The current commanded and achieved antenna positions:

\[
u_{A,\mathrm{cmd}}^{\mathrm{current}}
\]

and:

\[
\tilde{u}_{A}^{\mathrm{current}}.
\]

If the antenna has reliable position readback, the achieved value should be used for modeling and logging.

### 3.3 Allowed antenna domain

The antenna search domain:

\[
\mathcal{U}_{A,\mathrm{hard}}
\subset
\mathbb{R}^2.
\]

This domain contains all hard mechanical and geometric constraints on the antenna x/y motion.

Examples include:

- x/y travel limits,
- collision or clearance limits,
- cable-strain limits,
- maximum allowed single-step movement,
- known exclusion regions,
- and any CAD-defined safe envelope.

Any candidate outside this domain is rejected before it is ever sent to the hardware.

### 3.4 Alignment objective or proxy objective

Step 3 needs an alignment score. Ideally this score is directly connected to the same physics figure of merit used in the outer loop:

\[
J(q_0, \tilde{u}_B^{(t+1)}, u_A).
\]

In practice, evaluating the full high-fidelity boost-factor objective at every candidate antenna position may be too expensive. Therefore Step 3 may use a cheaper antenna-coupling proxy:

\[
A_\ell(u_A \mid \tilde{u}_B^{(t+1)}, W),
\]

where \(\ell\) denotes the measurement type or fidelity.

The proxy may be based on RF response, antenna-coupling observables, beam-overlap information, or another fast observable agreed upon with the experimental team.

The proxy must not be treated as final proof of improved boost factor. The final selected configuration still has to be evaluated in Step 4 using the measurement action selected by Step 1, and the final accepted best configuration must be validated with the high-fidelity boost-factor determination.

### 3.5 Previous antenna-alignment history

Step 3 should also receive earlier antenna-alignment data:

\[
D_A =
\{(\tilde{u}_{A,i}, \tilde{u}_{B,i}, t_i, \ell_i, A_i, \sigma_{A,i})\}.
\]

This history can be useful because the optimal antenna position may vary smoothly as the booster state changes. Previous alignments can provide a warm start for the current inner alignment.

### 3.6 Step-3 measurement budget

The antenna alignment should have its own local budget:

\[
B_A^{(t+1)}.
\]

This budget may be expressed in:

- number of antenna-position measurements,
- time spent on antenna alignment,
- allowed movement distance,
- or total measurement cost.

The inner alignment should not consume an uncontrolled amount of the total calibration budget.

---

## 4. Outputs of Step 3

Step 3 should output the antenna position chosen for the current booster state:

\[
u_A^{(t+1)}.
\]

More precisely, it should output both:

\[
u_{A,\mathrm{cmd}}^{(t+1)}
\]

and, when possible:

\[
\tilde{u}_{A}^{(t+1)}.
\]

The achieved value \(\tilde{u}_{A}^{(t+1)}\) is the value that should be passed to Step 4 and logged for the statistical model update.

Step 3 should also output:

- the alignment score or proxy score at the selected antenna position,
- the uncertainty of that score,
- the data collected during the alignment scan or BO process,
- a quality flag describing whether the alignment was reliable,
- and a record of whether the final selected antenna position was confirmed by a repeat measurement.

A compact Step-3 output record could be written as:

\[
R_A^{(t+1)} =
\left(
\tilde{u}_B^{(t+1)},
 u_{A,\mathrm{cmd}}^{(t+1)},
 \tilde{u}_A^{(t+1)},
 A^{(t+1)},
 \sigma_A^{(t+1)},
 \text{method},
 \text{quality flag},
 D_A^{(t+1)}
\right).
\]

---

## 5. The Step-3 optimization problem

For the fixed booster state, the ideal antenna-alignment problem is:

\[
u_A^\star(\tilde{u}_B)
=
\arg\max_{u_A \in \mathcal{U}_{A,\mathrm{hard}}}
J(q_0, \tilde{u}_B, u_A).
\]

Because the full high-fidelity objective may be too expensive for the inner antenna loop, the practical objective is usually:

\[
u_A^\star(\tilde{u}_B)
=
\arg\max_{u_A \in \mathcal{U}_{A,\mathrm{hard}}}
A_\ell(u_A \mid \tilde{u}_B, W),
\]

where \(A_\ell\) is an alignment score measured with a cheaper fidelity \(\ell\).

The important design requirement is:

> The alignment score \(A_\ell\) should be demonstrably correlated with the high-fidelity boost-factor objective for the current class of booster states.

If that correlation is not established, Step 3 should be treated as exploratory or diagnostic, not as a guaranteed improvement of the final objective.

---

## 6. Choosing the alignment score

The alignment score should be chosen together with the experimental team. At this technical-design level, the document should define the allowed categories of score rather than fixing one final observable.

### 6.1 High-fidelity score

The most direct score is the high-fidelity scalar objective:

\[
A_{\mathrm{HF}}(u_A) = J(q_0, \tilde{u}_B, u_A).
\]

This is conceptually clean but likely too expensive for many inner-loop antenna trials.

Use this only when:

- the antenna search is extremely small,
- the measurement budget allows it,
- or a final confirmation is needed.

### 6.2 RF or coupling proxy score

A practical alignment score may be based on faster electromagnetic response data:

\[
A_{\mathrm{proxy}}(u_A)
=
\mathcal{S}_A\left(y_{\mathrm{RF}}(\nu; \tilde{u}_B, u_A), W\right),
\]

where \(y_{\mathrm{RF}}\) denotes the measured RF observable and \(\mathcal{S}_A\) converts it into a scalar alignment score.

Possible proxy concepts include:

- coupling to the expected fundamental beam mode,
- received or reflected power in a selected frequency window,
- suppression of obviously bad coupling features,
- overlap with a simulated or measured beam profile,
- or a time-gated RF-response quantity if applicable.

The exact proxy must be validated experimentally against the final boost-factor objective.

### 6.3 Model-assisted score

If the beam shape and receiver coupling are simulated or measured in advance, Step 3 may use a model-assisted score:

\[
A_{\mathrm{model}}(u_A)
=
\mathbb{E}\left[J_{\mathrm{HF}}(\tilde{u}_B,u_A) \mid D, \text{coupling model}\right].
\]

This is useful when cheap measurements provide partial information but the optimizer still needs a prediction of the high-fidelity effect.

### 6.4 Multi-frequency alignment score

The antenna position should not necessarily be optimized at a single frequency if the booster setting targets a finite window \(W\).

A band-aware alignment score could be:

\[
A(u_A \mid \tilde{u}_B, W)
=
\operatorname{Aggregate}_{\nu \in W}
\left[
 w(\nu)\, C(\nu; \tilde{u}_B,u_A)
\right],
\]

where \(C\) is a coupling observable and \(w(\nu)\) reflects the physics objective.

For a narrow-band confirmation setting, \(w(\nu)\) may be concentrated near the target frequency.

For a broadband scan setting, \(w(\nu)\) should reflect the chosen scan-rate or sensitivity objective.

---

## 7. Recommended algorithmic structure

Because Step 3 is two-dimensional, the algorithm should be simpler than the outer booster-state optimizer.

The recommended structure is a **hybrid local alignment strategy**:

\[
\boxed{
\text{initial local scan or cross scan}
\rightarrow
\text{local fit}
\rightarrow
\text{optional 2D GP Bayesian optimization}
\rightarrow
\text{confirmation measurement}
}
\]

This hybrid strategy keeps the alignment efficient when the coupling surface is simple, but still allows more flexible optimization when the surface is distorted or noisy.

---

## 8. Stage 1: initialize the antenna search

Before any new antenna measurements are taken, Step 3 should choose a starting antenna position.

Useful starting points include:

1. the previous best antenna position for the previous booster state,
2. the antenna position predicted by the coupling or beam model,
3. the center of the known safe antenna domain,
4. a previously calibrated reference position,
5. or the current achieved antenna position if the booster change was small.

The initial position is denoted:

\[
u_{A,0}^{(t+1)}.
\]

If the booster-state change is small and the previous antenna alignment is expected to remain valid, Step 3 should first test the previous best antenna position before launching a full scan.

This avoids wasting time on unnecessary inner-loop alignment.

---

## 9. Stage 2: cheap validation at the incumbent position

A simple but important first measurement is the incumbent test.

Measure the alignment score at:

\[
u_{A,\mathrm{inc}}.
\]

Then compare it with the last known good alignment score, adjusted for any expected change in booster state.

If the score is still good within uncertainty, Step 3 may choose to keep the current antenna position and skip a full alignment.

Conceptually:

\[
A(u_{A,\mathrm{inc}})
\geq
A_{\mathrm{expected}} - \kappa\sigma_A.
\]

If this condition holds, the algorithm can return the incumbent antenna position with a quality flag such as:

\[
\text{alignment reused after validation}.
\]

If the score has degraded, Step 3 proceeds to active alignment.

---

## 10. Stage 3: local scan and local fit

The default first active-alignment method should be a small local scan.

For example, collect measurements at:

\[
(x_0,y_0),
\quad
(x_0 \pm \Delta x,y_0),
\quad
(x_0,y_0 \pm \Delta y),
\]

and, if needed, at diagonal points:

\[
(x_0 \pm \Delta x,y_0 \pm \Delta y).
\]

The exact scan pattern should be chosen according to the expected beam width, motor precision, and measurement cost.

After collecting the local scan data, fit a simple local response model.

Near a smooth maximum, a quadratic model is sufficient:

\[
A(x,y)
\approx
c
+ g_x(x-x_0)
+ g_y(y-y_0)
+ \frac{1}{2}
\begin{bmatrix}x-x_0 & y-y_0\end{bmatrix}
H
\begin{bmatrix}x-x_0 \\ y-y_0\end{bmatrix}.
\]

The fitted optimum is accepted only if:

- it lies inside the hard-safe domain,
- it is not too far outside the scanned region,
- the fitted curvature is physically plausible,
- the expected improvement is larger than measurement uncertainty,
- and a confirmation measurement at or near the predicted optimum agrees with the fit.

This local-fit mode is attractive because the expected receiver-coupling surface may often be smooth near the optimum.

---

## 11. Stage 4: switch to 2D Gaussian-process Bayesian optimization when needed

The local scan should not be forced to work when the response surface is distorted, noisy, or multi-modal.

Step 3 should switch to 2D Gaussian-process Bayesian optimization if any of the following occur:

- the local quadratic fit is unstable,
- the predicted maximum lies outside the scanned region,
- the curvature estimate is inconsistent with the expected beam scale,
- repeated measurements show significant noise or drift,
- the score surface appears multi-modal,
- RF features suggest antenna-booster resonances or higher-order-mode effects,
- or the local scan fails to produce a statistically meaningful improvement.

The GP-BO version models:

\[
A(u_A \mid \tilde{u}_B)
\sim
\mathcal{GP}(m_A(u_A), k_A(u_A,u_A')).
\]

Because the input dimension is only two, a standard GP is appropriate in principle.

The acquisition function may be one of:

- noisy expected improvement,
- upper confidence bound,
- Thompson sampling,
- or knowledge-gradient-style acquisition if information gain is important.

The practical acquisition should include:

\[
\text{expected improvement},
\quad
\text{measurement uncertainty},
\quad
\text{movement cost},
\quad
\text{safe-domain constraints}.
\]

Thus the selected next antenna point is conceptually:

\[
u_{A,\mathrm{next}}
=
\arg\max_{u_A \in \mathcal{U}_{A,\mathrm{hard}}}
\alpha_A(u_A),
\]

where \(\alpha_A\) is the chosen Step-3 acquisition function.

---

## 12. Hard constraints and learned alignment constraints

Step 3 must make the same hard-versus-learned distinction as the outer loop.

### 12.1 Known hard antenna constraints

Known hard constraints define the allowed antenna domain:

\[
\mathcal{U}_{A,\mathrm{hard}}.
\]

They include any movement or geometry limit that must never be violated.

Candidate antenna positions outside \(\mathcal{U}_{A,\mathrm{hard}}\) are never proposed, never measured, and never treated as learnable failures.

### 12.2 Unknown or soft alignment constraints

Other limitations may be empirical and non-damaging, for example:

- a proxy observable becomes unreliable,
- the VNA signal quality is poor,
- the coupling score is dominated by parasitic features,
- the antenna response is unstable,
- or the measurement has excessive uncertainty.

These can be treated as learned soft constraints or quality flags.

A candidate can then be scored not only by expected improvement, but also by probability of reliable measurement:

\[
\alpha_A^{\mathrm{constrained}}(u_A)
=
\alpha_A(u_A)\,P(\text{reliable measurement}\mid u_A,D_A).
\]

This logic is appropriate only for non-damaging empirical constraints. It is not a substitute for exact hard-constraint filtering.

---

## 13. Noise handling in Step 3

Antenna-alignment measurements may be noisy. Step 3 should not chase changes smaller than the measurement noise.

The alignment data should therefore include:

\[
\sigma_A(u_A,\ell,t),
\]

an uncertainty estimate for the alignment score.

Possible sources of uncertainty include:

- RF measurement noise,
- short-term drift,
- finite repeatability of antenna motion,
- beam or coupling fluctuations,
- frequency-dependent proxy variation,
- and imperfect time gating or background subtraction if those are used.

The Step-3 decision rule should compare expected improvement to the uncertainty:

\[
\Delta A_{\mathrm{expected}}
\gtrsim
\text{alignment uncertainty threshold}.
\]

If the predicted improvement is smaller than the uncertainty, Step 3 should either:

- repeat the measurement at the current best point,
- repeat the measurement at the candidate point,
- choose a cheaper diagnostic measurement,
- stop the inner alignment and return the incumbent,
- or report that the antenna alignment is not experimentally resolvable at the current precision.

---

## 14. Drift and hysteresis handling

Step 3 should record achieved antenna positions, not only commanded positions.

The alignment data should distinguish:

\[
u_{A,\mathrm{cmd}}
\]

from:

\[
\tilde{u}_{A,\mathrm{achieved}}.
\]

This is important because hysteresis, backlash, creep, and finite positioning reproducibility can smear the learned map if the optimizer only sees commanded positions.

The Step-3 data record should also include time or run index:

\[
D_{A,i}
=
(\tilde{u}_{A,i}, \tilde{u}_{B}, t_i, \ell_i, A_i, \sigma_{A,i}).
\]

For longer antenna alignments, Step 3 should periodically remeasure the incumbent or a reference point to detect drift.

If the reference measurement changes significantly, Step 3 should flag the alignment as drift-limited and pass that information to Step 5.

---

## 15. Relationship to Step 1 and Step 4

Step 3 sits between the outer-loop proposal and the measured objective.

### 15.1 Relationship to Step 1

Step 1 proposes the booster state and possibly the measurement action or fidelity.

Step 3 does not override the Step-1 booster proposal.

However, Step 3 may use information from Step 1, such as:

- the measurement budget allocated to this iteration,
- the selected measurement fidelity,
- the current posterior predictive model,
- and the expected value of antenna alignment for this booster state.

If Step 1 chooses a low-fidelity measurement action, Step 3 should avoid using a more expensive high-fidelity measurement unless the control system explicitly allows an exception.

### 15.2 Relationship to Step 4

Step 4 measures the selected observable for the complete detector configuration.

Therefore Step 3 must pass to Step 4:

\[
q(q_0, \tilde{u}_B^{(t+1)}, \tilde{u}_A^{(t+1)}).
\]

Step 4 should then use the achieved antenna position in the measurement record.

If Step 3 only used a proxy score, Step 4 is where that proxy-guided alignment is tested against the selected higher-level measurement.

---

## 16. How Step-3 data should feed the global statistical model

The inner antenna-alignment data should not be thrown away.

Even if Step 3 uses only a proxy observable, the data can help Step 5 infer coupling errors, antenna offsets, or discrepancy terms.

Therefore Step 3 should pass its full local dataset forward:

\[
D_A^{(t+1)}
=
\{(\tilde{u}_{A,j}, \tilde{u}_B^{(t+1)}, t_j, \ell_j, A_j, \sigma_{A,j})\}_{j=1}^{n_A}.
\]

Step 5 can then decide whether these data are:

- direct calibration data,
- lower-fidelity proxy data,
- diagnostic-only data,
- or data to be excluded because of quality failures.

This keeps Step 3 technically useful even when a local alignment does not improve the final boost-factor objective.

---

## 17. Recommended default policy

A practical default Step-3 policy is:

1. Use the previous best or model-predicted antenna position as the starting point.
2. Measure the alignment proxy at that point.
3. If the incumbent is still good within uncertainty, keep it and skip a full scan.
4. If not, perform a small local scan around the starting point.
5. Fit a local quadratic response model.
6. If the fit is reliable and predicts a safe optimum, move to that point.
7. Confirm the predicted optimum with a repeat measurement.
8. If the fit is unreliable, switch to 2D GP Bayesian optimization.
9. Stop the inner loop when improvement is smaller than uncertainty or the local budget is exhausted.
10. Return the best achieved antenna position and the full alignment record.

This default policy is intentionally conservative. It avoids spending too much calibration time on the antenna if the previous alignment is still adequate, but it has a more flexible GP-BO fallback when the response surface is not simple.

---

## 18. Minimal Step-3 algorithm in pseudocode

```text
Input:
    achieved booster state u_B_tilde
    current or previous antenna position u_A_current
    hard antenna domain U_A_hard
    alignment objective or proxy A_l
    local Step-3 budget B_A
    previous antenna-alignment data D_A

Output:
    selected antenna position u_A_selected
    achieved antenna position u_A_tilde_selected
    alignment score and uncertainty
    Step-3 alignment data record
    quality flag

Procedure:

    1. Choose initial antenna position:
           previous best, model-predicted center, or safe-domain center.

    2. Measure alignment score at the initial/incumbent position.

    3. If the incumbent remains good within uncertainty:
           return incumbent position with validation flag.

    4. Otherwise perform a small local scan around the initial point.

    5. Fit a local response model, preferably quadratic near the maximum.

    6. If the local fit is reliable:
           propose the fitted optimum,
           enforce hard antenna constraints,
           move to the candidate,
           confirm by measurement,
           return if confirmed.

    7. If the local fit is unreliable:
           initialize a 2D GP model from the collected data,
           use a constrained, noise-aware acquisition function
           to choose additional antenna positions.

    8. Continue until:
           improvement is smaller than uncertainty,
           the local alignment budget is exhausted,
           a confirmed optimum is found,
           or a quality/safety condition fails.

    9. Return the best achieved antenna position and all local data.
```

---

## 19. Quality flags

Step 3 should return a quality flag so that later steps know how much confidence to place in the antenna alignment.

Possible flags include:

| Flag | Meaning |
|---|---|
| `reused_incumbent` | Previous antenna alignment was checked and remained valid. |
| `local_fit_confirmed` | Local scan and fit produced a confirmed optimum. |
| `gp_bo_confirmed` | 2D GP-BO produced a confirmed optimum. |
| `budget_limited` | Alignment stopped because local budget was exhausted. |
| `noise_limited` | Expected improvement was not distinguishable from measurement noise. |
| `drift_limited` | Reference measurements changed during the alignment. |
| `proxy_unreliable` | Proxy score behaved inconsistently or failed validation. |
| `constraint_limited` | Best predicted point was outside or near the safe boundary. |
| `multimodal_or_parasitic` | Response surface showed suspicious multi-modal or parasitic features. |

These flags should be forwarded to Step 5 and Step 6. They are part of the calibration information.

---

## 20. Validation before hardware deployment

Before using Step 3 in a real calibration run, it should be tested offline.

Useful validation tests include:

### 20.1 Synthetic Gaussian beam test

Create a simulated smooth 2D coupling surface and verify that the local scan plus quadratic fit recovers the known maximum.

### 20.2 Distorted beam test

Add asymmetry, tilt, higher-order-mode contamination, or a shifted beam center. Verify that the algorithm switches from local fit to GP-BO when the local quadratic model is insufficient.

### 20.3 Noise test

Add realistic measurement noise and check that the stopping rule does not chase improvements smaller than uncertainty.

### 20.4 Drift test

Let the optimum move slowly during the inner loop and check that reference remeasurements detect the drift.

### 20.5 Hysteresis/readback test

Simulate a difference between commanded and achieved antenna positions. Verify that the optimizer learns the map from achieved positions.

### 20.6 Proxy-to-high-fidelity validation

Compare the chosen antenna position under the proxy score with the position that would be chosen under the high-fidelity objective. This test determines whether the proxy is adequate for Step 3.

### 20.7 Budget test

Measure how many inner-loop evaluations are required for reliable alignment under different noise and drift conditions.

---

## 21. Main failure modes and mitigations

| Failure mode | Consequence | Mitigation |
|---|---|---|
| Proxy score does not correlate with high-fidelity objective | Antenna appears aligned but final boost factor does not improve | Validate proxy periodically with high-fidelity or higher-fidelity measurements. |
| Response surface is multi-modal | Local scan may choose wrong maximum | Switch to 2D GP-BO or larger diagnostic scan. |
| Alignment improvement is smaller than measurement noise | Optimizer chases noise | Replicate measurements or stop as noise-limited. |
| Commanded and achieved positions differ | Learned response map is smeared | Use achieved position readback in all models. |
| Drift during inner loop | Earlier antenna measurements become stale | Remeasure incumbent/reference point and include timestamps. |
| Optimum lies outside safe domain | Best achievable coupling is constrained | Return boundary-safe optimum and flag as constraint-limited. |
| Booster state changes strongly between iterations | Previous antenna warm start becomes poor | Use model-predicted beam center or wider initial scan. |
| Parasitic RF features dominate proxy | Alignment optimizes wrong feature | Use quality filters, time-gating logic if appropriate, or change proxy. |

---

## 22. Recommended development stages

### Stage A: deterministic local scan baseline

Implement Step 3 first as:

\[
\text{incumbent check} \rightarrow \text{small 2D scan} \rightarrow \text{local quadratic fit} \rightarrow \text{confirmation}.
\]

This gives a simple, interpretable baseline.

### Stage B: GP-BO fallback

Add a 2D GP-BO mode for cases where the local fit is unreliable.

This should use the same hard antenna constraints and noise estimates as the local scan.

### Stage C: proxy-to-high-fidelity calibration

Use data from Step 4 and Step 5 to learn how well the Step-3 proxy predicts the high-fidelity objective.

If the proxy is reliable, Step 3 can remain cheap.

If the proxy is unreliable, Step 3 should either use a different proxy or request occasional high-fidelity validation.

### Stage D: global reuse of antenna history

Once enough data exist, use previous antenna alignments across different booster states to predict good starting points for new booster states.

This can reduce the number of inner-loop measurements.

---

## 23. What Step 3 should not do

Step 3 should not:

- change disk positions,
- change the reflecting mirror,
- change the focusing mirror unless that variable is explicitly moved from the booster-state block into the antenna block in a later design revision,
- override the hard feasibility domain,
- use high-fidelity boost-factor measurements at every antenna candidate unless the budget explicitly allows it,
- treat a proxy score as final proof of improved boost factor,
- log only commanded positions when achieved-position readback is available,
- or silently discard failed or noisy alignment measurements.

Step 3 should remain a focused inner loop:

\[
\boxed{
\text{fixed booster geometry} \rightarrow \text{best antenna x/y position}.
}
\]

---

## 24. Final Step-3 design summary

The technical design for Step 3 is:

\[
\boxed{
\text{Antenna alignment is a low-dimensional, fixed-booster inner optimization problem.}
}
\]

The recommended implementation concept is:

\[
\boxed{
\text{local validation and scan first, 2D GP-BO fallback if needed.}
}
\]

The selected antenna position should be based on:

- achieved booster geometry,
- achieved antenna position,
- a validated alignment proxy or high-fidelity score,
- hard antenna safety constraints,
- local measurement uncertainty,
- drift checks,
- and the available inner-loop budget.

The output of Step 3 is not only the antenna position. It is the antenna position **plus** the data and uncertainty needed for Step 4, Step 5, and Step 6.

---

# Source anchors

The design above is based on the parent proposal and the following external source anchors.

1. **Parent proposal:** Step 3 is defined as optimizing \(x_{\mathrm{ant}}\) and \(y_{\mathrm{ant}}\) for the current booster state using either 2D Bayesian optimization or a 2D scan plus local fit.  
   `madmax_closed_loop_calibration_proposal.md`

2. **MADMAX overview:** MADMAX uses a stack of dielectric disks in front of a mirror to enhance the axion-induced signal.  
   <https://arxiv.org/html/2409.20169v1>

3. **Reciprocity / antenna and focusing mirror context:** The focusing mirror couples the field to an antenna connected to the receiver chain; the antenna emits a beam focused onto the booster; transverse behavior is important for the complete signal-power description.  
   <https://arxiv.org/html/2311.13359v2>

4. **MADMAX 3D requirements:** 3D calculations derive the emitted beam shape, which is important for antenna design, and discuss effects on antenna coupling.  
   <https://arxiv.org/abs/2104.06553>

5. **Bayesian optimization tutorial:** Bayesian optimization is useful for expensive, noisy, continuous optimization problems and is typically modeled with Gaussian-process surrogates and acquisition functions.  
   <https://arxiv.org/abs/1807.02811>

6. **Autonomous beamline alignment:** Bayesian optimization with Gaussian-process models has been used for online alignment problems in experimental beamline settings.  
   <https://arxiv.org/abs/2402.16716>

7. **BoTorch constraints:** BoTorch distinguishes input/parameter constraints from modeled outcome constraints, supporting the hard-versus-learned constraint split used here.  
   <https://botorch.org/docs/constraints>

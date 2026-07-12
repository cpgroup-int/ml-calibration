# Step 5 — Joint inference update

**Module:** {mod}`madmax_calibration.steps.step5_inference` ·
**Design:** {doc}`../design/madmax_step5_joint_inference_technical_design`

Step 5 updates the statistical belief about the real detector:

$$
p(\theta,\; r,\; \sigma_J,\; \text{drift} \mid D_{1:t+1}).
$$

The full mathematical treatment is in {doc}`../user_guide/statistics`;
this page covers the procedural side.

## Inference and observation levels

The implementation is **Level A** of the design (§13.1): a *joint* MAP fit
in which every discrepancy GP is marginalized analytically inside the
objective — so $\theta$ is never estimated without the discrepancy channel
present — plus a Laplace approximation for uncertainty. Levels B/C
(partial/full posterior sampling) are future upgrades behind the same
{class}`~madmax_calibration.steps.step5_inference.Step5Result` interface.

The **observation level** (design §4) is selected by
`Step5Config.observation_level`. The default is the **curve-summary
level** (§4.2, roadmap Phase 1.1): each HF measurement contributes the
vector (J, log peak, band centroid, bandwidth, flatness) produced by
{class}`~madmax_calibration.summaries.CurveSummarizer`, with one
discrepancy GP per component — so frequency shifts, amplitude losses and
bandwidth changes constrain $\theta$ separately instead of collapsing
into one scalar. `"scalar"` selects the single-component special case
(kept for A/B benchmarking); records lacking summaries trigger a
diagnosed automatic fallback to it.

With `Step5Config.lf_channel = "physics"` (default; roadmap Phase 1.2)
a second observation channel joins the same joint fit: reflectivity/
group-delay summaries of the cheap LF measurements, predicted by the
simulator's reflection solve — the y_ℓ = S_ℓ(u, θ) + r_ℓ + ε structure
of design §12, taken literally. This is what identifies the loss
parameter and removes the geometry/loss quasi-degeneracy of boost-only
data; the affine LF→J link survives only as a fallback for proxies
without a simulator counterpart. See {doc}`../user_guide/statistics`
for the mathematics and the multi-start/amplitude-cap stability
safeguards.

## Update routine

Following the design's §14:

1. **Ingest and validate** — only records passing
   `usable_for_inference()` (valid, not measurement-failed) enter the
   fit; at least `min_hf_points_for_inference` HF points are required.
   Excluded records stay in the dataset with reasons.
2. **Joint MAP** over standardized $(\theta, d, \log s)$
   with Gaussian/half-normal priors, alternating with a penalized ML-II
   refit of each component's discrepancy-GP hyperparameters (two rounds;
   warm-started from the previous iteration's state, so later updates
   converge in a few steps). Each amplitude prior is floored at
   `discrepancy_sigma_floor` measurement sigmas so unmodelled
   systematics land in the discrepancy channel rather than in $\theta$.
3. **Laplace covariance** for $(\theta, d)$ by finite-difference Hessian,
   eigenvalue-floored at the prior curvature.
4. **Posterior checks** — standardized residuals before/after the GP,
   discrepancy-dominance warning, baseline-repeat drift estimate.
5. **Classification** — every $\theta$ component is labelled
   `correctable` or `diagnostic` against the online control basis
   (`DetectorState.CORRECTABLE`), implementing the
   $\theta_{\mathrm{corr}} / \theta_{\mathrm{diag}}$ split (§8).
6. **Identifiability** — the prior-sensitivity refit (§15.2) flags
   parameters as `weak` when widening the discrepancy prior moves them by
   more than their posterior sd.
7. **LF link model** — an affine LF↔HF relation with residual
   uncertainty, `validated` only when it demonstrably tracks the
   objective (§12).

## Output

{class}`~madmax_calibration.steps.step5_inference.Step5Result` — the
complete Step-6 handoff package (§21): MAP detector state + Laplace
covariance (+ sampling helper), drift rate ± sd, noise inflation, the
conditioned discrepancy GP with its training inputs, classification and
identifiability tables, the LF link, and a diagnostics dict
(`n_hf`, residual statistics, discrepancy amplitude vs prior,
`discrepancy_dominant`, baseline drift estimate, convergence flag).

## Failure-mode handling

| Design failure mode (§17) | Behaviour |
|---|---|
| 17.1 $\theta$–discrepancy confounding | `identifiability[param] = "weak"`; wide Laplace floor |
| 17.2 discrepancy absorbs everything | `discrepancy_dominant` diagnostic |
| 17.3 noise larger than improvement | noise inflation grows; upstream EI gates react |
| 17.5 drift invalidates older data | drift term + baseline-repeat estimate; Step-1 rebaseline gate |
| 17.6 unreliable measurement regions | quality-flagged records excluded; soft-constraint model learns the region |

## Configuration

{class}`~madmax_calibration.config.Step5Config` — priors (load-bearing,
see {doc}`../user_guide/statistics`), lengthscale bounds, minimum data
requirement, and the `prior_sensitivity_check` toggle.

## Tests

`tests/test_step5_inference.py` implements the design §20 checklist:

| Design check | Test |
|---|---|
| 20.1 synthetic recovery | `test_synthetic_recovery_of_correctable_parameters` |
| 20.2 confounding | `test_prior_sensitivity_flags_weak_identifiability` |
| 20.4 drift | `test_drift_detected` |
| 20.5 correctable vs diagnostic | `test_classification_labels_follow_control_basis` |
| 20.6 multi-fidelity consistency | `test_lf_link_learned_but_not_pooled` |
| minimum-data guard | `test_requires_minimum_hf_points` |

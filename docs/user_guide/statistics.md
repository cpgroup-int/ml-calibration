# Statistical model reference

This page collects the mathematics of the joint calibration model (Step 5)
and the posterior predictive construction (Step 6) as implemented. The
governing principle, from the parent proposal (§8): *detector-state
parameters and simulator discrepancy are confounded, so the discrepancy
model must be present while $\theta$ is inferred* — never fitted
afterwards on a leftover residual.

## The observation model

Every valid high-fidelity measurement $i$ contributes

$$
J_i \;=\; J_{\mathrm{sim}}(\tilde u_i, \theta)
\;+\; r(\tilde u_i)
\;+\; d\,(t_i - t_{\mathrm{ref}})
\;+\; \epsilon_i,
\qquad
\epsilon_i \sim \mathcal N\!\bigl(0,\; \sigma_i^2 + s_{\mathrm{extra}}^2\bigr)
$$

with

- $\tilde u_i$ — the **achieved** geometry (readback), never the commanded
  one (parent proposal §2.5);
- $J_{\mathrm{sim}}(u, \theta)$ — the fast simulator pushed through the
  scalar objective ({meth}`~madmax_calibration.simulator.BoostSimulator.predict_J`);
- $\theta = (\theta_z, \theta_c, \theta_{\log loss})$ — physically
  interpretable detector-state parameters
  ({class}`~madmax_calibration.simulator.DetectorState`);
- $r(\cdot) \sim \mathcal{GP}(0, k_{\mathrm{RBF}})$ — the systematic
  simulator–measurement discrepancy, defined over *normalized* control
  space;
- $d$ — a linear drift rate in $J$ per hour (minimal drift model, Step 5
  design §11.2);
- $\sigma_i$ — the Step-4 propagated measurement uncertainty;
- $s_{\mathrm{extra}}$ — a jointly inferred noise-inflation term guarding
  against optimistic $\sigma_i$.

This is the scalar-objective observation level (Step 5 design §4.1) — the
recommended first implementation. Curve-summary and curve-level inference
(§4.2–4.3) are future extensions; the measurement records already preserve
the full curves they would need.

## Joint MAP inference (Level A)

{func}`~madmax_calibration.steps.step5_inference.run_step5` maximizes the
log posterior over $(\theta, d, s_{\mathrm{extra}})$ **with the
discrepancy GP marginalized analytically**: for fixed parameters, the
residuals $e_i = J_i - J_{\mathrm{sim}}(\tilde u_i,\theta) - d\,\Delta t_i$
have the GP marginal likelihood

$$
\log p(e \mid \theta, d, s) =
-\tfrac12 e^\top K_e^{-1} e - \tfrac12 \log |K_e| - \tfrac{n}{2}\log 2\pi,
\qquad
K_e = K_{\mathrm{RBF}} + \mathrm{diag}(\sigma_i^2 + s^2),
$$

so the discrepancy is *inside* the objective during $\theta$-inference,
not a post-hoc fit. Priors:

- $\theta \sim \mathcal N(0, \mathrm{diag}(\text{prior sd}^2))$ — the
  informative priors of Step 5 design §15.1 (mechanical tolerances etc.);
- $d \sim \mathcal N(0, \text{drift prior}^2)$;
- $s_{\mathrm{extra}}$, GP amplitude — half-normal (the amplitude prior is
  what stops the discrepancy from absorbing the physics, §9.3).

The optimization alternates two rounds of (a) L-BFGS over
$(\theta, d, \log s)$ given GP hyperparameters and (b) penalized ML-II
refit of the GP amplitude/lengthscale on the residuals — with warm starts
from the previous iteration's posterior state.

### Laplace uncertainty

A finite-difference Hessian of the negative log posterior around the MAP
gives the covariance for $(\theta, d)$; eigenvalues are floored at the
prior curvature so the reported uncertainty never exceeds the prior. Step
6 consumes this as Gaussian posterior samples
({meth}`~madmax_calibration.steps.step5_inference.Step5Result.theta_samples`).

### Identifiability safeguards

Because $\theta$ and $r$ can explain the same data:

- **Prior-sensitivity check** (§15.2): refit with the discrepancy
  amplitude prior widened ×3; any $\theta$ component that moves by more
  than its posterior sd is flagged `weak` in
  `Step5Result.identifiability`.
- **Discrepancy-dominance warning**: raised when the fitted GP amplitude
  exceeds twice its prior scale *and* the residual scatter is much larger
  than the noise.
- **Classification**: every parameter is labelled `correctable` or
  `diagnostic` against the online control basis
  (`DetectorState.CORRECTABLE`), so the optimizer can never chase an
  uncorrectable error.

A practical illustration of *why* these safeguards exist: in the mock
detector, a stack offset, a gap compression and a mirror correction move
the same physical gaps, so the likelihood has a near-degenerate valley.
The MAP may land on an equivalent $(\theta_z, \theta_c)$ combination —
the proposed *correction* is unaffected, but the *physical
interpretation* of individual components must not be over-trusted. This is
exactly the Kennedy–O'Hagan confounding the design notes cite.

## The LF proxy link

Lower-fidelity records never enter the joint fit as objective values.
Instead an affine link (Step 5 design §12)

$$
y_{\mathrm{LF}} = \alpha\, J_{\mathrm{HF}} + \beta + \varepsilon,
\qquad \varepsilon \sim \mathcal N(0, s_{\mathrm{LF}}^2)
$$

is regressed on the model's own predictions at the LF locations. The link
is marked `validated` only when the slope is positive and the residual
scatter is small relative to the explained spread — until then Step 1
treats LF probes as link-calibration data, not as trustworthy objective
information.

## Step 6: the posterior predictive

{func}`~madmax_calibration.steps.step6_predictive.run_step6` converts the
Step-5 state into the optimizer-facing distribution (sample-based
construction, Step 6 design §8). For a candidate $u$ at future time $t$:

$$
\mu_{\mathrm{lat}}(u) =
\underbrace{\frac1S \sum_s J_{\mathrm{sim}}(u, \theta^{(s)})}_{\text{simulator, }\theta\text{-marginalized}}
+ \underbrace{\mu_r(u)}_{\text{discrepancy GP}}
+ \underbrace{\hat d\,(t - t_{\mathrm{ref}})}_{\text{drift}}
$$

$$
\sigma^2_{\mathrm{lat}}(u) =
\underbrace{\mathrm{Var}_s\,J_{\mathrm{sim}}(u,\theta^{(s)})}_{\text{detector-state}}
+ \underbrace{\sigma_r^2(u)}_{\text{discrepancy}}
+ \underbrace{\sigma_d^2\,(t - t_{\mathrm{ref}})^2}_{\text{drift}}
$$

and the **future observation** adds the measurement noise:

$$
\sigma^2_{\mathrm{obs}}(u) = \sigma^2_{\mathrm{lat}}(u)
+ \sigma_{J,\mathrm{HF}}^2 + s_{\mathrm{extra}}^2 .
$$

The latent/observation distinction (design §7) is what lets Step 1 ask two
different questions: *"is this configuration actually better?"* (latent)
and *"what will a noisy measurement of it look like?"* (observation —
used e.g. for replication decisions). Posterior *samples* of the latent
objective are also exposed for Thompson-style acquisition use.

### Double-counting discipline

Each uncertainty source lives in exactly one term (design §14):
simulator/detector-state → $\theta$; systematic mismatch → $r$;
repeat scatter → $\epsilon$; slow time dependence → drift. The
noise-inflation term only covers *unexplained excess* repeat scatter, and
the amplitude prior keeps $r$ from eating $\theta$'s share.

### Diagnostics attached to every prediction

- **Extrapolation regime** (design §17): distance in normalized control
  space to the nearest HF training point →
  `interpolation` (< 0.15), `mild extrapolation` (< 0.35),
  `strong extrapolation`.
- **Staleness**: hours since the last baseline/incumbent HF measurement
  ({meth}`~madmax_calibration.steps.step6_predictive.PredictiveModel.staleness_hours`).
- **Validation** (design §21): standardized residuals
  $z_i = (J_i - \mu(u_i))/\sigma_{\mathrm{obs}}(u_i)$ over the training
  data; `overconfident` (RMS z > 2) and `underconfident` flags are
  reported in `PredictiveModel.validation` before the model is handed to
  Step 1.

## What is deliberately *not* claimed

Following Step 5 design §24: the joint MAP is not a full posterior;
$\theta$ is not uniquely identified without the priors; the discrepancy is
systematic structure, not noise; LF proxies are not HF measurements; and
uncorrectable inferred errors are never presented to the optimizer as
correctable. Upgrades (partial/full posterior sampling — Levels B/C,
curve-level inference, richer drift models) slot in behind the same
`Step5Result` interface.

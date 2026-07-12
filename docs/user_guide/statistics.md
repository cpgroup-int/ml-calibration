# Statistical model reference

This page collects the mathematics of the joint calibration model (Step 5)
and the posterior predictive construction (Step 6) as implemented. The
governing principle, from the parent proposal (§8): *detector-state
parameters and simulator discrepancy are confounded, so the discrepancy
model must be present while $\theta$ is inferred* — never fitted
afterwards on a leftover residual.

## The observation model

Every valid high-fidelity measurement $i$ contributes a vector of curve
summaries $z_i$ (roadmap Phase 1.1; Step 5 design §4.2). Component by
component $k$:

$$
z_{i,k} \;=\; z_{\mathrm{sim},k}(\tilde u_i, \theta)
\;+\; r_k(\tilde u_i)
\;+\; [k{=}J]\; d\,(t_i - t_{\mathrm{ref}})
\;+\; \epsilon_{i,k},
\qquad
\epsilon_{i,k} \sim \mathcal N\!\bigl(0,\; \sigma_{i,k}^2 + (s\,\bar\sigma_k)^2\bigr)
$$

with

- $z = (J,\ \log\text{peak},\ \text{band centroid},\ \text{bandwidth},\
  \text{flatness})$ — smooth, physically meaningful summaries of
  $\beta^2(\nu)$ ({class}`~madmax_calibration.summaries.CurveSummarizer`;
  component 0 is the scalar objective). A frequency shift, an amplitude
  loss and a bandwidth change — indistinguishable at scalar level — now
  constrain $\theta$ separately, which is what breaks the detector-state
  degeneracies;
- $\tilde u_i$ — the **achieved** geometry (readback), never the commanded
  one (parent proposal §2.5);
- $z_{\mathrm{sim}}(u, \theta)$ — the fast simulator pushed through the
  summarizer ({meth}`~madmax_calibration.simulator.BoostSimulator.predict_summaries`);
- $\theta = (\theta_z, \theta_c, \theta_{\log loss})$ — physically
  interpretable detector-state parameters
  ({class}`~madmax_calibration.simulator.DetectorState`);
- $r_k(\cdot) \sim \mathcal{GP}(0, k_{\mathrm{RBF}})$ — one systematic
  simulator–measurement discrepancy GP per component, over *normalized*
  control space;
- $d$ — a linear drift rate acting on the objective component (minimal
  drift model, Step 5 design §11.2);
- $\sigma_{i,k}$ — the Step-4 Monte-Carlo-propagated summary
  uncertainties (cross-component correlations neglected at Level A —
  a documented approximation);
- $s$ — one shared, dimensionless noise-inflation factor, scaled per
  component by the typical measurement sigma $\bar\sigma_k$.

`Step5Config.observation_level = "scalar"` selects the pre-Phase-1.1
single-component special case ($z = (J)$) — retained so the benchmark
harness can A/B the two levels. Records lacking summaries trigger an
automatic, diagnosed fallback to scalar level. Full curve-level inference
(§4.3) remains a future extension; the records preserve the curves it
would need.

## The low-fidelity physics channel

With `Step5Config.lf_channel = "physics"` (default; roadmap Phase 1.2),
every cheap reflectivity measurement joins the same joint fit as a second
observation channel — the structure of Step 5 design §12, taken
literally:

$$
y_{i,\ell} \;=\; S_\ell(\tilde u_i, \theta) \;+\; r_\ell(\tilde u_i)
\;+\; \epsilon_{i,\ell},
$$

where $y_\ell$ are the reflectivity summaries (mean power reflectivity,
its slope across the window, group-delay centroid, mean group delay —
{class}`~madmax_calibration.summaries.ReflectivitySummarizer`) and
$S_\ell$ is the simulator's reflection solve
({meth}`~madmax_calibration.simulator.BoostSimulator.reflectivity_observables`).
Each LF component has its own discrepancy GP, so instrument systematics
(amplitude mis-calibration, cable-delay offset) live in the LF
discrepancy channel rather than biasing $\theta$.

Why this matters: with physically correct absorption, the dielectric
loss mimics geometry in *every boost-curve summary* (both reduce peak
and objective), leaving a geometry/loss quasi-degeneracy that HF data
cannot resolve — while $|\Gamma|^2$ dips measure absorption directly and
the group delay pins the resonance geometry. Measured on the benchmark:
adding six ~0.1 h reflectivity probes to ten HF points takes the
correctable-parameter recovery from ~100 µm (weakly identified loss) to
~5–20 µm with the loss known to ±0.015.

The affine LF→J link ({class}`~madmax_calibration.steps.step5_inference.LFLinkModel`)
remains only as the fallback for scalar proxies without a simulator
counterpart (`lf_channel = "affine"`); `"off"` ignores LF data entirely
(used by the benchmark A/B).

### Stability safeguards of the joint MAP

Two safeguards added after Phase-1.2 validation exposed a spurious mode
(the optimizer running the loss to its prior bound with few LF points):
every MAP round is **multi-started** (current point + prior mean, best
kept), and the ML-II amplitude refit is **capped at 4 prior sds** — with
very few points the marginal likelihood alone cannot rule out runaway
discrepancy explanations.

## Joint MAP inference (Level A)

{func}`~madmax_calibration.steps.step5_inference.run_step5` maximizes the
log posterior over $(\theta, d, s)$ **with every discrepancy GP
marginalized analytically**: for fixed parameters, each component's
residuals $e_{i,k} = z_{i,k} - z_{\mathrm{sim},k}(\tilde u_i,\theta) -
[k{=}J]\,d\,\Delta t_i$ contribute the GP marginal likelihood

$$
\log p(e_k \mid \theta, d, s) =
-\tfrac12 e_k^\top K_k^{-1} e_k - \tfrac12 \log |K_k| - \tfrac{n}{2}\log 2\pi,
\qquad
K_k = K_{\mathrm{RBF},k} + \mathrm{diag}\bigl(\sigma_{i,k}^2 + (s\bar\sigma_k)^2\bigr),
$$

summed over components — so the discrepancy channels are *inside* the
objective during $\theta$-inference, never a post-hoc fit. Priors:

- $\theta \sim \mathcal N(0, \mathrm{diag}(\text{prior sd}^2))$ — the
  informative priors of Step 5 design §15.1 (mechanical tolerances etc.);
- $d \sim \mathcal N(0, \text{drift prior}^2)$;
- $s$, GP amplitudes — half-normal. Each component's amplitude prior
  scales with its response magnitude **and is floored at a few
  measurement sigmas** (`discrepancy_sigma_floor`): unmodelled
  systematics — e.g. a receiver-chain tilt that shifts the band
  centroid — must have an affordable home in the discrepancy channel,
  otherwise the fit buys the same explanation with a biased $\theta$
  (the confounding mechanism of §9.3, observed directly in the
  benchmark before the floor was introduced).

The optimization alternates two rounds of (a) L-BFGS over
$(\theta, d, \log s)$ given all GP hyperparameters and (b) penalized
ML-II refit of each component's amplitude/lengthscale on its residuals —
with warm starts from the previous iteration's posterior state.

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

## Amortized inference engine (NPE)

`Step5Config.inference_engine = "amortized_npe"` (roadmap Phase 2)
replaces the joint-MAP estimate of $\theta$ with an **amortized neural
posterior estimator** — a conditional density $q(\theta \mid c)$ trained
offline against the simulator and evaluated online in a millisecond
forward pass. The Laplace approximation's weakness is exactly the
degeneracy valley above: a Gaussian centred at one MAP point is
**overconfident** and cannot represent the skewed/curved ridge. NPE
learns a flexible (Gaussian-mixture) posterior instead.

The engine ({mod}`madmax_calibration.amortized`):

- **Network.** A small mixture-density network (two tanh layers →
  diagonal Gaussian mixture over standardized $\theta$), implemented in
  pure numpy so the online runtime stays numpy/scipy-only. Trained
  weights are exported to a plain `.npz` blob — no PyTorch at inference.
- **Conditioning.** $\theta$ is global while measurements arrive at
  arbitrary $u_i$, so the network conditions on a fixed-length,
  permutation-invariant summary: each measurement's residual against the
  *nominal* simulation projected onto its design $[1, x_i]$, averaged
  over measurements (a Gauss-Newton sufficient direction). This handles
  any number of HF and LF measurements and appends cleanly to window
  features for the planned cross-window transfer (Phase 5).
- **Misspecification robustness.** Training injects a shared systematic
  bias into each episode's summaries (a stand-in for the tilt / focus /
  calibration systematics), so the amortized posterior is trained to be
  *appropriately wide* under the model error it will meet online — the
  robustness the roadmap requires.
- **Hybrid, not a replacement of the physics.** The forward model and
  the **online discrepancy GPs are kept**: NPE produces $p(\theta \mid
  D)$; the drift and noise-inflation terms are then fit with $\theta$
  fixed, and the per-channel discrepancy GPs are conditioned at the NPE
  estimate exactly as in the joint-MAP path.
  {meth}`~madmax_calibration.steps.step5_inference.Step5Result.theta_samples`
  becomes **exact mixture sampling** instead of a Laplace Gaussian draw,
  and Step 6 is unchanged.

**Validation (SBC).** Correctness is checked by simulation-based
calibration ({mod}`madmax_calibration.sbc`): for data simulated from
$\theta^\star \sim$ prior, a calibrated posterior makes the rank of
$\theta^\star$ among posterior samples uniform. On the shipped weights
the well-specified case passes (rank KS $p > 0.3$, 2σ coverage
0.93–0.95), and coverage degrades gracefully to ~0.9 under injected
discrepancy. This SBC replaces the Laplace-specific residual checks as
the acceptance instrument for the amortized engine.

**Measured trade-off.** On the full-loop benchmark (3 seeds) NPE gives
equal-or-better validated improvement at equal HF count and 100%
safety/budget, ~3× faster inference, and honestly calibrated (often
wider) posteriors — versus the joint-MAP's tighter but overconfident
estimates on the ridge.

**Why it is opt-in, not the default.** The trained weights are specific
to the control basis *and the frequency window* (the summaries depend on
the window's grid). The shipped `weights/npe_prototype.npz` is trained
for window 1 of the prototype settings; a different window or control
basis needs a retrain (`python examples/train_npe.py`). A dimension
guard falls the engine back to joint-MAP with a diagnostic if the
weights do not match the current basis. Window-conditioned amortization
that serves all windows from one network is roadmap Phase 5.

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

# Step 6 — The optimizer-facing predictive model

**Module:** {mod}`madmax_calibration.steps.step6_predictive` ·
**Design:** {doc}`../design/madmax_step6_predictive_model_technical_design`

Step 6 converts the Step-5 joint calibration state into the prediction
object Step 1 optimizes over:

$$
p\bigl(J_{\mathrm{HF}}(u) \mid D_{1:t+1}\bigr).
$$

It chooses no candidates and hard-codes no acquisition — it provides
calibrated distributions, samples and diagnostics
({class}`~madmax_calibration.steps.step6_predictive.PredictiveModel`).

## Construction

The **sample-based** construction of the design (§8): Laplace posterior
samples of $\theta$ are pushed through the fast simulator per candidate,
the discrepancy GP adds its mean and variance, and the drift term
extrapolates to the intended future measurement time. See
{doc}`../user_guide/statistics` for the formulas.

The prediction target is the **antenna-aligned booster-level objective**
$F(u_B)$ with a plug-in antenna optimum (§5, option 1) — the documented
approximation being that antenna-alignment uncertainty is not yet
propagated.

## The query interface

Matches the conceptual API of the design (§19):

| Design query | Implementation |
|---|---|
| `predict(candidates, …)` | {meth}`~madmax_calibration.steps.step6_predictive.PredictiveModel.predict` → {class}`~madmax_calibration.steps.step6_predictive.Prediction` with latent mean/sd, observation sd, latent samples, extrapolation regimes, nearest-data distances |
| `predict_fidelities(…)` | {meth}`~madmax_calibration.steps.step6_predictive.PredictiveModel.predict_lf` via the learned LF link |
| `diagnostics(…)` | extrapolation regime + `staleness_hours` + the `summary()` dict |
| cheap plug-in mean | {meth}`~madmax_calibration.steps.step6_predictive.PredictiveModel.predict_mean_map` (used by Step-1 candidate refinement) |

Latent vs future-observation predictions are kept distinct throughout
(§7): `latent_sd` answers *is this configuration better*, `obs_sd`
answers *what will a measurement of it look like* — the difference is the
measurement-noise floor $\sigma_{J,\mathrm{HF}}^2 + s_{\mathrm{extra}}^2$,
where $\sigma_{J,\mathrm{HF}}$ itself is the max of the stated
uncertainties and the empirical replicate scatter.

## Validation before hand-off

Per design §21, the model is checked before Step 1 may use it:
standardized residuals of the observation prediction over all training
points, with `overconfident` (RMS z > 2) and `underconfident` flags
stored in `PredictiveModel.validation`. The design's operational success
condition — measured values at proposed candidates falling inside the
predictive intervals at roughly the advertised rate — is what the
end-to-end suite exercises implicitly by requiring the loop to converge.

## Preserved distinctions

- **Correctable vs diagnostic** — predictions *include* the effect of
  diagnostic-only parameters (e.g. inferred extra loss lowers predicted
  performance) but no control direction can act on them, so the optimizer
  cannot chase them (§15).
- **Uncertainty decomposition** (§13) — the variance is built from
  identified sources ($\theta$ / discrepancy / drift / noise), enabling
  Step 1's replicate-vs-explore-vs-rebaseline choices.
- **Extrapolation honesty** (§17) — every prediction carries
  `interpolation` / `mild extrapolation` / `strong extrapolation` from
  its distance to the training data; Step-1 diagnostics surface it in
  each proposal's `reason` string.

## Tests

`tests/test_step6_predictive.py` implements the design §27 checklist:

| Design check | Test |
|---|---|
| 27.1 no-discrepancy closure | `test_no_discrepancy_closure` |
| 27.2 known discrepancy | `test_known_discrepancy_is_learned` |
| 27.3 biased theta / uncertainty propagation | `test_theta_uncertainty_propagates` |
| 27.5 drift | `test_drift_prediction_extrapolates_in_time` |
| §7 latent vs observation | `test_observation_sd_exceeds_latent_sd` |
| §17 extrapolation | `test_extrapolation_flagged` |

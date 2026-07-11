# Walkthrough: a synthetic calibration run

This page narrates what happens when you run

```bash
python examples/run_synthetic_calibration.py
```

and how to read the output. Everything about the run — the disk
configurations, the frequency window, the budgets, the simulated
detector's hidden errors — comes from the settings file
(`settings/prototype.toml` by default; see the settings-file section of
{doc}`user_guide/configuration`). The numbers below are from seed 0 on
`window_01` (18.25 GHz, half-wave spacings of 8.214 mm); other seeds
and windows differ in detail but not in structure.

## What the mock detector hides

The simulated detector ({class}`~madmax_calibration.hardware.MockHardware`
with {class}`~madmax_calibration.hardware.MockTruth`, both configured in
the `[mock]` section of the settings file) is the same physics as the
fast simulator **plus** everything the real detector would add on top of
an idealized model:

| Hidden effect | Default value | Visible to the algorithm? |
|---|---|---|
| Stack z-offset error $\theta_z$ | $+0.8$ mm | only through measurements |
| Inter-disk gap compression $\theta_c$ | $+0.4$ mm | only through measurements |
| Extra dielectric loss ($\log$ scale) | $+0.3$ | only through measurements |
| Antenna beam centre offset | $(2.5, -1.5)$ mm | found by Step 3 |
| Focus optimum offset | $+0.6$ mm | appears as *discrepancy* |
| Systematic curve tilt | $\pm 4\%$ across $W$ | appears as *discrepancy* |
| Stack drift | $2\,\mu$m/h | tracked by the drift term |
| Actuator hysteresis / repeatability | $10 / 5\,\mu$m | mitigated by achieved readback |
| HF measurement noise | ~1% norm + 1% per bin | estimated in Step 0 |

The combination of $\theta_z = +0.8$ mm and $\theta_c = +0.4$ mm degrades
the 3-disk stack response by roughly 30–50% depending on the window, and
the focus error costs more on top — so the loop starts from a detector
performing far below its nominal potential.

## Phase 1 — Step 0: baseline

```text
[step0] J0 = 0.1297 +/- 0.0028
```

The loop moves to the nominal configuration ($u_B = 0$, i.e. the
spacings from the active disk configuration), aligns the antenna (Step 3
finds the beam centre near $(2.5, -1.5)$ mm), then measures the boost
factor three times (`n_baseline_replicates`). The replication gives an
*empirical* repeatability estimate which is cross-checked against the
propagated per-measurement uncertainty; the larger of the two becomes
$\sigma_{J,0}$. If $\sigma_{J,0}$ were comparable to the whole objective,
the run would already be flagged as not resolvable (the feasibility
condition of the parent proposal, section 7).

## Phase 2 — exploration under the trust region

```text
[iter 1] action=new_candidate reason=EI 0.083 > 0.3 x HF noise 0.0028; regime: strong extrapolation
[iter 2] action=new_candidate ...
...
[iter 4] action=rebaseline reason=stale model state (4.0 h since last baseline/incumbent HF; drift scale 0.0073 vs noise 0.0034)
```

Each iteration:

1. **Step 5** re-fits the joint model — detector state $\theta$,
   per-summary discrepancy GPs, noise inflation, drift rate — on all
   valid HF data. Each HF record contributes its *curve-summary vector*
   (J, log peak, band centroid, bandwidth, flatness), so frequency
   shifts, amplitude losses and bandwidth changes constrain $\theta$
   separately (roadmap Phase 1.1; the scalar-only level remains
   available via `observation_level = "scalar"`).
2. **Step 6** rebuilds the posterior predictive $p(J_{\mathrm{HF}}(u)\mid D)$
   by pushing Laplace posterior samples of $\theta$ through the simulator
   and adding the discrepancy GP.
3. **Step 1** scores a Sobol candidate pool inside
   (trust region ∩ hard-feasible domain), refines the best predicted-mean
   point into a *physics-informed exploit candidate*, applies the
   noise/cost/soft-feasibility gates, and proposes the winner.
4. **Steps 2–4** execute: move (large moves are split into safe sub-steps),
   align the antenna (the incumbent position is revalidated and reused when
   still good), measure, record.

In the first iterations the $\theta$-posterior is wide, so the predictions
lean on exploration; after 3–5 HF points the inferred stack offset and
compression stabilize and the exploit candidate starts pointing at the
correcting configuration. Validated improvements recenter and expand the
trust region. Note iteration 4: the inferred drift rate times the model
staleness exceeded twice the measurement noise, so the drift-aware gate
forced a re-baseline before further exploration — the model state is
refreshed rather than trusted blindly.

## Phase 3 — noise-aware wind-down

```text
[iter 9]  action=lf_probe reason=expected HF improvement below noise threshold; ...
[iter 10] action=lf_probe ...
```

Once the predicted expected improvement of the best candidate drops below
`ei_noise_factor` × (HF measurement noise), Step 1 refuses to spend a
high-fidelity measurement on it. Because the LF proxy link is not yet
validated (fewer than 3 LF points), it spends cheap LF probes at the most
uncertain feasible candidates instead — information gathering at ~1/10 of
the HF cost.

After `patience` (default 3) consecutive iterations without resolvable
improvement, Step 7 stops the loop:

```text
stop_reason: expected improvement below 0.5 x HF noise (0.0035) for 3 consecutive iterations
```

## Phase 4 — final validation and report

The loop re-measures the best configuration with the conservative
`HF_validation` action and averages it with the incumbent measurement.
The **feasibility report** (parent proposal, section 3) summarizes the
run:

```text
J_baseline: 0.1297              sigma_J_baseline: 0.0028
J_best_validated: 0.1904        sigma_J_best: 0.0028
improvement: 0.0607             improvement_over_noise: 15.3
improvement_significant: True
n_hf_measurements: 12           n_lf_measurements: 3
total_cost_hours: 16.2
```

and the detector-state inference:

```text
z_offset     truth=+8.0e-04  estimate=+1.16e-03 +/- 1.2e-04  [correctable, weak]
compression  truth=+4.0e-04  estimate=+2.67e-04 +/- 9.3e-05  [correctable, ok]
log_loss     truth=+3.0e-01  estimate=+1.35e+00 +/- 2.9e-02  [diagnostic, ok]
```

Three things are worth reading carefully here:

- **The correction, not the parameters, is the deliverable.** The
  estimated $(\theta_z, \theta_c)$ pair differs from the truth along a
  direction that is nearly degenerate in the geometry (a stack offset can
  be traded against gap compression and the mirror correction; the final
  correction here indeed uses a mirror move as part of the compensation).
  The MAP landed on an *equivalent* explanation — and the proposed
  correction `u_B*` compensates the real errors regardless.
- **The `weak` flag on `z_offset` is the system being honest.** The mock
  detector's unmodelled systematics (the receiver-chain tilt, the focus
  offset) can masquerade as a stack offset; the prior-sensitivity check
  detects that the estimate moves when the discrepancy priors are
  widened and says so, instead of presenting a biased value with false
  confidence. Removing this residual confounding is precisely what the
  roadmap's Phase 2 (misspecification-robust amortized inference) and
  Phase 4.1 (drift on $\theta$) target.
- **`log_loss` is diagnostic-only** — with the curve summaries it is
  now *identifiable* (the amplitude-carrying summaries constrain it),
  but here it absorbs the focus-coupling amplitude deficit on top of
  the true dielectric loss. No control direction is spanned by it, so
  the optimizer never "chases" it either way.

## The history file

`calibration_history.json` contains one entry per iteration with the full
Step-1 proposal metadata (action, reason, predicted mean/sd, EI,
acquisition value, expected cost, soft feasibility, trust-region size),
the measured outcome and the running best — the audit trail required by
the Step-1 design note (section 24).

## Trying other windows and campaigns

```bash
# calibrate a different window from the same settings file
python examples/run_synthetic_calibration.py --window window_07

# generate a fresh campaign file (any range, any number of windows)
python examples/generate_settings.py -o settings/my_campaign.toml \
    --f-min 18 --f-max 24 --windows 12

# A/B-benchmark two pipeline configurations (roadmap Phase 0.2)
python -m madmax_calibration.benchmark settings/prototype.toml \
    settings/my_campaign.toml --runs 3
```

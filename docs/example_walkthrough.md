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
| Reflectivity calibration bias | $+1\%$ | absorbed by the LF discrepancy GPs |
| Group-delay cable offset | $+20$ ps | absorbed by the LF discrepancy GPs |

The combination of $\theta_z = +0.8$ mm and $\theta_c = +0.4$ mm degrades
the 3-disk stack response by roughly 30–50% depending on the window, and
the focus error costs more on top — so the loop starts from a detector
performing far below its nominal potential.

## Phase 1 — Step 0: baseline

```text
[step0] J0 = 0.1123 +/- 0.0024
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

## Phase 2 — identification, then exploration

```text
[iter 1] action=lf_probe reason=identification: 0 < 3 physics-channel LF measurements; ...
[iter 2] action=lf_probe reason=identification: 1 < 3 ...
[iter 3] action=lf_probe reason=identification: 2 < 3 ...
[iter 4] action=new_candidate reason=EI 0.092 > 0.3 x HF noise 0.0024; regime: strong extrapolation
[iter 5] action=new_candidate ...
[iter 6] action=rebaseline reason=stale model state (3.8 h since last baseline/incumbent HF; drift scale 0.0061 vs noise 0.0030)
```

The first three iterations are ~0.1 h **reflectivity identification probes**
(roadmap Phase 1.2): the reflectivity and group-delay curves are physics
observables the simulator can predict, so they constrain the detector
state at a tenth of the HF cost — in particular they measure the
dielectric loss directly and break the geometry/loss degeneracy that
boost curves alone leave open. Only then does the loop start spending
1 h boost-factor measurements.

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

Thanks to the identification probes the $\theta$-posterior is already
tight when HF spending starts, so the exploit candidate points at the
correcting configuration almost immediately. Validated improvements
recenter and expand the trust region. Note iteration 6: the inferred
drift rate times the model staleness exceeded twice the measurement
noise, so the drift-aware gate forced a re-baseline before further
exploration — the model state is refreshed rather than trusted blindly.

## Phase 3 — noise-aware wind-down

```text
[iter 9]  action=lf_probe reason=expected HF improvement below noise threshold; ...
[iter 10] action=lf_probe ...
```

Once the predicted expected improvement of the best candidate drops below
`ei_noise_factor` × (HF measurement noise), Step 1 refuses to spend a
high-fidelity measurement on it. With the physics LF channel active,
cheap reflectivity probes at the most uncertain feasible candidates are
always informative (they constrain $\theta$ directly), so the wind-down
gathers information at ~1/10 of the HF cost.

After `patience` (default 3) consecutive iterations without resolvable
improvement, Step 7 stops the loop:

```text
stop_reason: expected improvement below 0.5 x HF noise (0.0032) for 3 consecutive iterations
```

## Phase 4 — final validation and report

The loop re-measures the best configuration with the conservative
`HF_validation` action and averages it with the incumbent measurement.
The **feasibility report** (parent proposal, section 3) summarizes the
run:

```text
J_baseline: 0.1123              sigma_J_baseline: 0.0024
J_best_validated: 0.1872        sigma_J_best: 0.0026
improvement: 0.0749             improvement_over_noise: 21.4
improvement_significant: True
n_hf_measurements: 8            n_lf_measurements: 4
total_cost_hours: 11.5
```

and the detector-state inference:

```text
z_offset     truth=+8.0e-04  estimate=+6.68e-04 +/- 5.9e-05  [correctable, ok]
compression  truth=+4.0e-04  estimate=+5.21e-04 +/- 4.5e-05  [correctable, ok]
log_loss     truth=+3.0e-01  estimate=+3.63e-01 +/- 2.7e-02  [diagnostic, ok]
```

Three things are worth reading carefully here:

- **The correction, not the parameters, is the deliverable.** The
  estimated $(\theta_z, \theta_c)$ pair can trade against each other
  and the mirror correction along a near-degenerate geometry direction;
  the proposed correction `u_B*` compensates the real errors regardless
  of where on that ridge the estimate sits.
- **The reflectivity channel is what makes the estimates accurate.**
  Boost curves alone leave a geometry/loss quasi-degeneracy (with
  physically correct absorption, loss mimics geometry in every boost
  summary); the reflectivity dips measure absorption directly, and its
  own discrepancy GPs absorb the instrument systematics (calibration
  bias, cable delay) instead of biasing $\theta$. Compare the
  full-loop benchmark: without the LF channel, θ errors are ~170 µm
  with `weak` identifiability flags; with it, ~50 µm with honest
  posteriors and all flags `ok`.
- **`log_loss` is diagnostic-only** — now well measured
  (0.36 ± 0.03 vs truth 0.30), but no control direction is spanned by
  it, so the optimizer never "chases" it.

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

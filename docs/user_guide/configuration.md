# Configuration reference

All tunables live in {class}`~madmax_calibration.config.CalibrationConfig`,
a tree of dataclasses — and every one of them can be set from a **TOML
settings file**, which is the recommended interface. Units are **metres,
Hz and hours** throughout. Fields marked ⚠ are stand-ins for decisions
the design notes defer to the MADMAX team (see
{doc}`../design/DESIGN_DECISIONS`).

## The settings file

The repository ships `settings/prototype.toml`; nothing about the
campaign is hard-coded — the frequency range and the number of windows
are whatever the file says. Load and run with:

```python
from madmax_calibration.settings import load_settings
from madmax_calibration.loop import build_loop_from_settings

settings = load_settings("settings/prototype.toml")
loop = build_loop_from_settings(settings, window="window_07", seed=0)
result = loop.run(max_iterations=20)
```

File layout ({mod}`madmax_calibration.settings`):

- `[calibration]` — objective, baseline replicates, seed, and the
  `active_configuration` (the window calibrated by default);
- `[simulator]`, `[control]`, `[antenna]`, `[budget]`, `[cost]`,
  `[trust_region]`, `[step1]`, `[step5]`, `[step7]` — mirror the
  dataclasses below field-for-field (SI units). Unknown keys are
  **rejected at load time**, so typos fail loudly;
- `[mock.truth]` / `[mock.noise]` — the simulated detector's hidden
  errors and instrument noise (examples, benchmarks and tests only;
  ignored on real hardware);
- any number of `[[disk_configuration]]` tables — one per frequency
  window.

Each disk configuration defines the nominal detector state `q0(W)` for
one window:

```toml
[[disk_configuration]]
name = "window_07"
target_frequency_ghz = 21.25
window_half_width_ghz = 0.25
spacings_mm = [7.0539, 7.0539, 7.0539]   # mirror-d1, d1-d2, d2-d3
booster_antenna_distance_mm = 100
# disk_thickness_mm = 1.4108             # optional; default: half-wave
```

`spacings_mm` are the three physical gaps of the 3-disk prototype stack
(mirror–disk1, disk1–disk2, disk2–disk3). The
`booster_antenna_distance_mm` (disk3 → antenna, focusing mirror in
between) is deliberately **not** a stack spacing: the 1D solver treats
the region behind the last disk with a radiation condition, so it does
not change the boost curve; it parameterizes the coupling optics and is
carried through to the hardware layer.

Generate a complete file for an arbitrary campaign (the spacings are the
analytic half-wave stand-in for the offline disk optimization — replace
them with the real per-window values):

```bash
python examples/generate_settings.py -o settings/my_campaign.toml \
    --f-min 18 --f-max 24 --windows 12 --disks 3
```

## Programmatic configuration

The dataclasses remain available directly when scripting:

```python
from madmax_calibration.config import CalibrationConfig
cfg = CalibrationConfig()
cfg.objective = "smooth_min"
cfg.budget.max_hf_measurements = 15
cfg.step1.lambda_info = 0.3
```

## Top level

| Field | Default | Meaning |
|---|---|---|
| `objective` ⚠ | `"scan_rate"` | Scalar physics objective $J$; one of `scan_rate`, `smooth_min`, `peak`. Fixes the operating point on the peak–bandwidth trade-off (parent proposal §6). |
| `n_baseline_replicates` | 3 | Step-0 baseline repeats used for the empirical noise estimate. |
| `seed` | 0 | Seed for the loop's random generator. |

## `simulator` — {class}`~madmax_calibration.config.SimulatorConfig`

| Field | Default | Meaning |
|---|---|---|
| `n_disks` ⚠ | 3 | Number of dielectric disks in the 1D stack (MADMAX prototype). Derived from the active disk configuration when a settings file is used. |
| `disk_index` | 5.0 | Refractive index (LaAlO₃-like). |
| `disk_loss_tan` | 2e-3 | Nominal dielectric loss tangent; scaled by $e^{\theta_{\log loss}}$. |
| `target_frequency` | 22 GHz | Centre of the target window $W$. |
| `window_half_width` | 0.25 GHz | $W$ = target ± half-width. |
| `n_freq` | 81 | Frequency-grid points across $W$. |

## `control` — {class}`~madmax_calibration.config.ControlConfig`

Defines $u_B = (a_{\mathrm{disk}}, z_{\mathrm{global}}, z_{\mathrm{mirror}},
z_{\mathrm{focus}})$ and the hard control box.

| Field | Default | Meaning |
|---|---|---|
| `n_disk_modes` ⚠ | 2 | Active disk-correction modes (uniform gap change; linear gap gradient; optional quadratic as mode 3). |
| `disk_mode_limit` ⚠ | 0.5 mm | Travel half-width per disk-mode amplitude. |
| `z_global_limit` ⚠ | 1.0 mm | Global stack translation limit. |
| `z_mirror_limit` ⚠ | 0.5 mm | Reflecting-mirror correction limit. |
| `z_focus_limit` ⚠ | 2.0 mm | Focusing-mirror correction limit. |
| `max_step_normalized` ⚠ | 0.35 | Maximum safe *single move* per coordinate, in normalized [0, 1] units. Larger commanded moves are split automatically. |
| `min_gap` ⚠ | 3.0 mm | Minimum allowed physical gap (collision avoidance — enforced exactly, never learned). |

## `antenna` — {class}`~madmax_calibration.config.AntennaConfig`

| Field | Default | Meaning |
|---|---|---|
| `travel_limit` ⚠ | 10 mm | Hard $|x|,|y|$ antenna travel limit. |
| `initial_scan_step` | 1.5 mm | Local plus-pattern scan step (match to expected beam width). |
| `max_evaluations` | 20 | Step-3 local measurement budget $B_A$. |
| `kappa` | 2.0 | Incumbent revalidation / confirmation tolerance in units of the proxy noise. |

## `budget` — {class}`~madmax_calibration.config.BudgetConfig` ⚠

| Field | Default |
|---|---|
| `max_hf_measurements` | 25 |
| `max_lf_measurements` | 60 |
| `max_booster_moves` | 60 |
| `max_total_hours` | 60.0 |

## `cost` — {class}`~madmax_calibration.config.CostConfig` ⚠

Expected-cost model $C(u_B, \ell)$ in hours, used by the cost-aware
acquisition (Step 1 design §12).

| Field | Default |
|---|---|
| `hf_measurement` | 1.0 |
| `lf_measurement` | 0.10 |
| `antenna_alignment` | 0.20 |
| `move_base` | 0.05 |
| `move_per_normalized_distance` | 0.10 |

## `trust_region` — {class}`~madmax_calibration.config.TrustRegionConfig`

Policy from Step 1 design §14.

| Field | Default | Meaning |
|---|---|---|
| `initial_size` | 0.25 | Half-width in normalized space. |
| `min_size` / `max_size` | 0.02 / 0.6 | Bounds on the region size. |
| `expand_factor` | 1.6 | Applied after `success_tolerance` consecutive validated improvements. |
| `shrink_factor` | 0.5 | Applied after `failure_tolerance` consecutive failures. |
| `success_tolerance` | 2 | |
| `failure_tolerance` | 3 | |

## `step1` — {class}`~madmax_calibration.config.Step1Config`

Acquisition and decision gates (Step 1 design §9–13, 16–17).

| Field | Default | Meaning |
|---|---|---|
| `n_candidates` | 128 | Sobol candidate-pool size per proposal (use a power of two). |
| `lambda_info` | 0.15 | Weight of the information term (posterior sd) in the utility. Raise early in a campaign, lower for pure exploitation. |
| `soft_feasibility_threshold` | 0.5 | Candidates below this learned success probability are vetoed. |
| `ei_noise_factor` | 0.3 | A new HF measurement requires EI > factor × HF noise (the "don't chase noise" gate). |
| `incumbent_sd_factor` | 1.5 | Replicate the incumbent when its posterior sd exceeds this multiple of the HF noise. |
| `rebaseline_after_hours` | 12.0 | Staleness limit before a re-baseline action is forced (drift rule). |
| `n_theta_samples` | 8 | Posterior $\theta$ samples pushed through the simulator per prediction. More samples = smoother uncertainty, linearly more simulator calls. |
| `min_lf_identification` | 3 | Identify-first rule (minimal Phase-3.4 precursor): while fewer than this many physics-channel LF measurements exist, spend ~0.1 h reflectivity probes before new HF candidates. |
| `seed` | 0 | Seed for the Sobol sampler and prediction sampling. |

## `step5` — {class}`~madmax_calibration.config.Step5Config`

Joint-inference priors (Step 5 design §15: informative priors are *not
optional* — they are what separates $\theta$ from discrepancy).

| Field | Default | Meaning |
|---|---|---|
| `observation_level` | `"curve_summary"` | What each HF measurement contributes to the joint fit: the curve-summary vector (J, log peak, centroid, bandwidth, flatness — roadmap Phase 1.1) or, with `"scalar"`, only J (the pre-1.1 behaviour, kept for A/B benchmarking). |
| `lf_channel` | `"physics"` | How LF reflectivity data enter the joint fit (roadmap Phase 1.2): `"physics"` routes the reflectivity/group-delay summaries through the simulator so they constrain θ jointly; `"affine"` keeps only the scalar-link fallback; `"off"` ignores LF data. |
| `prior_sd_z_offset` ⚠ | 0.5 mm | Gaussian prior sd for the stack z-offset. |
| `prior_sd_compression` ⚠ | 0.25 mm | Gaussian prior sd for the uniform gap error. |
| `prior_sd_log_loss` | 0.7 | Gaussian prior sd for the log loss-scale. |
| `discrepancy_amplitude_prior` | 0.05 | Half-normal prior scale for each component's discrepancy GP amplitude, relative to that component's response scale. Tighter ⇒ residuals pushed into $\theta$/noise; looser ⇒ risk of discrepancy absorbing physics. |
| `discrepancy_sigma_floor` | 5.0 | Floor on each amplitude prior, in units of the component's median measurement sigma — gives unmodelled systematics (e.g. a curve tilt shifting the band centroid) an affordable home in the discrepancy channel instead of biasing $\theta$. |
| `discrepancy_lengthscale_bounds` | (0.1, 2.0) | GP lengthscale bounds in normalized control space. |
| `noise_inflation_prior` | 1.0 | Half-normal prior scale for the shared noise-inflation factor, in units of each component's typical measurement sigma. |
| `drift_rate_prior` | 0.002 | Gaussian prior sd for the linear drift rate ($J$/hour). |
| `min_hf_points_for_inference` | 3 | Below this, Step 5 refuses to fit. |
| `prior_sensitivity_check` | True | Refit with 3× wider discrepancy priors and flag parameters that move by more than their posterior sd as weakly identifiable. Costs one extra fit per iteration. |

## `step7` — {class}`~madmax_calibration.config.Step7Config`

| Field | Default | Meaning |
|---|---|---|
| `improvement_noise_factor` | 0.5 | Resolvability threshold relative to HF noise. |
| `patience` | 3 | Consecutive unresolvable iterations before stopping. |
| `target_objective` | None | Optional absolute early-stop target for $J$. |

## Tuning guidance

- **Exploration vs exploitation**: `step1.lambda_info` and
  `trust_region.initial_size` are the two main dials. The information term
  matters most while the $\theta$ posterior is still wide.
- **Chasing noise**: if the loop keeps measuring candidates that don't
  validate, raise `step1.ei_noise_factor` or `step7.improvement_noise_factor`.
- **Budget-starved runs**: lower `n_baseline_replicates` to 2 and rely on
  the noise-inflation term; raise `cost.hf_measurement` relative to
  `cost.lf_measurement` to push the acquisition toward LF probes.
- **Sluggish convergence to the correction**: check that the exploit
  candidate is not trust-region-limited (`trust_region.max_size`) and that
  `control` limits actually span the expected correction.
- **Overconfident models** (Step-6 validation reports `overconfident`):
  loosen `discrepancy_amplitude_prior` or `noise_inflation_prior`.

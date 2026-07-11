# Roadmap: conceptual upgrades to the calibration pipeline

**Status:** agreed development roadmap, v1
**Baseline:** the implemented seven-step loop (see {doc}`user_guide/architecture`)
and its documented defaults ({doc}`design/DESIGN_DECISIONS`).
**Guiding metric:** information gained per hour of cryostat time — the
pipeline is bound by high-fidelity (HF) boost-factor measurements at
~1 h each, so every item below is judged by how many HF hours it saves
per calibrated frequency window, or by how much it improves the quality
and trustworthiness of the final configuration.

## Operating context (fixed assumptions)

These facts anchor the priorities and scope:

- **Prototype MADMAX with 3 disks** is the relevant detector for this
  roadmap. Control and detector-state dimensionality stay small, the 1D
  transfer-matrix simulator stays millisecond-fast, and no emulator or
  dimension-reduction work is needed. Full-scale (80-disk) MADMAX is
  out of scope until the experiment gets there.
- **The scan campaign covers ~12 frequency windows between 18 and
  24 GHz.** Calibration is not a one-off: it will be repeated per
  window, which makes cross-window transfer (Phase 5) a first-class
  goal rather than an afterthought.
- **The gradient-method boost-factor determination remains a sealed
  oracle.** The loop wraps it, costs it, and consumes its output curve;
  it does not read or modify its internals. Where local sensitivity
  information is wanted, the loop commands its *own* measurements
  (Phase 1.3) rather than tapping the gradient method's intermediate
  data.

## Explicitly out of scope

| Dropped item | Reason |
|---|---|
| Harvesting the gradient method's internal small-movement/RF data as free Jacobian observations | The gradient method stays a self-contained, unmodified oracle. Substitute: self-commanded perturbation measurements (Phase 1.3). |
| Scaling mitigations for full-scale MADMAX (neural emulators for a slow 3D simulator, active subspaces, gradient-based acquisition at high dimension) | The prototype has 3 disks; the current architecture holds as-is. Revisit only when a full-scale campaign is planned. |

---

## Phase 0 — Alignment and benchmarking groundwork

Small enabling items; everything later is measured against these.

### 0.1 Align defaults with the prototype  *(effort: S)* — ✅ implemented

- Set `SimulatorConfig.n_disks = 3` as the default and re-tune the mock
  truth/limits so the synthetic problem stays meaningfully hard.
- Parameterize the campaign through a **settings file** rather than a
  fixed grid: `settings/prototype.toml` defines any number of
  `[[disk_configuration]]` tables (per window: the three prototype
  spacings mirror–d1, d1–d2, d2–d3 in mm, the booster–antenna distance,
  and the target window), plus every other pipeline parameter. The
  shipped default covers 12 windows over 18–24 GHz, but neither number
  is fixed — `examples/generate_settings.py` writes a file for any
  range/window count, and `madmax_calibration.settings.load_settings`
  rejects unknown keys loudly.
- Acceptance met: test suite green on the 3-disk defaults (including
  settings round-trip tests); the settings-driven example reproduces a
  statistically significant improvement (~12σ at seed 0, window 1).

### 0.2 A/B benchmark harness  *(effort: M)* — ✅ implemented

Every algorithmic change in this roadmap must be judged on the same
footing. `madmax_calibration.benchmark` executes the loop over an
ensemble of seeds (optionally jittering the mock truth) and reports:

- HF measurements (and total hours) to reach within measurement noise
  of the achievable optimum (computed analytically from the mock truth),
- final validated improvement and its significance,
- posterior calibration of the Step-6 predictions (2σ coverage, RMS
  standardized residual),
- safety and budget compliance (must always be 100%).

Acceptance met: one command compares any number of settings files —

```bash
python -m madmax_calibration.benchmark settings/a.toml settings/b.toml --runs 5
```

All subsequent phases cite this harness in their acceptance criteria.

---

## Phase 1 — Information architecture (largest per-run gains)

Change *what information each measurement yields* before changing how
decisions are made.

### 1.1 Curve-summary likelihood in Step 5  *(effort: M, highest priority)* — ✅ implemented

**Now:** Step 5 fits only the scalar objective J. **Problem:** an HF
measurement returns an ~80-bin β²(ν) curve; frequency shift, amplitude
loss and bandwidth change are distinguishable at curve level but
degenerate at scalar level. The synthetic runs showed the consequence
directly: a near-degenerate (z-offset, compression, mirror) likelihood
ridge with an overconfident Laplace approximation.

**Change (as implemented):** each HF record carries the smooth summary
vector (J, log peak, band centroid, bandwidth, flatness) with
Monte-Carlo-propagated uncertainties
(`madmax_calibration.summaries.CurveSummarizer`); Step 5 fits all
components jointly with one discrepancy GP per component, amplitude
priors floored at a few measurement sigmas
(`discrepancy_sigma_floor`) so unmodelled systematics stay in the
discrepancy channel, and one shared noise-inflation factor.
`observation_level = "scalar"` retains the old behaviour for A/B runs;
records without summaries fall back to it with a diagnostic. Full
curve-level inference (design §4.3) remains the optional follow-up once
the Phase-2 engine is in.

**Acceptance (measured):**

- *Clean-data A/B on identical datasets:* z-offset posterior sd
  62 → 8.5 µm (~7×), compression 41 → 8.6 µm (~5×), and the loss
  parameter goes from prior-dominated (sd 0.70, wrong sign) to
  identified (0.37 ± 0.044 vs truth 0.30). Enforced by
  `tests/test_curve_summaries.py`.
- *Full-loop benchmark (3 seeds, window 1):* HF-to-converge 14.0 → 11.7
  (−16%), hours 18.1 → 15.6, compression error 203 → 120 µm at equal
  improvement significance (3/3 both) and 100% safety/budget
  compliance; the example run's validated improvement rose from 12.5σ
  to 15.3σ at the same HF count.
- *Honest residual:* an in-loop z-offset bias of ~300 µm is **common to
  both observation levels** — it stems from unmodelled constant
  systematics (curve tilt, focus offset) buying the cheap θ_z
  explanation, and the prior-sensitivity check now flags `z_offset` as
  weakly identifiable in exactly these runs. Removing it is the job of
  Phase 2 (misspecification-robust amortized inference) and Phase 4.1
  (drift on θ), not of the observation level.

### 1.2 Physics-routed low-fidelity channel (reflectivity / group delay)  *(effort: L, deepest structural win)*

**Now:** the LF proxy is a scalar tied to J by a learned affine link —
the weakest element of the current design. **Insight:** for dielectric
haloscopes, reflectivity and group-delay curves strongly constrain the
disk geometry and losses; MADMAX's own proof-of-principle work infers
boost behaviour from exactly these observables.

**Change:**

1. extend the 1D transfer-matrix simulator with a reflection-coefficient
   solve `reflectivity(nu; u, theta)` (same matrix machinery, incoming
   wave instead of the axion source term);
2. model LF measurements as `y_LF = S_LF(u, theta) + r_LF + eps` — the
   structure the Step-5 design note already specifies — so cheap RF
   curves constrain θ *jointly* with HF data;
3. update the mock hardware to return a reflectivity curve (with its
   own noise/systematics) instead of the affine scalar;
4. keep the affine link only as a fallback for proxies with no
   simulator counterpart.

**Effect:** the information economy inverts — geometry calibration
flows mainly through ~0.1 h RF measurements; ~1 h HF measurements are
reserved for validating the objective scale and pinning the
discrepancy. **Acceptance:** benchmark shows a large reduction in HF
count at equal final quality, with the LF channel's discrepancy model
preventing overconfident pooling.

### 1.3 Optional: self-commanded perturbation measurements  *(effort: M, stretch)*

Instead of touching the gradient method, add a new Step-1 action type:
a *local sensitivity probe* — a short sequence of small, hard-feasible
disk/mirror offsets around the current configuration, each measured
with the cheap RF channel from 1.2. This yields finite-difference
sensitivity data the joint model can absorb (equivalently: derivative
information bought explicitly, at LF cost, when the acquisition decides
it is worth it). Depends on 1.2; the VoI layer (Phase 3) prices it.

---

## Phase 2 — Inference engine: amortized simulation-based inference

**Decision:** go directly to modern amortized SBI (neural posterior
estimation, NPE) as the Step-5 engine, replacing joint MAP + Laplace.

### 2.1 Amortized NPE for the detector state  *(effort: L)*

- **Training (offline, per campaign):** simulate large batches of
  (θ, u, window) → observable summaries (Phase-1 curve summaries and
  reflectivity summaries), and train a conditional neural density
  estimator q(θ | summaries, u, window). Condition on the window index
  or the nominal q0(W) so *one* network serves all ~12 windows between
  18 and 24 GHz.
- **Misspecification robustness:** the real detector is *not* the
  simulator. Two mandatory mitigations: (i) inject the stochastic
  discrepancy model (GP-sampled r, noise inflation, drift draws) into
  the training simulations so the amortized posterior is trained to be
  appropriately wide under model error; (ii) adopt a robust-NPE variant
  for outlier-resistant conditioning. The online GP discrepancy on the
  objective residual is *kept* — NPE handles θ, the discrepancy channel
  keeps absorbing what θ cannot explain.
- **Online use:** posterior evaluation is a millisecond forward pass —
  Step 5 becomes trivially cheap per iteration, and
  `Step5Result.theta_samples` is now exact sampling from the amortized
  posterior instead of a Gaussian Laplace draw. The Step-6 interface is
  unchanged by design.
- **Dependency hygiene:** training uses a standard SBI stack (e.g. the
  `sbi` toolkit / PyTorch) as an *offline, optional* dependency; the
  trained network is exported to plain weights so the online loop keeps
  its numpy/scipy-only runtime.

### 2.2 Calibration validation of the posterior  *(effort: M)*

Replace Laplace-specific tests with simulation-based calibration (SBC)
and empirical coverage checks in the benchmark harness: rank-uniformity
of true θ under the amortized posterior across the mock-truth ensemble,
including under injected discrepancy. The observed failure mode of the
current engine — overconfident uncertainty on the degeneracy ridge — is
the explicit regression target.

---

## Phase 3 — Decision layer: value of information end-to-end

### 3.1 Thompson sampling on θ as the new acquisition baseline  *(effort: S)*

Sample θ from the (now cheap, Phase-2) posterior, optimize
J_sim(u, θ_s) + r_s inside the safe domain, propose the argmax.
Hyperparameter-free, physically directed exploration, and an honest
baseline for 3.2. Retires the bolt-on "exploit candidate" refinement.

### 3.2 Value-of-information acquisition replacing the fallback ladder  *(effort: L)*

Replace EI + σ-bonus + the hand-tuned ladder (EI-vs-noise factor,
incumbent-sd factor, re-baseline timer) with one question: *which
action — new HF, LF probe, sensitivity probe (1.3), replicate,
re-baseline — maximizes expected improvement of the final validated J
per hour?* One-step-lookahead (knowledge-gradient-style) VoI per unit
cost over the discrete action set unifies fidelity choice, replication
and re-baselining. Acceptance: the benchmark shows equal-or-better HF
efficiency with *fewer tuned constants* (target: remove at least four
of the current Step-1 thresholds).

### 3.3 Cost-aware, campaign-grounded stopping  *(effort: M)*

Two layers replace the patience counter:

- **Per-window:** stop when max expected VoI per hour across all
  actions falls below a threshold — the principled acquisition-based
  stopping rule from the recent cost-aware BO literature.
- **Campaign-level:** the threshold itself comes from physics
  economics: marginal predicted scan-rate gain × *remaining operating
  time in this window and campaign* vs marginal calibration cost.
  With ~12 windows the remaining-campaign term is concrete and
  computable, and calibration depth automatically adapts (earlier
  windows justify more calibration than the last one).

### 3.4 Identification-first experiment design  *(effort: M)*

For the first measurements in a window, choose configurations that
maximize Fisher information about θ (D-optimal design; the 1D
transfer-matrix simulator is analytically differentiable) rather than
expected improvement. An explicit identify→exploit schedule replaces
the fixed λ_info weight. With Phase-5 transfer priors, the identify
phase shrinks automatically as windows accumulate.

---

## Phase 4 — Model-structure upgrades

### 4.1 Drift on θ, not on J  *(effort: M)*

Replace the linear-in-J drift with a random-walk/OU state on the
drifting geometry parameter(s) (stack offset first). Drift then
propagates to J *through the simulator* — correctly configuration-
dependent — and re-baselining becomes an inference-driven VoI action
(3.2) instead of a timer. The Phase-2 training already includes drift
draws, so the amortized posterior is drift-aware from the start.

### 4.2 Robust objective under achieved-geometry noise  *(effort: M)*

Optimize E[J(ũ)] under p(ũ | u) — actuator repeatability, hysteresis
and residual drift — instead of J(u). Given the area-law trade-off
(sharper peak ⇔ tighter tolerance), this can change *which*
configuration is optimal, not just its predicted value. Includes a
simple direction-dependent hysteresis model and an
approach-from-one-direction move policy to shrink ũ-uncertainty at the
source. Completes the achieved-geometry integral that Step 6 currently
documents as a plug-in approximation.

### 4.3 Contextual antenna model  *(effort: M)*

Model the coupling surface jointly as A(u_A; u_B) across iterations
instead of re-solving alignment per booster state. Target: alignment
cost ≈ one confirmation measurement after the first few iterations,
plus propagation of alignment uncertainty into the antenna-aligned
prediction F(u_B) (removing the documented plug-in gap).

### 4.4 Trust-region role separation  *(effort: S)*

The trust region currently conflates safety with model trust. Keep the
hard max-step limit (safety, non-negotiable); retire the size-adaptive
region and let extrapolation be governed by the discrepancy model's
growing variance plus the existing extrapolation diagnostics. The
physics surrogate is valid across the whole hard box — the synthetic
runs showed the trust region slowing early convergence by putting the
needed correction out of reach.

---

## Phase 5 — Campaign scale: transfer across the 12 windows

The sleeper with the largest cumulative payoff. Detector-state
parameters (misalignments, losses, actuator behaviour) persist when the
experiment retunes from one window to the next; the current design
restarts from priors every time.

### 5.1 Hierarchical θ-transfer between windows  *(effort: M)*

Split θ into window-independent components (mechanical offsets, loss
scale, actuator behaviour) and window-specific ones. The posterior of
the shared components from windows 1..k becomes the prior for window
k+1 (with a widening factor for slow drift between campaigns).
Target: after the first 2–3 windows, a new window needs ~2–3 HF
measurements instead of ~13.

### 5.2 Discrepancy transfer  *(effort: M)*

The simulator's systematic error is expected to vary smoothly with
frequency: model the discrepancy jointly across windows (window index /
centre frequency as an extra GP input, or a meta-learned prior over
discrepancy hyperparameters) so early windows inform later ones.

### 5.3 Window-conditioned amortization + campaign scheduler  *(effort: L)*

The Phase-2 NPE is already conditioned on the window — 5.3 closes the
loop at campaign level: a scheduler that allocates the calibration
budget *across* the 12 windows using the campaign-economics rule (3.3),
decides window order, and carries the transfer state. Output: a
campaign-level feasibility report (total calibration hours vs total
scan time, per-window improvements) alongside the per-window reports.

---

## Sequencing and dependencies

```text
Phase 0 ──► 1.1 ──► 2.1 ──► 2.2
              │        │
              ▼        ▼
             1.2 ─► (1.3, needs 1.2)      3.1 ──► 3.2 ──► 3.3
              │                             ▲              ▲
              └───── feeds NPE training ────┘              │
                                                4.1 ───────┘ (drift as VoI action)
             4.2, 4.3, 4.4: independent, any time after Phase 0
             5.1 ──► 5.2 ──► 5.3  (needs 2.1 for window-conditioned NPE)
```

Recommended order of attack: **0.1–0.2 → 1.1 → 1.2 → 2.1–2.2 → 3.1–3.3
→ 5.1**, interleaving Phase-4 items opportunistically.

## What deliberately stays as-is

The seven-step decomposition; the exact-hard / learned-soft constraint
split (damage-relevant limits are never learned); achieved-geometry
discipline; the auditability of every proposal and measurement record;
final HF validation of any accepted configuration. These are correct,
match the state of practice, and every phase above must preserve them —
the end-to-end safety and budget tests are non-negotiable acceptance
criteria for all items.

## Literature anchors

- Physics-informed priors for online machine tuning: Roussel et al.,
  *Bayesian optimization algorithms for accelerator physics*, PRAB 27,
  084801 (2024); *Leveraging prior mean models for faster BO of
  particle accelerators* (arXiv:2403.03225).
- Reflectivity/group delay as geometry information: *A first proof of
  principle booster setup for the MADMAX dielectric haloscope*, EPJ C
  80, 392 (2020); MADMAX reflectivity-calibration thesis work (2025).
- Amortized SBI / NPE and robustness under misspecification:
  simulation-based inference literature 2021–2026, incl. robust
  variational NPE (arXiv:2509.05724).
- Cost-aware acquisition and stopping: *Cost-aware Stopping for
  Bayesian Optimization* (arXiv:2507.12453); cost-sensitive
  multi-fidelity BO (arXiv:2405.17918).
- Transfer/meta-learned priors for recurring re-optimization:
  Rothfuss et al., *Meta-learning reliable priors in the function
  space* (NeurIPS 2021); *Provable accelerated BO with knowledge
  transfer* (arXiv:2511.03125).

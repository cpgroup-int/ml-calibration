# MADMAX Closed-Loop Calibration

Implementation of the seven-step closed-loop calibration algorithm for the
MADMAX detector described in
[`docs/madmax_closed_loop_calibration_proposal.md`](docs/madmax_closed_loop_calibration_proposal.md)
(version 3) and its step-specific technical design notes.

The system starts from an already-optimized nominal disk configuration,
treats the existing gradient-method boost-factor determination as the
high-fidelity experimental objective, and iteratively finds the best
calibrated real-detector configuration using **physics-informed, safe,
budget-aware Bayesian optimization** together with a **joint
detector-state / discrepancy / noise / drift calibration model**.

```
Step 0  initialize, measure baseline, estimate noise, freeze hard limits
      ┌─────────────────────────────────────────────────────────────┐
      │ Step 1  propose booster correction u_B + measurement action  │
      │ Step 2  set booster geometry, record achieved geometry       │
      │ Step 3  align antenna (x,y) for this booster state           │
      │ Step 4  measure selected observable / boost factor           │
      │ Step 5  jointly update theta, discrepancy, noise, drift      │
      │ Step 6  rebuild optimizer-facing posterior predictive model  │
      │ Step 7  stop or repeat ──────────────────────────────────────┘
      └── final high-fidelity validation + feasibility report
```

## Layout

| Module | Implements | Design doc |
|---|---|---|
| `config.py` | every tunable default in one place | Step 1 §26 open choices |
| `records.py` | measurement records, proposals, dataset `D_{1:t}` | Step 4 §13, Step 1 §21 |
| `control.py` | control basis `B`, `u_B = T(x_B)`, control→geometry map | proposal §4, Step 1 §5 |
| `simulator.py` | fast 1D transfer-matrix `β²_sim(ν; q, θ)` stand-in | proposal §2.3 |
| `objectives.py` | scalar physics objectives `J` + MC uncertainty | proposal §6, Step 4 §9.5 |
| `gp.py` | minimal exact GP (RBF, fixed noise, ML-II fit) | — |
| `constraints.py` | exact hard filtering + learned soft constraints | proposal §9 |
| `hardware.py` | `HardwareInterface` + `MockHardware` (hidden errors, hysteresis, drift, noise) | proposal §2.5, §10 |
| `steps/step0_initialize.py` | baseline + noise estimate + feasibility screen | proposal Step 0 |
| `steps/step1_propose.py` | trust-region constrained, noise/cost/budget-aware acquisition with replicate / re-baseline / LF-probe / stop fallbacks | Step 1 design |
| `steps/step2_set_geometry.py` | move + achieved-readback verification | proposal Step 2 |
| `steps/step3_antenna.py` | incumbent check → local scan + quadratic fit → 2D GP-BO fallback | Step 3 design |
| `steps/step4_measure.py` | measurement wrapper, QC flags, uncertainty, cost | Step 4 design |
| `steps/step5_inference.py` | joint MAP (θ, discrepancy GP, noise inflation, drift) + Laplace + identifiability checks | Step 5 design |
| `steps/step6_predictive.py` | sample-based posterior predictive `p(J_HF(u)|D)`, latent vs observation, extrapolation/staleness flags | Step 6 design |
| `steps/step7_stopping.py` | stopping rules | proposal Step 7 |
| `loop.py` | orchestrator + feasibility report | proposal §11–12 |

## Quick start

```bash
pip install -e ".[dev]"
pytest                                      # validation suite (~3 min)
python examples/run_synthetic_calibration.py
```

The example runs the loop against the simulated detector, which hides a
+0.6 mm stack offset, +0.25 mm gap-compression error, extra dielectric
loss, a mis-centred antenna beam, a mis-focused mirror, actuator
hysteresis, 2 µm/h drift and realistic measurement noise.  A typical run
recovers a statistically significant improvement (several σ above
measurement noise), an accurate estimate of the correctable detector-state
errors, and stops on its own when remaining improvement is not resolvable
above noise.

## Using it on real hardware

Implement `madmax_calibration.hardware.HardwareInterface` around the real
detector control + the existing gradient-method boost-factor routine, and
substitute the real MADMAX simulator behind the `BoostSimulator` interface
(`beta2`, `predict_J`).  Nothing else in the package knows about the mock.

Before that, read
[`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md): it lists every
default that stands in for a decision the design notes defer to the MADMAX
team (control basis, hard limits, objective choice, fidelities, budgets),
and what is deliberately not implemented yet (multi-objective/Pareto,
curve-level inference, full Bayesian sampling).

## Safety model

Damage-relevant constraints (travel limits, minimum gaps, maximum step
size) are enforced **exactly** before any candidate reaches the hardware —
they are never learned by failure.  Only non-damaging measurement-quality
failures train the statistical soft-constraint model.  Large commanded
moves are automatically split into steps below the maximum safe step size.

## Tests

The test suite implements the pre-hardware validation checklists from the
design notes: constraint filtering, noise/budget/drift responses and
trust-region behaviour (Step 1 §25), antenna-alignment beam tests (Step 3
§20), measurement-record integrity (Step 4 §21), synthetic
recovery/confounding/drift/multi-fidelity inference tests (Step 5 §20),
closure/discrepancy/extrapolation/coverage tests (Step 6 §27), and an
end-to-end closed-loop run that must find a validated improvement without
ever proposing an unsafe configuration.

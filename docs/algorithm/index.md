# The algorithm, step by step

One page per step of the seven-step closed-loop calibration algorithm.
Each page covers: the step's role as defined in its design note, the
implementation walk-through, its inputs/outputs, the relevant
configuration knobs, failure handling, and the tests that validate it.

```{toctree}
:maxdepth: 1

step0
step1
step2
step3
step4
step5
step6
step7
```

The corresponding original design documents are under
{doc}`../design/index`; the implementation-level architecture is in
{doc}`../user_guide/architecture`.

| Step | Module | One-line role |
|---|---|---|
| 0 | `steps/step0_initialize.py` | Baseline, noise estimate, feasibility screen |
| 1 | `steps/step1_propose.py` | Propose next booster correction + measurement action |
| 2 | `steps/step2_set_geometry.py` | Move booster, verify achieved geometry |
| 3 | `steps/step3_antenna.py` | Align antenna for the fixed booster state |
| 4 | `steps/step4_measure.py` | Measure, reduce, estimate uncertainty, flag, cost |
| 5 | `steps/step5_inference.py` | Joint detector-state/discrepancy/noise/drift update |
| 6 | `steps/step6_predictive.py` | Optimizer-facing posterior predictive model |
| 7 | `steps/step7_stopping.py` | Stop-or-repeat decision |

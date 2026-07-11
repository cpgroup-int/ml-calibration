# Getting started

## Requirements

- Python ≥ 3.10
- `numpy` ≥ 1.24, `scipy` ≥ 1.10 (installed automatically)
- `pytest` for the validation suite

The package is deliberately dependency-light: the Gaussian processes, the
acquisition logic and the physics simulator are self-contained, so no
BoTorch/PyTorch stack is required.

## Installation

From the repository root:

```bash
pip install -e ".[dev]"
```

To also build this documentation:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Verify the installation

Run the validation suite (~2–3 minutes; it implements the pre-hardware
validation checklists from the design notes, see {doc}`testing`):

```bash
pytest
```

## Run a full synthetic calibration

```bash
python examples/run_synthetic_calibration.py        # optional: seed argument
```

This runs the complete closed loop against the simulated detector
({class}`~madmax_calibration.hardware.MockHardware`), which hides detector
state errors, actuator hysteresis, drift and measurement noise from the
algorithm. A detailed narrative of what happens during this run is in
{doc}`example_walkthrough`.

## Minimal programmatic use

The convenience factory builds the whole stack (nominal geometry →
control map → simulator → mock hardware → loop):

```python
from madmax_calibration.loop import build_default_loop

loop = build_default_loop(seed=0)
result = loop.run(max_iterations=20, verbose=True)

print(result.feasibility_report["improvement_over_noise"])
print(result.u_B_star)           # best validated booster correction [m]
print(result.step5.theta_map)    # inferred detector-state parameters
```

Assembling the pieces explicitly — which is what you will do when
substituting the real simulator and detector control — looks like this:

```python
import numpy as np

from madmax_calibration.config import CalibrationConfig
from madmax_calibration.control import ControlMap
from madmax_calibration.hardware import MockHardware   # replace with your hardware
from madmax_calibration.loop import CalibrationLoop
from madmax_calibration.simulator import BoostSimulator, nominal_half_wave_geometry

config = CalibrationConfig()
config.budget.max_hf_measurements = 20            # tune everything via config
config.objective = "scan_rate"                    # or "smooth_min", "peak"

gaps, thicknesses = nominal_half_wave_geometry(config.simulator)   # q0(W)
control_map = ControlMap(config.control, config.simulator, gaps, thicknesses)
simulator = BoostSimulator(config.simulator, control_map)
hardware = MockHardware(simulator, config, seed=1234)

loop = CalibrationLoop(hardware, simulator, config)
result = loop.run(max_iterations=25)
```

Every default in {class}`~madmax_calibration.config.CalibrationConfig` is
documented in {doc}`user_guide/configuration`; the ones that stand in for
decisions the design notes defer to the MADMAX team are flagged in
{doc}`design/DESIGN_DECISIONS`.

## Where to go next

- {doc}`example_walkthrough` — an annotated synthetic calibration run.
- {doc}`user_guide/architecture` — how the modules map onto the seven-step
  algorithm.
- {doc}`user_guide/hardware` — what to implement to run against the real
  detector.
- {doc}`algorithm/index` — every step in detail, with the decision logic
  and its configuration knobs.

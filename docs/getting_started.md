# Getting started

## Requirements

- Python ≥ 3.11
- `numpy` ≥ 1.24, `scipy` ≥ 1.10 (installed automatically)
- `torch` ≥ 2.1 and `zuko` ≥ 1.1 for the amortized NPE engine
  (installed automatically; a CPU-only torch build is sufficient:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- `pytest` for the validation suite

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
python examples/run_synthetic_calibration.py                       # active window
python examples/run_synthetic_calibration.py --window window_07    # another window
```

This runs the complete closed loop against the simulated detector
({class}`~madmax_calibration.hardware.MockHardware`), which hides detector
state errors, actuator hysteresis, drift and measurement noise from the
algorithm. **Everything is driven by the settings file**
(`settings/prototype.toml`): the disk configurations — three spacings plus
a booster–antenna distance per frequency window — the budgets, the
acquisition constants, and the mock detector's hidden errors. A detailed
narrative of what happens during this run is in {doc}`example_walkthrough`;
the file format is documented in {doc}`user_guide/configuration`.

Generate a settings file for a different campaign (any frequency range,
any number of windows) and benchmark pipeline variants against each other:

```bash
python examples/generate_settings.py -o settings/my_campaign.toml \
    --f-min 18 --f-max 24 --windows 12
python -m madmax_calibration.benchmark settings/prototype.toml \
    settings/my_campaign.toml --runs 3
```

## Minimal programmatic use

The settings-based factory builds the whole stack (nominal geometry →
control map → simulator → mock hardware → loop):

```python
from madmax_calibration.loop import build_loop_from_settings
from madmax_calibration.settings import load_settings

settings = load_settings("settings/prototype.toml")
loop = build_loop_from_settings(settings, seed=0)
result = loop.run(max_iterations=20, verbose=True)

print(result.feasibility_report["improvement_over_noise"])
print(result.u_B_star)           # best validated booster correction [m]
print(result.step5.theta_map)    # inferred detector-state parameters
```

(`build_default_loop(seed=0)` remains as a shorthand: it uses the
repository settings file when present.)

To use the amortized neural-posterior inference engine (roadmap Phase 2 —
calibrated, ~3× faster), set `inference_engine = "amortized_npe"` in the
settings file (the shipped `weights/npe_prototype.pt` covers window 1 of
the prototype). Retrain for another window or control basis with:

```bash
python examples/train_npe.py --window window_07 --out weights/npe_window07.pt
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

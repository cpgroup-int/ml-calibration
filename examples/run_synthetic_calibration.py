"""Run the full closed-loop calibration against the simulated detector.

The mock detector hides detector-state errors (stack offset +0.6 mm, gap
compression +0.25 mm, extra dielectric loss), a mis-centred antenna beam,
a mis-focused mirror, actuator hysteresis, slow drift and measurement
noise.  The loop starts from the degraded nominal configuration and must
find a validated improvement within budget.

Usage:  python examples/run_synthetic_calibration.py [seed]
"""

import json
import sys
import time

import numpy as np

from madmax_calibration.loop import build_default_loop


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    loop = build_default_loop(seed=seed)
    result = loop.run(max_iterations=20, verbose=True)

    print("\n================ feasibility report ================")
    for key, value in result.feasibility_report.items():
        print(f"  {key}: {value}")

    truth = loop.hardware.truth.theta
    est = result.step5.theta_map
    sd = np.sqrt(np.diag(result.step5.theta_cov))
    print("\n================ detector-state inference ================")
    for i, name in enumerate(("z_offset", "compression", "log_loss")):
        t = getattr(truth, name)
        e = getattr(est, name)
        print(f"  {name:12s} truth={t:+.3e}  estimate={e:+.3e} +/- {sd[i]:.1e}"
              f"  [{result.step5.classification[name]}, {result.step5.identifiability[name]}]")

    print(f"\nbest correction u_B* (um): {np.round(result.u_B_star * 1e6, 1)}")
    print(f"elapsed: {time.time() - t0:.1f}s")

    with open("calibration_history.json", "w") as fh:
        json.dump(result.history, fh, indent=2, default=str)
    print("per-iteration history written to calibration_history.json")


if __name__ == "__main__":
    main()

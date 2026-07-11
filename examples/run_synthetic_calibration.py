"""Run the full closed-loop calibration against the simulated detector.

All parameters — including the disk configurations (three spacings plus
booster-antenna distance per frequency window) and the simulated
detector's hidden errors — come from a TOML settings file.

Usage:
    python examples/run_synthetic_calibration.py [--settings PATH]
        [--window NAME] [--seed N]
"""

import argparse
import json
import time

import numpy as np

from madmax_calibration.loop import build_loop_from_settings
from madmax_calibration.settings import default_settings_path, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None,
                        help="settings TOML file (default: settings/prototype.toml)")
    parser.add_argument("--window", default=None,
                        help="disk-configuration name (default: file's active one)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    path = args.settings or default_settings_path()
    if path is None:
        raise SystemExit("no settings file found; pass --settings or create settings/prototype.toml")
    settings = load_settings(path)
    window = settings.disk_configuration(args.window)
    print(f"settings: {path}")
    print(f"window:   {window.name} @ {window.target_frequency / 1e9:.2f} GHz "
          f"(+/- {window.window_half_width / 1e9:.2f} GHz), "
          f"spacings {np.round(window.spacings * 1e3, 3)} mm, "
          f"booster-antenna distance {window.booster_antenna_distance * 1e3:.0f} mm")

    t0 = time.time()
    loop = build_loop_from_settings(settings, window=args.window, seed=args.seed)
    result = loop.run(max_iterations=args.iterations, verbose=True)

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

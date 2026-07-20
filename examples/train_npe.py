"""Train and export the amortized NPE weights (roadmap Phase 2.1).

Offline training of the conditional spline flow (PyTorch + zuko) against
the fast simulator, exported as a ``.pt`` checkpoint. Re-run this
whenever the simulator, control basis, or objective in the settings file
changes — the conditioning statistic and network are tied to them.

Usage:
    python examples/train_npe.py [--settings PATH] [--window NAME]
        [--out weights/npe_prototype.pt] [--episodes 24000] [--epochs 300]
"""

import argparse
import time

import numpy as np

from madmax_calibration.amortized import TrainingConfig, train_amortized_posterior
from madmax_calibration.control import ControlMap
from madmax_calibration.settings import default_settings_path, load_settings
from madmax_calibration.simulator import BoostSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default=None)
    parser.add_argument("--window", default=None)
    parser.add_argument("--out", default="weights/npe_prototype.pt")
    parser.add_argument("--episodes", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = args.settings or default_settings_path()
    settings = load_settings(path)
    d = settings.disk_configuration(args.window)
    sim_cfg = settings.simulator_config_for(d.name)
    config = settings.config
    config.simulator = sim_cfg
    thicknesses = np.full(d.n_disks, d.thickness(sim_cfg.disk_index))
    control_map = ControlMap(config.control, sim_cfg, d.spacings, thicknesses)
    simulator = BoostSimulator(
        sim_cfg, control_map, booster_antenna_distance=d.booster_antenna_distance
    )

    print(f"training NPE: window {d.name} @ {d.target_frequency/1e9:.2f} GHz, "
          f"{args.episodes} episodes, {args.epochs} epochs")
    t0 = time.time()
    post = train_amortized_posterior(
        simulator, control_map, config,
        TrainingConfig(n_episodes=args.episodes, epochs=args.epochs, seed=args.seed),
        verbose=True,
    )
    post.save(args.out)
    print(f"wrote {args.out} (final NLL {post.metadata['final_nll']:.4f}, "
          f"{time.time()-t0:.1f}s, conditioning dim {post.featurizer.dim})")


if __name__ == "__main__":
    main()

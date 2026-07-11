"""Generate a complete calibration settings file for an arbitrary campaign.

The frequency range and the number of windows are free choices — nothing
in the pipeline hard-codes them. The generated per-window disk spacings
are the analytic half-wave stand-in for the offline MADMAX disk
optimization; edit the file and replace them with the real values.

Usage:
    python examples/generate_settings.py -o settings/my_campaign.toml \
        --f-min 18 --f-max 24 --windows 12 --disks 3
"""

import argparse

from madmax_calibration.settings import write_settings_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="settings/campaign.toml")
    parser.add_argument("--f-min", type=float, default=18.0, help="lower edge [GHz]")
    parser.add_argument("--f-max", type=float, default=24.0, help="upper edge [GHz]")
    parser.add_argument("--windows", type=int, default=12, help="number of frequency windows")
    parser.add_argument("--disks", type=int, default=3, help="disks per configuration")
    args = parser.parse_args()

    path = write_settings_file(
        args.output,
        f_min=args.f_min * 1e9,
        f_max=args.f_max * 1e9,
        n_windows=args.windows,
        n_disks=args.disks,
    )
    print(f"wrote {path} ({args.windows} windows, {args.f_min}-{args.f_max} GHz, "
          f"{args.disks} disks each)")
    print("Edit the [[disk_configuration]] spacings_mm entries to use real "
          "per-window optimized spacings.")


if __name__ == "__main__":
    main()

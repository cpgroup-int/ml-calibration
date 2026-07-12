"""Settings-file support (roadmap Phase 0.1).

Every tunable of the calibration pipeline can be set in a single TOML
settings file; nothing about the campaign is hard-coded.  In particular
the file defines **any number of named disk configurations** — one per
frequency window — each with:

- the three physical disk spacings of the 3-disk prototype
  (mirror–disk1, disk1–disk2, disk2–disk3), in millimetres,
- the booster–antenna distance (disk3 → antenna, with the focusing
  mirror in between).  This is *not* a stack spacing: in the 1D solver
  the region behind the last disk is handled by the radiation
  condition, so it does not change the boost curve; it parameterizes
  the coupling optics and is carried through to the hardware layer.
- the target frequency window (centre and half-width, in GHz),
- optionally a disk thickness override.

Sections mirror the configuration dataclasses in
:mod:`madmax_calibration.config` field-for-field (SI units: metres, Hz,
hours), plus ``[calibration]`` for top-level choices and an optional
``[mock]`` section for the simulated detector used in examples,
benchmarks and tests.  Unknown keys are rejected loudly rather than
silently ignored.

Use :func:`write_settings_file` to generate a complete, commented
settings file for an arbitrary campaign (frequency range and number of
windows are free choices, e.g. 12 windows between 18 and 24 GHz); the
generated spacings are the analytic half-wave stand-in for the offline
MADMAX disk optimization and are meant to be replaced by real values.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import (
    C_LIGHT,
    AntennaConfig,
    BudgetConfig,
    CalibrationConfig,
    ControlConfig,
    CostConfig,
    SimulatorConfig,
    Step1Config,
    Step5Config,
    Step7Config,
    TrustRegionConfig,
)
from .hardware import MockTruth, NoiseModel
from .simulator import DetectorState


@dataclass
class DiskConfiguration:
    """One nominal detector configuration q0(W) for one frequency window.

    ``spacings`` are the physical gaps (mirror–disk1, disk1–disk2,
    disk2–disk3, ...) in metres; ``booster_antenna_distance`` is the
    disk3→antenna optics path (not a stack spacing).
    """

    name: str
    target_frequency: float          # [Hz]
    window_half_width: float         # [Hz]
    spacings: np.ndarray             # (n_disks,) [m]
    booster_antenna_distance: float  # [m]
    disk_thickness: float | None = None   # [m]; None -> half-wave default

    @property
    def n_disks(self) -> int:
        return len(self.spacings)

    def thickness(self, disk_index: float) -> float:
        if self.disk_thickness is not None:
            return self.disk_thickness
        return C_LIGHT / self.target_frequency / (2.0 * disk_index)


@dataclass
class Settings:
    """Everything loaded from one settings file."""

    config: CalibrationConfig
    disk_configurations: list[DiskConfiguration]
    active: str
    mock_truth: MockTruth = field(default_factory=MockTruth)
    mock_noise: NoiseModel = field(default_factory=NoiseModel)
    path: str = ""

    @property
    def window_names(self) -> list[str]:
        return [d.name for d in self.disk_configurations]

    def disk_configuration(self, name: str | None = None) -> DiskConfiguration:
        name = name or self.active
        for d in self.disk_configurations:
            if d.name == name:
                return d
        raise KeyError(
            f"unknown disk configuration '{name}'; available: {self.window_names}"
        )

    def simulator_config_for(self, name: str | None = None) -> SimulatorConfig:
        """Per-window simulator configuration derived from the base one."""
        d = self.disk_configuration(name)
        return dataclasses.replace(
            self.config.simulator,
            n_disks=d.n_disks,
            target_frequency=d.target_frequency,
            window_half_width=d.window_half_width,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_SECTIONS: dict[str, type] = {
    "simulator": SimulatorConfig,
    "control": ControlConfig,
    "antenna": AntennaConfig,
    "budget": BudgetConfig,
    "cost": CostConfig,
    "trust_region": TrustRegionConfig,
    "step1": Step1Config,
    "step5": Step5Config,
    "step7": Step7Config,
}

_TOP_LEVEL_KEYS = {"objective", "n_baseline_replicates", "seed", "active_configuration"}


def _apply_section(obj, section_name: str, values: dict) -> None:
    valid = {f.name: f for f in dataclasses.fields(type(obj))}
    for key, value in values.items():
        if key not in valid:
            raise ValueError(
                f"unknown key '{key}' in [{section_name}]; "
                f"valid keys: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if isinstance(current, tuple):
            value = tuple(value)
        setattr(obj, key, value)


def _parse_disk_configuration(entry: dict, index: int) -> DiskConfiguration:
    known = {
        "name",
        "target_frequency_ghz",
        "window_half_width_ghz",
        "spacings_mm",
        "booster_antenna_distance_mm",
        "disk_thickness_mm",
    }
    unknown = set(entry) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [[disk_configuration]] #{index}; "
            f"valid keys: {sorted(known)}"
        )
    for required in ("name", "target_frequency_ghz", "spacings_mm",
                     "booster_antenna_distance_mm"):
        if required not in entry:
            raise ValueError(
                f"[[disk_configuration]] #{index} is missing required key '{required}'"
            )
    spacings = np.asarray(entry["spacings_mm"], dtype=float) * 1e-3
    if len(spacings) < 1 or np.any(spacings <= 0):
        raise ValueError(
            f"[[disk_configuration]] '{entry['name']}': spacings_mm must be positive"
        )
    thickness = entry.get("disk_thickness_mm")
    return DiskConfiguration(
        name=str(entry["name"]),
        target_frequency=float(entry["target_frequency_ghz"]) * 1e9,
        window_half_width=float(entry.get("window_half_width_ghz", 0.25)) * 1e9,
        spacings=spacings,
        booster_antenna_distance=float(entry["booster_antenna_distance_mm"]) * 1e-3,
        disk_thickness=None if thickness is None else float(thickness) * 1e-3,
    )


def _parse_mock(data: dict) -> tuple[MockTruth, NoiseModel]:
    truth = MockTruth()
    noise = NoiseModel()
    truth_data = dict(data.get("truth", {}))
    theta_kwargs = {}
    for key in ("z_offset", "compression", "log_loss"):
        if key in truth_data:
            theta_kwargs[key] = float(truth_data.pop(key))
    if theta_kwargs:
        truth.theta = DetectorState(**{**dataclasses.asdict(truth.theta), **theta_kwargs})
    valid_truth = {
        "beam_center",
        "focus_optimum",
        "discrepancy_tilt",
        "drift_rate_z",
        "refl_calibration_bias",
        "gd_delay_offset",
    }
    for key, value in truth_data.items():
        if key not in valid_truth:
            raise ValueError(
                f"unknown key '{key}' in [mock.truth]; valid keys: "
                f"{sorted(valid_truth | {'z_offset', 'compression', 'log_loss'})}"
            )
        if key == "beam_center":
            value = np.asarray(value, dtype=float)
        setattr(truth, key, value)
    _apply_section(noise, "mock.noise", data.get("noise", {}))
    return truth, noise


def load_settings(path: str | Path, active: str | None = None) -> Settings:
    """Load a TOML settings file into a :class:`Settings` bundle."""
    path = Path(path)
    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    known_tables = set(_SECTIONS) | {"calibration", "disk_configuration", "mock"}
    unknown = set(data) - known_tables
    if unknown:
        raise ValueError(
            f"unknown table(s) {sorted(unknown)} in {path.name}; "
            f"valid tables: {sorted(known_tables)}"
        )

    config = CalibrationConfig()

    top = data.get("calibration", {})
    unknown = set(top) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [calibration]; "
            f"valid keys: {sorted(_TOP_LEVEL_KEYS)}"
        )
    if "objective" in top:
        config.objective = str(top["objective"])
    if "n_baseline_replicates" in top:
        config.n_baseline_replicates = int(top["n_baseline_replicates"])
    if "seed" in top:
        config.seed = int(top["seed"])

    for section_name, _cls in _SECTIONS.items():
        if section_name in data:
            _apply_section(getattr(config, section_name), section_name, data[section_name])

    entries = data.get("disk_configuration", [])
    if not entries:
        raise ValueError(
            f"{path.name} defines no [[disk_configuration]] tables; at least one "
            "nominal disk configuration is required"
        )
    disk_configurations = [
        _parse_disk_configuration(entry, i) for i, entry in enumerate(entries)
    ]
    names = [d.name for d in disk_configurations]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate disk-configuration names in {path.name}: {names}")

    truth, noise = _parse_mock(data.get("mock", {}))

    active = active or top.get("active_configuration") or names[0]
    settings = Settings(
        config=config,
        disk_configurations=disk_configurations,
        active=active,
        mock_truth=truth,
        mock_noise=noise,
        path=str(path),
    )
    settings.disk_configuration(active)  # validate the choice
    return settings


def default_settings_path() -> Path | None:
    """The repository's shipped settings file, if present."""
    candidate = Path(__file__).resolve().parents[2] / "settings" / "prototype.toml"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def half_wave_spacings(frequency: float, n_disks: int) -> np.ndarray:
    """Analytic transparent-mode spacings: half a wavelength per gap.

    Stand-in for the offline MADMAX disk-spacing optimization; replace
    the generated values with the real per-window optimized spacings.
    """
    return np.full(n_disks, C_LIGHT / frequency / 2.0)


def generate_disk_configurations(
    f_min: float,
    f_max: float,
    n_windows: int,
    n_disks: int = 3,
    window_half_width: float | None = None,
    booster_antenna_distance: float = 0.10,
) -> list[DiskConfiguration]:
    """Evenly cover [f_min, f_max] with ``n_windows`` adjacent windows."""
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    width = (f_max - f_min) / n_windows
    half = window_half_width if window_half_width is not None else width / 2.0
    configs = []
    for i in range(n_windows):
        centre = f_min + (i + 0.5) * width
        configs.append(
            DiskConfiguration(
                name=f"window_{i + 1:02d}",
                target_frequency=centre,
                window_half_width=half,
                spacings=half_wave_spacings(centre, n_disks),
                booster_antenna_distance=booster_antenna_distance,
            )
        )
    return configs


def _fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return f'"{value}"'
    if value is None:
        raise ValueError("cannot format None; omit the key instead")
    return str(value)


def _section_lines(name: str, obj, skip: set[str] = frozenset()) -> list[str]:
    lines = [f"[{name}]"]
    for f in dataclasses.fields(type(obj)):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        if value is None:
            continue
        lines.append(f"{f.name} = {_fmt(value)}")
    lines.append("")
    return lines


def write_settings_file(
    path: str | Path,
    disk_configurations: list[DiskConfiguration] | None = None,
    config: CalibrationConfig | None = None,
    mock_truth: MockTruth | None = None,
    mock_noise: NoiseModel | None = None,
    f_min: float = 18e9,
    f_max: float = 24e9,
    n_windows: int = 12,
    n_disks: int = 3,
) -> Path:
    """Write a complete, commented settings file.

    If ``disk_configurations`` is omitted, an adjacent-window grid over
    [f_min, f_max] is generated with half-wave stand-in spacings.
    """
    config = config or CalibrationConfig()
    mock_truth = mock_truth or MockTruth()
    mock_noise = mock_noise or NoiseModel()
    if disk_configurations is None:
        disk_configurations = generate_disk_configurations(f_min, f_max, n_windows, n_disks)

    lines: list[str] = [
        "# Calibration settings for the MADMAX closed-loop calibration pipeline.",
        "#",
        "# Every tunable of the pipeline lives in this file. Sections mirror the",
        "# configuration dataclasses in madmax_calibration.config field-for-field;",
        "# units are SI (metres, Hz, hours) except where a key name carries an",
        "# explicit unit suffix (_ghz, _mm). Unknown keys are rejected at load time.",
        "#",
        "# Disk configurations: one [[disk_configuration]] table per frequency",
        "# window, in any number. spacings_mm are the physical gaps of the 3-disk",
        "# prototype stack: mirror-disk1, disk1-disk2, disk2-disk3.",
        "# booster_antenna_distance_mm is the disk3->antenna optics path (focusing",
        "# mirror in between) - not a stack spacing: it does not enter the 1D boost",
        "# solve and parameterizes the coupling optics. The generated spacings are",
        "# the analytic half-wave stand-in for the offline disk optimization;",
        "# replace them with the real per-window optimized values.",
        "",
        "[calibration]",
        f"objective = {_fmt(config.objective)}                # scan_rate | smooth_min | peak",
        f"n_baseline_replicates = {config.n_baseline_replicates}",
        f"seed = {config.seed}",
        f'active_configuration = "{disk_configurations[0].name}"',
        "",
    ]
    lines += _section_lines(
        "simulator", config.simulator,
        skip={"target_frequency", "window_half_width", "n_disks"},
    )[:-1]
    lines += [
        "# n_disks, target_frequency and window_half_width come from the active",
        "# [[disk_configuration]] below.",
        "",
    ]
    for name in ("control", "antenna", "budget", "cost", "trust_region",
                 "step1", "step5", "step7"):
        lines += _section_lines(name, getattr(config, name))

    lines += [
        "# Simulated-detector ground truth and instrument noise, used by the",
        "# example, the benchmark harness and the tests. Ignored on real hardware.",
        "[mock.truth]",
        f"z_offset = {_fmt(mock_truth.theta.z_offset)}",
        f"compression = {_fmt(mock_truth.theta.compression)}",
        f"log_loss = {_fmt(mock_truth.theta.log_loss)}",
        f"beam_center = [{mock_truth.beam_center[0]:g}, {mock_truth.beam_center[1]:g}]",
        f"focus_optimum = {_fmt(mock_truth.focus_optimum)}",
        f"discrepancy_tilt = {_fmt(mock_truth.discrepancy_tilt)}",
        f"drift_rate_z = {_fmt(mock_truth.drift_rate_z)}",
        f"refl_calibration_bias = {_fmt(mock_truth.refl_calibration_bias)}",
        f"gd_delay_offset = {_fmt(mock_truth.gd_delay_offset)}",
        "",
    ]
    lines += _section_lines("mock.noise", mock_noise)

    for d in disk_configurations:
        lines += [
            "[[disk_configuration]]",
            f'name = "{d.name}"',
            f"target_frequency_ghz = {d.target_frequency / 1e9:g}",
            f"window_half_width_ghz = {d.window_half_width / 1e9:g}",
            "spacings_mm = ["
            + ", ".join(f"{s * 1e3:.4f}" for s in d.spacings)
            + "]   # mirror-d1, d1-d2, d2-d3",
            f"booster_antenna_distance_mm = {d.booster_antenna_distance * 1e3:g}",
        ]
        if d.disk_thickness is not None:
            lines.append(f"disk_thickness_mm = {d.disk_thickness * 1e3:.4f}")
        lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path

"""Settings-file loading, validation and generation (roadmap Phase 0.1)."""

import numpy as np
import pytest

from madmax_calibration.loop import build_loop_from_settings
from madmax_calibration.settings import (
    default_settings_path,
    generate_disk_configurations,
    load_settings,
    write_settings_file,
)

MINIMAL = """
[calibration]
objective = "smooth_min"
n_baseline_replicates = 2
active_configuration = "low"

[budget]
max_hf_measurements = 7

[step5]
discrepancy_lengthscale_bounds = [0.2, 1.5]

[mock.truth]
z_offset = 5e-4
beam_center = [1e-3, -1e-3]

[mock.noise]
lf_refl_noise = 0.008

[[disk_configuration]]
name = "low"
target_frequency_ghz = 19.0
window_half_width_ghz = 0.3
spacings_mm = [7.9, 7.9, 7.9]
booster_antenna_distance_mm = 120.0

[[disk_configuration]]
name = "high"
target_frequency_ghz = 23.0
spacings_mm = [6.5, 6.5, 6.5]
booster_antenna_distance_mm = 120.0
disk_thickness_mm = 1.30
"""


@pytest.fixture
def settings_file(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(MINIMAL)
    return path


def test_load_settings_applies_all_sections(settings_file):
    s = load_settings(settings_file)
    assert s.config.objective == "smooth_min"
    assert s.config.n_baseline_replicates == 2
    assert s.config.budget.max_hf_measurements == 7
    assert s.config.step5.discrepancy_lengthscale_bounds == (0.2, 1.5)
    assert s.mock_truth.theta.z_offset == 5e-4
    assert np.allclose(s.mock_truth.beam_center, [1e-3, -1e-3])
    assert s.mock_noise.lf_refl_noise == 0.008
    assert s.window_names == ["low", "high"]
    assert s.active == "low"


def test_disk_configuration_units_and_fields(settings_file):
    s = load_settings(settings_file)
    low = s.disk_configuration("low")
    assert low.n_disks == 3
    assert np.allclose(low.spacings, 7.9e-3)      # mm -> m
    assert low.target_frequency == 19.0e9         # GHz -> Hz
    assert low.booster_antenna_distance == 0.120
    high = s.disk_configuration("high")
    assert high.disk_thickness == pytest.approx(1.30e-3)
    # thickness default: half-wave in the dielectric at the window centre
    assert low.thickness(disk_index=5.0) == pytest.approx(
        299792458.0 / 19e9 / (2 * 5.0)
    )


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(MINIMAL.replace("max_hf_measurements", "max_hf_measurments"))
    with pytest.raises(ValueError, match="max_hf_measurments"):
        load_settings(bad)

    bad2 = tmp_path / "bad2.toml"
    bad2.write_text(MINIMAL + "\n[nonsense]\nx = 1\n")
    with pytest.raises(ValueError, match="nonsense"):
        load_settings(bad2)


def test_missing_disk_configuration_rejected(tmp_path):
    path = tmp_path / "empty.toml"
    path.write_text("[calibration]\nseed = 1\n")
    with pytest.raises(ValueError, match="disk_configuration"):
        load_settings(path)


def test_unknown_active_window_rejected(settings_file):
    with pytest.raises(KeyError):
        load_settings(settings_file, active="window_99")


def test_spacings_become_nominal_geometry(settings_file):
    """The configured spacings ARE the nominal gaps of the control map."""
    s = load_settings(settings_file, active="high")
    s.config.simulator.n_freq = 21  # keep the loop build cheap
    loop = build_loop_from_settings(s, seed=1)
    assert np.allclose(loop.control_map.nominal_gaps, 6.5e-3)
    assert loop.config.simulator.target_frequency == 23.0e9
    assert loop.config.simulator.n_disks == 3
    assert loop.simulator.booster_antenna_distance == 0.120
    # booster-antenna distance is not a stack spacing: only 3 gaps exist.
    assert len(loop.control_map.nominal_gaps) == 3


def test_generate_and_reload_round_trip(tmp_path):
    """The generator writes a file the loader accepts, with any window count."""
    for n_windows in (1, 5):
        path = write_settings_file(
            tmp_path / f"gen_{n_windows}.toml",
            f_min=20e9, f_max=22e9, n_windows=n_windows, n_disks=3,
        )
        s = load_settings(path)
        assert len(s.disk_configurations) == n_windows
        centres = [d.target_frequency for d in s.disk_configurations]
        assert min(centres) > 20e9 and max(centres) < 22e9
        assert all(d.n_disks == 3 for d in s.disk_configurations)


def test_generated_windows_cover_range_evenly():
    configs = generate_disk_configurations(18e9, 24e9, 12, n_disks=3)
    assert len(configs) == 12
    centres = np.array([d.target_frequency for d in configs])
    assert centres[0] == pytest.approx(18.25e9)
    assert centres[-1] == pytest.approx(23.75e9)
    assert np.allclose(np.diff(centres), 0.5e9)
    # adjacent windows tile the range: half-width = half the spacing
    assert all(d.window_half_width == pytest.approx(0.25e9) for d in configs)


def test_repository_settings_file_loads():
    """The shipped settings/prototype.toml is valid and complete."""
    path = default_settings_path()
    assert path is not None, "settings/prototype.toml missing from repository"
    s = load_settings(path)
    assert len(s.disk_configurations) >= 1
    assert all(d.n_disks == 3 for d in s.disk_configurations)
    assert all(np.all(d.spacings > 0) for d in s.disk_configurations)

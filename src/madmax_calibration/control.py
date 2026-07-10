"""Control variables and the control-to-geometry map.

Implements the booster-state control vector (parent proposal, section 4.1)

    u_B = (a_disk, z_global, z_mirror, z_focus),

the low-dimensional disk-correction basis ``q_disk = q0_disk + B a_disk``
and the normalized internal representation ``x_B in [0,1]^d`` with the
bijective map ``u_B = T(x_B)`` (Step 1 design, section 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ControlConfig, SimulatorConfig


@dataclass
class BoosterControl:
    """Physical booster-state correction u_B."""

    a_disk: np.ndarray       # disk-correction mode amplitudes [m]
    z_global: float          # global stack translation [m]
    z_mirror: float          # reflecting-mirror correction [m]
    z_focus: float           # focusing-mirror correction [m]

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [np.atleast_1d(self.a_disk), [self.z_global, self.z_mirror, self.z_focus]]
        )

    @staticmethod
    def from_vector(v: np.ndarray, n_disk_modes: int) -> "BoosterControl":
        v = np.asarray(v, dtype=float)
        return BoosterControl(
            a_disk=v[:n_disk_modes].copy(),
            z_global=float(v[n_disk_modes]),
            z_mirror=float(v[n_disk_modes + 1]),
            z_focus=float(v[n_disk_modes + 2]),
        )

    @staticmethod
    def zero(n_disk_modes: int) -> "BoosterControl":
        return BoosterControl(np.zeros(n_disk_modes), 0.0, 0.0, 0.0)


def disk_mode_basis(n_disks: int, n_modes: int) -> np.ndarray:
    """The disk-correction basis B: (n_disks x n_modes).

    Column j gives the displacement of each disk (along z, away from the
    mirror) per unit mode amplitude:

      mode 0: disk i moves by (i+1)  -> uniform inter-disk gap change
      mode 1: disk i moves by (i+1)^2 / n -> linear gradient in gaps
      mode 2: disk i moves by (i+1)^3 / n^2 -> quadratic gap profile

    Modes are scaled so a unit amplitude produces order-unity gap changes.
    """
    i = np.arange(1, n_disks + 1, dtype=float)
    cols = []
    for m in range(n_modes):
        col = i ** (m + 1) / (n_disks ** m)
        cols.append(col / col[0])
    return np.stack(cols, axis=1)


@dataclass
class Geometry:
    """Physical 1D booster geometry: gaps and disk thicknesses.

    ``gaps[0]`` is the mirror-to-first-disk vacuum gap; ``gaps[i]`` for
    ``i >= 1`` are the inter-disk gaps.  ``z_focus`` is carried through to
    the antenna-coupling model (it does not enter the 1D stack).
    """

    gaps: np.ndarray
    thicknesses: np.ndarray
    z_focus: float = 0.0


class ControlMap:
    """Maps a control correction u_B onto a physical geometry.

    q_B = q_{0,B} + Delta q_B(u_B)  (Step 1 design, section 4.1),
    with the normalized representation T: [0,1]^d -> physical u_B.
    """

    def __init__(
        self,
        control_cfg: ControlConfig,
        sim_cfg: SimulatorConfig,
        nominal_gaps: np.ndarray,
        nominal_thicknesses: np.ndarray,
    ):
        self.cfg = control_cfg
        self.sim_cfg = sim_cfg
        self.nominal_gaps = np.asarray(nominal_gaps, dtype=float)
        self.nominal_thicknesses = np.asarray(nominal_thicknesses, dtype=float)
        self.basis = disk_mode_basis(sim_cfg.n_disks, control_cfg.n_disk_modes)
        self._limits = control_cfg.limits()

    @property
    def dim(self) -> int:
        return self.cfg.dim

    # ---- normalized <-> physical --------------------------------------

    def to_physical(self, x: np.ndarray) -> np.ndarray:
        """u_B = T(x), x in [0,1]^d."""
        x = np.asarray(x, dtype=float)
        return (2.0 * x - 1.0) * self._limits

    def to_normalized(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return 0.5 * (u / self._limits + 1.0)

    # ---- control -> geometry -------------------------------------------

    def displacements(self, u_B: np.ndarray) -> np.ndarray:
        """Per-disk z displacements produced by u_B (positive = away from
        mirror). Includes disk modes and the global stack translation."""
        u_B = np.asarray(u_B, dtype=float)
        a = u_B[: self.cfg.n_disk_modes]
        z_global = u_B[self.cfg.n_disk_modes]
        return self.basis @ a + z_global

    def geometry(
        self,
        u_B: np.ndarray,
        extra_disk_displacement: np.ndarray | None = None,
        extra_mirror_shift: float = 0.0,
    ) -> Geometry:
        """Physical geometry for correction u_B.

        ``extra_disk_displacement`` / ``extra_mirror_shift`` inject
        detector-state errors theta (used by the simulator and the mock
        hardware); the control map itself is error-free.
        """
        u_B = np.asarray(u_B, dtype=float)
        z_mirror = u_B[self.cfg.n_disk_modes + 1]
        z_focus = u_B[self.cfg.n_disk_modes + 2]

        disp = self.displacements(u_B)
        if extra_disk_displacement is not None:
            disp = disp + extra_disk_displacement

        # Mirror motion (+z_mirror moves the mirror toward the stack,
        # shrinking gap 0); disk displacement changes consecutive gaps.
        gaps = self.nominal_gaps.copy()
        mirror_shift = z_mirror + extra_mirror_shift
        gaps[0] += disp[0] - mirror_shift
        for i in range(1, len(gaps)):
            gaps[i] += disp[i] - disp[i - 1]
        return Geometry(gaps=gaps, thicknesses=self.nominal_thicknesses.copy(), z_focus=z_focus)

    # ---- hard feasibility ----------------------------------------------

    def within_travel_limits(self, u_B: np.ndarray) -> bool:
        return bool(np.all(np.abs(np.asarray(u_B)) <= self._limits + 1e-15))

    def min_gap(self, u_B: np.ndarray) -> float:
        return float(np.min(self.geometry(u_B).gaps))

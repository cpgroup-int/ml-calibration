"""Fast 1D physics simulator for the boosted axion-induced field.

Implements ``beta^2_sim(nu; q, theta)`` (parent proposal, section 2.3) as a
one-dimensional transfer-matrix model of a dielectric haloscope: a perfect
mirror followed by ``N`` dielectric disks separated by vacuum gaps, with an
axion-induced source field ``E_a = -E0 / n^2`` in each region.  The boost
factor is the squared amplitude of the outgoing wave in the semi-infinite
vacuum region on the antenna side, for ``E0 = 1``.

This is a stand-in with the same qualitative behaviour as the real MADMAX
simulation (resonant boost, strong sensitivity to disk spacing, peak/
bandwidth area-law trade-off, loss-induced amplitude reduction).  The rest
of the calibration package only uses it through :class:`BoostSimulator`, so
the real MADMAX code can be substituted behind the same interface.

Detector-state parameters theta (parent proposal, section 5):

- ``z_offset``      global stack z-offset error [m]        (correctable)
- ``compression``   uniform inter-disk gap error [m]       (correctable)
- ``log_loss``      log of the dielectric-loss scale factor (diagnostic-only)

The antenna/receiver coupling is modelled as a separable efficiency
``eta(u_A, z_focus)`` (Gaussian beam overlap x focus factor); the simulator
assumes the beam is centred at the origin and focus optimum at z_focus = 0.
Real-detector offsets of these optima live in the mock hardware and show up
as discrepancy — which is exactly the confounding structure the Step-5
model is designed to handle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import C_LIGHT, SimulatorConfig
from .control import ControlMap, Geometry


@dataclass
class DetectorState:
    """Detector-state / nuisance parameters theta."""

    z_offset: float = 0.0       # global stack offset error [m]
    compression: float = 0.0    # uniform inter-disk gap error [m]
    log_loss: float = 0.0       # log loss-scale (0 -> nominal loss)

    def to_vector(self) -> np.ndarray:
        return np.array([self.z_offset, self.compression, self.log_loss])

    @staticmethod
    def from_vector(v: np.ndarray) -> "DetectorState":
        return DetectorState(float(v[0]), float(v[1]), float(v[2]))

    # Names + correctability labels (Step 5 design, section 8).
    NAMES = ("z_offset", "compression", "log_loss")
    CORRECTABLE = {"z_offset": True, "compression": True, "log_loss": False}


def _beta2_curves(
    freqs: np.ndarray,
    gaps: np.ndarray,
    thicknesses: np.ndarray,
    n_disk: complex,
) -> np.ndarray:
    """Boost-factor curve |E_out/E0|^2 via a 1D transfer-matrix solve.

    Regions (left to right): mirror | gap_0 | disk_1 | gap_1 | ... |
    disk_N | semi-infinite vacuum.  In region r the field is
    ``A_r e^{ik_r x} + B_r e^{-ik_r x} + S_r`` with axion-induced
    ``S_r = -1/n_r^2``.  Boundary conditions: E = 0 at the mirror,
    continuity of E and dE/dx at interfaces, outgoing-only radiation in
    the last region.  Solved as one banded linear system per frequency
    (vectorized over the frequency grid).
    """
    n_disks = len(thicknesses)
    # Region refractive indices and lengths: gap0, d1, g1, d2, ..., dN, vac.
    indices = [1.0 + 0j]
    lengths = [gaps[0]]
    for i in range(n_disks):
        indices.append(n_disk)
        lengths.append(thicknesses[i])
        if i + 1 < len(gaps):
            indices.append(1.0 + 0j)
            lengths.append(gaps[i + 1])
    indices.append(1.0 + 0j)   # semi-infinite vacuum
    lengths.append(0.0)

    R = len(indices)
    n_r = np.array(indices)                       # (R,)
    d_r = np.array(lengths)                       # (R,)
    S_r = -1.0 / n_r**2                           # axion-induced field

    F = len(freqs)
    k = 2.0 * np.pi * freqs[:, None] * n_r[None, :] / C_LIGHT   # (F, R)
    phase = np.exp(1j * k[:, :-1] * d_r[None, :-1])             # (F, R-1)

    # Unknowns per frequency: A_0..A_{R-1}, B_0..B_{R-2}  (B_{R-1} = 0).
    n_unk = 2 * R - 1
    M = np.zeros((F, n_unk, n_unk), dtype=complex)
    rhs = np.zeros((F, n_unk), dtype=complex)

    def a_idx(r: int) -> int:
        return r

    def b_idx(r: int) -> int:
        return R + r  # only r <= R-2 exists

    row = 0
    # Mirror: A_0 + B_0 = -S_0
    M[:, row, a_idx(0)] = 1.0
    M[:, row, b_idx(0)] = 1.0
    rhs[:, row] = -S_r[0]
    row += 1

    for r in range(R - 1):
        pr = phase[:, r]
        ipr = 1.0 / pr
        # E continuity: A_r p + B_r /p - A_{r+1} - B_{r+1} = S_{r+1} - S_r
        M[:, row, a_idx(r)] = pr
        M[:, row, b_idx(r)] = ipr
        M[:, row, a_idx(r + 1)] = -1.0
        if r + 1 <= R - 2:
            M[:, row, b_idx(r + 1)] = -1.0
        rhs[:, row] = S_r[r + 1] - S_r[r]
        row += 1
        # dE/dx continuity: n_r (A_r p - B_r /p) = n_{r+1} (A_{r+1} - B_{r+1})
        M[:, row, a_idx(r)] = n_r[r] * pr
        M[:, row, b_idx(r)] = -n_r[r] * ipr
        M[:, row, a_idx(r + 1)] = -n_r[r + 1]
        if r + 1 <= R - 2:
            M[:, row, b_idx(r + 1)] = n_r[r + 1]
        row += 1

    sol = np.linalg.solve(M, rhs[..., None])[..., 0]      # (F, n_unk)
    a_out = sol[:, a_idx(R - 1)]
    return np.abs(a_out) ** 2


@dataclass
class BoostSimulator:
    """Fast simulator: geometry + theta -> boost curve and coupling."""

    cfg: SimulatorConfig
    control_map: ControlMap
    # Antenna-coupling model parameters (assumed known to the simulator).
    beam_width: float = 4e-3         # Gaussian beam width [m]
    focus_curvature: float = 2e5     # coupling loss per m^2 of focus offset
    # Nominal disk3 -> antenna optics path (focusing mirror in between).
    # Not a stack spacing: the 1D solve treats the region behind the last
    # disk with a radiation condition, so this does not change beta^2; it
    # anchors the coupling optics and is carried to the hardware layer.
    booster_antenna_distance: float = 0.10

    def __post_init__(self) -> None:
        self._freqs = self.cfg.frequency_grid()

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs

    # ---- geometry with detector-state errors ---------------------------

    def geometry_with_state(self, u_B: np.ndarray, theta: DetectorState) -> Geometry:
        n_disks = self.cfg.n_disks
        # z_offset moves the whole stack; compression adds a uniform
        # inter-disk gap error (disk i displaced by i * compression).
        extra = theta.z_offset + theta.compression * np.arange(n_disks, dtype=float)
        return self.control_map.geometry(u_B, extra_disk_displacement=extra)

    # ---- boost curve ----------------------------------------------------

    def beta2(self, u_B: np.ndarray, theta: DetectorState) -> np.ndarray:
        geom = self.geometry_with_state(u_B, theta)
        loss = self.cfg.disk_loss_tan * float(np.exp(theta.log_loss))
        n_disk = self.cfg.disk_index * (1.0 - 0.5j * loss)
        return _beta2_curves(self._freqs, geom.gaps, geom.thicknesses, n_disk)

    # ---- antenna/receiver coupling --------------------------------------

    def coupling(
        self,
        u_A: np.ndarray,
        z_focus: float,
        beam_center: np.ndarray | None = None,
        focus_optimum: float = 0.0,
    ) -> float:
        """Coupling efficiency eta in (0, 1]."""
        u_A = np.asarray(u_A, dtype=float)
        center = np.zeros(2) if beam_center is None else np.asarray(beam_center)
        r2 = float(np.sum((u_A - center) ** 2))
        beam = np.exp(-r2 / (2.0 * self.beam_width**2))
        focus = 1.0 / (1.0 + self.focus_curvature * (z_focus - focus_optimum) ** 2)
        return float(beam * focus)

    def aligned_coupling(self, u_B: np.ndarray, theta: DetectorState) -> float:
        """Coupling after ideal Step-3 antenna alignment (beam term = 1)."""
        z_focus = float(np.asarray(u_B)[-1])
        return self.coupling(np.zeros(2), z_focus)

    # ---- measured-curve prediction --------------------------------------

    def beta2_measured(self, u_B: np.ndarray, u_A: np.ndarray, theta: DetectorState) -> np.ndarray:
        """Predicted measured curve: stack boost x antenna coupling."""
        z_focus = float(np.asarray(u_B)[-1])
        return self.beta2(u_B, theta) * self.coupling(u_A, z_focus)

    def predict_J(self, u_B: np.ndarray, theta: DetectorState, objective) -> float:
        """Antenna-aligned scalar objective J_sim(u_B, theta).

        Assumes Step 3 will centre the antenna on the beam, so only the
        focus factor of the coupling remains (Step 6 design, section 5:
        plug-in antenna optimum).
        """
        curve = self.beta2(u_B, theta) * self.aligned_coupling(u_B, theta)
        return float(objective(curve))

    def predict_summaries(self, u_B: np.ndarray, theta: DetectorState, summarizer) -> np.ndarray:
        """Antenna-aligned curve-summary vector z_sim(u_B, theta).

        Same plug-in-antenna convention as :meth:`predict_J`; component 0
        of the returned vector is the scalar objective.
        """
        curve = self.beta2(u_B, theta) * self.aligned_coupling(u_B, theta)
        return summarizer(curve)


def nominal_half_wave_geometry(cfg: SimulatorConfig) -> tuple[np.ndarray, np.ndarray]:
    """Analytic transparent-mode seed: half-wave gaps and disks.

    Stands in for the existing offline MADMAX disk-spacing optimization
    that produces q0(W) (parent proposal, section 2.1).  Use
    :func:`optimize_nominal_gaps` to polish it on the simulator.
    """
    lam = cfg.wavelength
    gaps = np.full(cfg.n_disks, lam / 2.0)
    thicknesses = np.full(cfg.n_disks, lam / (2.0 * cfg.disk_index))
    return gaps, thicknesses


def optimize_nominal_gaps(
    cfg: SimulatorConfig,
    objective,
    maxiter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Crude offline optimization of the nominal gaps (Nelder-Mead).

    ``objective`` maps a beta^2 curve on ``cfg.frequency_grid()`` to a
    scalar to maximize.
    """
    from scipy.optimize import minimize

    gaps0, thick = nominal_half_wave_geometry(cfg)
    freqs = cfg.frequency_grid()
    n_disk = cfg.disk_index * (1.0 - 0.5j * cfg.disk_loss_tan)

    def neg_j(gaps: np.ndarray) -> float:
        if np.any(gaps < 1e-4):
            return 1e6
        curve = _beta2_curves(freqs, gaps, thick, n_disk)
        return -float(objective(curve))

    res = minimize(neg_j, gaps0, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-3})
    return np.asarray(res.x), thick

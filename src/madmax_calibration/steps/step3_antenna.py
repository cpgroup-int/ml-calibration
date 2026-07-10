"""Step 3: align the antenna for a fixed booster state.

Implements the hybrid local alignment strategy of the Step 3 design note
(section 7):

    incumbent validation -> local plus-pattern scan -> quadratic fit ->
    confirmation -> 2D GP Bayesian-optimization fallback.

The alignment score is the cheap coupling proxy provided by the hardware
(section 6.2); hard antenna travel limits are enforced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AntennaConfig
from ..gp import GaussianProcess
from ..hardware import HardwareInterface


@dataclass
class Step3Result:
    u_A_cmd: np.ndarray
    u_A_achieved: np.ndarray
    score: float
    score_sigma: float
    method: str                 # reused_incumbent | local_fit_confirmed | gp_bo_confirmed | budget_limited | ...
    quality_flag: str
    n_evaluations: int
    data: list = field(default_factory=list)   # (u_A_achieved, score, sigma) tuples


def _measure_at(hardware: HardwareInterface, u_A: np.ndarray, data: list) -> tuple[float, float, np.ndarray]:
    achieved = hardware.move_antenna(u_A)
    val, sig = hardware.measure_alignment_proxy()
    data.append((achieved.copy(), val, sig))
    return val, sig, achieved


def _clip_to_domain(u_A: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(u_A, -limit, limit)


def _quadratic_fit(points: np.ndarray, values: np.ndarray):
    """Least-squares quadratic surface; returns (optimum, curvature ok)."""
    x, y = points[:, 0], points[:, 1]
    A = np.stack([np.ones_like(x), x, y, x * y, x**2, y**2], axis=1)
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    _, gx, gy, cxy, cxx, cyy = coef
    H = np.array([[2 * cxx, cxy], [cxy, 2 * cyy]])
    # Maximum requires negative-definite Hessian.
    eigvals = np.linalg.eigvalsh(H)
    if np.any(eigvals >= 0):
        return None, False
    opt = np.linalg.solve(H, -np.array([gx, gy]))
    return opt, True


def run_step3(
    hardware: HardwareInterface,
    cfg: AntennaConfig,
    start: np.ndarray | None = None,
    expected_score: float | None = None,
    rng: np.random.Generator | None = None,
) -> Step3Result:
    """Align the antenna for the current (fixed) booster state."""
    rng = rng or np.random.default_rng(0)
    data: list = []
    limit = cfg.travel_limit
    u0 = _clip_to_domain(np.zeros(2) if start is None else np.asarray(start, dtype=float), limit)

    # Stage 2: incumbent validation (design section 9).
    val0, sig0, ach0 = _measure_at(hardware, u0, data)
    if expected_score is not None and val0 >= expected_score - cfg.kappa * sig0:
        return Step3Result(u0, ach0, val0, sig0, "reused_incumbent", "reused_incumbent", len(data), data)

    # Stage 3: local plus-pattern scan + quadratic fit (section 10).
    step = cfg.initial_scan_step
    offsets = np.array(
        [[step, 0], [-step, 0], [0, step], [0, -step], [step, step], [-step, -step], [step, -step], [-step, step]]
    )
    pts = [ach0]
    vals = [val0]
    for off in offsets:
        v, s, a = _measure_at(hardware, _clip_to_domain(u0 + off, limit), data)
        pts.append(a)
        vals.append(v)
    pts_arr = np.stack(pts)
    vals_arr = np.array(vals)

    opt, fit_ok = _quadratic_fit(pts_arr, vals_arr)
    best_idx = int(np.argmax(vals_arr))
    incumbent_pos, incumbent_val = pts_arr[best_idx], float(vals_arr[best_idx])
    sigma_typ = float(np.median([d[2] for d in data]))

    if fit_ok and opt is not None:
        within_reach = np.all(np.abs(opt - u0) <= 3.0 * step)  # not too far outside the scan
        inside = np.all(np.abs(opt) <= limit)
        if within_reach and inside:
            v_opt, s_opt, a_opt = _measure_at(hardware, opt, data)
            # Confirmation: fitted optimum must not be worse than the best
            # scanned point by more than the noise (section 10).
            if v_opt >= incumbent_val - cfg.kappa * max(s_opt, sigma_typ):
                if v_opt >= incumbent_val:
                    return Step3Result(opt, a_opt, v_opt, s_opt, "local_fit_confirmed", "local_fit_confirmed", len(data), data)
                # fall through with the scanned best as incumbent
                incumbent_pos, incumbent_val = (a_opt, v_opt) if v_opt > incumbent_val else (incumbent_pos, incumbent_val)

    # Stage 4: 2D GP-BO fallback (section 11), noise-aware UCB acquisition.
    while len(data) < cfg.max_evaluations:
        x = np.stack([d[0] for d in data]) / limit
        y = np.array([d[1] for d in data])
        noise = np.array([d[2] for d in data])
        y_mean = float(np.mean(y))
        amp = max(float(np.std(y)), 1e-3)
        gp = GaussianProcess(amplitude=amp, lengthscales=np.array([0.3, 0.3]))
        gp.fit(x, y - y_mean, noise)
        cand = rng.uniform(-1.0, 1.0, size=(256, 2))
        mean, sd = gp.predict(cand)
        ucb = mean + y_mean + 2.0 * sd
        u_next = cand[int(np.argmax(ucb))] * limit
        v, s, a = _measure_at(hardware, u_next, data)
        if v > incumbent_val:
            incumbent_pos, incumbent_val = a, v
        # Stop when predicted improvement is below the noise floor.
        if float(np.max(ucb) - incumbent_val) < max(s, sigma_typ):
            break

    method = "gp_bo_confirmed" if len(data) < cfg.max_evaluations else "budget_limited"
    # Return to the best found position so downstream measurement happens there.
    ach = hardware.move_antenna(incumbent_pos)
    return Step3Result(incumbent_pos, ach, incumbent_val, sigma_typ, method, method, len(data), data)

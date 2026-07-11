"""A/B benchmark harness for the calibration pipeline (roadmap Phase 0.2).

Runs the full closed loop against the simulated detector over an
ensemble of seeds (and optionally jittered mock truths) and reports the
roadmap's agreed metrics per settings file:

- HF measurements (and hours) to reach within measurement noise of the
  achievable optimum,
- final validated improvement and its significance,
- posterior calibration of the Step-6 predictions (2-sigma coverage,
  RMS standardized residual),
- safety and budget compliance (must always be 100%).

Every algorithmic change on the roadmap is judged with this harness:

    python -m madmax_calibration.benchmark settings/prototype.toml \
        [other_settings.toml ...] --runs 5 --iterations 15

The achievable optimum is computed from the mock ground truth (the
analytic cancellation of the correctable errors plus the true focus
optimum), which is exactly what a *simulation-only* benchmark may use.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .constraints import HardConstraints
from .loop import CalibrationLoop, build_loop_from_settings
from .objectives import Objective
from .settings import Settings, load_settings
from .simulator import DetectorState


@dataclass
class RunResult:
    """Metrics of one benchmark run (one seed, one window)."""

    label: str
    window: str
    seed: int
    J_baseline: float
    J_best: float
    sigma_J_best: float
    improvement: float
    improvement_over_noise: float
    significant: bool
    J_achievable: float
    gap_to_achievable: float          # J_achievable - J_best
    hf_to_within_noise: int | None    # HF count when J first within 2 sigma
    n_hf: int
    n_lf: int
    hours: float
    coverage_2sigma: float | None
    rms_standardized_residual: float | None
    theta_z_error: float | None       # |estimate - truth| of the stack offset [m]
    theta_z_sd: float | None          # its posterior sd [m]
    theta_c_error: float | None       # |estimate - truth| of the compression [m]
    theta_c_sd: float | None
    safety_ok: bool
    budget_ok: bool
    stop_reason: str


@dataclass
class BenchmarkSummary:
    label: str
    runs: list[RunResult] = field(default_factory=list)

    def _values(self, name: str) -> np.ndarray:
        vals = [getattr(r, name) for r in self.runs]
        return np.array([v for v in vals if v is not None], dtype=float)

    def row(self) -> dict:
        imp = self._values("improvement")
        sig = self._values("significant")
        gap = self._values("gap_to_achievable")
        hf = self._values("n_hf")
        hf_conv = self._values("hf_to_within_noise")
        hours = self._values("hours")
        cov = self._values("coverage_2sigma")
        tz_err = self._values("theta_z_error")
        tz_sd = self._values("theta_z_sd")
        tc_err = self._values("theta_c_error")
        tc_sd = self._values("theta_c_sd")
        return {
            "label": self.label,
            "runs": len(self.runs),
            "improvement": f"{imp.mean():.3f} ± {imp.std(ddof=1):.3f}" if len(imp) > 1 else f"{imp.mean():.3f}",
            "significant": f"{int(sig.sum())}/{len(sig)}",
            "gap_to_achievable": f"{gap.mean():.3f}",
            "hf_to_converge": (f"{hf_conv.mean():.1f}" if len(hf_conv) else "n/a")
            + f" (of {hf.mean():.1f})",
            "hours": f"{hours.mean():.1f}",
            "coverage_2sigma": f"{cov.mean():.2f}" if len(cov) else "n/a",
            "theta_z_err_um": f"{tz_err.mean() * 1e6:.0f} (sd {tz_sd.mean() * 1e6:.0f})" if len(tz_err) else "n/a",
            "theta_c_err_um": f"{tc_err.mean() * 1e6:.0f} (sd {tc_sd.mean() * 1e6:.0f})" if len(tc_err) else "n/a",
            "safety": f"{sum(r.safety_ok for r in self.runs)}/{len(self.runs)}",
            "budget": f"{sum(r.budget_ok for r in self.runs)}/{len(self.runs)}",
        }


def achievable_objective(loop: CalibrationLoop) -> float:
    """Best objective reachable by the control basis, from the mock truth.

    Cancels the correctable errors analytically (a0 = -c, z_g = -z + c),
    sets the focusing mirror to the true optimum, assumes a perfectly
    aligned antenna, and includes the systematic curve tilt — i.e. the
    ideal end state of a successful calibration at t = 0 (drift-free).
    """
    truth = loop.hardware.truth
    theta = truth.theta
    cm = loop.control_map
    n = loop.config.control.n_disk_modes
    limits = cm.cfg.limits()

    u_fix = np.zeros(cm.dim)
    u_fix[0] = -theta.compression
    u_fix[n] = -theta.z_offset + theta.compression
    u_fix[n + 2] = truth.focus_optimum
    u_fix = np.clip(u_fix, -limits, limits)

    sim = loop.simulator
    beta2 = sim.beta2(u_fix, theta)
    eta = sim.coupling(
        truth.beam_center, float(u_fix[-1]),
        beam_center=truth.beam_center, focus_optimum=truth.focus_optimum,
    )
    xi = np.linspace(-1.0, 1.0, len(beta2))
    curve = beta2 * eta * (1.0 + truth.discrepancy_tilt * xi)
    return float(Objective(loop.config.objective)(curve))


def run_one(
    settings: Settings,
    label: str,
    seed: int,
    window: str | None = None,
    max_iterations: int = 15,
    truth_jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> RunResult:
    """One full closed-loop run plus metric extraction."""
    settings = copy.deepcopy(settings)
    if truth_jitter > 0.0:
        rng = rng or np.random.default_rng(seed)
        theta = settings.mock_truth.theta
        settings.mock_truth.theta = DetectorState(
            z_offset=theta.z_offset * (1 + truth_jitter * rng.standard_normal()),
            compression=theta.compression * (1 + truth_jitter * rng.standard_normal()),
            log_loss=theta.log_loss,
        )
    settings.config.seed = seed

    loop = build_loop_from_settings(settings, window=window, seed=seed)
    j_achievable = achievable_objective(loop)
    result = loop.run(max_iterations=max_iterations)
    report = result.feasibility_report

    # HF count (in measurement order) at which J first reaches the
    # achievable optimum within 2 sigma of that measurement.
    hf_records = sorted(result.dataset.hf_records(), key=lambda r: r.time_start)
    hf_to_within = None
    for k, rec in enumerate(hf_records, start=1):
        if rec.J >= j_achievable - 2.0 * rec.sigma_J:
            hf_to_within = k
            break

    hard = HardConstraints(loop.control_map, loop.config.control)
    safety_ok = all(
        hard.feasible(r.u_B_cmd)
        for r in result.dataset.records
        if r.u_B_cmd is not None
    )
    counts = result.dataset.counts()
    b = loop.config.budget
    budget_ok = (
        counts["hf"] <= b.max_hf_measurements
        and counts["lf"] <= b.max_lf_measurements
        and loop.hardware.now <= b.max_total_hours + 1e-9
    )

    # Posterior-calibration and theta-recovery metrics.
    coverage = rms_z = None
    tz_err = tz_sd = tc_err = tc_sd = None
    if result.step5 is not None:
        from .steps.step6_predictive import run_step6

        model = run_step6(
            result.step5, loop.simulator, loop.control_map, result.dataset,
            loop.config, loop.objective,
        )
        coverage = model.validation.get("coverage_2sigma")
        rms_z = model.validation.get("rms_standardized_residual")
        truth = loop.hardware.truth
        est = result.step5.theta_map
        sd = np.sqrt(np.diag(result.step5.theta_cov))
        # Compare against the drift-adjusted truth at the mid-time of the
        # HF data (the stack offset drifts during the run).
        t_mid = float(np.median([r.time for r in result.dataset.hf_records()]))
        truth_z = truth.theta.z_offset + truth.drift_rate_z * t_mid
        tz_err = abs(est.z_offset - truth_z)
        tz_sd = float(sd[0])
        tc_err = abs(est.compression - truth.theta.compression)
        tc_sd = float(sd[1])

    return RunResult(
        label=label,
        window=settings.disk_configuration(window).name,
        seed=seed,
        J_baseline=result.J_baseline,
        J_best=result.J_best,
        sigma_J_best=result.sigma_J_best,
        improvement=report["improvement"],
        improvement_over_noise=report["improvement_over_noise"],
        significant=report["improvement_significant"],
        J_achievable=j_achievable,
        gap_to_achievable=j_achievable - result.J_best,
        hf_to_within_noise=hf_to_within,
        n_hf=counts["hf"],
        n_lf=counts["lf"],
        hours=loop.hardware.now,
        coverage_2sigma=coverage,
        rms_standardized_residual=rms_z,
        theta_z_error=tz_err,
        theta_z_sd=tz_sd,
        theta_c_error=tc_err,
        theta_c_sd=tc_sd,
        safety_ok=safety_ok,
        budget_ok=budget_ok,
        stop_reason=result.stop_reason,
    )


def run_benchmark(
    settings_path: str,
    n_runs: int = 3,
    window: str | None = None,
    max_iterations: int = 15,
    truth_jitter: float = 0.0,
    seed0: int = 0,
    label: str | None = None,
) -> BenchmarkSummary:
    settings = load_settings(settings_path)
    summary = BenchmarkSummary(label=label or settings_path)
    for i in range(n_runs):
        summary.runs.append(
            run_one(
                settings,
                label=summary.label,
                seed=seed0 + i,
                window=window,
                max_iterations=max_iterations,
                truth_jitter=truth_jitter,
            )
        )
    return summary


def format_table(summaries: list[BenchmarkSummary]) -> str:
    rows = [s.row() for s in summaries]
    columns = list(rows[0].keys())
    widths = {
        c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns
    }
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = [
        "  ".join(str(r[c]).ljust(widths[c]) for c in columns) for r in rows
    ]
    return "\n".join([header, sep, *body])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the calibration pipeline for one or more settings files."
    )
    parser.add_argument("settings", nargs="+", help="settings TOML file(s) to compare")
    parser.add_argument("--runs", "-n", type=int, default=3, help="seeds per settings file")
    parser.add_argument("--iterations", type=int, default=15, help="max outer iterations")
    parser.add_argument("--window", default=None, help="disk-configuration name (default: active)")
    parser.add_argument("--truth-jitter", type=float, default=0.0,
                        help="relative jitter of the mock truth per run (e.g. 0.25)")
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args(argv)

    summaries = []
    for path in args.settings:
        print(f"[benchmark] {path}: {args.runs} run(s), window={args.window or 'active'}")
        summaries.append(
            run_benchmark(
                path,
                n_runs=args.runs,
                window=args.window,
                max_iterations=args.iterations,
                truth_jitter=args.truth_jitter,
                seed0=args.seed0,
            )
        )
    print()
    print(format_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

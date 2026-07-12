"""Amortized NPE engine tests (roadmap Phase 2).

Covers the pure-numpy mixture density network (gradient correctness,
sampling, persistence), the residual-projection conditioning, the Step-5
integration (engine selection + fallback), and simulation-based
calibration of the shipped weights.
"""

import numpy as np
import pytest

from madmax_calibration.amortized import (
    AmortizedPosterior,
    MixtureDensityNetwork,
    ResidualProjectionFeaturizer,
    TrainingConfig,
    build_conditioning,
    train_amortized_posterior,
)
from madmax_calibration.settings import default_settings_path, load_settings
from madmax_calibration.steps.step5_inference import _default_weights_path
from tests.conftest import make_setup


# ---------------------------------------------------------------------------
# Mixture density network
# ---------------------------------------------------------------------------

def test_mdn_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    net = MixtureDensityNetwork(input_dim=6, theta_dim=3, n_components=3, hidden=8, seed=1)
    C = rng.standard_normal((5, 6))
    Th = rng.standard_normal((5, 3))
    _, grad = net._nll_and_grad(C, Th)
    eps = 1e-6
    max_err = 0.0
    for pi, gi in zip(net.p.as_list(), grad.as_list()):
        fp, fg = pi.ravel(), gi.ravel()
        for _ in range(15):
            j = rng.integers(len(fp))
            old = fp[j]
            fp[j] = old + eps
            lp, _ = net._nll_and_grad(C, Th)
            fp[j] = old - eps
            lm, _ = net._nll_and_grad(C, Th)
            fp[j] = old
            max_err = max(max_err, abs((lp - lm) / (2 * eps) - fg[j]))
    assert max_err < 1e-5


def test_mdn_learns_conditional_mean():
    """A tiny MDN recovers a linear conditional mean from noisy data."""
    rng = np.random.default_rng(0)
    C = rng.uniform(-1, 1, size=(2000, 2))
    W = np.array([[1.0, -0.5, 0.3], [0.2, 0.8, -0.4]])
    Th = C @ W + 0.05 * rng.standard_normal((2000, 3))
    net = MixtureDensityNetwork(input_dim=2, theta_dim=3, n_components=2, hidden=16, seed=1)
    hist = net.fit(C, Th, epochs=150, lr=5e-3, seed=0)
    assert hist[-1] < hist[0]
    c_test = np.array([0.5, -0.3])
    mean, cov = net.posterior_mean_cov(c_test)
    assert np.allclose(mean, c_test @ W, atol=0.1)
    assert np.all(np.diag(cov) < 0.1)          # tight around the true mean


def test_mdn_sampling_matches_mean_cov():
    net = MixtureDensityNetwork(input_dim=3, theta_dim=3, n_components=3, hidden=8, seed=2)
    c = np.array([0.1, -0.2, 0.3])
    mean, cov = net.posterior_mean_cov(c)
    samples = net.sample(c, 20000, np.random.default_rng(0))
    assert np.allclose(samples.mean(0), mean, atol=0.05)
    assert np.allclose(np.cov(samples.T), cov, atol=0.1)


# ---------------------------------------------------------------------------
# Featurizer + persistence
# ---------------------------------------------------------------------------

def test_featurizer_dimension_and_permutation_invariance():
    feat = ResidualProjectionFeaturizer(
        control_dim=5, hf_component_scale=np.ones(5), lf_component_scale=np.ones(4),
    )
    assert feat.dim == 5 * 6 + 4 * 6 + 3
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 5))
    r = rng.standard_normal((4, 5))
    c1 = feat.from_arrays(x, r, np.zeros((0, 5)), np.zeros((0, 4)))
    perm = rng.permutation(4)
    c2 = feat.from_arrays(x[perm], r[perm], np.zeros((0, 5)), np.zeros((0, 4)))
    assert np.allclose(c1, c2)                  # permutation invariant


def test_posterior_save_load_round_trip(tmp_path):
    config, cm, sim, _ = make_setup()
    post = train_amortized_posterior(
        sim, cm, config, TrainingConfig(n_episodes=300, epochs=20, hidden=16, seed=0),
    )
    path = tmp_path / "npe.npz"
    post.save(path)
    reloaded = AmortizedPosterior.load(path)
    rng = np.random.default_rng(0)
    c = np.zeros(post.featurizer.dim)
    m1, cov1, _ = post.infer(c, rng)
    m2, cov2, _ = reloaded.infer(c, rng)
    assert np.allclose(m1.to_vector(), m2.to_vector())
    assert np.allclose(cov1, cov2)


# ---------------------------------------------------------------------------
# Training recovers the detector state
# ---------------------------------------------------------------------------

def test_training_recovers_theta_direction():
    """A quickly-trained NPE tracks the true detector state better than the
    prior mean on clean data."""
    from madmax_calibration.objectives import Objective
    from madmax_calibration.simulator import DetectorState
    from madmax_calibration.summaries import CurveSummarizer, ReflectivitySummarizer

    config, cm, sim, _ = make_setup()
    post = train_amortized_posterior(
        sim, cm, config,
        TrainingConfig(n_episodes=3000, epochs=150, hidden=48, n_components=4, seed=0),
    )
    obj = Objective(config.objective)
    summ = CurveSummarizer(obj, sim.freqs)
    refl = ReflectivitySummarizer(sim.freqs)
    rng = np.random.default_rng(3)
    prior_sd = post.prior_sd
    limits = cm.cfg.limits()
    errs = []
    for _ in range(30):
        theta = DetectorState.from_vector(np.clip(rng.standard_normal(3), -2, 2) * prior_sd)
        hf_u = np.stack([rng.uniform(-0.5, 0.5, cm.dim) * limits for _ in range(8)])
        hf_z = np.stack([sim.predict_summaries(u, theta, summ) for u in hf_u])
        lf_u = np.stack([rng.uniform(-0.5, 0.5, cm.dim) * limits for _ in range(5)])
        lf_z = np.stack([sim.predict_reflectivity_summaries(u, theta, refl) for u in lf_u])
        c = build_conditioning(post.featurizer, hf_u, hf_z, lf_u, lf_z, cm, sim, obj, summ, refl)
        mean, _, _ = post.infer(c, rng)
        errs.append(np.abs((mean.to_vector() - theta.to_vector()) / prior_sd))
    errs = np.array(errs).mean(0)
    # Standardized error well below 1 (the prior sd) for the correctable pair.
    assert errs[0] < 0.6 and errs[1] < 0.6


# ---------------------------------------------------------------------------
# Step-5 integration and fallback
# ---------------------------------------------------------------------------

def _prototype_setup():
    settings = load_settings(default_settings_path())
    from madmax_calibration.control import ControlMap
    from madmax_calibration.simulator import BoostSimulator

    d = settings.disk_configuration()
    sim_cfg = settings.simulator_config_for(d.name)
    config = settings.config
    config.simulator = sim_cfg
    thick = np.full(d.n_disks, d.thickness(sim_cfg.disk_index))
    cm = ControlMap(config.control, sim_cfg, d.spacings, thick)
    sim = BoostSimulator(sim_cfg, cm, booster_antenna_distance=d.booster_antenna_distance)
    return settings, config, cm, sim


@pytest.mark.skipif(not __import__("pathlib").Path(_default_weights_path()).exists(),
                    reason="shipped NPE weights not present")
def test_step5_uses_npe_engine_with_shipped_weights():
    from madmax_calibration.hardware import MockHardware
    from madmax_calibration.loop import CalibrationLoop

    settings, config, cm, sim = _prototype_setup()
    config.step5.inference_engine = "amortized_npe"
    hw = MockHardware(sim, config, truth=settings.mock_truth, noise=settings.mock_noise, seed=1000)
    loop = CalibrationLoop(hw, sim, config)
    result = loop.run(max_iterations=6)
    assert result.step5.inference_engine == "amortized_npe"
    assert result.step5.sampler is not None
    # Exact mixture sampling is used for theta_samples.
    samples = result.step5.theta_samples(50, np.random.default_rng(0))
    assert len(samples) == 50
    assert result.J_best > result.J_baseline


def test_step5_falls_back_when_weights_missing():
    """A non-existent weights path degrades to joint_map with a diagnostic."""
    from madmax_calibration.objectives import Objective
    from madmax_calibration.simulator import DetectorState
    from madmax_calibration.steps.step5_inference import run_step5
    from madmax_calibration.summaries import CurveSummarizer
    from tests.test_curve_summaries import _make_summary_dataset

    config, cm, sim, _ = make_setup()
    config.step5.inference_engine = "amortized_npe"
    config.step5.npe_weights_path = "/nonexistent/weights.npz"
    summ = CurveSummarizer(Objective(config.objective), sim.freqs)
    ds = _make_summary_dataset(sim, cm, summ, DetectorState(z_offset=0.3e-3), n_points=6)
    res = run_step5(ds, sim, cm, config, Objective(config.objective))
    assert res.inference_engine == "joint_map"
    assert "amortized_npe requested but unavailable" in res.diagnostics.get("warning", "")


# ---------------------------------------------------------------------------
# Simulation-based calibration (Phase 2.2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not __import__("pathlib").Path(_default_weights_path()).exists(),
                    reason="shipped NPE weights not present")
def test_sbc_well_specified_is_calibrated():
    """SBC of the shipped weights: well-specified ranks are ~uniform and
    2-sigma coverage is near 0.95 (roadmap Phase 2.2 acceptance)."""
    from madmax_calibration.sbc import run_sbc

    _, config, cm, sim = _prototype_setup()
    post = AmortizedPosterior.load(_default_weights_path())
    r = run_sbc(post, sim, cm, config, n_trials=250, n_posterior=200,
                discrepancy_injection=0.0, seed=5)
    assert np.all(r.ks_uniform_p > 0.05), f"non-uniform ranks: {r.ks_uniform_p}"
    assert np.all(r.coverage_2sigma > 0.88), f"under-coverage: {r.coverage_2sigma}"


@pytest.mark.skipif(not __import__("pathlib").Path(_default_weights_path()).exists(),
                    reason="shipped NPE weights not present")
def test_sbc_degrades_gracefully_under_misspecification():
    from madmax_calibration.sbc import run_sbc

    _, config, cm, sim = _prototype_setup()
    post = AmortizedPosterior.load(_default_weights_path())
    r = run_sbc(post, sim, cm, config, n_trials=200, n_posterior=150,
                discrepancy_injection=0.06, seed=6)
    # Still broadly calibrated under injected systematics (not collapsing).
    assert np.all(r.coverage_2sigma > 0.82)

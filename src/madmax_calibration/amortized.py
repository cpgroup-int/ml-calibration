"""Amortized simulation-based inference for the detector state (Phase 2.1).

Roadmap Phase 2 replaces the joint-MAP + Laplace estimate of the
detector state ``theta`` with an **amortized neural posterior estimator**
(NPE): a conditional density ``q(theta | c)`` trained offline against the
fast simulator, evaluated online in a millisecond forward pass.  The
density model is a **conditional neural spline flow** built on the
established SBI stack — PyTorch for the network and training loop,
`zuko <https://zuko.readthedocs.io>`_ for the normalizing flow — rather
than anything hand-rolled; trained weights are shipped as a standard
``.pt`` checkpoint.

Why NPE here.  With a cheap, tractable forward model the value of SBI is
*not* to replace the simulator (we keep it) but to produce a
**calibrated, possibly non-Gaussian posterior** that is robust to model
misspecification — the failure mode of the Laplace approximation on the
detector-state degeneracy ridge (overconfident, Gaussian).  Robustness
is trained in by injecting the discrepancy model into the training
simulations (see :func:`train_amortized_posterior`), so the amortized
posterior is appropriately wide under systematics it will meet online.

Conditioning statistic.  The detector state is *global* across a window
while measurements arrive at arbitrary control inputs ``u_i``.  The
amortization is made well-posed with a fixed-length, permutation-
invariant summary of the (variable-size) dataset: the projection of each
measurement's *residual against the nominal simulation* onto its design,
averaged over measurements (a Gauss-Newton-style sufficient direction).
This handles any number of HF and LF measurements and appends cleanly to
window features for the planned cross-window transfer (Phase 5).

The trained posterior feeds Step 5 through
:mod:`madmax_calibration.steps.step5_inference`; the discrepancy GPs are
still fitted online (NPE handles theta, the discrepancy channel keeps
absorbing what theta cannot explain), and
``Step5Result.theta_samples`` becomes exact sampling from the flow
instead of a Gaussian Laplace draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import zuko

from .simulator import DetectorState
from .summaries import (
    REFLECTIVITY_SUMMARY_NAMES,
    SUMMARY_NAMES,
    CurveSummarizer,
    ReflectivitySummarizer,
)


# ---------------------------------------------------------------------------
# Conditioning statistic
# ---------------------------------------------------------------------------

@dataclass
class ResidualProjectionFeaturizer:
    """Fixed-length permutation-invariant summary of a measurement set.

    For each channel and each summary component, the residual of the
    measurement against the *nominal* simulation (``theta = 0``) is
    projected onto ``[1, x]`` (x = normalized control input) and averaged
    over measurements.  This is the local sufficient direction for the
    detector state, is informative regardless of measurement count, and
    is what the network learns to invert into a posterior.
    """

    control_dim: int
    hf_component_scale: np.ndarray     # (n_hf_comp,) response scales
    lf_component_scale: np.ndarray     # (n_lf_comp,) response scales
    hf_names: tuple = SUMMARY_NAMES
    lf_names: tuple = REFLECTIVITY_SUMMARY_NAMES

    @property
    def n_hf_comp(self) -> int:
        return len(self.hf_names)

    @property
    def n_lf_comp(self) -> int:
        return len(self.lf_names)

    @property
    def dim(self) -> int:
        g = 1 + self.control_dim
        return self.n_hf_comp * g + self.n_lf_comp * g + 3

    def _project(self, x: np.ndarray, resid: np.ndarray, n_comp: int) -> np.ndarray:
        """Mean over measurements of resid[:, k] outer [1, x_i]."""
        if len(x) == 0:
            return np.zeros(n_comp * (1 + self.control_dim))
        g = np.concatenate([np.ones((len(x), 1)), x], axis=1)     # (n, 1+d)
        # (n, K, 1) * (n, 1, 1+d) -> mean over n -> (K, 1+d)
        proj = np.einsum("nk,ng->kg", resid, g) / len(x)
        return proj.reshape(-1)

    def from_arrays(
        self,
        x_hf: np.ndarray, resid_hf: np.ndarray,
        x_lf: np.ndarray, resid_lf: np.ndarray,
    ) -> np.ndarray:
        """Build the conditioning vector from pre-computed residuals.

        ``resid`` are (n, K) residuals already divided by the component
        response scales.
        """
        s_hf = self._project(x_hf, resid_hf, self.n_hf_comp)
        s_lf = self._project(x_lf, resid_lf, self.n_lf_comp)
        meta = np.array([
            min(len(x_hf), 20) / 20.0,
            min(len(x_lf), 20) / 20.0,
            1.0 if len(x_lf) > 0 else 0.0,
        ])
        return np.concatenate([s_hf, s_lf, meta])


# ---------------------------------------------------------------------------
# Conditional normalizing flow (PyTorch + zuko)
# ---------------------------------------------------------------------------

class ConditionalFlow(torch.nn.Module):
    """Conditional posterior q(theta_std | c) as a neural spline flow.

    A thin wrapper around :class:`zuko.flows.NSF` (masked autoregressive
    rational-quadratic spline flow) that adds context standardization,
    a plain minibatch training loop (:meth:`fit`), and numpy-facing
    moment/sampling helpers so the rest of the package never touches
    torch tensors directly.
    """

    #: Monte-Carlo sample count used for posterior moments.
    N_MOMENT_SAMPLES = 4096

    def __init__(self, context_dim: int, theta_dim: int = 3, *,
                 transforms: int = 3, hidden: int = 96, bins: int = 8,
                 seed: int = 0):
        super().__init__()
        self.context_dim = context_dim
        self.theta_dim = theta_dim
        self.hparams = {"transforms": transforms, "hidden": hidden, "bins": bins}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.flow = zuko.flows.NSF(
                theta_dim, context=context_dim, transforms=transforms,
                hidden_features=(hidden, hidden), bins=bins,
            )
        # Context standardization, set from the training set in fit().
        self.register_buffer("c_loc", torch.zeros(context_dim))
        self.register_buffer("c_scale", torch.ones(context_dim))

    def _context(self, c: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(c), dtype=torch.float32)
        return (t - self.c_loc) / self.c_scale

    # ---- numpy-facing inference helpers --------------------------------

    def sample_np(self, c: np.ndarray, n: int, seed: int) -> np.ndarray:
        """Draw n posterior samples of theta_std for one conditioning c."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                s = self.flow(self._context(c)).sample((n,))
        return s.double().numpy()

    def posterior_mean_cov(self, c: np.ndarray):
        """Monte-Carlo mean and covariance of q(theta_std | c).

        Uses a fixed internal seed so repeated calls (and save/load
        round trips) are exactly reproducible.
        """
        s = self.sample_np(c, self.N_MOMENT_SAMPLES, seed=0)
        return s.mean(axis=0), np.cov(s.T)

    # ---- training ------------------------------------------------------

    def fit(self, C: np.ndarray, Theta: np.ndarray, *, epochs: int = 300,
            batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-5,
            val_fraction: float = 0.1, patience: int = 30,
            seed: int = 0, verbose: bool = False) -> list[float]:
        """Maximum-likelihood training; returns the per-epoch mean NLL.

        Uses **validation-based early stopping** (the standard NPE recipe,
        e.g. the ``sbi`` toolkit's default): a held-out fraction of the
        episodes tracks generalization, training stops after ``patience``
        epochs without improvement, and the best-validation weights are
        restored.  Without this the flow overfits the training episodes
        and the posterior becomes overconfident (SBC under-coverage).
        """
        import copy

        self.c_loc = torch.as_tensor(C.mean(axis=0), dtype=torch.float32)
        self.c_scale = torch.as_tensor(
            np.maximum(C.std(axis=0), 1e-6), dtype=torch.float32
        )
        Ct = self._context(C)
        Tt = torch.as_tensor(Theta, dtype=torch.float32)
        opt = torch.optim.AdamW(self.flow.parameters(), lr=lr,
                                weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        gen = torch.Generator().manual_seed(seed)
        n = len(Ct)
        n_val = int(round(n * val_fraction))
        split = torch.randperm(n, generator=gen)
        val_idx, train_idx = split[:n_val], split[n_val:]
        best_val = np.inf
        best_state = None
        stale = 0
        history = []
        self.flow.train()
        for epoch in range(epochs):
            perm = train_idx[torch.randperm(len(train_idx), generator=gen)]
            losses = []
            for start in range(0, len(perm), batch_size):
                idx = perm[start:start + batch_size]
                loss = -self.flow(Ct[idx]).log_prob(Tt[idx]).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
            sched.step()
            history.append(float(np.mean(losses)))
            if n_val > 0:
                with torch.no_grad():
                    val_nll = float(-self.flow(Ct[val_idx]).log_prob(Tt[val_idx]).mean())
                if val_nll < best_val - 1e-4:
                    best_val = val_nll
                    best_state = copy.deepcopy(self.flow.state_dict())
                    stale = 0
                else:
                    stale += 1
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                msg = f"  epoch {epoch:4d}: NLL {history[-1]:.4f}"
                if n_val > 0:
                    msg += f"  val {val_nll:.4f} (best {best_val:.4f})"
                print(msg)
            if n_val > 0 and stale >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val NLL {best_val:.4f})")
                break
        if best_state is not None:
            self.flow.load_state_dict(best_state)
        self.flow.eval()
        return history


# ---------------------------------------------------------------------------
# The amortized posterior object
# ---------------------------------------------------------------------------

@dataclass
class AmortizedPosterior:
    """Trained NPE bundle: featurizer + conditional flow + standardization."""

    featurizer: ResidualProjectionFeaturizer
    flow: ConditionalFlow
    prior_sd: np.ndarray               # (3,) theta standardization
    metadata: dict = field(default_factory=dict)

    def infer(self, c: np.ndarray, rng: np.random.Generator):
        """Return (theta_map, theta_cov) in physical units + a sampler."""
        mean_std, cov_std = self.flow.posterior_mean_cov(c)
        theta_map = DetectorState.from_vector(mean_std * self.prior_sd)
        cov = cov_std * np.outer(self.prior_sd, self.prior_sd)

        def sampler(n: int, rng_local: np.random.Generator) -> list[DetectorState]:
            seed = int(rng_local.integers(0, 2**31 - 1))
            s = self.flow.sample_np(c, n, seed) * self.prior_sd
            return [DetectorState.from_vector(v) for v in s]

        return theta_map, cov, sampler

    # ---- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.flow.state_dict(),
            "context_dim": self.flow.context_dim,
            "theta_dim": self.flow.theta_dim,
            "hparams": self.flow.hparams,
            "prior_sd": torch.as_tensor(self.prior_sd),
            "control_dim": self.featurizer.control_dim,
            "hf_component_scale": torch.as_tensor(self.featurizer.hf_component_scale),
            "lf_component_scale": torch.as_tensor(self.featurizer.lf_component_scale),
            "metadata": self.metadata,
        }, path)

    @staticmethod
    def load(path: str) -> "AmortizedPosterior":
        d = torch.load(path, map_location="cpu", weights_only=True)
        featurizer = ResidualProjectionFeaturizer(
            control_dim=int(d["control_dim"]),
            hf_component_scale=d["hf_component_scale"].double().numpy(),
            lf_component_scale=d["lf_component_scale"].double().numpy(),
        )
        flow = ConditionalFlow(
            int(d["context_dim"]), int(d["theta_dim"]), **d["hparams"],
        )
        flow.load_state_dict(d["state_dict"])
        flow.eval()
        return AmortizedPosterior(
            featurizer, flow, d["prior_sd"].double().numpy(),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# Conditioning from a live dataset (shared by training and Step 5)
# ---------------------------------------------------------------------------

def build_conditioning(
    featurizer: ResidualProjectionFeaturizer,
    hf_u: np.ndarray, hf_z: np.ndarray,
    lf_u: np.ndarray, lf_z: np.ndarray,
    control_map,
    simulator,
    objective,
    summarizer: CurveSummarizer,
    refl_summarizer: ReflectivitySummarizer,
) -> np.ndarray:
    """Residual-projection conditioning vector for one measurement set.

    ``hf_z`` / ``lf_z`` are the measured summary vectors; residuals are
    taken against the nominal simulation (theta = 0) and scaled by the
    featurizer's per-component response scales.
    """
    theta0 = DetectorState()
    if len(hf_u):
        x_hf = np.stack([control_map.to_normalized(u) for u in hf_u])
        sim_hf = np.stack([simulator.predict_summaries(u, theta0, summarizer) for u in hf_u])
        resid_hf = (hf_z - sim_hf) / featurizer.hf_component_scale
    else:
        x_hf = np.zeros((0, featurizer.control_dim))
        resid_hf = np.zeros((0, featurizer.n_hf_comp))
    if len(lf_u):
        x_lf = np.stack([control_map.to_normalized(u) for u in lf_u])
        sim_lf = np.stack([
            simulator.predict_reflectivity_summaries(u, theta0, refl_summarizer) for u in lf_u
        ])
        resid_lf = (lf_z - sim_lf) / featurizer.lf_component_scale
    else:
        x_lf = np.zeros((0, featurizer.control_dim))
        resid_lf = np.zeros((0, featurizer.n_lf_comp))
    return featurizer.from_arrays(x_hf, resid_hf, x_lf, resid_lf)


@dataclass
class TrainingConfig:
    """Settings for offline NPE training (roadmap Phase 2.1)."""

    n_episodes: int = 24000
    hf_range: tuple = (3, 12)          # measurements per episode
    lf_range: tuple = (0, 8)
    transforms: int = 3                # spline-flow transform layers
    hidden: int = 96                   # hyper-network hidden width
    bins: int = 8                      # spline bins per transform
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    val_fraction: float = 0.1          # held-out episodes for early stopping
    patience: int = 40                 # epochs without val improvement
    # Misspecification robustness: amplitude of the smooth systematic
    # bias injected into each episode's summaries, in units of each
    # component's response scale (0 disables — do not use for production).
    discrepancy_injection: float = 0.06
    hf_noise_rel: float = 0.02         # per-summary relative measurement noise
    lf_noise_rel: float = 0.03
    seed: int = 0


def _component_scales(simulator, control_map, objective, n: int, rng):
    """Typical magnitude/spread of each summary component over the domain."""
    summ = CurveSummarizer(objective, simulator.freqs)
    refl = ReflectivitySummarizer(simulator.freqs)
    limits = control_map.cfg.limits()
    hf, lf = [], []
    for _ in range(n):
        u = rng.uniform(-0.5, 0.5, control_map.dim) * limits
        theta = DetectorState()
        hf.append(simulator.predict_summaries(u, theta, summ))
        lf.append(simulator.predict_reflectivity_summaries(u, theta, refl))
    hf = np.stack(hf)
    lf = np.stack(lf)
    hf_scale = np.maximum(np.std(hf, axis=0), 0.1 * np.abs(np.mean(hf, axis=0)) + 1e-9)
    lf_scale = np.maximum(np.std(lf, axis=0), 0.1 * np.abs(np.mean(lf, axis=0)) + 1e-9)
    return summ, refl, hf_scale, lf_scale


def generate_training_episodes(
    simulator,
    control_map,
    config,
    train_config: TrainingConfig | None = None,
):
    """Simulate the NPE training set (episode generation only).

    Draws ``theta`` from the Step-5 prior and measurement sets at random
    control inputs, simulates the summary vectors with measurement noise
    and an injected smooth systematic (misspecification robustness), and
    returns ``(featurizer, prior_sd, C, Theta)`` where ``C`` are the
    conditioning vectors and ``Theta`` the standardized true states.
    Split out from :func:`train_amortized_posterior` because episode
    generation dominates the training cost and can be reused across
    hyperparameter settings.
    """
    tc = train_config or TrainingConfig()
    rng = np.random.default_rng(tc.seed)
    from .objectives import Objective

    objective = Objective(config.objective)
    prior_sd = np.array([
        config.step5.prior_sd_z_offset,
        config.step5.prior_sd_compression,
        config.step5.prior_sd_log_loss,
    ])
    summ, refl, hf_scale, lf_scale = _component_scales(
        simulator, control_map, objective, 400, rng
    )
    featurizer = ResidualProjectionFeaturizer(
        control_dim=control_map.dim,
        hf_component_scale=hf_scale,
        lf_component_scale=lf_scale,
    )
    limits = control_map.cfg.limits()

    C = np.empty((tc.n_episodes, featurizer.dim))
    Theta = np.empty((tc.n_episodes, 3))
    for e in range(tc.n_episodes):
        theta_std = np.clip(rng.standard_normal(3), -4, 4)
        theta = DetectorState.from_vector(theta_std * prior_sd)
        n_hf = int(rng.integers(tc.hf_range[0], tc.hf_range[1] + 1))
        n_lf = int(rng.integers(tc.lf_range[0], tc.lf_range[1] + 1))
        hf_u = np.stack([rng.uniform(-0.5, 0.5, control_map.dim) * limits for _ in range(n_hf)])
        # Shared per-episode systematic bias (calibration/tilt/focus stand-in).
        hf_bias = tc.discrepancy_injection * hf_scale * rng.standard_normal(featurizer.n_hf_comp)
        hf_z = np.stack([simulator.predict_summaries(u, theta, summ) for u in hf_u])
        hf_z = hf_z + hf_bias + tc.hf_noise_rel * hf_scale * rng.standard_normal(hf_z.shape)
        if n_lf > 0:
            lf_u = np.stack([rng.uniform(-0.5, 0.5, control_map.dim) * limits for _ in range(n_lf)])
            lf_bias = tc.discrepancy_injection * lf_scale * rng.standard_normal(featurizer.n_lf_comp)
            lf_z = np.stack([simulator.predict_reflectivity_summaries(u, theta, refl) for u in lf_u])
            lf_z = lf_z + lf_bias + tc.lf_noise_rel * lf_scale * rng.standard_normal(lf_z.shape)
        else:
            lf_u = np.zeros((0, control_map.dim))
            lf_z = np.zeros((0, featurizer.n_lf_comp))
        C[e] = build_conditioning(
            featurizer, hf_u, hf_z, lf_u, lf_z, control_map, simulator,
            objective, summ, refl,
        )
        Theta[e] = theta_std
    return featurizer, prior_sd, C, Theta


def train_amortized_posterior(
    simulator,
    control_map,
    config,
    train_config: TrainingConfig | None = None,
    verbose: bool = False,
    episodes: tuple | None = None,
) -> AmortizedPosterior:
    """Train an :class:`AmortizedPosterior` offline against the simulator.

    Generates training episodes (see :func:`generate_training_episodes`;
    pass a pre-generated tuple via ``episodes`` to skip this) and fits
    the conditional spline flow to recover ``theta`` from the
    residual-projection conditioning vector, with validation-based early
    stopping.
    """
    tc = train_config or TrainingConfig()
    featurizer, prior_sd, C, Theta = episodes or generate_training_episodes(
        simulator, control_map, config, tc
    )
    flow = ConditionalFlow(
        featurizer.dim, 3, transforms=tc.transforms, hidden=tc.hidden,
        bins=tc.bins, seed=tc.seed,
    )
    history = flow.fit(
        C, Theta, epochs=tc.epochs, batch_size=tc.batch_size, lr=tc.lr,
        val_fraction=tc.val_fraction, patience=tc.patience,
        seed=tc.seed, verbose=verbose,
    )
    return AmortizedPosterior(
        featurizer=featurizer,
        flow=flow,
        prior_sd=prior_sd,
        metadata={
            "n_episodes": len(C),
            "final_nll": history[-1],
            "objective": config.objective,
            "discrepancy_injection": tc.discrepancy_injection,
        },
    )

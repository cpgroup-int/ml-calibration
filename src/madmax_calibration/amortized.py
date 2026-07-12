"""Amortized simulation-based inference for the detector state (Phase 2.1).

Roadmap Phase 2 replaces the joint-MAP + Laplace estimate of the
detector state ``theta`` with an **amortized neural posterior estimator**
(NPE): a conditional density ``q(theta | c)`` trained offline against the
fast simulator, evaluated online in a millisecond forward pass.  The
online runtime stays numpy/scipy-only — the network is a small mixture
density network implemented here in pure numpy, and trained weights are
exported to a plain ``.npz`` blob (no PyTorch at inference time).

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
``Step5Result.theta_samples`` becomes exact sampling from the amortized
mixture instead of a Gaussian Laplace draw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simulator import DetectorState
from .summaries import (
    REFLECTIVITY_SUMMARY_NAMES,
    SUMMARY_NAMES,
    CurveSummarizer,
    ReflectivitySummarizer,
)


def _logsumexp(a: np.ndarray, axis=-1, keepdims=False):
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


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
# Mixture density network (pure numpy)
# ---------------------------------------------------------------------------

@dataclass
class MDNParams:
    """Weights of the MLP + mixture-density head."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    W_out: np.ndarray
    b_out: np.ndarray

    def copy(self) -> "MDNParams":
        return MDNParams(*(a.copy() for a in self.as_list()))

    def as_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W_out, self.b_out]


class MixtureDensityNetwork:
    """Conditional Gaussian-mixture posterior q(theta_std | c).

    Two tanh hidden layers into a diagonal-covariance mixture over the
    standardized detector state.  Forward evaluation, exact sampling and
    log-density are pure numpy; training (:meth:`fit`) uses hand-derived
    gradients with Adam, needed only offline.
    """

    def __init__(self, input_dim: int, theta_dim: int = 3, n_components: int = 4,
                 hidden: int = 64, params: MDNParams | None = None, seed: int = 0):
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.K = n_components
        self.hidden = hidden
        self.out_dim = n_components * (1 + 2 * theta_dim)
        if params is None:
            rng = np.random.default_rng(seed)
            scale1 = np.sqrt(1.0 / input_dim)
            scale2 = np.sqrt(1.0 / hidden)
            params = MDNParams(
                W1=rng.normal(0, scale1, (input_dim, hidden)),
                b1=np.zeros(hidden),
                W2=rng.normal(0, scale2, (hidden, hidden)),
                b2=np.zeros(hidden),
                W_out=rng.normal(0, 0.01, (hidden, self.out_dim)),
                b_out=np.zeros(self.out_dim),
            )
        self.p = params

    # ---- forward -------------------------------------------------------

    def _forward_hidden(self, C: np.ndarray):
        z1 = C @ self.p.W1 + self.p.b1
        h1 = np.tanh(z1)
        z2 = h1 @ self.p.W2 + self.p.b2
        h2 = np.tanh(z2)
        out = h2 @ self.p.W_out + self.p.b_out
        return h1, h2, out

    def _split(self, out: np.ndarray):
        K, D = self.K, self.theta_dim
        logits = out[..., :K]
        means = out[..., K:K + K * D].reshape(*out.shape[:-1], K, D)
        log_sd = out[..., K + K * D:].reshape(*out.shape[:-1], K, D)
        log_sd = np.clip(log_sd, -6.0, 3.0)
        return logits, means, log_sd

    def distribution(self, C: np.ndarray):
        """Return (log mixture weights, means, sds) for a batch of c."""
        C = np.atleast_2d(C)
        _, _, out = self._forward_hidden(C)
        logits, means, log_sd = self._split(out)
        log_w = logits - _logsumexp(logits, axis=-1, keepdims=True)
        return log_w, means, np.exp(log_sd)

    def posterior_mean_cov(self, c: np.ndarray):
        """Mean and covariance of q(theta_std | c) for a single c."""
        log_w, means, sds = self.distribution(c)
        w = np.exp(log_w)[0]                     # (K,)
        m = means[0]                              # (K, D)
        sd = sds[0]                               # (K, D)
        mean = w @ m                              # (D,)
        # Law of total covariance for a diagonal mixture.
        cov = np.zeros((self.theta_dim, self.theta_dim))
        for k in range(self.K):
            diff = (m[k] - mean)[:, None]
            cov += w[k] * (np.diag(sd[k] ** 2) + diff @ diff.T)
        return mean, cov

    def sample(self, c: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        log_w, means, sds = self.distribution(c)
        w = np.exp(log_w)[0]
        comps = rng.choice(self.K, size=n, p=w / w.sum())
        eps = rng.standard_normal((n, self.theta_dim))
        return means[0][comps] + sds[0][comps] * eps

    # ---- training ------------------------------------------------------

    def _nll_and_grad(self, C: np.ndarray, Theta: np.ndarray):
        """Mean NLL and parameter gradients over a minibatch."""
        n = len(C)
        K, D = self.K, self.theta_dim
        h1, h2, out = self._forward_hidden(C)
        logits, means, log_sd = self._split(out)
        log_w = logits - _logsumexp(logits, axis=-1, keepdims=True)   # (n, K)
        sd = np.exp(log_sd)
        th = Theta[:, None, :]                                         # (n,1,D)
        zscore = (th - means) / sd                                    # (n,K,D)
        log_norm = -0.5 * np.sum(zscore ** 2 + 2 * log_sd + np.log(2 * np.pi), axis=-1)
        log_joint = log_w + log_norm                                  # (n,K)
        log_q = _logsumexp(log_joint, axis=-1)                        # (n,)
        nll = -np.mean(log_q)

        gamma = np.exp(log_joint - log_q[:, None])                    # responsibilities (n,K)
        # Grad wrt head pre-activations.
        d_logits = (np.exp(log_w) - gamma)                           # (n,K)
        d_means = (-gamma[..., None]) * (zscore / sd)                # (n,K,D)
        d_log_sd = (-gamma[..., None]) * (zscore ** 2 - 1.0)         # (n,K,D)
        # respect the clip on log_sd (zero grad outside range)
        d_log_sd = np.where((log_sd > -6.0) & (log_sd < 3.0), d_log_sd, 0.0)
        d_out = np.concatenate(
            [d_logits, d_means.reshape(n, K * D), d_log_sd.reshape(n, K * D)], axis=1
        ) / n                                                         # (n, out_dim)

        gW_out = h2.T @ d_out
        gb_out = d_out.sum(0)
        d_h2 = d_out @ self.p.W_out.T
        d_z2 = d_h2 * (1 - h2 ** 2)
        gW2 = h1.T @ d_z2
        gb2 = d_z2.sum(0)
        d_h1 = d_z2 @ self.p.W2.T
        d_z1 = d_h1 * (1 - h1 ** 2)
        gW1 = C.T @ d_z1
        gb1 = d_z1.sum(0)
        return nll, MDNParams(gW1, gb1, gW2, gb2, gW_out, gb_out)

    def fit(self, C: np.ndarray, Theta: np.ndarray, *, epochs: int = 400,
            batch_size: int = 256, lr: float = 3e-3, weight_decay: float = 1e-4,
            seed: int = 0, verbose: bool = False) -> list[float]:
        rng = np.random.default_rng(seed)
        n = len(C)
        params = self.p.as_list()
        m = [np.zeros_like(a) for a in params]
        v = [np.zeros_like(a) for a in params]
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        step = 0
        for epoch in range(epochs):
            perm = rng.permutation(n)
            losses = []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                loss, grad = self._nll_and_grad(C[idx], Theta[idx])
                losses.append(loss)
                glist = grad.as_list()
                step += 1
                for i, (pi, gi) in enumerate(zip(params, glist)):
                    gi = gi + weight_decay * pi
                    m[i] = b1 * m[i] + (1 - b1) * gi
                    v[i] = b2 * v[i] + (1 - b2) * gi * gi
                    mhat = m[i] / (1 - b1 ** step)
                    vhat = v[i] / (1 - b2 ** step)
                    pi -= lr * mhat / (np.sqrt(vhat) + eps)
            history.append(float(np.mean(losses)))
            if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch:4d}: NLL {history[-1]:.4f}")
        return history


# ---------------------------------------------------------------------------
# The amortized posterior object
# ---------------------------------------------------------------------------

@dataclass
class AmortizedPosterior:
    """Trained NPE bundle: featurizer + MDN + standardization."""

    featurizer: ResidualProjectionFeaturizer
    mdn: MixtureDensityNetwork
    prior_sd: np.ndarray               # (3,) theta standardization
    metadata: dict = field(default_factory=dict)

    def infer(self, c: np.ndarray, rng: np.random.Generator):
        """Return (theta_map, theta_cov) in physical units + a sampler."""
        mean_std, cov_std = self.mdn.posterior_mean_cov(c)
        theta_map = DetectorState.from_vector(mean_std * self.prior_sd)
        cov = cov_std * np.outer(self.prior_sd, self.prior_sd)

        def sampler(n: int, rng_local: np.random.Generator) -> list[DetectorState]:
            s = self.mdn.sample(c, n, rng_local) * self.prior_sd
            return [DetectorState.from_vector(v) for v in s]

        return theta_map, cov, sampler

    # ---- persistence ---------------------------------------------------

    def save(self, path: str) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        p = self.mdn.p
        np.savez(
            path,
            W1=p.W1, b1=p.b1, W2=p.W2, b2=p.b2, W_out=p.W_out, b_out=p.b_out,
            prior_sd=self.prior_sd,
            control_dim=self.featurizer.control_dim,
            hf_component_scale=self.featurizer.hf_component_scale,
            lf_component_scale=self.featurizer.lf_component_scale,
            n_components=self.mdn.K,
            hidden=self.mdn.hidden,
        )

    @staticmethod
    def load(path: str) -> "AmortizedPosterior":
        d = np.load(path, allow_pickle=False)
        featurizer = ResidualProjectionFeaturizer(
            control_dim=int(d["control_dim"]),
            hf_component_scale=d["hf_component_scale"],
            lf_component_scale=d["lf_component_scale"],
        )
        params = MDNParams(d["W1"], d["b1"], d["W2"], d["b2"], d["W_out"], d["b_out"])
        mdn = MixtureDensityNetwork(
            input_dim=params.W1.shape[0], n_components=int(d["n_components"]),
            hidden=int(d["hidden"]), params=params,
        )
        return AmortizedPosterior(featurizer, mdn, d["prior_sd"])


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

    n_episodes: int = 12000
    hf_range: tuple = (3, 12)          # measurements per episode
    lf_range: tuple = (0, 8)
    n_components: int = 5
    hidden: int = 96
    epochs: int = 600
    batch_size: int = 256
    lr: float = 3e-3
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


def train_amortized_posterior(
    simulator,
    control_map,
    config,
    train_config: TrainingConfig | None = None,
    verbose: bool = False,
) -> AmortizedPosterior:
    """Train an :class:`AmortizedPosterior` offline against the simulator.

    Draws ``theta`` from the Step-5 prior and measurement sets at random
    control inputs, simulates the summary vectors with measurement noise
    and an injected smooth systematic (misspecification robustness), and
    fits the mixture-density network to recover ``theta`` from the
    residual-projection conditioning vector.
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

    mdn = MixtureDensityNetwork(
        input_dim=featurizer.dim, n_components=tc.n_components, hidden=tc.hidden,
        seed=tc.seed,
    )
    history = mdn.fit(
        C, Theta, epochs=tc.epochs, batch_size=tc.batch_size, lr=tc.lr,
        seed=tc.seed, verbose=verbose,
    )
    return AmortizedPosterior(
        featurizer=featurizer,
        mdn=mdn,
        prior_sd=prior_sd,
        metadata={
            "n_episodes": tc.n_episodes,
            "final_nll": history[-1],
            "objective": config.objective,
            "discrepancy_injection": tc.discrepancy_injection,
        },
    )



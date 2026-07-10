"""A small exact Gaussian-process module (RBF kernel, fixed per-point noise).

Used for the Step-5 scalar discrepancy model, the Step-3 antenna-alignment
surrogate and the learned soft-constraint model.  Deliberately minimal:
the design notes only require a GP with fixed/heteroscedastic observation
noise and marginal-likelihood hyperparameter fitting; heavy BO frameworks
are avoided so the package stays dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def rbf_kernel(x1: np.ndarray, x2: np.ndarray, amplitude: float, lengthscales: np.ndarray) -> np.ndarray:
    d = (x1[:, None, :] - x2[None, :, :]) / lengthscales[None, None, :]
    return amplitude**2 * np.exp(-0.5 * np.sum(d**2, axis=-1))


@dataclass
class GaussianProcess:
    """Exact GP regression with fixed per-point observation noise.

    Posterior for f (the latent function) given y_i = f(x_i) + eps_i,
    eps_i ~ N(0, noise_sd_i^2).  A zero prior mean is assumed; callers
    subtract their own mean model (e.g. the physics simulator) first.
    """

    amplitude: float
    lengthscales: np.ndarray
    jitter: float = 1e-10

    _x: np.ndarray | None = None
    _y: np.ndarray | None = None
    _noise: np.ndarray | None = None
    _chol = None
    _alpha: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, noise_sd: np.ndarray) -> "GaussianProcess":
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        noise = np.broadcast_to(np.asarray(noise_sd, dtype=float), y.shape).copy()
        k = rbf_kernel(x, x, self.amplitude, self.lengthscales)
        k[np.diag_indices_from(k)] += noise**2 + self.jitter
        self._chol = cho_factor(k, lower=True)
        self._alpha = cho_solve(self._chol, y)
        self._x, self._y, self._noise = x, y, noise
        return self

    def predict(self, x_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Latent mean and sd at x_star."""
        x_star = np.atleast_2d(np.asarray(x_star, dtype=float))
        if self._x is None:
            mean = np.zeros(len(x_star))
            sd = np.full(len(x_star), self.amplitude)
            return mean, sd
        k_star = rbf_kernel(x_star, self._x, self.amplitude, self.lengthscales)
        mean = k_star @ self._alpha
        v = cho_solve(self._chol, k_star.T)
        var = self.amplitude**2 - np.sum(k_star * v.T, axis=1)
        return mean, np.sqrt(np.clip(var, 1e-16, None))

    def log_marginal_likelihood(self) -> float:
        if self._x is None:
            raise RuntimeError("fit first")
        n = len(self._y)
        log_det = 2.0 * np.sum(np.log(np.diag(self._chol[0])))
        return float(
            -0.5 * self._y @ self._alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
        )


def fit_gp_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    noise_sd: np.ndarray,
    amplitude_bounds: tuple[float, float],
    lengthscale_bounds: tuple[float, float],
    amplitude_prior_sd: float | None = None,
    n_restarts: int = 2,
    seed: int = 0,
) -> GaussianProcess:
    """Fit (amplitude, isotropic lengthscale) by penalized marginal likelihood.

    ``amplitude_prior_sd`` adds a half-normal prior on the amplitude,
    implementing the informative discrepancy-amplitude prior required by
    the Step 5 design (section 9.3) to limit theta/discrepancy confounding.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    dim = x.shape[1]
    rng = np.random.default_rng(seed)

    def neg_obj(log_params: np.ndarray) -> float:
        amp = float(np.exp(log_params[0]))
        ls = float(np.exp(log_params[1]))
        gp = GaussianProcess(amplitude=amp, lengthscales=np.full(dim, ls))
        try:
            gp.fit(x, y, noise_sd)
            lml = gp.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            return 1e10
        penalty = 0.0
        if amplitude_prior_sd is not None:
            penalty = 0.5 * (amp / amplitude_prior_sd) ** 2
        return -lml + penalty

    lo = np.log([amplitude_bounds[0], lengthscale_bounds[0]])
    hi = np.log([amplitude_bounds[1], lengthscale_bounds[1]])
    best = None
    starts = [0.5 * (lo + hi)] + [rng.uniform(lo, hi) for _ in range(n_restarts)]
    for x0 in starts:
        res = minimize(neg_obj, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    amp = float(np.exp(best.x[0]))
    ls = float(np.exp(best.x[1]))
    gp = GaussianProcess(amplitude=amp, lengthscales=np.full(dim, ls))
    gp.fit(x, y, noise_sd)
    return gp

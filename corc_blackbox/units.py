"""
units.py
--------
Hopf / Stuart-Landau oscillator with one-dimensional slow adaptation.
Node state: (x_i, y_i, a_i). Amplitude squared r_i^2 = x_i^2 + y_i^2.

Equations (per CLAUDE.md §3):
    dx_i/dt = (mu_i - r_i^2 - beta_i*a_i)*x_i - omega_i*y_i + input_x + coupling_x + noise_x
    dy_i/dt = (mu_i - r_i^2 - beta_i*a_i)*y_i + omega_i*x_i + coupling_y + noise_y
    tau_a_i * da_i/dt = -a_i + alpha_i * r_i^2

All arrays are NumPy ndarrays for vectorised integration.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class HopfSlowUnit:
    """
    Vectorised ensemble of Hopf+slow-adaptation oscillators.

    Parameters
    ----------
    N : int
        Number of oscillators.
    mu : np.ndarray, shape (N,)
        Oscillation-gain parameter(s).
    omega : np.ndarray, shape (N,)
        Intrinsic angular frequency (rad/time).
    tau_a : np.ndarray, shape (N,)
        Slow adaptation time constant.
    alpha : np.ndarray, shape (N,)
        Amplitude -> slow variable gain.
    beta : np.ndarray, shape (N,)
        Slow variable -> amplitude suppression strength.
    sigma : np.ndarray, shape (N,)
        Gaussian white-noise intensity (sqrt(power)).
    dt : float
        Integration step.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        N: int,
        mu: np.ndarray,
        omega: np.ndarray,
        tau_a: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        sigma: np.ndarray,
        dt: float,
        rng: Optional[np.random.Generator] = None,
    ):
        assert N >= 1
        self.N = N
        self.dt = dt
        self.mu = np.asarray(mu, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.tau_a = np.asarray(tau_a, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.beta = np.asarray(beta, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

        # state
        self.x = self.rng.normal(0.0, 0.1, size=N)
        self.y = self.rng.normal(0.0, 0.1, size=N)
        self.a = np.zeros(N)

    @classmethod
    def heterogeneous(
        cls,
        N: int,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
        mu_range: Tuple[float, float] = (0.08, 0.18),
        f_range: Tuple[float, float] = (0.8, 2.5),
        tau_a_range: Tuple[float, float] = (0.8, 2.5),
        alpha_range: Tuple[float, float] = (0.8, 1.5),
        beta_range: Tuple[float, float] = (0.8, 1.5),
        sigma_range: Tuple[float, float] = (0.005, 0.02),
    ) -> HopfSlowUnit:
        """
        Factory: sample parameters uniformly as recommended in CLAUDE.md §10.
        omega is built from frequency in Hz: omega = 2*pi*f.
        """
        rng = rng if rng is not None else np.random.default_rng()
        mu = rng.uniform(*mu_range, size=N)
        omega = 2.0 * np.pi * rng.uniform(*f_range, size=N)
        tau_a = rng.uniform(*tau_a_range, size=N)
        alpha = rng.uniform(*alpha_range, size=N)
        beta = rng.uniform(*beta_range, size=N)
        sigma = rng.uniform(*sigma_range, size=N)
        return cls(N, mu, omega, tau_a, alpha, beta, sigma, dt, rng)

    @classmethod
    def homogeneous(
        cls,
        N: int,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
        mu: float = 0.12,
        f: float = 1.5,
        tau_a: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        sigma: float = 0.01,
    ) -> HopfSlowUnit:
        """All nodes share identical parameters (for ablation)."""
        rng = rng if rng is not None else np.random.default_rng()
        return cls(
            N,
            np.full(N, mu),
            np.full(N, 2.0 * np.pi * f),
            np.full(N, tau_a),
            np.full(N, alpha),
            np.full(N, beta),
            np.full(N, sigma),
            dt,
            rng,
        )

    @property
    def r2(self) -> np.ndarray:
        """Amplitude squared."""
        return self.x ** 2 + self.y ** 2

    @property
    def r(self) -> np.ndarray:
        return np.sqrt(self.r2)

    @property
    def phi(self) -> np.ndarray:
        return np.arctan2(self.y, self.x)

    def step(
        self,
        input_x: np.ndarray,
        coupling_x: np.ndarray,
        coupling_y: np.ndarray,
        freq_mod: Optional[np.ndarray] = None,
    ) -> None:
        """
        One Euler-Maruyama step.

        Parameters
        ----------
        input_x : (N,)  gain*projected input added to dx only
        coupling_x, coupling_y : (N,)  from coupling module
        freq_mod : (N,) optional additive modulation to omega
        """
        x, y, a = self.x, self.y, self.a
        r2 = x * x + y * y
        omega = self.omega + (freq_mod if freq_mod is not None else 0.0)

        # deterministic drift
        dx = (self.mu - r2 - self.beta * a) * x - omega * y + input_x + coupling_x
        dy = (self.mu - r2 - self.beta * a) * y + omega * x + coupling_y
        da = (-a + self.alpha * r2) / self.tau_a

        # noise (scaled by sqrt(dt) for Euler-Maruyama)
        if np.any(self.sigma > 0):
            noise_x = self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.N)
            noise_y = self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.N)
            dx += noise_x
            dy += noise_y

        self.x = x + dx * self.dt
        self.y = y + dy * self.dt
        self.a = a + da * self.dt

    def reset_state(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = self.rng.normal(0.0, 0.1, size=self.N)
        self.y = self.rng.normal(0.0, 0.1, size=self.N)
        self.a = np.zeros(self.N)

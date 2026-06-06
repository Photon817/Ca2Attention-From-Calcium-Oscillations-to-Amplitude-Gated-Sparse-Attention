"""
units.py — CORC v2
==================
Calcium-inspired Excitable Unit (CEU): simplified two-pool calcium model.

State: (c_i, s_i, a_i)
  c_i : cytosolic calcium (fast, excitable)
  s_i : ER stored calcium (slow recovery)
  a_i : ultra-slow adaptation (IP3R-like slow inactivation)

Equations (dimensionless):
  tau_c * dc_i/dt = -c_i + J_rel(c_i, s_i) + I_ext_i + C_i + sigma_c * xi
  tau_s * ds_i/dt = (1 - s_i) / tau_pump - gamma * J_rel(c_i, s_i)
  tau_a * da_i/dt = -a_i + alpha_a * H(c_i - c_th)

Release term:
  J_rel = g_rel * s_i * sigmoid(c_i - c_th, k) * (1 - a_i)
  sigmoid(x, k) = 1 / (1 + exp(-k*x))
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


def sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    """Sigmoid with steepness k. Clamped for numerical safety."""
    return 1.0 / (1.0 + np.exp(-np.clip(k * x, -50, 50)))


class CalciumExcitableUnit:
    """
    Vectorised ensemble of calcium-inspired excitable units.

    Parameters
    ----------
    N : int
        Number of units.
    c_th : np.ndarray (N,)
        Calcium release threshold.
    g_rel : np.ndarray (N,)
        Release conductance.
    tau_pump : np.ndarray (N,)
        Pump time constant for ER refill.
    k_rel : np.ndarray (N,)
        Sigmoid steepness for release.
    tau_a : np.ndarray (N,)
        Ultra-slow adaptation time constant.
    alpha_a : np.ndarray (N,)
        Adaptation gain.
    sigma_c : np.ndarray (N,)
        Noise intensity.
    tau_c : float
        Cytosolic calcium time constant (fast).
    tau_s : float
        ER store time constant (slow).
    gamma : float
        Release consumption rate from ER.
    dt : float
        Integration step.
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        N: int,
        c_th: np.ndarray,
        g_rel: np.ndarray,
        tau_pump: np.ndarray,
        k_rel: np.ndarray,
        tau_a: np.ndarray,
        alpha_a: np.ndarray,
        sigma_c: np.ndarray,
        tau_c: float = 0.1,
        tau_s: float = 1.0,
        gamma: float = 0.3,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        assert N >= 1
        self.N = N
        self.dt = dt
        self.tau_c = tau_c
        self.tau_s = tau_s
        self.gamma = gamma
        self.c_th = np.asarray(c_th, dtype=float)
        self.g_rel = np.asarray(g_rel, dtype=float)
        self.tau_pump = np.asarray(tau_pump, dtype=float)
        self.k_rel = np.asarray(k_rel, dtype=float)
        self.tau_a = np.asarray(tau_a, dtype=float)
        self.alpha_a = np.asarray(alpha_a, dtype=float)
        self.sigma_c = np.asarray(sigma_c, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initial state: near-threshold calcium, full ER, no adaptation
        self.c = self.rng.uniform(0.05, 0.25, size=N).astype(np.float32)  # cytosolic Ca
        self.s = np.ones(N, dtype=np.float32)  # ER store (full)
        self.a = np.zeros(N, dtype=np.float32)  # adaptation

    @classmethod
    def heterogeneous(
        cls,
        N: int,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
        c_th_range: Tuple[float, float] = (0.20, 0.50),
        g_rel_range: Tuple[float, float] = (1.0, 2.0),
        tau_pump_range: Tuple[float, float] = (3.0, 10.0),
        k_rel_range: Tuple[float, float] = (10.0, 30.0),
        tau_a_mu: float = 1.5,
        tau_a_sigma: float = 0.5,
        alpha_a_range: Tuple[float, float] = (0.3, 0.8),
        sigma_c_range: Tuple[float, float] = (0.05, 0.15),
        tau_c: float = 0.1,
        tau_s: float = 1.0,
        gamma: float = 0.3,
    ) -> CalciumExcitableUnit:
        """Factory: heterogeneous parameter sampling as specified in the design doc."""
        rng = rng if rng is not None else np.random.default_rng()
        c_th = rng.uniform(*c_th_range, size=N)
        g_rel = rng.uniform(*g_rel_range, size=N)
        tau_pump = rng.uniform(*tau_pump_range, size=N)
        k_rel = rng.uniform(*k_rel_range, size=N)
        tau_a = np.exp(rng.normal(tau_a_mu, tau_a_sigma, size=N))  # log-normal
        alpha_a = rng.uniform(*alpha_a_range, size=N)
        sigma_c = rng.uniform(*sigma_c_range, size=N)
        return cls(N, c_th, g_rel, tau_pump, k_rel, tau_a, alpha_a, sigma_c,
                   tau_c=tau_c, tau_s=tau_s, gamma=gamma, dt=dt, rng=rng)

    @classmethod
    def homogeneous(
        cls,
        N: int,
        dt: float = 0.01,
        rng: Optional[np.random.Generator] = None,
        c_th: float = 0.5,
        g_rel: float = 1.0,
        tau_pump: float = 4.0,
        k_rel: float = 20.0,
        tau_a: float = 4.5,
        alpha_a: float = 0.1,
        sigma_c: float = 0.01,
        tau_c: float = 0.1,
        tau_s: float = 1.0,
        gamma: float = 0.3,
    ) -> CalciumExcitableUnit:
        """Factory: all nodes share identical parameters (for ablation)."""
        rng = rng if rng is not None else np.random.default_rng()
        return cls(
            N,
            np.full(N, c_th),
            np.full(N, g_rel),
            np.full(N, tau_pump),
            np.full(N, k_rel),
            np.full(N, tau_a),
            np.full(N, alpha_a),
            np.full(N, sigma_c),
            tau_c=tau_c, tau_s=tau_s, gamma=gamma, dt=dt, rng=rng,
        )

    def j_rel(self, c: np.ndarray, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Calcium-induced calcium release (CICR) term."""
        sig = sigmoid(c - self.c_th, self.k_rel)
        return self.g_rel * s * sig * (1.0 - a)

    def step(
        self,
        input_c: np.ndarray,
        coupling_c: np.ndarray,
    ) -> None:
        """
        One Euler-Maruyama step for the CEU ensemble.

        Parameters
        ----------
        input_c : (N,)  external input (gain * projected input)
        coupling_c : (N,)  coupling contribution from coupling module
        """
        c, s, a = self.c, self.s, self.a
        jrel = self.j_rel(c, s, a)

        # Cytosolic calcium: fast dynamics
        dc = (-c + jrel + input_c + coupling_c) / self.tau_c

        # ER store: slow recovery
        ds = ((1.0 - s) / self.tau_pump - self.gamma * jrel) / self.tau_s

        # Ultra-slow adaptation
        da = (-a + self.alpha_a * (c > self.c_th).astype(float)) / self.tau_a

        # Additive noise on c
        if np.any(self.sigma_c > 0):
            noise = self.sigma_c * np.sqrt(self.dt) * self.rng.standard_normal(self.N)
            dc += noise / self.tau_c  # scale noise by 1/tau_c for consistency

        self.c = np.clip(c + dc * self.dt, -0.5, 5.0)  # prevent blowup
        self.s = np.clip(s + ds * self.dt, 0.0, 2.0)
        self.a = np.clip(a + da * self.dt, 0.0, 2.0)

    def reset_state(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.c = self.rng.uniform(0.05, 0.25, size=self.N).astype(np.float32)
        self.s = np.ones(self.N, dtype=np.float32)
        self.a = np.zeros(self.N, dtype=np.float32)
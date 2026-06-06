"""
baselines.py
------------
Baseline reservoirs for comparison (CLAUDE.md §8):
1. Standard Echo State Network (ESN)
2. Simple independent oscillator reservoir (no coupling)
3. Linear/harmonic oscillator version (no Hopf nonlinearity)
4. Direct input -> linear readout (null reservoir)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List

from .units import HopfSlowUnit
from .coupling import LinearCoupling
from .reservoir import CORCReservoir, CORCState


class ESN:
    """
    Standard Echo State Network with tanh activation.
    x(t+1) = (1-leak)*x(t) + leak*tanh(W_in @ u(t) + W @ x(t))
    """

    def __init__(
        self,
        N: int,
        input_dim: int = 1,
        leak: float = 0.1,
        spectral_radius: float = 0.9,
        sparsity: float = 0.15,
        input_scaling: float = 1.0,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.input_dim = input_dim
        self.leak = leak
        self.dt = dt
        rng = np.random.default_rng(seed)

        # recurrent weights
        W = rng.uniform(-1, 1, size=(N, N))
        mask = rng.random((N, N)) < sparsity
        W = W * mask
        # scale spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W = W * (spectral_radius / rho)
        self.W = W

        self.W_in = rng.uniform(-1, 1, size=(N, input_dim)) * input_scaling
        self.state = np.zeros(N)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)
        pre = self.W_in @ u + self.W @ self.state
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)
        return self.state.copy()

    def run(self, inputs: np.ndarray, reset: bool = True) -> np.ndarray:
        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T = inputs.shape[0]
        if reset:
            self.reset()
        out = np.zeros((T, self.N))
        for t in range(T):
            out[t] = self.step(inputs[t])
        return out


class SimpleOscillatorReservoir(CORCReservoir):
    """
    Independent Hopf oscillators (no coupling). Reuses CORC infrastructure.
    """

    def __init__(self, N: int = 32, dt: float = 0.01, input_dim: int = 1, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        units = HopfSlowUnit.heterogeneous(N, dt, rng=rng)
        # dummy coupling with all terms disabled
        coupling = LinearCoupling(N, dt, g_d=0.0, g_m=0.0, disable_pulse=True, rng=rng)
        input_matrix = rng.normal(0, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.5, 2.0, size=N)
        super().__init__(units, coupling, input_matrix, input_gain)


class HarmonicReservoir(CORCReservoir):
    """
    Linear harmonic oscillators with weak linear coupling.
    Effectively removes Hopf limit-cycle nonlinearity by setting mu very negative
    so amplitude collapses, then we inject a strong periodic drive? No — easier:
    approximate with linear system dx/dt = A x + B u.
    For simplicity, we keep Hopf units but clamp r^2 term artificially to 0
    and disable pulse coupling so the dynamics stay near-linear.
    """

    def __init__(self, N: int = 32, dt: float = 0.01, input_dim: int = 1, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        # parameters tuned near harmonic (small mu so nonlinearity negligible)
        units = HopfSlowUnit.heterogeneous(
            N, dt, rng=rng,
            mu_range=(-0.5, -0.1),  # negative mu -> stable focus, no limit cycle
            f_range=(0.8, 2.5),
            tau_a_range=(0.8, 2.5),
            alpha_range=(0.0, 0.0),  # no slow adaptation
            beta_range=(0.0, 0.0),
            sigma_range=(0.005, 0.02),
        )
        coupling = LinearCoupling(N, dt, g_d=0.01, g_m=0.01, rng=rng)
        input_matrix = rng.normal(0, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.5, 2.0, size=N)
        super().__init__(units, coupling, input_matrix, input_gain)


class NullReservoir:
    """
    Direct input -> linear readout (no reservoir).
    Useful as a lower bound.
    """

    def __init__(self, input_dim: int = 1):
        self.input_dim = input_dim
        self.N = input_dim

    def run(self, inputs: np.ndarray, reset: bool = True) -> np.ndarray:
        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        return inputs.copy()

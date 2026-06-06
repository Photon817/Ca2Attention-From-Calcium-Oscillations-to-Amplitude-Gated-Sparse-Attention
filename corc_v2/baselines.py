"""
baselines.py — CORC v2
======================
Four baselines:
  1. Standard ESN (N=64/128, tanh, optimized spectral radius)
  2. Hopf v1 (original Hopf+slow adaptation, no pulse coupling, same readout)
  3. No-pulse CEU (diffusion-only, g_p=0)
  4. Linear baseline (input delay line + Ridge)
"""

from __future__ import annotations
import numpy as np
from typing import Optional


class ESN:
    """Standard Echo State Network with tanh activation."""

    def __init__(
        self,
        N: int,
        input_dim: int = 1,
        leak: float = 0.3,
        spectral_radius: float = 0.95,
        sparsity: float = 0.15,
        input_scaling: float = 0.5,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.input_dim = input_dim
        self.leak = leak
        self.dt = dt
        rng = np.random.default_rng(seed)

        W = rng.uniform(-1, 1, size=(N, N))
        mask = rng.random((N, N)) < sparsity
        W = W * mask
        rho = np.max(np.abs(np.linalg.eigvals(W)))
        if rho > 0:
            W = W * (spectral_radius / rho)
        self.W = W
        self.W_in = rng.uniform(-1, 1, size=(N, input_dim)) * input_scaling
        self.state = np.zeros(N)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(np.squeeze(u))
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


class HopfV1:
    """Original Hopf+slow-adaptation reservoir (v1), imported for comparison."""

    def __init__(self, N: int = 32, dt: float = 0.01, input_dim: int = 1, seed: Optional[int] = None):
        from corc_blackbox.units import HopfSlowUnit
        from corc_blackbox.coupling import Coupling
        from corc_blackbox.reservoir import CORCReservoir
        rng = np.random.default_rng(seed)
        units = HopfSlowUnit.heterogeneous(N, dt, rng=rng)
        coupling = Coupling(N, dt, rng=rng)
        input_matrix = rng.normal(0, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.5, 2.0, size=N)
        self._reservoir = CORCReservoir(units, coupling, input_matrix, input_gain)
        self.N = N

    def run(self, inputs: np.ndarray, reset: bool = True) -> tuple:
        return self._reservoir.run(inputs, reset=reset)


class CEUNoPulse:
    """CEU reservoir without pulse coupling (diffusion only, g_p=0)."""

    def __init__(self, N: int = 32, dt: float = 0.01, input_dim: int = 1, seed: Optional[int] = None):
        from corc_v2.units import CalciumExcitableUnit
        from corc_v2.coupling import PulseCoupling
        from corc_v2.reservoir import CEUReservoir
        rng = np.random.default_rng(seed)
        units = CalciumExcitableUnit.heterogeneous(N, dt, rng=rng)
        coupling = PulseCoupling(N, dt, g_p=0.0, g_d=0.015, rng=rng)
        input_matrix = rng.uniform(0.1, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.3, 1.5, size=N)
        self._reservoir = CEUReservoir(units, coupling, input_matrix, input_gain)
        self.N = N

    def run(self, inputs: np.ndarray, reset: bool = True) -> tuple:
        return self._reservoir.run(inputs, reset=reset)


class LinearBaseline:
    """Input delay line + Ridge (no reservoir)."""

    def __init__(self, input_dim: int = 1, delay_taps: int = 10, delay_step: int = 3):
        self.input_dim = input_dim
        self.delay_taps = delay_taps
        self.delay_step = delay_step
        self.N = input_dim * delay_taps

    def run(self, inputs: np.ndarray, reset: bool = True) -> np.ndarray:
        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T, D = inputs.shape
        out = []
        for lag in range(self.delay_taps):
            shift = lag * self.delay_step
            if shift == 0:
                out.append(inputs)
            else:
                padded = np.zeros_like(inputs)
                padded[:-shift] = inputs[shift:]
                out.append(padded)
        return np.concatenate(out, axis=1)
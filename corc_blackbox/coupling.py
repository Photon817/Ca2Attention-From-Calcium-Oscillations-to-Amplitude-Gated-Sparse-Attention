"""
coupling.py
-----------
Diffusion + mean-field + pulse (spike-response) coupling, with optional
amplitude-dependent scaling (CLAUDE.md §4).

All methods accept (N,) state vectors and return (N,) coupling arrays.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class Coupling:
    """
    C_i = C_diff + C_mf + C_pulse
    """

    def __init__(
        self,
        N: int,
        dt: float,
        g_d: float = 0.01,
        g_m: float = 0.01,
        g_p: float = 0.08,
        p_conn: float = 0.15,
        eta: float = 0.4,
        tau_p: float = 0.08,
        theta: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        disable_pulse: bool = False,
        disable_diffusion: bool = False,
        disable_meanfield: bool = False,
    ):
        self.N = N
        self.dt = dt
        self.g_d = g_d
        self.g_m = g_m
        self.g_p = g_p
        self.p_conn = p_conn
        self.eta = eta
        self.tau_p = tau_p
        self.theta = theta
        self.rng = rng if rng is not None else np.random.default_rng()

        self._disable_pulse = disable_pulse
        self._disable_diffusion = disable_diffusion
        self._disable_meanfield = disable_meanfield

        # sparse random diffusion weight matrix (asymmetric allowed)
        self.W = (self.rng.random((N, N)) < p_conn).astype(float)
        np.fill_diagonal(self.W, 0.0)
        # normalise row-wise loosely so that sum_j W_ij ~ p_conn*N on average
        row_sums = self.W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.W = self.W / row_sums

        # pulse coupling uses same topology (could be different in extensions)
        self.W_p = self.W.copy()

        # pulse trace variable s_j(t)
        self.s = np.zeros(N)
        # track previous amplitude for threshold crossing
        self._r_prev = np.zeros(N)
        self._crossed = np.zeros(N, dtype=bool)

    def compute(
        self,
        x: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (C_x, C_y).
        """
        cx = np.zeros(self.N)
        cy = np.zeros(self.N)

        if not self._disable_diffusion:
            cx += self._diffusion(x, r)
            cy += self._diffusion(y, r)

        if not self._disable_meanfield:
            cx += self._meanfield(x)
            cy += self._meanfield(y)

        if not self._disable_pulse:
            cx += self._pulse(r)
            # pulse is isotropic (affects x channel only by convention here,
            # or optionally both; we keep it scalar-like on x for simplicity)
            # To keep symmetry, also add to y:
            cy += self._pulse(r) * 0.0  # placeholder: no y-pulse in original design

        return cx, cy

    def _diffusion(self, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        # C_i = g_d * sum_j W_ij * (z_j - z_i) * (1 + eta * r_j)
        amp_factor = 1.0 + self.eta * r
        diff = self.W * (z[:, None] - z[None, :])  # (N,N)
        diff = diff * amp_factor[None, :]  # scale by source amplitude
        return self.g_d * diff.sum(axis=1)

    def _meanfield(self, z: np.ndarray) -> np.ndarray:
        z_mean = z.mean()
        return self.g_m * (z_mean - z)

    def _pulse(self, r: np.ndarray) -> np.ndarray:
        # Detect up-crossings of threshold
        crossed = (r >= self.theta) & (self._r_prev < self.theta)
        # Decay existing trace first, then add new spikes
        self.s -= self.s * (self.dt / self.tau_p)
        self.s += crossed.astype(float)  # add delta spikes
        self.s = np.maximum(self.s, 0.0)
        self._r_prev = r.copy()
        # Drive to other nodes
        return self.g_p * (self.W_p @ self.s)

    def reset(self) -> None:
        self.s[:] = 0.0
        self._r_prev[:] = 0.0
        self._crossed[:] = False


class LinearCoupling(Coupling):
    """Placeholder linear/harmonic version for ablation: removes Hopf nonlinearity
    from coupling by setting eta=0 and using only diffusion/mean-field."""

    def __init__(self, N: int, dt: float, g_d: float = 0.01, g_m: float = 0.01, **kw):
        super().__init__(N, dt, g_d=g_d, g_m=g_m, g_p=0.0, eta=0.0, **kw)

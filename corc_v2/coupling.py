"""
coupling.py — CORC v2
=====================
Pulse coupling + weak diffusion, supporting critical-state dynamics.

- Event detection: c_i crosses theta_event upward
- Pulse trace: ds_pulse/dt = -s_pulse/tau_p + sum delta(t - t_k)
- Pulse coupling: C_i = g_p * sum_j W_ij * s_pulse_j * (1 + eta * c_j)
- Weak diffusion: C_i += g_d * sum_j W_ij * (c_j - c_i)
- Connection matrix: sparse random (p=0.15), row-normalized, no self-loops
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class PulseCoupling:
    """
    Event-triggered pulse coupling with weak diffusion.

    C_i = C_pulse_i + C_diff_i

    Parameters
    ----------
    N : int
    dt : float
    g_p : float
        Pulse coupling strength.
    g_d : float
        Diffusion coupling strength.
    p_conn : float
        Connection probability for sparse random matrix.
    eta : float
        Amplitude-dependent modulation factor.
    tau_p : float
        Pulse trace decay time constant.
    theta_event : float
        Threshold for event triggering (c_i crossing upward).
    rng : np.random.Generator, optional
    """

    def __init__(
        self,
        N: int,
        dt: float,
        g_p: float = 0.25,
        g_d: float = 0.015,
        p_conn: float = 0.15,
        eta: float = 0.4,
        tau_p: float = 0.08,
        theta_event: float = 0.6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.N = N
        self.dt = dt
        self.g_p = g_p
        self.g_d = g_d
        self.p_conn = p_conn
        self.eta = eta
        self.tau_p = tau_p
        self.theta_event = theta_event
        self.rng = rng if rng is not None else np.random.default_rng()

        # Sparse random connection matrix (same topology for pulse and diffusion)
        W = (self.rng.random((N, N)) < p_conn).astype(float)
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.W = W / row_sums  # normalized rows

        # Pulse trace variable
        self.s_pulse = np.zeros(N, dtype=np.float32)
        self._c_prev = np.zeros(N, dtype=np.float32)  # for crossing detection

    def compute(self, c: np.ndarray) -> np.ndarray:
        """
        Compute total coupling contribution C_i.

        Parameters
        ----------
        c : (N,)  current cytosolic calcium

        Returns
        -------
        coupling : (N,)  total coupling to add to dc/dt
        """
        coupling = np.zeros(self.N)

        # 1. Weak diffusion
        if self.g_d > 0:
            diff = self.W @ c - c  # sum_j W_ij * (c_j - c_i)
            coupling += self.g_d * diff

        # 2. Pulse coupling with event detection
        if self.g_p > 0:
            # Detect upward crossings
            crossed = (c >= self.theta_event) & (self._c_prev < self.theta_event)
            # Decay existing traces
            self.s_pulse -= self.s_pulse * (self.dt / self.tau_p)
            # Add new spikes
            self.s_pulse += crossed.astype(float)
            self.s_pulse = np.maximum(self.s_pulse, 0.0)
            # Amplitude-dependent modulation
            amp_mod = 1.0 + self.eta * c
            coupling += self.g_p * (self.W @ (self.s_pulse * amp_mod))

        self._c_prev = c.copy()
        return coupling

    def reset(self) -> None:
        self.s_pulse[:] = 0.0
        self._c_prev[:] = 0.0

    @property
    def pulse_trace(self) -> np.ndarray:
        return self.s_pulse.copy()
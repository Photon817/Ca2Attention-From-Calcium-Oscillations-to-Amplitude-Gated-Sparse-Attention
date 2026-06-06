"""
reservoir.py
------------
CORC network integration: units + coupling + input projection + state readout.
Matches CLAUDE.md §5 and §6.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .units import HopfSlowUnit
from .coupling import Coupling


@dataclass
class CORCState:
    x: np.ndarray
    y: np.ndarray
    a: np.ndarray
    r: np.ndarray
    phi: np.ndarray
    s: np.ndarray  # pulse trace


class CORCReservoir:
    """
    Reservoir composed of HopfSlowUnit nodes with Coupling.
    """

    def __init__(
        self,
        units: HopfSlowUnit,
        coupling: Coupling,
        input_matrix: Optional[np.ndarray] = None,
        input_gain: Optional[np.ndarray] = None,
        freq_mod_gain: Optional[np.ndarray] = None,
        warmup_steps: int = 500,
    ):
        self.units = units
        self.coupling = coupling
        self.N = units.N
        self.dt = units.dt
        self.warmup_steps = warmup_steps

        if input_matrix is None:
            # default: 1-D input projected randomly to all nodes
            self.input_matrix = np.ones((self.N, 1))
        else:
            self.input_matrix = np.asarray(input_matrix, dtype=float)
        self.input_dim = self.input_matrix.shape[1]

        if input_gain is None:
            self.input_gain = np.ones(self.N)
        else:
            self.input_gain = np.asarray(input_gain, dtype=float)

        self.freq_mod_gain = freq_mod_gain  # (N,) or None

    @classmethod
    def create_default(
        cls,
        N: int = 32,
        dt: float = 0.01,
        input_dim: int = 1,
        seed: Optional[int] = None,
        heterogeneous: bool = True,
        **unit_kwargs,
    ) -> "CORCReservoir":
        rng = np.random.default_rng(seed)
        if heterogeneous:
            units = HopfSlowUnit.heterogeneous(N, dt, rng=rng, **unit_kwargs)
        else:
            units = HopfSlowUnit.homogeneous(N, dt, rng=rng, **unit_kwargs)
        coupling = Coupling(N, dt, rng=rng)
        # random projection matrix
        input_matrix = rng.normal(0, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.5, 2.0, size=N)
        return cls(units, coupling, input_matrix, input_gain)

    def reset(self, seed: Optional[int] = None) -> None:
        self.units.reset_state(seed)
        self.coupling.reset()

    def _warmup(self, steps: int) -> None:
        for _ in range(steps):
            u = np.zeros(self.input_dim)
            self.step(u)

    def step(self, u: np.ndarray) -> CORCState:
        """
        u: (input_dim,) or scalar
        """
        u = np.atleast_1d(u)
        projected = self.input_matrix @ u  # (N,)
        input_x = self.input_gain * projected

        r = self.units.r
        cx, cy = self.coupling.compute(self.units.x, self.units.y, r)

        freq_mod = None
        if self.freq_mod_gain is not None:
            freq_mod = self.freq_mod_gain * projected

        self.units.step(input_x, cx, cy, freq_mod=freq_mod)

        return CORCState(
            x=self.units.x.copy(),
            y=self.units.y.copy(),
            a=self.units.a.copy(),
            r=self.units.r.copy(),
            phi=self.units.phi.copy(),
            s=self.coupling.s.copy(),
        )

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[List[CORCState], np.ndarray]:
        """
        inputs: (T, input_dim) or (T,)
        Returns (list_of_states, event_count_array)
        event_count_array: (T, N) spike counts per step per node.
        """
        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T = inputs.shape[0]

        if reset:
            self.reset(seed)
            self._warmup(self.warmup_steps)

        states: List[CORCState] = []
        event_counts = np.zeros((T, self.N), dtype=int)
        r_prev = self.units.r.copy()

        for t in range(T):
            st = self.step(inputs[t])
            states.append(st)
            # event detection (threshold crossing up)
            crossed = (st.r >= self.coupling.theta) & (r_prev < self.coupling.theta)
            event_counts[t] = crossed.astype(int)
            r_prev = st.r.copy()

        return states, event_counts

    def state_matrix(
        self,
        states: List[CORCState],
        features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Stack selected features into (T, N*len(features)) matrix.
        features in {'x','y','a','r','phi','s'}.
        """
        if features is None:
            features = ["x", "y", "a", "r", "s"]
        mats = []
        for f in features:
            vec = np.stack([getattr(s, f) for s in states], axis=0)  # (T,N)
            mats.append(vec)
        return np.concatenate(mats, axis=1)  # (T, N*F)

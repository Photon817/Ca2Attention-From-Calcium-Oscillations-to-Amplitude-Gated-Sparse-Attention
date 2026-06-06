"""
reservoir.py — CORC v2
======================
CEU network integration: units + coupling + input projection + time-scale elasticity.

Key feature: time-scale elasticity via scaling factor lambda_t.
For fast tasks like NARMA, lambda_t=0.1 speeds up oscillations 10x to match
the task's temporal statistics.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from .units import CalciumExcitableUnit
from .coupling import PulseCoupling


@dataclass
class CEUState:
    c: np.ndarray   # cytosolic calcium
    s: np.ndarray   # ER store
    a: np.ndarray   # adaptation
    pulse: np.ndarray  # pulse traces


class CEUReservoir:
    """
    Reservoir composed of CalciumExcitableUnit nodes with PulseCoupling.

    Supports time-scale elasticity: rescale dynamics to match task timescale.
    """

    def __init__(
        self,
        units: CalciumExcitableUnit,
        coupling: PulseCoupling,
        input_matrix: Optional[np.ndarray] = None,
        input_gain: Optional[np.ndarray] = None,
        warmup_steps: int = 500,
    ):
        self.units = units
        self.coupling = coupling
        self.N = units.N
        self.dt = units.dt
        self.warmup_steps = warmup_steps

        if input_matrix is None:
            self.input_matrix = np.ones((self.N, 1))
        else:
            self.input_matrix = np.asarray(input_matrix, dtype=float)
        self.input_dim = self.input_matrix.shape[1]

        if input_gain is None:
            self.input_gain = np.ones(self.N) * 2.0
        else:
            self.input_gain = np.asarray(input_gain, dtype=float)

    @classmethod
    def create_default(
        cls,
        N: int = 32,
        dt: float = 0.01,
        input_dim: int = 1,
        seed: Optional[int] = None,
        heterogeneous: bool = True,
        **unit_kwargs,
    ) -> CEUReservoir:
        rng = np.random.default_rng(seed)
        if heterogeneous:
            units = CalciumExcitableUnit.heterogeneous(N, dt, rng=rng, **unit_kwargs)
        else:
            units = CalciumExcitableUnit.homogeneous(N, dt, rng=rng, **unit_kwargs)
        coupling = PulseCoupling(N, dt, rng=rng)
        input_matrix = rng.uniform(0.1, 1.0, size=(N, input_dim))
        input_gain = rng.uniform(0.3, 1.5, size=N)
        return cls(units, coupling, input_matrix, input_gain)

    def reset(self, seed: Optional[int] = None) -> None:
        self.units.reset_state(seed)
        self.coupling.reset()

    def _warmup(self, steps: int) -> None:
        for _ in range(steps):
            u = np.zeros(self.input_dim)
            self.step(u)

    def step(self, u: np.ndarray) -> CEUState:
        u = np.atleast_1d(np.squeeze(u))
        projected = self.input_matrix @ u  # (N,)
        input_c = self.input_gain * projected

        coupling_c = self.coupling.compute(self.units.c)
        self.units.step(input_c, coupling_c)

        return CEUState(
            c=self.units.c.copy(),
            s=self.units.s.copy(),
            a=self.units.a.copy(),
            pulse=self.coupling.pulse_trace.copy(),
        )

    def run(
        self,
        inputs: np.ndarray,
        reset: bool = True,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Run reservoir on input sequence.

        Parameters
        ----------
        inputs : (T,) or (T, input_dim)
        reset : bool

        Returns
        -------
        states : list of CEUState
        event_counts : (T, N) int array of event occurrences per step
        """
        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T = inputs.shape[0]

        if reset:
            self.reset(seed)
            self._warmup(self.warmup_steps)

        states: List[CEUState] = []
        event_counts = np.zeros((T, self.N), dtype=int)
        c_prev = self.units.c.copy()

        for t in range(T):
            st = self.step(inputs[t])
            states.append(st)
            crossed = (st.c >= self.coupling.theta_event) & (c_prev < self.coupling.theta_event)
            event_counts[t] = crossed.astype(int)
            c_prev = st.c.copy()

        return states, event_counts

    def run_with_timescale(
        self,
        inputs: np.ndarray,
        lambda_t: float = 1.0,
        reset: bool = True,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Run with time-scale elasticity.
        lambda_t < 1 speeds up dynamics; lambda_t > 1 slows down.

        Strategy: subsample/interpolate input to effectively change
        the time unit relative to the model's natural timescale.
        """
        if lambda_t == 1.0:
            return self.run(inputs, reset=reset, seed=seed)

        inputs = np.atleast_1d(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, None]
        T_orig = inputs.shape[0]

        # Rescale: stretch/compress input in time
        # lambda_t < 1: fewer steps (speed up) — downsample with interpolation
        T_new = max(10, int(T_orig * lambda_t))
        t_old = np.linspace(0, 1, T_orig)
        t_new = np.linspace(0, 1, T_new)
        inputs_rescaled = np.zeros((T_new, inputs.shape[1]))
        for d in range(inputs.shape[1]):
            inputs_rescaled[:, d] = np.interp(t_new, t_old, inputs[:, d])

        return self.run(inputs_rescaled, reset=reset, seed=seed)

    def state_matrix(
        self,
        states: List[CEUState],
        features: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Stack selected features into (T, N*len(features)) matrix."""
        if features is None:
            features = ["c", "s", "a", "pulse"]
        mats = []
        for f in features:
            vec = np.stack([getattr(s, f) for s in states], axis=0)
            mats.append(vec)
        return np.concatenate(mats, axis=1)
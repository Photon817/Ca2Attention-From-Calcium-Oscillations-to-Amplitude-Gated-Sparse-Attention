"""
tasks.py
--------
Four task families (CLAUDE.md §7):
A. Rhythm / pattern binary classification
B. Short-term memory capacity (MC)
C. NARMA-10
D. Event-driven temporal classification (temporal XOR, switching rhythm)

All tasks expose:
    generate(n_trials, T, dt) -> inputs, targets
    score(predictions, targets) -> metric_dict
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass


def _make_input(T: int, dt: float, dim: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=(T, dim))


# ---------------------------------------------------------------------------
# A. Rhythm / pattern classification
# ---------------------------------------------------------------------------

@dataclass
class RhythmClassification:
    """
    Binary classification of input rhythms.
    Two classes:
        0: low-frequency sinusoid (~0.5 Hz)
        1: high-frequency sinusoid (~2.0 Hz)
    Optionally also vary phase offset / duty cycle (pulse trains).
    """

    dt: float = 0.01
    f0: float = 0.5
    f1: float = 2.0
    A: float = 0.5
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200, T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs = []
        targets = []
        for _ in range(n_trials):
            label = rng.integers(0, 2)
            t = np.arange(T) * self.dt
            if label == 0:
                u = self.A * np.sin(2 * np.pi * self.f0 * t)
            else:
                u = self.A * np.sin(2 * np.pi * self.f1 * t)
            # add small noise
            u += rng.normal(0, 0.02, size=T)
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)  # (trials, T, 1), (trials,)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        acc = np.mean(y_pred == y_true)
        return {"accuracy": float(acc)}


# ---------------------------------------------------------------------------
# B. Short-term memory capacity (MC)
# ---------------------------------------------------------------------------

@dataclass
class MemoryCapacity:
    """
    Memory function MC_k = corr^2(y(t), u(t-k)) for varying delay k.
    Input is i.i.d. from Uniform(-1,1).
    """

    dt: float = 0.01
    max_lag: int = 20
    rng: Optional[np.random.Generator] = None

    def generate(self, n_samples: int = 5000) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        u = rng.uniform(-1.0, 1.0, size=n_samples)
        targets = {}
        for k in range(1, self.max_lag + 1):
            targets[k] = np.roll(u, k)
            targets[k][:k] = 0.0  # padding
        return u[:, None], targets  # (n_samples, 1), dict[k] -> (n_samples,)

    @staticmethod
    def compute_mc(reservoir_states: np.ndarray, u: np.ndarray, max_lag: int = 20) -> Dict[str, float]:
        """
        reservoir_states: (T, D)
        u: (T,) original input
        Fits ridge regression for each delay k and returns MC_k.
        """
        from sklearn.linear_model import Ridge
        T, D = reservoir_states.shape
        # discard transient
        skip = 100
        X = reservoir_states[skip:]
        mc_vals = {}
        total_mc = 0.0
        for k in range(1, max_lag + 1):
            y = np.roll(u, k)[skip:]
            model = Ridge(alpha=1e-6)
            model.fit(X, y)
            y_pred = model.predict(X)
            # normalised correlation squared
            num = np.corrcoef(y, y_pred)[0, 1] ** 2
            mc_vals[k] = float(num)
            total_mc += mc_vals[k]
        mc_vals["total"] = float(total_mc)
        return mc_vals


# ---------------------------------------------------------------------------
# C. NARMA-10
# ---------------------------------------------------------------------------

@dataclass
class NARMA10:
    """
    Classic NARMA-10 benchmark.
    y(t+1) = 0.3*y(t) + 0.05*y(t)*sum_{i=0}^{9} y(t-i) + 1.5*u(t-9)*u(t) + 0.1
    Input u(t) ~ Uniform(0, 0.5).
    """

    dt: float = 0.01
    rng: Optional[np.random.Generator] = None

    def generate(self, T: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        u = rng.uniform(0.0, 0.5, size=T)
        y = np.zeros(T)
        for t in range(T):
            if t == 0:
                y[t] = 0.1
            else:
                window = y[max(0, t - 9):t + 1]
                term2 = 0.05 * y[t - 1] * window.sum()
                term3 = 1.5 * u[max(0, t - 10)] * u[max(0, t - 1)] if t >= 10 else 0.0
                y[t] = 0.3 * y[t - 1] + term2 + term3 + 0.1
        return u[:, None], y

    def score(self, y_pred: np.ndarray, y_true: np.ndarray, skip: int = 100) -> Dict[str, float]:
        yp = y_pred[skip:]
        yt = y_true[skip:]
        mse = np.mean((yp - yt) ** 2)
        nmse = mse / (np.var(yt) + 1e-12)
        return {"mse": float(mse), "nmse": float(nmse)}


# ---------------------------------------------------------------------------
# D. Event-driven temporal classification
# ---------------------------------------------------------------------------

@dataclass
class TemporalXOR:
    """
    Temporal XOR: two pulses at t1 and t2; label = XOR of their polarities.
    """

    dt: float = 0.01
    T: int = 500
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs = []
        targets = []
        for _ in range(n_trials):
            u = np.zeros(self.T)
            p1 = rng.integers(50, 150)
            p2 = rng.integers(250, 350)
            pol1 = rng.choice([-1.0, 1.0])
            pol2 = rng.choice([-1.0, 1.0])
            u[p1] = pol1
            u[p2] = pol2
            # slight width
            u = np.convolve(u, np.ones(5) / 5, mode="same")
            label = int((pol1 > 0) ^ (pol2 > 0))
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


@dataclass
class SwitchingRhythm:
    """
    Sequence switches from f_low to f_high at midpoint; classify which half is which.
    Simplified: output label corresponds to majority frequency.
    """

    dt: float = 0.01
    f_low: float = 0.5
    f_high: float = 2.0
    T: int = 1000
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs = []
        targets = []
        for _ in range(n_trials):
            t = np.arange(self.T) * self.dt
            label = rng.integers(0, 2)
            if label == 0:
                u = np.sin(2 * np.pi * self.f_low * t)
            else:
                u = np.sin(2 * np.pi * self.f_high * t)
            u += rng.normal(0, 0.02, size=self.T)
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ---------------------------------------------------------------------------
# Helper: train/test split for reservoir outputs
# ---------------------------------------------------------------------------

def train_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "regression",
    alpha: float = 1e-4,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Train ridge classifier / regressor on reservoir states.
    Returns predictions, metrics.
    """
    if task_type == "regression":
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = float(np.mean((y_pred - y_test) ** 2))
        nmse = mse / (np.var(y_test) + 1e-12)
        return y_pred, {"mse": mse, "nmse": nmse}
    else:
        from sklearn.linear_model import RidgeClassifier
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = float(np.mean(y_pred == y_test))
        return y_pred, {"accuracy": acc}

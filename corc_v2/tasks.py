"""
tasks.py — CORC v2
==================
Task families:
  A: Rhythm classification (1Hz vs 3Hz sine)
  B: Complex rhythm classification (duty cycle, modulation patterns)
  C: Temporal XOR
  D: Memory capacity (target >15)
  E: NARMA-10 (target NMSE <2.0, optimal <0.5)
  F: NARMA-20 (optional)
  G: Hard rhythm – close frequencies (2.0 vs 2.3 Hz)
  H: Phase-noise pattern recognition (different noise levels)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# ============================================================================
# A: Rhythm Classification
# ============================================================================

@dataclass
class RhythmClassification:
    dt: float = 0.01
    f0: float = 1.0
    f1: float = 3.0
    amp: float = 0.5
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200, T: int = 800) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs, targets = [], []
        for _ in range(n_trials):
            label = rng.integers(0, 2)
            t = np.arange(T) * self.dt
            freq = self.f0 if label == 0 else self.f1
            u = self.amp * np.sin(2 * np.pi * freq * t)
            u += rng.normal(0, 0.02, size=T)
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ============================================================================
# B: Complex Rhythm Classification
# ============================================================================

@dataclass
class ComplexRhythmClassification:
    """
    Classify patterns with different duty cycles and modulation.
    Class 0: sine with 20% duty cycle (pulse train)
    Class 1: sine with 50% duty cycle (standard sine)
    Class 2: sine with amplitude modulation (1Hz carrier, 0.1Hz modulation)
    """
    dt: float = 0.01
    n_classes: int = 3
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 300, T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs, targets = [], []
        t = np.arange(T) * self.dt
        for _ in range(n_trials):
            label = rng.integers(0, self.n_classes)
            if label == 0:
                u = np.where(np.sin(2*np.pi*1.0*t) > 0.6, 0.5, -0.5)
            elif label == 1:
                u = 0.5 * np.sin(2 * np.pi * 1.5 * t)
            else:
                mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
                u = 0.5 * mod * np.sin(2 * np.pi * 1.0 * t)
            u += rng.normal(0, 0.02, size=T)
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ============================================================================
# C: Temporal XOR
# ============================================================================

@dataclass
class TemporalXOR:
    dt: float = 0.01
    T: int = 600
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs, targets = [], []
        for _ in range(n_trials):
            u = np.zeros(self.T)
            p1 = rng.integers(60, 180)
            p2 = rng.integers(300, 420)
            pol1 = rng.choice([-1.0, 1.0])
            pol2 = rng.choice([-1.0, 1.0])
            u[p1] = pol1
            u[p2] = pol2
            u = np.convolve(u, np.ones(5)/5, mode='same')
            label = int((pol1 > 0) ^ (pol2 > 0))
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ============================================================================
# D: Memory Capacity
# ============================================================================

@dataclass
class MemoryCapacity:
    dt: float = 0.01
    max_lag: int = 30
    rng: Optional[np.random.Generator] = None

    def generate(self, n_samples: int = 6000) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        u = rng.uniform(-1.0, 1.0, size=n_samples)
        targets = {}
        for k in range(1, self.max_lag + 1):
            targets[k] = np.roll(u, k)
            targets[k][:k] = 0.0
        return u[:, None], targets

    @staticmethod
    def compute_mc(
        reservoir_states: np.ndarray,
        u: np.ndarray,
        max_lag: int = 30,
    ) -> Dict[str, float]:
        from sklearn.linear_model import Ridge
        T, D = reservoir_states.shape
        skip = 100
        X = reservoir_states[skip:]
        mc_vals = {}
        total_mc = 0.0
        for k in range(1, max_lag + 1):
            y = np.roll(u, k)[skip:]
            model = Ridge(alpha=1e-6)
            model.fit(X, y)
            y_pred = model.predict(X)
            num = np.corrcoef(y, y_pred)[0, 1] ** 2
            mc_vals[k] = float(num)
            total_mc += mc_vals[k]
        mc_vals["total"] = float(total_mc)
        return mc_vals


# ============================================================================
# E: NARMA-10
# ============================================================================

@dataclass
class NARMA10:
    dt: float = 0.01
    rng: Optional[np.random.Generator] = None

    def generate(self, T: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        u = rng.uniform(0.0, 0.5, size=T)
        y = np.zeros(T)
        for t in range(T):
            if t < 10:
                y[t] = 0.1
                continue
            window_sum = y[max(0, t-10):t].sum()
            y[t] = (0.3 * y[t-1] +
                    0.05 * y[t-1] * window_sum +
                    1.5 * u[t-10] * u[t-1] +
                    0.1)
            # clamp to prevent overflow
            y[t] = np.clip(y[t], -5.0, 5.0)
        return u[:, None], y

    def score(self, y_pred: np.ndarray, y_true: np.ndarray, skip: int = 100) -> Dict[str, float]:
        yp = y_pred[skip:]
        yt = y_true[skip:]
        mse = float(np.mean((yp - yt) ** 2))
        nmse = mse / (np.var(yt) + 1e-12)
        return {"mse": mse, "nmse": nmse}


# ============================================================================
# F: NARMA-20 (optional)
# ============================================================================

@dataclass
class NARMA20:
    dt: float = 0.01
    rng: Optional[np.random.Generator] = None

    def generate(self, T: int = 6000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        u = rng.uniform(0.0, 0.5, size=T)
        y = np.zeros(T)
        for t in range(T):
            if t < 20:
                y[t] = 0.1
                continue
            window_sum = y[max(0, t-20):t].sum()
            y[t] = (np.tanh(0.3 * y[t-1] +
                            0.05 * y[t-1] * window_sum +
                            1.5 * u[t-20] * u[t-1] +
                            0.01))
            y[t] = np.clip(y[t], -3.0, 3.0)
        return u[:, None], y

    def score(self, y_pred: np.ndarray, y_true: np.ndarray, skip: int = 100) -> Dict[str, float]:
        yp = y_pred[skip:]
        yt = y_true[skip:]
        mse = float(np.mean((yp - yt) ** 2))
        nmse = mse / (np.var(yt) + 1e-12)
        return {"mse": mse, "nmse": nmse}


# ============================================================================
# G: Hard Rhythm — close frequencies
# ============================================================================

@dataclass
class HardRhythmClassification:
    """
    Close-frequency rhythm classification: 2.0 vs 2.3 Hz.
    Designed to break the ceiling effect of easy rhythm tasks.
    """
    dt: float = 0.01
    f0: float = 2.0
    f1: float = 2.3
    amp: float = 0.5
    noise_std: float = 0.05
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 200, T: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs, targets = [], []
        for _ in range(n_trials):
            label = rng.integers(0, 2)
            t = np.arange(T) * self.dt
            freq = self.f0 if label == 0 else self.f1
            # Random phase offset to prevent phase-locking
            phase = rng.uniform(0, 2 * np.pi)
            u = self.amp * np.sin(2 * np.pi * freq * t + phase)
            u += rng.normal(0, self.noise_std, size=T)
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ============================================================================
# H: Phase-Noise Pattern Recognition
# ============================================================================

@dataclass
class PhaseNoiseClassification:
    """
    Classify 2 Hz sine with different phase noise levels.
    Class 0: low phase noise (sigma_phase=0.05 rad)
    Class 1: medium phase noise (sigma_phase=0.20 rad)
    Class 2: high phase noise (sigma_phase=0.50 rad)

    Phase noise is introduced by perturbing the instantaneous phase
    with a Wiener process (integrated Gaussian noise).
    """
    dt: float = 0.01
    n_classes: int = 3
    amp: float = 0.5
    noise_levels: Tuple[float, float, float] = (0.05, 0.20, 0.50)
    base_freq: float = 2.0
    rng: Optional[np.random.Generator] = None

    def generate(self, n_trials: int = 300, T: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        rng = self.rng if self.rng is not None else np.random.default_rng()
        inputs, targets = [], []
        t = np.arange(T) * self.dt
        for _ in range(n_trials):
            label = rng.integers(0, self.n_classes)
            sigma_phase = self.noise_levels[label]
            # Wiener phase noise
            dW = rng.normal(0, sigma_phase * np.sqrt(self.dt), size=T)
            phase_noise = np.cumsum(dW)
            phase = 2 * np.pi * self.base_freq * t + phase_noise
            u = self.amp * np.sin(phase)
            u += rng.normal(0, 0.01, size=T)  # small observation noise
            inputs.append(u[:, None])
            targets.append(label)
        return np.stack(inputs, axis=0), np.array(targets)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        return {"accuracy": float(np.mean(y_pred == y_true))}


# ============================================================================
# Helper: train readout
# ============================================================================

def train_readout(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str = "regression",
    alpha: float = 1e-4,
    use_poly: bool = False,
    use_mlp: bool = False,
    z_score: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Train readout on reservoir states.

    Parameters
    ----------
    task_type : 'regression' or 'classification'
    use_poly : degree-2 polynomial expansion
    use_mlp : lightweight MLP (single hidden layer, 64 units)
    z_score : standardize features
    """
    # Filter dead dimensions (zero variance)
    train_std = X_train.std(axis=0)
    alive = train_std > 1e-10
    X_train = X_train[:, alive]
    X_test = X_test[:, alive]

    if z_score:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if use_poly:
        from .observables import polynomial_expand
        X_train = polynomial_expand(X_train, degree=2)
        X_test = polynomial_expand(X_test, degree=2)

    if use_mlp:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            if task_type == "regression":
                model = MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation='relu',
                    alpha=alpha,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                )
            else:
                model = MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation='relu',
                    alpha=alpha,
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        except Exception:
            use_mlp = False  # fallback

    if not use_mlp:
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

    # MLP path
    if task_type == "regression":
        mse = float(np.mean((y_pred - y_test) ** 2))
        nmse = mse / (np.var(y_test) + 1e-12)
        return y_pred, {"mse": mse, "nmse": nmse}
    else:
        acc = float(np.mean(y_pred == y_test))
        return y_pred, {"accuracy": acc}
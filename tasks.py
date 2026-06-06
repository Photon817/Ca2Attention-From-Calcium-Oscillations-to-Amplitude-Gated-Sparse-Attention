"""
tasks.py
========
Benchmark tasks and baseline models for the NPU oscillatory reservoir.

Task A – Temporal binary classification (rhythm discrimination)
Task B – NARMA-10 memory benchmark

Baselines
---------
- ESN : traditional Echo State Network with same reservoir size.
- Raw : calcium signals fed directly into a linear read-out (no feature
  extraction).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from typing import Callable, Dict, List, Tuple

from model import (
    ESN,
    FeatureExtractor,
    InputEncoder,
    NPUArray,
    SingleNPU,
    generate_narma10,
    generate_rhythm_classification_trials,
    set_seed,
)

# ---------------------------------------------------------------------------
# Helper: split data into train / validation / test
# ---------------------------------------------------------------------------

def _split(arr: np.ndarray, ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2)):
    n = len(arr)
    i1 = int(n * ratios[0])
    i2 = int(n * (ratios[0] + ratios[1]))
    return arr[:i1], arr[i1:i2], arr[i2:]


# ---------------------------------------------------------------------------
# Task A – Temporal binary classification
# ---------------------------------------------------------------------------

class TaskAClassifier:
    """
    Discriminate between two rhythmic input patterns (low-frequency vs.
    high-frequency modulation) using reservoir features.
    """

    def __init__(
        self,
        reservoir: NPUArray,
        feature_extractor: FeatureExtractor,
        encoder: InputEncoder,
    ):
        self.reservoir = reservoir
        self.fe = feature_extractor
        self.encoder = encoder
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.is_trained = False

    def _trials_to_features(
        self, stimuli: np.ndarray
    ) -> np.ndarray:
        """
        stimuli : (n_trials, n_steps)
        Returns flattened feature vectors, one per trial.
        """
        n_trials = stimuli.shape[0]
        feats_list = []
        for i in range(n_trials):
            self.reservoir.reset()
            s = stimuli[i, :]
            dt = self.reservoir.dt

            def I_func(t: float) -> np.ndarray:
                idx = int(round(t / dt))
                idx = min(idx, len(s) - 1)
                return np.full(self.reservoir.N, s[idx])

            C = self.reservoir.run(duration=len(s) * dt, I_func=I_func)
            feats, _ = self.fe.extract(C)
            if feats.shape[0] == 0:
                feats_list.append(
                    np.concatenate([C.mean(axis=0), C.std(axis=0)])
                )
            else:
                feats_list.append(feats.mean(axis=0))
        return np.vstack(feats_list)

    def train(self, stimuli: np.ndarray, labels: np.ndarray) -> None:
        X = self._trials_to_features(stimuli)
        self.clf.fit(X, labels)
        self.is_trained = True

    def evaluate(self, stimuli: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("Classifier not trained yet.")
        X = self._trials_to_features(stimuli)
        y_pred = self.clf.predict(X)
        y_prob = self.clf.predict_proba(X)[:, 1]

        acc = accuracy_score(labels, y_pred)
        cm = confusion_matrix(labels, y_pred)
        try:
            auc = roc_auc_score(labels, y_prob)
        except ValueError:
            auc = np.nan

        return {"accuracy": acc, "confusion_matrix": cm, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Task A – Baseline using raw calcium signals (no feature extraction)
# ---------------------------------------------------------------------------

class TaskARawBaseline:
    """
    Directly pool raw calcium traces and feed to logistic regression.
    """

    def __init__(
        self,
        reservoir: NPUArray,
        encoder: InputEncoder,
        pool: str = "meanstd",
    ):
        self.reservoir = reservoir
        self.encoder = encoder
        self.pool = pool
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.is_trained = False

    def _trials_to_features(self, stimuli: np.ndarray) -> np.ndarray:
        n_trials = stimuli.shape[0]
        feats_list = []
        for i in range(n_trials):
            self.reservoir.reset()
            s = stimuli[i, :]

            def I_func(t: float) -> np.ndarray:
                idx = int(round(t / self.reservoir.dt))
                idx = min(idx, len(s) - 1)
                return self.encoder.encode_vector(t, s[idx])

            C = self.reservoir.run(duration=len(s) * self.reservoir.dt, I_func=I_func)
            if self.pool == "meanstd":
                feats_list.append(np.concatenate([C.mean(axis=0), C.std(axis=0)]))
            else:
                feats_list.append(C.flatten())
        return np.vstack(feats_list)

    def train(self, stimuli: np.ndarray, labels: np.ndarray) -> None:
        X = self._trials_to_features(stimuli)
        self.clf.fit(X, labels)
        self.is_trained = True

    def evaluate(self, stimuli: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("Baseline not trained yet.")
        X = self._trials_to_features(stimuli)
        y_pred = self.clf.predict(X)
        y_prob = self.clf.predict_proba(X)[:, 1]
        acc = accuracy_score(labels, y_pred)
        cm = confusion_matrix(labels, y_pred)
        try:
            auc = roc_auc_score(labels, y_prob)
        except ValueError:
            auc = np.nan
        return {"accuracy": acc, "confusion_matrix": cm, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Task A – ESN baseline
# ---------------------------------------------------------------------------

class TaskAESNBaseline:
    """
    Echo State Network baseline for rhythm classification.
    """

    def __init__(
        self,
        n_reservoir: int,
        dt: float = 0.01,
        seed: int = 99,
    ):
        self.esn = ESN(
            n_reservoir=n_reservoir,
            spectral_radius=0.95,
            input_scaling=1.5,
            leaking_rate=0.3,
            dt=dt,
            seed=seed,
        )
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.is_trained = False

    def _trials_to_features(self, stimuli: np.ndarray) -> np.ndarray:
        n_trials = stimuli.shape[0]
        feats_list = []
        for i in range(n_trials):
            self.esn.reset()
            s = stimuli[i, :]
            X = self.esn.run(s)
            # Mean-pool reservoir states over time
            feats_list.append(X.mean(axis=0))
        return np.vstack(feats_list)

    def train(self, stimuli: np.ndarray, labels: np.ndarray) -> None:
        X = self._trials_to_features(stimuli)
        self.clf.fit(X, labels)
        self.is_trained = True

    def evaluate(self, stimuli: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("ESN baseline not trained yet.")
        X = self._trials_to_features(stimuli)
        y_pred = self.clf.predict(X)
        y_prob = self.clf.predict_proba(X)[:, 1]
        acc = accuracy_score(labels, y_pred)
        cm = confusion_matrix(labels, y_pred)
        try:
            auc = roc_auc_score(labels, y_prob)
        except ValueError:
            auc = np.nan
        return {"accuracy": acc, "confusion_matrix": cm, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Task B – NARMA-10 prediction
# ---------------------------------------------------------------------------

class TaskBNARMA:
    """
    NARMA-10 next-step prediction via ridge regression read-out.
    """

    def __init__(
        self,
        reservoir: NPUArray,
        feature_extractor: FeatureExtractor,
        encoder: InputEncoder,
        ridge_alpha: float = 1e-4,
    ):
        self.reservoir = reservoir
        self.fe = feature_extractor
        self.encoder = encoder
        self.ridge = Ridge(alpha=ridge_alpha)
        self.is_trained = False

    def _generate_reservoir_states(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run reservoir with input stream u and return time-aligned feature
        vectors and their centre times.
        """
        self.reservoir.reset()
        n_steps = len(u)
        dt = self.reservoir.dt

        def I_func(t: float) -> np.ndarray:
            idx = int(round(t / dt))
            idx = min(idx, n_steps - 1)
            return self.encoder.encode_vector(t, u[idx])

        C = self.reservoir.run(duration=n_steps * dt, I_func=I_func)
        feats, times = self.fe.extract(C)

        if feats.shape[0] == 0:
            # Fallback: use raw C with simple downsampling
            step = max(1, int(round(self.fe.window_size / dt)))
            feats = C[::step, :]
            times = np.arange(len(feats)) * step * dt
        return feats, times

    def train(self, u: np.ndarray, y_target: np.ndarray) -> None:
        """
        u         : input stream (already normalized to [0,1])
        y_target  : target output y(t+1)
        """
        feats, times = self._generate_reservoir_states(u)
        dt = self.reservoir.dt
        # Map window centre times to y_target indices
        indices = np.round(times / dt).astype(int)
        indices = np.clip(indices, 0, len(y_target) - 1)
        # Skip first few windows as transient
        skip = min(5, len(indices) // 10)
        X = feats[skip:]
        y = y_target[indices[skip:]]
        self.ridge.fit(X, y)
        self.is_trained = True

    def predict(self, u: np.ndarray) -> np.ndarray:
        feats, _ = self._generate_reservoir_states(u)
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return self.ridge.predict(feats)

    def evaluate(self, u: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(u)
        # Align lengths
        min_len = min(len(y_pred), len(y_target))
        y_pred = y_pred[:min_len]
        y_target = y_target[:min_len]

        mse = np.mean((y_target - y_pred) ** 2)
        nmse = mse / np.var(y_target)
        # Memory capacity (simplified): correlation between predicted and true
        mc = float(np.corrcoef(y_target, y_pred)[0, 1] ** 2)
        return {"mse": mse, "nmse": nmse, "mc": mc}


# ---------------------------------------------------------------------------
# Task B – Raw baseline (no feature extraction)
# ---------------------------------------------------------------------------

class TaskBRawBaseline:
    """
    Feed raw calcium traces directly to ridge regression for NARMA-10.
    """

    def __init__(
        self,
        reservoir: NPUArray,
        encoder: InputEncoder,
        ridge_alpha: float = 1e-4,
    ):
        self.reservoir = reservoir
        self.encoder = encoder
        self.ridge = Ridge(alpha=ridge_alpha)
        self.is_trained = False

    def _generate_states(self, u: np.ndarray) -> np.ndarray:
        self.reservoir.reset()
        n_steps = len(u)
        dt = self.reservoir.dt

        def I_func(t: float) -> np.ndarray:
            idx = int(round(t / dt))
            idx = min(idx, n_steps - 1)
            return self.encoder.encode_vector(t, u[idx])

        C = self.reservoir.run(duration=n_steps * dt, I_func=I_func)
        return C

    def train(self, u: np.ndarray, y_target: np.ndarray) -> None:
        X = self._generate_states(u)
        skip = 100
        X = X[skip:]
        if len(X) > len(y_target) - skip:
            X = X[: len(y_target) - skip]
        y = y_target[skip : skip + len(X)]
        self.ridge.fit(X, y)
        self.is_trained = True

    def predict(self, u: np.ndarray) -> np.ndarray:
        X = self._generate_states(u)
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return self.ridge.predict(X)

    def evaluate(self, u: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(u)
        min_len = min(len(y_pred), len(y_target))
        y_pred = y_pred[:min_len]
        y_target = y_target[:min_len]
        mse = np.mean((y_target - y_pred) ** 2)
        nmse = mse / np.var(y_target)
        mc = float(np.corrcoef(y_target, y_pred)[0, 1] ** 2)
        return {"mse": mse, "nmse": nmse, "mc": mc}


# ---------------------------------------------------------------------------
# Task B – ESN baseline
# ---------------------------------------------------------------------------

class TaskBESNBaseline:
    """
    Echo State Network baseline for NARMA-10.
    """

    def __init__(
        self,
        n_reservoir: int,
        dt: float = 0.01,
        ridge_alpha: float = 1e-4,
        seed: int = 99,
    ):
        self.esn = ESN(
            n_reservoir=n_reservoir,
            spectral_radius=0.95,
            input_scaling=1.5,
            leaking_rate=0.3,
            dt=dt,
            seed=seed,
        )
        self.ridge = Ridge(alpha=ridge_alpha)
        self.is_trained = False

    def train(self, u: np.ndarray, y_target: np.ndarray) -> None:
        self.esn.reset()
        X = self.esn.run(u)
        skip = 100
        X = X[skip:]
        if len(X) > len(y_target) - skip:
            X = X[: len(y_target) - skip]
        y = y_target[skip : skip + len(X)]
        self.ridge.fit(X, y)
        self.is_trained = True

    def predict(self, u: np.ndarray) -> np.ndarray:
        self.esn.reset()
        X = self.esn.run(u)
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        return self.ridge.predict(X)

    def evaluate(self, u: np.ndarray, y_target: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(u)
        min_len = min(len(y_pred), len(y_target))
        y_pred = y_pred[:min_len]
        y_target = y_target[:min_len]
        mse = np.mean((y_target - y_pred) ** 2)
        nmse = mse / np.var(y_target)
        mc = float(np.corrcoef(y_target, y_pred)[0, 1] ** 2)
        return {"mse": mse, "nmse": nmse, "mc": mc}


# ---------------------------------------------------------------------------
# Convenience runner for multiple seeds
# ---------------------------------------------------------------------------

def run_taskA_multi_seed(
    n_seeds: int = 10,
    N: int = 8,
    n_trials: int = 200,
    trial_duration: float = 10.0,
    dt: float = 0.01,
    g_couple: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Run Task A across multiple random initial conditions / parameter draws.
    Returns dictionary mapping method name -> array of accuracies (n_seeds,).
    """
    acc_npu = []
    acc_raw = []
    acc_esn = []

    for seed in range(n_seeds):
        set_seed(seed)

        # Data
        stimuli, labels = generate_rhythm_classification_trials(
            n_trials=n_trials,
            trial_duration=trial_duration,
            dt=dt,
            seed=seed + 100,
        )
        n_train = int(0.7 * n_trials)
        stim_train, stim_test = stimuli[:n_train], stimuli[n_train:]
        lbl_train, lbl_test = labels[:n_train], labels[n_train:]

        # NPU with feature extraction
        reservoir = NPUArray(N=N, dt=dt, g_couple=g_couple, seed=seed, gain_range=(1.5, 2.5))
        fe = FeatureExtractor(N=N, dt=dt, window_size=2.0, overlap=1.0)
        encoder = InputEncoder(mode="frequency", n_channels=N)
        clf = TaskAClassifier(reservoir, fe, encoder)
        clf.train(stim_train, lbl_train)
        res = clf.evaluate(stim_test, lbl_test)
        acc_npu.append(res["accuracy"])

        # Raw baseline
        reservoir_raw = NPUArray(N=N, dt=dt, g_couple=g_couple, seed=seed)
        encoder_raw = InputEncoder(mode="frequency", n_channels=N)
        raw_bl = TaskARawBaseline(reservoir_raw, encoder_raw)
        raw_bl.train(stim_train, lbl_train)
        res_raw = raw_bl.evaluate(stim_test, lbl_test)
        acc_raw.append(res_raw["accuracy"])

        # ESN baseline
        esn_bl = TaskAESNBaseline(n_reservoir=N, dt=dt, seed=seed + 200)
        esn_bl.train(stim_train, lbl_train)
        res_esn = esn_bl.evaluate(stim_test, lbl_test)
        acc_esn.append(res_esn["accuracy"])

    return {
        "NPU_features": np.array(acc_npu),
        "Raw": np.array(acc_raw),
        "ESN": np.array(acc_esn),
    }


def run_taskB_multi_seed(
    n_seeds: int = 10,
    N: int = 8,
    n_steps: int = 5000,
    dt: float = 0.01,
    g_couple: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    Run Task B across multiple seeds.
    Returns dictionary mapping method -> NMSE array (n_seeds,).
    """
    nmse_npu = []
    nmse_raw = []
    nmse_esn = []
    mc_npu = []
    mc_raw = []
    mc_esn = []

    for seed in range(n_seeds):
        set_seed(seed)

        # NARMA data
        u, y = generate_narma10(n_steps=n_steps, seed=seed + 300)
        n_train = int(0.6 * n_steps)
        n_val = int(0.2 * n_steps)
        u_train, u_val, u_test = u[:n_train], u[n_train : n_train + n_val], u[n_train + n_val :]
        y_train, y_val, y_test = y[:n_train], y[n_train : n_train + n_val], y[n_train + n_val :]

        # NPU + features
        reservoir = NPUArray(N=N, dt=dt, g_couple=g_couple, seed=seed)
        fe = FeatureExtractor(N=N, dt=dt, window_size=2.0, overlap=1.0)
        encoder = InputEncoder(mode="frequency", n_channels=N)
        model = TaskBNARMA(reservoir, fe, encoder, ridge_alpha=1e-4)
        model.train(u_train, y_train)
        res = model.evaluate(u_test, y_test)
        nmse_npu.append(res["nmse"])
        mc_npu.append(res["mc"])

        # Raw baseline
        reservoir_raw = NPUArray(N=N, dt=dt, g_couple=g_couple, seed=seed)
        encoder_raw = InputEncoder(mode="frequency", n_channels=N)
        raw_bl = TaskBRawBaseline(reservoir_raw, encoder_raw, ridge_alpha=1e-4)
        raw_bl.train(u_train, y_train)
        res_raw = raw_bl.evaluate(u_test, y_test)
        nmse_raw.append(res_raw["nmse"])
        mc_raw.append(res_raw["mc"])

        # ESN baseline
        esn_bl = TaskBESNBaseline(n_reservoir=N, dt=dt, ridge_alpha=1e-4, seed=seed + 400)
        esn_bl.train(u_train, y_train)
        res_esn = esn_bl.evaluate(u_test, y_test)
        nmse_esn.append(res_esn["nmse"])
        mc_esn.append(res_esn["mc"])

    return {
        "NPU_features_nmse": np.array(nmse_npu),
        "Raw_nmse": np.array(nmse_raw),
        "ESN_nmse": np.array(nmse_esn),
        "NPU_features_mc": np.array(mc_npu),
        "Raw_mc": np.array(mc_raw),
        "ESN_mc": np.array(mc_esn),
    }

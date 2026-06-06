"""
observables.py — CORC v2
========================
Enhanced state observation: delay embedding, event statistics, virtual nodes.

Features per node:
  [c_i, a_i, s_pulse_i, n_events_i, mean_IEI_i, cv_IEI_i]
  where IEI = inter-event interval.

Global: concatenation F(t), then delay-embedded: z(t) = [F(t), F(t-Delta), ...]
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict
from .reservoir import CEUState


def compute_event_statistics(
    event_counts: np.ndarray,
    window_steps: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Compute per-node event statistics over rolling windows.

    Parameters
    ----------
    event_counts : (T, N)  binary event indicators per step
    window_steps : int  sliding window size

    Returns
    -------
    dict with keys: n_events, mean_iei, cv_iei
      Each (T, N) array (padded at start with zeros)
    """
    T, N = event_counts.shape
    n_events = np.zeros((T, N))
    mean_iei = np.zeros((T, N))
    cv_iei = np.zeros((T, N))

    for i in range(N):
        # Find event times for node i
        event_times = np.where(event_counts[:, i] > 0)[0].astype(float)
        if len(event_times) < 2:
            continue

        ieis = np.diff(event_times)
        for t in range(T):
            # Events in window [t-window_steps, t]
            start = max(0, t - window_steps)
            in_window = (event_times >= start) & (event_times <= t)
            n_events[t, i] = in_window.sum()

            # IEI stats: use event times within window
            window_events = event_times[in_window]
            if len(window_events) >= 2:
                window_ieis = np.diff(window_events)
                mean_iei[t, i] = window_ieis.mean()
                cv_iei[t, i] = window_ieis.std() / (window_ieis.mean() + 1e-12)

    return {"n_events": n_events, "mean_iei": mean_iei, "cv_iei": cv_iei}


def build_feature_vector(
    states: List[CEUState],
    event_counts: np.ndarray,
    window_steps: int = 50,
) -> np.ndarray:
    """
    Build per-time-step feature vector F(t) of shape (T, N * 4).

    Node features: [c_i, s_i, a_i, s_pulse_i]
    (Event statistics omitted to avoid dead dimensions; they are indirectly
    captured by pulse traces and adaptation.)
    """
    T = len(states)
    N = states[0].c.shape[0]

    features = np.zeros((T, N * 4), dtype=np.float32)
    for t in range(T):
        s = states[t]
        base = 0
        features[t, base:base+N] = s.c                      # c
        base += N
        features[t, base:base+N] = s.s                      # s (ER store)
        base += N
        features[t, base:base+N] = s.a                      # a
        base += N
        features[t, base:base+N] = s.pulse                  # s_pulse
    return features


def delay_embed(
    feature_matrix: np.ndarray,
    taps: int = 5,
    interval_steps: int = 4,
) -> np.ndarray:
    """
    Delay embedding of feature matrix.

    Parameters
    ----------
    feature_matrix : (T, D)
    taps : int  number of delay taps (including current)
    interval_steps : int  steps between taps

    Returns
    -------
    embedded : (T, D * taps)
    """
    T, D = feature_matrix.shape
    out = []
    for lag in range(taps):
        shift = lag * interval_steps
        if shift == 0:
            out.append(feature_matrix)
        else:
            padded = np.zeros_like(feature_matrix)
            padded[:-shift] = feature_matrix[shift:]
            out.append(padded)
    return np.concatenate(out, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Virtual node utilities
# ---------------------------------------------------------------------------

def virtual_node_embed(
    states: List[CEUState],
    n_real: int,
    n_virtual_per_real: int,
    interval_steps: int = 4,
) -> np.ndarray:
    """
    Create virtual nodes from a small number of real CEUs.
    Uses delay taps as virtual nodes, mimicking time-multiplexing.

    Parameters
    ----------
    states : list of CEUState  from reservoir with N_real nodes
    n_real : int  number of real CEUs to use
    n_virtual_per_real : int  number of virtual nodes per real node
    interval_steps : int  delay between virtual nodes

    Returns
    -------
    virtual_states : (T, n_real * n_virtual_per_real)  delay-embedded
    """
    T = len(states)
    c_all = np.stack([s.c for s in states], axis=0)  # (T, N)
    c_selected = c_all[:, :n_real]  # (T, n_real)

    virtual = []
    for lag in range(n_virtual_per_real):
        shift = lag * interval_steps
        if shift == 0:
            virtual.append(c_selected)
        else:
            padded = np.zeros_like(c_selected)
            padded[:-shift] = c_selected[shift:]
            virtual.append(padded)
    return np.concatenate(virtual, axis=1)


def polynomial_expand(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Polynomial feature expansion (degree-2).
    Includes: 1, x_i, x_i^2, x_i * x_j (j>i).
    Uses sklearn if available, else manual fallback.
    """
    try:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)
        return poly.fit_transform(X)
    except ImportError:
        T, D = X.shape
        out = [np.ones((T, 1)), X]  # bias, linear
        if degree >= 2:
            out.append(X ** 2)
            # pairwise products (limited to first 32 features to avoid blowup)
            limit = min(D, 32)
            for i in range(limit):
                for j in range(i+1, limit):
                    out.append((X[:, i] * X[:, j])[:, None])
        return np.concatenate(out, axis=1)
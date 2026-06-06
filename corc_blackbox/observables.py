"""
observables.py
--------------
Delay embedding + event detection + derived metrics (CLAUDE.md §6).

- Delay taps on high-dimensional mixed features
- Kuramoto order parameter for synchrony
- Optional PCA/t-SNE/UMAP wrappers
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple
from .reservoir import CORCState


def delay_embed(
    states: List[CORCState],
    features: Optional[List[str]] = None,
    taps: int = 4,
    interval_steps: int = 3,
) -> np.ndarray:
    """
    Build delay-embedded feature matrix.
    Output shape: (T, N * len(features) * taps)
    First K columns are current time, next K are t-interval, etc.
    """
    if features is None:
        features = ["x", "y", "a", "r", "s"]
    T = len(states)
    N = states[0].x.shape[0]
    F = len(features)
    K = N * F

    # raw matrix (T, K)
    raw = np.stack(
        [np.concatenate([getattr(s, f) for f in features]) for s in states],
        axis=0,
    )

    out = []
    for lag in range(taps):
        shift = lag * interval_steps
        if shift == 0:
            out.append(raw)
        else:
            # pad with zeros at the end so length stays T
            padded = np.zeros_like(raw)
            padded[:-shift] = raw[shift:]
            out.append(padded)
    return np.concatenate(out, axis=1)


def kuramoto_order(r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    r: (T,N) amplitudes
    phi: (T,N) phases
    Returns complex order parameter Z(t) = |<r_i e^{i phi_i}>| / <r_i>
    """
    z = r * np.exp(1j * phi)
    z_mean = z.mean(axis=1)
    r_mean = r.mean(axis=1)
    return np.abs(z_mean) / (r_mean + 1e-12)


def event_raster(event_counts: np.ndarray, dt: float) -> dict:
    """
    event_counts: (T,N) binary or integer counts per step.
    Returns dict with spike times per node.
    """
    T, N = event_counts.shape
    times = np.arange(T) * dt
    raster = {}
    for i in range(N):
        idx = np.where(event_counts[:, i] > 0)[0]
        raster[i] = times[idx]
    return raster


def sliding_event_rate(event_counts: np.ndarray, window_steps: int) -> np.ndarray:
    """
    event_counts: (T,N)
    Returns (T,N) smoothed rate (events per step).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if (T := event_counts.shape[0]) < window_steps:
        return event_counts.astype(float)
    # pad front
    padded = np.vstack([np.zeros((window_steps - 1, event_counts.shape[1])), event_counts])
    sw = sliding_window_view(padded, window_steps, axis=0)  # (T, N, W)
    return sw.sum(axis=2)


def state_covariance_dimension(states_mat: np.ndarray, threshold: float = 0.95) -> float:
    """
    states_mat: (T, D)
    Returns effective dimensionality = number of PCA components to explain threshold variance.
    """
    X = states_mat - states_mat.mean(axis=0)
    cov = (X.T @ X) / X.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    cumsum = np.cumsum(eigvals)
    total = cumsum[-1] + 1e-12
    dim = np.searchsorted(cumsum / total, threshold) + 1
    return float(dim)

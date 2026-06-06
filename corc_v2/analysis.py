"""
analysis.py — CORC v2
=====================
Criticality and avalanche analysis:
  - Event avalanche size/duration distributions
  - Branching ratio estimation
  - Synchrony order parameter (Kuramoto-like for calcium)
  - State covariance participation ratio (PR)
  - Coupling strength scan for critical point
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import stats


def avalanche_sizes_durations(event_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract avalanche sizes and durations.

    Avalanche: contiguous time bins with >=1 event anywhere in the network.
    Size = total events within avalanche.
    Duration = number of contiguous time bins.
    """
    T, N = event_counts.shape
    active = (event_counts.sum(axis=1) > 0).astype(int)
    if active.sum() == 0:
        return np.array([]), np.array([])

    diffs = np.diff(np.concatenate([[0], active, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    sizes, durations = [], []
    for s, e in zip(starts, ends):
        sizes.append(event_counts[s:e].sum())
        durations.append(e - s)
    return np.array(sizes, dtype=float), np.array(durations, dtype=float)


def powerlaw_fit(data: np.ndarray, xmin: Optional[float] = None) -> Dict[str, float]:
    """Log-binning + linear fit to estimate power-law exponent."""
    if len(data) == 0:
        return {"alpha": np.nan, "r2": np.nan}
    data = data[data > 0]
    if xmin is None:
        xmin = data.min()
    data = data[data >= xmin]
    if len(data) < 10:
        return {"alpha": np.nan, "r2": np.nan}

    bins = np.logspace(np.log10(xmin), np.log10(data.max()), num=15)
    counts, edges = np.histogram(data, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    if mask.sum() < 3:
        return {"alpha": np.nan, "r2": np.nan}
    log_c = np.log(counts[mask])
    log_x = np.log(centers[mask])
    slope, _, r_value, _, _ = stats.linregress(log_x, log_c)
    return {"alpha": float(-slope), "r2": float(r_value ** 2)}


def branching_ratio(event_counts: np.ndarray) -> float:
    """
    Branching ratio proxy: mean(E[t+1] / E[t] | E[t] > 0).
    """
    Et = event_counts.sum(axis=1).astype(float)
    if Et.sum() == 0:
        return 0.0
    mask = Et[:-1] > 0
    if mask.sum() == 0:
        return 0.0
    br = np.mean(Et[1:][mask] / Et[:-1][mask])
    return float(min(br, 5.0))


def synchrony_order(c: np.ndarray) -> np.ndarray:
    """
    Synchrony order parameter for calcium traces.
    Uses normalized temporal coherence R(t) = |mean(complex representation)|.

    Convert to phase-like representation via Hilbert-like: normalise and use as proxy.
    """
    T, N = c.shape
    c_norm = c / (c.std(axis=0, keepdims=True) + 1e-12)
    R = np.zeros(T)
    for t in range(T):
        R[t] = np.abs(np.mean(np.exp(1j * c_norm[t])))
    return R


def state_participation_ratio(X: np.ndarray) -> float:
    """
    Participation ratio of state covariance eigenvalues.
    PR = (sum lambda_i)^2 / sum(lambda_i^2).
    Higher PR = richer dynamics.
    """
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / Xc.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum() + 1e-12
    pr = (total ** 2) / (np.sum(eigvals ** 2) + 1e-12)
    return float(pr)


def critical_scan(
    reservoir_factory,
    g_values: np.ndarray,
    T: int = 2000,
    dt: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Scan coupling strength g_p to identify critical regime.

    Returns dict with:
      g_values, order_means, pr_values, br_values, size_alpha, dur_alpha
    """
    order_means, order_stds = [], []
    pr_values, br_values = [], []
    size_alphas, dur_alphas = [], []

    for g in g_values:
        res = reservoir_factory(g)
        u = np.zeros((T, 1))
        states, events = res.run(u, reset=True)
        c = np.stack([s.c for s in states], axis=0)

        # Synchrony
        R = synchrony_order(c)
        order_means.append(R.mean())
        order_stds.append(R.std())

        # Participation ratio
        X = c
        pr_values.append(state_participation_ratio(X))

        # Branching ratio
        br_values.append(branching_ratio(events))

        # Avalanche power-law fits
        sizes, durations = avalanche_sizes_durations(events)
        fit_s = powerlaw_fit(sizes)
        fit_d = powerlaw_fit(durations)
        size_alphas.append(fit_s["alpha"])
        dur_alphas.append(fit_d["alpha"])

    return {
        "g": g_values,
        "order_mean": np.array(order_means),
        "order_std": np.array(order_stds),
        "pr": np.array(pr_values),
        "branching_ratio": np.array(br_values),
        "size_alpha": np.array(size_alphas),
        "dur_alpha": np.array(dur_alphas),
    }
"""
analysis.py
-----------
Criticality and avalanche analysis (CLAUDE.md §9):
- event size / duration distributions
- branching ratio estimation
- synchrony order parameter vs coupling
- transient richness (state covariance dimension)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats


def avalanche_sizes_durations(event_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    event_counts: (T, N) binary array.
    Define avalanche as contiguous time bins with >=1 event anywhere in the network.
    Size = total number of events within avalanche.
    Duration = number of contiguous time bins.
    """
    T, N = event_counts.shape
    active = (event_counts.sum(axis=1) > 0).astype(int)
    if active.sum() == 0:
        return np.array([]), np.array([])

    # find contiguous runs
    diffs = np.diff(np.concatenate([[0], active, [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    sizes = []
    durations = []
    for s, e in zip(starts, ends):
        sizes.append(event_counts[s:e].sum())
        durations.append(e - s)
    return np.array(sizes, dtype=float), np.array(durations, dtype=float)


def powerlaw_fit(data: np.ndarray, xmin: Optional[float] = None) -> Dict[str, float]:
    """
    Simple log-binning + linear fit to estimate power-law exponent.
    Returns {'alpha': slope, 'r2': quality}.
    """
    if len(data) == 0:
        return {"alpha": np.nan, "r2": np.nan}
    data = data[data > 0]
    if xmin is None:
        xmin = data.min()
    data = data[data >= xmin]
    if len(data) < 10:
        return {"alpha": np.nan, "r2": np.nan}

    # log bins
    bins = np.logspace(np.log10(xmin), np.log10(data.max()), num=15)
    counts, edges = np.histogram(data, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    if mask.sum() < 3:
        return {"alpha": np.nan, "r2": np.nan}
    log_c = np.log(counts[mask])
    log_x = np.log(centers[mask])
    slope, intercept, r_value, _, _ = stats.linregress(log_x, log_c)
    return {"alpha": float(-slope), "r2": float(r_value ** 2)}


def branching_ratio_estimate(event_counts: np.ndarray, window: int = 5) -> float:
    """
    Crude branching ratio estimate:
    ratio of descendants to ancestors in sliding windows.
    Here we approximate as corr( E(t+1), E(t) ) where E(t)=total events.
    A more rigorous approach fits autoregressive model.
    """
    Et = event_counts.sum(axis=1).astype(float)
    if Et.sum() == 0:
        return 0.0
    # Use ratio of mean(E[t+1]) / mean(E[t]) conditioned on E[t]>0
    mask = Et[:-1] > 0
    if mask.sum() == 0:
        return 0.0
    br = np.mean(Et[1:][mask] / Et[:-1][mask])
    return float(min(br, 5.0))


def synchrony_scan(
    reservoir_factory,
    g_values: np.ndarray,
    T: int = 2000,
    dt: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Scan coupling strength and return Kuramoto order parameter time-averages.
    reservoir_factory(g) -> reservoir instance.
    """
    from .observables import kuramoto_order
    order_means = []
    order_stds = []
    for g in g_values:
        res = reservoir_factory(g)
        u = np.zeros((T, 1))
        states, events = res.run(u, reset=True)
        r = np.stack([s.r for s in states], axis=0)
        phi = np.stack([s.phi for s in states], axis=0)
        R = kuramoto_order(r, phi)
        order_means.append(R.mean())
        order_stds.append(R.std())
    return {
        "g": g_values,
        "order_mean": np.array(order_means),
        "order_std": np.array(order_stds),
    }


def state_richness(states_mat: np.ndarray) -> Dict[str, float]:
    """
    states_mat: (T, D)
    Returns covariance dimension and participation ratio.
    """
    from .observables import state_covariance_dimension
    X = states_mat - states_mat.mean(axis=0)
    cov = (X.T @ X) / X.shape[0]
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum() + 1e-12

    # participation ratio
    pr = (total ** 2) / (np.sum(eigvals ** 2) + 1e-12)
    cov_dim = state_covariance_dimension(states_mat, threshold=0.95)
    return {"cov_dim": cov_dim, "participation_ratio": float(pr)}

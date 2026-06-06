"""
plots.py
--------
Seven required chart families (CLAUDE.md §11):
1. Continuous states, event raster, synchrony, avalanche distributions
2. Dynamics comparison across coupling mechanisms
3. Criticality analysis (avalanche size/duration, order parameter, branching ratio)
4. Main result bar chart (CORC vs baselines)
5. Ablation result matrix
6. Robustness curves (noise, drift, node failure)
7. State projections (PCA / t-SNE / UMAP)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

from .observables import kuramoto_order
from .analysis import avalanche_sizes_durations, powerlaw_fit

plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 150


def save_or_show(fig: plt.Figure, path: Optional[str] = None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 1. Continuous state, event raster, synchrony, avalanche distributions
# ---------------------------------------------------------------------------

def plot_dynamics_overview(
    states_list,
    event_counts: np.ndarray,
    dt: float,
    path: Optional[str] = None,
    max_nodes: int = 8,
):
    """
    states_list: list of CORCState (T steps)
    event_counts: (T, N)
    """
    T = len(states_list)
    t = np.arange(T) * dt
    r = np.stack([s.r for s in states_list], axis=0)
    phi = np.stack([s.phi for s in states_list], axis=0)
    a = np.stack([s.a for s in states_list], axis=0)
    R = kuramoto_order(r, phi)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    # amplitude
    for i in range(min(max_nodes, r.shape[1])):
        axes[0].plot(t, r[:, i], alpha=0.7, lw=0.8)
    axes[0].set_ylabel("Amplitude r")
    axes[0].set_title("Continuous Amplitudes")

    # slow variable
    for i in range(min(max_nodes, a.shape[1])):
        axes[1].plot(t, a[:, i], alpha=0.7, lw=0.8)
    axes[1].set_ylabel("Slow var a")
    axes[1].set_title("Slow Adaptation")

    # event raster
    N = event_counts.shape[1]
    for i in range(N):
        times = t[event_counts[:, i] > 0]
        axes[2].scatter(times, [i] * len(times), s=3, c="k")
    axes[2].set_ylabel("Node")
    axes[2].set_title("Event Raster")

    # synchrony
    axes[3].plot(t, R, c="C3", lw=1.2)
    axes[3].set_ylabel("Order param R")
    axes[3].set_xlabel("Time")
    axes[3].set_title("Kuramoto Order Parameter")

    fig.tight_layout()
    save_or_show(fig, path)


def plot_avalanche_distributions(
    event_counts: np.ndarray,
    path: Optional[str] = None,
):
    sizes, durations = avalanche_sizes_durations(event_counts)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    if len(sizes) > 0:
        ax = axes[0]
        bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max()), 20)
        ax.hist(sizes, bins=bins, density=True, log=True, alpha=0.7, color="C0")
        fit = powerlaw_fit(sizes)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Avalanche size")
        ax.set_ylabel("PDF")
        ax.set_title(f"Size dist (alpha={fit['alpha']:.2f}, r2={fit['r2']:.2f})")

        ax = axes[1]
        bins = np.logspace(np.log10(max(1, durations.min())), np.log10(durations.max()), 20)
        ax.hist(durations, bins=bins, density=True, log=True, alpha=0.7, color="C1")
        fit_d = powerlaw_fit(durations)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Avalanche duration")
        ax.set_ylabel("PDF")
        ax.set_title(f"Duration dist (alpha={fit_d['alpha']:.2f}, r2={fit_d['r2']:.2f})")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No events", ha="center", va="center")
    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 2. Dynamics comparison across coupling mechanisms
# ---------------------------------------------------------------------------

def plot_coupling_comparison(
    results: Dict[str, dict],
    dt: float,
    path: Optional[str] = None,
):
    """
    results: dict label -> {'states': list, 'events': array}
    """
    labels = list(results.keys())
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.5 * n), sharex="col")
    if n == 1:
        axes = axes.reshape(1, -1)
    for i, lab in enumerate(labels):
        states = results[lab]["states"]
        events = results[lab]["events"]
        T = len(states)
        t = np.arange(T) * dt
        r = np.stack([s.r for s in states], axis=0)
        for j in range(min(5, r.shape[1])):
            axes[i, 0].plot(t, r[:, j], alpha=0.7, lw=0.7)
        axes[i, 0].set_ylabel("r")
        axes[i, 0].set_title(lab)
        N = events.shape[1]
        for node in range(N):
            times = t[events[:, node] > 0]
            axes[i, 1].scatter(times, [node] * len(times), s=2, c="k")
        axes[i, 1].set_ylabel("Node")
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 3. Criticality analysis
# ---------------------------------------------------------------------------

def plot_criticality_analysis(
    scan_results: Dict[str, np.ndarray],
    event_counts_dict: Dict[str, np.ndarray],
    path: Optional[str] = None,
):
    """
    scan_results: from synchrony_scan -> {'g', 'order_mean', 'order_std'}
    event_counts_dict: g_value -> event_counts array
    """
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    g = scan_results["g"]
    ax1.errorbar(g, scan_results["order_mean"], yerr=scan_results["order_std"], marker="o")
    ax1.set_xlabel("Coupling strength g")
    ax1.set_ylabel("Kuramoto R")
    ax1.set_title("Synchrony vs coupling")

    # pick a few g values for avalanche distributions
    chosen = list(event_counts_dict.keys())[:4]
    colors = plt.cm.viridis(np.linspace(0, 1, len(chosen)))
    for c, gval in zip(colors, chosen):
        sizes, _ = avalanche_sizes_durations(event_counts_dict[gval])
        if len(sizes) > 0:
            bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max()), 15)
            ax2.hist(sizes, bins=bins, density=True, alpha=0.4, color=c, label=f"g={gval:.3f}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Avalanche size")
    ax2.set_ylabel("PDF")
    ax2.set_title("Avalanche size distributions")
    ax2.legend(fontsize=7)

    # branching ratio proxy
    brs = [branching_ratio_estimate(event_counts_dict[gval]) for gval in chosen]
    ax3.bar(range(len(chosen)), brs, color=colors)
    ax3.set_xticks(range(len(chosen)))
    ax3.set_xticklabels([f"{gval:.3f}" for gval in chosen], rotation=45)
    ax3.set_xlabel("Coupling g")
    ax3.set_ylabel("Branching ratio proxy")
    ax3.set_title("Branching ratio")
    ax3.axhline(1.0, color="k", ls="--", lw=0.8)

    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 4. Main result bar chart (CORC vs baselines)
# ---------------------------------------------------------------------------

def plot_main_results(
    metrics: Dict[str, Dict[str, float]],
    task_names: List[str],
    metric_key: str = "accuracy",
    path: Optional[str] = None,
):
    """
    metrics: {method_name: {task_name: value}}
    """
    methods = list(metrics.keys())
    x = np.arange(len(task_names))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(methods):
        vals = [metrics[m].get(t, np.nan) for t in task_names]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_ylabel(metric_key)
    ax.set_title(f"Benchmark comparison ({metric_key})")
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 5. Ablation result matrix
# ---------------------------------------------------------------------------

def plot_ablation_matrix(
    ablation_scores: Dict[str, Dict[str, float]],
    path: Optional[str] = None,
):
    """
    ablation_scores: {condition: {task: score}}
    """
    conditions = list(ablation_scores.keys())
    tasks = list(next(iter(ablation_scores.values())).keys())
    mat = np.zeros((len(conditions), len(tasks)))
    for i, cond in enumerate(conditions):
        for j, task in enumerate(tasks):
            mat[i, j] = ablation_scores[cond].get(task, np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    ax.set_yticklabels(conditions)
    for i in range(len(conditions)):
        for j in range(len(tasks)):
            text = ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="k", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Ablation matrix")
    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 6. Robustness curves
# ---------------------------------------------------------------------------

def plot_robustness_curves(
    perturbations: np.ndarray,
    results: Dict[str, np.ndarray],
    xlabel: str = "Noise level",
    path: Optional[str] = None,
):
    """
    results: {method_name: array of scores same length as perturbations}
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, vals in results.items():
        ax.plot(perturbations, vals, marker="o", label=m, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")
    ax.set_title("Robustness")
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, path)


# ---------------------------------------------------------------------------
# 7. State projections (PCA / t-SNE / UMAP)
# ---------------------------------------------------------------------------

def plot_state_projection(
    states_mat: np.ndarray,
    colors: Optional[np.ndarray] = None,
    method: str = "pca",
    path: Optional[str] = None,
):
    """
    states_mat: (T, D)
    colors: (T,) e.g. input value or time
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(states_mat)
        title = "PCA"
    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE
        # subsample if too large
        if states_mat.shape[0] > 2000:
            idx = np.random.choice(states_mat.shape[0], 2000, replace=False)
            proj = TSNE(n_components=2, perplexity=30).fit_transform(states_mat[idx])
            colors = colors[idx] if colors is not None else None
        else:
            proj = TSNE(n_components=2, perplexity=30).fit_transform(states_mat)
        title = "t-SNE"
    elif method.lower() == "umap":
        try:
            import umap
            proj = umap.UMAP(n_components=2).fit_transform(states_mat)
            title = "UMAP"
        except Exception:
            ax.text(0.5, 0.5, "UMAP not installed", ha="center", va="center")
            title = "UMAP (missing)"
            proj = None
    else:
        raise ValueError(f"Unknown method {method}")

    if proj is not None:
        if colors is not None:
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=5, cmap="viridis", alpha=0.6)
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.6)
    ax.set_title(title)
    fig.tight_layout()
    save_or_show(fig, path)


def branching_ratio_estimate(event_counts: np.ndarray, window: int = 5) -> float:
    """Local copy to avoid circular import in plotting only."""
    Et = event_counts.sum(axis=1).astype(float)
    if Et.sum() == 0:
        return 0.0
    mask = Et[:-1] > 0
    if mask.sum() == 0:
        return 0.0
    br = np.mean(Et[1:][mask] / Et[:-1][mask])
    return float(min(br, 5.0))

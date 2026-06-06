"""
plots.py — CORC v2
==================
Eight required chart families:
  1. CEU single-node dynamics (c,s,a traces, bursting, phase portrait)
  2. Network avalanche statistics (size/duration, branching ratio, PR)
  3. Critical g_p vs NARMA/XOR performance curves
  4. Main results comparison (Hopf v1, CEU v2, ESN, linear) all tasks
  5. Memory capacity curves comparison
  6. Ablation matrix
  7. PCA/t-SNE state projections (Hopf vs CEU, different coupling)
  8. Virtual-node vs real-node performance
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

from .analysis import avalanche_sizes_durations, powerlaw_fit

plt.rcParams["font.size"] = 9
plt.rcParams["figure.dpi"] = 150


def save_or_show(fig: plt.Figure, path: Optional[str] = None) -> None:
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception:
            fig.savefig(path)
        plt.close(fig)


# ===================================================================
# 1. CEU single-node dynamics
# ===================================================================

def plot_ceu_single_node(
    states_list,
    event_counts: np.ndarray,
    dt: float,
    path: Optional[str] = None,
    max_nodes: int = 4,
):
    T = len(states_list)
    t = np.arange(T) * dt
    c_arr = np.stack([s.c for s in states_list], axis=0)
    s_arr = np.stack([s.s for s in states_list], axis=0)
    a_arr = np.stack([s.a for s in states_list], axis=0)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for i in range(min(max_nodes, c_arr.shape[1])):
        axes[0].plot(t, c_arr[:, i], alpha=0.7, lw=0.8, label=f"Node {i}")
    axes[0].set_ylabel("c (cytosolic Ca)")
    axes[0].set_title("CEU calcium traces")
    axes[0].legend(fontsize=6, loc='upper right')

    for i in range(min(max_nodes, s_arr.shape[1])):
        axes[1].plot(t, s_arr[:, i], alpha=0.7, lw=0.8)
    axes[1].set_ylabel("s (ER store)")
    axes[1].set_title("ER store recovery")

    for i in range(min(max_nodes, a_arr.shape[1])):
        axes[2].plot(t, a_arr[:, i], alpha=0.7, lw=0.8)
    axes[2].set_ylabel("a (adaptation)")
    axes[2].set_title("Ultra-slow adaptation")

    N = event_counts.shape[1]
    for i in range(N):
        times = t[event_counts[:, i] > 0]
        axes[3].scatter(times, [i]*len(times), s=3, c='k')
    axes[3].set_ylabel("Node")
    axes[3].set_xlabel("Time")
    axes[3].set_title("Event raster")

    fig.tight_layout()
    save_or_show(fig, path)


def plot_ceu_phase_portrait(
    c_arr: np.ndarray,
    s_arr: np.ndarray,
    a_arr: np.ndarray,
    node_idx: int = 0,
    path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(c_arr[:, node_idx], s_arr[:, node_idx], lw=0.5)
    axes[0].set_xlabel("c"); axes[0].set_ylabel("s")
    axes[0].set_title("(c, s) phase plane")
    axes[1].plot(c_arr[:, node_idx], a_arr[:, node_idx], lw=0.5)
    axes[1].set_xlabel("c"); axes[1].set_ylabel("a")
    axes[1].set_title("(c, a) phase plane")
    axes[2].plot(s_arr[:, node_idx], a_arr[:, node_idx], lw=0.5)
    axes[2].set_xlabel("s"); axes[2].set_ylabel("a")
    axes[2].set_title("(s, a) phase plane")
    fig.suptitle(f"CEU Node {node_idx} Phase Portrait")
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 2. Network avalanche statistics
# ===================================================================

def plot_avalanche_stats(
    event_counts: np.ndarray,
    c_arr: np.ndarray,
    path: Optional[str] = None,
):
    sizes, durations = avalanche_sizes_durations(event_counts)
    from .analysis import synchrony_order, branching_ratio, state_participation_ratio

    R = synchrony_order(c_arr)
    br = branching_ratio(event_counts)
    pr = state_participation_ratio(c_arr)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    if len(sizes) > 0:
        bins = np.logspace(np.log10(max(1, sizes.min())), np.log10(sizes.max()), 20)
        ax1.hist(sizes, bins=bins, density=True, log=True, alpha=0.7, color="C0")
        fit = powerlaw_fit(sizes)
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("Size"); ax1.set_ylabel("PDF")
        ax1.set_title(f"Size dist (alpha={fit['alpha']:.2f})")

    ax2 = fig.add_subplot(gs[0, 1])
    if len(durations) > 0:
        bins = np.logspace(np.log10(max(1, durations.min())), np.log10(durations.max()), 20)
        ax2.hist(durations, bins=bins, density=True, log=True, alpha=0.7, color="C1")
        fit_d = powerlaw_fit(durations)
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Duration"); ax2.set_ylabel("PDF")
        ax2.set_title(f"Duration dist (alpha={fit_d['alpha']:.2f})")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(R, lw=1.0, color="C3")
    ax3.set_xlabel("Time step"); ax3.set_ylabel("Synchrony R")
    ax3.set_title(f"Synchrony (mean={R.mean():.3f})")

    ax4 = fig.add_subplot(gs[1, 1])
    metrics_text = f"Branching ratio: {br:.3f}\nParticipation ratio: {pr:.1f}"
    ax4.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14,
             transform=ax4.transAxes)
    ax4.set_title("Criticality metrics")
    ax4.axis('off')

    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 3. Critical g_p vs performance curves
# ===================================================================

def plot_critical_performance(
    scan_results: Dict[str, np.ndarray],
    g_values: np.ndarray,
    performance: Dict[str, np.ndarray],
    path: Optional[str] = None,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    g = scan_results["g"]
    axes[0, 0].errorbar(g, scan_results["order_mean"], yerr=scan_results["order_std"],
                        marker='o', capsize=3)
    axes[0, 0].set_xlabel("g_p"); axes[0, 0].set_ylabel("Synchrony R")
    axes[0, 0].set_title("Synchrony vs coupling")

    axes[0, 1].plot(g, scan_results["branching_ratio"], marker='s', color="C2")
    axes[0, 1].axhline(1.0, color='k', ls='--', lw=0.8)
    axes[0, 1].set_xlabel("g_p"); axes[0, 1].set_ylabel("Branching ratio")
    axes[0, 1].set_title("Branching ratio vs coupling")

    axes[1, 0].plot(g, scan_results["pr"], marker='^', color="C4")
    axes[1, 0].set_xlabel("g_p"); axes[1, 0].set_ylabel("Participation ratio")
    axes[1, 0].set_title("State richness vs coupling")

    for label, vals in performance.items():
        axes[1, 1].plot(g_values, vals, marker='o', label=label, lw=1.2)
    axes[1, 1].set_xlabel("g_p"); axes[1, 1].set_ylabel("NMSE / Accuracy")
    axes[1, 1].set_title("Task performance vs coupling")
    axes[1, 1].legend(fontsize=7)

    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 4. Main results comparison
# ===================================================================

def plot_main_results(
    metrics: Dict[str, Dict[str, float]],
    task_names: List[str],
    metric_key: str = "accuracy",
    path: Optional[str] = None,
):
    methods = list(metrics.keys())
    x = np.arange(len(task_names))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    for i, m in enumerate(methods):
        vals = [metrics[m].get(t, 0) for t in task_names]
        ax.bar(x + i * width, vals, width, label=m, color=colors[i])

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(task_names, rotation=30, ha='right')
    ax.set_ylabel(metric_key)
    ax.set_title(f"CORC v2 Benchmark ({metric_key})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 5. Memory capacity curves
# ===================================================================

def plot_memory_capacity(
    mc_results: Dict[str, Dict[int, float]],
    path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, mc_dict in mc_results.items():
        ks = sorted([k for k in mc_dict if isinstance(k, int)])
        vals = [mc_dict[k] for k in ks]
        ax.plot(ks, vals, marker='o', label=label, lw=1.2, markersize=4)
    ax.set_xlabel("Delay k")
    ax.set_ylabel("MC_k")
    ax.set_title("Memory capacity curves")
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 6. Ablation matrix
# ===================================================================

def plot_ablation_matrix(
    ablation_scores: Dict[str, Dict[str, float]],
    path: Optional[str] = None,
):
    conditions = list(ablation_scores.keys())
    tasks = list(next(iter(ablation_scores.values())).keys())
    mat = np.zeros((len(conditions), len(tasks)))
    for i, cond in enumerate(conditions):
        for j, task in enumerate(tasks):
            mat[i, j] = ablation_scores[cond].get(task, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(tasks)*1.2), max(4, len(conditions)*0.6)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_yticklabels(conditions)
    for i in range(len(conditions)):
        for j in range(len(tasks)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha='center', va='center', fontsize=8,
                    color='k' if 0.2 < mat[i, j] < 0.8 else 'w')
    plt.colorbar(im, ax=ax)
    ax.set_title("Ablation matrix")
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 7. PCA/t-SNE state projections
# ===================================================================

def plot_state_projection(
    X_list: List[Dict],
    path: Optional[str] = None,
):
    """
    X_list: list of dicts with keys 'X' (array), 'label' (str), 'colors' (optional)
    """
    n = len(X_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, X_list):
        X = d['X']
        if X.shape[0] > 2000:
            idx = np.random.choice(X.shape[0], 2000, replace=False)
            X = X[idx]
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2).fit_transform(X)
        if 'colors' in d and d['colors'] is not None:
            c = d['colors']
            if len(c) > X.shape[0]:
                c = c[:X.shape[0]]
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=c, s=5, cmap='viridis', alpha=0.6)
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.6)
        ax.set_title(d['label'])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    fig.suptitle("State PCA projections")
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 8. Virtual vs real node performance
# ===================================================================

def plot_virtual_node_comparison(
    results: Dict[str, Dict[str, float]],
    path: Optional[str] = None,
):
    configs = list(results.keys())
    tasks = list(next(iter(results.values())).keys())
    x = np.arange(len(tasks))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cfg in enumerate(configs):
        vals = [results[cfg].get(t, 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=cfg)
    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Virtual vs Real Node Performance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_or_show(fig, path)


# ===================================================================
# 9. Robustness experiments
# ===================================================================

def plot_robustness_input_noise(
    noise_levels: np.ndarray,
    perf_ceu: np.ndarray,
    perf_esn: np.ndarray,
    task_label: str,
    metric_label: str = "Accuracy",
    path: Optional[str] = None,
):
    """Performance degradation under increasing input noise."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(noise_levels, perf_ceu, 'o-', label="CEU v2", lw=1.5, color="C0")
    ax.plot(noise_levels, perf_esn, 's--', label="ESN", lw=1.5, color="C1")
    ax.set_xlabel("Input noise std")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Input noise robustness — {task_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, path)


def plot_robustness_dropout(
    dropout_rates: np.ndarray,
    perf_ceu: np.ndarray,
    perf_esn: np.ndarray,
    task_label: str,
    metric_label: str = "Accuracy",
    path: Optional[str] = None,
):
    """Performance degradation under random node dropout."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(dropout_rates, perf_ceu, 'o-', label="CEU v2", lw=1.5, color="C0")
    ax.plot(dropout_rates, perf_esn, 's--', label="ESN", lw=1.5, color="C1")
    ax.set_xlabel("Dropout fraction")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Node dropout robustness — {task_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, path)


def plot_robustness_drift(
    drift_scales: np.ndarray,
    perf_ceu: np.ndarray,
    perf_esn: np.ndarray,
    task_label: str,
    metric_label: str = "Accuracy",
    path: Optional[str] = None,
):
    """Performance degradation under slow parameter drift."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(drift_scales, perf_ceu, 'o-', label="CEU v2", lw=1.5, color="C0")
    ax.plot(drift_scales, perf_esn, 's--', label="ESN", lw=1.5, color="C1")
    ax.set_xlabel("c_th drift scale")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Parameter drift robustness — {task_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, path)


def plot_robustness_summary(
    metrics: Dict[str, Dict[str, float]],
    path: Optional[str] = None,
):
    """Bar chart comparing robustness: relative retention at 50% perturbation."""
    conditions = list(metrics.keys())
    tasks = list(next(iter(metrics.values())).keys())
    x = np.arange(len(tasks))
    width = 0.8 / len(conditions)
    fig, ax = plt.subplots(figsize=(max(6, len(tasks)*2), 4))
    for i, cond in enumerate(conditions):
        vals = [metrics[cond].get(t, 0) for t in tasks]
        ax.bar(x + i * width, vals, width, label=cond)
    ax.set_xticks(x + width * (len(conditions) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=30, ha='right')
    ax.set_ylabel("Relative retention @ 50% perturbation")
    ax.set_title("Robustness comparison")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
    fig.tight_layout()
    save_or_show(fig, path)
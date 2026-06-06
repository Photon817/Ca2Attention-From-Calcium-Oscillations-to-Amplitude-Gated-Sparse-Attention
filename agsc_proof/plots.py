#!/usr/bin/env python3
"""
AGSC Proof-of-Concept: Publication-quality figures.
Style: Nature/Science-inspired — clean, minimal, high contrast.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

# ── Style config ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})
COLORS = ["#2166AC", "#B2182B", "#4DAF4A", "#FF7F00", "#A65628"]
MARKERS = ["o", "s", "D", "^", "v"]

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Fig 1: AGSC Mechanism Schematic
# ═══════════════════════════════════════════════════════════════

def plot_agsc_mechanism():
    """Illustrate AGSC as a 3-step pipeline: threshold → gate → aggregate."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))

    # Step 1: Input signal
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t) + 0.3 * np.sin(3 * t)
    ax = axes[0]
    ax.plot(t, x, color=COLORS[0], lw=1.2)
    ax.fill_between(t, x, alpha=0.15, color=COLORS[0])
    ax.axhline(y=0, color='gray', lw=0.5, ls='--')
    ax.set_title("Raw Signal", fontweight="bold")
    ax.set_xlabel("t"); ax.set_ylabel("x")

    # Step 2: Threshold + Gate
    theta = 0.15
    S = 1.0 / (1.0 + np.exp(-(x - theta) / 0.05))  # sigmoid approx
    G = (1.0 + 0.5 * x) * S
    ax = axes[1]
    ax.plot(t, S, color=COLORS[1], lw=1.2, label="sparsity S")
    ax.plot(t, G, color=COLORS[2], lw=1.2, label="score G")
    ax.axhline(y=theta, color='gray', lw=0.5, ls='--')
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Threshold + Gate", fontweight="bold")
    ax.set_xlabel("t"); ax.set_ylabel("score")

    # Step 3: Causal aggregation
    cum = np.cumsum(G)
    norm_cum = cum / (np.arange(len(cum)) + 1)
    ax = axes[2]
    ax.plot(t, np.cumsum(G * x) / (cum + 1e-8), color=COLORS[3], lw=1.2)
    ax.set_title("Causal Aggregate", fontweight="bold")
    ax.set_xlabel("t"); ax.set_ylabel("output")

    # Step 4: Complexity comparison
    ax = axes[3]
    N = np.array([10, 20, 50, 100, 200, 500])
    d = 16
    self_attn = N**2 * d
    lin_attn = N * d**2
    agsc = N * d
    ax.loglog(N, self_attn, 's-', color=COLORS[1], label="Self-Attn O(N²d)", lw=1.2)
    ax.loglog(N, lin_attn, 'o-', color=COLORS[2], label="Linear Attn O(Nd²)", lw=1.2)
    ax.loglog(N, agsc, 'D-', color=COLORS[3], label="AGSC O(Nd)", lw=1.5, markersize=5)
    ax.set_title("Complexity", fontweight="bold")
    ax.set_xlabel("N"); ax.set_ylabel("Operations")
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(True, alpha=0.2, lw=0.5)

    fig.suptitle("AGSC Pipeline: Threshold → Gate → Aggregate", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "01_agsc_mechanism.png")


# ═══════════════════════════════════════════════════════════════
# Fig 2: Copy Memory Results
# ═══════════════════════════════════════════════════════════════

def plot_copy_memory_results(results):
    """Bar chart: test loss & accuracy for Copy Memory task."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    models = list(results.keys())
    x = np.arange(len(models))
    w = 0.35

    losses = [results[m]["test_loss"] for m in models]
    accs = [results[m]["accuracy"] * 100 for m in models]
    params = [results[m]["params"] for m in models]

    bars1 = ax1.bar(x, losses, w, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    ax1.set_ylabel("Test Loss (BCE)")
    ax1.set_title("Copy Memory: Test Loss", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha='right')

    bars2 = ax2.bar(x, accs, w, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{p}p", ha='center', fontsize=6, color='gray')
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Copy Memory: Accuracy", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha='right')

    fig.tight_layout()
    _save(fig, "02_copy_memory_results.png")


# ═══════════════════════════════════════════════════════════════
# Fig 3: Mackey-Glass Results
# ═══════════════════════════════════════════════════════════════

def plot_mackey_glass_results(results):
    """Bar chart: test MSE & forward time for Mackey-Glass."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    models = list(results.keys())
    x = np.arange(len(models))
    w = 0.35

    mses = [results[m]["test_mse"] for m in models]
    fwds = [results[m]["forward_ms"] for m in models]
    params = [results[m]["params"] for m in models]

    bars1 = ax1.bar(x, mses, w, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    ax1.set_ylabel("Test MSE")
    ax1.set_title("Mackey-Glass: Test MSE", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha='right')
    ax1.set_yscale('symlog', linthresh=1e-6)

    bars2 = ax2.bar(x, fwds, w, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{p}p", ha='center', fontsize=6, color='gray')
    ax2.set_ylabel("Forward Time (ms)")
    ax2.set_title("Mackey-Glass: Speed", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha='right')

    fig.tight_layout()
    _save(fig, "03_mackey_glass_results.png")


# ═══════════════════════════════════════════════════════════════
# Fig 4: Scaling Study
# ═══════════════════════════════════════════════════════════════

def plot_scaling_study(seq_lengths, times):
    """Forward time vs sequence length, log-log."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for i, name in enumerate(times):
        ax.plot(seq_lengths, times[name], f'{MARKERS[i]}-', color=COLORS[i],
                label=name, lw=1.5, markersize=5)

    # Reference slopes
    N_ref = np.array(seq_lengths)
    ax.plot(N_ref, N_ref * 0.01, '--', color='gray', lw=0.8, alpha=0.5, label='O(N)')
    ax.plot(N_ref, N_ref**2 * 1e-4, ':', color='gray', lw=0.8, alpha=0.5, label='O(N²)')

    ax.set_xlabel("Sequence Length T")
    ax.set_ylabel("Forward Time (ms)")
    ax.set_title("Scaling: Forward Time vs Sequence Length", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2, lw=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')

    fig.tight_layout()
    _save(fig, "04_scaling_study.png")


# ═══════════════════════════════════════════════════════════════
# Fig 5: Parameter Efficiency
# ═══════════════════════════════════════════════════════════════

def plot_parameter_efficiency(results_copy, results_mg):
    """Parameter count vs performance scatter."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Copy Memory
    for i, (name, r) in enumerate(results_copy.items()):
        ax1.scatter(r["params"], r["test_loss"], s=80, color=COLORS[i],
                    marker=MARKERS[i], edgecolors='white', lw=0.5, zorder=3)
        ax1.annotate(name, (r["params"], r["test_loss"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax1.set_xlabel("Parameters")
    ax1.set_ylabel("Test Loss (BCE)")
    ax1.set_title("Copy Memory: Param Efficiency", fontweight="bold")
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.2, lw=0.5)

    # Mackey-Glass
    for i, (name, r) in enumerate(results_mg.items()):
        ax2.scatter(r["params"], r["test_mse"], s=80, color=COLORS[i],
                    marker=MARKERS[i], edgecolors='white', lw=0.5, zorder=3)
        ax2.annotate(name, (r["params"], r["test_mse"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=6)
    ax2.set_xlabel("Parameters")
    ax2.set_ylabel("Test MSE")
    ax2.set_title("Mackey-Glass: Param Efficiency", fontweight="bold")
    ax2.set_xscale('log')
    ax2.set_yscale('symlog', linthresh=1e-6)
    ax2.grid(True, alpha=0.2, lw=0.5)

    fig.tight_layout()
    _save(fig, "05_parameter_efficiency.png")


# ═══════════════════════════════════════════════════════════════
# Fig 6: Attention Heatmap (AGSC vs Self-Attention)
# ═══════════════════════════════════════════════════════════════

def plot_attention_comparison():
    """Conceptual comparison: AGSC causal scores vs self-attention matrix."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))

    T = 20
    # AGSC scores: causal, diagonal-dominant
    agsc_scores = np.zeros((T, T))
    for t in range(T):
        for j in range(t + 1):
            # Score decays with distance, modulated by signal
            agsc_scores[t, j] = np.exp(-0.3 * (t - j)) * (0.5 + 0.5 * np.sin(j / 2))
    # Normalize rows
    row_sums = agsc_scores.sum(axis=1, keepdims=True) + 1e-8
    agsc_scores = agsc_scores / row_sums

    im1 = ax1.imshow(agsc_scores, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.3)
    ax1.set_title("AGSC: Causal Salience", fontweight="bold")
    ax1.set_xlabel("Key position j"); ax1.set_ylabel("Query position t")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Self-attention: pairwise, content-dependent
    np.random.seed(42)
    sa_scores = np.abs(np.random.randn(T, T)) * 0.5
    # Causal mask
    sa_scores = np.tril(sa_scores)
    row_sums = sa_scores.sum(axis=1, keepdims=True) + 1e-8
    sa_scores = sa_scores / row_sums

    im2 = ax2.imshow(sa_scores, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.3)
    ax2.set_title("Self-Attn: Content Matching", fontweight="bold")
    ax2.set_xlabel("Key position j"); ax2.set_ylabel("Query position t")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Attention Pattern Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "06_attention_comparison.png")


# ═══════════════════════════════════════════════════════════════
# Fig 7: Training Curves
# ═══════════════════════════════════════════════════════════════

def plot_training_curves(results_copy, results_mg):
    """Not available from current run — placeholder for extended training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Simplified: show final loss comparison
    models = list(results_copy.keys())
    x = np.arange(len(models))
    copy_losses = [results_copy[m]["test_loss"] for m in models]
    mg_losses = [results_mg.get(m, {}).get("test_mse", 0) for m in models]

    ax1.bar(x, copy_losses, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=25, ha='right')
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Copy Memory", fontweight="bold")

    ax2.bar(x, mg_losses, color=COLORS[:len(models)], edgecolor='white', lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=25, ha='right')
    ax2.set_ylabel("Test MSE")
    ax2.set_title("Mackey-Glass", fontweight="bold")
    ax2.set_yscale('symlog', linthresh=1e-6)

    fig.suptitle("Model Performance Comparison", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "07_performance_summary.png")


# ═══════════════════════════════════════════════════════════════
# Fig 8: Connection to CORC v2
# ═══════════════════════════════════════════════════════════════

def plot_agsc_corc_connection():
    """Show how AGSC emerges from CORC v2 pulse coupling."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    # (a) CORC v2 pulse coupling equation
    ax = axes[0, 0]
    ax.axis('off')
    eq_text = (
        r"$\mathbf{CORC\ v2\ Pulse\ Coupling:}$" + "\n\n"
        r"$C_{i,c}^{\mathrm{pulse}} = g_p \sum_j W_{ij} \cdot s_j^{\mathrm{pulse}} \cdot \mathbf{(1 + \eta \cdot c_j)}$" + "\n\n"
        r"$\uparrow$" + "\n"
        r"$\mathbf{Amplitude\ Gate}$"
    )
    ax.text(0.5, 0.5, eq_text, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # (b) AGSC abstraction
    ax = axes[0, 1]
    ax.axis('off')
    eq_text = (
        r"$\mathbf{AGSC\ Abstraction:}$" + "\n\n"
        r"$\mathrm{score}_j = (1 + \eta \cdot x_j) \cdot \sigma(x_j - \theta)$" + "\n\n"
        r"$\mathrm{output} = \frac{\sum_j \mathrm{score}_j \cdot x_j}{\sum_j \mathrm{score}_j}$" + "\n\n"
        r"$\mathbf{Complexity:\ } O(N \cdot d)$"
    )
    ax.text(0.5, 0.5, eq_text, transform=ax.transAxes, ha='center', va='center',
            fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # (c) NARMA ablation: η=0 effect
    ax = axes[1, 0]
    conditions = [r"$\eta=0.5$ (AGSC)", r"$\eta=0$ (no gate)"]
    values = [0.71, 6774]
    bars = ax.bar(conditions, values, color=[COLORS[0], COLORS[1]], edgecolor='white', lw=0.5)
    ax.set_ylabel("NARMA-10 NMSE")
    ax.set_title("CORC v2 Ablation: η → 0", fontweight="bold")
    ax.set_yscale('log')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{v}", ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.2, lw=0.5, axis='y')

    # (d) Biological → Computational mapping
    ax = axes[1, 1]
    ax.axis('off')
    mapping = (
        r"$\mathbf{Biological\ \rightarrow\ Computational}$" + "\n\n"
        r"$Ca^{2+}\ threshold\ \rightarrow\ \theta$" + "\n"
        r"$IP_3R\ open\ probability\ \rightarrow\ S(x_j > \theta)$" + "\n"
        r"$Cytosolic\ [Ca^{2+}]\ \rightarrow\ (1 + \eta \cdot x_j)$" + "\n"
        r"$ER\ release\ \rightarrow\ \sum_j score_j \cdot v_j$" + "\n\n"
        r"$\mathbf{Principle:}$" + "\n"
        r"$\mathbf{Signal\ strength\ modulates\ influence}$"
    )
    ax.text(0.5, 0.5, mapping, transform=ax.transAxes, ha='center', va='center',
            fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='aliceblue', alpha=0.8))

    fig.suptitle("AGSC: From CORC v2 to Linear Attention", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "08_agsc_corc_connection.png")


# ═══════════════════════════════════════════════════════════════
# Fig 9: Biological Distillation Framework
# ═══════════════════════════════════════════════════════════════

def plot_biological_distillation():
    """High-level diagram: biological computation → distilled principles → AI."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Three columns
    col_colors = ['#FFF3E0', '#E8F5E9', '#E3F2FD']
    col_titles = ['Biological System', 'Distilled Principle', 'AI Application']
    col_x = [0.5, 4.5, 8.5]

    for i, (cx, color, title) in enumerate(zip(col_x, col_colors, col_titles)):
        rect = mpatches.FancyBboxPatch((cx, 0.3), 3.2, 5.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='gray', lw=0.8, alpha=0.7)
        ax.add_patch(rect)
        ax.text(cx + 1.6, 5.7, title, ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows between columns
    for x_start in [3.7, 7.7]:
        ax.annotate("", xy=(x_start + 0.5, 3.5), xytext=(x_start, 3.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#333333'))

    # Biological items
    bio_items = [
        ("Calcium\nOscillation", 0.2),
        ("Neuronal\nExcitability", 0.3),
        ("Synaptic\nPlasticity", 0.4),
        ("Critical\nAvalanches", 0.5),
    ]
    for (text, alpha), y in zip(bio_items, [4.5, 3.5, 2.5, 1.5]):
        ax.text(col_x[0] + 1.6, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5))

    # Distilled principles
    dist_items = [
        ("Amplitude\nGating", 0.2),
        ("Sparse\nThresholding", 0.3),
        ("Multi-scale\nAdaptation", 0.4),
        ("Event-driven\nCoupling", 0.5),
    ]
    for (text, alpha), y in zip(dist_items, [4.5, 3.5, 2.5, 1.5]):
        ax.text(col_x[1] + 1.6, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5))

    # AI applications
    ai_items = [
        ("AGSC Attention\nO(N·d)", 0.2),
        ("Sparse MoE\nRouting", 0.3),
        ("Adaptive\nLearning Rates", 0.4),
        ("Event-cameras\nSNNs", 0.5),
    ]
    for (text, alpha), y in zip(ai_items, [4.5, 3.5, 2.5, 1.5]):
        ax.text(col_x[2] + 1.6, y, text, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', lw=0.5))

    # Connecting lines
    for y in [4.5, 3.5, 2.5, 1.5]:
        ax.plot([col_x[0] + 3.2, col_x[1] + 0.1], [y, y], '--', color='gray', lw=0.5, alpha=0.5)
        ax.plot([col_x[1] + 3.2, col_x[2] + 0.1], [y, y], '--', color='gray', lw=0.5, alpha=0.5)

    ax.set_title("Biological Distillation: From Nature to AI", fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, "09_biological_distillation.png")


# ═══════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════

def generate_all_agsc_plots(results_copy, results_mg, seq_lengths, times):
    """Generate all AGSC figures."""
    print("Generating AGSC figures...")
    plot_agsc_mechanism()
    plot_copy_memory_results(results_copy)
    plot_mackey_glass_results(results_mg)
    plot_scaling_study(seq_lengths, times)
    plot_parameter_efficiency(results_copy, results_mg)
    plot_attention_comparison()
    plot_agsc_corc_connection()
    plot_biological_distillation()
    print(f"All AGSC figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    # Dummy data for standalone run
    dummy_copy = {
        "AGSC-RNN": {"test_loss": 0.234, "accuracy": 0.066, "params": 298, "forward_ms": 0.812},
        "AGSC-Trans": {"test_loss": 0.237, "accuracy": 0.064, "params": 330, "forward_ms": 0.124},
        "StandardRNN": {"test_loss": 0.256, "accuracy": 0.147, "params": 1432, "forward_ms": 0.334},
        "SelfAttn": {"test_loss": 0.234, "accuracy": 0.102, "params": 1416, "forward_ms": 0.320},
    }
    dummy_mg = {
        "AGSC-RNN": {"test_mse": 0.000, "params": 51, "forward_ms": 2.319},
        "AGSC-Trans": {"test_mse": 0.000, "params": 83, "forward_ms": 0.165},
        "StandardRNN": {"test_mse": 0.010, "params": 929, "forward_ms": 0.856},
        "SelfAttn": {"test_mse": 0.000, "params": 1169, "forward_ms": 0.628},
    }
    dummy_seq = [20, 50, 100]
    dummy_times = {
        "AGSC-RNN": [0.818, 1.956, 3.756],
        "AGSC-Trans": [0.088, 0.099, 0.114],
        "StandardRNN": [0.267, 0.565, 1.070],
        "SelfAttn": [0.205, 0.308, 0.411],
    }
    generate_all_agsc_plots(dummy_copy, dummy_mg, dummy_seq, dummy_times)
#!/usr/bin/env python3
"""
CORC v2: Schematic diagrams and architecture figures.
Style: Nature Computational Science / PNAS-inspired.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Circle, Rectangle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures_v2")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {
    "ceu": "#2166AC",
    "pulse": "#B2182B",
    "adapt": "#4DAF4A",
    "er": "#FF7F00",
    "diff": "#A65628",
    "input": "#7570B3",
    "output": "#D95F02",
    "readout": "#1B9E77",
    "bio": "#E41A1C",
    "comp": "#377EB8",
    "hardware": "#984EA3",
}


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Schematic 1: System Architecture Overview
# ═══════════════════════════════════════════════════════════════

def plot_system_architecture():
    """Full CORC v2 system architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ── Biological layer (left) ──
    bio_rect = FancyBboxPatch((0.3, 0.5), 3.5, 7, boxstyle="round,pad=0.15",
                               facecolor='#FFF3E0', edgecolor='#E65100', lw=1.5, alpha=0.5)
    ax.add_patch(bio_rect)
    ax.text(2.05, 7.7, "Biological", ha='center', fontsize=11, fontweight='bold', color='#E65100')

    bio_items = [
        (2.05, 6.5, "Ca²⁺\nOscillation"),
        (2.05, 5.2, "IP₃R\nExcitability"),
        (2.05, 3.9, "Two-Pool\nModel"),
        (2.05, 2.6, "Slow\nInactivation"),
        (2.05, 1.3, "Neuronal\nAvalanches"),
    ]
    for x, y, label in bio_items:
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#E65100', lw=0.8, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5)

    # ── Arrow: Biological → Computational ──
    ax.annotate("", xy=(4.5, 5.5), xytext=(3.9, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333', connectionstyle="arc3,rad=0.2"))
    ax.text(4.2, 6.2, "Abstract", ha='center', fontsize=8, color='#333', fontstyle='italic')

    # ── Computational layer (center) ──
    comp_rect = FancyBboxPatch((4.6, 0.5), 4.5, 7, boxstyle="round,pad=0.15",
                                facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5, alpha=0.5)
    ax.add_patch(comp_rect)
    ax.text(6.85, 7.7, "CORC v2 Framework", ha='center', fontsize=11, fontweight='bold', color='#2E7D32')

    comp_items = [
        (6.85, 6.5, "CEU\nDynamics"),
        (6.85, 5.2, "Pulse\nCoupling"),
        (6.85, 3.9, "Multi-scale\nτ_a"),
        (6.85, 2.6, "Time-scale\nElasticity"),
        (6.85, 1.3, "Event\nEncoding"),
    ]
    for x, y, label in comp_items:
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#2E7D32', lw=0.8, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5)

    # ── Arrow: Computational → Hardware ──
    ax.annotate("", xy=(9.8, 5.5), xytext=(9.2, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#333', connectionstyle="arc3,rad=0.2"))
    ax.text(9.5, 6.2, "Deploy", ha='center', fontsize=8, color='#333', fontstyle='italic')

    # ── Hardware layer (right) ──
    hw_rect = FancyBboxPatch((9.9, 0.5), 3.8, 7, boxstyle="round,pad=0.15",
                              facecolor='#EDE7F6', edgecolor='#4527A0', lw=1.5, alpha=0.5)
    ax.add_patch(hw_rect)
    ax.text(11.8, 7.7, "NPU Hardware", ha='center', fontsize=11, fontweight='bold', color='#4527A0')

    hw_items = [
        (11.8, 6.5, "Optical\nPulses"),
        (11.8, 5.2, "Thermal\nThresh"),
        (11.8, 3.9, "Slow\nFeedback"),
        (11.8, 2.6, "O(N·d)\nComplexity"),
        (11.8, 1.3, "Edge\nInference"),
    ]
    for x, y, label in hw_items:
        box = FancyBboxPatch((x - 1.2, y - 0.5), 2.4, 1.0, boxstyle="round,pad=0.05",
                              facecolor='white', edgecolor='#4527A0', lw=0.8, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=7.5)

    fig.suptitle("CORC v2: From Biological Calcium to Computational Framework to Hardware",
                 fontweight="bold", fontsize=12, y=0.98)
    fig.tight_layout()
    _save(fig, "s01_system_architecture.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 2: CEU Dynamics — Resting vs Bursting
# ═══════════════════════════════════════════════════════════════

def plot_ceu_dynamics_schematic():
    """CEU node dynamics: resting vs bursting states."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

    # ── Row 1: Resting ──
    t = np.linspace(0, 4, 400)
    # Resting: low baseline, no crossing
    c_rest = 0.15 + 0.05 * np.sin(2 * np.pi * 1.5 * t) + 0.02 * np.random.randn(400)
    c_rest[c_rest < 0] = 0

    ax = axes[0, 0]
    ax.plot(t, c_rest, color=COLORS["ceu"], lw=1.2)
    ax.axhline(y=0.3, color='red', ls='--', lw=0.8, label='c_th')
    ax.fill_between(t, 0, c_rest, alpha=0.15, color=COLORS["ceu"])
    ax.set_title("Resting: Sub-threshold", fontweight="bold", fontsize=9)
    ax.set_ylabel("c (cytosolic Ca²⁺)")
    ax.legend(fontsize=6)

    # Bursting
    c_burst = 0.3 + 0.4 * np.sin(2 * np.pi * 2.0 * t) * (np.sin(np.pi * t / 4) > 0.3)
    c_burst += 0.05 * np.random.randn(400)
    c_burst = np.clip(c_burst, 0, 1.2)

    ax = axes[0, 1]
    ax.plot(t, c_burst, color=COLORS["pulse"], lw=1.2)
    ax.axhline(y=0.3, color='red', ls='--', lw=0.8)
    ax.fill_between(t, 0.3, c_burst, where=c_burst > 0.3, alpha=0.3, color=COLORS["pulse"])
    ax.set_title("Bursting: Super-threshold → Events", fontweight="bold", fontsize=9)
    ax.set_ylabel("c (cytosolic Ca²⁺)")

    # Phase portrait
    ax = axes[0, 2]
    c_cycle = np.sin(np.linspace(0, 2 * np.pi, 200)) * 0.4 + 0.5
    s_cycle = 1 - c_cycle * 0.6
    ax.plot(c_cycle, s_cycle, color=COLORS["ceu"], lw=1.5)
    ax.axvline(x=0.3, color='red', ls='--', lw=0.8, alpha=0.7)
    # Annotate: threshold crossing
    ax.annotate("threshold\ncrossing", xy=(0.5, 0.7), xytext=(0.7, 0.85),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='red'),
                fontsize=6, color='red')
    ax.set_xlabel("c"); ax.set_ylabel("s (ER store)")
    ax.set_title("Phase Portrait: c vs s", fontweight="bold", fontsize=9)

    # ── Row 2: Adaptation + Coupling ──
    ax = axes[1, 0]
    a1 = 0.5 * (1 - np.exp(-t / 1.5))
    ax.plot(t, a1, color=COLORS["adapt"], lw=1.5, label=r'$\tau_a=1.5$')
    a2 = 0.5 * (1 - np.exp(-t / 0.5))
    ax.plot(t, a2, color=COLORS["adapt"], alpha=0.5, lw=1.5, label=r'$\tau_a=0.5$')
    a3 = 0.5 * (1 - np.exp(-t / 5.0))
    ax.plot(t, a3, color=COLORS["adapt"], alpha=0.3, lw=1.5, label=r'$\tau_a=5.0$')
    ax.set_title("Multi-scale Adaptation", fontweight="bold", fontsize=9)
    ax.set_ylabel("a (adaptation)")
    ax.legend(fontsize=6)

    ax = axes[1, 1]
    g_p_vals = [0.0, 0.15, 0.25, 0.35]
    nmse_vals = [1147, 0.71, 0.71, 2.5]
    ax.plot(g_p_vals, nmse_vals, 'o-', color=COLORS["pulse"], lw=1.5, markersize=6)
    ax.axvspan(0.15, 0.30, alpha=0.1, color='green')
    ax.text(0.22, 0.5, "Critical\nZone", ha='center', fontsize=7, color='green',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.set_xlabel("g_p (pulse coupling)")
    ax.set_ylabel("NARMA NMSE")
    ax.set_title("Critical Coupling", fontweight="bold", fontsize=9)
    ax.set_yscale('log')

    ax = axes[1, 2]
    tasks = ["Rhythm", "HardRhythm", "TempXOR", "NARMA"]
    ceu = [1.0, 0.98, 1.0, 0.71]
    esn = [1.0, 0.43, 0.50, 0.31]
    x = np.arange(len(tasks))
    w = 0.35
    ax.bar(x - w/2, ceu, w, label="CEU v2", color=COLORS["ceu"], edgecolor='white', lw=0.5)
    ax.bar(x + w/2, esn, w, label="ESN", color=COLORS["diff"], edgecolor='white', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=7)
    ax.set_ylabel("Score")
    ax.set_title("Task Performance", fontweight="bold", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("CEU Node Dynamics: From Resting to Bursting to Computation",
                 fontweight="bold", fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, "s02_ceu_dynamics.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 3: AGSC → Linear Attention Derivation
# ═══════════════════════════════════════════════════════════════

def plot_agsc_derivation():
    """Formal derivation: AGSC as linear attention special case."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.8, "AGSC as a Degenerate Case of Linear Attention",
            ha='center', fontsize=13, fontweight='bold')

    # Step 1: Standard linear attention
    box1 = FancyBboxPatch((0.3, 4.5), 9.4, 2.0, boxstyle="round,pad=0.1",
                           facecolor='#E3F2FD', edgecolor='#1565C0', lw=1.2, alpha=0.6)
    ax.add_patch(box1)
    ax.text(5, 6.2, "Standard Linear Attention (Katharopoulos et al., 2020)",
            ha='center', fontsize=10, fontweight='bold', color='#1565C0')
    ax.text(5, 5.5,
            r"$\mathrm{Attn}(Q,K,V) = \frac{\phi(Q) \cdot (\phi(K)^T V)}{\phi(Q) \cdot \phi(K)^T \mathbf{1}}$",
            ha='center', fontsize=11, fontfamily='monospace')
    ax.text(5, 4.8, "Complexity: O(N·d²)  |  φ is a non-negative kernel (e.g., ELU+1, ReLU)",
            ha='center', fontsize=8, color='gray')

    # Arrow
    ax.annotate("", xy=(5, 4.3), xytext=(5, 4.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    # Step 2: AGSC specialization
    box2 = FancyBboxPatch((0.3, 1.8), 9.4, 2.0, boxstyle="round,pad=0.1",
                           facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.2, alpha=0.6)
    ax.add_patch(box2)
    ax.text(5, 3.5, "AGSC Specialization: Q = 1,  K = V = X",
            ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
    ax.text(5, 2.9,
            r"$\phi_{\mathrm{AGSC}}(x_j) = (1 + \eta \cdot x_j) \cdot \sigma\left(\frac{x_j - \theta}{\tau}\right)$",
            ha='center', fontsize=11, fontfamily='monospace')
    ax.text(5, 2.3,
            r"$\mathrm{output} = \frac{\sum_j (1 + \eta \cdot x_j) \cdot \mathbb{1}_{x_j > \theta} \cdot x_j}{\sum_j (1 + \eta \cdot x_j) \cdot \mathbb{1}_{x_j > \theta}}$",
            ha='center', fontsize=10, fontfamily='monospace')

    # Arrow to complexity
    ax.annotate("", xy=(5, 1.6), xytext=(5, 1.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#333'))

    # Step 3: Complexity
    box3 = FancyBboxPatch((0.3, 0.2), 9.4, 0.9, boxstyle="round,pad=0.1",
                           facecolor='#FFF3E0', edgecolor='#E65100', lw=1.2, alpha=0.6)
    ax.add_patch(box3)
    ax.text(5, 0.65,
            r"Complexity: O(N·d)  |  No Q·Kᵀ  |  No d²  |  Only element-wise + cumsum",
            ha='center', fontsize=10, fontweight='bold', color='#E65100')

    # Right side annotations
    ax.text(9.8, 5.5, "← Q = 1 eliminates\nquery-dependency",
            fontsize=7, color='#1565C0', ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(9.8, 2.9, "← Amplitude gate\nfrom CORC v2",
            fontsize=7, color='#2E7D32', ha='left', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.tight_layout()
    _save(fig, "s03_agsc_derivation.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 4: Bio → Comp Mapping Table
# ═══════════════════════════════════════════════════════════════

def plot_bio_comp_mapping():
    """Biological feature → Computational abstraction mapping table."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(5, 5.3, "Biological Calcium Dynamics → Computational Abstraction",
            ha='center', fontsize=13, fontweight='bold')

    # Table-like layout
    rows = [
        ("Biological Feature", "Computational Abstraction", "Effect"),
        ("IP₃R opening threshold", r"$c_{\mathrm{th}}$ (excitability)", "Sparse activation"),
        ("Cytosolic [Ca²⁺]", r"$(1 + \eta \cdot c_j)$ (amplitude gate)", "Prevents synchrony collapse"),
        ("ER Ca²⁺ store → s", r"$\tau_s$ (slow recovery)", "Refractory period"),
        ("IP₃R slow inactivation", r"$a_i$ (adaptation)", "Multi-scale memory"),
        ("Ca²⁺ spike events", r"$s^{\mathrm{pulse}}$ (event trace)", "Temporal coding"),
        ("Neuronal avalanches", "Critical branching ≈ 1", "Maximal dynamic range"),
        ("Two-pool model", "CEU state (c, s, a)", "Resting ↔ Bursting bistability"),
    ]

    y_positions = [4.8, 4.2, 3.6, 3.0, 2.4, 1.8, 1.2, 0.6]
    col_x = [1.5, 5.0, 8.5]
    col_widths = [3.0, 3.5, 2.0]

    for i, (row, y) in enumerate(zip(rows, y_positions)):
        if i == 0:
            # Header
            for cx, cw, text in zip(col_x, col_widths, row):
                box = FancyBboxPatch((cx - cw/2, y - 0.25), cw, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor='#333', edgecolor='#333', lw=0.5)
                ax.add_patch(box)
                ax.text(cx, y, text, ha='center', va='center', fontsize=8, color='white',
                        fontweight='bold')
        else:
            bg = '#F5F5F5' if i % 2 == 0 else 'white'
            for cx, cw, text in zip(col_x, col_widths, row):
                box = FancyBboxPatch((cx - cw/2, y - 0.25), cw, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=bg, edgecolor='#CCC', lw=0.3, alpha=0.7)
                ax.add_patch(box)
                ax.text(cx, y, text, ha='center', va='center', fontsize=7.5)

    fig.tight_layout()
    _save(fig, "s04_bio_comp_mapping.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 5: NARMA improvement trajectory
# ═══════════════════════════════════════════════════════════════

def plot_narma_trajectory():
    """NARMA NMSE trajectory: v1 → v2 → ESN gap."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    milestones = ["Hopf v1", "CEU v2\n(no pulse)", "CEU v2\n(full)", "ESN\n(upper bound)"]
    nmse = [2730, 0.63, 0.71, 0.31]
    colors = ['#B2182B', '#FF7F00', '#2166AC', '#4DAF4A']

    x = np.arange(len(milestones))
    bars = ax.bar(x, nmse, color=colors, edgecolor='white', lw=0.8, width=0.6)

    for bar, v in zip(bars, nmse):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{v:.2f}", ha='center', fontsize=9, fontweight='bold')

    # Annotate improvement
    ax.annotate("3845×", xy=(1.5, 1), xytext=(2.5, 2),
                fontsize=10, fontweight='bold', color='#2166AC',
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#2166AC'))

    ax.set_xticks(x)
    ax.set_xticklabels(milestones, fontsize=8)
    ax.set_ylabel("NARMA-10 NMSE")
    ax.set_yscale('log')
    ax.set_title("NARMA-10 Performance: v1 → v2", fontweight="bold")
    ax.grid(True, alpha=0.2, lw=0.5, axis='y')

    fig.tight_layout()
    _save(fig, "s05_narma_trajectory.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 6: Five Upgrades
# ═══════════════════════════════════════════════════════════════

def plot_five_upgrades():
    """The five key upgrades from v1 to v2."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    ax.text(5, 4.8, "CORC v1 → v2: Five Key Upgrades", ha='center',
            fontsize=13, fontweight='bold')

    upgrades = [
        (1, 3.5, "CEU\nDynamics", "Resting-Bursting\nbistability", "#2166AC"),
        (3, 3.5, "Pulse\nCoupling", "Critical\navalanches", "#B2182B"),
        (5, 3.5, "Multi-scale\nτ_a", "Continuous\nmemory spectrum", "#4DAF4A"),
        (7, 3.5, "Time-scale\nElasticity", "Task-specific\nfrequency match", "#FF7F00"),
        (9, 3.5, "Event\nEncoding", "Sparse temporal\nfeatures", "#A65628"),
    ]

    for x, y, title, desc, color in upgrades:
        # Circle
        circle = Circle((x, y), 0.7, facecolor=color, edgecolor='white', lw=2, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, y, title, ha='center', va='center', fontsize=7.5, color='white',
                fontweight='bold')
        # Description below
        ax.text(x, y - 1.1, desc, ha='center', va='center', fontsize=7, color='#333')

    # Arrows between circles
    for i in range(4):
        ax.annotate("", xy=(upgrades[i+1][0] - 0.75, 3.5),
                    xytext=(upgrades[i][0] + 0.75, 3.5),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color='#555'))

    # v1 → v2 label
    ax.text(5, 2.0, "v1: Hopf limit cycle → v2: Calcium excitable unit",
            ha='center', fontsize=9, fontstyle='italic', color='gray',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    fig.tight_layout()
    _save(fig, "s06_five_upgrades.png")


# ═══════════════════════════════════════════════════════════════
# Schematic 7: Model Comparison Legend
# ═══════════════════════════════════════════════════════════════

def plot_model_legend():
    """Visual legend: CEU vs ESN vs Hopf vs Linear."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))

    t = np.linspace(0, 2, 200)

    # CEU
    ax = axes[0, 0]
    c = 0.3 + 0.4 * np.sin(2 * np.pi * 3 * t) * (np.sin(np.pi * t / 2) > 0.2)
    ax.plot(t, c, color=COLORS["ceu"], lw=1.5)
    ax.fill_between(t, 0.3, c, where=c > 0.3, alpha=0.3, color=COLORS["ceu"])
    ax.axhline(0.3, color='red', ls='--', lw=0.8)
    ax.set_title("CEU (CORC v2)", fontweight="bold", color=COLORS["ceu"])
    ax.set_ylabel("c, s, a")
    ax.text(0.5, 0.95, "3 state vars\nResting/Bursting\nAmplitude gate",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ESN
    ax = axes[0, 1]
    h = np.tanh(np.random.randn(200) * 0.5 + 0.3 * np.sin(2 * np.pi * 2 * t))
    ax.plot(t, h, color=COLORS["diff"], lw=1.5)
    ax.set_title("ESN (Standard)", fontweight="bold", color=COLORS["diff"])
    ax.set_ylabel("tanh state")
    ax.text(0.5, 0.95, "1 state var\nContinuous\nNo events",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Hopf
    ax = axes[1, 0]
    r = 0.5 + 0.1 * np.random.randn(200)
    theta = np.linspace(0, 8 * np.pi, 200)
    ax.plot(t, r * np.sin(theta), color=COLORS["adapt"], lw=1.5)
    ax.set_title("Hopf (v1)", fontweight="bold", color=COLORS["adapt"])
    ax.set_ylabel("complex amp")
    ax.text(0.5, 0.95, "2 state vars\nFixed limit cycle\nNo events",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Linear
    ax = axes[1, 1]
    ax.plot(t, np.sin(2 * np.pi * 2 * t), color=COLORS["input"], lw=1.5)
    ax.set_title("Linear", fontweight="bold", color=COLORS["input"])
    ax.set_ylabel("delay line")
    ax.text(0.5, 0.95, "0 state vars\nInput delay only\nNo dynamics",
            transform=ax.transAxes, fontsize=7, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle("Model Comparison", fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "s07_model_comparison.png")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def generate_all_schematics():
    print("Generating CORC v2 schematic figures...")
    plot_system_architecture()
    plot_ceu_dynamics_schematic()
    plot_agsc_derivation()
    plot_bio_comp_mapping()
    plot_narma_trajectory()
    plot_five_upgrades()
    plot_model_legend()
    print(f"All schematic figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    generate_all_schematics()
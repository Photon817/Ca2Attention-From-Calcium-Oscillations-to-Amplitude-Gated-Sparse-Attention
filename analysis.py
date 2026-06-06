"""
analysis.py
===========
High-resolution plotting routines for publication-quality figures.

Figure list (matching CLAUDE.md requirements)
----------------------------------------------
1. Single-NPU step response + phase portrait  (nonlinearity demonstration)
2. Multi-NPU synchronization / desynchronization under dual-frequency drive
3. Task A / Task B performance bar charts with error bars
4. Robustness curves (noise vs accuracy, drift vs NMSE, coupling vs metric)
5. Reservoir dynamics projected onto first principal components (PCA / t-SNE)
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Optional, Tuple

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

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "figure.figsize": (5.5, 4.0),
    "lines.linewidth": 1.2,
})

COLORS = {
    "npu": "#2E86AB",
    "raw": "#A23B72",
    "esn": "#F18F01",
    "noise": "#C73E1D",
    "drift": "#3B1F2B",
}


def _savefig(fig, name: str, outdir: str = "figures") -> None:
    os.makedirs(outdir, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(f"{outdir}/{name}.png", dpi=300)
    fig.savefig(f"{outdir}/{name}.pdf", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 – Single NPU step response & phase portrait
# ---------------------------------------------------------------------------

def fig_single_npu_response(dt: float = 0.01, duration: float = 30.0, outdir: str = "figures"):
    """
    Show a single NPU responding to a step increase in stimulus intensity,
    demonstrating amplitude saturation and frequency pulling (Arnold tongue).
    """
    set_seed(0)
    npu = SingleNPU(f0=1.0, tau_ca=1.0, gain=1.0, sigma=0.01, dt=dt)

    # Piecewise constant input: low -> high -> low
    t = np.arange(0, duration, dt)
    I = np.piecewise(
        t,
        [t < 10, (t >= 10) & (t < 20), t >= 20],
        [0.2, 0.8, 0.2],
    )

    # Simulate
    out = np.empty_like(t)
    for k, tk in enumerate(t):
        out[k] = npu.step(I[k])

    t_hist, v_hist, w_hist, c_hist, I_hist = npu.get_signals()

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig)

    # (a) Time trace
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_hist, c_hist, color=COLORS["npu"], label=r"Calcium signal $c(t)$")
    ax1.plot(t_hist, I_hist, color="gray", linestyle="--", alpha=0.7, label="Input $I(t)$")
    ax1.axvline(10, color="red", linestyle=":", alpha=0.5)
    ax1.axvline(20, color="red", linestyle=":", alpha=0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.set_title("(a) Single-NPU step response: amplitude saturation & frequency pulling")
    ax1.legend(loc="upper right")

    # (b) Phase portrait (v vs w)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(v_hist, w_hist, color=COLORS["npu"], alpha=0.6)
    ax2.set_xlabel(r"$v$ (fast variable)")
    ax2.set_ylabel(r"$w$ (recovery)")
    ax2.set_title("(b) Phase portrait $(v, w)$")

    # (c) 3D projection (v, w, c)
    ax3 = fig.add_subplot(gs[1, 1], projection="3d")
    ax3.plot(v_hist, w_hist, c_hist, color=COLORS["npu"], alpha=0.6, lw=0.8)
    ax3.set_xlabel(r"$v$")
    ax3.set_ylabel(r"$w$")
    ax3.set_zlabel(r"$c$")
    ax3.set_title("(c) 3D trajectory $(v, w, c)$")

    _savefig(fig, "fig01_single_npu_response", outdir)


# ---------------------------------------------------------------------------
# Figure 2 – Multi-NPU synchronization under dual-frequency drive
# ---------------------------------------------------------------------------

def fig_multi_npu_sync(
    N: int = 8,
    dt: float = 0.01,
    duration: float = 20.0,
    outdir: str = "figures",
):
    """
    Two groups of NPUs driven at different carrier frequencies.
    Show phase differences, PLV, and whether the reservoir splits into
    distinct synchronization clusters.
    """
    set_seed(1)
    reservoir = NPUArray(N=N, f0_range=(0.5, 2.5), dt=dt, g_couple=0.02, seed=1)

    t = np.arange(0, duration, dt)
    n_steps = len(t)

    # Dual-frequency drive: first half of channels @ 0.8 Hz, second half @ 2.2 Hz
    n_half = N // 2
    drive_freqs = np.empty(N)
    drive_freqs[:n_half] = 0.8
    drive_freqs[n_half:] = 2.2

    def I_func(tk: float) -> np.ndarray:
        return 0.5 * (1.0 + np.sin(2.0 * np.pi * drive_freqs * tk))

    C = reservoir.run(duration=duration, I_func=I_func)

    # Hilbert phase
    phase = np.angle(hilbert(C, axis=0))

    # Compute pairwise phase difference distributions in the last 5 s
    t_mask = t >= (duration - 5.0)
    dphi = []
    for i in range(N):
        for j in range(i + 1, N):
            dphi.append(np.mod(phase[t_mask, i] - phase[t_mask, j] + np.pi, 2 * np.pi) - np.pi)
    dphi = np.concatenate(dphi)

    # Within-group vs across-group PLV
    plv_within = []
    plv_across = []
    for i in range(N):
        for j in range(i + 1, N):
            d = phase[t_mask, i] - phase[t_mask, j]
            plv = np.abs(np.mean(np.exp(1j * d)))
            if (i < n_half and j < n_half) or (i >= n_half and j >= n_half):
                plv_within.append(plv)
            else:
                plv_across.append(plv)

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 2, figure=fig)

    # (a) Calcium traces
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(N):
        ax1.plot(t, C[:, i], alpha=0.6, lw=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel(r"$c_i(t)$")
    ax1.set_title("(a) Multi-NPU calcium traces under dual-frequency drive")

    # (b) Phase difference histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(dphi, bins=60, color=COLORS["npu"], edgecolor="white", alpha=0.8)
    ax2.set_xlabel(r"Phase difference $\Delta\phi$ (rad)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Pairwise phase-difference distribution (last 5 s)")

    # (c) PLV bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(
        [0, 1],
        [np.mean(plv_within), np.mean(plv_across)],
        yerr=[np.std(plv_within), np.std(plv_across)],
        color=[COLORS["npu"], COLORS["raw"]],
        capsize=4,
    )
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Within-group", "Across-group"])
    ax3.set_ylabel("PLV")
    ax3.set_title("(c) Phase-locking value")
    ax3.set_ylim(0, 1)

    _savefig(fig, "fig02_multi_npu_sync", outdir)


# ---------------------------------------------------------------------------
# Figure 3 – Performance comparison bar charts (Task A & B)
# ---------------------------------------------------------------------------

def fig_performance_bars(
    taskA_results: Dict[str, np.ndarray],
    taskB_results: Dict[str, np.ndarray],
    outdir: str = "figures",
):
    """
    taskA_results : dict with keys "NPU_features", "Raw", "ESN"
                    each -> array of accuracies (n_seeds,)
    taskB_results : dict with keys "NPU_features_nmse", "Raw_nmse", "ESN_nmse"
                    each -> array of NMSE (n_seeds,)
    """
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # --- Task A ---
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ["NPU\n(features)", "Raw\n(signal)", "ESN\n(baseline)"]
    means = [
        taskA_results["NPU_features"].mean(),
        taskA_results["Raw"].mean(),
        taskA_results["ESN"].mean(),
    ]
    stds = [
        taskA_results["NPU_features"].std(),
        taskA_results["Raw"].std(),
        taskA_results["ESN"].std(),
    ]
    bars = ax1.bar(
        methods,
        means,
        yerr=stds,
        color=[COLORS["npu"], COLORS["raw"], COLORS["esn"]],
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_ylabel("Classification accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Task A: Rhythm classification")
    # Annotate bars
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    # --- Task B ---
    ax2 = fig.add_subplot(gs[0, 1])
    methods_b = ["NPU\n(features)", "Raw\n(signal)", "ESN\n(baseline)"]
    means_b = [
        taskB_results["NPU_features_nmse"].mean(),
        taskB_results["Raw_nmse"].mean(),
        taskB_results["ESN_nmse"].mean(),
    ]
    stds_b = [
        taskB_results["NPU_features_nmse"].std(),
        taskB_results["Raw_nmse"].std(),
        taskB_results["ESN_nmse"].std(),
    ]
    bars_b = ax2.bar(
        methods_b,
        means_b,
        yerr=stds_b,
        color=[COLORS["npu"], COLORS["raw"], COLORS["esn"]],
        capsize=5,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_ylabel("NMSE")
    ax2.set_title("Task B: NARMA-10 prediction")
    # Annotate
    for bar, m in zip(bars_b, means_b):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    _savefig(fig, "fig03_performance_bars", outdir)


# ---------------------------------------------------------------------------
# Figure 4 – Robustness analysis curves
# ---------------------------------------------------------------------------

def fig_robustness_curves(
    noise_taskA: Dict[str, np.ndarray],
    noise_taskB: Dict[str, np.ndarray],
    drift_taskA: Dict[str, np.ndarray],
    drift_taskB: Dict[str, np.ndarray],
    coupling_taskA: Dict[str, np.ndarray],
    coupling_taskB: Dict[str, np.ndarray],
    outdir: str = "figures",
):
    """
    noise_taskA : dict with keys "sigma_values", "NPU", "Raw", "ESN"
    drift_taskA : dict with keys "drift_values", "NPU", "Raw", "ESN"
    coupling_taskA : dict with keys "couple_values", "NPU", "Raw", "ESN"
    (and corresponding _taskB for NMSE)
    Each value array shape = (n_conditions, n_seeds)
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    def _plot_curve(ax, xvals, data_dict, key_order, title, ylabel):
        for key, color in zip(key_order, [COLORS["npu"], COLORS["raw"], COLORS["esn"]]):
            arr = data_dict[key]  # (n_conditions, n_seeds)
            ax.errorbar(
                xvals,
                arr.mean(axis=1),
                yerr=arr.std(axis=1),
                label=key,
                color=color,
                marker="o",
                markersize=4,
                capsize=3,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()

    key_order = ["NPU", "Raw", "ESN"]

    # Row 1: Noise
    ax11 = fig.add_subplot(gs[0, 0])
    _plot_curve(ax11, noise_taskA["sigma_values"], noise_taskA, key_order,
                "(a) Noise sensitivity – Task A", "Accuracy")
    ax12 = fig.add_subplot(gs[0, 1])
    _plot_curve(ax12, noise_taskB["sigma_values"], noise_taskB, key_order,
                "(b) Noise sensitivity – Task B", "NMSE")

    # Row 2: Parameter drift
    ax21 = fig.add_subplot(gs[1, 0])
    _plot_curve(ax21, drift_taskA["drift_values"], drift_taskA, key_order,
                "(c) Parameter drift – Task A", "Accuracy")
    ax22 = fig.add_subplot(gs[1, 1])
    _plot_curve(ax22, drift_taskB["drift_values"], drift_taskB, key_order,
                "(d) Parameter drift – Task B", "NMSE")

    # Row 3: Coupling
    ax31 = fig.add_subplot(gs[2, 0])
    _plot_curve(ax31, coupling_taskA["couple_values"], coupling_taskA, key_order,
                "(e) Coupling crosstalk – Task A", "Accuracy")
    ax32 = fig.add_subplot(gs[2, 1])
    _plot_curve(ax32, coupling_taskB["couple_values"], coupling_taskB, key_order,
                "(f) Coupling crosstalk – Task B", "NMSE")

    _savefig(fig, "fig04_robustness_curves", outdir)


# ---------------------------------------------------------------------------
# Figure 5 – Reservoir dynamics PCA / t-SNE projection
# ---------------------------------------------------------------------------

def fig_reservoir_projection(
    N: int = 8,
    dt: float = 0.01,
    duration: float = 30.0,
    outdir: str = "figures",
):
    """
    Drive the NPU array with a composite stimulus, collect high-dimensional
    reservoir features, and project onto first 2 principal components.
    Also show t-SNE for comparison.
    """
    set_seed(5)
    reservoir = NPUArray(N=N, dt=dt, g_couple=0.01, seed=5)
    fe = FeatureExtractor(N=N, dt=dt, window_size=2.0, overlap=1.5)
    encoder = InputEncoder(mode="frequency", n_channels=N)

    # Composite stimulus: chirp + amplitude modulation
    t = np.arange(0, duration, dt)
    f_chirp = 0.5 + 1.5 * (t / duration)
    u = 0.5 * (1.0 + np.sin(2.0 * np.pi * f_chirp * t))

    def I_func(tk: float) -> np.ndarray:
        idx = int(round(tk / dt))
        idx = min(idx, len(u) - 1)
        return encoder.encode_vector(tk, u[idx])

    C = reservoir.run(duration=duration, I_func=I_func)
    feats, times = fe.extract(C)

    # Color by time (to show trajectory evolution)
    colors_time = plt.cm.viridis(times / times.max())

    # PCA
    pca = PCA(n_components=2)
    proj_pca = pca.fit_transform(feats)

    # t-SNE (subsample if too large)
    if len(feats) > 500:
        idx_tsne = np.linspace(0, len(feats) - 1, 500, dtype=int)
    else:
        idx_tsne = np.arange(len(feats))
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    proj_tsne = tsne.fit_transform(feats[idx_tsne])

    fig = plt.figure(figsize=(12, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # (a) Input stimulus
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, u, color="black", lw=1.0)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Input $u(t)$")
    ax1.set_title("(a) Composite chirp stimulus")

    # (b) PCA projection colored by time
    ax2 = fig.add_subplot(gs[0, 1])
    sc = ax2.scatter(proj_pca[:, 0], proj_pca[:, 1], c=times, cmap="viridis",
                     s=15, alpha=0.7, edgecolors="none")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("(b) PCA projection of reservoir features")
    plt.colorbar(sc, ax=ax2, label="Time (s)")

    # (c) t-SNE projection
    ax3 = fig.add_subplot(gs[0, 2])
    sc2 = ax3.scatter(proj_tsne[:, 0], proj_tsne[:, 1],
                      c=times[idx_tsne], cmap="viridis",
                      s=15, alpha=0.7, edgecolors="none")
    ax3.set_xlabel("t-SNE 1")
    ax3.set_ylabel("t-SNE 2")
    ax3.set_title("(c) t-SNE projection (subsampled)")
    plt.colorbar(sc2, ax=ax3, label="Time (s)")

    _savefig(fig, "fig05_reservoir_projection", outdir)


# ---------------------------------------------------------------------------
# Supplementary: Arnold tongue scan (frequency vs. input amplitude)
# ---------------------------------------------------------------------------

def fig_arnold_tongue(
    f0: float = 1.0,
    dt: float = 0.01,
    duration: float = 20.0,
    outdir: str = "figures",
):
    """
    Scan driving frequency and amplitude to show Arnold tongue-like
    synchronization regions for a single NPU.
    """
    set_seed(6)
    npu = SingleNPU(f0=f0, tau_ca=1.0, gain=1.0, sigma=0.005, dt=dt)

    freqs = np.linspace(0.3, 2.5, 30)
    amps = np.linspace(0.0, 1.0, 30)
    sync_map = np.empty((len(amps), len(freqs)))

    for i, amp in enumerate(amps):
        for j, fd in enumerate(freqs):
            npu.reset(v0=0.1, w0=0.1, c0=0.1)
            t_arr = np.arange(0, duration, dt)
            I = amp * np.sin(2.0 * np.pi * fd * t_arr)
            out = np.empty_like(t_arr)
            for k in range(len(t_arr)):
                out[k] = npu.step(I[k])

            # Synchronization index = ratio of dominant response freq to drive freq
            yf = np.fft.rfft(out[-int(5.0 / dt):] * np.hanning(int(5.0 / dt)))
            xf = np.fft.rfftfreq(int(5.0 / dt), d=dt)
            dom_freq = xf[np.argmax(np.abs(yf[1:])) + 1]
            # Normalized deviation
            sync_map[i, j] = np.abs(dom_freq - fd) / fd if fd > 0.1 else 0.0

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(
        sync_map,
        origin="lower",
        aspect="auto",
        extent=[freqs[0], freqs[-1], amps[0], amps[-1]],
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=0.5,
    )
    ax.set_xlabel("Drive frequency (Hz)")
    ax.set_ylabel("Drive amplitude")
    ax.set_title("Arnold tongue: normalized frequency deviation")
    plt.colorbar(im, ax=ax, label="Normalized frequency deviation")
    _savefig(fig, "figS1_arnold_tongue", outdir)

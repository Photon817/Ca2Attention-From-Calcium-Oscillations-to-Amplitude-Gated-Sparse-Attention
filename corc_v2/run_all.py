#!/usr/bin/env python3
"""
run_all.py — CORC v2
====================
One-click benchmark runner for CORC v2.
Produces figures under ../figures_v2/ and summary JSON.

Usage:
    python3 -m corc_v2.run_all
"""

from __future__ import annotations
import os, sys, json, gc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corc_v2.reservoir import CEUReservoir
from corc_v2.baselines import ESN, HopfV1, CEUNoPulse, LinearBaseline
from corc_v2.coupling import PulseCoupling
from corc_v2.units import CalciumExcitableUnit
from corc_v2.tasks import (
    RhythmClassification, ComplexRhythmClassification, TemporalXOR,
    MemoryCapacity, NARMA10, NARMA20, train_readout,
    HardRhythmClassification, PhaseNoiseClassification,
)
from corc_v2.observables import (
    build_feature_vector, delay_embed, virtual_node_embed,
)
from corc_v2.analysis import critical_scan
from corc_v2.plots import (
    plot_ceu_single_node, plot_ceu_phase_portrait,
    plot_avalanche_stats, plot_critical_performance,
    plot_main_results, plot_memory_capacity,
    plot_ablation_matrix, plot_state_projection, plot_virtual_node_comparison,
    plot_robustness_input_noise, plot_robustness_dropout, plot_robustness_drift,
    plot_robustness_summary,
)

SEED = 42
RNG = np.random.default_rng(SEED)
DT = 0.01
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures_v2")
os.makedirs(FIG_DIR, exist_ok=True)


# ===================================================================
# Builders
# ===================================================================

def build_ceu(N=32, seed=None, **overrides):
    """Build a CEU reservoir with default parameters (now tuned for excitability)."""
    rng = np.random.default_rng(seed)
    units = CalciumExcitableUnit.heterogeneous(N, DT, rng=rng)
    coupling = PulseCoupling(N, DT, rng=rng, **overrides)
    input_matrix = rng.uniform(0.1, 1.0, size=(N, 1))
    input_gain = rng.uniform(0.3, 1.5, size=N)
    return CEUReservoir(units, coupling, input_matrix, input_gain)


def extract_features_ceu(reservoir, inputs, taps=5, interval=4, window_steps=50):
    """Run CEU reservoir and extract features. Returns raw [c,s,a] concatenation (float32)."""
    states, events = reservoir.run(inputs, reset=True)
    c_arr = np.stack([s.c for s in states], axis=0).astype(np.float32)
    s_arr = np.stack([s.s for s in states], axis=0).astype(np.float32)
    a_arr = np.stack([s.a for s in states], axis=0).astype(np.float32)
    X = np.concatenate([c_arr, s_arr, a_arr], axis=1)
    return X, events


def extract_features_hopf_v1(reservoir, inputs, taps=5, interval=4):
    """Run Hopf v1 reservoir and extract delay-embedded features."""
    from corc_blackbox.observables import delay_embed as de_v1
    states, events = reservoir.run(inputs, reset=True)
    X = de_v1(states, features=["x", "y", "a", "r", "s"], taps=taps,
              interval_steps=interval)
    return X, events


def _trajectory_readout(X_traj: np.ndarray) -> np.ndarray:
    """
    Convert a full trajectory feature matrix (T, D) into a fixed-size
    summary vector by concatenating: mean of 3 equal segments + final_state.
    This preserves temporal dynamics for classification.
    """
    T = X_traj.shape[0]
    seg1 = X_traj[:T//3].mean(axis=0)
    seg2 = X_traj[T//3:2*T//3].mean(axis=0)
    seg3 = X_traj[2*T//3:].mean(axis=0)
    return np.concatenate([seg1, seg2, seg3, X_traj[-1]])


# ===================================================================
# Step 1: Single-node dynamics demo
# ===================================================================

def run_single_node_demo():
    print("[1/8] CEU single-node dynamics...")
    res = build_ceu(N=8, seed=SEED)
    T = 2000
    t_arr = np.arange(T) * DT
    u = 0.4 * np.sin(2 * np.pi * 1.0 * t_arr) + RNG.normal(0, 0.03, T)
    states, events = res.run(u, reset=True)

    c_arr = np.stack([s.c for s in states], axis=0)
    s_arr = np.stack([s.s for s in states], axis=0)
    a_arr = np.stack([s.a for s in states], axis=0)

    plot_ceu_single_node(states, events, DT,
                         path=os.path.join(FIG_DIR, "01_ceu_dynamics.png"))
    plot_ceu_phase_portrait(c_arr, s_arr, a_arr, node_idx=0,
                            path=os.path.join(FIG_DIR, "01_ceu_phase.png"))
    print("      Saved 01_ceu_*.png")


# ===================================================================
# Step 2: Avalanche statistics
# ===================================================================

def run_avalanche_stats():
    print("[2/8] Network avalanche statistics...")
    res = build_ceu(N=32, seed=SEED)
    T = 3000
    u = 0.3 * np.sin(2 * np.pi * 1.0 * np.arange(T) * DT) + RNG.normal(0, 0.05, T)
    states, events = res.run(u, reset=True)
    c_arr = np.stack([s.c for s in states], axis=0)
    plot_avalanche_stats(events, c_arr,
                         path=os.path.join(FIG_DIR, "02_avalanche_stats.png"))
    print("      Saved 02_avalanche_stats.png")


# ===================================================================
# Step 3: Critical coupling scan
# ===================================================================

def run_critical_scan():
    print("[3/8] Critical coupling scan...")
    g_values = np.linspace(0.0, 0.4, 9)

    def factory(g):
        return build_ceu(N=32, seed=SEED, g_p=g)
    scan = critical_scan(factory, g_values, T=2000, dt=DT)

    # Quick performance at each g: NARMA and XOR
    narma_perf = []
    xor_perf = []
    task_narma = NARMA10(dt=DT, rng=RNG)
    u_n, y_n = task_narma.generate(T=2000)
    task_xor = TemporalXOR(dt=DT, rng=RNG)
    u_x, t_x = task_xor.generate(n_trials=100)
    n_train_x = 70

    for g in g_values:
        res = factory(g)
        # NARMA — full trajectory using raw states
        X_n, _ = extract_features_ceu(res, u_n)
        split = 1500
        _, met_n = train_readout(X_n[:split], y_n[:split], X_n[split:], y_n[split:],
                                 task_type="regression", alpha=100.0)
        narma_perf.append(met_n["nmse"])
        # XOR — trajectory summary
        X_x_summary = []
        for i in range(len(t_x)):
            X_traj, _ = extract_features_ceu(res, u_x[i])
            X_x_summary.append(_trajectory_readout(X_traj))
        X_x_all = np.stack(X_x_summary)
        _, met_x = train_readout(X_x_all[:n_train_x], t_x[:n_train_x],
                                 X_x_all[n_train_x:], t_x[n_train_x:],
                                 task_type="classification")
        xor_perf.append(met_x["accuracy"])

    perf = {"NARMA NMSE": np.array(narma_perf), "XOR Accuracy": np.array(xor_perf)}
    plot_critical_performance(scan, g_values, perf,
                              path=os.path.join(FIG_DIR, "03_critical_performance.png"))
    print("      Saved 03_critical_performance.png")
    return scan, perf


# ===================================================================
# Step 4: Main benchmark (classification)
# ===================================================================

def benchmark_classification(task, task_name, N=32):
    """
    Run classification benchmark using trajectory summary readout.
    Uses per-trial reservoir re-initialization (no_events approach)
    for CEU v2 to eliminate cross-trial state noise.
    """
    inputs, targets = task.generate()
    n_trials = len(targets)
    n_train = int(0.7 * n_trials)

    results = {}

    # --- CEU v2 (per-trial fresh reservoir, same seed) ---
    X_ceu = []
    for i in range(n_trials):
        ceu = build_ceu(N=N, seed=SEED)
        X_traj, _ = extract_features_ceu(ceu, inputs[i])
        X_ceu.append(_trajectory_readout(X_traj))
        del ceu
    X_all = np.stack(X_ceu)
    _, met = train_readout(X_all[:n_train], targets[:n_train],
                           X_all[n_train:], targets[n_train:],
                           task_type="classification")
    results["CEU v2"] = met["accuracy"]
    del X_ceu, X_all

    # --- ESN (per-trial fresh) ---
    X_esn = []
    for i in range(n_trials):
        esn = ESN(N=N*2, input_dim=1, seed=SEED)
        S = esn.run(inputs[i])
        X_esn.append(_trajectory_readout(S))
        del esn
    X_esn_all = np.stack(X_esn)
    _, met_e = train_readout(X_esn_all[:n_train], targets[:n_train],
                             X_esn_all[n_train:], targets[n_train:],
                             task_type="classification")
    results["ESN"] = met_e["accuracy"]
    del X_esn, X_esn_all

    # --- Hopf v1 ---
    try:
        hopf = HopfV1(N=N, seed=SEED)
        X_h = []
        for i in range(n_trials):
            X_hv, _ = extract_features_hopf_v1(hopf, inputs[i])
            X_h.append(_trajectory_readout(X_hv))
        X_h = np.stack(X_h)
        _, met_h = train_readout(X_h[:n_train], targets[:n_train],
                                 X_h[n_train:], targets[n_train:],
                                 task_type="classification")
        results["Hopf v1"] = met_h["accuracy"]
    except Exception:
        results["Hopf v1"] = float("nan")

    # --- No-pulse CEU (per-trial) ---
    X_np = []
    for i in range(n_trials):
        no_p = CEUNoPulse(N=N, seed=SEED)
        X_npv, _ = extract_features_ceu(no_p, inputs[i])
        X_np.append(_trajectory_readout(X_npv))
        del no_p
    X_np_all = np.stack(X_np)
    _, met_np = train_readout(X_np_all[:n_train], targets[:n_train],
                              X_np_all[n_train:], targets[n_train:],
                              task_type="classification")
    results["CEU no-pulse"] = met_np["accuracy"]
    del X_np, X_np_all

    # --- Linear baseline ---
    lb = LinearBaseline(input_dim=1, delay_taps=10, delay_step=3)
    X_lb = []
    for i in range(n_trials):
        X_l = lb.run(inputs[i])
        X_lb.append(_trajectory_readout(X_l))
    X_lb = np.stack(X_lb)
    _, met_lb = train_readout(X_lb[:n_train], targets[:n_train],
                              X_lb[n_train:], targets[n_train:],
                              task_type="classification")
    results["Linear"] = met_lb["accuracy"]

    return results


def run_main_benchmark():
    print("[4/8] Main benchmark...")
    tasks = {
        "Rhythm": RhythmClassification(dt=DT, rng=RNG),
        "ComplexRhythm": ComplexRhythmClassification(dt=DT, rng=RNG, n_classes=3),
        "TempXOR": TemporalXOR(dt=DT, rng=RNG),
    }
    all_results = {}
    for name, task in tasks.items():
        all_results[name] = benchmark_classification(task, name, N=32)

    methods = ["CEU v2", "ESN", "Hopf v1", "CEU no-pulse", "Linear"]
    metrics = {m: {t: all_results[t].get(m, 0) for t in tasks} for m in methods}
    plot_main_results(metrics, list(tasks.keys()), metric_key="accuracy",
                      path=os.path.join(FIG_DIR, "04_main_results.png"))
    print("      Saved 04_main_results.png")
    return all_results


# ===================================================================
# Step 5: NARMA-10 regression
# ===================================================================

def run_narma():
    print("[5/8] NARMA-10 regression...")
    task = NARMA10(dt=DT, rng=RNG)
    u, y = task.generate(T=4000)
    split = 3000

    results = {}

    # CEU v2 (with time rescaling for speed)
    ceu = build_ceu(N=64, seed=SEED)
    X_c, _ = extract_features_ceu(ceu, u)
    _, met_c = train_readout(X_c[:split], y[:split], X_c[split:], y[split:],
                             task_type="regression", alpha=100.0)
    results["CEU v2"] = met_c

    # CEU v2 + poly
    _, met_cp = train_readout(X_c[:split], y[:split], X_c[split:], y[split:],
                               task_type="regression", use_poly=True, alpha=100.0)
    results["CEU v2+poly"] = met_cp

    # ESN (bigger)
    esn = ESN(N=128, input_dim=1, seed=SEED, leak=0.3)
    Se = esn.run(u)
    _, met_e = train_readout(Se[:split], y[:split], Se[split:], y[split:],
                             task_type="regression", alpha=1e-4)
    results["ESN"] = met_e

    # Hopf v1
    try:
        hopf = HopfV1(N=32, seed=SEED)
        X_h, _ = extract_features_hopf_v1(hopf, u)
        _, met_h = train_readout(X_h[:split], y[:split], X_h[split:], y[split:],
                                 task_type="regression", alpha=10.0)
        results["Hopf v1"] = met_h
    except Exception:
        results["Hopf v1"] = {"nmse": float("nan"), "mse": float("nan")}

    # Linear
    lb = LinearBaseline(input_dim=1, delay_taps=10, delay_step=3)
    X_lb = lb.run(u)
    _, met_lb = train_readout(X_lb[:split], y[:split], X_lb[split:], y[split:],
                              task_type="regression", alpha=1e-4)
    results["Linear"] = met_lb

    print(f"      CEU v2       NMSE: {results['CEU v2']['nmse']:.4f}")
    print(f"      CEU v2+poly  NMSE: {results['CEU v2+poly']['nmse']:.4f}")
    print(f"      ESN          NMSE: {results['ESN']['nmse']:.4f}")
    print(f"      Linear       NMSE: {results['Linear']['nmse']:.4f}")

    # Bar chart for NARMA
    methods_n = ["CEU v2", "CEU v2+poly", "ESN", "Linear"]
    narma_metrics = {m: {"NARMA-10": results[m]["nmse"]} for m in methods_n
                     if not np.isnan(results[m]["nmse"])}
    plot_main_results(narma_metrics, ["NARMA-10"], metric_key="NMSE",
                      path=os.path.join(FIG_DIR, "05_narma_bars.png"))
    print("      Saved 05_narma_bars.png")
    return results


# ===================================================================
# Step 6: Memory capacity
# ===================================================================

def run_memory_capacity():
    print("[6/8] Memory capacity...")
    mc = MemoryCapacity(dt=DT, max_lag=30, rng=RNG)
    u, _ = mc.generate(n_samples=6000)

    mc_results = {}

    # CEU v2
    ceu = build_ceu(N=64, seed=SEED)
    X_c, _ = extract_features_ceu(ceu, u)
    mc_c = MemoryCapacity.compute_mc(X_c, u[:, 0], max_lag=30)
    mc_results["CEU v2"] = mc_c

    # ESN
    esn = ESN(N=64, input_dim=1, seed=SEED)
    Se = esn.run(u)
    mc_e = MemoryCapacity.compute_mc(Se, u[:, 0], max_lag=30)
    mc_results["ESN"] = mc_e

    print(f"      CEU v2 total MC: {mc_c['total']:.2f}")
    print(f"      ESN    total MC: {mc_e['total']:.2f}")

    plot_memory_capacity(mc_results, path=os.path.join(FIG_DIR, "06_memory_capacity.png"))
    print("      Saved 06_memory_capacity.png")
    return mc_results


# ===================================================================
# Step 7: Ablation
# ===================================================================

def run_ablation():
    print("[7/8] Ablation experiments...")
    ablation_scores = {}

    # --- XOR ablation ---
    task_xor = TemporalXOR(dt=DT, rng=RNG)
    inputs_xor, targets_xor = task_xor.generate(n_trials=150)
    n_train_x = 105

    def score_ablation_xor(res):
        X_summary = []
        for i in range(len(targets_xor)):
            X_traj, _ = extract_features_ceu(res, inputs_xor[i])
            X_summary.append(_trajectory_readout(X_traj))
        X_all = np.stack(X_summary)
        _, met = train_readout(X_all[:n_train_x], targets_xor[:n_train_x],
                               X_all[n_train_x:], targets_xor[n_train_x:],
                               task_type="classification")
        return met["accuracy"]

    # Full model
    ablation_scores["full"] = {"XOR": score_ablation_xor(build_ceu(N=32, seed=SEED))}

    # No pulse
    ablation_scores["no_pulse"] = {"XOR": score_ablation_xor(build_ceu(N=32, seed=SEED, g_p=0.0))}

    # No adaptation
    rng = np.random.default_rng(SEED)
    units = CalciumExcitableUnit.heterogeneous(32, DT, rng=rng, alpha_a_range=(0, 0))
    coup = PulseCoupling(32, DT, rng=rng)
    imat = rng.uniform(0.1, 1.0, size=(32, 1))
    igain = rng.uniform(0.3, 1.5, size=32)
    res_no_adapt = CEUReservoir(units, coup, imat, igain)
    ablation_scores["no_adapt"] = {"XOR": score_ablation_xor(res_no_adapt)}

    # No amp modulation (eta=0)
    ablation_scores["eta=0"] = {"XOR": score_ablation_xor(build_ceu(N=32, seed=SEED, eta=0.0))}

    # Homogeneous
    rng2 = np.random.default_rng(SEED)
    units_h = CalciumExcitableUnit.homogeneous(32, DT, rng=rng2)
    coup_h = PulseCoupling(32, DT, rng=rng2)
    imat_h = rng2.uniform(0.1, 1.0, size=(32, 1))
    igain_h = rng2.uniform(0.3, 1.5, size=32)
    res_homo = CEUReservoir(units_h, coup_h, imat_h, igain_h)
    ablation_scores["homogeneous"] = {"XOR": score_ablation_xor(res_homo)}

    # No event features (use only c, s, a — same as extract_features_ceu)
    def extract_features_no_events(reservoir, inputs):
        states, events = reservoir.run(inputs, reset=True)
        c_arr = np.stack([s.c for s in states], axis=0)
        s_arr = np.stack([s.s for s in states], axis=0)
        a_arr = np.stack([s.a for s in states], axis=0)
        X = np.concatenate([c_arr, s_arr, a_arr], axis=1)
        return X

    X_noev_list = []
    for i in range(len(targets_xor)):
        ceu_tmp = build_ceu(N=32, seed=SEED)
        X_noev = extract_features_no_events(ceu_tmp, inputs_xor[i])
        X_noev_list.append(_trajectory_readout(X_noev))
    X_noev_all = np.stack(X_noev_list)
    _, met_ne = train_readout(X_noev_all[:n_train_x], targets_xor[:n_train_x],
                               X_noev_all[n_train_x:], targets_xor[n_train_x:],
                               task_type="classification")
    ablation_scores["no_events"] = {"XOR": met_ne["accuracy"]}

    # --- NARMA ablation ---
    task_n = NARMA10(dt=DT, rng=RNG)
    u_n, y_n = task_n.generate(T=2000)
    split = 1500

    def score_ablation_narma(res):
        X_n, _ = extract_features_ceu(res, u_n)
        _, met_n = train_readout(X_n[:split], y_n[:split], X_n[split:], y_n[split:],
                                 task_type="regression", alpha=100.0)
        return met_n["nmse"]

    ablation_scores["full"]["NARMA"] = score_ablation_narma(build_ceu(N=32, seed=SEED))
    ablation_scores["no_pulse"]["NARMA"] = score_ablation_narma(build_ceu(N=32, seed=SEED, g_p=0.0))
    ablation_scores["no_adapt"]["NARMA"] = score_ablation_narma(res_no_adapt)
    ablation_scores["eta=0"]["NARMA"] = score_ablation_narma(build_ceu(N=32, seed=SEED, eta=0.0))
    ablation_scores["homogeneous"]["NARMA"] = score_ablation_narma(res_homo)

    # no_events for NARMA
    X_noev_n = extract_features_no_events(build_ceu(N=64, seed=SEED), u_n)
    _, met_nen = train_readout(X_noev_n[:split], y_n[:split], X_noev_n[split:], y_n[split:],
                               task_type="regression", alpha=10.0)
    ablation_scores["no_events"]["NARMA"] = met_nen["nmse"]

    plot_ablation_matrix(ablation_scores,
                         path=os.path.join(FIG_DIR, "07_ablation_matrix.png"))
    print("      Saved 07_ablation_matrix.png")
    return ablation_scores


# ===================================================================
# Step 8: Projections
# ===================================================================

def run_projections():
    print("[8/8] State projections...")
    T = 1500
    t_arr = np.arange(T) * DT
    u = 0.4 * np.sin(2 * np.pi * 1.0 * t_arr)

    X_list = []

    # CEU v2
    ceu = build_ceu(N=32, seed=SEED)
    states, _ = ceu.run(u, reset=True)
    c_arr = np.stack([s.c for s in states], axis=0)
    s_arr = np.stack([s.s for s in states], axis=0)
    a_arr = np.stack([s.a for s in states], axis=0)
    X_c = np.concatenate([c_arr, s_arr, a_arr], axis=1)
    X_list.append({"X": X_c, "label": "CEU v2", "colors": u})

    # CEU no-pulse
    ceu_np = build_ceu(N=32, seed=SEED, g_p=0.0)
    states_np, _ = ceu_np.run(u, reset=True)
    c_np = np.stack([s.c for s in states_np], axis=0)
    s_np = np.stack([s.s for s in states_np], axis=0)
    a_np = np.stack([s.a for s in states_np], axis=0)
    X_np = np.concatenate([c_np, s_np, a_np], axis=1)
    X_list.append({"X": X_np, "label": "CEU no pulse", "colors": u})

    # Hopf v1 (compare)
    try:
        hopf = HopfV1(N=32, seed=SEED)
        states_h, _ = hopf.run(u, reset=True)
        from corc_blackbox.observables import delay_embed as de_v1
        X_h = de_v1(states_h, features=["x", "y", "a", "r", "s"], taps=3, interval_steps=4)
        X_list.append({"X": X_h, "label": "Hopf v1", "colors": u})
    except Exception:
        pass

    plot_state_projection(X_list, path=os.path.join(FIG_DIR, "08_projections.png"))
    print("      Saved 08_projections.png")

    # Virtual node comparison
    vn_results = {}
    task_xor = TemporalXOR(dt=DT, rng=RNG)
    in_x, t_x = task_xor.generate(n_trials=100)
    n_train_v = 70

    # 8 CEU + 40 virtual taps
    ceu_small = build_ceu(N=8, seed=SEED)
    X_vn_list = []
    for i in range(len(t_x)):
        st_vn, _ = ceu_small.run(in_x[i], reset=True)
        X_vn = virtual_node_embed(st_vn, n_real=8, n_virtual_per_real=40, interval_steps=4)
        X_vn_list.append(_trajectory_readout(X_vn))
    X_vn_all = np.stack(X_vn_list)
    _, met_vn = train_readout(X_vn_all[:n_train_v], t_x[:n_train_v],
                               X_vn_all[n_train_v:], t_x[n_train_v:],
                               task_type="classification")
    vn_results["8 CEU+40 virtual"] = {"TempXOR": met_vn["accuracy"]}

    # 32 CEU + 5 taps (baseline)
    X_32_list = []
    for i in range(len(t_x)):
        X_32_traj, _ = extract_features_ceu(build_ceu(N=32, seed=SEED), in_x[i])
        X_32_list.append(_trajectory_readout(X_32_traj))
    X_32_all = np.stack(X_32_list)
    _, met_32 = train_readout(X_32_all[:n_train_v], t_x[:n_train_v],
                               X_32_all[n_train_v:], t_x[n_train_v:],
                               task_type="classification")
    vn_results["32 CEU+5 taps"] = {"TempXOR": met_32["accuracy"]}

    # 128 CEU no taps
    X_128_list = []
    for i in range(len(t_x)):
        ceu_128 = build_ceu(N=128, seed=SEED)
        X_128_traj, _ = extract_features_ceu(ceu_128, in_x[i])
        X_128_list.append(_trajectory_readout(X_128_traj))
    X_128_all = np.stack(X_128_list)
    _, met_128 = train_readout(X_128_all[:n_train_v], t_x[:n_train_v],
                               X_128_all[n_train_v:], t_x[n_train_v:],
                               task_type="classification")
    vn_results["128 CEU"] = {"TempXOR": met_128["accuracy"]}

    plot_virtual_node_comparison(vn_results,
                                 path=os.path.join(FIG_DIR, "08_virtual_nodes.png"))
    print("      Saved 08_virtual_nodes.png")
    return vn_results


# ===================================================================
# Step 9: Hard tasks (close-frequency rhythm + phase noise)
# ===================================================================

def run_hard_tasks():
    print("[9/10] Hard tasks benchmark...")

    tasks = {
        "HardRhythm_2vs2.3Hz": HardRhythmClassification(dt=DT, rng=RNG),
        "PhaseNoise_3class": PhaseNoiseClassification(dt=DT, rng=RNG, n_classes=3),
    }
    all_results = {}
    for name, task in tasks.items():
        print(f"      Running {name}...")
        all_results[name] = benchmark_classification(task, name, N=32)
        gc.collect()

    methods = ["CEU v2", "ESN", "Hopf v1", "CEU no-pulse", "Linear"]
    metrics = {m: {t: all_results[t].get(m, 0) for t in tasks} for m in methods}
    plot_main_results(metrics, list(tasks.keys()), metric_key="accuracy",
                      path=os.path.join(FIG_DIR, "09_hard_tasks.png"))
    print("      Saved 09_hard_tasks.png")

    for tname, tres in all_results.items():
        for mname, val in tres.items():
            print(f"        {tname} / {mname}: {val:.3f}")
    return all_results


# ===================================================================
# Step 10: Robustness experiments
# ===================================================================

def run_robustness():
    print("[10/10] Robustness experiments...")
    robust = {}

    # --- Task setup ---
    hard_rhythm = HardRhythmClassification(dt=DT, rng=RNG)
    u_hr, t_hr = hard_rhythm.generate(n_trials=80)
    n_train = 56
    narma_task = NARMA10(dt=DT, rng=RNG)
    u_nr, y_nr = narma_task.generate(T=1500)
    split_nr = 1100

    # --------- 10a: Input noise sweep ---------
    print("      10a: Input noise sweep...")
    noise_levels = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3])
    ceu_noise_acc, esn_noise_acc = [], []
    ceu_noise_nmse, esn_noise_nmse = [], []

    for sigma_n in noise_levels:
        # Hard rhythm with added noise
        u_noisy = u_hr + np.random.default_rng(SEED).normal(0, sigma_n, u_hr.shape).astype(np.float32)
        br = benchmark_classification_on_data(u_noisy, t_hr, "HardRhythm", N=32)
        ceu_noise_acc.append(br["CEU v2"])
        esn_noise_acc.append(br["ESN"])
        del br; gc.collect()

    for sigma_n in noise_levels:
        # NARMA with noisy input
        u_n_noisy = u_nr + np.random.default_rng(SEED).normal(0, sigma_n * 0.5, u_nr.shape).astype(np.float32)
        nr = benchmark_narma_on_data(u_n_noisy, y_nr, split_nr)
        ceu_noise_nmse.append(nr["CEU v2"])
        esn_noise_nmse.append(nr["ESN"])
        del nr; gc.collect()

    plot_robustness_input_noise(noise_levels, np.array(ceu_noise_acc), np.array(esn_noise_acc),
                                "HardRhythm", "Accuracy",
                                path=os.path.join(FIG_DIR, "10a_noise_rhythm.png"))
    plot_robustness_input_noise(noise_levels, np.array(ceu_noise_nmse), np.array(esn_noise_nmse),
                                "NARMA-10", "NMSE",
                                path=os.path.join(FIG_DIR, "10a_noise_narma.png"))
    # Relative retention at 50% noise
    r_n50_ceu = ceu_noise_acc[2] / max(ceu_noise_acc[0], 1e-9)
    r_n50_esn = esn_noise_acc[2] / max(esn_noise_acc[0], 1e-9)
    robust["input_noise"] = {"HardRhythm_CEU": min(r_n50_ceu, 2.0), "HardRhythm_ESN": min(r_n50_esn, 2.0)}
    print(f"        Noise 50% retention: CEU={r_n50_ceu:.3f}, ESN={r_n50_esn:.3f}")

    # --------- 10b: Node dropout sweep ---------
    print("      10b: Node dropout sweep...")
    dropout_rates = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ceu_drop_acc, esn_drop_acc = [], []

    for dr in dropout_rates:
        br = benchmark_classification_on_data_dropout(u_hr, t_hr, "HardRhythm", N=32, dropout=dr)
        ceu_drop_acc.append(br["CEU v2"])
        esn_drop_acc.append(br["ESN"])
        del br; gc.collect()

    plot_robustness_dropout(dropout_rates, np.array(ceu_drop_acc), np.array(esn_drop_acc),
                            "HardRhythm", "Accuracy",
                            path=os.path.join(FIG_DIR, "10b_dropout.png"))
    r_d50_ceu = ceu_drop_acc[5] / max(ceu_drop_acc[0], 1e-9)
    r_d50_esn = esn_drop_acc[5] / max(esn_drop_acc[0], 1e-9)
    robust["dropout"] = {"HardRhythm_CEU": min(r_d50_ceu, 2.0), "HardRhythm_ESN": min(r_d50_esn, 2.0)}
    print(f"        Dropout 50% retention: CEU={r_d50_ceu:.3f}, ESN={r_d50_esn:.3f}")

    # --------- 10c: Parameter drift ---------
    print("      10c: Parameter drift sweep...")
    drift_scales = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    ceu_drift_acc, esn_drift_acc = [], []

    for drift in drift_scales:
        br = benchmark_classification_on_data_drift(u_hr, t_hr, "HardRhythm", N=32, drift_scale=drift)
        ceu_drift_acc.append(br["CEU v2"])
        esn_drift_acc.append(br["ESN"])
        del br; gc.collect()

    plot_robustness_drift(drift_scales, np.array(ceu_drift_acc), np.array(esn_drift_acc),
                          "HardRhythm", "Accuracy",
                          path=os.path.join(FIG_DIR, "10c_drift.png"))
    r_p50_ceu = ceu_drift_acc[-1] / max(ceu_drift_acc[0], 1e-9)
    r_p50_esn = esn_drift_acc[-1] / max(esn_drift_acc[0], 1e-9)
    robust["param_drift"] = {"HardRhythm_CEU": min(r_p50_ceu, 2.0), "HardRhythm_ESN": min(r_p50_esn, 2.0)}
    print(f"        Drift 0.3 retention: CEU={r_p50_ceu:.3f}, ESN={r_p50_esn:.3f}")

    # Summary plot
    plot_robustness_summary(robust,
                            path=os.path.join(FIG_DIR, "10_robustness_summary.png"))
    print("      Saved 10_*.png")
    return robust


# -------------------------------------------------------------------
# Robustness helpers: run benchmark on pre-generated data
# -------------------------------------------------------------------

def benchmark_classification_on_data(inputs, targets, label, N=32):
    """Same as benchmark_classification but takes pre-generated data."""
    n_trials = len(targets)
    n_train = int(0.7 * n_trials)
    results = {}

    # CEU v2
    X_ceu = []
    for i in range(n_trials):
        ceu = build_ceu(N=N, seed=SEED)
        X_traj, _ = extract_features_ceu(ceu, inputs[i])
        X_ceu.append(_trajectory_readout(X_traj))
        del ceu
    X_all = np.stack(X_ceu)
    _, met = train_readout(X_all[:n_train], targets[:n_train],
                           X_all[n_train:], targets[n_train:],
                           task_type="classification")
    results["CEU v2"] = met["accuracy"]
    del X_ceu, X_all

    # ESN
    X_esn = []
    for i in range(n_trials):
        esn = ESN(N=N*2, input_dim=1, seed=SEED)
        S = esn.run(inputs[i])
        X_esn.append(_trajectory_readout(S))
        del esn
    X_esn_all = np.stack(X_esn)
    _, met_e = train_readout(X_esn_all[:n_train], targets[:n_train],
                             X_esn_all[n_train:], targets[n_train:],
                             task_type="classification")
    results["ESN"] = met_e["accuracy"]
    return results


def benchmark_classification_on_data_dropout(inputs, targets, label, N=32, dropout=0.2):
    """Benchmark with random node dropout (nodes silenced during inference)."""
    n_trials = len(targets)
    n_train = int(0.7 * n_trials)
    results = {}

    # CEU v2 with dropout
    X_ceu = []
    for i in range(n_trials):
        ceu = build_ceu(N=N, seed=SEED)
        # Randomly select nodes to silence
        rng_drop = np.random.default_rng(SEED + i + 1000)
        mask = (rng_drop.random(N) > dropout).astype(np.float32)
        states, _ = ceu.run(inputs[i], reset=True)
        c_arr = np.stack([s.c * mask for s in states], axis=0).astype(np.float32)
        s_arr = np.stack([s.s * mask for s in states], axis=0).astype(np.float32)
        a_arr = np.stack([s.a * mask for s in states], axis=0).astype(np.float32)
        X_traj = np.concatenate([c_arr, s_arr, a_arr], axis=1)
        X_ceu.append(_trajectory_readout(X_traj))
        del ceu
    results["CEU v2"] = train_readout(
        np.stack(X_ceu)[:n_train], targets[:n_train],
        np.stack(X_ceu)[n_train:], targets[n_train:],
        task_type="classification")[1]["accuracy"]

    # ESN with dropout
    X_esn = []
    for i in range(n_trials):
        esn = ESN(N=N*2, input_dim=1, seed=SEED)
        rng_drop = np.random.default_rng(SEED + i + 2000)
        mask = (rng_drop.random(N*2) > dropout).astype(np.float32)
        S = esn.run(inputs[i])
        S = S * mask
        X_esn.append(_trajectory_readout(S))
        del esn
    results["ESN"] = train_readout(
        np.stack(X_esn)[:n_train], targets[:n_train],
        np.stack(X_esn)[n_train:], targets[n_train:],
        task_type="classification")[1]["accuracy"]
    return results


def benchmark_classification_on_data_drift(inputs, targets, label, N=32, drift_scale=0.1):
    """Benchmark with slow parameter drift (c_th drifts linearly over time)."""
    n_trials = len(targets)
    n_train = int(0.7 * n_trials)
    results = {}

    # CEU v2 with drift: c_th shifts over the trial
    X_ceu = []
    T = inputs.shape[1]
    for i in range(n_trials):
        ceu = build_ceu(N=N, seed=SEED)
        # Drift: linearly shift c_th over the trial duration
        drift = np.linspace(0, drift_scale, T)
        orig_c_th = ceu.units.c_th.copy()
        states_list = []
        for t in range(T):
            ceu.units.c_th = np.clip(orig_c_th + drift[t], 0.05, 0.95)
            st = ceu.step(inputs[i, t])
            states_list.append(st)
        ceu.units.c_th = orig_c_th
        c_arr = np.stack([s.c for s in states_list], axis=0).astype(np.float32)
        s_arr = np.stack([s.s for s in states_list], axis=0).astype(np.float32)
        a_arr = np.stack([s.a for s in states_list], axis=0).astype(np.float32)
        X_traj = np.concatenate([c_arr, s_arr, a_arr], axis=1)
        X_ceu.append(_trajectory_readout(X_traj))
        del ceu
    results["CEU v2"] = train_readout(
        np.stack(X_ceu)[:n_train], targets[:n_train],
        np.stack(X_ceu)[n_train:], targets[n_train:],
        task_type="classification")[1]["accuracy"]

    # ESN with drift: input_scaling drifts
    X_esn = []
    for i in range(n_trials):
        esn = ESN(N=N*2, input_dim=1, seed=SEED)
        rng_drift = np.random.default_rng(SEED + i + 3000)
        # ESN drift: add random walk noise to W_in
        orig_W_in = esn.W_in.copy()
        drift_noise = np.cumsum(rng_drift.normal(0, drift_scale / np.sqrt(T), size=T))
        states_list = []
        for t in range(T):
            esn.W_in = orig_W_in * (1.0 + drift_noise[t])
            s_t = esn.step(inputs[i, t])
            states_list.append(s_t)
        esn.W_in = orig_W_in
        S = np.stack(states_list, axis=0).astype(np.float32)
        X_esn.append(_trajectory_readout(S))
        del esn
    results["ESN"] = train_readout(
        np.stack(X_esn)[:n_train], targets[:n_train],
        np.stack(X_esn)[n_train:], targets[n_train:],
        task_type="classification")[1]["accuracy"]
    return results


def benchmark_narma_on_data(u, y, split):
    """Quick NARMA benchmark on pre-generated data (CEU vs ESN only)."""
    results = {}
    # CEU v2
    ceu = build_ceu(N=64, seed=SEED)
    X_c, _ = extract_features_ceu(ceu, u)
    _, met_c = train_readout(X_c[:split], y[:split], X_c[split:], y[split:],
                             task_type="regression", alpha=100.0)
    results["CEU v2"] = met_c["nmse"]
    del ceu, X_c
    # ESN
    esn = ESN(N=128, input_dim=1, seed=SEED)
    Se = esn.run(u)
    _, met_e = train_readout(Se[:split], y[:split], Se[split:], y[split:],
                             task_type="regression", alpha=1e-4)
    results["ESN"] = met_e["nmse"]
    del esn, Se
    return results


# ===================================================================
# Main
# ===================================================================

def main():
    print("="*60)
    print("CORC v2 — Calcium-inspired Oscillatory Reservoir Computing")
    print("="*60)

    run_single_node_demo(); gc.collect()
    run_avalanche_stats(); gc.collect()
    scan_results, _ = run_critical_scan(); gc.collect()
    bench = run_main_benchmark(); gc.collect()
    narma = run_narma(); gc.collect()
    mc = run_memory_capacity(); gc.collect()
    ablation = run_ablation(); gc.collect()
    projections = run_projections(); gc.collect()
    hard_tasks = run_hard_tasks(); gc.collect()
    robustness = run_robustness(); gc.collect()

    # Save summary
    summary = {
        "classification": bench,
        "narma": {k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in narma.items() if not np.isnan(list(v.values())[0])},
        "memory_capacity": {k: {kk: float(vv) for kk, vv in v.items()}
                            for k, v in mc.items()},
        "ablation": {k: {kk: float(vv) for kk, vv in v.items()}
                     for k, v in ablation.items()},
        "hard_tasks": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in hard_tasks.items()},
        "robustness": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in robustness.items()},
    }
    with open(os.path.join(FIG_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    print("\n" + "="*60)
    print(f"All figures saved to: {FIG_DIR}")
    print(f"Summary JSON: {os.path.join(FIG_DIR, 'summary.json')}")
    print("Done.")


if __name__ == "__main__":
    main()
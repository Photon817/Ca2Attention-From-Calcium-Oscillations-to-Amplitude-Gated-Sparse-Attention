"""
run_experiments.py
==================
Master script to reproduce all figures and benchmark results.

Usage
-----
    python run_experiments.py

All PNG / PDF figures are written to ./figures/.
A text report is appended to ./results_report.txt.

The script follows the 8-item checklist from CLAUDE.md:
1. Single-NPU model demo               -> fig01
2. Multi-NPU synchronization           -> fig02
3. Task A + B performance bars         -> fig03
4. Robustness sweeps (noise/drift/coupling) -> fig04
5. Reservoir PCA / t-SNE               -> fig05
6. Baselines (ESN, raw) included in 3 & 4
7. Supplementary Arnold tongue         -> figS1
"""

from __future__ import annotations

import os
import time
import numpy as np
from typing import Dict

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

from tasks import (
    TaskAClassifier,
    TaskAESNBaseline,
    TaskARawBaseline,
    TaskBNARMA,
    TaskBESNBaseline,
    TaskBRawBaseline,
    run_taskA_multi_seed,
    run_taskB_multi_seed,
)

from analysis import (
    fig_single_npu_response,
    fig_multi_npu_sync,
    fig_performance_bars,
    fig_robustness_curves,
    fig_reservoir_projection,
    fig_arnold_tongue,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
N_NPU = 8
DT = 0.02  # 50 Hz sampling
N_SEEDS_MAIN = 10
N_SEEDS_ROBUST = 5  # smaller for speed; increase for paper

os.makedirs("figures", exist_ok=True)


def log(msg: str) -> None:
    print(f"[run_experiments] {msg}")
    with open("results_report.txt", "a") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# 1. Single-NPU response & phase portrait
# ---------------------------------------------------------------------------

def run_fig01() -> None:
    log("=== Fig 1: Single-NPU step response ===")
    fig_single_npu_response(dt=0.01, duration=30.0)
    log("Saved fig01_single_npu_response.{png,pdf}")


# ---------------------------------------------------------------------------
# 2. Multi-NPU synchronization
# ---------------------------------------------------------------------------

def run_fig02() -> None:
    log("=== Fig 2: Multi-NPU synchronization ===")
    fig_multi_npu_sync(N=N_NPU, dt=DT, duration=20.0)
    log("Saved fig02_multi_npu_sync.{png,pdf}")


# ---------------------------------------------------------------------------
# 3. Task A & B performance bars (main benchmarks)
# ---------------------------------------------------------------------------

def run_fig03() -> Dict:
    log("=== Fig 3: Performance bars (Task A & B) ===")
    log(f"Running Task A with {N_SEEDS_MAIN} seeds...")
    taskA_res = run_taskA_multi_seed(
        n_seeds=N_SEEDS_MAIN,
        N=N_NPU,
        n_trials=200,
        trial_duration=15.0,
        dt=DT,
        g_couple=0.0,
    )
    log(
        f"Task A accuracy: NPU={taskA_res['NPU_features'].mean():.3f}+-{taskA_res['NPU_features'].std():.3f}, "
        f"Raw={taskA_res['Raw'].mean():.3f}+-{taskA_res['Raw'].std():.3f}, "
        f"ESN={taskA_res['ESN'].mean():.3f}+-{taskA_res['ESN'].std():.3f}"
    )

    log(f"Running Task B with {N_SEEDS_MAIN} seeds...")
    taskB_res = run_taskB_multi_seed(
        n_seeds=N_SEEDS_MAIN,
        N=N_NPU,
        n_steps=5000,
        dt=DT,
        g_couple=0.0,
    )
    log(
        f"Task B NMSE: NPU={taskB_res['NPU_features_nmse'].mean():.3f}+-{taskB_res['NPU_features_nmse'].std():.3f}, "
        f"Raw={taskB_res['Raw_nmse'].mean():.3f}+-{taskB_res['Raw_nmse'].std():.3f}, "
        f"ESN={taskB_res['ESN_nmse'].mean():.3f}+-{taskB_res['ESN_nmse'].std():.3f}"
    )

    fig_performance_bars(taskA_res, taskB_res)
    log("Saved fig03_performance_bars.{png,pdf}")
    return {"taskA": taskA_res, "taskB": taskB_res}


# ---------------------------------------------------------------------------
# 4. Robustness analysis (noise, drift, coupling)
# ---------------------------------------------------------------------------

def run_fig04() -> None:
    log("=== Fig 4: Robustness curves ===")

    # --- Noise sweep ---
    sigma_vals = np.linspace(0.01, 0.2, 7)
    noise_taskA = {"sigma_values": sigma_vals, "NPU": [], "Raw": [], "ESN": []}
    noise_taskB = {"sigma_values": sigma_vals, "NPU": [], "Raw": [], "ESN": []}

    for sigma in sigma_vals:
        log(f"  Noise sigma={sigma:.3f}")
        # Quick runs with reduced seeds
        acc_npu, acc_raw, acc_esn = [], [], []
        nmse_npu, nmse_raw, nmse_esn = [], [], []

        for seed in range(N_SEEDS_ROBUST):
            set_seed(seed)
            # Task A mini
            stimuli, labels = generate_rhythm_classification_trials(
                n_trials=100, trial_duration=8.0, dt=DT, seed=seed + 500
            )
            n_train = 70
            stim_train, stim_test = stimuli[:n_train], stimuli[n_train:]
            lbl_train, lbl_test = labels[:n_train], labels[n_train:]

            reservoir = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir.units:
                u.sigma = sigma
            fe = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc = InputEncoder(mode="frequency", n_channels=N_NPU)
            clf = TaskAClassifier(reservoir, fe, enc)
            clf.train(stim_train, lbl_train)
            acc_npu.append(clf.evaluate(stim_test, lbl_test)["accuracy"])

            reservoir_raw = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_raw.units:
                u.sigma = sigma
            enc_raw = InputEncoder(mode="frequency", n_channels=N_NPU)
            raw_bl = TaskARawBaseline(reservoir_raw, enc_raw)
            raw_bl.train(stim_train, lbl_train)
            acc_raw.append(raw_bl.evaluate(stim_test, lbl_test)["accuracy"])

            esn_bl = TaskAESNBaseline(n_reservoir=N_NPU, dt=DT, seed=seed + 600)
            esn_bl.train(stim_train, lbl_train)
            acc_esn.append(esn_bl.evaluate(stim_test, lbl_test)["accuracy"])

            # Task B mini
            u_narma, y_narma = generate_narma10(n_steps=3000, seed=seed + 700)
            n_tr = 1800
            u_tr, u_te = u_narma[:n_tr], u_narma[n_tr:]
            y_tr, y_te = y_narma[:n_tr], y_narma[n_tr:]

            reservoir_b = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_b.units:
                u.sigma = sigma
            fe_b = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc_b = InputEncoder(mode="frequency", n_channels=N_NPU)
            model_b = TaskBNARMA(reservoir_b, fe_b, enc_b, ridge_alpha=1e-4)
            model_b.train(u_tr, y_tr)
            nmse_npu.append(model_b.evaluate(u_te, y_te)["nmse"])

            reservoir_b_raw = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_b_raw.units:
                u.sigma = sigma
            enc_b_raw = InputEncoder(mode="frequency", n_channels=N_NPU)
            raw_b = TaskBRawBaseline(reservoir_b_raw, enc_b_raw, ridge_alpha=1e-4)
            raw_b.train(u_tr, y_tr)
            nmse_raw.append(raw_b.evaluate(u_te, y_te)["nmse"])

            esn_b = TaskBESNBaseline(n_reservoir=N_NPU, dt=DT, ridge_alpha=1e-4, seed=seed + 800)
            esn_b.train(u_tr, y_tr)
            nmse_esn.append(esn_b.evaluate(u_te, y_te)["nmse"])

        noise_taskA["NPU"].append(acc_npu)
        noise_taskA["Raw"].append(acc_raw)
        noise_taskA["ESN"].append(acc_esn)
        noise_taskB["NPU"].append(nmse_npu)
        noise_taskB["Raw"].append(nmse_raw)
        noise_taskB["ESN"].append(nmse_esn)

    for key in ["NPU", "Raw", "ESN"]:
        noise_taskA[key] = np.array(noise_taskA[key])
        noise_taskB[key] = np.array(noise_taskB[key])

    # --- Parameter drift sweep ---
    drift_fracs = np.linspace(-0.15, 0.15, 7)
    drift_taskA = {"drift_values": drift_fracs, "NPU": [], "Raw": [], "ESN": []}
    drift_taskB = {"drift_values": drift_fracs, "NPU": [], "Raw": [], "ESN": []}

    for dfrac in drift_fracs:
        log(f"  Drift fraction={dfrac:.2f}")
        acc_npu, acc_raw, acc_esn = [], [], []
        nmse_npu, nmse_raw, nmse_esn = [], [], []

        for seed in range(N_SEEDS_ROBUST):
            set_seed(seed)
            # Train on nominal
            stimuli, labels = generate_rhythm_classification_trials(
                n_trials=100, trial_duration=8.0, dt=DT, seed=seed + 900
            )
            n_train = 70
            stim_train, stim_test = stimuli[:n_train], stimuli[n_train:]
            lbl_train, lbl_test = labels[:n_train], labels[n_train:]

            reservoir = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            fe = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc = InputEncoder(mode="frequency", n_channels=N_NPU)
            clf = TaskAClassifier(reservoir, fe, enc)
            clf.train(stim_train, lbl_train)

            # Test on drifted
            reservoir_test = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_test.units:
                u.f0 *= (1.0 + dfrac)
                u.tau_w = max(0.1, 2.5 / (2.0 * np.pi * u.f0))
                u.tau_ca *= (1.0 + dfrac)
            fe_test = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            clf_test = TaskAClassifier(reservoir_test, fe_test, enc)
            clf_test.clf = clf.clf  # reuse trained read-out
            clf_test.is_trained = True
            acc_npu.append(clf_test.evaluate(stim_test, lbl_test)["accuracy"])

            reservoir_raw_test = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_raw_test.units:
                u.f0 *= (1.0 + dfrac)
                u.tau_w = max(0.1, 2.5 / (2.0 * np.pi * u.f0))
                u.tau_ca *= (1.0 + dfrac)
            enc_raw = InputEncoder(mode="frequency", n_channels=N_NPU)
            raw_bl = TaskARawBaseline(reservoir_raw_test, enc_raw)
            raw_bl.train(stim_train, lbl_train)
            acc_raw.append(raw_bl.evaluate(stim_test, lbl_test)["accuracy"])

            esn_bl = TaskAESNBaseline(n_reservoir=N_NPU, dt=DT, seed=seed + 1000)
            esn_bl.train(stim_train, lbl_train)
            acc_esn.append(esn_bl.evaluate(stim_test, lbl_test)["accuracy"])

            # Task B drift
            u_narma, y_narma = generate_narma10(n_steps=3000, seed=seed + 1100)
            n_tr = 1800
            u_tr, u_te = u_narma[:n_tr], u_narma[n_tr:]
            y_tr, y_te = y_narma[:n_tr], y_narma[n_tr:]

            reservoir_b = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            fe_b = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc_b = InputEncoder(mode="frequency", n_channels=N_NPU)
            model_b = TaskBNARMA(reservoir_b, fe_b, enc_b, ridge_alpha=1e-4)
            model_b.train(u_tr, y_tr)

            reservoir_b_test = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_b_test.units:
                u.f0 *= (1.0 + dfrac)
                u.tau_w = max(0.1, 2.5 / (2.0 * np.pi * u.f0))
                u.tau_ca *= (1.0 + dfrac)
            model_b_test = TaskBNARMA(reservoir_b_test, fe_b, enc_b, ridge_alpha=1e-4)
            model_b_test.ridge = model_b.ridge
            model_b_test.is_trained = True
            nmse_npu.append(model_b_test.evaluate(u_te, y_te)["nmse"])

            reservoir_b_raw_test = NPUArray(N=N_NPU, dt=DT, g_couple=0.0, seed=seed)
            for u in reservoir_b_raw_test.units:
                u.f0 *= (1.0 + dfrac)
                u.tau_w = max(0.1, 2.5 / (2.0 * np.pi * u.f0))
                u.tau_ca *= (1.0 + dfrac)
            raw_b = TaskBRawBaseline(reservoir_b_raw_test, enc_b, ridge_alpha=1e-4)
            raw_b.train(u_tr, y_tr)
            nmse_raw.append(raw_b.evaluate(u_te, y_te)["nmse"])

            esn_b = TaskBESNBaseline(n_reservoir=N_NPU, dt=DT, ridge_alpha=1e-4, seed=seed + 1200)
            esn_b.train(u_tr, y_tr)
            nmse_esn.append(esn_b.evaluate(u_te, y_te)["nmse"])

        drift_taskA["NPU"].append(acc_npu)
        drift_taskA["Raw"].append(acc_raw)
        drift_taskA["ESN"].append(acc_esn)
        drift_taskB["NPU"].append(nmse_npu)
        drift_taskB["Raw"].append(nmse_raw)
        drift_taskB["ESN"].append(nmse_esn)

    for key in ["NPU", "Raw", "ESN"]:
        drift_taskA[key] = np.array(drift_taskA[key])
        drift_taskB[key] = np.array(drift_taskB[key])

    # --- Coupling sweep ---
    couple_vals = np.linspace(0.0, 0.1, 6)
    couple_taskA = {"couple_values": couple_vals, "NPU": [], "Raw": [], "ESN": []}
    couple_taskB = {"couple_values": couple_vals, "NPU": [], "Raw": [], "ESN": []}

    for gc in couple_vals:
        log(f"  Coupling g={gc:.3f}")
        acc_npu, acc_raw, acc_esn = [], [], []
        nmse_npu, nmse_raw, nmse_esn = [], [], []

        for seed in range(N_SEEDS_ROBUST):
            set_seed(seed)
            stimuli, labels = generate_rhythm_classification_trials(
                n_trials=100, trial_duration=8.0, dt=DT, seed=seed + 1300
            )
            n_train = 70
            stim_train, stim_test = stimuli[:n_train], stimuli[n_train:]
            lbl_train, lbl_test = labels[:n_train], labels[n_train:]

            reservoir = NPUArray(N=N_NPU, dt=DT, g_couple=gc, seed=seed)
            fe = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc = InputEncoder(mode="frequency", n_channels=N_NPU)
            clf = TaskAClassifier(reservoir, fe, enc)
            clf.train(stim_train, lbl_train)
            acc_npu.append(clf.evaluate(stim_test, lbl_test)["accuracy"])

            reservoir_raw = NPUArray(N=N_NPU, dt=DT, g_couple=gc, seed=seed)
            enc_raw = InputEncoder(mode="frequency", n_channels=N_NPU)
            raw_bl = TaskARawBaseline(reservoir_raw, enc_raw)
            raw_bl.train(stim_train, lbl_train)
            acc_raw.append(raw_bl.evaluate(stim_test, lbl_test)["accuracy"])

            esn_bl = TaskAESNBaseline(n_reservoir=N_NPU, dt=DT, seed=seed + 1400)
            esn_bl.train(stim_train, lbl_train)
            acc_esn.append(esn_bl.evaluate(stim_test, lbl_test)["accuracy"])

            u_narma, y_narma = generate_narma10(n_steps=3000, seed=seed + 1500)
            n_tr = 1800
            u_tr, u_te = u_narma[:n_tr], u_narma[n_tr:]
            y_tr, y_te = y_narma[:n_tr], y_narma[n_tr:]

            reservoir_b = NPUArray(N=N_NPU, dt=DT, g_couple=gc, seed=seed)
            fe_b = FeatureExtractor(N=N_NPU, dt=DT, window_size=2.0, overlap=1.0)
            enc_b = InputEncoder(mode="frequency", n_channels=N_NPU)
            model_b = TaskBNARMA(reservoir_b, fe_b, enc_b, ridge_alpha=1e-4)
            model_b.train(u_tr, y_tr)
            nmse_npu.append(model_b.evaluate(u_te, y_te)["nmse"])

            reservoir_b_raw = NPUArray(N=N_NPU, dt=DT, g_couple=gc, seed=seed)
            raw_b = TaskBRawBaseline(reservoir_b_raw, enc_b, ridge_alpha=1e-4)
            raw_b.train(u_tr, y_tr)
            nmse_raw.append(raw_b.evaluate(u_te, y_te)["nmse"])

            esn_b = TaskBESNBaseline(n_reservoir=N_NPU, dt=DT, ridge_alpha=1e-4, seed=seed + 1600)
            esn_b.train(u_tr, y_tr)
            nmse_esn.append(esn_b.evaluate(u_te, y_te)["nmse"])

        couple_taskA["NPU"].append(acc_npu)
        couple_taskA["Raw"].append(acc_raw)
        couple_taskA["ESN"].append(acc_esn)
        couple_taskB["NPU"].append(nmse_npu)
        couple_taskB["Raw"].append(nmse_raw)
        couple_taskB["ESN"].append(nmse_esn)

    for key in ["NPU", "Raw", "ESN"]:
        couple_taskA[key] = np.array(couple_taskA[key])
        couple_taskB[key] = np.array(couple_taskB[key])

    fig_robustness_curves(
        noise_taskA, noise_taskB,
        drift_taskA, drift_taskB,
        couple_taskA, couple_taskB,
    )
    log("Saved fig04_robustness_curves.{png,pdf}")


# ---------------------------------------------------------------------------
# 5. Reservoir dynamics projection (PCA / t-SNE)
# ---------------------------------------------------------------------------

def run_fig05() -> None:
    log("=== Fig 5: Reservoir PCA / t-SNE projection ===")
    fig_reservoir_projection(N=N_NPU, dt=DT, duration=30.0)
    log("Saved fig05_reservoir_projection.{png,pdf}")


# ---------------------------------------------------------------------------
# S1. Arnold tongue
# ---------------------------------------------------------------------------

def run_figS1() -> None:
    log("=== Fig S1: Arnold tongue ===")
    fig_arnold_tongue(f0=1.0, dt=0.01, duration=20.0)
    log("Saved figS1_arnold_tongue.{png,pdf}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()
    # Clear old report
    if os.path.exists("results_report.txt"):
        os.remove("results_report.txt")

    log("=" * 60)
    log(f"NPU Oscillatory Reservoir Computing Benchmark")
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"N_NPU={N_NPU}, DT={DT}, N_SEEDS_MAIN={N_SEEDS_MAIN}")
    log("=" * 60)

    run_fig01()
    run_fig02()
    bench_results = run_fig03()
    run_fig04()
    run_fig05()
    run_figS1()

    elapsed = time.time() - t0
    log(f"All experiments finished in {elapsed:.1f} s")
    log("Figures saved to ./figures/")
    print("\nDone. Check ./figures/ and ./results_report.txt")

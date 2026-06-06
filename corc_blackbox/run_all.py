#!/usr/bin/env python3
"""
run_all.py
----------
One-click demonstration and benchmark runner for CORC.
Produces figures under ../figures/ and prints a summary table.

Usage:
    python -m corc_blackbox.run_all
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier

# ensure imports work when run as script or module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corc_blackbox.reservoir import CORCReservoir
from corc_blackbox.baselines import ESN, SimpleOscillatorReservoir, HarmonicReservoir, NullReservoir
from corc_blackbox.coupling import Coupling
from corc_blackbox.units import HopfSlowUnit
from corc_blackbox.tasks import (
    RhythmClassification, TemporalXOR, SwitchingRhythm, NARMA10, MemoryCapacity,
    train_readout,
)
from corc_blackbox.observables import delay_embed, kuramoto_order
from corc_blackbox.analysis import (
    avalanche_sizes_durations, powerlaw_fit, branching_ratio_estimate,
    synchrony_scan, state_richness,
)
from corc_blackbox.plots import (
    plot_dynamics_overview, plot_avalanche_distributions,
    plot_coupling_comparison, plot_criticality_analysis,
    plot_main_results, plot_ablation_matrix, plot_robustness_curves,
    plot_state_projection,
)

SEED = 42
RNG = np.random.default_rng(SEED)
DT = 0.01
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def build_corc(N=32, seed=None, **overrides):
    rng = np.random.default_rng(seed)
    units = HopfSlowUnit.heterogeneous(N, DT, rng=rng)
    coupling = Coupling(N, DT, rng=rng, **overrides)
    input_matrix = rng.normal(0, 1.0, size=(N, 1))
    input_gain = rng.uniform(0.5, 2.0, size=N)
    return CORCReservoir(units, coupling, input_matrix, input_gain)


def run_dynamics_demo():
    print("[1/7] Dynamics overview and avalanche distributions...")
    res = build_corc(N=32, seed=SEED)
    T = 3000
    u = 0.3 * np.sin(2 * np.pi * 1.0 * np.arange(T) * DT) + RNG.normal(0, 0.05, T)
    states, events = res.run(u, reset=True)
    plot_dynamics_overview(states, events, DT, path=os.path.join(FIG_DIR, "01_dynamics_overview.png"))
    plot_avalanche_distributions(events, path=os.path.join(FIG_DIR, "01_avalanche_distributions.png"))
    print("      Saved 01_*.png")


def run_coupling_comparison():
    print("[2/7] Coupling mechanism comparison...")
    T = 2000
    u = 0.3 * np.sin(2 * np.pi * 1.0 * np.arange(T) * DT) + RNG.normal(0, 0.05, T)
    results = {}
    configs = {
        "full": {},
        "no_pulse": {"disable_pulse": True},
        "no_diffusion": {"disable_diffusion": True},
        "no_meanfield": {"disable_meanfield": True},
        "no_coupling": {"disable_pulse": True, "disable_diffusion": True, "disable_meanfield": True},
    }
    for label, kw in configs.items():
        res = build_corc(N=32, seed=SEED, **kw)
        states, events = res.run(u, reset=True)
        results[label] = {"states": states, "events": events}
    plot_coupling_comparison(results, DT, path=os.path.join(FIG_DIR, "02_coupling_comparison.png"))
    print("      Saved 02_*.png")


def run_criticality():
    print("[3/7] Criticality analysis (synchrony scan)...")
    g_values = np.linspace(0.0, 0.15, 7)

    def factory(g):
        return build_corc(N=32, seed=SEED, g_d=g, g_m=g, g_p=g * 3)

    scan = synchrony_scan(factory, g_values, T=2000, dt=DT)
    event_dict = {}
    for g in g_values:
        res = factory(g)
        _, events = res.run(np.zeros((2000, 1)), reset=True)
        event_dict[g] = events
    plot_criticality_analysis(scan, event_dict, path=os.path.join(FIG_DIR, "03_criticality.png"))
    print("      Saved 03_*.png")


def extract_features(reservoir, inputs, taps=4, interval=3):
    states, events = reservoir.run(inputs, reset=True)
    X = delay_embed(states, features=["x", "y", "a", "r", "s"], taps=taps, interval_steps=interval)
    return X, events


def benchmark_classification(task, task_name, metric_key="accuracy"):
    inputs, targets = task.generate()
    n_trials = len(targets)
    n_train = int(0.7 * n_trials)

    # CORC
    corc = build_corc(N=32, seed=SEED)
    X_all = []
    for i in range(n_trials):
        X, _ = extract_features(corc, inputs[i])
        X_all.append(X[-1])  # take final state vector as trial summary
    X_all = np.stack(X_all, axis=0)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = targets[:n_train], targets[n_train:]
    _, met_corc = train_readout(X_train, y_train, X_test, y_test, task_type="classification")

    # ESN
    esn = ESN(N=32, input_dim=1, seed=SEED)
    X_esn = []
    for i in range(n_trials):
        S = esn.run(inputs[i])
        X_esn.append(S[-1])
    X_esn = np.stack(X_esn, axis=0)
    _, met_esn = train_readout(X_esn[:n_train], y_train, X_esn[n_train:], y_test, task_type="classification")

    # Simple oscillator (no coupling)
    simple = SimpleOscillatorReservoir(N=32, seed=SEED)
    X_sim = []
    for i in range(n_trials):
        X, _ = extract_features(simple, inputs[i])
        X_sim.append(X[-1])
    X_sim = np.stack(X_sim, axis=0)
    _, met_sim = train_readout(X_sim[:n_train], y_train, X_sim[n_train:], y_test, task_type="classification")

    # Harmonic
    harm = HarmonicReservoir(N=32, seed=SEED)
    X_har = []
    for i in range(n_trials):
        X, _ = extract_features(harm, inputs[i])
        X_har.append(X[-1])
    X_har = np.stack(X_har, axis=0)
    _, met_har = train_readout(X_har[:n_train], y_train, X_har[n_train:], y_test, task_type="classification")

    return {
        "CORC": met_corc[metric_key],
        "ESN": met_esn[metric_key],
        "SimpleOsc": met_sim[metric_key],
        "Harmonic": met_har[metric_key],
    }


def run_main_benchmarks():
    print("[4/7] Main benchmark bar chart...")
    results = {}
    # Rhythm classification
    rc = RhythmClassification(dt=DT, rng=RNG)
    results["Rhythm"] = benchmark_classification(rc, "Rhythm")
    # Temporal XOR
    tx = TemporalXOR(dt=DT, rng=RNG)
    results["TempXOR"] = benchmark_classification(tx, "TempXOR")
    # Switching rhythm
    sr = SwitchingRhythm(dt=DT, rng=RNG)
    results["Switch"] = benchmark_classification(sr, "Switch")

    # Prepare for plotting
    methods = ["CORC", "ESN", "SimpleOsc", "Harmonic"]
    tasks_names = list(results.keys())
    metrics = {m: {t: results[t][m] for t in tasks_names} for m in methods}
    plot_main_results(metrics, tasks_names, metric_key="accuracy", path=os.path.join(FIG_DIR, "04_main_results.png"))
    print("      Saved 04_*.png")
    return results


def run_ablations():
    print("[5/7] Ablation matrix...")
    task = TemporalXOR(dt=DT, rng=RNG)
    inputs, targets = task.generate(n_trials=200)
    n_train = 140

    ablation_configs = {
        "full": lambda: build_corc(N=32, seed=SEED),
        "beta=0": lambda: _build_ablated_unit(beta_range=(0.0, 0.0), alpha_range=(0.0, 0.0)),
        "no_pulse": lambda: build_corc(N=32, seed=SEED, disable_pulse=True),
        "eta=0": lambda: build_corc(N=32, seed=SEED, eta=0.0),
        "homogeneous": lambda: _build_ablated_unit(homogeneous=True),
        "no_events": lambda: build_corc(N=32, seed=SEED),
    }
    ablation_scores = {}
    for cond_name, builder in ablation_configs.items():
        res = builder()
        feats = ["x", "y", "a", "r"] if cond_name == "no_events" else ["x", "y", "a", "r", "s"]
        X_all = []
        for i in range(len(targets)):
            states, _ = res.run(inputs[i], reset=True)
            from corc_blackbox.observables import delay_embed
            X = delay_embed(states, features=feats, taps=4, interval_steps=3)
            X_all.append(X[-1])
        X_all = np.stack(X_all, axis=0)
        _, met = train_readout(X_all[:n_train], targets[:n_train], X_all[n_train:], targets[n_train:], task_type="classification")
        ablation_scores[cond_name] = {"TempXOR": met["accuracy"]}

    plot_ablation_matrix(ablation_scores, path=os.path.join(FIG_DIR, "05_ablation_matrix.png"))
    print("      Saved 05_*.png")
    return ablation_scores


def _build_ablated_unit(beta_range=(0.8, 1.5), alpha_range=(0.8, 1.5), homogeneous=False):
    rng = np.random.default_rng(SEED)
    if homogeneous:
        units = HopfSlowUnit.homogeneous(32, DT, rng=rng)
    else:
        units = HopfSlowUnit.heterogeneous(32, DT, rng=rng, beta_range=beta_range, alpha_range=alpha_range)
    coupling = Coupling(32, DT, rng=rng)
    input_matrix = rng.normal(0, 1.0, size=(32, 1))
    input_gain = rng.uniform(0.5, 2.0, size=32)
    return CORCReservoir(units, coupling, input_matrix, input_gain)


def run_robustness():
    print("[6/7] Robustness curves...")
    noise_levels = np.linspace(0.0, 0.3, 6)
    task = RhythmClassification(dt=DT, rng=RNG)
    inputs, targets = task.generate(n_trials=200)
    n_train = 140
    methods = {}
    for name, builder in [("CORC", lambda: build_corc(N=32, seed=SEED)),
                          ("ESN", lambda: ESN(N=32, input_dim=1, seed=SEED)),
                          ("SimpleOsc", lambda: SimpleOscillatorReservoir(N=32, seed=SEED))]:
        scores = []
        for nl in noise_levels:
            res = builder()
            X_all = []
            for i in range(len(targets)):
                noisy = inputs[i] + RNG.normal(0, nl, size=inputs[i].shape)
                if name == "ESN":
                    S = res.run(noisy)
                    X_all.append(S[-1])
                else:
                    states, _ = res.run(noisy, reset=True)
                    from corc_blackbox.observables import delay_embed
                    X = delay_embed(states, taps=4, interval_steps=3)
                    X_all.append(X[-1])
            X_all = np.stack(X_all, axis=0)
            _, met = train_readout(X_all[:n_train], targets[:n_train], X_all[n_train:], targets[n_train:], task_type="classification")
            scores.append(met["accuracy"])
        methods[name] = np.array(scores)
    plot_robustness_curves(noise_levels, methods, xlabel="Input noise std", path=os.path.join(FIG_DIR, "06_robustness.png"))
    print("      Saved 06_*.png")


def run_projections():
    print("[7/7] State projections...")
    res = build_corc(N=32, seed=SEED)
    T = 2000
    t = np.arange(T)
    u = 0.3 * np.sin(2 * np.pi * 0.5 * t * DT)
    states, _ = res.run(u, reset=True)
    from corc_blackbox.observables import delay_embed
    X = delay_embed(states, features=["x", "y", "a", "r", "s"], taps=4, interval_steps=3)
    plot_state_projection(X, colors=u, method="pca", path=os.path.join(FIG_DIR, "07_projection_pca.png"))
    plot_state_projection(X, colors=u, method="tsne", path=os.path.join(FIG_DIR, "07_projection_tsne.png"))
    print("      Saved 07_*.png")


def run_narma():
    print("[Extra] NARMA-10 regression benchmark...")
    task = NARMA10(dt=DT, rng=RNG)
    u, y = task.generate(T=4000)
    split = 3000

    corc = build_corc(N=32, seed=SEED)
    Xc, _ = extract_features(corc, u)
    _, met_c = train_readout(Xc[:split], y[:split], Xc[split:], y[split:], task_type="regression")

    esn = ESN(N=32, input_dim=1, seed=SEED)
    Se = esn.run(u)
    _, met_e = train_readout(Se[:split], y[:split], Se[split:], y[split:], task_type="regression")

    simple = SimpleOscillatorReservoir(N=32, seed=SEED)
    Xs, _ = extract_features(simple, u)
    _, met_s = train_readout(Xs[:split], y[:split], Xs[split:], y[split:], task_type="regression")

    print(f"      CORC  NMSE: {met_c['nmse']:.4f}")
    print(f"      ESN   NMSE: {met_e['nmse']:.4f}")
    print(f"      Simp  NMSE: {met_s['nmse']:.4f}")
    return {"CORC": met_c, "ESN": met_e, "SimpleOsc": met_s}


def run_memory_capacity():
    print("[Extra] Memory capacity...")
    mc = MemoryCapacity(dt=DT, max_lag=20, rng=RNG)
    u, _ = mc.generate(n_samples=5000)
    corc = build_corc(N=32, seed=SEED)
    Xc, _ = extract_features(corc, u, taps=4, interval=3)
    mc_vals = MemoryCapacity.compute_mc(Xc, u[:, 0], max_lag=20)
    print(f"      Total MC: {mc_vals['total']:.2f}")
    return mc_vals


def main():
    print("=" * 60)
    print("CORC / AHER Benchmark Suite")
    print("=" * 60)
    run_dynamics_demo()
    run_coupling_comparison()
    run_criticality()
    bench = run_main_benchmarks()
    abl = run_ablations()
    run_robustness()
    run_projections()
    narma = run_narma()
    mc = run_memory_capacity()

    # Save JSON summary
    summary = {
        "classification": bench,
        "ablation": abl,
        "narma": {k: {m: float(v) for m, v in narma[k].items()} for k in narma},
        "memory_capacity": {k: float(v) for k, v in mc.items()},
    }
    with open(os.path.join(FIG_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nAll figures saved to:", FIG_DIR)
    print("Summary JSON saved to:", os.path.join(FIG_DIR, "summary.json"))
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()

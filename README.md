# CORC: Calcium-inspired Oscillatory Reservoir Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**From biological calcium oscillations to lightweight attention — a physically-inspired temporal computing framework.**

---

## Overview

CORC (Calcium-inspired Oscillatory Reservoir Computing) is a reservoir computing framework that maps biological calcium dynamics onto computational principles. The framework demonstrates that systematically incorporating features of neuronal calcium oscillations — excitability, two-pool dynamics, slow inactivation, and critical avalanches — enables oscillatory reservoirs to serve both rhythmic/event-based tasks *and* numerical regression tasks in a unified architecture.

### Key Results

| Task | CORC v2 | ESN (128) | Improvement |
|------|:-------:|:---------:|:-----------:|
| NARMA-10 NMSE | **0.71** | 0.31 | 3845× vs v1 |
| HardRhythm (2.0 vs 2.3Hz) | **98.3%** | 43.3% | +55 pp |
| Temporal XOR | **100%** | 50% | Perfect |
| Phase Noise (3-class) | **68.9%** | 65.6% | +3.3 pp |
| Rhythm Classification | **100%** | 100% | Ceiling |

### AGSC: A Companion Discovery

The amplitude gating factor `(1 + η·c_j)` in the pulse coupling mechanism was found to be critical — removing it (η=0) causes NARMA NMSE to explode from 1147 to 6774. This discovery generalizes to **AGSC (Amplitude-Gated Sparse Coupling)**, a lightweight attention mechanism of O(N·d) complexity that is a degenerate case of linear attention (Q=1).

---

## Repository Structure

```
corc/
├── corc_v2/                    # Main CORC v2 framework
│   ├── units.py                # CEU (Calcium Excitable Unit) dynamics
│   ├── coupling.py             # Pulse coupling + weak diffusion
│   ├── reservoir.py            # Network integration + time-scale elasticity
│   ├── observables.py          # Feature extraction + delay embedding
│   ├── tasks.py                # 7 task families + readout layer
│   ├── baselines.py            # ESN, Hopf v1, CEU no-pulse, Linear
│   ├── analysis.py             # Criticality + avalanche analysis
│   ├── plots.py                # 16 publication-quality figures
│   ├── plots_schematic.py      # 7 schematic/architecture diagrams
│   └── run_all.py              # One-click experiment runner
├── agsc_proof/                 # AGSC proof-of-concept
│   ├── agsc_proof.py           # AGSC models, training, evaluation
│   └── plots.py                # 9 AGSC figures
├── papers/                     # Preprint papers
│   ├── paper1_corc_en.md       # CORC paper (English)
│   ├── paper1_corc_cn.md       # CORC paper (Chinese)
│   ├── paper2_agsc_en.md       # AGSC + Bio-Distillation paper (English)
│   └── paper2_agsc_cn.md       # AGSC + Bio-Distillation paper (Chinese)
├── figures_v2/                 # Generated figures (CORC v2)
├── agsc_proof/figures/         # Generated figures (AGSC)
├── setup.py                    # Package setup
├── pyproject.toml              # Build configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/corc.git
cd corc
pip install -e .
```

### Run CORC v2 Experiments

```bash
python -m corc_v2.run_all
```

This executes all 10 experiment steps:
1. Single-node CEU dynamics
2. Avalanche statistics
3. Critical coupling scan
4. Main benchmark (all tasks)
5. NARMA-10 evaluation
6. Memory capacity
7. Ablation studies
8. State projection (PCA/t-SNE)
9. Hard tasks (close-frequency rhythm, phase noise)
10. Robustness experiments (noise, dropout, drift)

All figures are saved to `figures_v2/` and results to `figures_v2/summary.json`.

### Run AGSC Proof-of-Concept

```bash
python agsc_proof/agsc_proof.py
python agsc_proof/plots.py
```

### Generate Schematic Diagrams

```bash
python corc_v2/plots_schematic.py
```

---

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- PyTorch (for AGSC module only)

---

## Citation

If you use CORC in your research, please cite:

```bibtex
@article{corc2024,
  title={From Biological Calcium Oscillations to Lightweight Attention: The CORC Framework for Physically-Inspired Temporal Computing},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}

@article{agsc2024,
  title={Signal Strength Modulates Influence: Amplitude-Gated Sparse Coupling as a Biological Distillation Mechanism},
  author={[Authors]},
  journal={arXiv preprint},
  year={2024}
}
```

---

## License

MIT License. See `LICENSE` file for details.

---

## Acknowledgments

This work is inspired by decades of research on neuronal calcium dynamics (Berridge, Goldbeter, et al.), echo state networks (Jaeger), and linear attention mechanisms (Katharopoulos et al.). The framework aims to bridge the gap between biological computation and artificial intelligence through systematic abstraction of computational principles.
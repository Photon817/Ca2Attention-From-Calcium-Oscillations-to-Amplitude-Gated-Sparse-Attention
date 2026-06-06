# CORC：钙振荡启发储备池计算

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**从生物钙振荡到轻量注意力——面向物理启发时序计算的计算框架。**

---

## 概述

CORC（Calcium-inspired Oscillatory Reservoir Computing，钙振荡启发储备池计算）是一个将生物钙动力学映射到计算原理的储备池计算框架。该框架证明，系统性地引入神经元钙振荡的关键特征——可兴奋性、两池动力学、慢失活和临界雪崩——使振荡储备池能够在统一架构中同时服务节律/事件任务和数值回归任务。

### 核心结果

| 任务 | CORC v2 | ESN (128) | 改善 |
|------|:-------:|:---------:|:----:|
| NARMA-10 NMSE | **0.71** | 0.31 | 3845× vs v1 |
| HardRhythm (2.0 vs 2.3Hz) | **98.3%** | 43.3% | +55pp |
| Temporal XOR | **100%** | 50% | 完美 |
| Phase Noise (3-class) | **68.9%** | 65.6% | +3.3pp |
| Rhythm Classification | **100%** | 100% | 天花板 |

### AGSC：伴随发现

脉冲耦合机制中的振幅门控因子 `(1 + η·c_j)` 被发现至关重要——去除它（η=0）导致 NARMA NMSE 从 1147 爆炸至 6774。此发现推广为 **AGSC（振幅门控稀疏耦合）**，一种 O(N·d) 复杂度的轻量注意力机制，是线性注意力（Q=1）的退化特例。

---

## 仓库结构

```
corc/
├── corc_v2/                    # CORC v2 主框架
│   ├── units.py                # CEU（钙可兴奋单元）动力学
│   ├── coupling.py             # 脉冲耦合 + 弱扩散
│   ├── reservoir.py            # 网络集成 + 时间尺度弹性
│   ├── observables.py          # 特征提取 + 延迟嵌入
│   ├── tasks.py                # 7 类任务 + 读出层
│   ├── baselines.py            # ESN, Hopf v1, 无脉冲CEU, 线性
│   ├── analysis.py             # 临界态 + 雪崩分析
│   ├── plots.py                # 16 张论文级图表
│   ├── plots_schematic.py      # 7 张原理/架构示意图
│   └── run_all.py              # 一键实验运行器
├── agsc_proof/                 # AGSC 概念验证
│   ├── agsc_proof.py           # AGSC 模型、训练、评估
│   └── plots.py                # 9 张 AGSC 图表
├── papers/                     # 预印本论文
│   ├── paper1_corc_en.md       # CORC 论文（英文）
│   ├── paper1_corc_cn.md       # CORC 论文（中文）
│   ├── paper2_agsc_en.md       # AGSC + 生物蒸馏论文（英文）
│   └── paper2_agsc_cn.md       # AGSC + 生物蒸馏论文（中文）
├── figures_v2/                 # 生成的图表（CORC v2）
├── agsc_proof/figures/         # 生成的图表（AGSC）
├── setup.py                    # 包安装配置
├── pyproject.toml              # 构建配置
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/yourusername/corc.git
cd corc
pip install -e .
```

### 运行 CORC v2 实验

```bash
python -m corc_v2.run_all
```

执行全部 10 个实验步骤：
1. 单节点 CEU 动力学
2. 雪崩统计
3. 临界耦合扫描
4. 主 Benchmark（全部任务）
5. NARMA-10 评估
6. 记忆容量
7. 消融实验
8. 状态投影（PCA/t-SNE）
9. 硬任务（相近频率节律、相位噪声）
10. 鲁棒性实验（噪声、失活、漂移）

所有图表保存至 `figures_v2/`，结果保存至 `figures_v2/summary.json`。

### 运行 AGSC 概念验证

```bash
python agsc_proof/agsc_proof.py
python agsc_proof/plots.py
```

### 生成原理示意图

```bash
python corc_v2/plots_schematic.py
```

---

## 依赖

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- PyTorch（仅 AGSC 模块需要）

---

## 引用

如果您在研究中使用了 CORC，请引用：

```bibtex
@article{corc2024,
  title={从生物钙振荡到轻量注意力：面向物理启发时序计算的 CORC 框架},
  author={[作者]},
  journal={arXiv preprint},
  year={2024}
}

@article{agsc2024,
  title={信号强度调节影响力：振幅门控稀疏耦合作为一种生物蒸馏机制},
  author={[作者]},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 许可证

MIT License。详见 `LICENSE` 文件。

---

## 致谢

本工作受神经元钙动力学（Berridge、Goldbeter 等）、回声状态网络（Jaeger）和线性注意力机制（Katharopoulos 等）数十年的研究启发。该框架旨在通过计算原则的系统抽象，连接生物计算与人工智能。
# Summary — 图表与论文假设对照

本文件说明每张生成图表在论文中对应的假设、结论及如何引用。

---

## 图1 — `fig01_single_npu_response`  
**对应假设：H1** — 单个NPU对外部光热刺激表现出可控的非线性振荡响应。  
**内容：**
- (a) 单NPU对阶梯刺激（低→高→低）的钙信号时程，展示振幅饱和与频率牵引现象。
- (b) $(v, w)$ 相图，验证极限环行为。
- (c) 三维轨迹 $(v, w, c)$，展示钙慢变量如何调制快动力学。

**论文位置：** Results → "Single-NPU dynamics" 或 Methods → "Model validation"。

---

## 图2 — `fig02_multi_npu_sync`  
**对应假设：H2** — NPU阵列在不同频率驱动下可形成同步/去同步簇，体现储备池的高维分离能力。  
**内容：**
- (a) 8个NPU在双频（0.8 Hz / 2.2 Hz）驱动下的钙振荡轨迹。
- (b) 配对相位差分布 histogram。
- (c) 组内 vs 组间 PLV 对比，显示频率特异性聚类。

**论文位置：** Results → "Reservoir clustering and synchronization"。

---

## 图3 — `fig03_performance_bars`  
**对应假设：H3** — 基于钙振荡特征提取的读出层在时序任务上优于原始信号基线，并与传统ESN可比。  
**内容：**
- (左) Task A（节律分类）准确率对比：NPU(features) vs Raw vs ESN。
- (右) Task B（NARMA-10）NMSE对比。
- 误差条来自10次独立随机初始条件。

**论文位置：** Results → "Benchmark performance"。

---

## 图4 — `fig04_robustness_curves`  
**对应假设：H4** — 钙振荡储备池对生物不稳定性（噪声、参数漂移、耦合串扰）具有鲁棒性。  
**内容：** 6个子图：
- (a)(b) 噪声水平 $\sigma \in [0.01, 0.2]$ 对 Task A/B 的影响。
- (c)(d) 训练-测试间参数漂移（$\pm 15\%$ $f_0, \tau_{ca}$）的影响。
- (e)(f) 弱耦合强度 $g_{couple} \in [0, 0.1]$ 的影响。

**论文位置：** Results → "Robustness analysis"。

---

## 图5 — `fig05_reservoir_projection`  
**对应假设：H5** — 储备池在复合刺激下遍历高维特征空间，形成丰富的动态轨迹，支持线性可分读出。  
**内容：**
- (a) 复合 chirp 刺激（频率扫描）。
- (b) 储备池特征的前两个主成分投影（PCA），颜色映射时间。
- (c) t-SNE 非线性投影，验证动态轨迹的拓扑结构。

**论文位置：** Results → "Reservoir state space dynamics"。

---

## 图S1 — `figS1_arnold_tongue`  
**对应假设：H1 补充** — NPU的 entrained 区域符合 Arnold tongue 结构，证明其可作为频率调谐元件。  
**内容：** 驱动频率 vs 驱动振幅的热图，颜色为归一化频率偏差。

**论文位置：** Supplementary Information → "Arnold tongue characterization"。

---

## 数值结果报告

运行 `run_experiments.py` 后，`results_report.txt` 将包含：
- Task A 各方法平均准确率 $\pm$ 标准差
- Task B 各方法平均 NMSE $\pm$ 标准差
- 各鲁棒性扫描的详细数值

可直接复制到论文 Results 段落。

---

## Methods 文本块（可直接复制到论文）

```latex
\subsection*{NPU Model}
Each neuro-photonic unit (NPU) is described by a modified FitzHugh--Nagumo
system with an additional slow calcium variable $c$:
\begin{align}
\dot v &= v - \frac{v^3}{3} - w + c + g I(t) + \xi_v(t), \\
\dot w &= \frac{v + a - b w}{\tau_w}, \\
\dot c &= \frac{-c + \kappa \max(0,v)}{\tau_{ca}} + \xi_c(t),
\end{align}
where $v$ is a fast voltage-like variable, $w$ a slow recovery variable,
and $c$ tracks cytosolic Ca$^{2+}$.  $I(t)$ denotes photothermal stimulus
intensity, $g$ the input gain, and $\xi$ independent Gaussian noise
($\sigma_v=0.02$, $\sigma_c=0.006$).  The intrinsic frequency $f_0$ is set
by $\tau_w = 2.5 / (2\pi f_0)$, yielding limit-cycle oscillations in the
range $0.5$--$2.5$~Hz.  Calcium recovery time $\tau_{ca}\in[0.5,2]$~s
introduces short-term memory.

\subsection*{Reservoir Architecture}
An array of $N=8$ NPUs with heterogeneous parameters
($f_0$, $\tau_{ca}$, gain, noise) receives encoded optical inputs.
Weak diffusive coupling $g_{\text{couple}}( \bar v - v_i )$ may be added
to mimic microfluidic crosstalk.  Read-out features are extracted in
$2$~s sliding windows (50\% overlap): instantaneous amplitude envelope
and phase (Hilbert transform), dominant frequency (FFT peak), band-pass
energies (0.5--1, 1--2, 2--4~Hz), and pairwise phase-locking values.

\subsection*{Training}
Only the linear read-out layer is trained.
Task~A (rhythm classification): logistic regression ($C=1.0$) on
mean-pooled feature vectors.
Task~B (NARMA-10): ridge regression ($\alpha=10^{-4}$) predicting
$y(t+1)$ from reservoir features.
All random seeds are fixed; reported metrics are mean$\pm$s.d. over
$\ge 10$ independent trials.
```

# CORC / AHER 框架概述

## 1. 定位与动机

**CORC (Calcium-inspired Oscillatory Reservoir Computing)** 是一个由钙振荡群体动力学启发的黑箱储层计算框架。它**不精确建模离子通道、受体或分子机制**，而是抽象出以下核心特征：

- 慢–快变量耦合
- 非线性振荡与极限环
- 频率牵引 / 锁相
- 短时记忆
- 同步 / 去同步转变
- 事件爆发 / 雪崩活动
- 噪声鲁棒性

本工作的**重点是算法与计算原理**，而非生物物理细节。

## 2. 黑箱视角

从物理储层计算（Physical Reservoir Computing）的角度，复杂动力学被视为一个**可驱动、可观测、只训练读出层的黑箱动力系统**。我们关心的是：

1. 给定输入 $u(t)$，系统的瞬态响应是否具备高分离度（high separability）？
2. 系统内部状态是否保留足够的历史信息以支持记忆任务？
3. 在事件驱动或节律任务中，复杂耦合与临界性是否带来可量化的性能提升？

## 3. NPU 的角色

**NPU（神经元处理单元）仅作为硬件实现示例之一**，不是模型本体。CORC 的核心是一个通用的复杂动力学黑箱储层计算框架；它可以映射到模拟电路、光子系统、或任何具有慢-快振荡和脉冲耦合特性的物理介质上。

## 4. 单节点：Hopf + 慢适应

节点 $i$ 的状态为 $(x_i, y_i, a_i)$，其中 $a_i$ 是慢变量：

$$
\begin{aligned}
\dot{x}_i &= (\mu_i - r_i^2 - \beta_i a_i)x_i - \omega_i y_i + G_i u_i(t) + C_{i,x}(t) + \sigma_i \xi_{i,x} \\
\dot{y}_i &= (\mu_i - r_i^2 - \beta_i a_i)y_i + \omega_i x_i + C_{i,y}(t) + \sigma_i \xi_{i,y} \\
\tau_{a,i} \dot{a}_i &= -a_i + \alpha_i r_i^2
\end{aligned}
$$

- $\mu_i$：振荡增益；$\omega_i$：本征频率
- $\beta_i, \tau_{a,i}, \alpha_i$：慢适应的抑制强度、时间常数、振幅驱动强度
- $C_i(t)$：来自耦合模块的总耦合
- $\sigma_i \xi_i$：独立高斯白噪声

## 5. 耦合：三层叠加

总耦合 $C_i = C_{i,\text{diff}} + C_{i,\text{mf}} + C_{i,\text{pulse}}$：

- **弱扩散耦合**：$C_{i,\text{diff}} = g_d \sum_j W_{ij}(z_j - z_i)$，$W$ 为稀疏随机连接
- **平均场耦合**：$C_{i,\text{mf}} = g_m (\bar{z} - z_i)$
- **脉冲耦合**：当振幅 $r_i$ 上穿阈值 $\theta$ 时触发事件，生成短时脉冲痕迹 $s_j(t)$，对其他节点施加脉冲驱动
- **振幅依赖**：耦合强度可乘以 $(1 + \eta r_j)$，使高振幅节点影响力更大

## 6. 状态观测与读出

**绝不依赖纯手工压缩特征**。采用**混合高维状态表示**：

- 基础连续状态：$x_i, y_i, a_i$
- 振幅与相位：$r_i, \phi_i$
- 事件状态：脉冲痕迹 $s_i(t)$

对上述变量施加**延迟嵌入**（delay taps）：

$$
z(t) = [h(t), h(t-\Delta), h(t-2\Delta), \dots, h(t-(K-1)\Delta)]
$$

其中 $h(t)$ 可取所有节点的 $[x_i, y_i, a_i, r_i, s_i]$ 拼接。推荐 $K=4$，间隔 $2$–$5$ 个仿真步长。

**最终读出层输入**为 $[x, y, a, r, s]$ 的延迟嵌入版本，保留原始高维信息，不依赖 Hilbert 提取单一振幅/频率。

## 7. 关键叙事原则

CORC 是复杂动力学黑箱储层计算框架，NPU 仅是可能的物理实现之一。若 NARMA 性能不及 ESN，必须正面叙述并转向其在节律/事件/模式分类任务的优势。

## 8. 仓库结构

```
corc_blackbox/
├── __init__.py
├── units.py          # Hopf + 慢适应单元
├── coupling.py       # 扩散 + 平均场 + 脉冲耦合
├── reservoir.py      # 网络集成与运行
├── observables.py    # 延迟嵌入、事件检测、同步指标
├── tasks.py          # 4 类 Benchmark 任务
├── baselines.py      # ESN、简单振荡器、谐波、线性基线
├── analysis.py       # 临界性、雪崩统计、分支比
├── plots.py          # 7 张必要图表
├── run_all.py        # 一键运行与演示
├── summary.md        # 本文件
├── methods_for_paper.md
└── results_interpretation.md
```

# Methods for Paper — CORC v2

## 1. CEU 节点动力学

### 1.1 简化钙两池模型

每个节点 $i$ 的状态为 $(c_i, s_i, a_i)$：$c_i$ 为胞质钙（快变量），$s_i$ 为内质网储存钙（慢恢复），$a_i$ 为超慢适应（IP₃R 样慢失活）。无量纲动力学：

$$
\begin{aligned}
\tau_c \dot{c}_i &= -c_i + J_{\text{rel}}(c_i, s_i) + I_i^{\text{ext}} + C_{i,c} + \sigma_{c,i} \xi_{i,c} \\
\tau_s \dot{s}_i &= \frac{1 - s_i}{\tau_{\text{pump},i}} - \gamma J_{\text{rel}}(c_i, s_i) \\
\tau_a \dot{a}_i &= -a_i + \alpha_{a,i} \cdot H(c_i - c_{\text{th},i})
\end{aligned}
$$

钙诱导钙释放项（CICR）：

$$J_{\text{rel}} = g_{\text{rel},i} \cdot s_i \cdot \text{sig}(c_i - c_{\text{th},i}, k_i) \cdot (1 - a_i)$$

其中 $\text{sig}(x, k) = 1/(1+e^{-kx})$，$H$ 为单位阶跃函数。$J_{\text{rel}}$ 受三个因素调制：(1) ER 储存 $s_i$，(2) 钙浓度对阈值 $c_{\text{th}}$ 的 sigmoid 穿越，(3) 慢失活 $a_i$。

### 1.2 参数异质性采样

| 参数 | 分布 | 区间/参数 | 物理含义 |
|------|------|----------|----------|
| $c_{\text{th},i}$ | Uniform | [0.20, 0.50] | 钙释放阈值 |
| $g_{\text{rel},i}$ | Uniform | [1.0, 2.0] | 释放电导 |
| $\tau_{\text{pump},i}$ | Uniform | [3.0, 10.0] | 泵回时间常数 |
| $k_i$ | Uniform | [10, 30] | 释放陡度 |
| $\tau_{a,i}$  | LogNormal | μ=1.5, σ=0.5 | 适应时间常数 |
| $\alpha_{a,i}$ | Uniform | [0.3, 0.8] | 适应增益 |
| $\sigma_{c,i}$ | Uniform | [0.05, 0.15] | 噪声强度 |

固定参数：$\tau_c = 0.1$, $\tau_s = 1.0$, $\gamma = 0.3$。积分：Euler-Maruyama，$dt = 0.01$。状态数据类型：float32（内存优化）。

初始条件：$c_i \sim U(0.05, 0.25)$（接近阈值），$s_i = 1.0$（ER 满载），$a_i = 0$（无适应）。

## 2. 耦合机制

### 2.1 事件检测

当 $c_i$ 向上穿越阈值 $\theta_{\text{event}} = 0.6$ 时触发事件。

### 2.2 脉冲耦合

脉冲痕迹 $s_i^{\text{pulse}}$ 在事件触发时加一，按指数 $\tau_p = 0.08$ 衰减：

$$\dot{s}_i^{\text{pulse}} = -s_i^{\text{pulse}}/\tau_p + \sum\nolimits_k \delta(t - t_{i,k})$$

耦合驱动（带振幅调制）：

$$C_{i,c}^{\text{pulse}} = g_p \sum_j W_{ij} \cdot s_j^{\text{pulse}} \cdot (1 + \eta \cdot c_j)$$

默认 $g_p = 0.25$, $\eta = 0.4$。

### 2.3 弱扩散

$$C_{i,c}^{\text{diff}} = g_d \sum_j W_{ij} (c_j - c_i), \quad g_d = 0.015$$

总耦合：$C_{i,c} = C_{i,c}^{\text{pulse}} + C_{i,c}^{\text{diff}}$。

连接矩阵 $W$：稀疏随机（$p = 0.15$），行归一化，无自连接。脉冲与扩散共享拓扑。

## 3. 输入注入

输入 $u(t) \in \mathbb{R}^{d_{\text{in}}}$ 经固定随机矩阵 $M \in \mathbb{R}^{N \times d_{\text{in}}}$（$M_{ik} \sim U(0.1, 1.0)$）投影，再乘以节点增益 $G_i \sim U(0.3, 1.5)$：

$$I_i^{\text{ext}} = G_i \sum_k M_{ik} u_k(t)$$

## 4. 特征提取与读出

### 4.1 节点状态特征

每节点提取 3 维原始状态：$(c_i, s_i, a_i)$。总特征维度 $N \times 3$。主实验使用 $N=64$（NARMA/MC）或 $N=32$（分类/消融）。

不提取事件统计特征（事件数、平均 IEI、CV IEI），因为在低事件率下这些特征存在零方差死维度问题。事件信息间接通过脉冲痕迹 $s_i^{\text{pulse}}$（在全体状态中已编码）和慢适应 $a_i$ 捕获。

### 4.2 时序摘要（分类任务）

分类任务使用三分段均值+终态将时序轨迹编码为固定维度：

$$\text{feat} = [\text{mean}(F_{1:T/3}), \text{mean}(F_{T/3:2T/3}), \text{mean}(F_{2T/3:T}), F_T]$$

总维度 $N \times 3 \times 4 = 12N$。

### 4.3 延迟嵌入（可选）

$$z(t) = [F(t), F(t - \Delta), F(t - 2\Delta), \dots, F(t - (K-1)\Delta)]$$

默认 $K = 5$, $\Delta = 4$ 步（0.04 时间单位）。当前主实验未使用延迟嵌入以避免维度爆炸。

### 4.4 虚拟节点

用少量真实 CEU + 大量等间隔延迟 taps 构造虚拟节点：
- 8 CEU + 40 virtual taps = 320 维
- 32 CEU + 5 taps = 960 维（基准）
- 128 CEU = 384 维（上限）

### 4.5 读出层

- 线性：Ridge 回归（分类：RidgeClassifier，回归：Ridge）
- NARMA 任务：$\alpha = 100$（高正则化抑制过拟合）
- 其他回归任务：$\alpha = 10^{-4}$
- 多项式：degree-2 特征扩展 + Ridge（可选，仅 NARMA）
- 轻量 MLP：单隐层 64 神经元（ReLU, $\alpha = 10^{-4}$, early stopping，可选）

特征标准化（z-score）应用于所有读出。死维度过滤（训练集方差 < 1e-10）在标准化前执行。

## 5. Benchmark 协议

| 任务 | 类型 | 协议 |
|------|------|------|
| A. 节律分类 | 二分类 | 1Hz vs 3Hz 正弦，200 trials，T=800，70/30 split |
| B. 复杂节律 | 三分类 | 脉冲占空比 vs 标准正弦 vs 调幅正弦，300 trials，T=1000 |
| C. Temporal XOR | 二分类 | 两极性脉冲 XOR，200 trials，T=600, 70/30 split |
| D. 记忆容量 | 回归 | 延迟 $k = 1\dots30$，6000 步，前 100 丢弃 |
| E. NARMA-10 | 回归 | 4000 步，前 3000 训练 / 后 1000 测试，跳过前 100 |
| F. NARMA-20 | 回归 | 6000 步，前 4000 训练 / 后 2000 测试（可选） |

评价指标：分类用 Accuracy，回归用 NMSE 和 MSE。

## 6. 基线

| 方法 | 配置 |
|------|------|
| 标准 ESN | N=64（分类/MC）/ 128（NARMA），tanh，谱半径 0.95，leak 0.3，sparsity 0.15 |
| Hopf v1 | N=32，完整耦合（扩散+平均场+脉冲），延迟嵌入 5 taps |
| CEU 无脉冲 | N=32，仅弱扩散 ($g_p = 0$)，相同读出 |
| 线性基线 | 输入延迟线（10 taps，delay_step=3）+ Ridge |

## 7. 消融实验

1. 去脉冲耦合 ($g_p = 0$) → 验证事件耦合
2. 去慢适应 ($\alpha_a = 0$) → 验证超慢变量
3. 去振幅依赖调制 ($\eta = 0$) → 验证振幅门控
4. 同质参数 → 验证异质性
5. 去事件统计特征 → 验证事件编码（主实验已默认不使用）
6. 临界 vs 亚临界扫描 → 验证临界性优势（$g_p \in [0, 0.4]$）

## 8. 临界性分析

扫描 $g_p \in [0, 0.4]$（9 点），计算：
- 雪崩大小/持续时间分布及幂律指数
- 分支比 $\hat{\sigma} = \langle E(t+1)/E(t) \mid E(t) > 0 \rangle$
- 同步序参量 $R = |\langle e^{i c_i} \rangle|$
- 状态协方差参与率 PR

临界区定义为分支比 $\approx 1$ 且 PR 最大的 $g_p$ 区间。

## 9. 默认超参数

- 主实验节点数：$N = 64$（NARMA/MC），$N = 32$（分类/消融/扫描）
- 稀疏连接 $p = 0.15$
- $dt = 0.01$，暖机 500 步
- $g_p = 0.25, g_d = 0.015, \eta = 0.4, \tau_p = 0.08, \theta_{\text{event}} = 0.6$
- 读出 $\alpha = 100$（NARMA），$\alpha = 10^{-4}$（其他）
- 状态类型：float32（内存优化）

## 10. 内存优化

- 所有状态数组（$c_i, s_i, a_i, s_i^{\text{pulse}}$）使用 `np.float32`
- 特征矩阵在构造时显式转换 `.astype(np.float32)`
- 每个实验步骤后调用 `gc.collect()`
- 估计全流程峰值内存 < 500 MB，可在 8 GB MacBook Air 上运行
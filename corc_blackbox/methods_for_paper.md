# Methods for Paper

## 1. 节点动力学

### 1.1 Hopf / Stuart–Landau 振子

每个节点 $i$ 的二维快变量 $(x_i, y_i)$ 遵循：

$$
\begin{aligned}
\dot{x}_i &= \bigl(\mu_i - r_i^2 - \beta_i a_i\bigr) x_i - \omega_i y_i + G_i u_i(t) + C_{i,x}(t) + \sigma_i \xi_{i,x}(t) \\
\dot{y}_i &= \bigl(\mu_i - r_i^2 - \beta_i a_i\bigr) y_i + \omega_i x_i + C_{i,y}(t) + \sigma_i \xi_{i,y}(t)
\end{aligned}
$$

其中 $r_i^2 = x_i^2 + y_i^2$，$\xi_{i,x}, \xi_{i,y}$ 为独立标准高斯白噪声。

### 1.2 一维慢适应

$$\tau_{a,i} \, \dot{a}_i = -a_i + \alpha_i r_i^2$$

- $\tau_{a,i}$ 控制慢变量时间尺度（通常 $\gg$ 振荡周期）
- $\alpha_i$ 调节振幅对慢变量的驱动
- $\beta_i$ 决定慢变量对振幅的负反馈强度

### 1.3 参数采样

所有节点参数均独立均匀采样：

| 参数 | 区间 | 说明 |
|------|------|------|
| $\mu_i$ | $[0.08, 0.18]$ | 振荡增益 |
| $f_i$ | $[0.8, 2.5]$ Hz | 本征频率，$\omega_i = 2\pi f_i$ |
| $\tau_{a,i}$ | $[0.8, 2.5]$ | 慢变量时间常数 |
| $\alpha_i$ | $[0.8, 1.5]$ | 振幅→慢变量增益 |
| $\beta_i$ | $[0.8, 1.5]$ | 慢变量→振幅抑制 |
| $\sigma_i$ | $[0.005, 0.02]$ | 噪声强度 |

积分步长 $dt = 0.01$，采用 Euler–Maruyama 数值积分。

---

## 2. 耦合机制

总耦合为三项之和：$C_i = C_{i,\text{diff}} + C_{i,\text{mf}} + C_{i,\text{pulse}}$。

### 2.1 弱扩散耦合

$$C_{i,\text{diff}} = g_d \sum_{j=1}^N W_{ij} (z_j - z_i) \cdot \bigl(1 + \eta \, r_j(t)\bigr)$$

- $W_{ij}$ 为稀疏随机连接矩阵（连接概率 $p=0.15$，对角元为 0），行归一化
- $g_d \in [0, 0.05]$ 为扩散耦合强度
- $\eta \in [0, 1]$ 为振幅依赖调制因子

### 2.2 平均场耦合

$$C_{i,\text{mf}} = g_m \bigl(\bar{z} - z_i\bigr), \qquad g_m \in [0, 0.05]$$

### 2.3 脉冲耦合

事件触发条件：振幅 $r_i$ **向上穿越**阈值 $\theta$（$\theta \approx 1.0$）。

当节点 $j$ 触发事件时，向其脉冲痕迹变量注入单位脉冲：

$$s_j(t^+) = s_j(t^-) + 1$$

痕迹按指数衰减：

$$\dot{s}_j = -\frac{s_j}{\tau_p}, \qquad \tau_p \in [0.03, 0.2]$$

脉冲耦合对节点 $i$ 的驱动为：

$$C_{i,\text{pulse}}(t) = g_p \sum_j W_{p,ij} \, s_j(t), \qquad g_p \in [0, 0.3]$$

脉冲耦合拓扑 $W_p$ 可与扩散耦合共享或独立设定。本实现默认复用 $W$。

---

## 3. 输入注入

输入信号 $u(t) \in \mathbb{R}^{d_{\text{in}}}$ 经固定随机矩阵 $M \in \mathbb{R}^{N \times d_{\text{in}}}$ 投影：

$$u_i(t) = \sum_k M_{ik} u_k(t)$$

再以节点增益 $G_i$ 加入 $x_i$ 方程：

$$\text{input}_{i,x} = G_i \, u_i(t)$$

辅助频率调制（可选）：

$$\omega_i(t) = \omega_{i,0} + \gamma_i u_i(t)$$

在本基准套件中，主要驱动为幅值驱动，频率调制仅作可选项，避免破坏瞬态。

---

## 4. 状态观测与读出特征

### 4.1 原始状态变量

- 连续状态：$x_i, y_i, a_i$
- 派生：$r_i = \sqrt{x_i^2 + y_i^2}$，$\phi_i = \arctan2(y_i, x_i)$
- 事件状态：脉冲痕迹 $s_i(t)$，事件计数 $e_i(t)$（滑动窗口内阈值穿越次数）

### 4.2 延迟嵌入（Delay Embedding）

对每个选定的特征向量 $h(t) \in \mathbb{R}^{N \cdot F}$（$F$ 为每节点特征数），构造：

$$z(t) = \bigl[ h(t), \; h(t-\Delta), \; \dots, \; h\bigl(t-(K-1)\Delta\bigr) \bigr]$$

默认参数：$K=4$，$\Delta = 3$ 个仿真步长（$0.03$ 时间单位）。

### 4.3 读出层训练

- **分类**：岭分类器（RidgeClassifier，$\alpha = 10^{-4}$）
- **回归**：岭回归（Ridge，$\alpha = 10^{-4}$）
- 可选上界：轻量 MLP（本基准以线性读出为主，MLP 仅作参考）

所有读出模型均在延迟嵌入状态上进行训练与测试。

---

## 5. Benchmark 协议

### 5.1 任务定义

| 任务 | 类型 | 说明 |
|------|------|------|
| A. Rhythm | 二分类 | 区分低频（$0.5$ Hz）与高频（$2.0$ Hz）正弦输入 |
| B. Memory Capacity | 回归 | 计算延迟 $k$ 的记忆函数 $MC_k$ 及总容量 |
| C. NARMA-10 | 回归 | $y_{t+1} = 0.3 y_t + 0.05 y_t \sum_{i=0}^9 y_{t-i} + 1.5 u_{t-9} u_t + 0.1$ |
| D. Temporal XOR | 二分类 | 两个带极性脉冲的时间 XOR |
| D. Switching Rhythm | 二分类 | 持续低频 vs 高频节律分类 |

### 5.2 训练 / 测试划分

- 分类任务：70% 训练，30% 测试
- NARMA-10：前 3000 步训练，后 1000 步测试
- Memory Capacity：全程 5000 步，前 100 步丢弃（瞬态），计算 $k=1\dots 20$ 的 $MC_k$

### 5.3 评价指标

- 分类：**Accuracy**
- 回归：**NMSE**（归一化均方误差），**MSE**
- 记忆容量：$MC_k = \text{corr}^2(y_{\text{pred}}, u_{\text{delayed}})$，总容量 $MC_{\text{tot}} = \sum_k MC_k$

---

## 6. 临界性与雪崩分析

### 6.1 雪崩定义

将网络事件视为二值化活动：若某时间步 $t$ 存在至少一个节点触发阈值穿越，则该步为**活跃**。雪崩定义为连续活跃时间步的集合。

- **Size** $S$：雪崩期间全网累计事件数
- **Duration** $D$：雪崩持续的时间步数

### 6.2 统计方法

对 Size 与 Duration 进行 log-binning 直方图，并用线性回归估计幂律指数 $\alpha$：

$$\log P(S) \sim -\alpha \log S$$

### 6.3 分支比估计

采用简化代理指标：

$$\hat{\sigma} = \left\langle \frac{E(t+1)}{E(t)} \biggm| E(t) > 0 \right\rangle$$

其中 $E(t) = \sum_i e_i(t)$ 为全局事件计数。

### 6.4 同步序参量

Kuramoto 复序参量：

$$Z(t) = \frac{\bigl|\sum_i r_i(t) e^{i\phi_i(t)}\bigr|}{\sum_i r_i(t)}$$

$\langle Z \rangle \approx 0$ 表示异步；$\langle Z \rangle \approx 1$ 表示强同步。

### 6.5 瞬态丰富度

状态协方差矩阵的前 $95\%$ 方差对应的 PCA 维数记为 $D_{\text{cov}}$。参与率（Participation Ratio）$PR = (\sum \lambda_i)^2 / \sum \lambda_i^2$。

---

## 7. 消融实验设计

| 条件 | 操作 | 目的 |
|------|------|------|
| 去掉慢变量 | $\beta = 0, \alpha = 0$ | 验证慢适应的必要性 |
| 去掉脉冲耦合 | `disable_pulse=True` | 验证脉冲事件机制 |
| 去掉振幅依赖 | $\eta = 0$ | 验证振幅门控耦合 |
| 去掉异质性 | 所有节点同参数 | 验证参数异质性 |
| 去掉事件读出 | 特征不含 $s_i$ | 验证事件特征贡献 |
| 扫描耦合强度 | $g \in [0, 0.15]$ | 定位临界区 |

---

## 8. 统计检验

所有分类任务重复 5 次独立随机种子，报告均值 ± 标准差。采用双样本 $t$-检验比较 CORC 与 ESN 的准确率差异（$\alpha = 0.05$）。

---

## 9. 默认超参数（全文一致）

- $N = 32$
- 稀疏连接概率 $p = 0.15$
- $dt = 0.01$
- $g_d = 0.01, \; g_m = 0.01, \; g_p = 0.08$
- $\eta = 0.4, \; \tau_p = 0.08, \; \theta = 1.0$
- 延迟嵌入 taps $K = 4$，间隔 3 步
- 暖机步数 500 步
- 读出正则化 $\alpha = 10^{-4}$

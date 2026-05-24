# Group Sequence Policy Optimization

## 基本信息

- **标题**: Group Sequence Policy Optimization
- **作者**: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin
- **发表日期**: 2025年7月
- **arXiv ID**: 2507.18071
- **arXiv 链接**: <https://arxiv.org/abs/2507.18071>
- **领域/关键词**: GSPO, Sequence-level Policy Optimization, RLVR, LLM Training, Qwen3, GRPO, Importance Sampling, MoE

## 一句话总结
>
> GSPO 提出用序列似然（sequence likelihood）定义重要性比率替代 GRPO 的 token 级重要性比率，从根源上解决 GRPO 在大规模 RL 训练中的不稳定性问题，并在 Qwen3 系列模型上验证了其优越性。

## 论文精读

### 背景与前置知识

**GRPO（Group Relative Policy Optimization）** 是当前训练大语言模型的主流 RL 算法之一。它通过为同一查询生成一组响应（group），并利用组内奖励的相对排名来估计优势函数（advantage），从而绕过了 PPO 中对价值模型（value model）的依赖。GRPO 的核心优化目标对每个 token 使用 token 级重要性比率 w_{i,t}(θ) = π_θ(y_{i,t}|x, y_{i,<t}) / π_{θ_old}(y_{i,t}|x, y_{i,<t})，并通过裁剪（clipping）机制限制策略更新幅度。

**Token 级重要性比率** 是 PPO 和 GRPO 共同的设计选择，即在序列的每个 token 位置分别计算新旧策略的概率比值。这一设计在理论上源于重要性采样（importance sampling），旨在纠正从旧策略采样带来的分布偏差。

**RLVR（Reinforcement Learning from Verifiable Rewards）** 是指利用可验证奖励（如数学题的正确性、代码的通过率）进行强化学习的范式。随着模型规模和响应长度的增长，RLVR 的训练稳定性成为关键挑战。

### 核心思想详解

GSPO 的核心洞察在于：**GRPO 的 token 级重要性比率是对重要性采样的错误应用**。

重要性采样的基本原理要求通过多个样本来估计期望值，即用 π_tar(z)/π_beh(z) 对函数值进行加权。但在 GRPO 中，每个 token 位置的重要性权重仅基于单个样本（即该位置实际生成的 token），而非对整个 next-token 分布进行采样平均。因此，这些 token 级权重无法有效纠正分布偏差，反而引入了高方差的训练噪声。

这种噪声会随着响应长度的增加而累积，并在裁剪机制的作用下被放大，最终导致灾难性的模型崩溃（model collapse）。一旦崩溃发生，即使回退到之前的检查点并精心调整超参数，也无法恢复训练。

GSPO 的解决方案是：**将优化单元从 token 级别提升到序列级别**。具体而言，GSPO 基于序列似然定义重要性比率 s_i(θ) = [π_θ(y_i|x) / π_{θ_old}(y_i|x)]^{1/|y_i|}，在序列整体层面执行裁剪、奖励和优化，使得优化的单元与奖励的单元（整个响应）完全一致。

### 方法逐步拆解

1. **序列似然（Sequence Likelihood）的定义**：对于响应 y_i，GSPO 计算其在策略 π_θ 下的似然 π_θ(y_i|x)，即所有 token 条件概率的乘积。这与 token 级概率不同，它反映的是整个响应在新旧策略下的分布偏差程度。

2. **长度归一化的序列级重要性比率**：为避免响应长度差异导致重要性比率数值范围不一致，GSPO 对重要性比率取长度归一化：s_i(θ) = [π_θ(y_i|x) / π_{θ_old}(y_i|x)]^{1/|y_i|}。这等价于对 token 级对数似然的平均取指数。长度归一化确保了不同长度响应的重要性比率处于统一的数值范围，也避免了少数 token 的似然变化引起序列级重要性比率的剧烈波动。

3. **序列级裁剪（Sequence-level Clipping）**：GSPO 对整个响应的重要性比率 s_i(θ) 进行裁剪，而非对每个 token 单独裁剪。裁剪范围（论文中设为 3e-4 和 4e-4 左右）与 GRPO 的裁剪范围（0.2 和 0.27）在数量级上完全不同，这是因为两种重要性比率的定义方式不同。

4. **组级优势估计**：与 GRPO 相同，GSPO 使用组内奖励的标准化值作为优势估计：Ã_i = [r(x, y_i) - mean] / std。所有 token 共享同一个优势值。

5. **梯度分析的关键差异**：从梯度角度看，GSPO 对响应内所有 token 的对数似然梯度赋予相同的权重（即序列级重要性比率），而 GRPO 则为不同 token 赋予不同的权重（即各自的 token 级重要性比率）。GRPO 中这些不相等的权重可以在 (0, 1+ε] 或 [1-ε, +∞) 范围内变化，随着训练推进会产生不可预测的累积效应。

6. **GSPO-token 变体**：在需要更细粒度的优势调整（如多轮 RL）的场景下，GSPO 还提供了 token 级变体。其巧妙之处在于用 stop-gradient 技术将序列级重要性比率与 token 级调整解耦，使数值上等价于 GSPO，但允许按 token 定制优势值。

### 关键公式/算法解读

**GSPO 的核心目标函数**（公式 5）：

J_GSPO(θ) = E_{x~D, {y_i}~π_{θ_old}} [1/G * Σ_{i=1}^{G} min(s_i(θ) * Ã_i, clip(s_i(θ), 1-ε, 1+ε) * Ã_i)]

其中：
- s_i(θ) = [π_θ(y_i|x) / π_{θ_old}(y_i|x)]^{1/|y_i|}（公式 7）：长度归一化的序列级重要性比率
- Ã_i = [r(x,y_i) - mean{r(x,y_i)}] / std{r(x,y_i)}（公式 6）：组级标准化优势

**核心区别**：与 GRPO 的目标函数（公式 2）相比，GSPO 将求和维度从"组 x 序列长度"降为"组"，即裁剪和重要性加权都在序列级别进行。

**梯度形式**（公式 10）：GSPO 的梯度为所有 token 的对数似然梯度乘以统一的序列级权重，而 GRPO 的梯度（公式 12）中每个 token 有各自的权重。这一区别解释了 GSPO 为何更稳定——它消除了 token 级权重差异带来的方差。

### 实验设计分析

论文基于 Qwen3-30B-A3B-Base 的冷启动模型进行了系统实验：

- **训练设置**：每个批次的 rollout 数据被分为 4 个 mini-batch 进行梯度更新。GSPO 的裁剪范围设为 (3e-4, 4e-4)，GRPO 的裁剪范围设为 (0.2, 0.27)（经过仔细调参以确保公平比较）。
- **评估基准**：AIME'24（32 次采样的平均 Pass@1）、LiveCodeBench 202410-202502（8 次采样的平均 Pass@1）、CodeForces（Elo 等级分）。
- **主要发现**：
  - GSPO 在整个训练过程中保持稳定，可通过增加训练计算量、定期更新查询集和延长生成长度持续提升性能。
  - GSPO 展现出更高的训练效率，在相同的训练计算量和查询消耗下，实现了更好的训练精度和基准性能。
  - 在 MoE 模型上，GSPO 天然解决了 GRPO 所需的 Routing Replay 策略带来的稳定性问题。

- **关于裁剪比例的反直觉发现**：GSPO 裁剪的 token 比例比 GRPO 高两个数量级（约 10% vs 约 0.13%），但仍然取得了更高的训练效率。这说明 GRPO 的 token 级梯度估计本身就存在噪声和低效问题。

- **MoE 模型的关键优势**：在 48 层 Qwen3-30B-A3B MoE 模型中，每次梯度更新后约有 10% 的专家激活发生变化，这使得 token 级重要性比率剧烈波动。GSPO 仅关注序列似然，不受个别 token 的专家切换影响，从根本上解决了这一问题，消除了对 Routing Replay 策略的依赖。

## 问题与动机

- **核心问题**：GRPO 在训练大规模语言模型时存在严重的不稳定性，经常导致灾难性且不可逆的模型崩溃。
- **根本原因**：GRPO 的 token 级重要性比率是对重要性采样原理的错误应用——仅基于单个样本无法有效纠正分布偏差，反而引入高方差噪声。
- **关键洞察**：优化目标的单元应与奖励的单元保持一致。既然奖励是针对整个序列的，off-policy 修正也应在序列级别进行。

## 核心方法

- 基于序列似然定义重要性比率，配合长度归一化控制数值范围
- 在序列级别执行裁剪、奖励和优化，使三者单元统一
- 通过梯度分析证明所有 token 获得等权更新，消除了 GRPO 中 token 级权重差异的不稳定性
- 提供 GSPO-token 变体支持更细粒度的优势调整需求

## 实验结果

- **数据集**: AIME'24, LiveCodeBench (202410-202502), CodeForces
- **主要指标**: Pass@1 (AIME'24, LiveCodeBench), Elo Rating (CodeForces), Training Reward
- **关键结果**: GSPO 在所有基准上优于 GRPO，训练效率显著更高；MoE 模型无需 Routing Replay 即可稳定训练；已成功应用于 Qwen3 系列模型的 RL 训练
- **对比方法**: GRPO（含/不含 Routing Replay）

## 关键图表

- **Figure 1**: GSPO 与 GRPO 在训练奖励、AIME'24、LiveCodeBench 和 CodeForces 上的训练曲线对比，GSPO 在所有指标上展现出更高的训练效率
- **Figure 2**: GSPO 与 GRPO 的裁剪 token 比例对比——GSPO 裁剪了约 10% 的 token，比 GRPO（约 0.13%）高两个数量级，但反而更高效
- **Figure 3**: GRPO 在 MoE 模型上需要 Routing Replay 才能正常收敛，否则训练崩溃

## 局限性

- 论文仅在 Qwen3-30B-A3B 一个模型规模上进行了对比实验，未展示在其他规模或架构上的泛化性
- 裁剪范围的选择（3e-4 到 4e-4）与 GRPO 的量级完全不同，可能需要针对不同场景重新调参
- 论文未讨论 GSPO 在非 MoE 的 dense 模型上相比 GRPO 的具体收益幅度
- 对 GSPO-token 变体缺乏充分的实验验证
- 未探讨长度归一化在极短或极长响应场景下的行为

## 个人思考

GSPO 的核心贡献在于从理论层面揭示了 GRPO 的根本缺陷（token 级重要性比率的不当使用），并提出了一个优雅的替代方案。其"优化单元应与奖励单元一致"的设计原则具有很强的启发性。

值得注意的是，GSPO 的裁剪范围（3e-4 量级）远小于 GRPO（0.2 量级），这意味着在 GSPO 中，只要序列似然发生微小变化就会触发裁剪。这看似更严格，但由于裁剪是在序列级别进行的，它过滤的是整个"过于偏离"的响应，而非单个 token，因此反而更合理。

论文中关于 GSPO 可简化 RL 基础设施的讨论也很有价值——由于仅需要序列级似然，可以直接使用推理引擎返回的似然值，无需训练引擎重新计算，这对于 training-inference disaggregated 框架具有重要意义。

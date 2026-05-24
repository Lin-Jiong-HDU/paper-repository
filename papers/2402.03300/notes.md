# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

## 基本信息

- **标题**: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- **作者**: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
- **发表日期**: 2024年2月
- **arXiv ID**: 2402.03300
- **arXiv 链接**: <https://arxiv.org/abs/2402.03300>
- **领域/关键词**: Mathematical Reasoning, GRPO, Reinforcement Learning, LLM, DeepSeek, PPO, Data Selection, Common Crawl

## 一句话总结
>
> DeepSeekMath 通过从 Common Crawl 中精选 120B 数学 token 进行持续预训练，并提出 GRPO（Group Relative Policy Optimization）强化学习算法，使 7B 模型在竞赛级 MATH 基准上达到 51.7%，接近 GPT-4 水平。

## 论文精读

### 背景与前置知识

**数学推理与语言模型：** 数学推理是语言模型面临的核心挑战之一，因其要求复杂的多步逻辑推理和精确的计算能力。此前，闭源模型 GPT-4 和 Gemini-Ultra 在数学推理方面遥遥领先，而开源模型的表现则显著落后。

**PPO（Proximal Policy Optimization）：** PPO 是一种经典的 actor-critic 强化学习算法，广泛应用于 LLM 的 RLHF 阶段。其核心思路是通过截断概率比（clipping）来限制策略更新幅度，保证训练稳定性。但 PPO 需要同时维护策略模型（policy model）和价值函数（value function），后者通常与策略模型同等规模，带来显著的内存和计算开销。

**RL for LLM 的关键挑战：** 在 LLM 场景下，奖励模型通常只对最后一个 token 给出奖励分数，这使得训练一个在每个 token 上都准确的价值函数变得困难。同时，PPO 中使用 GAE（Generalized Advantage Estimation）计算优势函数时依赖价值函数的质量。

### 核心思想详解

DeepSeekMath 的核心贡献分为两大支柱：

**支柱一：可扩展的数学数据工程。** 论文提出了一个迭代式的数据收集管道，从 Common Crawl 的 40B HTML 页面中筛选出 120B 数学相关 token（DeepSeekMath Corpus），规模约为 OpenWebMath 的 9 倍、Minerva 所用数据的 7 倍。关键创新在于利用 fastText 分类器配合人工标注的迭代式领域发现方法，逐步扩充正例种子集合。

**支柱二：GRPO 算法。** 论文提出的 Group Relative Policy Optimization 是 PPO 的高效变体。其核心思想极其优雅：放弃 PPO 中独立的价值模型（critic），转而对同一问题采样一组输出，用组内奖励的统计量（均值和标准差）来归一化计算优势函数。这一设计既减少了约一半的模型内存开销，又天然契合了奖励模型的比较性质（reward model 本身就是在同一问题的不同输出之间做比较训练的）。

### 方法逐步拆解

**第一步：数据收集与去污染**

1. **种子数据选择：** 以 OpenWebMath 作为初始种子语料
2. **fastText 模型训练：** 使用 50 万正例（OpenWebMath）和 50 万负例（Common Crawl 通用网页）训练二分类器
3. **数学网页召回：** 从去重后的 40B HTML 页面中召回数学内容
4. **领域发现与迭代：** 将 Common Crawl 按域名分组，统计每个域名中被召回页面的比例，超过 10% 的域名标记为数学相关域；人工标注域内的数学 URL 路径，将未收集的页面加入种子集合
5. **迭代终止：** 经过 4 次迭代后停止（第 4 次迭代时 98% 的数据已在第 3 次被收集），最终获得 35.5M 数学网页、120B token
6. **去污染：** 使用 10-gram 精确匹配移除包含 GSM8K、MATH 等基准题目和答案的网页

**第二步：两阶段预训练**

1. **基础模型选择：** 以 DeepSeek-Coder-Base-v1.5 7B 为起点（而非通用 LLM），因为代码训练已被证明能提升数学推理能力
2. **数据配比：** 56% DeepSeekMath Corpus + 4% AlgebraicStack + 10% arXiv + 20% GitHub 代码 + 10% 中英通用数据
3. **训练配置：** 500B token，学习率上限 4.2e-4，batch size 10M token，4K 上下文长度

**第三步：指令微调（SFT）**

- 收集 776K 数学指令数据，涵盖思维链（CoT）、程序思维（PoT）和工具集成推理格式
- 训练 500 步，batch size 256，恒定学习率 5e-5

**第四步：GRPO 强化学习**

1. **采样阶段：** 对每个问题 q，从旧策略采样 G 个输出 {o_1, ..., o_G}
2. **奖励计算：** 使用奖励模型对每个输出打分，得到 {r_1, ..., r_G}
3. **优势计算（核心）：** 对组内奖励做归一化：A_hat = (r_i - mean(r)) / std(r)
4. **策略更新：** 最大化 GRPO 目标函数（包含 clip 机制 + KL 正则化）
5. **训练细节：** 学习率 1e-6，KL 系数 0.04，每组 64 个输出，最大长度 1024

### 关键公式/算法解读

**PPO 目标函数（公式 1）：** 标准 PPO 对每个问题采样一个输出，通过 GAE 计算每个 token 的优势 A_t，使用截断概率比进行策略更新。需要额外的价值函数 V_psi 来估计基线。

**GRPO 目标函数（公式 3）：** GRPO 的核心改动体现在两个层面：

1. **Group Relative Advantage：** 对同一问题采样一组 G 个输出，奖励模型对每个输出打分后，在组内做 Z-score 归一化得到优势值。这完全替代了 PPO 中的价值函数和 GAE。

2. **KL 正则化位置变更：** PPO 将 KL 惩罚加入每步奖励中（公式 2），而 GRPO 直接将 KL 散度项加到最终损失函数中（公式 3 中的 -beta * D_KL），避免 KL 惩罚影响优势函数的计算。

**KL 散度估计（公式 4）：** 使用 Schulman (2020) 提出的无偏估计器：D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1，保证非负性。

**Outcome Supervision vs Process Supervision：**
- Outcome Supervision（结果监督）：整个输出使用同一个归一化奖励作为所有 token 的优势
- Process Supervision（过程监督）：对每个推理步骤分别打分，每个 token 的优势为后续所有步骤归一化奖励之和

**Iterative GRPO（算法 1）：** 支持多轮迭代训练。每轮迭代中：(1) 更新参考模型为当前策略；(2) 用策略模型采样训练数据更新奖励模型（含 10% 历史数据回放）；(3) 用更新后的奖励模型继续训练策略模型。

### 实验设计分析

**数学基准评估：** 覆盖英汉双语，从小学到大学不同难度：
- 英文：GSM8K（小学算术）、MATH（竞赛级）、SAT、OCW（大学课程）、MMLU-STEM
- 中文：MGSM-zh、CMATH、Gaokao-MathCloze、Gaokao-MathQA

**核心实验结果：**
- DeepSeekMath-Base 7B 在 MATH 上达到 36.2%，超过 Minerva 540B（33.6%）
- DeepSeekMath-RL 7B 在 MATH 上达到 51.7%，在 GSM8K 上达到 88.2%
- GRPO 在域内和域外任务上均带来显著提升（GSM8K: 82.9% -> 88.2%，MATH: 46.8% -> 51.7%，CMATH: 84.6% -> 88.8%）

**RL 消融实验关键发现：**
- 在线采样（Online RFT）显著优于离线采样（RFT）
- GRPO 优于 Online RFT，证明基于奖励值调整梯度系数的有效性
- 过程监督（GRPO+PS）优于结果监督（GRPO+OS）
- 迭代 RL 带来进一步提升，尤其是第一轮迭代
- RL 提升 Maj@K 但不提升 Pass@K，说明 RL 主要使输出分布更鲁棒，而非提升基础能力

## 问题与动机

论文试图解决两个核心问题：(1) 如何从公开的网络数据中大规模获取高质量数学训练数据？(2) 如何在有限的计算资源下高效地通过强化学习提升 LLM 的数学推理能力？此前的工作要么依赖小规模的人工整理语料，要么在 RL 阶段使用资源密集的 PPO 算法。DeepSeekMath 的动机是证明：通过精心的数据工程和算法创新，小模型（7B）也能在数学推理上接近甚至媲美大得多的模型。

## 核心方法

**数据工程：** 迭代式 fastText 分类器 + 领域发现管道，从 Common Crawl 中提取 120B 数学 token。

**GRPO 算法：** 用组内相对奖励替代 PPO 的价值函数，将 KL 正则化从奖励函数移至损失函数，实现更高效的策略优化。

**统一范式（Unified Paradigm）：** 论文提供了一个统一框架来理解 SFT、RFT、DPO、Online RFT、PPO、GRPO 等方法——所有方法的梯度都可以表示为 E[GC * grad log pi_theta] 的形式，区别仅在于数据来源（Data Source）、奖励函数（Reward Function）和梯度系数（Gradient Coefficient）三个维度。

## 实验结果

- **数据集**: GSM8K, MATH, SAT, OCW, MMLU-STEM, MGSM-zh, CMATH, Gaokao-MathCloze, Gaokao-MathQA, miniF2F, MMLU, BBH, HumanEval, MBPP
- **主要指标**: Top-1 准确率（主）、Maj@K、Pass@K
- **关键结果**: MATH 51.7%（开源最优，接近 GPT-4 的 52.9%），GSM8K 88.2%（开源最优），Self-consency 64 样本在 MATH 上达 60.9%
- **对比方法**: GPT-4, Gemini-Ultra, Minerva 540B, Llemma 34B, WizardMath, MetaMath 70B, InternLM2-Math 20B, Qwen 72B

## 关键图表

- **Figure 1**: MATH 基准上开源模型的 Top-1 准确率对比，DeepSeekMath-RL 7B 遥遥领先
- **Figure 2**: 迭代式数学数据收集管道示意图，展示从种子到大规模语料的四轮迭代过程
- **Figure 4**: PPO 与 GRPO 的架构对比图，直观展示了 GRPO 用组内计算替代价值模型的设计
- **Figure 5**: 不同 RL 方法（RFT、Online RFT、GRPO+OS、GRPO+PS）的训练曲线对比
- **Figure 6**: 迭代 RL 的性能变化，第一轮迭代带来最大提升
- **Figure 7**: Maj@K 和 Pass@K 的对比分析，揭示 RL 的工作机制——增强分布鲁棒性而非基础能力
- **Table 2**: 基础模型在 8 个数学基准上的全面对比，DeepSeekMath-Base 7B 全面领先开源模型
- **Table 5**: 最终模型（Instruct + RL）与开源/闭源模型的全面对比

## 局限性

1. **几何与定理证明较弱：** 模型在处理三角形、椭圆等几何问题时表现不佳，可能反映了预训练和微调中的数据选择偏差
2. **Few-shot 能力受限：** 受限于 7B 模型规模，DeepSeekMath 的 few-shot 能力不如 GPT-4，零样本和少样本评估表现相似
3. **RL 提升机制有限：** RL 主要提升 Maj@K 而非 Pass@K，说明其改善的是输出分布的鲁棒性而非模型的根本推理能力
4. **arXiv 数据效果存疑：** 论文发现 arXiv 论文在提升数学推理方面似乎无效，但作者也承认该结论有待进一步验证（不同模型规模、不同任务组合等）

## 个人思考

这篇论文最值得关注的是 GRPO 算法的设计哲学——用一个简单而优雅的"组内相对"机制替代了复杂的价值函数。这一思路后来被 DeepSeek 团队进一步发展，在 DeepSeek-R1 等后续工作中发挥了关键作用。GRPO 的核心洞察在于：奖励模型本质上就是在做比较判断，与其训练一个独立的价值函数来估计基线，不如直接利用多次采样的组内统计量。这种设计不仅减少了计算开销，还天然与奖励模型的训练范式对齐。

从数据工程角度看，论文证明了一个重要观点：公开网络数据中蕴含着大量高质量数学内容，关键在于如何精准地筛选。迭代式领域发现方法具有很强的通用性，可以推广到代码、科学论文等其他领域。

关于 RL 的局限性分析（提升 Maj@K 而非 Pass@K）值得深思。这暗示当前的 RL 方法可能更多是在做"对齐"而非"提升"——让模型更可靠地输出它已经知道的东西，而不是让它学会解决新问题。如何突破这一限制，让 RL 真正提升模型的基础能力，仍然是一个开放的研究方向。

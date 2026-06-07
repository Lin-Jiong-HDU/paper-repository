# UI-Ins: Enhancing GUI Grounding with Multi-Perspective Instruction-as-Reasoning

## 基本信息

- **标题**: UI-Ins: Enhancing GUI Grounding with Multi-Perspective Instruction-as-Reasoning
- **作者**: Liangyu Chen, Hanzhang Zhou, Chenglin Cai, Jianan Zhang, Panrong Tong, Quyu Kong, Xu Zhang, Chen Liu, Yuqi Liu, Wenxuan Wang, Yue Wang, Qin Jin, Steven HOI
- **发表日期**: 2025-10-23
- **arXiv ID**: 2510.20286
- **arXiv 链接**: <https://arxiv.org/abs/2510.20286>
- **领域/关键词**: GUI Grounding, Instruction-as-Reasoning, Multi-Perspective Reasoning, Reinforcement Learning, GRPO, GUI Agent, Vision-Language Model

## 一句话总结
>
> 本文提出 Instruction-as-Reasoning 范式，将 GUI grounding 中的自然语言指令从静态输入升级为动态推理路径，通过 SFT+GRPO 两阶段训练让模型学会从多视角分析指令并选择最优推理路径，在五个主流 benchmark 上取得 SOTA。

## 论文精读

### 背景与前置知识

**GUI Grounding（GUI 元素定位）**：给定一张 GUI 截图和一条自然语言指令，模型需要预测目标 UI 元素在屏幕上的坐标点。这是 GUI Agent 完成用户任务的核心基础能力——如果模型连"点哪里"都搞错了，后面的自动化操作就无从谈起。

**指令的多样性（Instruction Diversity）**：人类在描述一个 UI 操作时，会自然地切换不同视角。比如"关闭窗口"这个意图，可以说"点那个红色的 X"（外观视角），也可以说"关闭文件管理器"（功能视角），或者说"点右上角的按钮"（位置视角），或者说"把这个屏幕关掉"（意图视角）。这四种描述指向同一个操作，但从不同角度表述。

**GRPO（Group Relative Policy Optimization）**：一种强化学习算法，是 DeepSeek-R1 中使用的优化方法。核心思想是对同一输入生成多个 rollout，用组内相对优势（Z-score 归一化后的奖励）来更新策略，不需要单独的 critic 模型。

**OmniParser**：一个纯视觉的 GUI 解析工具，能从截图中检测出所有可交互的 UI 元素及其边界框。

### 核心思想详解

这篇论文的核心洞察可以用一个简单的类比来理解：

想象你去餐厅点菜。服务员给你菜单，你不会用"物理学"的方式描述想吃的菜（"我要一份由碳氢化合物组成的、经过美拉德反应的蛋白质混合物"），而是会说"我要一份红烧肉"。但如果菜单上有两道看起来很相似的菜，你可能会切换到外观视角："就是图片上那个红红的、装在砂锅里的那个"。如果还不行，你会加上位置信息："第三页左上角那个"。

这就是 **Instruction-as-Reasoning** 的本质：**不同的指令表述不是简单的同义改写，而是不同的分析视角（reasoning pathway）**。优秀的 GUI Agent 不应该只会机械地匹配指令文字和 UI 元素，而应该能从多个角度理解用户意图，并自动选择最有效的一个。

作者在 ScreenSpot-Pro 上做了一个关键实验：把原始指令手动改写成四种不同视角（外观、功能、位置、意图），用同一个模型（Qwen2.5-VL-7B）零样本测试。结果发现：四种视角的准确率差异巨大（18.9% → 35.2%），而且如果能针对每个样本选择最佳视角（"Combined"），准确率能达到 64.1%——相比原始指令的 36.4%，相对提升高达 **76%**。

这个发现直接推导出方法设计：**先用 SFT 教会模型从多种视角推理，再用 RL 让模型学会选择最优视角。**

### 方法逐步拆解

**第一步：数据清洗与增强（Data Pipeline）**

1. **预处理**：用 OmniParser V2 检测截图中所有 UI 元素，通过 IoU 匹配找到与 ground-truth 最匹配的边界框，过滤掉不匹配的噪声样本。这一步把原数据中 23.3% 的质量问题降到了 8% 以下。

2. **多视角指令增强**：对每个样本，用 GPT-4.1 生成四种视角的指令：
   - **外观（Appearance）**："点那个红色的 X 图标"
   - **功能（Function）**："关闭当前文件管理器窗口"
   - **位置（Spatial/Location）**："点右上角的按钮"
   - **意图（Goal/Intent）**："关掉这个屏幕"

3. **验证过滤**：GPT-4.1 再次检查每条生成的指令，确认它唯一指向目标元素（不能有歧义，不能指向多个元素）。

**第二步：SFT 阶段——学习多样推理**

在 SFT 阶段，训练数据被构造成：输入 =（截图 + 一种视角的指令），输出 =（另一种视角的推理文本 + 最终坐标）。

举个例子：
- 输入指令（功能视角）："点击 CSDN 书签"
- 模型思考输出（外观视角）："我将从外观角度分析——点击书签栏中带有红色 C 图标和 CSDN 标签的书签"
- 最终输出：坐标 [588, 67]

这种设计巧妙的地方在于：它强制模型学会**把一种指令翻译成另一种视角的描述**，然后再根据翻译后的描述定位元素。经过大量训练后，模型内化了"多视角推理"的能力。

**第三步：RL 阶段——学习选择最优视角**

SFT 教会了模型"会"从多视角推理，但没教它"什么时候用什么视角"。RL 阶段解决这个问题。

- 训练时**不再指定使用哪种视角**，只要求模型"think"然后再输出坐标
- 用 GRPO 算法优化，奖励函数是简单的 point-in-box：预测点在 ground-truth 框内 → reward=1，否则 → reward=0
- 模型通过探索（生成 8 个 rollout），自己发现哪些推理路径能带来更高准确率

**关键设计细节**：
- SFT 阶段的 prompt 明确列出了四种视角的定义（Appearance/Function/Spatial/Goal Perspective）
- RL 阶段的 prompt 去掉了这些定义，只说"你先想想再回答"，鼓励自由探索
- 这样设计是因为：如果 RL 阶段还给定视角，模型就只会从这四个里选；去掉限制后，模型甚至能涌现出训练时没见过的全新推理视角

### 关键公式/算法解读

**SFT 阶段的目标函数**（公式 1）：

```
max Σ log P(Y_gt | S, I; θ)
Y_gt = R_gt ⊕ p_gt
```

翻译成自然语言：最大化模型生成正确输出序列的概率。其中 `Y_gt` 是目标序列，由两部分拼接而成——推理文本 `R_gt`（从增强的指令视角中随机采样一个）和坐标点 `p_gt`。模型被要求先"说"出推理过程，再"点"出坐标。

**GRPO 的优势计算**（公式 2）：

```
Â_i = (r_i - mean(r)) / std(r)
```

对一组 G 个 rollout 的奖励做 Z-score 标准化。这个设计的直觉是：不是关心"这个动作的绝对好坏"，而是关心"在当前这组尝试里，这个动作比平均好多少"。这避免了奖励尺度不一致的问题。

**GRPO 的优化目标**（公式 3）：

```
L = - (1/G) Σ [π(o_i|I,S) / π_old(o_i|I,S)] · Â_i
```

翻译：对于每个 rollout `o_i`，如果它的优势 `Â_i` 是正的（比平均好），就增大它在新策略下被选中的概率；如果是负的（比平均差），就减小。除以旧策略概率是为了做重要性采样修正。乘以 `G` 分之一是对所有 rollout 取平均。

### 实验设计分析

**为什么选这些 benchmark？**
- **MMBench-GUI L2**：测试对层次化指令的理解，分 Basic（直接描述外观）和 Advanced（描述用途/意图）两个难度，能区分模型是真的理解了指令还是只会匹配文字
- **UI-I2E-Bench**：分 Explicit（显式）和 Implicit（隐式）指令，测试模型的语义推理能力
- **ScreenSpot-Pro**：高分辨率专业软件截图（CAD、开发工具、创意软件），测试模型对复杂界面的感知能力
- **ScreenSpot-V2 & ShowDown**：跨平台（Mobile/Desktop/Web）的通用 grounding 测试
- **AndroidWorld**：动态在线环境，测试真实场景中 grounding 的稳定性和泛化能力

**实验结果说明了什么？**
1. UI-Ins-32B 在所有 benchmark 上都是 SOTA，特别是在理解和推理类任务上优势更明显
2. 在 MMBench-GUI L2 上，UI-Ins 相对基础模型的提升在 Advanced 子集上更大（24.5% vs 12.3%），说明多视角推理确实增强了复杂指令理解
3. AndroidWorld 上 74.1% 的成功率（比 Qwen2.5-VL-7B 的 50.0% 高 24.1%）说明强大的 grounding 能直接转化为更好的 agent 表现

**消融实验的关键发现：**
- 去掉中间推理步骤 → UI-I2E 准确率下降超过 10%，说明推理过程是必需的，不是花架子
- 只用 SFT 或只用 RL → 都不如 SFT+RL 组合，说明两个阶段不可偏废
- Free-Form Reasoning 在 GRPO 中会**降低**性能（UI-Tars-1.5-7B 下降 6.4%），而 Instruction-as-Reasoning 始终提升（Qwen2.5-VL-7B 提升 9.9%）
- 标准 SFT+RL 框架容易出现策略崩溃（RL 后性能反而下降），但 Instruction-as-Reasoning 的 SFT 阶段提供了多样化探索能力，避免了这个问题

## 问题与动机

GUI grounding 中，自然语言指令是最关键的输入之一——它承载了用户意图。但现有工作存在两大问题：

1. **数据质量问题**：作者人工检查了 1,909 条数据，发现 23.3% 存在质量问题（歧义匹配、无匹配等），这会显著损害模型训练效果
2. **指令多样性未被利用**：人类能灵活切换多种描述视角来定位 UI 元素，但现有模型被训练在单一、固定的指令风格上，无法利用视角多样性带来的信息增益

## 核心方法

**Instruction-as-Reasoning** 是一个两阶段训练框架：

1. **数据管线**：清洗噪声标注（OmniParser 检测 + IoU 匹配）+ GPT-4.1 多视角指令增强（外观/功能/位置/意图）+ 验证过滤
2. **SFT 阶段**：在约 283k 样本上训练，每条样本随机选两种视角（一种作输入指令、一种作推理输出），教会模型从多视角推理后再输出坐标
3. **RL 阶段**：在约 100k 样本上用 GRPO 优化，不再指定视角，让模型自由探索并学习选择最优推理路径

基于 Qwen2.5-VL-7B 和 Qwen2.5-VL-32B 两个 backbone，分别训练出 UI-Ins-7B 和 UI-Ins-32B。

## 实验结果

- **数据集**: OS-Atlas, Omniact, Android Control, AMEX, AgentNet（训练）；MMBench-GUI L2, UI-I2E-Bench, ScreenSpot-Pro, ScreenSpot-V2, ShowDown, AndroidWorld（评测）
- **主要指标**: Point-in-box Accuracy
- **关键结果**:
  - UI-Ins-32B: 87.3% UI-I2E, 57.0% ScreenSpot-Pro, 84.9% MMBench-GUI L2（全部 SOTA）
  - UI-Ins-7B: 83.1% MMBench-GUI L2, 81.1% UI-I2E, 52.2% ScreenSpot-Pro
  - AndroidWorld: 74.1% 成功率（UI-Ins-7B + GPT-5 planner）
  - 数据管线将错误率从 23.3% 降至 <8%
  - 消融实验证实推理中间步骤对性能提升超 10%
- **对比方法**: GPT-4o, Claude 3.7, Qwen2.5-VL, UI-TARS-1.5, UGround, GTA1, GUI-Actor, SE-GUI, InfiGUI-G1, GUI-G2, UI-Venus, OpenCUA, InternVL3 等

## 关键图表

**Figure 1**：UI-Ins 的 scaling 性能对比图。随参数量增加（2B→32B），UI-Ins 在 MMBench-GUI L2 和 UI-I2E Bench 上均保持对其他方法的显著优势，且 7B 模型就超过了某些 72B 模型。

**Figure 2**：(a) 四种指令视角下的模型性能对比，Combined（最优选择）远超任何单一视角；(b) 原始数据的质量问题分布：23.3% 有缺陷；(c) 清洗数据训练后的模型在多个 benchmark 上一致优于原始数据训练的模型。

**Figure 6**：Instruction-as-Reasoning 框架全景图。左侧是 SFT 阶段（给定视角→推理→坐标），右侧是 RL 阶段（自由探索→GRPO 优化）。

**Figure 8**：(a) 模型在推理时能组合多个视角（1-6 个）进行分析；(b) RL 后模型探索出了训练时未见过的全新视角（如 Structural Relationship, Component Type, State 等）。

**Figure 9**：展示了 UI-Ins 的三种涌现推理能力：(1) 单视角推理、(2) 多视角组合推理、(3) 组合+涌现视角推理（包含了训练中未见的 Group Affiliation、State、Prediction 等视角）。

## 局限性

1. **缺乏领域知识**：模型无法将抽象描述（如"以积木玩具闻名的公司"）与具体品牌实体（如"MEGA"）关联，需要外部知识库支持
2. **界面布局理解不足**：在复杂布局中，模型有时无法准确判断哪个区域是可点击的操作目标
3. **视觉歧义和幻觉**：当界面中存在视觉相似的图标时，模型容易混淆并选错目标

## 个人思考

这篇论文最值得学习的是"把问题本身拆解清楚"的方法论。他们没有急着提一个新模型，而是先做了两个扎实的预分析——"指令多样性到底重不重要？"和"现有数据质量到底可不可靠？"——这两个问题的答案直接推导出整个方法的设计逻辑。

Instruction-as-Reasoning 这个概念本身也很优雅。它不是简单地做数据增强（"多生成几种指令让模型更鲁棒"），而是把"多样性"提升到"推理路径"的层面——不同视角不仅是不同的输入，更是不同的认知策略。这使得 RL 阶段可以真正学到"策略选择"而不仅仅是"模式匹配"。

关于 Free-Form Reasoning 在 grounding 中失败的发现也很有价值。它说明了一个更普遍的道理：**在视觉 grounding 这种精确空间定位任务中，无约束的推理不如结构化的推理有效**。Grounding 需要的是对视觉证据的精确描述，而不是开放式联想。

不过，该方法依赖于 GPT-4.1 做数据增强，训练成本较高（283k SFT + 100k RL 样本），且四种视角的定义是人工设计的。如果有更好的自动发现视角的方法，或者能将视角类型拓展到更多维度，可能会有更大提升。

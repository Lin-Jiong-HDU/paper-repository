# PKB - Paper Knowledge Base

基于 [Claude Code](https://docs.anthropic.com/en/docs/claude-code) 的本地论文知识库。输入 arXiv 链接，自动下载论文、生成结构化中文精读笔记和可视化展示页。

## 功能

- **一键添加论文** — 从 arXiv 下载 PDF，自动生成笔记和展示页（目前仅支持 arXiv）
- **结构化精读笔记** — 包含背景知识、核心思想、方法拆解、公式解读、实验分析等模块
- **可视化展示页** — 可折叠的卡片式 HTML 页面，支持浏览器直接打开和打印
- **个人思考** — 每篇论文附带个人思考空间，记录阅读心得

## 快速开始

### 前置条件

- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) 已安装并登录

### 使用方法

在 Claude Code 中直接使用 `/pkb` 命令：

```
# 添加一篇新论文（下载 + 生成笔记 + 生成展示页）
/pkb add https://arxiv.org/abs/1706.03762

# 重新生成某篇论文的笔记
/pkb notes 1706.03762

# 重新生成某篇论文的展示页
/pkb html 1706.03762

# 列出所有论文
/pkb list

# 在浏览器中打开展示页
/pkb open 1706.03762
```

## 目录结构

```
.
├── papers/                  # 论文存储（已 gitignore）
│   └── {arxiv-id}/
│       ├── paper.pdf        # 原文
│       ├── notes.md         # 结构化笔记
│       └── detail.html      # 可视化展示页
├── templates/               # 模板
│   ├── notes-template.md    # 笔记模板
│   └── detail-template.html # 展示页模板
├── .claude/skills/pkb/      # Skill 定义
│   └── SKILL.md
├── CLAUDE.md                # Claude Code 项目配置
└── README.md
```

## 笔记内容

每篇论文的笔记包含以下模块：

| 模块 | 说明 |
|------|------|
| 一句话总结 | 核心贡献的简洁概括 |
| 背景与前置知识 | 理解论文所需的基础概念 |
| 核心思想详解 | 用类比和直觉解释论文核心思路 |
| 方法逐步拆解 | 编号步骤，每步说明做什么、为什么 |
| 关键公式/算法解读 | 用自然语言翻译数学符号和步骤 |
| 实验设计分析 | 数据集选择、指标含义、结果解读 |
| 关键图表 | 核心图表描述 |
| 局限性 | 论文不足之处 |
| 个人思考 | 阅读心得和延伸思考 |

## 自定义

笔记和展示页的格式由 `templates/` 目录下的模板控制，可根据需要修改：

- `templates/notes-template.md` — 修改笔记结构和字段
- `templates/detail-template.html` — 修改展示页样式和布局

## License

MIT

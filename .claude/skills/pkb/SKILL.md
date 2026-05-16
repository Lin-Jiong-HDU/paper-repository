---
description: 本地论文知识库管理。从 arXiv 下载论文，生成结构化笔记（MD）和可视化展示页（HTML）。使用 /pkb add <url> 添加论文，/pkb notes <id> 重生成笔记，/pkb html <id> 重生成展示页，/pkb list 列出论文，/pkb open <id> 打开展示页。
---

# PKB - Paper Knowledge Base

本地论文知识库管理工具。从 arXiv 下载论文并自动生成结构化笔记和可视化展示页。

## 命令

### `/pkb add <arxiv-url>` - 全流程添加论文

1. **提取 arXiv ID**
   - 从 URL 中提取 ID（如 `https://arxiv.org/abs/1706.03762` → `1706.03762`）
   - 也支持 `https://arxiv.org/pdf/1706.03762` 格式
   - 去除版本号（如 `v1`、`v2`），保留纯 ID

2. **创建论文目录**

   ```bash
   mkdir -p papers/{arxiv_id}
   ```

3. **下载 PDF**

   ```bash
   wget -O papers/{arxiv_id}/paper.pdf https://arxiv.org/pdf/{arxiv_id}
   ```

4. **读取并分析论文**
   - 使用 Read 工具，或者相关的 PDF 阅读 Skill 读取 `papers/{arxiv_id}/paper.pdf`
   - 深入理解论文内容

5. **生成笔记**
   - 读取 `templates/notes-template.md` 作为模板
   - 根据论文内容填充所有占位符
   - 写入 `papers/{arxiv_id}/notes.md`
   - 笔记要求：
     - **背景与前置知识**：列出理解该论文需要的基础概念，用通俗语言解释
     - **核心思想详解**：用类比和直觉解释，不要直接复制摘要
     - **方法逐步拆解**：用编号步骤，每步说明做什么、为什么这么做
     - **关键公式/算法解读**：用自然语言翻译每个数学符号和步骤
     - **实验设计分析**：解释为什么选这些数据集/指标，结果说明什么
     - 所有内容用中文撰写

6. **生成 HTML 展示页**
   - 读取 `templates/detail-template.html` 作为模板
   - 将笔记中的内容填充到 HTML 模板中：
     - `{title}` → 论文标题
     - `{authors}` → 作者列表
     - `{date}` → 发表日期
     - `{arxiv_id}` → arXiv ID
     - `{tags_html}` → 关键词转为 `<span class="tag">关键词</span>` 格式
     - `{one_sentence_summary}` → 一句话总结
     - `{background_and_prerequisites}` → 背景与前置知识（转为 HTML 段落）
     - `{core_idea_explained}` → 核心思想详解（转为 HTML 段落）
     - `{method_step_by_step}` → 方法拆解（转为 HTML 段落）
     - `{key_formulas_and_algorithms}` → 公式解读（转为 HTML 段落）
     - `{experimental_design_analysis}` → 实验分析（转为 HTML 段落）
     - `{key_figures}` → 关键图表描述（转为 HTML 段落）
     - `{limitations}` → 局限性（转为 HTML 段落）
     - `{field}` → 领域
     - `{datasets}` → 数据集
     - `{metrics}` → 主要指标
     - `{key_results}` → 关键结果
     - `{baselines}` → 对比方法
     - `{personal_thoughts}` → 个人思考
   - MD 内容转 HTML 规则：`**bold**` → `<strong>`，段落间加 `<p>`，列表用 `<ul><li>`
   - 写入 `papers/{arxiv_id}/detail.html`

### `/pkb notes <arxiv-id>` - 重新生成笔记

1. 确认 `papers/{arxiv_id}/paper.pdf` 存在，不存在则报错
2. 读取 PDF
3. 读取模板 `templates/notes-template.md`
4. 填充并写入 `papers/{arxiv_id}/notes.md`

### `/pkb html <arxiv-id>` - 重新生成展示页

1. 确认 `papers/{arxiv_id}/notes.md` 存在，不存在则报错
2. 读取 notes.md 获取内容
3. 读取模板 `templates/detail-template.html`
4. 填充并写入 `papers/{arxiv_id}/detail.html`

### `/pkb list` - 列出所有论文

1. 列出 `papers/` 下的所有子目录
2. 对每个论文目录，读取 `notes.md` 的第一行（标题）和基本信息中的关键词
3. 以表格形式展示：arXiv ID | 标题 | 关键词

### `/pkb open <arxiv-id>` - 在浏览器中打开展示页

```bash
xdg-open papers/{arxiv_id}/detail.html
```

## 注意事项

- arXiv ID 提取要兼容多种 URL 格式
- 如果 `papers/{id}/` 目录已存在，提示用户是否覆盖
- 生成笔记时确保内容充实，不要用占位符糊弄
- HTML 必须是自包含的单文件，无外部依赖
- 所有笔记内容使用中文

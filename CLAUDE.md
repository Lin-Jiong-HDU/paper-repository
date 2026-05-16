# Paper Knowledge Base (PKB)

本地论文知识库，通过 `/pkb` skill 管理论文。

## 目录结构

- `papers/{arxiv-id}/` - 每篇论文独立文件夹
  - `paper.pdf` - 原文
  - `notes.md` - 结构化笔记
  - `detail.html` - 可视化展示页
- `templates/` - 笔记和 HTML 模板
- `skills/pkb.md` - skill 定义

## 使用方式

- `/pkb add <arxiv-url>` - 下载论文并生成笔记和展示页
- `/pkb notes <arxiv-id>` - 重新生成笔记
- `/pkb html <arxiv-id>` - 重新生成展示页
- `/pkb list` - 列出所有论文
- `/pkb open <arxiv-id>` - 在浏览器中打开展示页

# VideoToText

将视频课程自动转换为 NotebookLM 风格的结构化笔记。

```
视频 (.mp4)  →  音频转录 (SRT)  →  AI 笔记 (Markdown)  →  Obsidian 知识库
```

---

## 快速开始

### 1. 环境配置

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 GROQ_API_KEY
```

### 2. 添加视频并处理

```bash
# 推荐：添加单个视频并自动触发完整流水线
bash scripts/add_video.sh /path/to/00_03_第三课：示例课程丙_01_第一节内容.mp4

# 或将视频放入 data/input/ 后统一运行
bash scripts/run_pipeline.sh
```

> **命名规范**：`{课程编号}_{课程名}_{段编号}_{段标题}.mp4`
> 同课程前缀的多个视频段自动合并为一份笔记。

### 3. 查看笔记

```
notes/
├── 00_01_第一课：示例课程甲.md    ← 4 个视频段合并
├── 00_02_第二课：示例课程乙.md    ← 3 个视频段合并
├── 00_03_第三课：示例课程丙.md    ← 2 个视频段合并
└── INDEX.md                       ← 自动维护的索引
```

---

## 项目结构

```
VideoToText/
├── data/
│   ├── input/               # 📥 新视频放这里（唯一入口）
│   ├── output/              # 🔄 提取产物（音频、关键帧、转录）
│   │   └── {视频名}/
│   │       ├── audio.mp3
│   │       ├── frame_*.jpg
│   │       └── transcript/
│   │           ├── audio.srt
│   │           └── transcript.md
│   └── srt_exports/         # 📦 SRT 按课程聚合（供 Cursor @引用）
│       └── {课程名}/
│           └── {段落名}.srt
│
├── notes/                   # 📚 最终笔记（Obsidian 知识库）
│   ├── 00_XX_课程名.md
│   ├── INDEX.md             # ← 自动维护
│   └── 000_我的视频知识大盘.md  # ← Dataview 动态看板
│
├── src/
│   ├── extract_media.py     # 提取音频和关键帧
│   ├── transcribe_audio.py  # Groq Whisper 增量转录 + SRT 聚合
│   └── generate_note.py     # Groq LLM 笔记生成
│
├── scripts/
│   ├── add_video.sh         # ⭐ 标准入口：添加视频并触发流水线
│   └── run_pipeline.sh      # 一键执行全流程
│
├── prompts/
│   └── notebooklm_prompt.md # 笔记生成提示词文档（与代码同步）
│
├── .env                     # API 密钥（不提交到 git）
├── .env.example             # 密钥配置模板
└── requirements.txt
```

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **增量处理** | 已处理的步骤自动跳过，重新运行仅处理新增内容 |
| **自动分片** | 音频 >18MB 自动按 5 分钟分片，解决 API 大小限制 |
| **课程合并** | 同前缀多视频段自动合并为一份笔记 |
| **SRT 聚合** | 转录完成后自动按课程整理到 `data/srt_exports/`，便于批量引用 |
| **结构化笔记** | 每份笔记含 Mind Map（markmap）、数据表格、分段详情 |
| **INDEX 自动维护** | 每次生成后自动更新 `notes/INDEX.md` |
| **幻觉防护** | Prompt + Python 双层清洗，过滤 LLM 编造内容 |

---

## 常用命令

```bash
# 完整流水线
bash scripts/run_pipeline.sh

# 单步执行
.venv/bin/python src/extract_media.py                                        # 仅提取
.venv/bin/python src/transcribe_audio.py                                     # 仅转录
.venv/bin/python src/generate_note.py --all                                  # 仅生成笔记
.venv/bin/python src/generate_note.py --prefix "00_03_第三课：示例课程丙"   # 指定课程
.venv/bin/python src/generate_note.py --update-index                         # 仅重建索引

# 强制重处理
bash scripts/run_pipeline.sh --force-note        # 强制重新生成所有笔记
TRANSCRIBE_FORCE=true .venv/bin/python src/transcribe_audio.py  # 强制重新转录
```

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GROQ_API_KEY` | 必需 | Groq API 密钥 |
| `NOTE_MODEL` | `llama-3.3-70b-versatile` | 笔记生成模型 |
| `NOTE_MAX_TOKENS` | `8192` | 笔记最大输出长度 |
| `TRANSCRIBE_FORCE` | `false` | 强制重新转录所有 |
| `TRANSCRIBE_CHUNK_SECONDS` | `300` | 音频分片时长（秒） |

---

## 技术栈

- **转录**：[Groq API](https://console.groq.com) + `whisper-large-v3`
- **笔记生成**：Groq API + `llama-3.3-70b-versatile`
- **媒体处理**：`ffmpeg`（音频提取）、`opencv-python`（关键帧提取）
- **笔记格式**：Obsidian Markdown，支持 Mindmap NextGen、Dataview 插件

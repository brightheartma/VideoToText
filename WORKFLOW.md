# VideoToText 完整工作流指南

从"有新视频"到"Obsidian 笔记"的完整说明。

---

## 第一步：准备视频文件

### 命名规范

视频文件名决定最终笔记的名称和结构：

```
{课程编号}_{课程名}_{段编号}_{段标题}.mp4
```

**示例：**

```
✅ 00_03_第三课：示例课程丙_01_第一节内容.mp4
✅ 00_03_第三课：示例课程丙_02_第二节内容.mp4
   → 生成笔记：notes/00_03_第三课：示例课程丙.md

❌ lesson03.mp4                  （太简洁，无法识别）
❌ 视频_final_final2_v3.mp4      （无课程编号）
```

> **规则**：相同的课程前缀（`00_03_第三课：示例课程丙`）的所有视频段，会自动合并为一份课程笔记。

---

## 第二步：添加视频

三种方式，任选其一：

```bash
# 方式 A（推荐）：自动复制到 data/input/ 并触发完整流水线
bash scripts/add_video.sh /path/to/00_03_第三课：示例课程丙_01_第一节内容.mp4

# 方式 B：仅复制，延后手动处理
bash scripts/add_video.sh --no-run /path/to/video.mp4

# 方式 C：手动拖入文件夹，稍后运行
cp /path/to/video.mp4 data/input/
```

---

## 第三步：执行流水线

如果在第二步选了**方式 A**，可跳过（已自动执行）。否则手动运行：

```bash
bash scripts/run_pipeline.sh
```

流水线依次执行以下 4 个步骤：

### Step 1 — 提取媒体

```bash
.venv/bin/python src/extract_media.py
```

- 输入：`data/input/` 中的所有 `.mp4` 视频
- 输出：
  - `data/output/{视频名}/audio.mp3`（提取的音频）
  - `data/output/{视频名}/frame_XXXX_HH-MM-SS.jpg`（关键帧，每 2 秒一张）
- **增量**：已提取的视频自动跳过

### Step 2 — 转录音频

```bash
.venv/bin/python src/transcribe_audio.py
```

- 调用 **Groq API**（`whisper-large-v3` 模型）
- 音频 >18MB 自动按 5 分钟分片上传，解决 API 大小限制
- **增量**：已转录的自动跳过（检测 `audio.srt` 是否存在）
- 输出：
  - `data/output/{视频名}/transcript/audio.srt`（SRT 格式，含时间戳）
  - `data/output/{视频名}/transcript/transcript.md`（Markdown 格式）
  - `data/srt_exports/{课程名}/{段落名}.srt`（聚合 SRT，供 Cursor 批量引用）

### Step 3 — 生成笔记

```bash
.venv/bin/python src/generate_note.py --all
```

- 调用 **Groq LLM**（`llama-3.3-70b-versatile`）
- 自动合并同课程前缀的所有视频段
- 生成结构化笔记，包含：
  - **Mind Map**（markmap 格式，Obsidian Mindmap NextGen 插件可渲染）
  - **数据表格**（各段关键时间点、术语、结论）
  - **分段详情**（核心大纲 + 关键术语 + 详细解析）
- 输出：
  - `notes/{课程名}.md`
  - `notes/INDEX.md`（自动更新）

### Step 4 — 清理临时文件

```bash
rm -rf data/output/*/transcript/_chunks_work/
```

清理音频分片的工作目录（自动执行，无需手动）。

---

## 第四步：查看结果

| 文件 | 说明 |
|---|---|
| `notes/{课程名}.md` | 课程笔记，在 Obsidian 中打开 |
| `notes/INDEX.md` | 全局索引，含所有课程统计 |
| `notes/000_我的视频知识大盘.md` | Dataview 动态看板 |
| `data/srt_exports/{课程名}/` | 原始转录文件，供 Cursor 引用 |

---

## 附录 A：手动执行单个步骤

```bash
# 仅提取媒体（不转录）
.venv/bin/python src/extract_media.py

# 仅转录（增量，已完成跳过）
.venv/bin/python src/transcribe_audio.py

# 强制重新转录所有
TRANSCRIBE_FORCE=true .venv/bin/python src/transcribe_audio.py

# 生成所有课程笔记（已有笔记跳过）
.venv/bin/python src/generate_note.py --all

# 强制重新生成所有笔记
.venv/bin/python src/generate_note.py --all --force

# 为指定课程生成笔记
.venv/bin/python src/generate_note.py --prefix "00_03_第三课：示例课程丙"

# 仅重建 INDEX.md（不生成笔记）
.venv/bin/python src/generate_note.py --update-index
```

---

## 附录 B：流水线参数

```bash
bash scripts/run_pipeline.sh                    # 标准执行（增量）
bash scripts/run_pipeline.sh --force-note       # 强制重新生成所有笔记
bash scripts/run_pipeline.sh --force-transcribe # 强制重新转录
bash scripts/run_pipeline.sh --force-all        # 全部强制重新处理
bash scripts/run_pipeline.sh --skip-extract     # 跳过提取（仅转录+笔记）
bash scripts/run_pipeline.sh --skip-transcribe  # 跳过转录（仅提取+笔记）
bash scripts/run_pipeline.sh --note-only        # 仅生成笔记
```

---

## 附录 C：环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GROQ_API_KEY` | 必需 | Groq API 密钥，填入 `.env` 文件 |
| `NOTE_MODEL` | `llama-3.3-70b-versatile` | 笔记生成模型 |
| `NOTE_MAX_TOKENS` | `8192` | 笔记最大输出长度（token 数） |
| `TRANSCRIBE_FORCE` | `false` | 设为 `true` 可强制重新转录所有音频 |
| `TRANSCRIBE_CHUNK_SECONDS` | `300` | 音频分片时长（秒），默认 5 分钟 |

---

## 附录 D：关键特性说明

### 增量处理

每个步骤都有完成标志检测，重新运行流水线只处理新增内容：

- **提取**：检测 `audio.mp3` 是否存在
- **转录**：检测 `audio.srt` 是否存在
- **笔记**：检测 `notes/{课程名}.md` 是否存在

使用 `--force` 参数或 `FORCE=true` 环境变量可覆盖。

### 自动分片（超大文件处理）

Groq API 上传限制约 25MB。`transcribe_audio.py` 在文件 >18MB 时：

1. 自动将音频按 `TRANSCRIBE_CHUNK_SECONDS` 分片
2. 逐片上传并转录
3. 合并所有片段的 SRT，修正时间戳偏移
4. 清理工作目录 `_chunks_work/`

### SRT 聚合导出

`transcribe_audio.py` 转录完成后，自动将所有 `audio.srt` 按课程前缀整理到：

```
data/srt_exports/
└── 00_03_第三课：示例课程丙/
    ├── 01_第一节：课程片段丙一.srt
    └── 02_第二节：课程片段丙二.srt
```

在 Cursor 中可一次性引用整课转录：

```
@data/srt_exports/00_03_第三课：示例课程丙/
```

### 结构化笔记与幻觉防护

笔记生成采用两层防护：

1. **Prompt 层**：明确禁止编造代码、复读指令、添加转录中不存在的内容
2. **Python 清洗层**：`_clean_mindmap`、`_clean_datatable`、`_clean_detail` 函数过滤残留幻觉

---

## 附录 E：完整项目结构

```
VideoToText/
├── data/
│   ├── input/               # 📥 新视频放这里
│   ├── output/              # 🔄 提取产物
│   │   └── {视频名}/
│   │       ├── audio.mp3
│   │       ├── frame_*.jpg
│   │       └── transcript/
│   │           ├── audio.srt
│   │           └── transcript.md
│   └── srt_exports/         # 📦 SRT 按课程聚合
│       └── {课程名}/
│           └── {段落名}.srt
│
├── notes/                   # 📚 最终笔记
│   ├── 00_XX_课程名.md
│   ├── INDEX.md
│   └── 000_我的视频知识大盘.md
│
├── src/
│   ├── extract_media.py
│   ├── transcribe_audio.py
│   └── generate_note.py
│
├── scripts/
│   ├── add_video.sh         # ⭐ 标准入口
│   └── run_pipeline.sh
│
├── prompts/
│   └── notebooklm_prompt.md
│
├── .env
├── .env.example
├── requirements.txt
├── README.md                # 项目简介
├── WORKFLOW.md              # ⭐ 本文件
├── PROJECT_MAP.txt          # 项目地图（目录速查）
└── QUICK_START.txt          # 快速入门卡
```

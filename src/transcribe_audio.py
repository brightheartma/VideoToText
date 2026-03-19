import os
import re
import shutil
import datetime
import subprocess
import time
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

# ================= 配置区域 =================
# 强烈建议将 API Key 写在 .env 文件中（不要写进代码/笔记/提交到仓库）
# 例如：export GROQ_API_KEY="gsk_***"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# ============================================

CHUNK_SECONDS = int(os.getenv("TRANSCRIBE_CHUNK_SECONDS", "300"))  # 5 minutes
# 经验阈值：大于该体积时，直接走分片，避免 413 / 超时
CHUNK_WHEN_OVER_BYTES = int(os.getenv("TRANSCRIBE_CHUNK_WHEN_OVER_BYTES", str(18 * 1024 * 1024)))
# 分片压缩参数：更小体积、更稳（语音足够）
ENC_AUDIO_RATE = os.getenv("TRANSCRIBE_AR", "16000")
ENC_AUDIO_CHANNELS = os.getenv("TRANSCRIBE_AC", "1")
ENC_AUDIO_BITRATE = os.getenv("TRANSCRIBE_AB", "32k")
# 增量转录配置
FORCE_RETRANSCRIBE = os.getenv("TRANSCRIBE_FORCE", "").lower() == "true"  # 强制重新转录所有文件

def format_time(seconds):
    """将秒数转换为 HH:MM:SS"""
    return str(datetime.timedelta(seconds=int(seconds)))

def format_srt_time(seconds: float) -> str:
    # SRT: HH:MM:SS,mmm
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    hh = ms // 3_600_000
    mm = (ms % 3_600_000) // 60_000
    ss = (ms % 60_000) // 1000
    mmm = ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"

def write_srt(segments: Iterable[dict], srt_path: str) -> None:
    lines = []
    idx = 1
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")  # blank line
        idx += 1
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def write_markdown_transcript(segments: Iterable[dict], md_path: str, title: str) -> None:
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 视频逐字稿（{title}）\n\n")
        for seg in segments:
            start = format_time(seg.get("start", 0))
            end = format_time(seg.get("end", 0))
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            f.write(f"**[{start} - {end}]** {text}\n\n")

def _ensure_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError("未检测到 ffmpeg。请先安装：brew install ffmpeg") from e

def _file_size_bytes(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0

def _run_ffmpeg_segment_encode(input_path: str, out_dir: str, chunk_seconds: int) -> List[Tuple[str, float]]:
    """
    把音频转码为更小 mp3 并按固定长度分片。
    返回 [(chunk_path, offset_seconds), ...]
    """
    _ensure_ffmpeg()
    os.makedirs(out_dir, exist_ok=True)

    # 清理旧分片
    for name in os.listdir(out_dir):
        if name.startswith("chunk_") and name.endswith(".mp3"):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass

    out_pattern = os.path.join(out_dir, "chunk_%03d.mp3")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-vn",
        "-ac",
        str(ENC_AUDIO_CHANNELS),
        "-ar",
        str(ENC_AUDIO_RATE),
        "-b:a",
        str(ENC_AUDIO_BITRATE),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-reset_timestamps",
        "1",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)

    chunks = sorted(
        [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.startswith("chunk_") and f.endswith(".mp3")]
    )
    results: List[Tuple[str, float]] = []
    for i, p in enumerate(chunks):
        results.append((p, float(i * chunk_seconds)))
    if not results:
        raise RuntimeError("音频分片失败：未生成任何 chunk_*.mp3")
    return results

def _is_request_too_large(err: Exception) -> bool:
    s = str(err)
    return ("413" in s) or ("Request Entity Too Large" in s) or ("request_too_large" in s)

def _is_timeout(err: Exception) -> bool:
    s = str(err).lower()
    return ("timed out" in s) or ("timeout" in s)

def transcribe_via_groq(
    client: Groq,
    audio_path: str,
    *,
    model: str = "whisper-large-v3",
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
) -> List[dict]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), f.read()),
                    model=model,
                    response_format="verbose_json",
                )
            segments = list(getattr(transcription, "segments", []) or [])
            if not segments:
                raise RuntimeError("API 返回的 segments 为空（response_format=verbose_json 但无分段数据）")
            return segments
        except Exception as e:
            last_err = e
            # 413 直接抛给上层做分片
            if _is_request_too_large(e):
                raise
            # 超时：重试
            if attempt < max_retries and _is_timeout(e):
                time.sleep(retry_backoff_s * attempt)
                continue
            raise
    raise RuntimeError("转录失败") from last_err

def transcribe_with_chunking(
    client: Groq,
    audio_path: str,
    work_dir: str,
    *,
    chunk_seconds: int,
    model: str = "whisper-large-v3",
) -> List[dict]:
    chunks = _run_ffmpeg_segment_encode(audio_path, work_dir, chunk_seconds)
    all_segments: List[dict] = []

    for i, (chunk_path, offset) in enumerate(chunks, start=1):
        print(f"    - 分片 {i}/{len(chunks)}: {os.path.basename(chunk_path)} (offset={int(offset)}s)", flush=True)
        segs = transcribe_via_groq(client, chunk_path, model=model, max_retries=3, retry_backoff_s=2.0)
        for s in segs:
            # 将分片时间轴拼回原时间轴
            s2 = dict(s)
            s2["start"] = float(s.get("start", 0.0)) + offset
            s2["end"] = float(s.get("end", 0.0)) + offset
            all_segments.append(s2)

    # 兜底排序
    all_segments.sort(key=lambda x: float(x.get("start", 0.0)))
    return all_segments

def _check_transcription_exists(srt_path: str, md_path: str) -> bool:
    """检查转录文件是否已存在（增量转录的关键逻辑）"""
    return os.path.exists(srt_path) and os.path.exists(md_path)

def _get_transcript_info(srt_path: str) -> Optional[dict]:
    """读取已有的 SRT 文件，返回统计信息"""
    if not os.path.exists(srt_path):
        return None
    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        line_count = len(lines)
        file_size_kb = os.path.getsize(srt_path) / 1024
        return {
            "lines": line_count,
            "size_kb": file_size_kb,
            "exists": True
        }
    except Exception:
        return None

def batch_transcribe_with_api(base_dir):
    """使用 Groq API 批量极速转录音频（支持增量转录）"""
    output_dir = os.path.join(base_dir, "data", "output")
    if not os.path.exists(output_dir):
        print(f"错误：找不到目录 {output_dir}")
        return

    # 初始化 Groq 客户端（给足超时，减少大文件偶发失败）
    client = Groq(api_key=GROQ_API_KEY, timeout=600.0)
    
    # 统计信息
    stats = {
        "total_folders": 0,
        "skipped": 0,
        "transcribed": 0,
        "failed": 0,
    }
    
    folders = sorted(os.listdir(output_dir))
    print(f"\n📊 开始扫描转录任务（共 {len(folders)} 个文件夹）...", flush=True)
    print("=" * 70, flush=True)
    
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        stats["total_folders"] += 1
        audio_path = os.path.join(folder_path, "audio.mp3")
        transcript_dir = os.path.join(folder_path, "transcript")
        os.makedirs(transcript_dir, exist_ok=True)
        srt_path = os.path.join(transcript_dir, "audio.srt")
        md_path = os.path.join(transcript_dir, "transcript.md")
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"⏭️  跳过: {folder}（未找到 audio.mp3）", flush=True)
            stats["skipped"] += 1
            continue
        
        # 增量转录：检查转录文件是否已存在
        transcript_exists = _check_transcription_exists(srt_path, md_path)
        
        if transcript_exists and not FORCE_RETRANSCRIBE:
            transcript_info = _get_transcript_info(srt_path)
            print(f"✅ 已转录: {folder}", flush=True)
            if transcript_info:
                print(f"   📄 SRT: {transcript_info['lines']} 行, {transcript_info['size_kb']:.1f}KB", flush=True)
            stats["skipped"] += 1
            continue
        
        if transcript_exists and FORCE_RETRANSCRIBE:
            print(f"\n🔄 强制重新转录: {folder}（--force 标志已启用）", flush=True)
        else:
            print(f"\n🚀 正在通过 Groq API 极速转录: {folder} ...", flush=True)
        
        try:
            size_b = _file_size_bytes(audio_path)
            need_chunk = size_b >= CHUNK_WHEN_OVER_BYTES
            work_dir = os.path.join(transcript_dir, "_chunks_work")

            if need_chunk:
                print(
                    f"  -> 文件较大（{size_b/1024/1024:.1f}MB），启用分片：{CHUNK_SECONDS}s/片，转码 {ENC_AUDIO_CHANNELS}ch {ENC_AUDIO_RATE}Hz {ENC_AUDIO_BITRATE}",
                    flush=True,
                )
                segments = transcribe_with_chunking(
                    client,
                    audio_path,
                    work_dir,
                    chunk_seconds=CHUNK_SECONDS,
                    model="whisper-large-v3",
                )
            else:
                try:
                    segments = transcribe_via_groq(client, audio_path, model="whisper-large-v3")
                except Exception as e:
                    # 413 / 超时等：自动降级到分片
                    if _is_request_too_large(e) or _is_timeout(e):
                        print("  -> 单次请求失败（413/超时），改为分片重试。", flush=True)
                        segments = transcribe_with_chunking(
                            client,
                            audio_path,
                            work_dir,
                            chunk_seconds=CHUNK_SECONDS,
                            model="whisper-large-v3",
                        )
                    else:
                        raise

            write_srt(segments, srt_path)
            write_markdown_transcript(segments, md_path, folder)

            print(f"  -> ✅ 成功: SRT 已保存至 {srt_path}", flush=True)
            print(f"  -> ✅ 成功: Markdown 已保存至 {md_path}", flush=True)
            stats["transcribed"] += 1
        except Exception as e:
            print(f"  -> ❌ 失败: API 请求出错 ({e})", flush=True)
            stats["failed"] += 1
    
    # 聚合 SRT 文件，按课程前缀分类到 data/srt_exports/
    srt_exports_dir = os.path.join(base_dir, "data", "srt_exports")
    os.makedirs(srt_exports_dir, exist_ok=True)

    print("\n📦 正在将分散的 SRT 文件按课程聚合导出...", flush=True)
    exported_count = 0
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        srt_path = os.path.join(folder_path, "transcript", "audio.srt")
        if not os.path.exists(srt_path):
            continue

        matches = list(re.finditer(r"_\d{2}_", folder))
        if matches:
            split_idx = matches[-1].start()
            prefix = folder[:split_idx]
            segment_name = folder[split_idx + 1:]
        else:
            prefix = "未分类课程"
            segment_name = folder

        target_course_dir = os.path.join(srt_exports_dir, prefix)
        os.makedirs(target_course_dir, exist_ok=True)
        target_srt_path = os.path.join(target_course_dir, f"{segment_name}.srt")
        shutil.copy2(srt_path, target_srt_path)
        exported_count += 1

    print(f"  -> ✅ 成功聚合 {exported_count} 个转录文件至: data/srt_exports/", flush=True)

    # 打印总结
    print("=" * 70, flush=True)
    print(f"\n📈 转录总结:", flush=True)
    print(f"  - 总文件夹数: {stats['total_folders']}", flush=True)
    print(f"  - ✅ 已转录: {stats['transcribed']}", flush=True)
    print(f"  - ⏭️  已跳过（已有转录）: {stats['skipped']}", flush=True)
    print(f"  - ❌ 转录失败: {stats['failed']}", flush=True)
    print(f"  - 运行模式: {'强制重新转录所有' if FORCE_RETRANSCRIBE else '增量转录（仅处理新文件）'}", flush=True)
    print("", flush=True)

def print_usage():
    """打印使用说明"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║           Groq API 增量转录脚本 - 使用说明                        ║
╚════════════════════════════════════════════════════════════════════╝

📌 基本用法：
   .venv/bin/python transcribe_audio.py

📌 增量转录（默认）：
   - 自动检查每个视频是否已有 audio.srt 和 transcript.md
   - 如果已存在，则跳过；否则进行转录
   - 已有的转录文件除非被手动删除，否则不会被重复处理

📌 强制重新转录所有文件：
   TRANSCRIBE_FORCE=true .venv/bin/python transcribe_audio.py

📌 调整转录参数：
   TRANSCRIBE_CHUNK_SECONDS=120 .venv/bin/python transcribe_audio.py
   TRANSCRIBE_CHUNK_WHEN_OVER_BYTES=$((15*1024*1024)) .venv/bin/python transcribe_audio.py

📌 环境变量：
   GROQ_API_KEY          - Groq API 密钥（必需，在 .env 中设置）
   TRANSCRIBE_FORCE      - 强制重新转录（true/false）
   TRANSCRIBE_CHUNK_SECONDS - 分片时长秒数（默认 300）
   TRANSCRIBE_CHUNK_WHEN_OVER_BYTES - 何时启用分片（字节数，默认 18MB）
   TRANSCRIBE_AR         - 音频采样率（默认 16000）
   TRANSCRIBE_AC         - 音频通道数（默认 1）
   TRANSCRIBE_AB         - 音频码率（默认 32k）

📌 文件位置：
   - 输入：data/output/<视频名>/audio.mp3
   - 输出：data/output/<视频名>/transcript/audio.srt
          data/output/<视频名>/transcript/transcript.md
          data/srt_exports/<课程前缀>/<段落名>.srt  （聚合副本，供 @引用）

""")

if __name__ == "__main__":
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    if not GROQ_API_KEY:
        print("❌ 错误：未检测到 GROQ_API_KEY。")
        print("✅ 解决方法：")
        print("   1. 创建或编辑 .env 文件")
        print("   2. 添加一行：GROQ_API_KEY=gsk_your_api_key_here")
        print("   3. 保存文件")
        print('   4. 或临时设置环境变量：export GROQ_API_KEY="gsk_***"')
        print("")
        print_usage()
        raise SystemExit(1)

    # 脚本在 src/ 下，项目根目录在上一层
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    batch_transcribe_with_api(BASE_DIR)
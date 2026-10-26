"""Batch-transcribe lecture recordings locally with Cohere Transcribe (MLX).

Usage:
    .venv/bin/python src/transcribe.py                    # everything in data/recordings
    .venv/bin/python src/transcribe.py lecture01.m4a      # specific files or folders
    .venv/bin/python src/transcribe.py --overwrite        # redo files already transcribed

The language is detected per file (English or Mandarin; Mandarin keeps English
terms as English) and Mandarin output is converted to Traditional Chinese.

For each recording it writes, into processed/transcripts:
    <name>.txt  transcript in paragraphs with [hh:mm:ss] markers (paste this into Claude)
    <name>.srt  subtitles, one ~30 s segment per cue
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import opencc
import soundfile as sf

from mlx_speech._hub import get_model_path
from mlx_speech.generation import CohereAsrModel
from mlx_speech.models.cohere_asr.feature_extraction import split_audio_chunks

SR = 16000
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".aiff", ".aif", ".caf", ".flac", ".aac", ".mp4", ".mov", ".ogg", ".webm"}
ROOT = Path(__file__).resolve().parent.parent  # project root (this file is in src/)
TO_TRADITIONAL = opencc.OpenCC("s2twp")  # Simplified -> Taiwan Traditional, incl. phrasing
# A phrase of 2-80 chars repeated 3+ times back to back: greedy decoding looping on
# silence or noise ("The color model is the color model. The color model is ...").
REPEAT_RE = re.compile(r"(.{2,80}?)(?:[\s,，.。]*\1){2,}", re.S)


def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Decode any audio file to a 16 kHz mono float32 waveform.

    Uses macOS's built-in afconvert, falling back to ffmpeg for formats it can't read.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        cmds = [["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(path), str(wav)]]
        if shutil.which("ffmpeg"):
            cmds.append(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                         "-ac", "1", "-ar", str(SR), str(wav)])
        for cmd in cmds:
            if subprocess.run(cmd, capture_output=True).returncode == 0 and wav.exists():
                audio, sr = sf.read(wav, dtype="float32", always_2d=False)
                assert sr == SR, sr
                return audio
    hint = "" if shutil.which("ffmpeg") else " (try `brew install ffmpeg`)"
    raise RuntimeError(f"could not decode {path.name}{hint}")


def fmt_ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def cjk_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return sum("一" <= c <= "鿿" for c in chars) / max(1, len(chars))


def detect_language(model: CohereAsrModel, audio: np.ndarray) -> str:
    """Cohere Transcribe has no language ID, so decode a few 30 s samples in
    Mandarin mode: English speech still comes out in English, Mandarin in hanzi."""
    votes = 0
    for frac in (0.2, 0.4, 0.6, 0.8):
        s = int(len(audio) * frac)
        text = REPEAT_RE.sub(r"\1", model.transcribe(audio[s:s + 30 * SR], sample_rate=SR, language="zh").text)
        votes += cjk_ratio(text) > 0.3
    return "zh" if votes >= 2 else "en"


def clean(text: str, language: str) -> str:
    text = REPEAT_RE.sub(r"\1", text.strip())
    return TO_TRADITIONAL.convert(text) if language == "zh" else text


def transcribe_file(model: CohereAsrModel, path: Path, language: str) -> tuple[list[tuple[float, float, str]], str]:
    audio = load_audio_16k_mono(path)
    duration = len(audio) / SR
    if language == "auto":
        language = detect_language(model, audio)
    print(f"  language: {language}")
    # Same energy-based splitting the model uses internally (<=35 s, cut at the
    # quietest point), done here so each segment keeps its timestamp.
    chunks = split_audio_chunks(audio, sr=SR, max_clip_s=35.0, overlap_s=5.0)

    segments: list[tuple[float, float, str]] = []
    offset = 0
    started = time.time()
    for i, chunk in enumerate(chunks, 1):
        start, end = offset / SR, (offset + len(chunk)) / SR
        offset += len(chunk)
        text = clean(model.transcribe(chunk, sample_rate=SR, language=language).text, language)
        if text:
            segments.append((start, end, text))
        elapsed = time.time() - started
        eta = elapsed / end * (duration - end) if end else 0
        print(f"\r  {fmt_ts(end)} / {fmt_ts(duration)}  ({i}/{len(chunks)})  "
              f"elapsed {elapsed:.0f}s, ~{eta:.0f}s left", end="", flush=True)
    elapsed = time.time() - started
    print(f"\r  done: {fmt_ts(duration)} of audio in {elapsed:.0f}s "
          f"({duration / max(elapsed, 1e-6):.0f}x real time)" + " " * 20)
    return segments, language


def write_outputs(segments: list[tuple[float, float, str]], stem: Path, language: str,
                  paragraph_s: float = 120.0) -> None:
    sep = "" if language == "zh" else " "
    # .txt: group segments into ~2-minute paragraphs, each led by its start time.
    paragraphs: list[str] = []
    current: list[str] = []
    para_start = 0.0
    for start, _end, text in segments:
        if current and start - para_start >= paragraph_s:
            paragraphs.append(f"[{fmt_ts(para_start)}] " + sep.join(current))
            current = []
        if not current:
            para_start = start
        current.append(text)
    if current:
        paragraphs.append(f"[{fmt_ts(para_start)}] " + sep.join(current))
    Path(f"{stem}.txt").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")

    cues = [f"{n}\n{fmt_ts(s, True)} --> {fmt_ts(e, True)}\n{t}\n"
            for n, (s, e, t) in enumerate(segments, 1)]
    Path(f"{stem}.srt").write_text("\n".join(cues), encoding="utf-8")


def collect_inputs(paths: list[Path]) -> list[tuple[Path, Path]]:
    """Return (audio file, output stem relative to --out); subfolders such as
    data/recordings/<course>/ are mirrored as processed/transcripts/<course>/."""
    files: list[tuple[Path, Path]] = []
    for p in paths:
        if p.is_dir():
            files += [(f, f.relative_to(p).with_suffix(""))
                      for f in sorted(p.rglob("*")) if f.suffix.lower() in AUDIO_EXTS]
        elif p.is_file():
            files.append((p, Path(p.stem)))
        else:
            print(f"skip: {p} not found", file=sys.stderr)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path, default=[ROOT / "data" / "recordings"],
                    help="audio files or folders (default: data/recordings)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "processed" / "transcripts", help="output folder")
    ap.add_argument("--language", default="auto", help="auto (English or Mandarin, per file), or an ISO 639-1 code such as en, zh")
    ap.add_argument("--model", default="cohere-asr", help="mlx-speech alias, HF repo, or local path")
    ap.add_argument("--overwrite", action="store_true", help="re-transcribe files that already have a .txt")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("No audio files found. Put recordings in data/recordings or pass paths.", file=sys.stderr)
        return 1
    todo = [(f, rel) for f, rel in files
            if args.overwrite or not Path(f"{args.out / rel}.txt").exists()]
    if len(todo) < len(files):
        print(f"Skipping {len(files) - len(todo)} already transcribed (use --overwrite to redo).")
    if not todo:
        return 0

    print("Loading model...")
    model = CohereAsrModel.from_dir(get_model_path(args.model))
    # MLX keeps freed GPU buffers cached for reuse; across a long batch that
    # cache grew past 20 GB, so keep it small.
    mx.set_cache_limit(512 * 1024**2)

    failed = 0
    for n, (f, rel) in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {rel}")
        try:
            segments, language = transcribe_file(model, f, args.language)
        except Exception as e:  # keep going with the rest of the batch
            print(f"\n  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        stem = args.out / rel
        stem.parent.mkdir(parents=True, exist_ok=True)
        write_outputs(segments, stem, language)
        print(f"  -> {stem}.txt / .srt")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

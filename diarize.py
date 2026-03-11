#!/usr/bin/env python3
"""
Speaker diarization using NVIDIA NeMo.

Features:
  - Auto-converts audio to 16kHz mono WAV via ffmpeg
  - Supports single file or batch (multiple files / directory)
  - Prints per-file metrics: segments, speakers, speech duration

Usage:
    python diarize.py audio.wav
    python diarize.py audio.wav --speakers 2
    python diarize.py a.wav b.wav c.wav
    python diarize.py ./recordings/           # all .wav files in directory
    python diarize.py audio.mp3 --output ./results
    python diarize.py audio.wav --no-msdd     # clustering-only, no MSDD
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

# NeMo writes RTTM output here: {out_dir}/pred_rttms/{stem}.rttm
RTTM_SUBDIR = "pred_rttms"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def needs_conversion(audio_path: Path) -> bool:
    """Return True if the file is not already 16kHz mono WAV."""
    import soundfile as sf
    if audio_path.suffix.lower() != ".wav":
        return True
    info = sf.info(str(audio_path))
    return info.samplerate != 16000 or info.channels != 1


def convert_audio(src: Path, dst: Path) -> None:
    """Convert audio to 16kHz mono WAV using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def get_audio_duration(audio_path: Path) -> float:
    """Return duration in seconds using soundfile."""
    import soundfile as sf
    info = sf.info(str(audio_path))
    return info.duration


# ---------------------------------------------------------------------------
# Manifest / NeMo interface
# ---------------------------------------------------------------------------

def build_manifest(audio_path: Path, num_speakers: Optional[int], manifest_path: str) -> None:
    entry = {
        "audio_filepath": str(audio_path.resolve()),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": num_speakers,
        "rttm_filepath": None,
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(entry) + "\n")


def load_nemo_model(cfg, no_msdd: bool):
    try:
        from nemo.collections.asr.models import ClusteringDiarizer, NeuralDiarizer
    except ImportError:
        print("Error: NeMo not installed. Run: pip install nemo_toolkit[asr]", file=sys.stderr)
        sys.exit(1)
    return ClusteringDiarizer(cfg=cfg) if no_msdd else NeuralDiarizer(cfg=cfg)


# ---------------------------------------------------------------------------
# RTTM parsing and metrics
# ---------------------------------------------------------------------------

def parse_rttm(rttm_path: Path) -> list:
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            segments.append({
                "start": start,
                "duration": duration,
                "end": start + duration,
                "speaker": parts[7],
            })
    if not segments:
        print("Warning: no speaker segments found in RTTM output.", file=sys.stderr)
    return sorted(segments, key=lambda s: s["start"])


def compute_metrics(segments: list, audio_duration: float) -> dict:
    unique_speakers = len(set(s["speaker"] for s in segments))
    total_speech = sum(s["duration"] for s in segments)
    coverage = (total_speech / audio_duration * 100) if audio_duration > 0 else 0.0
    return {
        "segments": len(segments),
        "speakers": unique_speakers,
        "speech_sec": round(total_speech, 2),
        "coverage_pct": round(coverage, 1),
    }


def print_results(segments: list) -> None:
    print("\n--- Diarization Results ---")
    for seg in segments:
        m_s, s_s = divmod(seg["start"], 60)
        m_e, s_e = divmod(seg["end"], 60)
        print(f"[{int(m_s):02d}:{s_s:05.2f} --> {int(m_e):02d}:{s_e:05.2f}]  {seg['speaker']}")


def print_metrics(metrics: dict) -> None:
    print("\n--- Metrics ---")
    print(f"  Speakers : {metrics['speakers']}")
    print(f"  Segments : {metrics['segments']}")
    print(f"  Speech   : {metrics['speech_sec']}s  ({metrics['coverage_pct']}% of audio)")


# ---------------------------------------------------------------------------
# Core single-file processing
# ---------------------------------------------------------------------------

def process_single(
    audio_path: Path,
    num_speakers: Optional[int],
    config_path: Path,
    output_dir: Path,
    no_msdd: bool,
) -> Optional[dict]:
    """Run diarization on one file. Returns metrics dict or None on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert audio if needed
    if needs_conversion(audio_path):
        converted = output_dir / "input_converted.wav"
        print(f"  Converting {audio_path.name} → 16kHz mono WAV...")
        convert_audio(audio_path, converted)
        audio_path = converted

    duration = get_audio_duration(audio_path)

    cfg = OmegaConf.load(config_path)
    cfg.diarizer.out_dir = str(output_dir)
    if num_speakers is not None:
        cfg.diarizer.clustering.parameters.oracle_num_speakers = True

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    try:
        build_manifest(audio_path, num_speakers, tmp.name)
        tmp.close()
        cfg.diarizer.manifest_filepath = tmp.name
        model = load_nemo_model(cfg, no_msdd)
        model.diarize()
    finally:
        os.unlink(tmp.name)

    # Stem is taken from the audio file NeMo actually saw
    rttm_path = output_dir / RTTM_SUBDIR / f"{audio_path.stem}.rttm"
    if not rttm_path.exists():
        print(f"  Warning: RTTM not found at {rttm_path}", file=sys.stderr)
        return None

    segments = parse_rttm(rttm_path)
    metrics = compute_metrics(segments, duration)
    print_results(segments)
    print_metrics(metrics)
    print(f"\n  RTTM saved to: {rttm_path}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker diarization with NeMo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python diarize.py audio.wav\n"
               "  python diarize.py a.wav b.wav --speakers 2\n"
               "  python diarize.py ./recordings/\n",
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="Audio file(s) or a directory of .wav files",
    )
    parser.add_argument("--speakers", type=int, default=None, help="Known number of speakers (optional)")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--no-msdd", action="store_true", help="Clustering-only diarizer (skip MSDD)")
    args = parser.parse_args()

    if args.speakers is not None and args.speakers < 1:
        parser.error("--speakers must be a positive integer")

    config_path = Path(args.config)
    if not config_path.exists():
        parser.error(f"config file not found: {config_path}")

    output_dir = Path(args.output)

    # Resolve audio paths (file list or directory)
    audio_inputs = [Path(p) for p in args.audio]
    if len(audio_inputs) == 1 and audio_inputs[0].is_dir():
        audio_paths = sorted(audio_inputs[0].glob("*.wav"))
        if not audio_paths:
            print(f"Error: no .wav files found in {audio_inputs[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_paths = audio_inputs

    missing = [p for p in audio_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: audio file not found: {p}", file=sys.stderr)
        sys.exit(1)

    batch = len(audio_paths) > 1
    summary = []

    for audio_path in audio_paths:
        if batch:
            file_out = output_dir / audio_path.stem
            print(f"\n=== {audio_path.name} ===")
        else:
            file_out = output_dir
            print(f"Audio:    {audio_path}")
            print(f"Speakers: {args.speakers or 'auto-detect'}")
            print(f"Output:   {file_out}")

        metrics = process_single(audio_path, args.speakers, config_path, file_out, args.no_msdd)
        if batch and metrics:
            summary.append((audio_path.name, metrics))

    if batch and summary:
        print("\n=== Batch Summary ===")
        print(f"{'File':<30} {'Speakers':>8} {'Segments':>9} {'Speech':>10} {'Coverage':>9}")
        print("-" * 70)
        for name, m in summary:
            print(f"{name:<30} {m['speakers']:>8} {m['segments']:>9} {m['speech_sec']:>9}s {m['coverage_pct']:>8}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Post-processing for NeMo diarization output.

Applies corrections in order:
  0. Speaker consolidation — auto-merge speaker clusters with similar TitaNet embeddings
  1. Speaker unification within defined time blocks (explicit merge lists)
  2. Gap filling — merge same-speaker segments separated by < gap seconds
  3. Short segment absorption — segments < min_seg absorbed by dominant neighbor or deleted

Input:  RTTM file  OR  JSON file produced by handler.py
Output: corrected JSON (stdout or --output file)

Usage:
  python postprocess.py audio.rttm
  python postprocess.py audio.rttm --blocks blocks.json --gap 1.5 --min-seg 0.4
  python postprocess.py results.json --output corrected.json
  python postprocess.py audio.rttm --blocks blocks.json --output corrected.json
  python postprocess.py audio.rttm --audio audio.mp3 --consolidate 0.85

blocks.json format:
  [
    {
      "start": "01:29:19",
      "end": "01:41:10",
      "merge": ["speaker_0", "speaker_2"]
    }
  ]
  Each block defines a time window and an explicit list of speaker IDs to merge.
  The FIRST ID in "merge" is kept; the rest are relabeled to it within the window.
  Other speakers in the window are NOT affected.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> float:
    """Parse 'HH:MM:SS', 'MM:SS', or a plain float string to seconds."""
    parts = t.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(f"Cannot parse time: {t!r}")


def _fmt(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Load input
# ---------------------------------------------------------------------------

def load_rttm(path: Path) -> list:
    segments = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            segments.append({
                "start": start,
                "duration": dur,
                "end": start + dur,
                "speaker": parts[7],
            })
    return sorted(segments, key=lambda s: s["start"])


def load_input(path: Path) -> list:
    """Load segments from RTTM or handler JSON."""
    if path.suffix.lower() == ".rttm":
        return load_rttm(path)
    data = json.loads(path.read_text())
    # handler JSON has {"segments": [...], ...}
    segs = data.get("segments", data) if isinstance(data, dict) else data
    return sorted(segs, key=lambda s: s["start"])


# ---------------------------------------------------------------------------
# 1. Speaker unification within time blocks
# ---------------------------------------------------------------------------

def unify_speakers_in_blocks(segments: list, blocks: list) -> list:
    """
    For each defined time block, merge explicitly listed speaker IDs into one.

    blocks: list of {"start": float_sec, "end": float_sec, "merge": ["speaker_0", "speaker_2"]}
    The first ID in "merge" is kept; the rest are relabeled to it within the window.
    Speakers NOT listed in "merge" are never touched.
    """
    if not blocks:
        return segments

    result = [dict(s) for s in segments]

    for block in blocks:
        b_start, b_end = block["start"], block["end"]
        merge_ids: list = block.get("merge", [])

        if len(merge_ids) < 2:
            print(
                f"  Block [{_fmt(b_start)} → {_fmt(b_end)}]: "
                f"'merge' needs at least 2 speaker IDs — skipped",
                file=sys.stderr,
            )
            continue

        keep = merge_ids[0]
        replace = set(merge_ids[1:])

        changed = 0
        for s in result:
            if s["start"] < b_end and s["end"] > b_start and s["speaker"] in replace:
                s["speaker"] = keep
                changed += 1

        print(
            f"  Block [{_fmt(b_start)} → {_fmt(b_end)}]: "
            f"merged {replace} → {keep} ({changed} segment(s))",
            file=sys.stderr,
        )

    return result


# ---------------------------------------------------------------------------
# 2. Gap filling — merge same-speaker segments within gap_threshold
# ---------------------------------------------------------------------------

def fill_gaps(segments: list, gap: float) -> list:
    """Merge consecutive same-speaker segments separated by <= gap seconds."""
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            seg["speaker"] == prev["speaker"]
            and (seg["start"] - prev["end"]) <= gap
        ):
            prev["end"] = seg["end"]
            prev["duration"] = prev["end"] - prev["start"]
        else:
            merged.append(dict(seg))
    return merged


# ---------------------------------------------------------------------------
# 3. Short segment absorption
# ---------------------------------------------------------------------------

def absorb_short_segments(segments: list, min_duration: float, gap: float) -> list:
    """
    Segments shorter than min_duration are either:
      - Absorbed by a neighboring segment of the SAME speaker within gap seconds
      - Absorbed by the LONGER of the two adjacent segments (if neighbors differ)
      - Deleted if no neighbors exist within gap

    Runs iteratively until no more short segments remain.
    """
    if min_duration <= 0:
        return segments

    changed = True
    segs = [dict(s) for s in segments]

    while changed:
        changed = False
        new_segs = []
        i = 0
        while i < len(segs):
            s = segs[i]
            if s["duration"] >= min_duration:
                new_segs.append(s)
                i += 1
                continue

            # Short segment — find neighbors
            prev_seg = new_segs[-1] if new_segs else None
            next_seg = segs[i + 1] if i + 1 < len(segs) else None

            prev_gap = (s["start"] - prev_seg["end"]) if prev_seg else float("inf")
            next_gap = (next_seg["start"] - s["end"]) if next_seg else float("inf")

            # Prefer merging with same-speaker neighbor
            absorbed = False
            if prev_seg and prev_seg["speaker"] == s["speaker"] and prev_gap <= gap:
                prev_seg["end"] = s["end"]
                prev_seg["duration"] = prev_seg["end"] - prev_seg["start"]
                absorbed = True
            elif next_seg and next_seg["speaker"] == s["speaker"] and next_gap <= gap:
                next_seg["start"] = s["start"]
                next_seg["duration"] = next_seg["end"] - next_seg["start"]
                absorbed = True
            elif prev_seg and prev_gap <= gap and next_seg and next_gap <= gap:
                # Both neighbors reachable — pick the longer one
                if prev_seg["duration"] >= (next_seg["duration"] if next_seg else 0):
                    prev_seg["end"] = s["end"]
                    prev_seg["duration"] = prev_seg["end"] - prev_seg["start"]
                else:
                    next_seg["start"] = s["start"]
                    next_seg["duration"] = next_seg["end"] - next_seg["start"]
                absorbed = True
            elif prev_seg and prev_gap <= gap:
                prev_seg["end"] = s["end"]
                prev_seg["duration"] = prev_seg["end"] - prev_seg["start"]
                absorbed = True
            elif next_seg and next_gap <= gap:
                next_seg["start"] = s["start"]
                next_seg["duration"] = next_seg["end"] - next_seg["start"]
                absorbed = True
            # else: isolated short segment — delete (don't append)

            if absorbed:
                changed = True
            i += 1

        segs = new_segs

    return segs


# ---------------------------------------------------------------------------
# helpers shared by consolidation and identification
# ---------------------------------------------------------------------------

def _extract_embeddings(segments: list, audio_path: str, tmpdir: Path, device: str) -> dict:
    """Return {speaker_id: normalised_embedding} for each unique speaker in segments."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from diarize import get_speaker_embedding, _build_speaker_clip

    embeddings: dict = {}
    for spk in sorted({s["speaker"] for s in segments}):
        spk_segs = [s for s in segments if s["speaker"] == spk]
        clip = _build_speaker_clip(Path(audio_path), spk_segs, tmpdir, spk, max_sec=60.0)
        if clip is None:
            print(f"  Warning: could not build clip for {spk} — skipped", file=sys.stderr)
            continue
        try:
            emb = get_speaker_embedding(clip, device)
            norm = np.linalg.norm(emb)
            embeddings[spk] = emb / (norm + 1e-8)
        except Exception as exc:
            print(f"  Warning: embedding failed for {spk}: {exc}", file=sys.stderr)
    return embeddings


# ---------------------------------------------------------------------------
# 0. Automatic speaker consolidation via TitaNet pairwise similarity
# ---------------------------------------------------------------------------

def consolidate_speakers(
    segments: list,
    audio_path: str,
    threshold: float,
    tmpdir: Path,
    device: str = "cpu",
) -> list:
    """
    Extract a TitaNet embedding for each speaker cluster, compute pairwise
    cosine similarities, and merge clusters above threshold using union-find.
    """
    import numpy as np

    speakers = sorted({s["speaker"] for s in segments})
    if len(speakers) < 2:
        print("  Consolidation: fewer than 2 speakers, nothing to do", file=sys.stderr)
        return segments

    print(f"  Consolidation: extracting embeddings for {len(speakers)} speaker(s)…", file=sys.stderr)
    embeddings = _extract_embeddings(segments, audio_path, tmpdir, device)
    spk_list = list(embeddings.keys())
    if len(spk_list) < 2:
        return segments

    # Union-find
    parent = {spk: spk for spk in spk_list}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Compute all pairwise similarities, collect candidates
    pairs: list = []
    for i in range(len(spk_list)):
        for j in range(i + 1, len(spk_list)):
            a, b = spk_list[i], spk_list[j]
            sim = float(np.dot(embeddings[a], embeddings[b]))
            pairs.append((sim, a, b))

    # Print similarity matrix
    print("\n  Pairwise cosine similarities:", file=sys.stderr)
    for sim, a, b in sorted(pairs, key=lambda x: -x[0]):
        marker = " ← merge" if sim >= threshold else ""
        print(f"    {a} ↔ {b}: {sim:.3f}{marker}", file=sys.stderr)

    # Merge from highest similarity down
    merged_pairs: list = []
    for sim, a, b in sorted(pairs, key=lambda x: -x[0]):
        if sim < threshold:
            break
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa
            merged_pairs.append((a, b, sim))

    if not merged_pairs:
        print("  Consolidation: no pairs above threshold — nothing merged", file=sys.stderr)
        return segments

    print("\n  Consolidations applied:", file=sys.stderr)
    for a, b, sim in merged_pairs:
        print(f"    {b} → {find(a)}  (sim={sim:.3f})", file=sys.stderr)

    speaker_map = {spk: find(spk) for spk in spk_list}
    return [dict(s, speaker=speaker_map.get(s["speaker"], s["speaker"])) for s in segments]


# ---------------------------------------------------------------------------
# 0.5. Identification from known speaker signatures
# ---------------------------------------------------------------------------

def load_signatures_local(folder: Path) -> dict:
    """Load all .npy files from a local folder. Returns {speaker_name: normalised_embedding}."""
    import numpy as np
    sigs: dict = {}
    for npy in sorted(folder.glob("*.npy")):
        emb = np.load(str(npy)).astype(np.float32)
        norm = np.linalg.norm(emb)
        sigs[npy.stem] = emb / (norm + 1e-8)
    return sigs


def load_signatures_gcs(gcs_uri: str) -> dict:
    """
    Load .npy signatures from a GCS prefix.
    gcs_uri: 'gs://bucket-name/signatures/' or 'gs://bucket-name/signatures'
    Auth: GOOGLE_APPLICATION_CREDENTIALS env var (path to key.json)
       or GOOGLE_CREDENTIALS_JSON env var (base64-encoded key.json content)
    """
    import io
    import os
    import base64
    import numpy as np
    import tempfile

    # Bootstrap credentials from base64 env var if needed
    creds_b64 = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
    if creds_b64 and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(base64.b64decode(creds_b64))
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

    from google.cloud import storage

    # Parse gs://bucket/prefix
    without_scheme = gcs_uri[len("gs://"):]
    bucket_name, _, prefix = without_scheme.partition("/")
    prefix = prefix.rstrip("/") + "/"

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    sigs: dict = {}
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".npy"):
            continue
        name = Path(blob.name).stem
        emb = np.load(io.BytesIO(blob.download_as_bytes())).astype(np.float32)
        norm = np.linalg.norm(emb)
        sigs[name] = emb / (norm + 1e-8)
        print(f"  Loaded signature: {name}", file=sys.stderr)
    return sigs


def identify_from_signatures(
    segments: list,
    audio_path: str,
    signatures: dict,
    threshold: float,
    tmpdir: Path,
    device: str = "cpu",
) -> list:
    """
    For each anonymous speaker cluster, extract a TitaNet embedding and compare
    against known signatures. Replace speaker_X with the real name if similarity
    >= threshold. Prints the match table.
    """
    import numpy as np

    if not signatures:
        print("  Identification: no signatures loaded — skipped", file=sys.stderr)
        return segments

    print(
        f"  Identification: {len(signatures)} signature(s) loaded, threshold={threshold}",
        file=sys.stderr,
    )
    embeddings = _extract_embeddings(segments, audio_path, tmpdir, device)
    if not embeddings:
        return segments

    speaker_map: dict = {}
    print("\n  Speaker identification results:", file=sys.stderr)
    for spk, emb in sorted(embeddings.items()):
        best_name, best_sim = max(
            ((name, float(np.dot(emb, sig_emb))) for name, sig_emb in signatures.items()),
            key=lambda x: x[1],
        )
        if best_sim >= threshold:
            speaker_map[spk] = best_name
            print(f"    {spk} → {best_name}  (sim={best_sim:.3f})", file=sys.stderr)
        else:
            print(f"    {spk} → (unidentified, best={best_name} {best_sim:.3f})", file=sys.stderr)

    if not speaker_map:
        print("  Identification: no speakers matched above threshold", file=sys.stderr)
        return segments

    return [dict(s, speaker=speaker_map.get(s["speaker"], s["speaker"])) for s in segments]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process NeMo diarization output (RTTM or JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="RTTM or JSON file from NeMo / handler.py")
    parser.add_argument(
        "--blocks", "-b",
        help="JSON file with time blocks for speaker unification "
             '[{"start": "01:29:19", "end": "01:41:10"}, ...]',
    )
    parser.add_argument(
        "--gap", "-g", type=float, default=1.5,
        help="Max pause (sec) between same-speaker segments to merge (default: 1.5)",
    )
    parser.add_argument(
        "--min-seg", "-m", type=float, default=0.4,
        help="Segments shorter than this (sec) are absorbed/deleted (default: 0.4)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file (default: stdout)",
    )
    parser.add_argument(
        "--audio", "-a",
        help="Original audio file (required for --consolidate)",
    )
    parser.add_argument(
        "--consolidate", "-c", type=float, metavar="THRESHOLD",
        help="Auto-merge speaker clusters with cosine similarity >= THRESHOLD (e.g. 0.85). "
             "Requires --audio.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for TitaNet embedding (default: cpu). Use 'cuda' for GPU.",
    )
    parser.add_argument(
        "--signatures", "-s",
        help="Folder with speaker signature .npy files (local path or gs://bucket/prefix/). "
             "Requires --audio. Filename stem = speaker name.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.75,
        help="Cosine similarity threshold for signature matching (default: 0.75)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load
    segments = load_input(input_path)
    print(f"Loaded {len(segments)} segments from {input_path.name}", file=sys.stderr)

    # Load blocks
    blocks = []
    if args.blocks:
        raw_blocks = json.loads(Path(args.blocks).read_text())
        for b in raw_blocks:
            blocks.append({
                "start": _parse_time(str(b["start"])),
                "end": _parse_time(str(b["end"])),
                "merge": b.get("merge", []),
            })
        print(f"Loaded {len(blocks)} unification block(s)", file=sys.stderr)

    # Step 0 — auto-consolidate speakers via TitaNet embeddings
    if args.consolidate is not None:
        if not args.audio:
            print("Error: --audio is required when --consolidate is set", file=sys.stderr)
            sys.exit(1)
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)
        print(f"\n--- Step 0: Speaker consolidation (threshold={args.consolidate}) ---", file=sys.stderr)
        with tempfile.TemporaryDirectory() as _tmp:
            segments = consolidate_speakers(
                segments, str(audio_path), args.consolidate, Path(_tmp), args.device
            )
        print(f"After consolidation: {len(segments)} segments", file=sys.stderr)

    # Step 0.5 — identify speakers from signatures
    if args.signatures:
        if not args.audio:
            print("Error: --audio is required when --signatures is set", file=sys.stderr)
            sys.exit(1)
        audio_path_id = Path(args.audio)
        if not audio_path_id.exists():
            print(f"Error: audio file not found: {audio_path_id}", file=sys.stderr)
            sys.exit(1)
        print(f"\n--- Step 0.5: Speaker identification (threshold={args.threshold}) ---", file=sys.stderr)
        if args.signatures.startswith("gs://"):
            print(f"  Loading signatures from GCS: {args.signatures}", file=sys.stderr)
            signatures = load_signatures_gcs(args.signatures)
        else:
            sig_folder = Path(args.signatures)
            if not sig_folder.is_dir():
                print(f"Error: signatures folder not found: {sig_folder}", file=sys.stderr)
                sys.exit(1)
            print(f"  Loading signatures from: {sig_folder}", file=sys.stderr)
            signatures = load_signatures_local(sig_folder)
        print(f"  {len(signatures)} signature(s) loaded", file=sys.stderr)
        with tempfile.TemporaryDirectory() as _tmp:
            segments = identify_from_signatures(
                segments, str(audio_path_id), signatures, args.threshold, Path(_tmp), args.device
            )
        print(f"After identification: {len(segments)} segments", file=sys.stderr)

    # Step 1 — unify speakers in blocks
    if blocks:
        segments = unify_speakers_in_blocks(segments, blocks)
        print(f"After unification: {len(segments)} segments", file=sys.stderr)

    # Step 2 — fill gaps
    segments = fill_gaps(segments, args.gap)
    print(f"After gap fill (gap={args.gap}s): {len(segments)} segments", file=sys.stderr)

    # Step 3 — absorb short segments
    segments = absorb_short_segments(segments, args.min_seg, args.gap)
    print(f"After absorption (min={args.min_seg}s): {len(segments)} segments", file=sys.stderr)

    # Print summary
    print("\n--- Post-processed Segments ---", file=sys.stderr)
    for s in segments:
        print(f"  [{_fmt(s['start'])} → {_fmt(s['end'])}]  {s['speaker']}  ({s['duration']:.2f}s)", file=sys.stderr)

    # Output
    out = {"segments": segments}
    text = json.dumps(out, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
        print(f"\nSaved to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

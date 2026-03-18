#!/usr/bin/env python3
"""
RunPod serverless handler for NeMo speaker diarization.

Payload schema — diarize mode (default):
  {
    "input": {
      "gcs_audio_path": "recordings/audio.mp3",   # required
      "gcs_output_prefix": "results/",             # optional, default "results/"
      "speakers": null,                             # optional, int or null (auto-detect)
      "no_msdd": false,                             # optional, faster clustering-only mode
      "gap": 5,                                     # optional, overrides config.yaml
      "min_seg": 0.5,                               # optional, overrides config.yaml
      "identify_speakers": true,                    # optional, match clusters to GCS signatures
      "similarity_threshold": 0.75                  # optional, cosine sim cutoff (default 0.75)
    }
  }
  When identify_speakers=true:
    - Loads all *.npy from signatures/ in GCS bucket
    - Matches each speaker cluster to the best signature
    - Replaces speaker_X with real name if similarity >= threshold
    - Saves 30-sec review clip to unknown_samples/ for unmatched speakers
    - Reinforces matched signatures with the current audio (running average)

Payload schema — enroll mode:
  {
    "input": {
      "mode": "enroll",
      "speaker_name": "Ion Popescu",
      "gcs_audio_paths": ["samples/ion1.mp3", "samples/ion2.mp3"]
    }
  }
  Multiple clips are averaged into a single signature. If a signature already
  exists for that name, the new embedding is averaged with the existing one.

Environment variables:
  GCS_BUCKET                — GCS bucket name (required)
  GOOGLE_CREDENTIALS_JSON   — service account key as base64 or raw JSON
"""

import io
import json as _json
import os
import tempfile
from pathlib import Path

import numpy as np
import runpod
from google.cloud import storage
from omegaconf import OmegaConf

from diarize import (
    _build_speaker_clip,
    convert_audio,
    filter_short_segments,
    get_speaker_embedding,
    identify_speakers,
    merge_segments,
    needs_conversion,
    parse_rttm,
    process_single,
)

# ---------------------------------------------------------------------------
# GCS credentials bootstrap (support JSON string env var for RunPod secrets)
# ---------------------------------------------------------------------------

_CREDS_JSON = (os.environ.get("GOOGLE_CREDENTIALS_JSON") or "").strip()
if _CREDS_JSON and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    import base64, json as _json_boot
    try:
        decoded = base64.b64decode(_CREDS_JSON, validate=True).decode("utf-8").strip()
        _json_boot.loads(decoded)  # validate it decodes to valid JSON
        _CREDS_JSON = decoded
    except Exception:
        pass  # already raw JSON
    if not _CREDS_JSON:
        raise RuntimeError(
            "GOOGLE_CREDENTIALS_JSON is set but empty after decoding — check the value in RunPod secrets"
        )
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _tmp.write(_CREDS_JSON)
    _tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _gcs_client() -> storage.Client:
    return storage.Client()


def _download(blob_path: str, local_path: str) -> None:
    _gcs_client().bucket(GCS_BUCKET).blob(blob_path).download_to_filename(local_path)


def _upload(blob_path: str, local_path: str) -> str:
    _gcs_client().bucket(GCS_BUCKET).blob(blob_path).upload_from_filename(local_path)
    return f"gs://{GCS_BUCKET}/{blob_path}"


def _load_signatures(sigs_prefix: str) -> dict:
    """Load all speaker signature .npy files from GCS. Returns {name: embedding}."""
    client = _gcs_client()
    signatures: dict = {}
    for blob in client.bucket(GCS_BUCKET).list_blobs(prefix=sigs_prefix):
        if not blob.name.endswith(".npy"):
            continue
        name = Path(blob.name).stem
        emb = np.load(io.BytesIO(blob.download_as_bytes()))
        signatures[name] = emb
        print(f"  Loaded signature: {name}")
    return signatures


def _save_signature(name: str, embedding: np.ndarray, sigs_prefix: str) -> str:
    """Save (or average with existing) a speaker embedding to GCS."""
    client = _gcs_client()
    blob_path = f"{sigs_prefix.rstrip('/')}/{name}.npy"
    blob = client.bucket(GCS_BUCKET).blob(blob_path)
    if blob.exists():
        existing = np.load(io.BytesIO(blob.download_as_bytes()))
        embedding = (existing + embedding) / 2.0
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    buf = io.BytesIO()
    np.save(buf, embedding.astype(np.float32))
    buf.seek(0)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    return f"gs://{GCS_BUCKET}/{blob_path}"


# ---------------------------------------------------------------------------
# Enrollment mode
# ---------------------------------------------------------------------------

def _handle_enroll(inp: dict, tmpdir: Path) -> dict:
    """Enroll a new speaker from one or more audio clips stored in GCS."""
    import torch
    speaker_name: str = (inp.get("speaker_name") or "").strip()
    gcs_paths: list = inp.get("gcs_audio_paths") or []

    if not speaker_name:
        return {"error": "speaker_name is required for enrollment"}
    if not gcs_paths:
        return {"error": "gcs_audio_paths is required for enrollment"}
    if not GCS_BUCKET:
        return {"error": "GCS_BUCKET environment variable not set"}

    cfg_si = OmegaConf.load(_CONFIG_PATH).get("speaker_identification", {})
    sigs_prefix = cfg_si.get("signatures_prefix", "signatures/")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeddings = []
    for gcs_path in gcs_paths:
        local = tmpdir / Path(gcs_path).name
        try:
            _download(gcs_path, str(local))
        except Exception as e:
            return {"error": f"Failed to download {gcs_path}: {e}"}
        if needs_conversion(local):
            converted = tmpdir / f"_conv_{local.name}.wav"
            convert_audio(local, converted)
            local = converted
        try:
            emb = get_speaker_embedding(local, device)
            embeddings.append(emb)
        except Exception as e:
            print(f"  Warning: embedding failed for {gcs_path}: {e}")

    if not embeddings:
        return {"error": "Failed to extract embeddings from any provided clip"}

    final_emb = np.mean(embeddings, axis=0).astype(np.float32)
    uri = _save_signature(speaker_name, final_emb, sigs_prefix)
    print(f"  Enrolled '{speaker_name}' → {uri}")
    return {"enrolled": speaker_name, "clips_used": len(embeddings), "signature_uri": uri}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job):
    inp = job["input"]

    # Route to enrollment if requested
    if inp.get("mode") == "enroll":
        with tempfile.TemporaryDirectory() as _tmp:
            return _handle_enroll(inp, Path(_tmp))

    # --- Diarize mode ---
    import torch
    gcs_audio_path: str = inp["gcs_audio_path"]
    gcs_output_prefix: str = inp.get("gcs_output_prefix", "results/").rstrip("/")
    speakers = inp.get("speakers", None)
    no_msdd: bool = inp.get("no_msdd", False)
    do_identify: bool = inp.get("identify_speakers", False)

    cfg = OmegaConf.load(_CONFIG_PATH)
    cfg_pp = cfg.get("postprocessing", {})
    gap = float(inp.get("gap", cfg_pp.get("gap", 5.0)))
    min_seg = float(inp.get("min_seg", cfg_pp.get("min_seg", 0.0)))

    cfg_si = cfg.get("speaker_identification", {})
    sigs_prefix = cfg_si.get("signatures_prefix", "signatures/")
    unknown_prefix = cfg_si.get("unknown_prefix", "unknown_samples/")
    threshold = float(inp.get("similarity_threshold", cfg_si.get("similarity_threshold", 0.75)))

    if not GCS_BUCKET:
        return {"error": "GCS_BUCKET environment variable not set"}

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)

        # 1 — Download audio from GCS
        audio_filename = Path(gcs_audio_path).name
        local_audio = tmpdir / audio_filename
        print(f"Downloading gs://{GCS_BUCKET}/{gcs_audio_path} ...")
        try:
            _download(gcs_audio_path, str(local_audio))
        except Exception as e:
            return {"error": f"Failed to download audio from GCS: {e}"}

        # 2 — Run diarization
        output_dir = tmpdir / "output"
        try:
            metrics = process_single(
                audio_path=local_audio,
                num_speakers=speakers,
                config_path=_CONFIG_PATH,
                output_dir=output_dir,
                no_msdd=no_msdd,
                gap=gap,
                min_seg=min_seg,
            )
        except Exception as e:
            return {"error": f"Diarization failed: {e}"}

        if metrics is None:
            return {"error": "Diarization produced no output"}

        # 3 — Parse RTTM
        rttm_files = list((output_dir / "pred_rttms").glob("*.rttm"))
        if not rttm_files:
            return {"error": "RTTM file missing after diarization"}
        raw = parse_rttm(rttm_files[0])
        segments = merge_segments(filter_short_segments(raw, min_seg), gap)

        # 4 — Speaker identification (optional)
        unknown_speakers: dict = {}
        if do_identify:
            print("Loading voice signatures from GCS ...")
            signatures = _load_signatures(sigs_prefix)
            if signatures:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                id_tmpdir = tmpdir / "id_tmp"
                id_tmpdir.mkdir()
                segments, unknown_clips = identify_speakers(
                    segments, local_audio, signatures, threshold, id_tmpdir, device
                )

                # Upload unknown speaker samples to GCS for review
                stem = Path(gcs_audio_path).stem
                for spk_id, clip_path in unknown_clips.items():
                    gcs_clip = f"{unknown_prefix.rstrip('/')}/{stem}_{spk_id}.wav"
                    uri = _upload(gcs_clip, str(clip_path))
                    unknown_speakers[spk_id] = uri
                    print(f"  Unknown sample → {uri}")

                # Reinforce signatures for identified speakers using current audio
                identified_names = {s["speaker"] for s in segments} & set(signatures.keys())
                if identified_names:
                    print("Reinforcing signatures for identified speakers ...")
                    reinforce_tmpdir = id_tmpdir / "reinforce"
                    reinforce_tmpdir.mkdir()
                    for name in sorted(identified_names):
                        name_segs = [s for s in segments if s["speaker"] == name]
                        clip = _build_speaker_clip(
                            local_audio, name_segs, reinforce_tmpdir, name, max_sec=60.0
                        )
                        if clip is None:
                            continue
                        try:
                            emb = get_speaker_embedding(clip, device)
                            _save_signature(name, emb, sigs_prefix)
                            print(f"  Reinforced: {name}")
                        except Exception as e:
                            print(f"  Warning: reinforce failed for {name}: {e}")
            else:
                print("  No signatures found — skipping identification")

        # 5 — Build result JSON and upload to GCS
        stem = Path(gcs_audio_path).stem
        result_payload = {"segments": segments, "metrics": metrics}
        if unknown_speakers:
            result_payload["unknown_speakers"] = unknown_speakers
        json_local = tmpdir / f"{stem}.json"
        json_local.write_text(_json.dumps(result_payload, indent=2, default=str))

        gcs_json_path = f"{gcs_output_prefix}/{stem}.json"
        json_uri = _upload(gcs_json_path, str(json_local))
        print(f"Uploaded JSON → {json_uri}")

        result = {"segments": segments, "metrics": metrics, "gcs_uri": json_uri}
        if unknown_speakers:
            result["unknown_speakers"] = unknown_speakers
        return result


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

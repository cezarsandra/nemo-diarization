#!/usr/bin/env python3
"""
RunPod serverless handler for NeMo speaker diarization.

Payload schema:
  {
    "input": {
      "gcs_audio_path": "recordings/audio.mp3",   # path inside GCS_BUCKET
      "gcs_output_prefix": "results/",             # optional, default "results/"
      "speakers": null,                             # optional, int or null
      "no_msdd": false                              # optional, bool
    }
  }

Environment variables:
  GCS_BUCKET                  — GCS bucket name (required)
  GOOGLE_APPLICATION_CREDENTIALS      — path to service account key file
  GOOGLE_CREDENTIALS_JSON     — service account key as JSON string (alternative)
"""

import os
import tempfile
from pathlib import Path

import runpod
from google.cloud import storage
from omegaconf import OmegaConf

from diarize import (
    filter_short_segments,
    merge_segments,
    parse_rttm,
    process_single,
)

# ---------------------------------------------------------------------------
# GCS credentials bootstrap (support JSON string env var for RunPod secrets)
# ---------------------------------------------------------------------------

_CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if _CREDS_JSON and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    import base64
    # Support both raw JSON and base64-encoded JSON
    try:
        decoded = base64.b64decode(_CREDS_JSON).decode("utf-8")
        import json as _json; _json.loads(decoded)  # validate it's JSON
        _CREDS_JSON = decoded
    except Exception:
        pass  # already raw JSON
    _tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    _tmp.write(_CREDS_JSON)
    _tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _tmp.name

GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _download(blob_path: str, local_path: str) -> None:
    client = storage.Client()
    client.bucket(GCS_BUCKET).blob(blob_path).download_to_filename(local_path)


def _upload(blob_path: str, local_path: str) -> str:
    client = storage.Client()
    client.bucket(GCS_BUCKET).blob(blob_path).upload_from_filename(local_path)
    return f"gs://{GCS_BUCKET}/{blob_path}"


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job):
    inp = job["input"]
    gcs_audio_path: str = inp["gcs_audio_path"]
    gcs_output_prefix: str = inp.get("gcs_output_prefix", "results/").rstrip("/")
    speakers = inp.get("speakers", None)
    no_msdd: bool = inp.get("no_msdd", False)

    if not GCS_BUCKET:
        return {"error": "GCS_BUCKET environment variable not set"}

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)

        # 1 — Download audio from GCS
        audio_filename = Path(gcs_audio_path).name
        local_audio = tmpdir / audio_filename
        print(f"Downloading gs://{GCS_BUCKET}/{gcs_audio_path} ...")
        _download(gcs_audio_path, str(local_audio))

        # 2 — Run diarization
        output_dir = tmpdir / "output"
        metrics = process_single(
            audio_path=local_audio,
            num_speakers=speakers,
            config_path=_CONFIG_PATH,
            output_dir=output_dir,
            no_msdd=no_msdd,
        )

        if metrics is None:
            return {"error": "Diarization produced no output"}

        # 3 — Find the RTTM NeMo wrote
        rttm_files = list((output_dir / "pred_rttms").glob("*.rttm"))
        if not rttm_files:
            return {"error": "RTTM file missing after diarization"}
        rttm_local = rttm_files[0]

        # 4 — Upload RTTM to GCS
        stem = Path(gcs_audio_path).stem
        gcs_rttm_path = f"{gcs_output_prefix}/{stem}.rttm"
        rttm_uri = _upload(gcs_rttm_path, str(rttm_local))
        print(f"Uploaded RTTM → {rttm_uri}")

        # 5 — Re-parse RTTM to return segments in response
        cfg_pp = OmegaConf.load(_CONFIG_PATH).get("postprocessing", {})
        gap = float(cfg_pp.get("gap", 0.5))
        min_seg = float(cfg_pp.get("min_seg", 0.0))
        raw = parse_rttm(rttm_local)
        segments = merge_segments(filter_short_segments(raw, min_seg), gap)

        return {
            "segments": segments,
            "metrics": metrics,
            "rttm_gcs_uri": rttm_uri,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

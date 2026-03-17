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

_CREDS_JSON = (os.environ.get("GOOGLE_CREDENTIALS_JSON") or "").strip()
if _CREDS_JSON and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    import base64, json as _json
    # Try base64 decode first, fall back to raw JSON
    try:
        decoded = base64.b64decode(_CREDS_JSON, validate=True).decode("utf-8").strip()
        _json.loads(decoded)  # validate it decodes to valid JSON
        _CREDS_JSON = decoded
    except Exception:
        pass  # already raw JSON
    if not _CREDS_JSON:
        raise RuntimeError("GOOGLE_CREDENTIALS_JSON is set but empty after decoding — check the value in RunPod secrets")
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

    # Post-processing params — input overrides config.yaml defaults
    cfg_pp = OmegaConf.load(_CONFIG_PATH).get("postprocessing", {})
    gap = float(inp.get("gap", cfg_pp.get("gap", 0.5)))
    min_seg = float(inp.get("min_seg", cfg_pp.get("min_seg", 0.0)))

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

        # 3 — Find the RTTM NeMo wrote
        rttm_files = list((output_dir / "pred_rttms").glob("*.rttm"))
        if not rttm_files:
            return {"error": "RTTM file missing after diarization"}
        rttm_local = rttm_files[0]

        # 4 — Build processed segments JSON and upload to GCS
        import json as _json
        raw = parse_rttm(rttm_local)
        segments = merge_segments(filter_short_segments(raw, min_seg), gap)

        stem = Path(gcs_audio_path).stem
        result_payload = {"segments": segments, "metrics": metrics}
        json_local = tmpdir / f"{stem}.json"
        json_local.write_text(_json.dumps(result_payload, indent=2, default=str))

        gcs_json_path = f"{gcs_output_prefix}/{stem}.json"
        json_uri = _upload(gcs_json_path, str(json_local))
        print(f"Uploaded JSON → {json_uri}")

        return {
            "segments": segments,
            "metrics": metrics,
            "gcs_uri": json_uri,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

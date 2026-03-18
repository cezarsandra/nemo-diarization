# NeMo Speaker Diarization

Speaker diarization using NVIDIA NeMo (MSDD — Multi-Scale Diarization Decoder).
Supports local CLI, Docker, and RunPod serverless with Google Cloud Storage.

---

## Requirements

- Python 3.10
- ffmpeg
- CUDA-capable GPU (recommended; CPU works but is slow)
- NVIDIA NeMo (`nemo_toolkit[asr]`)

---

## Mode 1 — Local CLI

Run diarization directly on your machine using the virtual environment.

### Setup (one time)

```bash
# Install system dependencies
sudo apt-get install -y ffmpeg libsndfile1

# Create venv and install Python packages
make setup
```

### Run

```bash
# Single file
make run FILE=audio.mp3

# Single file, known speaker count
make run FILE=audio.mp3 SPEAKERS=2

# Batch — all .wav files in a directory
make run-batch DIR=./recordings

# Force faster clustering-only mode (no MSDD)
make run FILE=audio.mp3 NO_MSDD=1
```

### Options

| Variable | Default | Description |
|----------|---------|-------------|
| `FILE` | `audio.wav` | Audio file path |
| `DIR` | `./recordings` | Directory for batch mode |
| `OUTPUT` | `./output` | Output directory |
| `SPEAKERS` | _(auto)_ | Known number of speakers |
| `GAP` | `5` | Max pause (sec) between same-speaker turns to merge |
| `MIN_SEG` | `10` | Drop segments shorter than this (sec) |
| `NO_MSDD` | _(off)_ | Set to `1` to use clustering-only (faster) |
| `CONFIG` | `config.yaml` | NeMo config file |

### Output

Results are printed to stdout and saved to `./output/`:

```
--- Diarization Results ---
[00:00.46 --> 01:00.17]  speaker_1
[01:59.52 --> 04:00.21]  speaker_0
[05:13.64 --> 05:58.33]  speaker_2

--- Metrics ---
  Speakers : 3
  Segments : 3
  Speech   : 244.59 sec
  Coverage : 68.2%
```

### Tune post-processing

Edit `config.yaml` to change default values:

```yaml
postprocessing:
  gap: 5.0      # merge same-speaker turns separated by <= this many seconds
  min_seg: 0.5  # drop segments shorter than this (0 = keep all)
```

---

## Mode 2 — Local, simulating RunPod (with real GCS)

Test the full RunPod handler flow locally — downloads audio from GCS, runs diarization,
uploads result JSON back to GCS — without Docker or RunPod.

### Setup (one time)

```bash
cp .env.example .env
cp test_input.json.example test_input.json
```

**Edit `.env`:**

```bash
GCS_BUCKET=my-bucket-name
GOOGLE_CREDENTIALS_JSON=<base64-encoded key.json>
```

To encode your service account key:
```bash
cat key.json | base64 -w 0
```

**Edit `test_input.json`:**

```json
{
  "input": {
    "gcs_audio_path": "recordings/audio.mp3",
    "gcs_output_prefix": "results/",
    "speakers": null,
    "no_msdd": false,
    "gap": 5,
    "min_seg": 0.5
  }
}
```

### Run

```bash
make test-handler
```

The result JSON (`segments` + `metrics`) is uploaded to GCS at
`gs://<GCS_BUCKET>/<gcs_output_prefix>/<filename>.json` and printed to stdout.

---

## Mode 3 — Docker (local, with GPU)

Build and run the container locally before deploying to RunPod.

### Build

```bash
make docker-build
```

> First build takes ~20 minutes — NeMo models are pre-downloaded into the image.

### Test with a local file (no GCS)

```bash
make docker-test-local FILE=audio.mp3
```

### Test with GCS

```bash
make docker-test \
  GCS_BUCKET=my-bucket \
  GCS_AUDIO_PATH=recordings/audio.mp3 \
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

---

## Mode 4 — RunPod Serverless

### Deploy

1. Push code to GitHub
2. On RunPod: **Serverless** → **New Endpoint** → select **GitHub** as source
3. Point to your repo and `Dockerfile`
4. Set environment variables (see below)
5. Select a GPU (RTX 4090 or A100 recommended)

### Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GCS_BUCKET` | Yes | GCS bucket name |
| `GOOGLE_CREDENTIALS_JSON` | Yes | `key.json` content encoded in base64 |

To get the base64 value:
```bash
cat key.json | base64 -w 0
```

### Input payload

```json
{
  "input": {
    "gcs_audio_path": "recordings/audio.mp3",
    "gcs_output_prefix": "results/",
    "speakers": null,
    "no_msdd": false,
    "gap": 5,
    "min_seg": 0.5
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `gcs_audio_path` | string | Yes | Path to audio file inside the GCS bucket |
| `gcs_output_prefix` | string | No | Output folder in GCS (default: `"results/"`) |
| `speakers` | int or null | No | Known speaker count; `null` = auto-detect |
| `no_msdd` | bool | No | `true` = faster clustering-only mode (default: `false`) |
| `gap` | float | No | Max pause (sec) to merge same-speaker turns (default from config) |
| `min_seg` | float | No | Drop segments shorter than this in seconds (default from config) |

### Response

```json
{
  "segments": [
    {"start": 0.46, "end": 60.17, "speaker": "speaker_1", "duration": 59.71},
    {"start": 119.52, "end": 240.21, "speaker": "speaker_0", "duration": 120.69},
    {"start": 313.64, "end": 358.33, "speaker": "speaker_2", "duration": 44.69}
  ],
  "metrics": {
    "segments": 3,
    "speakers": 3,
    "speech_sec": 225.09,
    "coverage_pct": 62.5
  },
  "gcs_uri": "gs://my-bucket/results/audio.json"
}
```

The full result is also saved as JSON to GCS at the path in `gcs_uri`.

### GCS Service Account setup

1. **Google Cloud Console** → **IAM & Admin** → **Service Accounts**
2. Create a service account with role **Storage Object Admin**
3. **Keys** tab → **Add Key** → **JSON** → download `key.json`
4. Encode: `cat key.json | base64 -w 0`
5. Paste the output as `GOOGLE_CREDENTIALS_JSON` in RunPod environment variables

.PHONY: help venv install install-sys setup run run-batch lint clean docker-build docker-test docker-test-local test-handler postprocess

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip

# Defaults (override on command line: make run FILE=my.wav SPEAKERS=2)
IMAGE   ?= nemo-diarization
FILE    ?= audio.wav
DIR     ?= ./recordings
OUTPUT  ?= ./output
CONFIG  ?= config.yaml
SPEAKERS ?=
GAP      ?= 5
MIN_SEG  ?= 10

_SPEAKERS_FLAG := $(if $(SPEAKERS),--speakers $(SPEAKERS),)
_NO_MSDD_FLAG  := $(if $(NO_MSDD),--no-msdd,)
_MIN_SEG_FLAG  := $(if $(filter-out 0.0,$(MIN_SEG)),--min-seg $(MIN_SEG),)

help:
	@echo "Usage:"
	@echo "  make venv               Create .venv virtual environment"
	@echo "  make setup              Create venv + system deps + Python packages"
	@echo "  make install            Install Python packages into .venv"
	@echo "  make install-sys        Install system packages (ffmpeg, libsndfile1)"
	@echo "  make run FILE=audio.wav Run diarization on a single file"
	@echo "  make run-batch DIR=./r  Run diarization on all .wav files in a directory"
	@echo "  make lint               Check diarize.py syntax"
	@echo "  make clean              Remove output directory"
	@echo "  make clean-venv         Remove .venv"
	@echo ""
	@echo "Options (append to any run target):"
	@echo "  SPEAKERS=2              Force known speaker count"
	@echo "  OUTPUT=./results        Custom output directory"
	@echo "  CONFIG=config.yaml      Custom config file"
	@echo "  NO_MSDD=1               Skip MSDD (clustering-only, faster)"
	@echo "  GAP=0.5                 Max pause (sec) to merge same-speaker turns (default: 0.5)"
	@echo "  MIN_SEG=0.5             Drop segments shorter than this (sec) before merging (default: disabled)"
	@echo "  AUDIO=audio.mp3         Original audio file (required for CONSOLIDATE / SIGNATURES)"
	@echo "  CONSOLIDATE=0.85        Auto-merge speaker clusters with similarity >= threshold"
	@echo "  SIGNATURES=./sigs/      Folder (local or gs://bucket/prefix/) with .npy signatures"
	@echo "  THRESHOLD=0.75          Cosine similarity threshold for signature matching"
	@echo "  DEVICE=cuda             Torch device for TitaNet (default: cpu)"
	@echo ""
	@echo "Docker targets:"
	@echo "  make docker-build         Build Docker image (IMAGE=nemo-diarization)"
	@echo "  make docker-test-local    Test handler inside Docker with a LOCAL audio file"
	@echo "    Required: FILE=audio.mp3"
	@echo "  make docker-test          Test handler inside Docker with a GCS audio file"
	@echo "    Required: GCS_BUCKET=my-bucket GCS_AUDIO_PATH=recordings/audio.mp3"
	@echo "    Required: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"
	@echo ""
	@echo "Local handler test (fara Docker):"
	@echo "  make test-handler         Simuleaza un job RunPod local cu GCS real"
	@echo "    Setup: cp .env.example .env && cp test_input.json.example test_input.json"
	@echo "    Editeaza .env cu bucket + credentials, test_input.json cu calea audio"

# ----------------------------------------------------------------------------
# Virtual environment
# ----------------------------------------------------------------------------

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(VENV)/bin/activate

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

install-sys:
	sudo apt-get update && sudo apt-get install -y libsndfile1 ffmpeg

install: venv
	$(PIP) install -r requirements.txt

setup: install-sys install

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

RTTM        ?=
BLOCKS      ?=
AUDIO       ?=
CONSOLIDATE ?=
SIGNATURES  ?=
THRESHOLD   ?= 0.75
DEVICE      ?= cpu
_BLOCKS_FLAG      := $(if $(BLOCKS),--blocks $(BLOCKS),)
_AUDIO_FLAG       := $(if $(AUDIO),--audio $(AUDIO),)
_CONSOLIDATE_FLAG := $(if $(CONSOLIDATE),--consolidate $(CONSOLIDATE),)
_SIGNATURES_FLAG  := $(if $(SIGNATURES),--signatures $(SIGNATURES),)

postprocess: venv
	@test -n "$(RTTM)" || (echo "ERROR: RTTM=path/to/file.rttm is required" && exit 1)
	$(PYTHON) postprocess.py $(RTTM) \
		$(_BLOCKS_FLAG) \
		$(_AUDIO_FLAG) \
		$(_CONSOLIDATE_FLAG) \
		$(_SIGNATURES_FLAG) \
		$(if $(SIGNATURES),--threshold $(THRESHOLD),) \
		--device $(DEVICE) \
		--gap $(GAP) \
		--min-seg $(MIN_SEG)

run: venv
	$(PYTHON) diarize.py $(FILE) $(_SPEAKERS_FLAG) --output $(OUTPUT) --config $(CONFIG) --gap $(GAP) $(_NO_MSDD_FLAG) $(_MIN_SEG_FLAG)

run-batch: venv
	$(PYTHON) diarize.py $(DIR) $(_SPEAKERS_FLAG) --output $(OUTPUT) --config $(CONFIG) --gap $(GAP) $(_NO_MSDD_FLAG) $(_MIN_SEG_FLAG)

# ----------------------------------------------------------------------------
# Maintenance
# ----------------------------------------------------------------------------

lint: venv
	$(PYTHON) -m py_compile diarize.py && echo "diarize.py: OK"

clean:
	rm -rf $(OUTPUT)
	@echo "Removed $(OUTPUT)"

clean-venv:
	rm -rf $(VENV)
	@echo "Removed $(VENV)"

# ----------------------------------------------------------------------------
# Docker
# ----------------------------------------------------------------------------

GCS_BUCKET           ?=
GCS_AUDIO_PATH       ?=
GOOGLE_APPLICATION_CREDENTIALS ?=

docker-build:
	docker build -t $(IMAGE) .

# Test handler with a LOCAL file (no GCS needed — mounts the file directly)
docker-test-local:
	docker run --rm --gpus all --shm-size=1g \
		-v $(abspath $(FILE)):/tmp/test-audio$(suffix $(FILE)):ro \
		-e GCS_BUCKET=LOCAL_TEST \
		$(IMAGE) \
		python3 -c "\
import sys; sys.path.insert(0, '/app'); \
from pathlib import Path; \
from diarize import process_single; \
m = process_single(Path('/tmp/test-audio$(suffix $(FILE))'), None, Path('/app/config.yaml'), Path('/tmp/out'), False); \
print('metrics:', m)"

# Test handler locally (full end-to-end cu GCS real, fara Docker)
test-handler: venv
	@test -f .env || (echo "ERROR: .env not found. Ruleaza: cp .env.example .env" && exit 1)
	@test -f test_input.json || (echo "ERROR: test_input.json not found. Ruleaza: cp test_input.json.example test_input.json" && exit 1)
	$(PYTHON) test_handler.py

# Test handler with GCS (full end-to-end)
docker-test:
	docker run --rm --gpus all --shm-size=1g \
		-e GCS_BUCKET=$(GCS_BUCKET) \
		-e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcs-key.json \
		-v $(GOOGLE_APPLICATION_CREDENTIALS):/run/secrets/gcs-key.json:ro \
		$(IMAGE) \
		python3 handler.py --test_input '{"input": {"gcs_audio_path": "$(GCS_AUDIO_PATH)"}}'

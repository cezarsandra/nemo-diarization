.PHONY: help venv install install-sys setup run run-batch lint clean docker-build docker-test docker-test-local

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
	@echo ""
	@echo "Docker targets:"
	@echo "  make docker-build         Build Docker image (IMAGE=nemo-diarization)"
	@echo "  make docker-test-local    Test handler inside Docker with a LOCAL audio file"
	@echo "    Required: FILE=audio.mp3"
	@echo "  make docker-test          Test handler inside Docker with a GCS audio file"
	@echo "    Required: GCS_BUCKET=my-bucket GCS_AUDIO_PATH=recordings/audio.mp3"
	@echo "    Required: GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"

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

# Test handler with GCS (full end-to-end)
docker-test:
	docker run --rm --gpus all --shm-size=1g \
		-e GCS_BUCKET=$(GCS_BUCKET) \
		-e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcs-key.json \
		-v $(GOOGLE_APPLICATION_CREDENTIALS):/run/secrets/gcs-key.json:ro \
		$(IMAGE) \
		python3 handler.py --test_input '{"input": {"gcs_audio_path": "$(GCS_AUDIO_PATH)"}}'

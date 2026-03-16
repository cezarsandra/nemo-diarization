.PHONY: help venv install install-sys setup run run-batch lint clean

VENV    := .venv
PYTHON  := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip

# Defaults (override on command line: make run FILE=my.wav SPEAKERS=2)
FILE    ?= audio.wav
DIR     ?= ./recordings
OUTPUT  ?= ./output
CONFIG  ?= config.yaml
SPEAKERS ?=
GAP      ?= 0.5
MIN_SEG  ?= 0.0

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

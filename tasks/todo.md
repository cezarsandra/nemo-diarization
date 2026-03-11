# nemo-diarization — Task Tracker

## Setup
- [x] Create project structure (diarize.py, config.yaml, requirements.txt)
- [x] Refactor code to meet CLAUDE.md standards
- [x] Create tasks/ directory with todo.md and lessons.md

## Validation
- [x] Test `--help` output → clean usage printed
- [x] Test missing audio file error path → "Error: audio file not found"
- [x] Test `--speakers 0` validation → argparse error fired correctly
- [ ] Test full diarization run on a real audio file *(manual — requires NeMo + GPU)*
- [ ] Verify RTTM output is written correctly *(manual — requires NeMo + GPU)*

## Features — Completed
- [x] Add audio format conversion (to 16kHz mono WAV) via ffmpeg
- [x] Add confidence/quality metrics output (speakers, segments, speech %, coverage)
- [x] Support batch processing (multiple files or a directory)

## Review
All CLI validation tests pass. Features (audio conversion, metrics, batch) implemented
and integrated. Full diarization run pending — install `nemo_toolkit[asr]` to verify end-to-end.

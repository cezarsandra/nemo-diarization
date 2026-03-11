# Lessons Learned

## Session: Bug Fix (2026-03-11)
- NeMo hard-accesses (no `.get()`) many config keys — any missing one raises `ConfigAttributeError`. Full list of required top-level diarizer keys: `oracle_vad`, `collar`, `ignore_overlap`; under `vad`: `external_vad_manifest`; under `msdd_model.parameters`: `infer_batch_size`, `seq_eval_mode`, `split_infer`. Run this grep to audit any version: `grep -n "\._diarizer_params\.[a-z]" clustering_diarizer.py msdd_models.py | grep -v "\.get("`.
- Fix config gaps incrementally by running, hitting the next `ConfigAttributeError`, grepping source, adding key — but it's better to grep all hard accesses upfront in one pass.

## Session: Feature Implementation (2026-03-11)
- NeMo imports at module level block `--help` and validation error paths when NeMo is not installed. Always defer NeMo imports to the actual model instantiation call site so the CLI remains functional without the full ML stack.
- `soundfile.info()` is the right tool for inspecting sample rate and channels without loading the full audio — already in requirements, zero extra deps.
- For batch output, give each file its own subdirectory `{output_dir}/{stem}/` so RTTM and converted audio files don't collide.

## Session: Initial Setup (2026-03-11)
- NeMo diarization requires a JSON manifest file at runtime — not a static config
- RTTM output lands in `{out_dir}/pred_rttms/{audio_stem}.rttm` — this path is NeMo-internal
- Conditional imports inside functions hide dependency failures; prefer top-level imports with clear error messages
- `int | None` union syntax requires Python 3.10+; use `Optional[int]` from `typing` for broader compatibility

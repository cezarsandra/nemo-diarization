#!/usr/bin/env python3
"""
Local test runner — simuleaza un job RunPod pentru handler.py.

Folosire:
  cp .env.example .env          # completeaza cu credentials reale
  cp test_input.json.example test_input.json   # completeaza cu calea audio din GCS
  make test-handler
"""
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Incarca .env INAINTE de import handler
#    (handler.py are cod la nivel de modul care bootstrapeaza credentials GCS)
# ---------------------------------------------------------------------------
env_file = Path(".env")
if not env_file.exists():
    print("ERROR: .env not found. Copy .env.example to .env and fill in your values.")
    sys.exit(1)

for line in env_file.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip())

# ---------------------------------------------------------------------------
# 2. Citeste test_input.json
# ---------------------------------------------------------------------------
input_file = Path("test_input.json")
if not input_file.exists():
    print("ERROR: test_input.json not found. Copy test_input.json.example to test_input.json and fill in your values.")
    sys.exit(1)

job = json.loads(input_file.read_text())

# ---------------------------------------------------------------------------
# 3. Ruleaza handler
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from handler import handler  # noqa: E402 — import after env setup

print("Input:")
print(json.dumps(job, indent=2))
print("-" * 60)

result = handler(job)

print("-" * 60)
print("Result:")
print(json.dumps(result, indent=2, default=str))

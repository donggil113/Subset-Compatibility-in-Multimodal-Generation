"""Stable RNG keys (v2).

v1 (experiment.derive_seed) hashed positional parts and omitted the joint
and experiment names, so same-named conditions on different joints shared
noise. v1 is kept unchanged so that committed v1 results stay reproducible.

v2 keys are explicit named fields hashed with SHA-256 over canonical JSON;
Python's salted hash() is never used.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

REQUIRED = ("experiment", "joint", "condition", "example", "replicate", "branch")


def seed_v2(run_seed: int, **fields: Any) -> int:
    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"seed_v2 missing fields: {missing}")
    payload = json.dumps({"run_seed": run_seed, "v": 2, **fields}, sort_keys=True, separators=(",", ":"))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)

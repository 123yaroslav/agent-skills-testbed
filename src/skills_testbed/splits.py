"""Fixed split of tasks into a tuning part and a held-out test part.

The split is by task, not by run. Runs of one task look alike, a split by run would let
rules and prompts tuned on one run of a task be scored on its twin.
"""

from __future__ import annotations

import hashlib


def split_of(task: str) -> str:
    return "tune" if int(hashlib.sha256(task.encode()).hexdigest(), 16) % 2 == 0 else "test"

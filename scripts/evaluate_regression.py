"""Evaluate the "new version is worse" test, baseline and MVP, on SkillsBench runs.

There are no runs of our own stand yet, and SkillsBench has only 3 runs per task and mode,
too few for a real A/A split. So the checks are simulated, with every number taken from
real runs. Each Claude Code task gives a success chance with the skill and without it,
estimated from its 3 + 3 runs (Beta posterior with the Jeffreys prior, so 3 of 3 does not
become a certain success). A PR check takes 12 scenarios, 6 of the changed skill and 6 of
its neighbour, after the selection rule from the design doc. The checks are

    aa         nothing changed, both versions have the with-skill chance
    aa_reused  same, but the old runs are the selection runs, as if they were reused
    removed    the skill stops working on its 6 scenarios, chance drops to the no-skill one
    half       the skill loses half of its effect on its 6 scenarios

Tasks are split into tune and test with skills_testbed.splits. Thresholds are tuned on the
tune checks and scored once on the test checks.

    uv run python scripts/evaluate_regression.py

Writes data/processed/regression_checks.csv.gz, which notebooks/models.ipynb reads.
"""

from __future__ import annotations

import csv
import gzip
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from skills_testbed.regression import baseline_blocks, permutation_pvalue
from skills_testbed.splits import split_of

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "data" / "processed" / "trials.csv"
OUT = ROOT / "data" / "processed" / "regression_checks.csv.gz"

SEED = 20261122
OWN, NEIGHBOUR = 6, 6
RUNS = 3  # MVP runs per scenario and version
EXTRA = 2  # add-on runs per scenario and version in the two-stage rule
N_CHECKS = {"tune": {"aa": 3000, "aa_reused": 1000, "removed": 1500, "half": 1500},
            "test": {"aa": 6000, "aa_reused": 2000, "removed": 3000, "half": 3000}}  # fmt: skip


@dataclass(frozen=True)
class Task:
    name: str
    with_skill: int  # successes out of 3
    without: int


def load_tasks() -> list[Task]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(list)
    with TRIALS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["harness"] == "claude-code":
                counts[(row["task"], row["mode"])].append(int(row["status"] == "pass"))
    names = sorted({task for task, _ in counts})
    return [Task(n, sum(counts[(n, "with")]), sum(counts[(n, "without")])) for n in names]


def runs(rng: random.Random, p: float, k: int) -> list[int]:
    return [int(rng.random() < p) for _ in range(k)]


def scenario(rng: random.Random, tasks: list[Task]) -> tuple[float, float, list[int]]:
    """Draw scenarios until one passes the selection rule. Returns chances and selection runs."""
    while True:
        task = rng.choice(tasks)
        p_with = rng.betavariate(0.5 + task.with_skill, 0.5 + 3 - task.with_skill)
        p_without = rng.betavariate(0.5 + task.without, 0.5 + 3 - task.without)
        picked_with, picked_without = runs(rng, p_with, 3), runs(rng, p_without, 3)
        if sum(picked_with) >= 2 and sum(picked_without) < 3:
            return p_with, p_without, picked_with


def check(rng: random.Random, tasks: list[Task], kind: str) -> dict[str, float | int | str]:
    old_p, new_p, reused = [], [], []
    for i in range(OWN + NEIGHBOUR):
        p_with, p_without, selection_runs = scenario(rng, tasks)
        old_p.append(p_with)
        reused.append(selection_runs)
        own = i < OWN
        if kind == "removed" and own:
            new_p.append(p_without)
        elif kind == "half" and own:
            new_p.append(p_with - 0.5 * (p_with - p_without))
        else:
            new_p.append(p_with)
    old = reused if kind == "aa_reused" else [runs(rng, p, RUNS) for p in old_p]
    new = [runs(rng, p, RUNS) for p in new_p]
    old_more = [o + runs(rng, p, EXTRA) for o, p in zip(old, old_p, strict=True)]
    new_more = [n + runs(rng, p, EXTRA) for n, p in zip(new, new_p, strict=True)]
    true_drop = sum(old_p) / len(old_p) - sum(new_p) / len(new_p)
    return {
        "kind": kind,
        "true_drop": round(true_drop, 4),
        "baseline_block": int(baseline_blocks([o[:1] for o in old], [n[:1] for n in new])),
        "p3": float(f"{permutation_pvalue(old, new):.6g}"),
        "p5": float(f"{permutation_pvalue(old_more, new_more):.6g}"),
    }


def main() -> None:
    rng = random.Random(SEED)  # noqa: S311 (simulation, not cryptography)
    tasks = load_tasks()
    by_split = {s: [t for t in tasks if split_of(t.name) == s] for s in ("tune", "test")}
    rows = []
    for split, plan in N_CHECKS.items():
        for kind, n in plan.items():
            for i in range(n):
                rows.append({"split": split, "check": i, **check(rng, by_split[split], kind)})
    with gzip.open(OUT, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sizes = {s: len(t) for s, t in by_split.items()}
    print(f"tasks per split {sizes}, {len(rows)} checks, wrote {OUT}")


if __name__ == "__main__":
    main()

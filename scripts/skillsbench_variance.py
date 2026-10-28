"""Estimate run-to-run variance and skill lift from published SkillsBench results.

Numbers from this script are quoted in docs/ml_system_design_doc.md. The inputs are not
stored in the repo. Fetch them at the pinned commits:

    curl -sLO https://raw.githubusercontent.com/benchflow-ai/skillsbench-trajectories/\
37484902dabffa24e91fac710ddcb90395fc5114/website-data/results-registry.json
    git clone --filter=blob:none --sparse https://github.com/benchflow-ai/skillsbench.git
    git -C skillsbench checkout f860ecf02ef53412a030f8661c83d644b739ff01
    git -C skillsbench sparse-checkout set --no-cone 'tasks/*/task.md'

Run:
    uv run python scripts/skillsbench_variance.py results-registry.json skillsbench/tasks
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

DOC_DATA = frozenset({"office-white-collar", "finance-economics"})
CATEGORY = re.compile(r"^\s+category:\s*(\S+)", re.MULTILINE)


def task_categories(tasks_dir: Path) -> dict[str, str]:
    result = {}
    for task_md in sorted(tasks_dir.glob("*/task.md")):
        match = CATEGORY.search(task_md.read_text(encoding="utf-8"))
        if match:
            result[task_md.parent.name] = match.group(1)
    return result


def cell_variance(passes: int, trials: int) -> float:
    """Unbiased estimate of p * (1 - p) from one cell of Bernoulli trials."""
    return passes * (trials - passes) / (trials * (trials - 1))


def summarize(records: list[dict[str, Any]], label: str) -> None:
    variances = [cell_variance(r["passCount"], r["trials"]) for r in records]
    stable = sum(r["passCount"] in (0, r["trials"]) for r in records)
    print(
        f"{label}: cells={len(records)}, mean p(1-p)={fmean(variances):.3f}, "
        f"cells with all runs equal={stable / len(records):.0%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SkillsBench variance and lift")
    parser.add_argument("registry", type=Path)
    parser.add_argument("tasks_dir", type=Path)
    args = parser.parse_args()

    categories = task_categories(args.tasks_dir)
    records: list[dict[str, Any]] = json.loads(args.registry.read_text(encoding="utf-8"))
    for record in records:
        record["category"] = categories.get(record["task"], "unknown")
    trials = {r["trials"] for r in records}
    print(f"records={len(records)}, tasks={len(categories)}, trials per cell={trials}")

    doc = [r for r in records if r["category"] in DOC_DATA]
    summarize(records, "all tasks")
    summarize(doc, "documents and data")
    for condition in ("No Skills", "With Skills"):
        summarize(
            [r for r in doc if r["condition"] == condition], f"documents and data, {condition}"
        )

    # Paired lift per configuration on document and data tasks.
    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in doc:
        by_cell[(r["model"], r["task"])][r["condition"]] = r["passCount"] / r["trials"]
    lifts: dict[str, list[float]] = defaultdict(list)
    per_task: dict[str, list[float]] = defaultdict(list)
    for (model, task), arms in by_cell.items():
        lift = arms["With Skills"] - arms["No Skills"]
        lifts[model].append(lift)
        per_task[task].append(lift)

    print("\nconfiguration: mean lift, sd of per-task lift, tasks with negative lift")
    for model, values in sorted(lifts.items(), key=lambda kv: -fmean(kv[1])):
        negative = sum(v < 0 for v in values)
        print(f"  {model}: {fmean(values):+.3f}, {pstdev(values):.3f}, {negative}/{len(values)}")

    # Same task, different models: does the sign of the lift depend on the model?
    mixed = sorted(t for t, v in per_task.items() if max(v) >= 2 / 3 and min(v) <= -2 / 3)
    print(f"\ntasks with lift >= +2/3 for one model and <= -2/3 for another: {len(mixed)}")
    print(f"  of {len(per_task)}: {', '.join(mixed)}")


if __name__ == "__main__":
    main()

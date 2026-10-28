"""Print the sensitivity table used in docs/ml_system_design_doc.md.

Run: uv run python -m skills_testbed.power
"""

from __future__ import annotations

from skills_testbed.stats import minimum_detectable_effect

# (scenarios, runs per scenario): one skill, a PR check, all positive scenarios, all scenarios.
GRID = [(6, 3), (12, 3), (48, 3), (60, 3)]
# 0.25 is the worst case. 0.074 is the mean p(1-p) on SkillsBench document and data
# tasks, see scripts/skillsbench_variance.py.
VARIANCES = (0.25, 0.074)


def table() -> str:
    head = "| Сценариев | Повторов | " + " | ".join(f"MDE при p(1-p)={v}" for v in VARIANCES) + " |"
    rule = "|---" * (2 + len(VARIANCES)) + "|"
    rows = [
        f"| {n} | {k} | "
        + " | ".join(f"{100 * minimum_detectable_effect(n, k, v):.0f} п.п." for v in VARIANCES)
        + " |"
        for n, k in GRID
    ]
    return "\n".join([head, rule, *rows])


if __name__ == "__main__":
    print(table())

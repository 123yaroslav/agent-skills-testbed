"""Turn the raw SkillsBench downloads into the small tables in data/processed.

Run scripts/fetch_eda_data.py first. The tables keep only what notebooks/eda.ipynb needs,
with no raw log text, so they are safe to commit.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from skills_testbed.trajectory import read_events, summarize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = RAW / "hf"
OUT = ROOT / "data" / "processed"
TASKS = RAW / "skillsbench" / "tasks"
# Same definition as in the design doc: tasks on documents and data.
DOC_DATA = {"office-white-collar", "finance-economics"}


def write(name: str, rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def frontmatter(text: str) -> str:
    parts = text.split("---", 2)
    return parts[1] if text.startswith("---") and len(parts) == 3 else ""


def scalar(meta: str, key: str) -> str:
    match = re.search(rf"^\s*{key}:\s*(.+)$", meta, re.MULTILINE)
    return match.group(1).strip().strip("'\"") if match else ""


def items(meta: str, key: str) -> str:
    match = re.search(rf"^\s*{key}:\s*\n((?:\s+- .+\n)+)", meta, re.MULTILINE)
    return "|".join(re.findall(r"- (\S+)", match.group(1))) if match else ""


def skill_files() -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for line in (RAW / "skillsbench_files.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split("/")
        if len(parts) > 4 and parts[2:4] == ["environment", "skills"]:
            counts[(parts[1], parts[4])] += 1
    return counts


def build_tasks_and_skills() -> set[str]:
    files = skill_files()
    tasks: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []
    for task_md in sorted(TASKS.glob("*/task.md")):
        task = task_md.parent.name
        meta = frontmatter(task_md.read_text(encoding="utf-8"))
        agent = re.search(r"^agent:\s*\n\s+timeout_sec:\s*([\d.]+)", meta, re.MULTILINE)
        task_skills = sorted(task_md.parent.glob("environment/skills/*/SKILL.md"))
        tasks.append(
            {
                "task": task,
                "category": scalar(meta, "category"),
                "difficulty": scalar(meta, "difficulty"),
                "modalities": items(meta, "modality"),
                "task_types": items(meta, "task_type"),
                "skill_types": items(meta, "skill_type"),
                "agent_timeout_s": float(agent.group(1)) if agent else "",
                "n_skills": len(task_skills),
                "doc_data": scalar(meta, "category") in DOC_DATA,
            }
        )
        for skill_md in task_skills:
            text = skill_md.read_text(encoding="utf-8")
            meta_s = frontmatter(text)
            body = text.split("---", 2)[2] if meta_s else text
            skill = skill_md.parent.name
            n_files = files[(task, skill)]
            skills.append(
                {
                    "task": task,
                    "skill": skill,
                    "description": scalar(meta_s, "description"),
                    "description_words": len(scalar(meta_s, "description").split()),
                    "body_words": len(body.split()),
                    "n_files": n_files,
                    "has_scripts": n_files > 1,
                }
            )
    write("tasks.csv", tasks)
    write("skills.csv", skills)
    return {t["task"] for t in tasks}


def number(value: str) -> float | str:
    return float(value) if value else ""


def build_trials(release: set[str]) -> dict[str, dict[str, str]]:
    raw = read_csv(HF / "analysis/pr8_pr2_pr3_selected/selected_trials.csv")
    rows = [r for r in raw if r["task"] in release]
    write(
        "trials.csv",
        [
            {
                "harness": r["harness"],
                "model": r["model"],
                "mode": r["mode"].removesuffix("-skills"),
                "task": r["task"],
                "slot": int(r["trial_slot"]),
                "status": r["status_for_analysis"],
                "reward": number(r["reward"]),
                "total_s": number(r["total_s"]),
                "events": number(r["trajectory_events"]),
                "has_tokens": bool(r["total_tokens"]),
            }
            for r in rows
        ],
    )
    return {r["dest_path"].replace("/v0.1/", "/v1.1/", 1): r for r in rows}


def build_claude_code(trials: dict[str, dict[str, str]]) -> None:
    rows = []
    for path, trial in sorted(trials.items()):
        if trial["harness"] != "claude-code":
            continue
        trial_dir = HF / path
        result = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
        timing = result.get("timing") or {}
        run = summarize(read_events(trial_dir / "trajectory" / "acp_trajectory.jsonl"))
        rows.append(
            {
                "task": trial["task"],
                "mode": trial["mode"].removesuffix("-skills"),
                "slot": int(trial["trial_slot"]),
                "status": trial["status_for_analysis"],
                "reward": number(trial["reward"]),
                "environment_setup_s": timing.get("environment_setup", ""),
                "agent_s": timing.get("agent_execution", ""),
                "verifier_s": timing.get("verifier", ""),
                "total_s": timing.get("total", ""),
                "n_tool_calls": run.n_tool_calls,
                "n_failed_calls": run.n_failed,
                "n_environment_errors": run.n_environment_errors,
                "skills_opened": "|".join(dict.fromkeys(run.skills)),
                "n_skill_calls": len(run.skills),
                "first_skill_call": "" if run.first_skill_call is None else run.first_skill_call,
                "run_error": bool(result.get("error")),
                "partial_log": bool(result.get("partial_trajectory")),
            }
        )
    write("claude_code_runs.csv", rows)


def build_audit() -> None:
    audit = json.loads(
        (HF / "analysis/skillsbench/v1.1/official-selected/failure_audit_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    write(
        "failure_audit.csv",
        [
            {
                "model": r["model"],
                "mode": r["mode"],
                "task": r["task"],
                "fail_mode": r["fail_mode"],
                "verdict": r["verdict"],
                "category": r["category"],
            }
            for r in audit["rows"]
        ],
    )


def build_small_tables() -> None:
    board = read_csv(HF / "leaderboard/skillsbench/v1.1/official.csv")
    keep = ("model", "agent", "with_skills_score", "without_skills_score", "skill_lift")
    write("leaderboard.csv", [{k: r[k] for k in (*keep, "coverage_status")} for r in board])
    use = read_csv(HF / "analysis/opus48_pr11_rerun_20260611/skill_invocation_rate.csv")
    write(
        "openhands_skill_use.csv",
        [
            {
                "task": r["task"],
                "reward": number(r["reward"]),
                "n_task_skills": len([s for s in r["loaded_task_skills"].split(",") if s]),
                "n_skill_calls": int(r["n_skill_invocations"]),
                "opened_skill": r["invoked_any_skill"] == "True",
            }
            for r in use
        ],
    )


def main() -> None:
    release = build_tasks_and_skills()
    trials = build_trials(release)
    build_claude_code(trials)
    build_audit()
    build_small_tables()


if __name__ == "__main__":
    main()

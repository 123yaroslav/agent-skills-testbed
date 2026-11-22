"""Run the failure cause classifier, baseline and MVP, on the failures audited by SkillsBench.

The baseline is rules only. The MVP asks an LLM through the Claude Code CLI in headless mode,
answers are cached in data/processed/failure_llm_cache.jsonl, so a rerun costs nothing.

    uv run python scripts/classify_failures.py --split tune
    uv run python scripts/classify_failures.py --split test

Writes data/processed/failure_predictions.csv, which notebooks/models.ipynb reads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from skills_testbed.failure_cause import (
    SYSTEM_PROMPT,
    FailedRun,
    Verdict,
    baseline,
    build_prompt,
    parse_answer,
    sure_rules,
)
from skills_testbed.splits import split_of
from skills_testbed.trajectory import read_events

ROOT = Path(__file__).resolve().parents[1]
HF = ROOT / "data" / "raw" / "hf"
AUDIT = HF / "analysis/skillsbench/v1.1/official-selected/failure_audit_aggregate.json"
OUT = ROOT / "data" / "processed"
CACHE = OUT / "failure_llm_cache.jsonl"
LOCK = threading.Lock()


def load_runs() -> list[tuple[dict[str, Any], FailedRun]]:
    """Audited failures that have a log. The 12 excluded timeouts have no files at all."""
    rows = json.loads(AUDIT.read_text(encoding="utf-8"))["rows"]
    runs = []
    for row in rows:
        trial = HF / row["path"]
        if not (trial / "result.json").exists():
            continue
        run = FailedRun(
            trial=trial.name,
            with_skills=row["mode"] == "with",
            result=json.loads((trial / "result.json").read_text(encoding="utf-8")),
            events=read_events(trial / "trajectory" / "acp_trajectory.jsonl"),
        )
        runs.append((row, run))
    return runs


def read_cache() -> dict[str, dict[str, Any]]:
    if not CACHE.exists():
        return {}
    lines = CACHE.read_text(encoding="utf-8").splitlines()
    return {d["prompt_hash"]: d for d in (json.loads(line) for line in lines if line)}


def ask_claude(model: str, system: str, prompt: str) -> tuple[str, float]:
    """One headless Claude Code call without tools, settings or MCP servers."""
    result = subprocess.run(  # noqa: S603 (fixed arguments, prompt goes to stdin)
        [  # noqa: S607 (claude from PATH)
            "claude",
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--tools",
            "",
            "--system-prompt",
            system,
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    data = json.loads(result.stdout)
    return str(data.get("result", "")), float(data.get("total_cost_usd") or 0.0)


def classify(run: FailedRun, model: str, cache: dict[str, dict[str, Any]]) -> tuple[Verdict, float]:
    rule = sure_rules(run)
    if rule is not None:
        return rule, 0.0
    prompt = build_prompt(run)
    key = hashlib.sha256(f"{model}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()
    hit = cache.get(key)
    if hit is None:
        answer, cost = ask_claude(model, SYSTEM_PROMPT, prompt)
        hit = {
            "prompt_hash": key,
            "trial": run.trial,
            "model": model,
            "answer": answer,
            "cost": cost,
        }
        with LOCK, CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(hit, ensure_ascii=False) + "\n")
        cache[key] = hit
    return parse_answer(str(hit["answer"]), prompt), float(hit["cost"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tune", "test", "all"], default="tune")
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    runs = [(row, run) for row, run in load_runs() if args.split in ("all", split_of(row["task"]))]
    cache = read_cache()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(lambda item: classify(item[1], args.model, cache), runs))

    out = OUT / f"failure_predictions_{args.model}.csv"
    old = {}
    if out.exists():
        with out.open(encoding="utf-8", newline="") as f:
            old = {r["trial"]: r for r in csv.DictReader(f)}
    for (row, run), (verdict, cost) in zip(runs, verdicts, strict=True):
        base = baseline(run)
        old[run.trial] = {
            "trial": run.trial,
            "task": row["task"],
            "agent_model": row["model"],
            "mode": row["mode"],
            "split": split_of(row["task"]),
            "audit_verdict": row["verdict"],
            "audit_category": row["category"],
            "baseline_cause": base.cause.value,
            "baseline_rule": base.source,
            "mvp_cause": verdict.cause.value,
            "mvp_source": verdict.source,
            "mvp_quote": verdict.quote[:200],
            "llm_cost_usd": f"{cost:.6f}",
        }
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(old.values()))))
        writer.writeheader()
        writer.writerows(sorted(old.values(), key=lambda r: (r["split"], r["task"], r["trial"])))
    print(f"{len(runs)} runs, LLM cost {sum(c for _, c in verdicts):.4f} USD, wrote {out}")


if __name__ == "__main__":
    main()

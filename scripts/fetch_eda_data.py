"""Download the public SkillsBench data used in notebooks/eda.ipynb.

Sources are pinned, so data/processed can be rebuilt exactly. Raw files go to data/raw,
which is not tracked by git. Run:

    uv run python scripts/fetch_eda_data.py
    uv run python scripts/build_eda_tables.py
"""

from __future__ import annotations

import csv
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Hugging Face dataset with SkillsBench runs and audits, Apache-2.0.
HF_REPO = "benchflow/skillsbench-leaderboard"
HF_REV = "f104580363a9642563593c475620196ecd36687d"
HF_FILES = (
    "analysis/pr8_pr2_pr3_selected/selected_trials.csv",
    "analysis/skillsbench/v1.1/official-selected/failure_audit_aggregate.json",
    "analysis/opus48_pr11_rerun_20260611/skill_invocation_rate.csv",
    "leaderboard/skillsbench/v1.1/official.csv",
)
# Per-trial files for the Claude Code runs. Only Claude Code is the agent of the testbed.
TRIAL_FILES = ("result.json", "timing.json", "trajectory/acp_trajectory.jsonl")

# SkillsBench tasks and skills, release v1.1, Apache-2.0.
SB_REPO = "https://github.com/benchflow-ai/skillsbench.git"
SB_REV = "27738384b1df694ea2ae466e416f476e94d8fab9"
SB_SPARSE = ("tasks/*/task.md", "tasks/*/environment/skills/*/SKILL.md", "/taxonomy.yaml")


def hf_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REV}/{path}"


def download(url: str, dest: Path, attempts: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https url {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (https only, checked above)
                data = response.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def claude_code_trials(selected_trials: Path) -> list[str]:
    """Paths of Claude Code trials in the HF dataset.

    The CSV keeps the old v0.1 prefix, the files now live under v1.1.
    """
    with selected_trials.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["harness"] == "claude-code"]
    return [r["dest_path"].replace("/v0.1/", "/v1.1/", 1) for r in rows]


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(  # noqa: S603 (fixed git arguments, no shell)
        ["git", *args],  # noqa: S607 (git from PATH)
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def fetch_skillsbench() -> None:
    repo = RAW / "skillsbench"
    if not repo.exists():
        git("clone", "--quiet", "--filter=blob:none", "--no-checkout", SB_REPO, str(repo))
    git("sparse-checkout", "set", "--no-cone", *SB_SPARSE, cwd=repo)
    git("checkout", "--quiet", SB_REV, cwd=repo)
    # Full file list of the release, blobs are not downloaded for it.
    files = git("ls-tree", "-r", "--name-only", SB_REV, "tasks", cwd=repo)
    (RAW / "skillsbench_files.txt").write_text(files, encoding="utf-8")


def main() -> None:
    for path in HF_FILES:
        download(hf_url(path), RAW / "hf" / path)
    trials = claude_code_trials(RAW / "hf" / HF_FILES[0])
    jobs = [(hf_url(f"{t}/{f}"), RAW / "hf" / t / f) for t in trials for f in TRIAL_FILES]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda job: download(*job), jobs))
    print(f"downloaded {len(HF_FILES)} tables and {len(trials)} Claude Code trials")
    fetch_skillsbench()
    print(f"skillsbench at {SB_REV[:7]} in {RAW / 'skillsbench'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skills_testbed.trajectory import read_events, skills_opened, summarize, tool_calls


def call(title: str, text: str, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "tool_call",
        "title": title,
        "kind": "other",
        "status": status,
        "content": [{"type": "content", "content": {"type": "text", "text": text}}],
    }


EVENTS = [
    {"type": "user_message", "text": "Fill the report"},
    call("Terminal", "ls /root"),
    call("Skill", "Launching skill: xlsx"),
    call("Terminal", "EACCES: permission denied, mkdir '/home/agent/.claude'", status="failed"),
    call("Terminal", "Traceback (most recent call last)", status="failed"),
    call("Skill", "Launching skill: pdf"),
    {"type": "agent_message", "text": "Done"},
]


def test_skills_in_order() -> None:
    assert skills_opened(tool_calls(EVENTS)) == ["xlsx", "pdf"]


def test_summary_counts() -> None:
    summary = summarize(EVENTS)
    assert summary.n_tool_calls == 5
    assert summary.n_failed == 2
    assert summary.n_environment_errors == 1
    assert summary.first_skill_call == 1
    assert summary.opened_skill


def test_no_skill() -> None:
    summary = summarize([call("Terminal", "ls")])
    assert summary.first_skill_call is None
    assert not summary.opened_skill


def test_diff_content_is_read(tmp_path: Path) -> None:
    edit = {
        "type": "tool_call",
        "title": "Write",
        "status": "completed",
        "content": [{"type": "diff", "path": "/root/a.json", "newText": "{}"}],
    }
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps(edit) + "\n\n", encoding="utf-8")
    assert tool_calls(read_events(path))[0].text == "{}"

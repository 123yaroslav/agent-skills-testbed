"""Read agent logs in the ACP format (claude-agent-acp), as published by SkillsBench.

The testbed needs three things from a log. Which skills the agent opened, which tool calls
failed, and whether a failure looks like a broken environment rather than an agent mistake.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKILL_LAUNCH = re.compile(r"Launching skill: (\S+)")
# Tool errors caused by the container, not by the agent. Kept short on purpose: this is the
# rule baseline of the failure classifier, the LLM handles everything else.
ENVIRONMENT_ERRORS = re.compile(
    r"EACCES|permission denied|command not found|Exit code 127|is not installed",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolCall:
    title: str
    kind: str
    status: str
    text: str


@dataclass(frozen=True)
class RunSummary:
    n_tool_calls: int
    n_failed: int
    n_environment_errors: int
    skills: tuple[str, ...]
    first_skill_call: int | None  # index among tool calls, None if no skill was opened

    @property
    def opened_skill(self) -> bool:
        return bool(self.skills)


def read_events(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _text(content: object) -> str:
    """Collect text from the nested content blocks of a tool call."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_text(item) for item in content)
    if isinstance(content, dict):
        parts = [content.get("text"), content.get("content"), content.get("newText")]
        return "\n".join(_text(p) for p in parts if p)
    return ""


def tool_calls(events: Iterable[dict[str, Any]]) -> list[ToolCall]:
    return [
        ToolCall(
            title=str(e.get("title", "")),
            kind=str(e.get("kind", "")),
            status=str(e.get("status", "")),
            text=_text(e.get("content")),
        )
        for e in events
        if e.get("type") == "tool_call"
    ]


def skills_opened(calls: Iterable[ToolCall]) -> list[str]:
    """Skill names in the order the agent launched them, repeats kept."""
    names = []
    for call in calls:
        if call.title == "Skill":
            names.extend(SKILL_LAUNCH.findall(call.text))
    return names


def summarize(events: Iterable[dict[str, Any]]) -> RunSummary:
    calls = tool_calls(events)
    failed = [c for c in calls if c.status == "failed"]
    first = next((i for i, c in enumerate(calls) if c.title == "Skill"), None)
    return RunSummary(
        n_tool_calls=len(calls),
        n_failed=len(failed),
        n_environment_errors=sum(bool(ENVIRONMENT_ERRORS.search(c.text)) for c in failed),
        skills=tuple(skills_opened(calls)),
        first_skill_call=first,
    )

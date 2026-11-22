"""Name the cause of a failed agent run.

The baseline is a set of rules over the log. The MVP keeps the rules that are almost never
wrong and sends everything else to an LLM. The LLM must back its answer with a quote from
the log. An answer whose quote is not in the log becomes "unsure" and goes to a human.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from skills_testbed.trajectory import ENVIRONMENT_ERRORS, ToolCall, skills_opened, tool_calls


class Cause(StrEnum):
    SKILL_NOT_OPENED = "skill_not_opened"
    WRONG_SKILL = "wrong_skill"
    SKIPPED_STEP = "skipped_step"
    SKILL_BUG = "skill_bug"
    AGENT_ERROR = "agent_error"
    INFRASTRUCTURE = "infrastructure"
    SCENARIO_ERROR = "scenario_error"
    TAMPERING = "tampering"  # the agent tried to read or change the checker
    UNSURE = "unsure"


# What the stand does with each cause. Infrastructure failures are rerun and do not count in
# the comparison, the human queue gets everything a person has to look at.
Action = str
ACTION: dict[Cause, Action] = {
    Cause.INFRASTRUCTURE: "rerun",
    Cause.SCENARIO_ERROR: "human",
    Cause.TAMPERING: "human",
    Cause.UNSURE: "human",
}


def action(cause: Cause) -> Action:
    return ACTION.get(cause, "count")


@dataclass(frozen=True)
class FailedRun:
    trial: str  # trial directory name, the stand writes it before the run starts
    with_skills: bool
    result: Mapping[str, Any]  # result.json of the run
    events: Sequence[Mapping[str, Any]]  # the ACP log


@dataclass(frozen=True)
class Verdict:
    cause: Cause
    source: str  # which rule or "llm"
    quote: str = ""


def _skill_invocations(run: FailedRun, calls: Sequence[ToolCall]) -> int:
    return max(int(run.result.get("n_skill_invocations") or 0), len(skills_opened(calls)))


def baseline(run: FailedRun) -> Verdict:
    """Rules from the design doc, section 2.3, stage 2."""
    calls = tool_calls(run.events)
    if run.result.get("partial_trajectory"):
        return Verdict(Cause.INFRASTRUCTURE, "partial_log")
    if not run.with_skills and _skill_invocations(run, calls):
        return Verdict(Cause.INFRASTRUCTURE, "skill_leak")
    if any(c.status == "failed" and ENVIRONMENT_ERRORS.search(c.text) for c in calls):
        return Verdict(Cause.INFRASTRUCTURE, "environment_error")
    if run.result.get("verifier_error"):
        return Verdict(Cause.SCENARIO_ERROR, "verifier_error")
    if run.with_skills and not _skill_invocations(run, calls):
        return Verdict(Cause.SKILL_NOT_OPENED, "no_skill_call")
    return Verdict(Cause.AGENT_ERROR, "default")


def sure_rules(run: FailedRun) -> Verdict | None:
    """The MVP keeps only rules that do not need judgment. None means ask the LLM."""
    calls = tool_calls(run.events)
    if run.result.get("partial_trajectory"):
        return Verdict(Cause.INFRASTRUCTURE, "partial_log")
    if not run.with_skills and _skill_invocations(run, calls):
        return Verdict(Cause.INFRASTRUCTURE, "skill_leak")
    rollout = run.result.get("rollout_name")
    if rollout and rollout != run.trial:
        return Verdict(Cause.INFRASTRUCTURE, "provenance_mismatch")
    if run.result.get("verifier_error"):
        return Verdict(Cause.SCENARIO_ERROR, "verifier_error")
    return None


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _clean(text: str, limit: int) -> str:
    text = ANSI.sub("", text).replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) <= limit else text[:limit] + " [...]"


def compress(run: FailedRun, head: int = 15, tail: int = 60) -> str:
    """Short text view of a log. Tool calls with errors and the last messages, no file bodies."""
    calls = tool_calls(run.events)
    skills = skills_opened(calls)
    lines = [
        f"mode: {'with skills' if run.with_skills else 'without skills'}",
        f"reward: {run.result.get('rewards', {}).get('reward')}",
        f"tool calls: {len(calls)}, failed: {sum(c.status == 'failed' for c in calls)}",
        f"skills opened: {', '.join(skills) if skills else 'none'}",
        "",
        "TOOL CALLS",
    ]
    picked: list[tuple[int, ToolCall | None]] = list(enumerate(calls))
    if len(picked) > head + tail:
        omitted = len(picked) - head - tail
        picked = [*picked[:head], (-1, None), *picked[-tail:]]
    else:
        omitted = 0
    for i, c in picked:
        if c is None:
            lines.append(f"... {omitted} calls omitted ...")
            continue
        lines.append(f"[{i}] {c.kind} {c.status}: {_clean(c.title, 160)}")
        if c.status == "failed" or ENVIRONMENT_ERRORS.search(c.text):
            lines.append("    output: " + _clean(c.text, 400).replace("\n", "\n    "))
    messages = [
        e
        for e in run.events
        if e.get("type") in ("agent_message", "agent_thought") and e.get("text")
    ]
    lines += ["", "LAST AGENT MESSAGES"]
    lines += [f"({e['type']}) {_clean(str(e['text']), 700)}" for e in messages[-3:]]
    last = run.events[-1].get("type") if run.events else "none"
    lines.append(f"\nlog ends with: {last}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You review failed runs of an AI agent that solves a task in a container.
The task checker already said the run failed. Name the main cause from this list.

skill_not_opened: skills were available, the agent never opened one and the failure is about \
what the skill covers.
wrong_skill: the agent opened a skill made for a different kind of task and followed it.
skipped_step: the agent opened the right skill but skipped or changed a step it prescribes.
skill_bug: the agent followed the skill and the skill's own instructions or scripts are wrong.
agent_error: the agent worked to the end in a working environment and the result is wrong \
or incomplete. This is the usual case, a failed command that the agent fixed or worked \
around is still agent_error.
infrastructure: the run was broken from outside. The model provider cut answers short \
(messages end mid-word, tool calls repeatedly miss required arguments, API errors such as \
503), the log stops abruptly, or the container blocked work the agent could not route \
around (no write access to the output folder, a required program missing).
scenario_error: the task or checker itself is broken, the output looks right but the \
checker crashes or demands something the task never asked for.
tampering: the agent tried to read, run or change the checker or the expected answers.
unsure: the log does not show enough to decide.

Answer with one JSON object and nothing else:
{"cause": "<one name from the list>", "quote": "<exact text copied from the log, at most \
200 characters, that shows the cause>", "reason": "<one short sentence>"}"""


def build_prompt(run: FailedRun) -> str:
    return "Failed run log:\n\n" + compress(run)


Ask = Callable[[str, str], str]  # (system prompt, user prompt) -> model text


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", ANSI.sub("", text)).strip().lower()


def parse_answer(answer: str, log_text: str) -> Verdict:
    """Read the model answer. Unknown cause, bad JSON or a quote not in the log give UNSURE."""
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if not match:
        return Verdict(Cause.UNSURE, "llm_bad_json")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Verdict(Cause.UNSURE, "llm_bad_json")
    try:
        cause = Cause(str(data.get("cause", "")).strip())
    except ValueError:
        return Verdict(Cause.UNSURE, "llm_unknown_cause")
    quote = str(data.get("quote", "")).strip()
    if cause is not Cause.UNSURE and (
        len(quote) < 8 or _normalize(quote) not in _normalize(log_text)
    ):
        return Verdict(Cause.UNSURE, "llm_quote_not_found", quote)
    return Verdict(cause, "llm", quote)


def mvp(run: FailedRun, ask: Ask) -> Verdict:
    rule = sure_rules(run)
    if rule is not None:
        return rule
    prompt = build_prompt(run)
    return parse_answer(ask(SYSTEM_PROMPT, prompt), prompt)

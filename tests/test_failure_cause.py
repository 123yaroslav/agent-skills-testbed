from __future__ import annotations

from typing import Any

from skills_testbed.failure_cause import (
    Cause,
    FailedRun,
    action,
    baseline,
    build_prompt,
    mvp,
    parse_answer,
    sure_rules,
)


def call(title: str, text: str, status: str = "completed", kind: str = "execute") -> dict[str, Any]:
    return {
        "type": "tool_call",
        "title": title,
        "kind": kind,
        "status": status,
        "content": [{"type": "content", "content": {"type": "text", "text": text}}],
    }


SKILL = call("Use the xlsx skill", "Tool: invoke_skill\nResult:\n[skill: xlsx]\n", kind="skill")
DONE = {"type": "agent_message", "text": "I wrote the report to /root/output/report.xlsx"}


def run(
    events: list[dict[str, Any]],
    with_skills: bool = True,
    trial: str = "task__abc",
    **result: Any,
) -> FailedRun:
    return FailedRun(trial, with_skills, {"rollout_name": trial, **result}, events)


def test_baseline_rules_in_order() -> None:
    assert baseline(run([call("ls", "x")], partial_trajectory=True)).cause is Cause.INFRASTRUCTURE
    assert baseline(run([SKILL], with_skills=False)).source == "skill_leak"
    denied = call("mkdir", "mkdir: permission denied", status="failed")
    assert baseline(run([SKILL, denied])).source == "environment_error"
    assert baseline(run([call("ls", "x"), DONE])).cause is Cause.SKILL_NOT_OPENED
    assert baseline(run([SKILL, DONE])).cause is Cause.AGENT_ERROR


def test_sure_rules_leave_judgment_to_the_llm() -> None:
    denied = call("mkdir", "mkdir: permission denied", status="failed")
    assert sure_rules(run([SKILL, denied])) is None
    mismatch = FailedRun("task__abc", True, {"rollout_name": "task__def"}, [SKILL])
    verdict = sure_rules(mismatch)
    assert verdict is not None
    assert verdict.source == "provenance_mismatch"


def test_parse_answer_needs_a_quote_from_the_log() -> None:
    log = "[3] execute failed: write report\n    output: Permission denied: /root/output"
    good = '{"cause": "infrastructure", "quote": "Permission denied: /root/output", "reason": "x"}'
    made_up = '{"cause": "infrastructure", "quote": "disk is full on the host", "reason": "x"}'
    assert parse_answer(good, log).cause is Cause.INFRASTRUCTURE
    assert parse_answer(made_up, log).cause is Cause.UNSURE
    assert parse_answer("no json here", log).source == "llm_bad_json"
    assert parse_answer('{"cause": "aliens", "quote": "x"}', log).source == "llm_unknown_cause"


def test_mvp_asks_the_llm_only_when_rules_are_silent() -> None:
    asked: list[str] = []

    def ask(system: str, prompt: str) -> str:
        asked.append(system + prompt)
        return '{"cause": "agent_error", "quote": "I wrote the report", "reason": "wrong"}'

    assert mvp(run([SKILL, DONE]), ask).cause is Cause.AGENT_ERROR
    assert len(asked) == 1
    assert mvp(run([SKILL], partial_trajectory=True), ask).cause is Cause.INFRASTRUCTURE
    assert len(asked) == 1


def test_prompt_shows_failed_output_and_last_message() -> None:
    denied = call("mkdir out", "mkdir: permission denied", status="failed")
    prompt = build_prompt(run([SKILL, denied, DONE]))
    assert "skills opened: xlsx" in prompt
    assert "permission denied" in prompt
    assert "I wrote the report" in prompt


def test_actions() -> None:
    assert action(Cause.INFRASTRUCTURE) == "rerun"
    assert action(Cause.UNSURE) == "human"
    assert action(Cause.SKIPPED_STEP) == "count"

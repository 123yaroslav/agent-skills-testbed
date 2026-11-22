import itertools

import pytest

from skills_testbed.regression import (
    Sequential,
    baseline_blocks,
    permutation_pvalue,
    success_rate,
)


def brute_force_pvalue(old: list[list[int]], new: list[list[int]]) -> float:
    """Enumerate every within-scenario relabeling, the definition of the exact test."""
    per_scenario = []
    for o, n in zip(old, new, strict=True):
        runs = o + n
        splits = [
            sum(runs[i] for i in idx) for idx in itertools.combinations(range(len(runs)), len(o))
        ]
        per_scenario.append(splits)
    observed = sum(sum(o) for o in old)
    combos = list(itertools.product(*per_scenario))
    return sum(sum(c) >= observed for c in combos) / len(combos)


def test_success_rate() -> None:
    assert success_rate([[1, 0, 1], [0, 0, 0]]) == pytest.approx(2 / 6)


def test_baseline_blocks_only_on_a_drop_above_threshold() -> None:
    old = [[1]] * 10
    assert not baseline_blocks(old, [[1]] * 9 + [[0]])  # 10 p.p. is not more than 10 p.p.
    assert baseline_blocks(old, [[1]] * 8 + [[0]] * 2)
    assert not baseline_blocks(old, [[1]] * 10)
    assert not baseline_blocks([[1]] * 10 + [[0]] * 10, [[1]] * 9 + [[0]] * 11)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ([[1, 1, 1], [1, 0, 1]], [[0, 0, 1], [0, 1, 0]]),
        ([[1, 1, 0], [0, 0, 0], [1, 1, 1]], [[1, 0, 0], [0, 0, 0], [1, 1, 0]]),
        ([[1, 1], [0, 1]], [[1, 1, 1], [0, 0, 0]]),
    ],
)
def test_pvalue_matches_brute_force(old: list[list[int]], new: list[list[int]]) -> None:
    assert permutation_pvalue(old, new) == pytest.approx(brute_force_pvalue(old, new))


def test_same_results_give_pvalue_one_when_nothing_varies() -> None:
    assert permutation_pvalue([[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]) == 1.0


def test_clear_drop_is_significant() -> None:
    old = [[1, 1, 1]] * 6
    new = [[0, 0, 0]] * 6
    assert permutation_pvalue(old, new) == pytest.approx(1 / 20**6)


def test_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError, match="same"):
        permutation_pvalue([[1]], [[1], [0]])
    with pytest.raises(ValueError, match="0 or 1"):
        permutation_pvalue([[2]], [[1]])


def test_sequential_rule() -> None:
    rule = Sequential(alpha1=0.01, gray=0.2, alpha2=0.03)
    assert rule.stage_one(0.005) == "block"
    assert rule.stage_one(0.1) == "more"
    assert rule.stage_one(0.5) == "pass"
    assert rule.stage_two(0.02) == "block"
    assert rule.stage_two(0.04) == "pass"

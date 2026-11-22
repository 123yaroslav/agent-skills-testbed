"""Decide whether a new skill version is worse than the old one.

Both versions run on the same scenarios, every run is a success (1) or a failure (0).
The baseline looks at one run per scenario and blocks on a plain drop in success rate.
The MVP runs each scenario several times and uses an exact permutation test that only
swaps version labels inside a scenario, so differences between scenarios never count as
an effect of the skill.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

Runs = Sequence[Sequence[int]]  # outcomes per scenario, 0 or 1 per run


def success_rate(runs: Runs) -> float:
    total = sum(len(r) for r in runs)
    if total == 0:
        raise ValueError("no runs")
    return sum(sum(r) for r in runs) / total


def baseline_blocks(old: Runs, new: Runs, max_drop: float = 0.10) -> bool:
    """Baseline rule from the design doc. Block if success rate fell by more than max_drop."""
    return success_rate(old) - success_rate(new) > max_drop


def _check(old: Runs, new: Runs) -> None:
    if len(old) != len(new) or not old:
        raise ValueError("old and new need the same non-empty list of scenarios")
    for o, n in zip(old, new, strict=True):
        if not o or not n:
            raise ValueError("every scenario needs runs of both versions")
        if any(x not in (0, 1) for x in (*o, *n)):
            raise ValueError("outcomes must be 0 or 1")


def _hypergeom(n_old: int, n_new: int, successes: int) -> list[float]:
    """Distribution of old-version successes in one scenario if versions are the same."""
    total = comb(n_old + n_new, n_old)
    return [
        comb(successes, x) * comb(n_old + n_new - successes, n_old - x) / total
        for x in range(n_old + 1)
    ]


def _convolve(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, pa in enumerate(a):
        if pa:
            for j, pb in enumerate(b):
                out[i + j] += pa * pb
    return out


def permutation_pvalue(old: Runs, new: Runs) -> float:
    """One-sided exact p-value of "new is worse", labels permuted within scenarios.

    The statistic is the number of old-version successes. Inside a scenario all ways to
    split its runs into old and new are equally likely under the null, so the count follows
    a hypergeometric law there, and the total is a convolution over scenarios.
    """
    _check(old, new)
    dist = [1.0]
    observed = 0
    for o, n in zip(old, new, strict=True):
        dist = _convolve(dist, _hypergeom(len(o), len(n), sum(o) + sum(n)))
        observed += sum(o)
    return min(1.0, sum(dist[observed:]))


@dataclass(frozen=True)
class Sequential:
    """Two-stage rule. Run k1 times, add runs only when the first p-value is borderline.

    Block at stage one if p <= alpha1, stop and pass if p > gray. Otherwise every scenario
    gets more runs of both versions and the check blocks if the new p-value is <= alpha2.
    The three thresholds are tuned together on A/A checks, so the false block rate of the
    whole procedure, add-on runs included, stays under the target.
    """

    alpha1: float
    gray: float
    alpha2: float

    def stage_one(self, p: float) -> str:
        if p <= self.alpha1:
            return "block"
        if p > self.gray:
            return "pass"
        return "more"

    def stage_two(self, p: float) -> str:
        return "block" if p <= self.alpha2 else "pass"

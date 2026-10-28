import pytest

from skills_testbed.stats import minimum_detectable_effect, pass_hat_k


def test_pass_hat_k_edge_cases() -> None:
    assert pass_hat_k(n=5, c=5, k=3) == 1.0
    assert pass_hat_k(n=5, c=2, k=3) == 0.0
    assert pass_hat_k(n=4, c=3, k=1) == pytest.approx(0.75)


def test_pass_hat_k_matches_hand_count() -> None:
    # 4 runs, 3 successes, pick 2: 3 of the 6 pairs are both successes.
    assert pass_hat_k(n=4, c=3, k=2) == pytest.approx(0.5)


@pytest.mark.parametrize(("n", "c", "k"), [(3, 4, 1), (3, 1, 0), (3, 1, 4)])
def test_pass_hat_k_rejects_bad_input(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError, match="need"):
        pass_hat_k(n, c, k)


def test_mde_known_value() -> None:
    # 100 scenarios, 1 run, worst-case variance: (1.645 + 0.842) * sqrt(0.5 / 100).
    assert minimum_detectable_effect(100, 1) == pytest.approx(0.1759, abs=1e-3)


def test_mde_shrinks_with_more_runs() -> None:
    few = minimum_detectable_effect(10, 3)
    many = minimum_detectable_effect(10, 12)
    assert many == pytest.approx(few / 2)

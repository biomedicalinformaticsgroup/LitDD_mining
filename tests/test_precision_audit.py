"""Tests for the precision-audit statistics: Wilson interval and Cohen's kappa."""
from __future__ import annotations

import math

from litdd.evaluation.common import wilson_ci
from litdd.evaluation.precision_audit import score_audit


def test_wilson_ci_brackets_point_estimate():
    lo, hi = wilson_ci(84, 100)
    assert lo < 0.84 < hi
    assert 0 <= lo < hi <= 1
    # ~known Wilson interval for 84/100 is roughly 0.754-0.901
    assert math.isclose(lo, 0.754, abs_tol=0.01)
    assert math.isclose(hi, 0.901, abs_tol=0.01)


def test_wilson_ci_edges():
    assert wilson_ci(0, 0) != wilson_ci(0, 0)  # NaN for n=0 (NaN != NaN)
    lo, hi = wilson_ci(10, 10)  # all successes
    assert hi <= 1.0 and lo < 1.0


def test_cohen_kappa_perfect_and_chance():
    assert score_audit.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # total disagreement on a 2-class balanced split -> negative kappa
    assert score_audit.cohen_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"]) < 0


def test_cohen_kappa_partial():
    a = ["correct"] * 9 + ["incorrect"]
    b = ["correct"] * 8 + ["incorrect", "incorrect"]
    k = score_audit.cohen_kappa(a, b)
    assert 0 < k < 1


def test_cohen_kappa_empty():
    assert score_audit.cohen_kappa([], []) != score_audit.cohen_kappa([], [])  # NaN

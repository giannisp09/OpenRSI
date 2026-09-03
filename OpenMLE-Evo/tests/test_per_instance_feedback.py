"""Regression tests for the per-instance operator feedback breakdown.

The eval service returns a per-instance improvement map alongside the scalar
``aggregate_improvement``. Only the scalar used to reach the operator prompts,
so once most instances of a multi-instance task saturated, the search lost any
signal about which instance still had headroom.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

BASE_TASK = (
    Path(__file__).resolve().parents[1]
    / "third_party/aira-evo/examples/nature_bench/base_task.py"
)


def _load_renderer():
    """Extract the renderer without importing the heavy task module."""
    lines = BASE_TASK.read_text(encoding="utf-8").splitlines(True)
    start = next(
        i for i, l in enumerate(lines) if l.startswith("    def _per_instance_feedback")
    )
    end = next(
        i
        for i, l in enumerate(lines[start + 1 :], start + 1)
        if l.startswith("    def _annotate_eval_payload")
    )
    source = (
        "class _Stub:\n"
        "    def __init__(self, cfg): self.cfg = cfg\n"
        "    @staticmethod\n"
        "    def _coerce_score(value):\n"
        "        try: return float(value)\n"
        "        except (TypeError, ValueError): return None\n"
    ) + "".join(lines[start:end])
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(source, str(BASE_TASK), "exec"), namespace)
    return namespace["_Stub"]


@pytest.fixture(scope="module")
def renderer():
    return _load_renderer()


SATURATED = {
    "per_instance_improvement": {
        "german_credit": 0.021972,
        "ist_aspirin": 0.119695,
        "ist_heparin": 0.124562,
        "twin_mortality": -0.231296,
    },
    "raw_scores": {
        "german_credit": {"Counterfactual F1-Score": 1.0},
        "ist_aspirin": {"Counterfactual F1-Score": 1.0},
        "ist_heparin": {"Counterfactual F1-Score": 0.998724},
        "twin_mortality": {"Counterfactual F1-Score": 0.638024},
    },
}


def test_names_the_weakest_instance(renderer):
    text = renderer({})._per_instance_feedback(SATURATED)
    assert "twin_mortality" in text
    assert "weakest, most remaining headroom" in text
    # Exactly one instance is marked, otherwise the hint carries no information.
    assert text.count("weakest, most remaining headroom") == 1


def test_reports_every_instance_with_its_metrics(renderer):
    text = renderer({})._per_instance_feedback(SATURATED)
    for name in SATURATED["per_instance_improvement"]:
        assert name in text
    assert "Counterfactual F1-Score=0.638024" in text


def test_disabled_by_config_for_ablation(renderer):
    assert renderer({"per_instance_feedback": False})._per_instance_feedback(
        SATURATED
    ) == ""


def test_uniform_scores_add_nothing_over_the_aggregate(renderer):
    """A crashed candidate reports the same floor everywhere; the traceback is
    the useful feedback, so the breakdown must stay out of the debug prompt."""
    crashed = {
        "per_instance_improvement": {name: -1.0 for name in ("a", "b", "c", "d")},
        "raw_scores": {},
    }
    assert renderer({})._per_instance_feedback(crashed) == ""


def test_single_instance_task_is_skipped(renderer):
    assert renderer({})._per_instance_feedback(
        {"per_instance_improvement": {"only": -0.5}}
    ) == ""


def test_unscored_instance_does_not_break_rendering(renderer):
    payload = {
        "per_instance_improvement": {"a": -0.5, "b": 0.2, "c": None},
        "raw_scores": {"c": {"error": "boom", "Counterfactual F1-Score": None}},
    }
    text = renderer({})._per_instance_feedback(payload)
    assert "c: not scored" in text
    assert "boom" not in text  # error strings belong in the run traceback

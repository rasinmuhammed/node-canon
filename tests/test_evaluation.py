"""Tests for evaluation module."""
from __future__ import annotations

import pytest

from nodecanon.core.models import KGGraph, KGNode, MergeConflict, MergeRecord, ScoreVector
from nodecanon.evaluation.metrics import MergeReport, QADeltaReport, evaluate


def _sv() -> ScoreVector:
    return ScoreVector(0.9, 0.9, 1.0, 0.8, 0.5)


def _record(canonical: str, merged: list[str]) -> MergeRecord:
    return MergeRecord(canonical_id=canonical, merged_ids=merged, score=_sv(), strategy="rule_based")


class TestMergeReport:
    def test_reduction_pct_correct(self) -> None:
        report = MergeReport(original_node_count=100, resolved_node_count=60)
        assert report.reduction_pct == pytest.approx(40.0)

    def test_zero_original_nodes_no_division_error(self) -> None:
        report = MergeReport(original_node_count=0, resolved_node_count=0)
        assert report.reduction_pct == pytest.approx(0.0)

    def test_no_reduction(self) -> None:
        report = MergeReport(original_node_count=50, resolved_node_count=50)
        assert report.reduction_pct == pytest.approx(0.0)

    def test_aliases_absorbed_counts_all_merged_ids(self) -> None:
        report = MergeReport(
            original_node_count=10,
            resolved_node_count=7,
            merge_records=[
                _record("a", ["b", "c"]),
                _record("d", ["e"]),
            ],
        )
        assert report.aliases_absorbed == 3

    def test_str_contains_key_numbers(self) -> None:
        report = MergeReport(
            original_node_count=100,
            resolved_node_count=60,
            merge_records=[_record("a", ["b"])],
        )
        text = str(report)
        assert "100" in text
        assert "60" in text

    def test_str_mentions_conflicts(self) -> None:
        sv = _sv()
        conflict = MergeConflict(node_id_a="x", node_id_b="y", score=sv, conflict_reason="test")
        report = MergeReport(
            original_node_count=10,
            resolved_node_count=9,
            conflicts=[conflict],
        )
        assert "1" in str(report)


class TestQADeltaReport:
    def test_delta_positive(self) -> None:
        r = QADeltaReport(baseline_accuracy=0.70, resolved_accuracy=0.85, dataset="HotpotQA")
        assert r.delta == pytest.approx(0.15)

    def test_delta_negative(self) -> None:
        r = QADeltaReport(baseline_accuracy=0.80, resolved_accuracy=0.75, dataset="test")
        assert r.delta == pytest.approx(-0.05)

    def test_str_contains_dataset(self) -> None:
        r = QADeltaReport(baseline_accuracy=0.70, resolved_accuracy=0.85, dataset="HotpotQA")
        assert "HotpotQA" in str(r)


class TestEvaluate:
    def test_returns_merge_report(self) -> None:
        original = KGGraph(
            nodes=[KGNode(id=f"n{i}", name=f"Node {i}") for i in range(5)],
            edges=[],
        )
        resolved = KGGraph(
            nodes=[KGNode(id=f"n{i}", name=f"Node {i}") for i in range(3)],
            edges=[],
        )
        records = [_record("n0", ["n3", "n4"])]
        report = evaluate(original, resolved, records, [])
        assert report.original_node_count == 5
        assert report.resolved_node_count == 3
        assert report.reduction_pct == pytest.approx(40.0)

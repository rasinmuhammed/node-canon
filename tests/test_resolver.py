"""Tests for Resolver orchestrator.

Uses _PairwiseScorer to avoid loading the sentence-transformers model in CI.
All tests are fully deterministic.
"""
from __future__ import annotations

import pytest

from nodecanon.core.matching import RuleBasedMatcher
from nodecanon.core.models import KGEdge, KGGraph, KGNode, ScoreVector
from nodecanon.core.resolver import Resolver
from nodecanon.core.scoring import NodeScorer


# ---------------------------------------------------------------------------
# Test helper: fixed-score scorer that needs no embedding model
# ---------------------------------------------------------------------------

class _PairwiseScorer(NodeScorer):
    """Returns pre-defined ScoreVectors for specific node-id pairs."""

    def __init__(self, scores: dict[tuple[str, str], ScoreVector]) -> None:
        super().__init__()
        self._fixed: dict[tuple[str, str], ScoreVector] = {
            (min(a, b), max(a, b)): sv for (a, b), sv in scores.items()
        }

    def fit(self, graph: KGGraph) -> None:
        self._adjacency_index = graph.adjacency_index()
        self._node_index = graph.node_index()

    def score(self, node_a: KGNode, node_b: KGNode, _graph: KGGraph) -> ScoreVector:
        key = (min(node_a.id, node_b.id), max(node_a.id, node_b.id))
        return self._fixed.get(key, ScoreVector(0.0, 0.0, 0.0, 0.0, 0.0))


def _high_sv() -> ScoreVector:
    return ScoreVector(
        name_similarity=0.95,
        semantic_similarity=0.92,
        type_agreement=1.0,
        neighbor_overlap=0.88,
        description_similarity=0.80,
    )


def _low_sv() -> ScoreVector:
    return ScoreVector(0.1, 0.1, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Helpers that build resolvers with controlled scorers
# ---------------------------------------------------------------------------

def _resolver(scores: dict[tuple[str, str], ScoreVector], threshold: float = 0.75) -> Resolver:
    return Resolver(
        scorer=_PairwiseScorer(scores),
        matcher=RuleBasedMatcher(threshold=threshold),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResolver:
    def test_no_duplicates_unchanged(self, no_duplicate_graph: KGGraph) -> None:
        resolver = _resolver({})  # no pair will score high
        result = resolver.resolve(no_duplicate_graph)
        assert len(result.graph.nodes) == len(no_duplicate_graph.nodes)
        assert result.merge_records == []
        assert result.conflicts == []

    def test_duplicates_merged(self, simple_graph: KGGraph) -> None:
        # simple_graph: n1=IBM (ORG), n2=I.B.M. (COMPANY), n3=Ginni Rometty (PERSON)
        resolver = _resolver({("n1", "n2"): _high_sv()})
        result = resolver.resolve(simple_graph)

        node_ids = {n.id for n in result.graph.nodes}
        assert len(result.graph.nodes) == 2   # IBM + Ginni Rometty
        assert "n3" in node_ids               # Ginni Rometty unchanged
        assert len(result.merge_records) == 1
        assert set(result.merge_records[0].merged_ids) <= {"n1", "n2"}

    def test_incompatible_types_not_merged(self, conflict_graph: KGGraph) -> None:
        # conflict_graph: c1=Apple (PERSON), c2=Apple (ORGANIZATION).
        # TypeCompatibilityBlocker rejects this pair at blocking — nodes preserved.
        resolver = _resolver({("c1", "c2"): _high_sv()})
        result = resolver.resolve(conflict_graph)
        assert len(result.graph.nodes) == 2
        assert result.merge_records == []

    def test_conflict_detector_fires_when_type_filter_bypassed(
        self, conflict_graph: KGGraph
    ) -> None:
        # If a custom blocker bypasses the type filter, ConflictDetector must
        # surface the incompatible-type pair as a MergeConflict — not silently merge.
        from nodecanon.core.blocking import (
            NGramFingerprintBlocker,
            TokenOverlapBlocker,
            UnionBlocker,
        )

        resolver = Resolver(
            blocker=UnionBlocker([TokenOverlapBlocker(), NGramFingerprintBlocker()]),
            scorer=_PairwiseScorer({("c1", "c2"): _high_sv()}),
            matcher=RuleBasedMatcher(threshold=0.75),
        )
        result = resolver.resolve(conflict_graph)
        assert len(result.graph.nodes) == 2   # not merged
        assert len(result.conflicts) == 1     # surfaced as conflict
        assert result.merge_records == []

    def test_provenance_on_merged_node(self, simple_graph: KGGraph) -> None:
        resolver = _resolver({("n1", "n2"): _high_sv()})
        result = resolver.resolve(simple_graph)
        # The canonical from n1/n2 merge must carry provenance
        merged = next(n for n in result.graph.nodes if n._merged_from is not None)
        assert set(merged._merged_from) == {"n1", "n2"}
        assert merged._merge_strategy == "rule_based"
        assert merged._merge_evidence is not None

    def test_no_data_loss_all_nodes_accounted(self, simple_graph: KGGraph) -> None:
        resolver = _resolver({("n1", "n2"): _high_sv()})
        result = resolver.resolve(simple_graph)
        # Every original node should either be in the resolved graph or listed
        # as an absorbed alias in a MergeRecord.
        original_ids = {n.id for n in simple_graph.nodes}
        resolved_ids = {n.id for n in result.graph.nodes}
        alias_ids = {aid for r in result.merge_records for aid in r.merged_ids}
        assert original_ids == resolved_ids | alias_ids

    def test_result_has_merge_report(self, simple_graph: KGGraph) -> None:
        resolver = _resolver({("n1", "n2"): _high_sv()})
        result = resolver.resolve(simple_graph)
        report = result.merge_report()
        assert "→" in report
        assert "canonical" in report.lower() or "nodes" in report.lower()

    def test_transitive_merges_via_union_find(self) -> None:
        # A matches B, B matches C → all three should merge into one group.
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="IBM Corp"),
                KGNode(id="d", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="d", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="d", relation="PRODUCT"),
                KGEdge(source_id="c", target_id="d", relation="PRODUCT"),
            ],
        )
        resolver = _resolver({
            ("a", "b"): _high_sv(),
            ("b", "c"): _high_sv(),
        })
        result = resolver.resolve(graph)
        # a, b, c → one canonical + Watson = 2 nodes total
        assert len(result.graph.nodes) == 2
        assert len(result.merge_records) == 1
        assert len(result.merge_records[0].merged_ids) == 2  # two aliases absorbed

    def test_edges_redirected_after_merge(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
            ],
        )
        resolver = _resolver({("a", "b"): _high_sv()})
        result = resolver.resolve(graph)
        # Both edges should converge on the canonical node → deduplicated to one
        edges_to_watson = [
            e for e in result.graph.edges if e.target_id == "c"
        ]
        assert len(edges_to_watson) == 1

    def test_self_loop_edge_not_in_result(self) -> None:
        # If a→b merge and there's an a→b edge, it must not become a self-loop.
        graph = KGGraph(
            nodes=[KGNode(id="a", name="IBM"), KGNode(id="b", name="I.B.M.")],
            edges=[KGEdge(source_id="a", target_id="b", relation="ALIAS")],
        )
        resolver = _resolver({("a", "b"): _high_sv()})
        result = resolver.resolve(graph)
        for edge in result.graph.edges:
            assert edge.source_id != edge.target_id

    def test_empty_name_raises(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="a", name=""), KGNode(id="b", name="IBM")],
            edges=[],
        )
        with pytest.raises(ValueError, match="no name field"):
            Resolver().resolve(graph)

    def test_original_counts_recorded(self, simple_graph: KGGraph) -> None:
        resolver = _resolver({("n1", "n2"): _high_sv()})
        result = resolver.resolve(simple_graph)
        assert result.original_node_count == len(simple_graph.nodes)
        assert result.original_edge_count == len(simple_graph.edges)

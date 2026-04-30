"""Tests for merging layer."""
from __future__ import annotations

import pytest

from nodecanon.core.merging import ConflictDetector, EdgeMerger, NodeMerger
from nodecanon.core.models import KGEdge, KGGraph, KGNode, ScoreVector


def _sv(**kwargs: float) -> ScoreVector:
    defaults = dict(
        name_similarity=0.9,
        semantic_similarity=0.9,
        type_agreement=1.0,
        neighbor_overlap=0.8,
        description_similarity=0.5,
    )
    defaults.update(kwargs)
    return ScoreVector(**defaults)


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------

class TestConflictDetector:
    def test_person_vs_organization_is_conflict(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="Apple", type="PERSON")
        b = KGNode(id="b", name="Apple", type="ORGANIZATION")
        conflict = detector.detect(a, b, _sv())
        assert conflict is not None
        assert conflict.node_id_a == "a"
        assert conflict.node_id_b == "b"
        assert "PERSON" in conflict.conflict_reason

    def test_same_type_is_no_conflict(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="IBM", type="ORGANIZATION")
        b = KGNode(id="b", name="I.B.M.", type="ORGANIZATION")
        assert detector.detect(a, b, _sv()) is None

    def test_compatible_types_no_conflict(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="IBM", type="ORGANIZATION")
        b = KGNode(id="b", name="IBM Inc", type="COMPANY")
        assert detector.detect(a, b, _sv()) is None

    def test_null_type_no_conflict(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="IBM")  # type=None
        b = KGNode(id="b", name="IBM Corp", type="ORGANIZATION")
        assert detector.detect(a, b, _sv()) is None

    def test_both_null_types_no_conflict(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="X")
        b = KGNode(id="b", name="Y")
        assert detector.detect(a, b, _sv()) is None

    def test_conflict_carries_score(self) -> None:
        detector = ConflictDetector()
        a = KGNode(id="a", name="Apple", type="PERSON")
        b = KGNode(id="b", name="Apple", type="ORGANIZATION")
        sv = _sv(name_similarity=0.99)
        conflict = detector.detect(a, b, sv)
        assert conflict is not None
        assert conflict.score.name_similarity == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# NodeMerger — select_canonical
# ---------------------------------------------------------------------------

class TestNodeMergerSelectCanonical:
    def test_highest_degree_chosen(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Watson"),
                KGNode(id="d", name="Armonk"),
            ],
            edges=[
                # b has 2 edges, a has 1
                KGEdge(source_id="a", target_id="c", relation="R"),
                KGEdge(source_id="b", target_id="c", relation="R"),
                KGEdge(source_id="b", target_id="d", relation="R"),
            ],
        )
        merger = NodeMerger()
        canonical = merger.select_canonical(
            [graph.node_index()["a"], graph.node_index()["b"]], graph
        )
        assert canonical.id == "b"

    def test_description_length_tiebreaker(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM", description="Short."),
                KGNode(id="b", name="I.B.M.", description="A longer description that wins."),
            ],
            edges=[],  # equal degree (0), so description breaks the tie
        )
        merger = NodeMerger()
        canonical = merger.select_canonical(
            [graph.node_index()["a"], graph.node_index()["b"]], graph
        )
        assert canonical.id == "b"

    def test_single_node_returns_that_node(self) -> None:
        graph = KGGraph(nodes=[KGNode(id="a", name="IBM")], edges=[])
        merger = NodeMerger()
        assert merger.select_canonical([graph.node_index()["a"]], graph).id == "a"


# ---------------------------------------------------------------------------
# NodeMerger — merge
# ---------------------------------------------------------------------------

class TestNodeMergerMerge:
    def test_provenance_fields_set(self) -> None:
        canonical = KGNode(id="a", name="IBM", type="ORGANIZATION")
        alias = KGNode(id="b", name="I.B.M.", type="COMPANY")
        sv = _sv()
        merger = NodeMerger()
        merged, record = merger.merge(canonical, [alias], sv)

        assert merged._merge_strategy == "rule_based"
        assert merged._merge_evidence is not None
        assert "name_similarity" in merged._merge_evidence
        assert set(merged._merged_from) == {"a", "b"}

    def test_merge_record_correct(self) -> None:
        canonical = KGNode(id="a", name="IBM")
        aliases = [KGNode(id="b", name="I.B.M."), KGNode(id="c", name="IBM Corp")]
        merged, record = NodeMerger().merge(canonical, aliases, _sv())
        assert record.canonical_id == "a"
        assert set(record.merged_ids) == {"b", "c"}
        assert record.strategy == "rule_based"

    def test_longest_description_kept(self) -> None:
        canonical = KGNode(id="a", name="IBM", description="Short.")
        alias = KGNode(id="b", name="I.B.M.", description="A much longer description here.")
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged.description == "A much longer description here."

    def test_no_description_propagated(self) -> None:
        canonical = KGNode(id="a", name="IBM")
        alias = KGNode(id="b", name="I.B.M.", description="IBM is a tech company.")
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged.description == "IBM is a tech company."

    def test_canonical_type_preserved(self) -> None:
        canonical = KGNode(id="a", name="IBM", type="ORGANIZATION")
        alias = KGNode(id="b", name="IBM Inc", type="COMPANY")
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged.type == "ORGANIZATION"  # canonical's type wins

    def test_resolved_types_contains_all(self) -> None:
        canonical = KGNode(id="a", name="IBM", type="ORGANIZATION")
        alias = KGNode(id="b", name="IBM Inc", type="COMPANY")
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged._resolved_types is not None
        assert "ORGANIZATION" in merged._resolved_types
        assert "COMPANY" in merged._resolved_types

    def test_source_chunks_union(self) -> None:
        canonical = KGNode(id="a", name="IBM", source_chunks=["chunk_1", "chunk_2"])
        alias = KGNode(id="b", name="IBM Inc", source_chunks=["chunk_2", "chunk_3"])
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert set(merged.source_chunks) == {"chunk_1", "chunk_2", "chunk_3"}

    def test_no_data_loss_attributes(self) -> None:
        canonical = KGNode(id="a", name="IBM", attributes={"founded": 1911})
        alias = KGNode(id="b", name="I.B.M.", attributes={"hq": "Armonk"})
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged.attributes["founded"] == 1911
        assert merged.attributes["hq"] == "Armonk"

    def test_canonical_attribute_wins_on_conflict(self) -> None:
        canonical = KGNode(id="a", name="IBM", attributes={"founded": 1911})
        alias = KGNode(id="b", name="I.B.M.", attributes={"founded": 1910})  # wrong
        merged, _ = NodeMerger().merge(canonical, [alias], _sv())
        assert merged.attributes["founded"] == 1911

    def test_original_canonical_not_mutated(self) -> None:
        canonical = KGNode(id="a", name="IBM", type="ORGANIZATION")
        alias = KGNode(id="b", name="I.B.M.", type="COMPANY")
        original_type = canonical.type
        NodeMerger().merge(canonical, [alias], _sv())
        assert canonical.type == original_type
        assert canonical._merged_from is None  # untouched


# ---------------------------------------------------------------------------
# EdgeMerger
# ---------------------------------------------------------------------------

class TestEdgeMerger:
    def test_alias_edges_redirected_to_canonical(self) -> None:
        merger = EdgeMerger()
        edges = [
            KGEdge(source_id="b", target_id="c", relation="CEO_OF"),
            KGEdge(source_id="d", target_id="b", relation="WORKS_FOR"),
        ]
        # b is an alias → all its edges should point to canonical "a"
        result = merger.merge_edges(edges, alias_to_canonical={"b": "a"})
        for edge in result:
            assert edge.source_id != "b"
            assert edge.target_id != "b"

    def test_parallel_edges_deduplicated(self) -> None:
        merger = EdgeMerger()
        # After redirect, two edges a→c become parallel
        edges = [
            KGEdge(source_id="a", target_id="c", relation="CEO_OF", weight=1.0),
            KGEdge(source_id="b", target_id="c", relation="CEO_OF", weight=1.0),
        ]
        result = merger.merge_edges(edges, alias_to_canonical={"b": "a"})
        assert len(result) == 1
        assert result[0].weight == pytest.approx(2.0)

    def test_self_loops_dropped(self) -> None:
        merger = EdgeMerger()
        # a and b merge into a; edge a→b becomes a→a (self-loop)
        edges = [KGEdge(source_id="a", target_id="b", relation="ALIAS")]
        result = merger.merge_edges(edges, alias_to_canonical={"b": "a"})
        assert result == []

    def test_unrelated_edges_unchanged(self) -> None:
        merger = EdgeMerger()
        edges = [KGEdge(source_id="x", target_id="y", relation="R")]
        result = merger.merge_edges(edges, alias_to_canonical={"b": "a"})
        assert len(result) == 1
        assert result[0].source_id == "x"

    def test_parallel_weights_summed(self) -> None:
        merger = EdgeMerger()
        edges = [
            KGEdge(source_id="a", target_id="c", relation="R", weight=2.0),
            KGEdge(source_id="a", target_id="c", relation="R", weight=3.0),
        ]
        result = merger.merge_edges(edges, alias_to_canonical={})
        assert len(result) == 1
        assert result[0].weight == pytest.approx(5.0)

    def test_different_relations_not_merged(self) -> None:
        merger = EdgeMerger()
        edges = [
            KGEdge(source_id="a", target_id="c", relation="CEO_OF"),
            KGEdge(source_id="a", target_id="c", relation="FOUNDED"),
        ]
        result = merger.merge_edges(edges, alias_to_canonical={})
        assert len(result) == 2

    def test_empty_edges_returns_empty(self) -> None:
        assert EdgeMerger().merge_edges([], {}) == []

"""Tests for user-facing API: GraphBuilder, KGGraph.from_dicts, and
post-resolution editing (reject_merge, force_merge, accept_conflict, explain).
"""

from __future__ import annotations

import pytest

from nodecanon import GraphBuilder, KGGraph, KGNode, Resolver
from nodecanon.core.models import MergeConflict, ScoreVector
from nodecanon.core.resolver import ResolveResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_resolved_result() -> ResolveResult:
    """Resolve a tiny IBM / I.B.M. / Microsoft graph for editing tests."""
    graph = (
        GraphBuilder()
        .add_node("IBM", type="ORGANIZATION")
        .add_node("I.B.M.", type="ORGANIZATION")
        .add_node("Microsoft", type="ORGANIZATION")
        .add_node("Watson AI", type="PRODUCT")
        .add_edge("IBM", "Watson AI", "MADE")
        .add_edge("I.B.M.", "Watson AI", "MADE")
        .build()
    )
    from unittest.mock import patch

    from nodecanon.core.scoring import NodeScorer

    _W = {"name_similarity": 0.43, "semantic_similarity": 0.00,
          "type_agreement": 0.29, "neighbor_overlap": 0.29, "description_similarity": 0.00}

    def _fast_fit(self, g):
        self._adjacency_index = g.adjacency_index()
        self._node_index = g.node_index()

    def _zero(_self, _a, _b):
        return 0.0

    from nodecanon.core.matching import RuleBasedMatcher
    scorer = NodeScorer(weights=_W, cache_dir=None)
    matcher = RuleBasedMatcher(threshold=0.72, weights=_W)
    with (
        patch.object(NodeScorer, "fit", _fast_fit),
        patch.object(NodeScorer, "_semantic_similarity", _zero),
        patch.object(NodeScorer, "_description_similarity", _zero),
    ):
        return Resolver(scorer=scorer, matcher=matcher).resolve(graph)


# ---------------------------------------------------------------------------
# KGGraph.from_dicts
# ---------------------------------------------------------------------------


class TestKGGraphFromDicts:
    def test_basic_nodes(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[
                {"id": "n1", "name": "IBM", "type": "ORGANIZATION"},
                {"id": "n2", "name": "Microsoft", "type": "ORGANIZATION"},
            ]
        )
        assert len(graph.nodes) == 2
        idx = graph.node_index()
        assert idx["n1"].name == "IBM"
        assert idx["n2"].type == "ORGANIZATION"

    def test_auto_id_from_name(self) -> None:
        graph = KGGraph.from_dicts(nodes=[{"name": "IBM Corporation"}])
        assert len(graph.nodes) == 1
        assert graph.nodes[0].id == "ibm_corporation"

    def test_auto_id_collision_resolved(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[{"name": "IBM"}, {"name": "IBM"}]  # duplicate name → distinct ids
        )
        ids = [n.id for n in graph.nodes]
        assert len(set(ids)) == 2  # both got unique ids

    def test_edges_source_target(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            edges=[{"source": "a", "target": "b", "relation": "LINKS_TO"}],
        )
        assert len(graph.edges) == 1
        assert graph.edges[0].relation == "LINKS_TO"

    def test_edges_source_id_target_id(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[{"id": "a", "name": "A"}, {"id": "b", "name": "B"}],
            edges=[{"source_id": "a", "target_id": "b", "relation": "X"}],
        )
        assert graph.edges[0].source_id == "a"

    def test_extra_fields_land_in_attributes(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[{"id": "n1", "name": "IBM", "founded": 1911, "country": "USA"}]
        )
        attrs = graph.nodes[0].attributes
        assert attrs["founded"] == 1911
        assert attrs["country"] == "USA"

    def test_description_parsed(self) -> None:
        graph = KGGraph.from_dicts(
            nodes=[{"name": "IBM", "description": "A big tech company"}]
        )
        assert graph.nodes[0].description == "A big tech company"

    def test_none_edges_param(self) -> None:
        graph = KGGraph.from_dicts(nodes=[{"id": "n1", "name": "A"}], edges=None)
        assert graph.edges == []

    def test_repr(self) -> None:
        graph = KGGraph.from_dicts(nodes=[{"id": "n1", "name": "A"}])
        assert "1 nodes" in repr(graph)


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class TestGraphBuilder:
    def test_fluent_chain(self) -> None:
        graph = (
            GraphBuilder()
            .add_node("IBM", type="ORGANIZATION")
            .add_node("Watson AI", type="PRODUCT")
            .add_edge("IBM", "Watson AI", "MADE")
            .build()
        )
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1

    def test_add_node_idempotent(self) -> None:
        builder = GraphBuilder()
        builder.add_node("IBM")
        builder.add_node("IBM")  # second call is a no-op
        assert len(builder.build().nodes) == 1

    def test_edge_by_name(self) -> None:
        graph = (
            GraphBuilder()
            .add_node("A", id="node_a")
            .add_node("B", id="node_b")
            .add_edge("A", "B", "X")
            .build()
        )
        assert graph.edges[0].source_id == "node_a"
        assert graph.edges[0].target_id == "node_b"

    def test_edge_by_id(self) -> None:
        graph = (
            GraphBuilder()
            .add_node("A", id="node_a")
            .add_node("B", id="node_b")
            .add_edge("node_a", "node_b", "X")
            .build()
        )
        assert graph.edges[0].source_id == "node_a"

    def test_auto_create_missing_node_from_edge(self) -> None:
        graph = (
            GraphBuilder()
            .add_edge("IBM", "Watson", "MADE")  # neither node added yet
            .build()
        )
        assert len(graph.nodes) == 2

    def test_custom_id(self) -> None:
        graph = GraphBuilder().add_node("IBM Corp", id="ibm").build()
        assert graph.nodes[0].id == "ibm"
        assert graph.nodes[0].name == "IBM Corp"

    def test_extra_attributes_stored(self) -> None:
        graph = GraphBuilder().add_node("IBM", founded=1911).build()
        assert graph.nodes[0].attributes["founded"] == 1911

    def test_edge_weight(self) -> None:
        graph = (
            GraphBuilder()
            .add_edge("A", "B", "X", weight=3.5)
            .build()
        )
        assert graph.edges[0].weight == 3.5

    def test_build_returns_kggraph(self) -> None:
        result = GraphBuilder().build()
        assert isinstance(result, KGGraph)

    def test_repr_nodes(self) -> None:
        node = KGNode(id="n1", name="IBM", type="ORGANIZATION")
        assert "IBM" in repr(node)
        assert "ORGANIZATION" in repr(node)


# ---------------------------------------------------------------------------
# ResolveResult.explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_merged_node(self) -> None:
        result = _simple_resolved_result()
        merged = [r for r in result.merge_records]
        assert merged, "Expected at least one merge"
        explanation = result.explain(merged[0].canonical_id)
        assert "Canonical node" in explanation
        assert "Merged from" in explanation
        assert "weighted score" in explanation

    def test_explain_unmerged_node(self) -> None:
        result = _simple_resolved_result()
        # Find a node that was not merged (singleton)
        merged_ids = {r.canonical_id for r in result.merge_records}
        merged_ids |= {mid for r in result.merge_records for mid in r.merged_ids}
        singleton = next(n for n in result.graph.nodes if n.id not in merged_ids)
        explanation = result.explain(singleton.id)
        assert "was not merged" in explanation

    def test_explain_alias_id(self) -> None:
        result = _simple_resolved_result()
        for record in result.merge_records:
            if record.merged_ids:
                # Ask about an alias id, should still find the record
                explanation = result.explain(record.merged_ids[0])
                assert "Canonical node" in explanation
                return
        pytest.skip("No aliases found in merge records")

    def test_explain_returns_string(self) -> None:
        result = _simple_resolved_result()
        for r in result.merge_records:
            assert isinstance(result.explain(r.canonical_id), str)


# ---------------------------------------------------------------------------
# ResolveResult.reject_merge
# ---------------------------------------------------------------------------


class TestRejectMerge:
    def test_reject_restores_nodes(self) -> None:
        result = _simple_resolved_result()
        assert result.merge_records, "Expected at least one merge"
        record = result.merge_records[0]
        original_count = len(result.graph.nodes)

        corrected = result.reject_merge(record.canonical_id)

        # Should have more nodes now (aliases restored)
        assert len(corrected.graph.nodes) > original_count

    def test_reject_removes_merge_record(self) -> None:
        result = _simple_resolved_result()
        record = result.merge_records[0]
        corrected = result.reject_merge(record.canonical_id)
        ids = [r.canonical_id for r in corrected.merge_records]
        assert record.canonical_id not in ids

    def test_reject_does_not_mutate_original(self) -> None:
        result = _simple_resolved_result()
        original_node_count = len(result.graph.nodes)
        original_records = len(result.merge_records)
        record = result.merge_records[0]
        result.reject_merge(record.canonical_id)
        assert len(result.graph.nodes) == original_node_count
        assert len(result.merge_records) == original_records

    def test_reject_unknown_id_raises(self) -> None:
        result = _simple_resolved_result()
        with pytest.raises(ValueError, match="No merge record"):
            result.reject_merge("does_not_exist")

    def test_partial_reject(self) -> None:
        result = _simple_resolved_result()
        record = result.merge_records[0]
        if not record.merged_ids:
            pytest.skip("No aliases to partially restore")
        alias_id = record.merged_ids[0]
        corrected = result.reject_merge(record.canonical_id, restore=[alias_id])
        node_ids = {n.id for n in corrected.graph.nodes}
        assert alias_id in node_ids


# ---------------------------------------------------------------------------
# ResolveResult.force_merge
# ---------------------------------------------------------------------------


class TestForceMerge:
    def test_force_merge_two_nodes(self) -> None:
        result = _simple_resolved_result()
        node_ids = [n.id for n in result.graph.nodes]
        if len(node_ids) < 2:
            pytest.skip("Not enough nodes to force-merge")
        merged = result.force_merge(node_ids[0], node_ids[1])
        assert len(merged.graph.nodes) < len(result.graph.nodes)

    def test_force_merge_adds_record(self) -> None:
        result = _simple_resolved_result()
        node_ids = [n.id for n in result.graph.nodes]
        original_records = len(result.merge_records)
        merged = result.force_merge(node_ids[0], node_ids[1])
        assert len(merged.merge_records) == original_records + 1
        assert merged.merge_records[-1].strategy == "manual"

    def test_force_merge_unknown_id_raises(self) -> None:
        result = _simple_resolved_result()
        with pytest.raises(ValueError, match="not found"):
            result.force_merge("real_id_needed", "ghost_id")

    def test_force_merge_one_id_raises(self) -> None:
        result = _simple_resolved_result()
        node_id = result.graph.nodes[0].id
        with pytest.raises(ValueError, match="at least 2"):
            result.force_merge(node_id)

    def test_force_merge_does_not_mutate_original(self) -> None:
        result = _simple_resolved_result()
        node_ids = [n.id for n in result.graph.nodes]
        original_count = len(result.graph.nodes)
        result.force_merge(node_ids[0], node_ids[1])
        assert len(result.graph.nodes) == original_count


# ---------------------------------------------------------------------------
# ResolveResult.accept_conflict
# ---------------------------------------------------------------------------


class TestAcceptConflict:
    def _result_with_conflict(self) -> ResolveResult:
        conflict = MergeConflict(
            node_id_a="n1",
            node_id_b="n2",
            score=ScoreVector(0.9, 0.9, 0.0, 0.9, 0.0),
            conflict_reason="Incompatible types: PERSON vs ORGANIZATION",
        )
        graph = KGGraph.from_dicts(
            nodes=[
                {"id": "n1", "name": "IBM", "type": "ORGANIZATION"},
                {"id": "n2", "name": "IBM Person", "type": "PERSON"},
                {"id": "n3", "name": "Watson", "type": "PRODUCT"},
            ]
        )
        return ResolveResult(
            graph=graph,
            conflicts=[conflict],
            original_node_count=3,
            original_edge_count=0,
        )

    def test_accept_merges_nodes(self) -> None:
        result = self._result_with_conflict()
        accepted = result.accept_conflict(0)
        assert len(accepted.graph.nodes) == 2  # n1+n2 merged, n3 remains

    def test_accept_removes_conflict(self) -> None:
        result = self._result_with_conflict()
        accepted = result.accept_conflict(0)
        assert len(accepted.conflicts) == 0

    def test_accept_out_of_range_raises(self) -> None:
        result = self._result_with_conflict()
        with pytest.raises(IndexError):
            result.accept_conflict(5)

    def test_accept_negative_index_raises(self) -> None:
        result = self._result_with_conflict()
        with pytest.raises(IndexError):
            result.accept_conflict(-1)

"""Tests for core data models."""
from __future__ import annotations

import pytest

from nodecanon.core.models import KGEdge, KGGraph, KGNode, MergeConflict, MergeRecord, ScoreVector


class TestKGNode:
    def test_minimal_construction(self) -> None:
        node = KGNode(id="n1", name="IBM")
        assert node.id == "n1"
        assert node.name == "IBM"
        assert node.type is None
        assert node.attributes == {}
        assert node._merged_from is None

    def test_defaults_are_independent(self) -> None:
        a = KGNode(id="a", name="A")
        b = KGNode(id="b", name="B")
        a.attributes["x"] = 1
        assert "x" not in b.attributes


class TestKGGraph:
    def test_node_index(self, simple_graph: KGGraph) -> None:
        idx = simple_graph.node_index()
        assert "n1" in idx
        assert idx["n1"].name == "IBM"

    def test_adjacency_index(self, simple_graph: KGGraph) -> None:
        adj = simple_graph.adjacency_index()
        assert "n3" in adj["n1"]
        assert "n1" in adj["n3"]

    def test_isolated_node_has_empty_neighbors(self) -> None:
        graph = KGGraph(nodes=[KGNode(id="x", name="X")], edges=[])
        adj = graph.adjacency_index()
        assert adj["x"] == []


class TestScoreVector:
    def test_weighted_sum_defaults(self) -> None:
        sv = ScoreVector(
            name_similarity=1.0,
            semantic_similarity=1.0,
            type_agreement=1.0,
            neighbor_overlap=1.0,
            description_similarity=1.0,
        )
        assert sv.weighted_sum() == pytest.approx(1.0)

    def test_zero_vector(self) -> None:
        sv = ScoreVector(0.0, 0.0, 0.0, 0.0, 0.0)
        assert sv.weighted_sum() == pytest.approx(0.0)

    def test_custom_weights(self) -> None:
        sv = ScoreVector(1.0, 0.0, 0.0, 0.0, 0.0)
        result = sv.weighted_sum({"name_similarity": 1.0, "semantic_similarity": 0.0,
                                  "type_agreement": 0.0, "neighbor_overlap": 0.0,
                                  "description_similarity": 0.0})
        assert result == pytest.approx(1.0)

    def test_to_dict_keys(self) -> None:
        sv = ScoreVector(0.1, 0.2, 0.3, 0.4, 0.5)
        d = sv.to_dict()
        assert set(d.keys()) == {
            "name_similarity", "semantic_similarity", "type_agreement",
            "neighbor_overlap", "description_similarity",
        }

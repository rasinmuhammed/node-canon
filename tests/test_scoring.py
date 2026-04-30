"""Tests for scoring layer.

Embedding-dependent tests avoid the real SentenceTransformer by pre-populating
the NodeScorer's internal caches directly. Only _name_similarity, _type_agreement,
and _neighbor_overlap are tested without any mocking — they have no encoder dep.
"""
from __future__ import annotations

import numpy as np
import pytest

from nodecanon.core.models import KGEdge, KGGraph, KGNode
from nodecanon.core.scoring import NodeScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int = 384, axis: int = 0) -> np.ndarray:
    """Unit vector along a single axis — deterministic, normalized."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _seeded_vec(seed: int, dim: int = 384) -> np.ndarray:
    """Random but deterministic unit vector for a given seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _scorer_with_embeddings(
    nodes: list[KGNode],
    same_name_emb: bool = False,
    desc_embs: dict[str, np.ndarray] | None = None,
) -> NodeScorer:
    """Build a NodeScorer with name embeddings pre-loaded into cache.

    same_name_emb=True → all nodes share the same embedding (maximally similar).
    same_name_emb=False → each node gets a unique seeded embedding.
    """
    scorer = NodeScorer()
    for i, node in enumerate(nodes):
        emb = _unit_vec() if same_name_emb else _seeded_vec(i)
        scorer._name_emb_cache[node.id] = emb
    if desc_embs:
        scorer._desc_emb_cache.update(desc_embs)
    return scorer


# ---------------------------------------------------------------------------
# _name_similarity  (no encoder — safe to call directly)
# ---------------------------------------------------------------------------

class TestNameSimilarity:
    def test_identical_names_score_one(self) -> None:
        scorer = NodeScorer()
        assert scorer._name_similarity("IBM", "IBM") == pytest.approx(1.0)

    def test_case_insensitive(self) -> None:
        scorer = NodeScorer()
        assert scorer._name_similarity("ibm", "IBM") > 0.9

    def test_abbreviation_variants_score_high(self) -> None:
        # "IBM" and "I.B.M." share the same metaphone → phonetic path gives 1.0
        scorer = NodeScorer()
        assert scorer._name_similarity("IBM", "I.B.M.") > 0.7

    def test_completely_different_names_score_low(self) -> None:
        scorer = NodeScorer()
        assert scorer._name_similarity("Apple", "Zebra") < 0.5

    def test_result_in_zero_one_range(self) -> None:
        scorer = NodeScorer()
        for a, b in [
            ("IBM", "International Business Machines"),
            ("Dr. Sarah Chen", "S. Chen"),
            ("colour", "color"),
            ("AAPL", "Apple Inc"),
        ]:
            score = scorer._name_similarity(a, b)
            assert 0.0 <= score <= 1.0, f"Out of range for ({a!r}, {b!r}): {score}"


# ---------------------------------------------------------------------------
# _type_agreement  (no encoder)
# ---------------------------------------------------------------------------

class TestTypeAgreement:
    def test_identical_types_return_one(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM", type="ORGANIZATION")
        b = KGNode(id="b", name="IBM Corp", type="ORGANIZATION")
        assert scorer._type_agreement(a, b) == pytest.approx(1.0)

    def test_compatible_types_return_one(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM", type="ORGANIZATION")
        b = KGNode(id="b", name="IBM Inc", type="COMPANY")
        assert scorer._type_agreement(a, b) == pytest.approx(1.0)

    def test_incompatible_types_return_zero(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="Apple", type="ORGANIZATION")
        b = KGNode(id="b", name="Apple", type="PERSON")
        assert scorer._type_agreement(a, b) == pytest.approx(0.0)

    def test_both_null_types_neutral(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM")
        b = KGNode(id="b", name="I.B.M.")
        assert scorer._type_agreement(a, b) == pytest.approx(0.5)

    def test_one_null_type_neutral(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM", type="ORGANIZATION")
        b = KGNode(id="b", name="IBM Inc")
        assert scorer._type_agreement(a, b) == pytest.approx(0.5)

    def test_type_comparison_is_case_insensitive(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM", type="organization")
        b = KGNode(id="b", name="IBM Corp", type="ORGANIZATION")
        assert scorer._type_agreement(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _neighbor_overlap  (no encoder)
# ---------------------------------------------------------------------------

class TestNeighborOverlap:
    def test_zero_when_both_isolated(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="a", name="IBM"), KGNode(id="b", name="I.B.M.")],
            edges=[],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        assert scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx) == pytest.approx(0.0)

    def test_one_when_identical_neighbor_sets(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Ginni Rometty"),
                KGNode(id="d", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="CEO_OF"),
                KGEdge(source_id="a", target_id="d", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="c", relation="CEO_OF"),
                KGEdge(source_id="b", target_id="d", relation="PRODUCT"),
            ],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        assert scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx) == pytest.approx(1.0)

    def test_partial_overlap(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="IBM Corp"),
                KGNode(id="c", name="Watson"),
                KGNode(id="d", name="Armonk"),
                KGNode(id="e", name="Azure"),     # only b connects here
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="a", target_id="d", relation="LOCATED_IN"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="d", relation="LOCATED_IN"),
                KGEdge(source_id="b", target_id="e", relation="COMPETES_WITH"),
            ],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        # intersection = {watson, armonk}, union = {watson, armonk, azure}
        overlap = scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx)
        assert overlap == pytest.approx(2 / 3)

    def test_direct_connection_excluded_from_neighbor_sets(self) -> None:
        # a and b are directly connected; they should be excluded from each
        # other's neighbor sets so that direct connection doesn't inflate overlap.
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="b", relation="ALIAS"),
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
            ],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        # neighbours(a) excl b = {c}, neighbours(b) excl a = {c}
        # intersection={c}, union={c} → 1.0
        assert scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx) == pytest.approx(1.0)

    def test_one_isolated_one_not(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Watson"),
            ],
            edges=[KGEdge(source_id="a", target_id="c", relation="PRODUCT")],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        # a has {c}, b has {} → intersection={}, union={c} → 0.0
        assert scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx) == pytest.approx(0.0)

    def test_result_in_zero_one_range(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="IBM Corp"),
                KGNode(id="c", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="R"),
                KGEdge(source_id="b", target_id="c", relation="R"),
            ],
        )
        scorer = NodeScorer()
        adj = graph.adjacency_index()
        idx = graph.node_index()
        val = scorer._neighbor_overlap(idx["a"], idx["b"], adj, idx)
        assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# _semantic_similarity and _description_similarity  (mocked via cache)
# ---------------------------------------------------------------------------

class TestSemanticSimilarity:
    def test_same_embedding_returns_one(self) -> None:
        nodes = [KGNode(id="a", name="IBM"), KGNode(id="b", name="I.B.M.")]
        scorer = _scorer_with_embeddings(nodes, same_name_emb=True)
        a, b = nodes
        assert scorer._semantic_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_embeddings_return_zero(self) -> None:
        nodes = [KGNode(id="a", name="Apple"), KGNode(id="b", name="Zebra")]
        scorer = NodeScorer()
        scorer._name_emb_cache["a"] = _unit_vec(axis=0)
        scorer._name_emb_cache["b"] = _unit_vec(axis=1)
        assert scorer._semantic_similarity(nodes[0], nodes[1]) == pytest.approx(0.0)

    def test_result_clipped_to_zero_one(self) -> None:
        # Force a slightly negative dot product and confirm it's clipped to 0.
        nodes = [KGNode(id="a", name="X"), KGNode(id="b", name="Y")]
        scorer = NodeScorer()
        v = _unit_vec(axis=0)
        scorer._name_emb_cache["a"] = v
        scorer._name_emb_cache["b"] = -v  # opposite direction → dot = -1
        result = scorer._semantic_similarity(nodes[0], nodes[1])
        assert result == pytest.approx(0.0)


class TestDescriptionSimilarity:
    def test_no_description_returns_zero(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM")
        b = KGNode(id="b", name="IBM Corp")
        assert scorer._description_similarity(a, b) == pytest.approx(0.0)

    def test_one_missing_description_returns_zero(self) -> None:
        scorer = NodeScorer()
        a = KGNode(id="a", name="IBM", description="A tech company.")
        b = KGNode(id="b", name="IBM Corp")
        assert scorer._description_similarity(a, b) == pytest.approx(0.0)

    def test_same_description_embedding_returns_one(self) -> None:
        a = KGNode(id="a", name="IBM", description="A tech company.")
        b = KGNode(id="b", name="IBM Corp", description="A technology firm.")
        emb = _unit_vec(axis=0)
        scorer = NodeScorer()
        scorer._desc_emb_cache["a"] = emb
        scorer._desc_emb_cache["b"] = emb
        assert scorer._description_similarity(a, b) == pytest.approx(1.0)

    def test_result_clipped_to_zero(self) -> None:
        a = KGNode(id="a", name="IBM", description="A tech company.")
        b = KGNode(id="b", name="Rival Corp", description="A rival firm.")
        emb = _unit_vec(axis=0)
        scorer = NodeScorer()
        scorer._desc_emb_cache["a"] = emb
        scorer._desc_emb_cache["b"] = -emb
        assert scorer._description_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# score()  (end-to-end, mocked embeddings)
# ---------------------------------------------------------------------------

class TestNodeScorer:
    def test_identical_nodes_score_high(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM", type="ORGANIZATION"),
                KGNode(id="b", name="IBM", type="ORGANIZATION"),
                KGNode(id="c", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
            ],
        )
        scorer = _scorer_with_embeddings(graph.nodes, same_name_emb=True)
        sv = scorer.score(graph.node_index()["a"], graph.node_index()["b"], graph)
        assert sv.weighted_sum() > 0.8

    def test_completely_different_nodes_score_low(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="Apple", type="ORGANIZATION"),
                KGNode(id="b", name="Zebra", type="PERSON"),
            ],
            edges=[],
        )
        scorer = NodeScorer()
        scorer._name_emb_cache["a"] = _unit_vec(axis=0)
        scorer._name_emb_cache["b"] = _unit_vec(axis=1)
        sv = scorer.score(graph.node_index()["a"], graph.node_index()["b"], graph)
        assert sv.weighted_sum() < 0.5

    def test_neighbor_overlap_zero_when_both_isolated(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="a", name="IBM"), KGNode(id="b", name="I.B.M.")],
            edges=[],
        )
        scorer = _scorer_with_embeddings(graph.nodes)
        sv = scorer.score(graph.node_index()["a"], graph.node_index()["b"], graph)
        assert sv.neighbor_overlap == pytest.approx(0.0)

    def test_neighbor_overlap_one_when_identical_neighbors(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM"),
                KGNode(id="b", name="I.B.M."),
                KGNode(id="c", name="Watson"),
                KGNode(id="d", name="Ginni Rometty"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="a", target_id="d", relation="CEO_OF"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="d", relation="CEO_OF"),
            ],
        )
        scorer = _scorer_with_embeddings(graph.nodes)
        sv = scorer.score(graph.node_index()["a"], graph.node_index()["b"], graph)
        assert sv.neighbor_overlap == pytest.approx(1.0)

    def test_score_vector_fields_in_range(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="a", name="IBM", type="ORGANIZATION", description="Tech company."),
                KGNode(id="b", name="IBM Corp", type="COMPANY", description="Technology firm."),
                KGNode(id="c", name="Watson"),
            ],
            edges=[
                KGEdge(source_id="a", target_id="c", relation="PRODUCT"),
                KGEdge(source_id="b", target_id="c", relation="PRODUCT"),
            ],
        )
        emb = _unit_vec(axis=0)
        scorer = NodeScorer()
        for n in graph.nodes:
            scorer._name_emb_cache[n.id] = emb
            if n.description:
                scorer._desc_emb_cache[n.id] = emb
        sv = scorer.score(graph.node_index()["a"], graph.node_index()["b"], graph)
        for field, val in sv.to_dict().items():
            assert 0.0 <= val <= 1.0, f"{field} out of range: {val}"

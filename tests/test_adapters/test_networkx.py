"""Tests for NetworkX adapter."""
from __future__ import annotations

import pytest

from nodecanon.adapters.networkx import NetworkXAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode


def _sample_graph() -> KGGraph:
    return KGGraph(
        nodes=[
            KGNode(id="n1", name="IBM", type="ORGANIZATION", description="A tech company."),
            KGNode(id="n2", name="Ginni Rometty", type="PERSON"),
            KGNode(id="n3", name="Watson", attributes={"category": "AI"}),
        ],
        edges=[
            KGEdge(source_id="n1", target_id="n2", relation="CEO_OF", weight=1.0),
            KGEdge(source_id="n1", target_id="n3", relation="PRODUCT", weight=2.0),
        ],
    )


class TestNetworkXAdapter:
    def test_to_networkx_node_count(self) -> None:
        adapter = NetworkXAdapter()
        G = adapter.to_networkx(_sample_graph())
        assert len(G.nodes) == 3

    def test_to_networkx_edge_count(self) -> None:
        adapter = NetworkXAdapter()
        G = adapter.to_networkx(_sample_graph())
        assert len(G.edges) == 2

    def test_to_networkx_node_attributes(self) -> None:
        adapter = NetworkXAdapter()
        G = adapter.to_networkx(_sample_graph())
        assert G.nodes["n1"]["name"] == "IBM"
        assert G.nodes["n1"]["type"] == "ORGANIZATION"
        assert G.nodes["n1"]["description"] == "A tech company."

    def test_to_networkx_edge_attributes(self) -> None:
        adapter = NetworkXAdapter()
        G = adapter.to_networkx(_sample_graph())
        assert G["n1"]["n2"]["relation"] == "CEO_OF"
        assert G["n1"]["n3"]["weight"] == pytest.approx(2.0)

    def test_to_networkx_extra_attributes_preserved(self) -> None:
        adapter = NetworkXAdapter()
        G = adapter.to_networkx(_sample_graph())
        assert G.nodes["n3"]["category"] == "AI"

    def test_roundtrip_preserves_nodes(self) -> None:
        adapter = NetworkXAdapter()
        original = _sample_graph()
        G = adapter.to_networkx(original)
        restored = adapter.from_networkx(G)
        assert {n.id for n in restored.nodes} == {n.id for n in original.nodes}

    def test_roundtrip_preserves_edges(self) -> None:
        adapter = NetworkXAdapter()
        original = _sample_graph()
        G = adapter.to_networkx(original)
        restored = adapter.from_networkx(G)
        orig_edges = {(e.source_id, e.target_id, e.relation) for e in original.edges}
        rest_edges = {(e.source_id, e.target_id, e.relation) for e in restored.edges}
        assert orig_edges == rest_edges

    def test_roundtrip_preserves_node_name_and_type(self) -> None:
        adapter = NetworkXAdapter()
        original = _sample_graph()
        restored = adapter.from_networkx(adapter.to_networkx(original))
        idx = restored.node_index()
        assert idx["n1"].name == "IBM"
        assert idx["n1"].type == "ORGANIZATION"

    def test_roundtrip_preserves_edge_weight(self) -> None:
        adapter = NetworkXAdapter()
        original = _sample_graph()
        restored = adapter.from_networkx(adapter.to_networkx(original))
        edge = next(e for e in restored.edges if e.relation == "PRODUCT")
        assert edge.weight == pytest.approx(2.0)

    def test_missing_name_raises_descriptive_error(self) -> None:
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("n1")  # no "name" attribute
        with pytest.raises(ValueError, match="no 'name' attribute"):
            NetworkXAdapter().from_networkx(G)

    def test_missing_relation_defaults(self) -> None:
        import networkx as nx
        G = nx.DiGraph()
        G.add_node("a", name="A")
        G.add_node("b", name="B")
        G.add_edge("a", "b")  # no "relation" attribute
        graph = NetworkXAdapter().from_networkx(G)
        assert graph.edges[0].relation == "RELATED_TO"

    def test_provenance_fields_survive_roundtrip(self) -> None:
        adapter = NetworkXAdapter()
        node = KGNode(
            id="c",
            name="IBM Canon",
            _merged_from=["a", "b"],
            _merge_strategy="rule_based",
            _resolved_types=["ORGANIZATION", "COMPANY"],
        )
        graph = KGGraph(nodes=[node], edges=[])
        restored = adapter.from_networkx(adapter.to_networkx(graph))
        n = restored.node_index()["c"]
        assert n._merged_from == ["a", "b"]
        assert n._merge_strategy == "rule_based"
        assert n._resolved_types == ["ORGANIZATION", "COMPANY"]

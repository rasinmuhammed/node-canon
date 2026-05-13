"""Tests for Neo4j Cypher export adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nodecanon.adapters.neo4j import Neo4jAdapter, _props_map, _safe_label
from nodecanon.core.models import KGEdge, KGGraph, KGNode


def _simple_graph() -> KGGraph:
    return KGGraph(
        nodes=[
            KGNode(
                id="n1", name="IBM", type="ORGANIZATION", description="A tech company."
            ),
            KGNode(id="n2", name="Ginni Rometty", type="PERSON"),
        ],
        edges=[
            KGEdge(source_id="n2", target_id="n1", relation="CEO_OF", weight=1.0),
        ],
    )


class TestSafeLabel:
    def test_spaces_replaced(self) -> None:
        assert _safe_label("ORGANIZATION TYPE") == "ORGANIZATION_TYPE"

    def test_leading_digit_prefixed(self) -> None:
        assert _safe_label("3M").startswith("_")

    def test_empty_returns_entity(self) -> None:
        assert _safe_label("") == "ENTITY"

    def test_clean_label_unchanged(self) -> None:
        assert _safe_label("ORGANIZATION") == "ORGANIZATION"


class TestPropsMap:
    def test_empty_dict(self) -> None:
        assert _props_map({}) == ""

    def test_string_value_quoted(self) -> None:
        result = _props_map({"name": "IBM"})
        assert "name: 'IBM'" in result

    def test_float_value_unquoted(self) -> None:
        result = _props_map({"weight": 1.5})
        assert "weight: 1.5" in result

    def test_single_quote_escaped(self) -> None:
        result = _props_map({"name": "O'Brien"})
        assert "O\\'Brien" in result


class TestToCypher:
    def test_contains_node_ids(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert "n1" in cypher
        assert "n2" in cypher

    def test_contains_node_names(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert "IBM" in cypher
        assert "Ginni Rometty" in cypher

    def test_contains_relationship_type(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert "CEO_OF" in cypher

    def test_uses_merge_not_create(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert "MERGE" in cypher
        assert "CREATE" not in cypher.upper().replace("// GENERATED", "")

    def test_node_label_from_type(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert ":ORGANIZATION" in cypher
        assert ":PERSON" in cypher

    def test_null_type_uses_entity_label(self) -> None:
        graph = KGGraph(nodes=[KGNode(id="x", name="X")], edges=[])
        cypher = Neo4jAdapter().to_cypher(graph)
        assert ":Entity" in cypher

    def test_description_included(self) -> None:
        cypher = Neo4jAdapter().to_cypher(_simple_graph())
        assert "A tech company." in cypher

    def test_provenance_fields_included(self) -> None:
        node = KGNode(
            id="c",
            name="IBM Canon",
            _merged_from=["a", "b"],
            _merge_strategy="rule_based",
        )
        graph = KGGraph(nodes=[node], edges=[])
        cypher = Neo4jAdapter().to_cypher(graph)
        assert "_merged_from" in cypher
        assert "_merge_strategy" in cypher

    def test_empty_graph(self) -> None:
        cypher = Neo4jAdapter().to_cypher(KGGraph(nodes=[], edges=[]))
        assert cypher.strip().startswith("//")

    def test_edge_weight_in_cypher(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="a", name="A"), KGNode(id="b", name="B")],
            edges=[KGEdge(source_id="a", target_id="b", relation="KNOWS", weight=2.5)],
        )
        cypher = Neo4jAdapter().to_cypher(graph)
        assert "2.5" in cypher


class TestDump:
    def test_dump_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "graph.cypher"
            Neo4jAdapter().dump(_simple_graph(), out)
            content = out.read_text()
        assert "IBM" in content
        assert "CEO_OF" in content

    def test_dump_to_dict(self) -> None:
        dest: dict = {}
        Neo4jAdapter().dump(_simple_graph(), dest)
        assert "cypher" in dest
        assert "IBM" in dest["cypher"]

    def test_load_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="from_neo4j"):
            Neo4jAdapter().load("anything")

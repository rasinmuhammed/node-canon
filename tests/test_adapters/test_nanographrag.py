"""Tests for the nano-graphrag adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nodecanon.adapters.nanographrag import NanoGraphRAGAdapter, _graphml_filename
from nodecanon.core.models import KGEdge, KGGraph, KGNode


def _simple_graph() -> KGGraph:
    return KGGraph(
        nodes=[
            KGNode(id="IBM", name="IBM", type="ORGANIZATION", description="A tech firm."),
            KGNode(id="WATSON_AI", name="Watson AI", type="PRODUCT"),
        ],
        edges=[
            KGEdge(source_id="IBM", target_id="WATSON_AI", relation="MAKES", weight=1.0),
        ],
    )


class TestGraphmlFilename:
    def test_default_namespace(self) -> None:
        assert _graphml_filename("chunk_entity_relation") == "graph_chunk_entity_relation.graphml"

    def test_custom_namespace(self) -> None:
        assert _graphml_filename("my_graph") == "graph_my_graph.graphml"


class TestRoundtrip:
    def test_save_and_load_from_dir(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            NanoGraphRAGAdapter.save(graph, tmpdir)
            loaded = NanoGraphRAGAdapter.from_working_dir(tmpdir)

        assert len(loaded.nodes) == len(graph.nodes)
        assert len(loaded.edges) == len(graph.edges)

    def test_node_attributes_preserved(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            NanoGraphRAGAdapter.save(graph, tmpdir)
            loaded = NanoGraphRAGAdapter.from_working_dir(tmpdir)

        by_id = {n.id: n for n in loaded.nodes}
        assert by_id["IBM"].type == "ORGANIZATION"
        assert by_id["IBM"].description == "A tech firm."

    def test_edge_attributes_preserved(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            NanoGraphRAGAdapter.save(graph, tmpdir)
            loaded = NanoGraphRAGAdapter.from_working_dir(tmpdir)

        assert loaded.edges[0].relation == "MAKES"
        assert loaded.edges[0].weight == pytest.approx(1.0)

    def test_custom_namespace_roundtrip(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = NanoGraphRAGAdapter(namespace="my_ns")
            adapter.dump(graph, tmpdir)
            loaded = NanoGraphRAGAdapter.from_working_dir(tmpdir, namespace="my_ns")

        assert len(loaded.nodes) == len(graph.nodes)


class TestFromWorkingDir:
    def test_missing_dir_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="working directory not found"):
            NanoGraphRAGAdapter.from_working_dir("/does/not/exist")

    def test_empty_dir_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(FileNotFoundError, match="No GraphML file found"):
            NanoGraphRAGAdapter.from_working_dir(tmpdir)

    def test_auto_detects_graphml(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save with a non-default namespace
            NanoGraphRAGAdapter(namespace="custom").dump(graph, tmpdir)
            # from_working_dir with default namespace auto-detects
            loaded = NanoGraphRAGAdapter.from_working_dir(tmpdir, namespace="custom")
        assert len(loaded.nodes) == 2


class TestFromInstance:
    def test_from_graphrag_instance(self) -> None:
        import networkx as nx

        G = nx.Graph()
        G.add_node("IBM", entity_type="ORGANIZATION", description="Tech company.")
        G.add_node("WATSON", entity_type="PRODUCT")
        G.add_edge("IBM", "WATSON", description="MAKES", weight=1.0)

        storage_mock = MagicMock()
        storage_mock._graph = G

        rag_mock = MagicMock()
        rag_mock.chunk_entity_relation_graph = storage_mock

        loaded = NanoGraphRAGAdapter.from_instance(rag_mock)
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1

    def test_no_attribute_raises(self) -> None:
        with pytest.raises(AttributeError, match="chunk_entity_relation_graph"):
            NanoGraphRAGAdapter.from_instance(object())

    def test_no_graph_attr_raises(self) -> None:
        storage_mock = MagicMock(spec=[])  # no _graph attribute
        rag_mock = MagicMock()
        rag_mock.chunk_entity_relation_graph = storage_mock

        with pytest.raises(AttributeError, match="_graph"):
            NanoGraphRAGAdapter.from_instance(rag_mock)


class TestLoadInterface:
    def test_load_from_directory(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            NanoGraphRAGAdapter.save(graph, tmpdir)
            loaded = NanoGraphRAGAdapter().load(tmpdir)
        assert len(loaded.nodes) == 2

    def test_load_from_graphml_file(self) -> None:
        graph = _simple_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            graphml_path = Path(tmpdir) / "graph_chunk_entity_relation.graphml"
            NanoGraphRAGAdapter().dump(graph, graphml_path)
            loaded = NanoGraphRAGAdapter().load(graphml_path)
        assert len(loaded.nodes) == 2

    def test_load_from_instance(self) -> None:
        import networkx as nx

        G = nx.Graph()
        G.add_node("A", entity_type="PERSON")
        G.add_node("B", entity_type="ORG")

        storage_mock = MagicMock()
        storage_mock._graph = G
        rag_mock = MagicMock()
        rag_mock.chunk_entity_relation_graph = storage_mock

        loaded = NanoGraphRAGAdapter().load(rag_mock)
        assert len(loaded.nodes) == 2

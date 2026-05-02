"""Tests for LightRAG GraphML adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nodecanon.adapters.lightrag import _DEFAULT_GRAPHML, LightRAGAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode


def _sample_graph() -> KGGraph:
    return KGGraph(
        nodes=[
            KGNode(
                id="IBM",
                name="IBM",
                type="ORGANIZATION",
                description="A tech company.",
                source_chunks=["chunk1", "chunk2"],
            ),
            KGNode(id="WATSON", name="WATSON", type="PRODUCT"),
        ],
        edges=[
            KGEdge(source_id="IBM", target_id="WATSON", relation="PRODUCT", weight=1.5),
        ],
    )


def _roundtrip(graph: KGGraph) -> KGGraph:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / _DEFAULT_GRAPHML
        LightRAGAdapter.to_graphml(graph, out)
        return LightRAGAdapter.from_graphml(out)


class TestFromGraphML:
    def test_node_count(self) -> None:
        assert len(_roundtrip(_sample_graph()).nodes) == 2

    def test_edge_count(self) -> None:
        assert len(_roundtrip(_sample_graph()).edges) == 1

    def test_node_type_preserved(self) -> None:
        idx = _roundtrip(_sample_graph()).node_index()
        assert idx["IBM"].type == "ORGANIZATION"

    def test_node_description_preserved(self) -> None:
        idx = _roundtrip(_sample_graph()).node_index()
        assert idx["IBM"].description == "A tech company."

    def test_source_chunks_preserved(self) -> None:
        idx = _roundtrip(_sample_graph()).node_index()
        assert "chunk1" in idx["IBM"].source_chunks
        assert "chunk2" in idx["IBM"].source_chunks

    def test_edge_relation_preserved(self) -> None:
        graph = _roundtrip(_sample_graph())
        assert graph.edges[0].relation == "PRODUCT"

    def test_edge_weight_preserved(self) -> None:
        graph = _roundtrip(_sample_graph())
        assert graph.edges[0].weight == pytest.approx(1.5)

    def test_node_name_equals_id_in_lightrag_convention(self) -> None:
        idx = _roundtrip(_sample_graph()).node_index()
        assert idx["IBM"].name == "IBM"

    def test_missing_graphml_raises_descriptive_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="GraphML file not found"):
            LightRAGAdapter.from_graphml("/nonexistent/path.graphml")


class TestFromWorkingDir:
    def test_loads_graph_from_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            LightRAGAdapter.to_graphml(_sample_graph(), d / _DEFAULT_GRAPHML)
            graph = LightRAGAdapter.from_working_dir(d)
        assert len(graph.nodes) == 2

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="working directory not found"):
            LightRAGAdapter.from_working_dir("/nonexistent/dir")

    def test_missing_graphml_in_dir_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            pytest.raises(FileNotFoundError, match=_DEFAULT_GRAPHML),
        ):
            LightRAGAdapter.from_working_dir(tmp)


class TestDump:
    def test_dump_to_directory_creates_graphml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            LightRAGAdapter().dump(_sample_graph(), d)
            assert (d / _DEFAULT_GRAPHML).exists()

    def test_dump_to_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "custom.graphml"
            LightRAGAdapter().dump(_sample_graph(), out)
            assert out.exists()

    def test_load_via_base_interface_from_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            LightRAGAdapter().dump(_sample_graph(), d)
            graph = LightRAGAdapter().load(d)
        assert len(graph.nodes) == 2

    def test_load_via_base_interface_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / _DEFAULT_GRAPHML
            LightRAGAdapter.to_graphml(_sample_graph(), out)
            graph = LightRAGAdapter().load(out)
        assert len(graph.nodes) == 2


class TestEdgeCases:
    def test_node_with_no_type(self) -> None:
        graph = KGGraph(nodes=[KGNode(id="X", name="X")], edges=[])
        restored = _roundtrip(graph)
        assert restored.node_index()["X"].type is None

    def test_node_with_extra_attributes(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="X", name="X", attributes={"score": 0.9})], edges=[]
        )
        restored = _roundtrip(graph)
        assert restored.node_index()["X"].attributes.get("score") == pytest.approx(0.9)

    def test_empty_graph_roundtrip(self) -> None:
        restored = _roundtrip(KGGraph(nodes=[], edges=[]))
        assert restored.nodes == []
        assert restored.edges == []

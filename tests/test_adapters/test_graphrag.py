"""Tests for Microsoft GraphRAG adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from nodecanon.adapters.graphrag import GraphRAGAdapter


def _write_parquets(
    directory: Path,
    entities_rows: list[dict],
    relationships_rows: list[dict],
) -> None:
    """Write minimal parquet files matching GraphRAG's output schema."""
    import pandas as pd

    pd.DataFrame(entities_rows).to_parquet(directory / "entities.parquet", index=False)
    pd.DataFrame(relationships_rows).to_parquet(
        directory / "relationships.parquet", index=False
    )


class TestGraphRAGAdapter:
    def test_load_returns_kggraph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[
                    {"id": "e1", "title": "IBM", "type": "ORGANIZATION"},
                    {"id": "e2", "title": "Ginni Rometty", "type": "PERSON"},
                ],
                relationships_rows=[
                    {
                        "id": "r1",
                        "source": "e2",
                        "target": "e1",
                        "description": "CEO_OF",
                    },
                ],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        node_ids = {n.id for n in graph.nodes}
        assert node_ids == {"e1", "e2"}

    def test_entity_types_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[
                    {"id": "e1", "title": "IBM", "type": "ORGANIZATION"},
                    {"id": "e2", "title": "Watson", "type": "PRODUCT"},
                ],
                relationships_rows=[],
            )
            graph = GraphRAGAdapter.from_directory(d)

        idx = graph.node_index()
        assert idx["e1"].type == "ORGANIZATION"
        assert idx["e2"].type == "PRODUCT"

    def test_missing_directory_raises_descriptive_error(self) -> None:
        with pytest.raises(FileNotFoundError, match="entities.parquet"):
            GraphRAGAdapter.from_directory("/nonexistent/path/that/does/not/exist")

    def test_description_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[
                    {
                        "id": "e1",
                        "title": "IBM",
                        "type": "ORGANIZATION",
                        "description": "A major tech company.",
                    }
                ],
                relationships_rows=[],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert graph.node_index()["e1"].description == "A major tech company."

    def test_edge_weight_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[
                    {"id": "a", "title": "A"},
                    {"id": "b", "title": "B"},
                ],
                relationships_rows=[
                    {"id": "r1", "source": "a", "target": "b", "weight": 3.5},
                ],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert graph.edges[0].weight == pytest.approx(3.5)

    def test_missing_relation_defaults_to_related_to(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[
                    {"id": "a", "title": "A"},
                    {"id": "b", "title": "B"},
                ],
                relationships_rows=[
                    {"id": "r1", "source": "a", "target": "b"},
                ],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert graph.edges[0].relation == "RELATED_TO"

    def test_nested_output_directory_layout(self) -> None:
        """GraphRAG v2 puts parquets inside an output/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            output = d / "output"
            output.mkdir()
            _write_parquets(
                output,
                entities_rows=[{"id": "e1", "title": "Node"}],
                relationships_rows=[],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert len(graph.nodes) == 1

    def test_load_method_delegates_to_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[{"id": "e1", "title": "X"}],
                relationships_rows=[],
            )
            adapter = GraphRAGAdapter()
            graph = adapter.load(d)

        assert len(graph.nodes) == 1

    def test_extra_entity_columns_go_into_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[{"id": "e1", "title": "IBM", "frequency": 42}],
                relationships_rows=[],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert graph.node_index()["e1"].attributes.get("frequency") == 42

    def test_empty_relationships_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_parquets(
                d,
                entities_rows=[{"id": "e1", "title": "IBM"}],
                relationships_rows=[],
            )
            graph = GraphRAGAdapter.from_directory(d)

        assert graph.edges == []

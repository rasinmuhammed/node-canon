"""Tests for the nodecanon CLI."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from nodecanon.cli.main import _save_graph, app
from nodecanon.core.models import KGEdge, KGGraph, KGNode

runner = CliRunner()


def _write_graphrag_parquets(directory: Path) -> None:
    import pandas as pd

    pd.DataFrame(
        [
            {"id": "e1", "title": "IBM", "type": "ORGANIZATION"},
            {"id": "e2", "title": "I.B.M.", "type": "ORGANIZATION"},
            {"id": "e3", "title": "Ginni Rometty", "type": "PERSON"},
        ]
    ).to_parquet(directory / "entities.parquet", index=False)
    pd.DataFrame(
        [{"id": "r1", "source": "e3", "target": "e1", "description": "CEO_OF"}]
    ).to_parquet(directory / "relationships.parquet", index=False)


def _simple_resolved_graph() -> KGGraph:
    return KGGraph(
        nodes=[
            KGNode(
                id="n1",
                name="IBM",
                type="ORGANIZATION",
                _merged_from=["n1", "n2"],
                _merge_strategy="rule_based",
            ),
            KGNode(id="n3", name="Ginni Rometty", type="PERSON"),
        ],
        edges=[KGEdge(source_id="n3", target_id="n1", relation="CEO_OF")],
    )


# ---------------------------------------------------------------------------
# _save_graph / round-trip JSON
# ---------------------------------------------------------------------------


class TestSaveGraph:
    def test_creates_node_and_edge_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _save_graph(_simple_resolved_graph(), d)
            assert (d / "resolved_nodes.json").exists()
            assert (d / "resolved_edges.json").exists()

    def test_nodes_json_contains_all_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _save_graph(_simple_resolved_graph(), d)
            nodes = json.loads((d / "resolved_nodes.json").read_text())
        ids = {n["id"] for n in nodes}
        assert ids == {"n1", "n3"}

    def test_provenance_fields_serialised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _save_graph(_simple_resolved_graph(), d)
            nodes = json.loads((d / "resolved_nodes.json").read_text())
        merged = next(n for n in nodes if n["id"] == "n1")
        assert merged["_merged_from"] == ["n1", "n2"]
        assert merged["_merge_strategy"] == "rule_based"

    def test_edges_json_has_relation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _save_graph(_simple_resolved_graph(), d)
            edges = json.loads((d / "resolved_edges.json").read_text())
        assert edges[0]["relation"] == "CEO_OF"


# ---------------------------------------------------------------------------
# CLI: resolve command
# ---------------------------------------------------------------------------


class TestResolveCommand:
    def test_resolve_exit_code_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input"
            out = Path(tmp) / "output"
            inp.mkdir()
            _write_graphrag_parquets(inp)
            result = runner.invoke(app, ["resolve", str(inp), "--output", str(out)])
        assert result.exit_code == 0, result.output

    def test_resolve_creates_output_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input"
            out = Path(tmp) / "output"
            inp.mkdir()
            _write_graphrag_parquets(inp)
            runner.invoke(app, ["resolve", str(inp), "--output", str(out)])
            assert (out / "resolved_nodes.json").exists()
            assert (out / "resolved_edges.json").exists()
            assert (out / "merge_report.json").exists()

    def test_resolve_report_has_expected_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input"
            out = Path(tmp) / "output"
            inp.mkdir()
            _write_graphrag_parquets(inp)
            runner.invoke(app, ["resolve", str(inp), "--output", str(out)])
            report = json.loads((out / "merge_report.json").read_text())
            assert "original_node_count" in report
            assert "resolved_node_count" in report
            assert "merge_records" in report
            assert "conflicts" in report

    def test_resolve_output_mentions_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inp = Path(tmp) / "input"
            out = Path(tmp) / "output"
            inp.mkdir()
            _write_graphrag_parquets(inp)
            result = runner.invoke(app, ["resolve", str(inp), "--output", str(out)])
        assert "node" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    def _setup(self, tmp: str) -> Path:
        d = Path(tmp)
        inp = d / "input"
        out = d / "output"
        inp.mkdir()
        _write_graphrag_parquets(inp)
        runner.invoke(app, ["resolve", str(inp), "--output", str(out)])
        return out

    def test_inspect_exit_code_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._setup(tmp)
            result = runner.invoke(app, ["inspect", str(out)])
        assert result.exit_code == 0, result.output

    def test_inspect_shows_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self._setup(tmp)
            result = runner.invoke(app, ["inspect", str(out)])
        assert "→" in result.output

    def test_inspect_missing_report_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(app, ["inspect", tmp])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI: explain command
# ---------------------------------------------------------------------------


class TestExplainCommand:
    def _setup_with_merge(self, tmp: str) -> tuple[Path, str]:
        d = Path(tmp)
        inp = d / "input"
        out = d / "output"
        inp.mkdir()
        _write_graphrag_parquets(inp)
        runner.invoke(
            app, ["resolve", str(inp), "--output", str(out), "--threshold", "0.0"]
        )
        report = json.loads((out / "merge_report.json").read_text())
        if report["merge_records"]:
            canonical_id = report["merge_records"][0]["canonical_id"]
        else:
            nodes = json.loads((out / "resolved_nodes.json").read_text())
            canonical_id = nodes[0]["id"]
        return out, canonical_id

    def test_explain_exit_code_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, node_id = self._setup_with_merge(tmp)
            result = runner.invoke(app, ["explain", node_id, str(out)])
        assert result.exit_code == 0, result.output

    def test_explain_shows_node_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, node_id = self._setup_with_merge(tmp)
            nodes = json.loads((out / "resolved_nodes.json").read_text())
            node_name = next(n["name"] for n in nodes if n["id"] == node_id)
            result = runner.invoke(app, ["explain", node_id, str(out)])
        assert node_name in result.output

    def test_explain_unknown_node_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out, _ = self._setup_with_merge(tmp)
            result = runner.invoke(app, ["explain", "nonexistent_id_xyz", str(out)])
        assert result.exit_code != 0

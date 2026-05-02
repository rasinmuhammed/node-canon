from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from nodecanon.adapters.graphrag import GraphRAGAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode
from nodecanon.core.resolver import Resolver

app = typer.Typer(
    name="nodecanon",
    help="Entity resolution for LLM-extracted knowledge graphs.",
    no_args_is_help=True,
)

# File names written by the resolve command.
_NODES_FILE = "resolved_nodes.json"
_EDGES_FILE = "resolved_edges.json"
_REPORT_FILE = "merge_report.json"


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


@app.command()
def resolve(
    input_dir: Path = typer.Argument(..., help="Path to GraphRAG output directory."),  # noqa: B008
    output: Path = typer.Option(  # noqa: B008
        Path("./resolved/"), "--output", "-o", help="Output directory."
    ),
    threshold: float = typer.Option(0.75, "--threshold", "-t"),  # noqa: B008
    llm: bool = typer.Option(False, "--llm", help="Enable LLM-assisted matching."),  # noqa: B008
) -> None:
    """Resolve and deduplicate entities in a GraphRAG knowledge graph."""
    typer.echo(f"Loading graph from {input_dir} …")
    graph = GraphRAGAdapter.from_directory(input_dir)
    typer.echo(f"  Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges.")

    if llm:
        from nodecanon.core.matching import LLMAssistedMatcher

        matcher = LLMAssistedMatcher(threshold=threshold)
    else:
        from nodecanon.core.matching import RuleBasedMatcher

        matcher = RuleBasedMatcher(threshold=threshold)

    typer.echo("Running entity resolution …")
    resolver = Resolver(matcher=matcher)
    result = resolver.resolve(graph)

    typer.echo(result.merge_report())

    output.mkdir(parents=True, exist_ok=True)
    _save_graph(result.graph, output)

    report_data = {
        "original_node_count": result.original_node_count,
        "original_edge_count": result.original_edge_count,
        "resolved_node_count": len(result.graph.nodes),
        "resolved_edge_count": len(result.graph.edges),
        "merge_records": [
            {
                "canonical_id": r.canonical_id,
                "merged_ids": r.merged_ids,
                "strategy": r.strategy,
                "score": asdict(r.score),
            }
            for r in result.merge_records
        ],
        "conflicts": [
            {
                "node_id_a": c.node_id_a,
                "node_id_b": c.node_id_b,
                "conflict_reason": c.conflict_reason,
                "score": asdict(c.score),
            }
            for c in result.conflicts
        ],
    }
    (output / _REPORT_FILE).write_text(
        json.dumps(report_data, indent=2), encoding="utf-8"
    )

    typer.echo(f"\nOutput written to {output}/")
    typer.echo(f"  {_NODES_FILE}  — {len(result.graph.nodes)} canonical nodes")
    typer.echo(f"  {_EDGES_FILE}  — {len(result.graph.edges)} edges")
    typer.echo(f"  {_REPORT_FILE} — merge records and conflicts")


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    input_dir: Path = typer.Argument(..., help="Path to resolved output directory."),  # noqa: B008
) -> None:
    """Show a summary of a previously resolved graph."""
    report_path = input_dir / _REPORT_FILE
    if not report_path.exists():
        typer.echo(
            f"No {_REPORT_FILE} found in {input_dir}. Run 'nodecanon resolve' first.",
            err=True,
        )
        raise typer.Exit(1)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    orig = report["original_node_count"]
    resolved = report["resolved_node_count"]
    reduction = (1 - resolved / orig) * 100 if orig else 0.0
    n_merges = len(report["merge_records"])
    n_conflicts = len(report["conflicts"])
    aliases = sum(len(r["merged_ids"]) for r in report["merge_records"])

    typer.echo(
        f"→ {orig} nodes → {resolved} canonical nodes ({reduction:.1f}% reduction)"
    )
    typer.echo(f"→ {aliases} alias nodes absorbed in {n_merges} merge group(s)")
    typer.echo(f"→ {n_conflicts} conflict(s) flagged for human review")
    typer.echo(
        f"→ {report['original_edge_count']} edges → {report['resolved_edge_count']} edges"
    )

    if n_conflicts > 0:
        typer.echo("\nConflicts:")
        for c in report["conflicts"]:
            typer.echo(
                f"  [{c['node_id_a']} ↔ {c['node_id_b']}] {c['conflict_reason']}"
            )


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


@app.command()
def explain(
    node_id: str = typer.Argument(..., help="Canonical node ID to explain."),  # noqa: B008
    input_dir: Path = typer.Argument(..., help="Path to resolved output directory."),  # noqa: B008
) -> None:
    """Explain why a canonical node was formed from its aliases."""
    nodes_path = input_dir / _NODES_FILE
    report_path = input_dir / _REPORT_FILE

    for p in (nodes_path, report_path):
        if not p.exists():
            typer.echo(f"File not found: {p}. Run 'nodecanon resolve' first.", err=True)
            raise typer.Exit(1)

    nodes_data: list[dict] = json.loads(nodes_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    node_by_id = {n["id"]: n for n in nodes_data}
    if node_id not in node_by_id:
        typer.echo(f"Node '{node_id}' not found in resolved graph.", err=True)
        raise typer.Exit(1)

    node = node_by_id[node_id]
    merge_record = next(
        (r for r in report["merge_records"] if r["canonical_id"] == node_id), None
    )

    typer.echo(f"Canonical node: {node['name']}  (id: {node_id})")
    typer.echo(f"  type:        {node.get('type', 'unknown')}")
    if node.get("description"):
        typer.echo(f"  description: {node['description'][:120]}")

    if merge_record is None:
        typer.echo("\nThis node was not merged — it is an original unique node.")
        return

    aliases = merge_record["merged_ids"]
    typer.echo(f"\nMerged from {len(aliases) + 1} nodes:")
    typer.echo(f"  [canonical] {node['name']} (id: {node_id})")
    for aid in aliases:
        anode = node_by_id.get(aid)
        aname = anode["name"] if anode else aid
        typer.echo(f"  [alias]     {aname} (id: {aid})")

    sv = merge_record["score"]
    ws = (
        sv["name_similarity"] * 0.30
        + sv["semantic_similarity"] * 0.25
        + sv["type_agreement"] * 0.20
        + sv["neighbor_overlap"] * 0.20
        + sv["description_similarity"] * 0.05
    )
    typer.echo("\nMerge evidence (highest scoring pair):")
    typer.echo(f"  name_similarity:        {sv['name_similarity']:.2f}")
    typer.echo(f"  semantic_similarity:    {sv['semantic_similarity']:.2f}")
    typer.echo(f"  type_agreement:         {sv['type_agreement']:.2f}")
    typer.echo(f"  neighbor_overlap:       {sv['neighbor_overlap']:.2f}")
    typer.echo(f"  description_similarity: {sv['description_similarity']:.2f}")
    typer.echo(f"  {'─' * 36}")
    typer.echo(
        f"  weighted score:         {ws:.2f}  (strategy: {merge_record['strategy']})"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_graph(graph: KGGraph, directory: Path) -> None:
    """Serialise nodes and edges to JSON files."""

    def _node_dict(n: KGNode) -> dict:
        d: dict = {"id": n.id, "name": n.name}
        if n.type is not None:
            d["type"] = n.type
        if n.description is not None:
            d["description"] = n.description
        if n.source_chunks:
            d["source_chunks"] = n.source_chunks
        if n.attributes:
            d["attributes"] = n.attributes
        if n._merged_from is not None:
            d["_merged_from"] = n._merged_from
        if n._merge_strategy is not None:
            d["_merge_strategy"] = n._merge_strategy
        if n._resolved_types is not None:
            d["_resolved_types"] = n._resolved_types
        return d

    def _edge_dict(e: KGEdge) -> dict:
        d: dict = {
            "source_id": e.source_id,
            "target_id": e.target_id,
            "relation": e.relation,
            "weight": e.weight,
        }
        if e.attributes:
            d["attributes"] = e.attributes
        return d

    (directory / _NODES_FILE).write_text(
        json.dumps([_node_dict(n) for n in graph.nodes], indent=2), encoding="utf-8"
    )
    (directory / _EDGES_FILE).write_text(
        json.dumps([_edge_dict(e) for e in graph.edges], indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    app()

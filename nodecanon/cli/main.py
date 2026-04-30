from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="nodecanon",
    help="Entity resolution for LLM-extracted knowledge graphs.",
    no_args_is_help=True,
)


@app.command()
def resolve(
    input_dir: Path = typer.Argument(..., help="Path to GraphRAG output directory."),
    output: Path = typer.Option(
        Path("./resolved/"), "--output", "-o", help="Output directory."
    ),
    threshold: float = typer.Option(0.75, "--threshold", "-t"),
    llm: bool = typer.Option(False, "--llm", help="Enable LLM-assisted matching."),
) -> None:
    """Resolve and deduplicate entities in a knowledge graph."""
    raise NotImplementedError


@app.command()
def inspect(
    input_dir: Path = typer.Argument(..., help="Path to resolved graph directory."),
) -> None:
    """Show a summary of merge records and conflicts."""
    raise NotImplementedError


@app.command()
def explain(
    node_id: str = typer.Argument(..., help="Canonical node ID to explain."),
    input_dir: Path = typer.Argument(..., help="Path to resolved graph directory."),
) -> None:
    """Explain why a canonical node was formed from its aliases."""
    raise NotImplementedError


if __name__ == "__main__":
    app()

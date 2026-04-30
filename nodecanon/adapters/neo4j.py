from __future__ import annotations

from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGGraph


class Neo4jAdapter(BaseAdapter):
    """Export-only: write resolved KGGraph as Cypher CREATE statements."""

    def load(self, source: Path | str | Any) -> KGGraph:
        raise NotImplementedError("Neo4jAdapter is export-only.")

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        raise NotImplementedError

    def to_cypher(self, graph: KGGraph) -> str:
        raise NotImplementedError

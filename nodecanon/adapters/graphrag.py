from __future__ import annotations

from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGGraph


class GraphRAGAdapter(BaseAdapter):
    """Read/write Microsoft GraphRAG parquet output files."""

    def load(self, source: Path | str | Any) -> KGGraph:
        raise NotImplementedError

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        raise NotImplementedError

    @classmethod
    def from_directory(cls, directory: Path | str) -> KGGraph:
        raise NotImplementedError

from __future__ import annotations

from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGGraph


class LlamaIndexAdapter(BaseAdapter):
    """Convert between LlamaIndex PropertyGraphIndex and KGGraph."""

    def load(self, source: Path | str | Any) -> KGGraph:
        raise NotImplementedError

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        raise NotImplementedError

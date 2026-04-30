"""Resolve a Microsoft GraphRAG output directory."""
from __future__ import annotations

from pathlib import Path

from nodecanon import Resolver
from nodecanon.adapters import GraphRAGAdapter

graph = GraphRAGAdapter().load(Path("./graphrag_output/"))
result = Resolver().resolve(graph)
print(result.merge_report())

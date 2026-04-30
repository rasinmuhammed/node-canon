"""Resolve a LlamaIndex PropertyGraphIndex."""
from __future__ import annotations

from nodecanon import Resolver
from nodecanon.adapters import LlamaIndexAdapter

# Assumes you have a LlamaIndex PropertyGraphIndex object called `index`
graph = LlamaIndexAdapter().load(None)  # replace None with your index
result = Resolver().resolve(graph)
print(result.merge_report())

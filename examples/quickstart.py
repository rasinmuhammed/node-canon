"""Quickstart: resolve a manually constructed KGGraph."""
from __future__ import annotations

from nodecanon import KGEdge, KGGraph, KGNode, Resolver

nodes = [
    KGNode(id="n1", name="IBM", type="ORGANIZATION"),
    KGNode(id="n2", name="I.B.M.", type="COMPANY"),
    KGNode(id="n3", name="International Business Machines", type="ORGANIZATION"),
    KGNode(id="n4", name="Ginni Rometty", type="PERSON"),
]
edges = [
    KGEdge(source_id="n1", target_id="n4", relation="CEO_OF"),
    KGEdge(source_id="n2", target_id="n4", relation="CEO_OF"),
    KGEdge(source_id="n3", target_id="n4", relation="CEO_OF"),
]
graph = KGGraph(nodes=nodes, edges=edges)

resolver = Resolver()
result = resolver.resolve(graph)

print(result.merge_report())

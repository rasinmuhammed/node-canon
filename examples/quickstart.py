"""
nodecanon quickstart — entity resolution on a hand-built graph.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

from nodecanon import KGEdge, KGGraph, KGNode, Resolver

# The mess: four nodes that are all IBM, one that isn't.
# In a real GraphRAG graph these come from different chunks
# and have no idea they refer to the same company.
graph = KGGraph(
    nodes=[
        KGNode(id="n1", name="IBM", type="ORGANIZATION"),
        KGNode(id="n2", name="I.B.M.", type="COMPANY"),
        KGNode(id="n3", name="International Business Machines", type="ORGANIZATION"),
        KGNode(id="n4", name="IBM Corp", type="ORGANIZATION"),
        KGNode(id="n5", name="Ginni Rometty", type="PERSON"),
        KGNode(id="n6", name="Watson", type="PRODUCT"),
    ],
    edges=[
        # All IBM variants connect to the same real-world entities.
        # This is the structural signal nodecanon uses alongside name similarity.
        KGEdge(source_id="n1", target_id="n5", relation="HAS_CEO"),
        KGEdge(source_id="n2", target_id="n5", relation="HAS_CEO"),
        KGEdge(source_id="n3", target_id="n5", relation="HAS_CEO"),
        KGEdge(source_id="n4", target_id="n5", relation="HAS_CEO"),
        KGEdge(source_id="n1", target_id="n6", relation="MAKES"),
        KGEdge(source_id="n3", target_id="n6", relation="MAKES"),
    ],
)

print(f"Before: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

result = Resolver().resolve(graph)

print(f"After:  {len(result.graph.nodes)} nodes, {len(result.graph.edges)} edges")
print()
print(result.merge_report())

# Inspect the canonical IBM node.
canonical = next(n for n in result.graph.nodes if n._merged_from)
print(f"\nCanonical node:  {canonical.name!r}")
print(f"Merged from:     {canonical._merged_from}")
print(f"Strategy:        {canonical._merge_strategy}")
print(f"Score:           {canonical._merge_evidence}")

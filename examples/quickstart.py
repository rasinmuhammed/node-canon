"""
nodecanon quickstart — complete walkthrough of the API.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

from nodecanon import GraphBuilder, KGGraph, Resolver

# ──────────────────────────────────────────────────────────────────────────────
# 1. Build a graph
# ──────────────────────────────────────────────────────────────────────────────
#
# This simulates what a GraphRAG pipeline produces: the same real-world company
# extracted under four different surface forms, with no knowledge that they
# refer to the same entity.

print("─" * 60)
print("  Building graph")
print("─" * 60)

# Option A: fluent builder (great for programmatic construction)
graph = (
    GraphBuilder()
    # Four names for one company
    .add_node("IBM",                              type="ORGANIZATION")
    .add_node("I.B.M.",                          type="COMPANY")
    .add_node("International Business Machines", type="ORGANIZATION")
    .add_node("IBM Corp",                        type="ORGANIZATION")
    # Structural context — all IBM variants connect to the same real entities
    .add_node("Watson AI",       type="PRODUCT")
    .add_node("Ginni Rometty",   type="PERSON")
    .add_node("Armonk, New York", type="LOCATION")
    # One unrelated company
    .add_node("Microsoft",       type="ORGANIZATION")
    .add_node("Azure",           type="PRODUCT")
    .add_edge("Microsoft", "Azure", "MAKES")
    # IBM variants all connect to the same anchors — this is the topology signal
    .add_edge("IBM",                              "Watson AI",        "MAKES")
    .add_edge("IBM",                              "Ginni Rometty",    "HAS_CEO")
    .add_edge("IBM",                              "Armonk, New York", "HEADQUARTERED_IN")
    .add_edge("I.B.M.",                          "Watson AI",        "MAKES")
    .add_edge("I.B.M.",                          "Ginni Rometty",    "HAS_CEO")
    .add_edge("I.B.M.",                          "Armonk, New York", "HEADQUARTERED_IN")
    .add_edge("International Business Machines", "Watson AI",        "MAKES")
    .add_edge("International Business Machines", "Ginni Rometty",    "HAS_CEO")
    .add_edge("IBM Corp",                        "Armonk, New York", "HEADQUARTERED_IN")
    .build()
)

print(f"  {graph}")

# Option B: from dicts (great when loading from JSON / a database)
graph_b = KGGraph.from_dicts(
    nodes=[
        {"name": "IBM",   "type": "ORGANIZATION", "founded": 1911, "country": "USA"},
        {"name": "I.B.M.", "type": "COMPANY"},
    ],
    edges=[],
)
print(f"  from_dicts: {graph_b}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 2. Resolve
# ──────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Resolving (with embeddings — downloads model on first run)")
print("─" * 60)

result = Resolver().resolve(graph)
print()
print(result.merge_report())
print()

# ──────────────────────────────────────────────────────────────────────────────
# 3. Inspect results
# ──────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Inspecting canonical nodes")
print("─" * 60)

for node in result.graph.nodes:
    if node._merged_from and len(node._merged_from) > 1:
        print(f"\n  Canonical: {node.name!r}")
        print(f"  Merged from: {node._merged_from}")
        print(f"  Strategy: {node._merge_strategy}")
        score = node._merge_evidence or {}
        if score:
            print(f"  Evidence: name={score.get('name_similarity', 0):.2f}, "
                  f"semantic={score.get('semantic_similarity', 0):.2f}, "
                  f"neighbor={score.get('neighbor_overlap', 0):.2f}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Explain a merge
# ──────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  explain() — full merge breakdown")
print("─" * 60)

if result.merge_records:
    record = result.merge_records[0]
    print()
    print(result.explain(record.canonical_id))
    print()

# ──────────────────────────────────────────────────────────────────────────────
# 5. Post-resolution editing
# ──────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Post-resolution editing")
print("─" * 60)

if result.merge_records:
    record = result.merge_records[0]

    # Reject a merge — restores original nodes as separate entities
    corrected = result.reject_merge(record.canonical_id)
    print(f"\n  reject_merge: {len(result.graph.nodes)} nodes → {len(corrected.graph.nodes)} nodes")

    # Force a merge — manually merge nodes the resolver didn't
    node_ids = [n.id for n in result.graph.nodes if not n._merged_from]
    if len(node_ids) >= 2:
        forced = result.force_merge(node_ids[0], node_ids[1])
        print(f"  force_merge:  {len(result.graph.nodes)} nodes → {len(forced.graph.nodes)} nodes")

# Accept conflicts (none in this graph, but the API is the same)
print(f"  Conflicts flagged for review: {len(result.conflicts)}")
if result.conflicts:
    for i, c in enumerate(result.conflicts):
        print(f"    [{i}] {c.node_id_a} vs {c.node_id_b}: {c.conflict_reason}")
    accepted = result.accept_conflict(0)
    print(f"  accept_conflict(0): {len(accepted.conflicts)} conflicts remaining")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 6. Using the resolved graph
# ──────────────────────────────────────────────────────────────────────────────

print("─" * 60)
print("  Resolved graph")
print("─" * 60)
print(f"\n  {result.graph}")
print(f"  Nodes: {[n.name for n in result.graph.nodes]}")
print(f"  Edges: {len(result.graph.edges)}")
print()
print("Done. Pass result.graph to your downstream pipeline.")

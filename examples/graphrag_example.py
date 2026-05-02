"""
nodecanon + Microsoft GraphRAG — full workflow.

Prerequisites:
    pip install nodecanon
    # Run your GraphRAG indexing pipeline first, then:

    nodecanon resolve ./graphrag_output/ --output ./resolved/
    nodecanon inspect ./resolved/

Or do it in Python:

    python examples/graphrag_example.py
"""

from __future__ import annotations

from pathlib import Path

from nodecanon.adapters.graphrag import GraphRAGAdapter
from nodecanon.adapters.neo4j import Neo4jAdapter
from nodecanon.core.resolver import Resolver

GRAPHRAG_OUTPUT = Path("./graphrag_output/")  # change this
RESOLVED_DIR = Path("./resolved/")
NEO4J_CYPHER = RESOLVED_DIR / "graph.cypher"


def main() -> None:
    if not GRAPHRAG_OUTPUT.exists():
        print(f"No GraphRAG output found at {GRAPHRAG_OUTPUT}.")
        print("Run `graphrag index` first, then point GRAPHRAG_OUTPUT at the result.")
        return

    print("Loading graph …")
    graph = GraphRAGAdapter.from_directory(GRAPHRAG_OUTPUT)
    print(f"  {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    print("Resolving …")
    result = Resolver().resolve(graph)
    print(result.merge_report())

    # Write Cypher so you can load the clean graph into Neo4j.
    RESOLVED_DIR.mkdir(parents=True, exist_ok=True)
    Neo4jAdapter().dump(result.graph, NEO4J_CYPHER)
    print(f"\nCypher written to {NEO4J_CYPHER}")
    print("Load it with:  cypher-shell < resolved/graph.cypher")

    # Or inspect any merge decision programmatically.
    if result.merge_records:
        first = result.merge_records[0]
        canonical = result.graph.node_index()[first.canonical_id]
        print(f"\nExample merge — {canonical.name!r}")
        print(f"  absorbed:  {first.merged_ids}")
        print(f"  score:     weighted={first.score.weighted_sum():.3f}")


if __name__ == "__main__":
    main()

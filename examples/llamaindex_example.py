"""
nodecanon + LlamaIndex PropertyGraphIndex — full workflow.

Prerequisites:
    pip install nodecanon[llamaindex]
    pip install llama-index-core llama-index-graph-stores-simple

Run:
    python examples/llamaindex_example.py
"""

from __future__ import annotations

from nodecanon.adapters.llamaindex import LlamaIndexAdapter
from nodecanon.core.resolver import Resolver


def resolve_property_graph_index(index):  # type: ignore[no-untyped-def]
    """
    Resolve entity duplicates in a LlamaIndex PropertyGraphIndex.

    Parameters
    ----------
    index : PropertyGraphIndex
        Your existing LlamaIndex index. Build it with your documents first:

            from llama_index.core import PropertyGraphIndex
            index = PropertyGraphIndex.from_documents(documents)

    Returns
    -------
    PropertyGraphIndex
        A new index backed by the resolved graph.  The original is untouched.
    """
    adapter = LlamaIndexAdapter()

    # Pull the graph out of LlamaIndex's internal store.
    graph = adapter.load(index)
    print(f"Loaded {len(graph.nodes)} nodes, {len(graph.edges)} edges from index")

    # Resolve.
    result = Resolver().resolve(graph)
    print(result.merge_report())

    # Push back into a new PropertyGraphIndex.
    resolved_index = adapter.to_property_graph_index(result.graph)
    return resolved_index


# ── Demo with a tiny synthetic graph ──────────────────────────────────────────


def _demo() -> None:
    """
    Demonstrates the adapter without requiring a real LlamaIndex document set.
    Builds a minimal PropertyGraphIndex manually and resolves it.
    """
    try:
        from llama_index.core.graph_stores.simple_labelled import (
            SimplePropertyGraphStore,
        )
        from llama_index.core.graph_stores.types import EntityNode, Relation
        from llama_index.core.indices.property_graph import PropertyGraphIndex
    except ImportError:
        print("Install llama-index-core first:  pip install nodecanon[llamaindex]")
        return

    store = SimplePropertyGraphStore()
    store.upsert_nodes(
        [
            EntityNode(name="IBM", label="ORGANIZATION"),
            EntityNode(name="I.B.M.", label="COMPANY"),
            EntityNode(name="International Business Machines", label="ORGANIZATION"),
            EntityNode(name="Ginni Rometty", label="PERSON"),
        ]
    )

    ibm = EntityNode(name="IBM", label="ORGANIZATION")
    ibm2 = EntityNode(name="I.B.M.", label="COMPANY")
    ibm3 = EntityNode(name="International Business Machines", label="ORGANIZATION")
    ginni = EntityNode(name="Ginni Rometty", label="PERSON")

    store.upsert_relations(
        [
            Relation(label="HAS_CEO", source_id=ibm.id, target_id=ginni.id),
            Relation(label="HAS_CEO", source_id=ibm2.id, target_id=ginni.id),
            Relation(label="HAS_CEO", source_id=ibm3.id, target_id=ginni.id),
        ]
    )

    index = PropertyGraphIndex.from_existing(
        property_graph_store=store, show_progress=False
    )

    resolved = resolve_property_graph_index(index)
    print(
        f"\nResolved index has store type: {type(resolved.property_graph_store).__name__}"
    )


if __name__ == "__main__":
    _demo()

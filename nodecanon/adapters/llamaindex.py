from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode

if TYPE_CHECKING:
    pass


class LlamaIndexAdapter(BaseAdapter):
    """Convert between LlamaIndex PropertyGraphIndex / graph stores and KGGraph.

    LlamaIndex is an optional dependency.  Install it with::

        pip install nodecanon[llamaindex]

    Supported sources
    -----------------
    ``from_property_graph_index(index)``
        Accepts a ``PropertyGraphIndex`` instance.

    ``from_graph_store(store)``
        Accepts any LlamaIndex graph store that exposes ``get_triplets()``
        or ``get_rel_map()`` — e.g. ``SimplePropertyGraphStore``,
        ``Neo4jPropertyGraphStore``.

    ``to_property_graph_index(graph)``
        Returns a ``PropertyGraphIndex`` backed by a
        ``SimplePropertyGraphStore`` populated from the KGGraph.
    """

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self, source: Any) -> KGGraph:
        """Load from a PropertyGraphIndex or graph store."""
        try:
            from llama_index.core.indices.property_graph import PropertyGraphIndex
        except ImportError:
            raise ImportError(
                "llama-index-core is required for LlamaIndexAdapter. "
                "Install it with: pip install nodecanon[llamaindex]"
            ) from None

        if isinstance(source, PropertyGraphIndex):
            return self.from_property_graph_index(source)
        return self.from_graph_store(source)

    def dump(self, graph: KGGraph, destination: Any) -> None:
        """Write KGGraph into *destination* (a PropertyGraphIndex or graph store)."""
        try:
            from llama_index.core.indices.property_graph import PropertyGraphIndex
        except ImportError:
            raise ImportError(
                "llama-index-core is required for LlamaIndexAdapter. "
                "Install it with: pip install nodecanon[llamaindex]"
            ) from None

        index = self.to_property_graph_index(graph)
        if isinstance(destination, PropertyGraphIndex):
            destination.property_graph_store = index.property_graph_store
        elif isinstance(destination, dict):
            destination["index"] = index
        else:
            destination.index = index

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_property_graph_index(cls, index: Any) -> KGGraph:
        """Convert a ``PropertyGraphIndex`` to a KGGraph."""
        _require_llamaindex()
        return cls.from_graph_store(index.property_graph_store)

    @classmethod
    def from_graph_store(cls, store: Any) -> KGGraph:
        """Convert a LlamaIndex graph store to a KGGraph.

        Works with any store that implements the ``get_triplets()``
        or ``get_rel_map()`` protocol from llama-index-core.
        """
        _require_llamaindex()
        from llama_index.core.graph_stores.types import Relation

        node_map: dict[str, KGNode] = {}
        edges: list[KGEdge] = []

        triplets = _get_triplets(store)

        for subj, rel, obj in triplets:
            for li_node in (subj, obj):
                node_id = str(li_node.id)
                if node_id not in node_map:
                    node_map[node_id] = _entity_node_to_kgnode(li_node)

            if isinstance(rel, Relation):
                edges.append(
                    KGEdge(
                        source_id=str(subj.id),
                        target_id=str(obj.id),
                        relation=str(rel.label or "RELATED_TO"),
                        weight=float(rel.properties.get("weight", 1.0)),
                        attributes={
                            k: v for k, v in rel.properties.items() if k != "weight"
                        },
                    )
                )
            else:
                # Plain string relation label
                edges.append(
                    KGEdge(
                        source_id=str(subj.id),
                        target_id=str(obj.id),
                        relation=str(rel) if rel else "RELATED_TO",
                    )
                )

        return KGGraph(nodes=list(node_map.values()), edges=edges)

    @classmethod
    def to_property_graph_index(cls, graph: KGGraph) -> Any:
        """Build a ``PropertyGraphIndex`` from a KGGraph (in-memory store)."""
        _require_llamaindex()
        from llama_index.core.graph_stores.simple_labelled import (
            SimplePropertyGraphStore,
        )
        from llama_index.core.graph_stores.types import EntityNode, Relation
        from llama_index.core.indices.property_graph import PropertyGraphIndex

        store = SimplePropertyGraphStore()

        li_nodes: list[Any] = []
        for node in graph.nodes:
            li_node = EntityNode(
                name=node.name,
                label=node.type or "ENTITY",
                properties=_kgnode_extra_props(node),
            )
            li_nodes.append(li_node)

        li_rels: list[Any] = []
        node_index = graph.node_index()
        name_by_id = {n.id: n.name for n in graph.nodes}
        for edge in graph.edges:
            src_name = name_by_id.get(edge.source_id, edge.source_id)
            tgt_name = name_by_id.get(edge.target_id, edge.target_id)
            src_type = (
                (node_index[edge.source_id].type or "ENTITY")
                if edge.source_id in node_index
                else "ENTITY"
            )
            tgt_type = (
                (node_index[edge.target_id].type or "ENTITY")
                if edge.target_id in node_index
                else "ENTITY"
            )

            rel = Relation(
                label=edge.relation,
                source_id=EntityNode(name=src_name, label=src_type).id,
                target_id=EntityNode(name=tgt_name, label=tgt_type).id,
                properties={"weight": edge.weight, **edge.attributes},
            )
            li_rels.append(rel)

        store.upsert_nodes(li_nodes)
        store.upsert_relations(li_rels)

        return PropertyGraphIndex.from_existing(
            property_graph_store=store,
            show_progress=False,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _require_llamaindex() -> None:
    try:
        import llama_index.core  # noqa: F401
    except ImportError:
        raise ImportError(
            "llama-index-core is required for LlamaIndexAdapter. "
            "Install it with: pip install nodecanon[llamaindex]"
        ) from None


def _get_triplets(store: Any) -> list[tuple[Any, Any, Any]]:
    """Extract (subject, relation, object) triplets from a graph store."""
    if hasattr(store, "get_triplets"):
        return store.get_triplets() or []
    # Fallback: iterate node-level rel maps
    if hasattr(store, "get_rel_map"):
        triplets: list[tuple[Any, Any, Any]] = []
        rel_map = store.get_rel_map() or {}
        for subj, relations in rel_map.items():
            for rel, obj in relations:
                triplets.append((subj, rel, obj))
        return triplets
    raise AttributeError(
        f"Graph store {type(store).__name__!r} does not expose "
        "'get_triplets()' or 'get_rel_map()'. "
        "Ensure you are passing a supported LlamaIndex graph store."
    )


def _entity_node_to_kgnode(li_node: Any) -> KGNode:
    """Convert a LlamaIndex EntityNode to a KGNode."""
    props: dict = dict(li_node.properties) if hasattr(li_node, "properties") else {}
    _RESERVED = {
        "description",
        "source_chunks",
        "_merged_from",
        "_merge_strategy",
        "_resolved_types",
    }
    description: str | None = props.pop("description", None)
    source_chunks: list[str] = list(props.pop("source_chunks", None) or [])
    extra = {k: v for k, v in props.items() if k not in _RESERVED}

    return KGNode(
        id=str(li_node.id),
        name=str(li_node.name),
        type=str(li_node.label) if getattr(li_node, "label", None) else None,
        description=description or None,
        attributes=extra,
        source_chunks=source_chunks,
    )


def _kgnode_extra_props(node: KGNode) -> dict:
    """Collect all serialisable KGNode fields into a properties dict."""
    props: dict = dict(node.attributes)
    if node.description:
        props["description"] = node.description
    if node.source_chunks:
        props["source_chunks"] = node.source_chunks
    if node._merged_from is not None:
        props["_merged_from"] = node._merged_from
    if node._merge_strategy is not None:
        props["_merge_strategy"] = node._merge_strategy
    if node._resolved_types is not None:
        props["_resolved_types"] = node._resolved_types
    return props

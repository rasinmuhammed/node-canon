from __future__ import annotations

from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode


class NetworkXAdapter(BaseAdapter):
    """Bidirectional conversion between NetworkX DiGraph and KGGraph.

    Node convention:
      - Each graph node must have a "name" attribute.
      - Optional: "type", "description", "source_chunks", plus any extras
        stored under arbitrary attribute keys.

    Edge convention:
      - Each edge may carry "relation" (str), "weight" (float), and arbitrary
        attribute keys.  "relation" defaults to "RELATED_TO" if absent.
    """

    def load(self, source: Any) -> KGGraph:
        """Convert a networkx.DiGraph (or Graph) to KGGraph."""
        return self.from_networkx(source)

    def dump(self, graph: KGGraph, destination: Any) -> None:
        """Convert a KGGraph to networkx.DiGraph and assign to *destination*.

        Because NetworkX graphs are mutable Python objects, *destination* is
        expected to be a dict or namespace where the result is stored under
        the key/attribute "graph".  For direct use, call to_networkx() instead.
        """
        nx_graph = self.to_networkx(graph)
        if isinstance(destination, dict):
            destination["graph"] = nx_graph
        else:
            destination.graph = nx_graph

    def to_networkx(self, graph: KGGraph) -> Any:
        """Return a networkx.DiGraph representing the KGGraph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for NetworkXAdapter. "
                "Install it with: pip install networkx"
            ) from None

        G: Any = nx.DiGraph()

        for node in graph.nodes:
            attrs: dict[str, Any] = {
                "name": node.name,
            }
            if node.type is not None:
                attrs["type"] = node.type
            if node.description is not None:
                attrs["description"] = node.description
            if node.source_chunks:
                attrs["source_chunks"] = node.source_chunks
            if node._merged_from is not None:
                attrs["_merged_from"] = node._merged_from
            if node._merge_strategy is not None:
                attrs["_merge_strategy"] = node._merge_strategy
            if node._resolved_types is not None:
                attrs["_resolved_types"] = node._resolved_types
            attrs.update(node.attributes)
            G.add_node(node.id, **attrs)

        for edge in graph.edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                relation=edge.relation,
                weight=edge.weight,
                **edge.attributes,
            )

        return G

    def from_networkx(self, nx_graph: Any) -> KGGraph:
        """Convert a networkx.Graph or DiGraph to KGGraph."""
        _RESERVED = {
            "name",
            "type",
            "description",
            "source_chunks",
            "_merged_from",
            "_merge_strategy",
            "_resolved_types",
        }

        nodes: list[KGNode] = []
        for node_id, data in nx_graph.nodes(data=True):
            name = data.get("name")
            if not name:
                raise ValueError(
                    f"NetworkX node {node_id!r} has no 'name' attribute. "
                    "Set node['name'] before converting to KGGraph."
                )
            extra = {k: v for k, v in data.items() if k not in _RESERVED}
            nodes.append(
                KGNode(
                    id=str(node_id),
                    name=str(name),
                    type=data.get("type"),
                    description=data.get("description"),
                    attributes=extra,
                    source_chunks=list(data.get("source_chunks") or []),
                    _merged_from=data.get("_merged_from"),
                    _merge_strategy=data.get("_merge_strategy"),
                    _resolved_types=data.get("_resolved_types"),
                )
            )

        edges: list[KGEdge] = []
        for src, tgt, data in nx_graph.edges(data=True):
            _EDGE_RESERVED = {"relation", "weight"}
            extra = {k: v for k, v in data.items() if k not in _EDGE_RESERVED}
            edges.append(
                KGEdge(
                    source_id=str(src),
                    target_id=str(tgt),
                    relation=str(data.get("relation", "RELATED_TO")),
                    weight=float(data.get("weight", 1.0)),
                    attributes=extra,
                )
            )

        return KGGraph(nodes=nodes, edges=edges)

"""Fluent builder for KGGraph — no manual ID wrangling required."""

from __future__ import annotations

from typing import Any

from nodecanon.core.models import KGEdge, KGGraph, KGNode, _auto_id


class GraphBuilder:
    """Build a KGGraph incrementally with a simple, fluent API.

    Nodes can be referenced by name or id in ``add_edge`` — the builder
    resolves names automatically.  Calling ``add_node`` twice with the same
    name is a no-op (idempotent).

    Example::

        graph = (
            GraphBuilder()
            .add_node("IBM", type="ORGANIZATION")
            .add_node("I.B.M.", type="ORGANIZATION")
            .add_node("Watson AI", type="PRODUCT")
            .add_edge("IBM", "Watson AI", "MADE")
            .add_edge("I.B.M.", "Watson AI", "MADE")
            .build()
        )
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KGNode] = {}  # id → node
        self._name_to_id: dict[str, str] = {}  # name → id (for by-name lookup)
        self._edges: list[KGEdge] = []

    # ------------------------------------------------------------------
    # Fluent API
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        *,
        id: str | None = None,
        type: str | None = None,
        description: str | None = None,
        **attributes: Any,
    ) -> GraphBuilder:
        """Add a node, returning self for chaining.

        If a node with the same name already exists, this is a no-op.
        ``id`` is auto-derived from the name when not provided.
        """
        if name in self._name_to_id:
            return self
        node_id = id if id is not None else _auto_id(name, set(self._nodes))
        node = KGNode(
            id=node_id,
            name=name,
            type=type,
            description=description,
            attributes=attributes,
        )
        self._nodes[node_id] = node
        self._name_to_id[name] = node_id
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "RELATED_TO",
        *,
        weight: float = 1.0,
        **attributes: Any,
    ) -> GraphBuilder:
        """Add an edge between two nodes, returning self for chaining.

        ``source`` and ``target`` can be node names or IDs.  Nodes that
        don't exist yet are auto-created with just a name (no type).
        """
        src_id = self._resolve(source)
        tgt_id = self._resolve(target)
        self._edges.append(
            KGEdge(
                source_id=src_id,
                target_id=tgt_id,
                relation=relation,
                weight=weight,
                attributes=attributes,
            )
        )
        return self

    def build(self) -> KGGraph:
        """Return the completed KGGraph."""
        return KGGraph(nodes=list(self._nodes.values()), edges=list(self._edges))

    # ------------------------------------------------------------------
    # Convenience class methods for quick construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dicts(
        cls,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
    ) -> KGGraph:
        """Shortcut: ``GraphBuilder.from_dicts(...)`` delegates to ``KGGraph.from_dicts``."""
        return KGGraph.from_dicts(nodes, edges)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, ref: str) -> str:
        """Resolve a name-or-id reference to a node id, auto-creating if needed."""
        if ref in self._nodes:
            return ref
        if ref in self._name_to_id:
            return self._name_to_id[ref]
        # Not found — auto-create a bare node from the name
        self.add_node(ref)
        return self._name_to_id[ref]

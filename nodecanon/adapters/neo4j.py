"""Neo4j adapter — full roundtrip: load from live instance, write back resolved graph.

Install the driver:
    pip install nodecanon[neo4j]

Quickstart::

    from neo4j import GraphDatabase
    from nodecanon.adapters.neo4j import Neo4jAdapter
    from nodecanon import Resolver

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

    graph  = Neo4jAdapter.from_neo4j(driver)
    result = Resolver().resolve(graph)
    Neo4jAdapter.to_neo4j(driver, result)

    driver.close()

The write-back is non-destructive: canonical nodes are updated in place, alias
nodes are annotated with ``_canonical_id`` / ``_is_alias`` and linked via an
``IS_ALIAS_OF`` relationship.  Nothing is deleted — you keep full provenance
and your existing queries still work.  Run a cleanup query afterwards if you
want to prune aliases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode
from nodecanon.core.resolver import ResolveResult

if TYPE_CHECKING:
    from neo4j import Driver

# Characters that are not valid in Cypher labels or relationship types.
_INVALID_LABEL_RE = re.compile(r"[^A-Za-z0-9_]")

# Properties written to every node by nodecanon — excluded when round-tripping
# back to KGNode.attributes so they don't appear twice.
_INTERNAL_PROPS = frozenset(
    {
        "id", "name", "type", "description", "source_chunks",
        "_merged_from", "_merge_evidence", "_merge_strategy", "_resolved_types",
        "_is_alias", "_is_canonical", "_canonical_id",
    }
)


def _safe_label(text: str) -> str:
    cleaned = _INVALID_LABEL_RE.sub("_", text.strip())
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned or "ENTITY"


def _cypher_str(value: Any) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_cypher_str(v) for v in value) + "]"
    escaped = json.dumps(value).replace("'", "\\'")
    return f"'{escaped}'"


def _props_map(props: dict[str, Any]) -> str:
    if not props:
        return ""
    items = ", ".join(f"{k}: {_cypher_str(v)}" for k, v in sorted(props.items()))
    return "{" + items + "}"


class Neo4jAdapter(BaseAdapter):
    """Full Neo4j integration: load, resolve, write back.

    Parameters
    ----------
    node_id_property:
        The node property used as the nodecanon ``id`` field.  Defaults to
        ``"id"``.  If your nodes use a different unique key (e.g. ``"uid"``
        or ``"element_id"``), set this here.
    name_property:
        The node property used as the nodecanon ``name`` field.  Defaults to
        ``"name"``.
    """

    def __init__(
        self,
        node_id_property: str = "id",
        name_property: str = "name",
    ) -> None:
        self.node_id_property = node_id_property
        self.name_property = name_property

    # ------------------------------------------------------------------
    # Load from Neo4j
    # ------------------------------------------------------------------

    @classmethod
    def from_neo4j(
        cls,
        driver: Driver,
        *,
        database: str | None = None,
        node_label: str | None = None,
        node_id_property: str = "id",
        name_property: str = "name",
        limit: int | None = None,
    ) -> KGGraph:
        """Load a KGGraph from a live Neo4j database.

        Parameters
        ----------
        driver:
            An open ``neo4j.GraphDatabase.driver(...)`` instance.
        database:
            Name of the Neo4j database to query (default: the driver's default).
        node_label:
            Optional Cypher label filter, e.g. ``"Entity"``.  When omitted all
            nodes are loaded.
        node_id_property:
            Property that uniquely identifies each node (default ``"id"``).
        name_property:
            Property used as the human-readable entity name (default ``"name"``).
        limit:
            Cap the number of nodes loaded (useful for large databases).
        """
        adapter = cls(
            node_id_property=node_id_property, name_property=name_property
        )
        with driver.session(database=database) as session:
            nodes = adapter._load_nodes(session, node_label=node_label, limit=limit)
            node_ids = {n.id for n in nodes}
            edges = adapter._load_edges(session, node_ids=node_ids, node_label=node_label)
        return KGGraph(nodes=nodes, edges=edges)

    def _load_nodes(
        self,
        session: Any,
        node_label: str | None,
        limit: int | None,
    ) -> list[KGNode]:
        label_clause = f":{node_label}" if node_label else ""
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = f"MATCH (n{label_clause}) RETURN n{limit_clause}"

        nodes: list[KGNode] = []
        result = session.run(query)
        for record in result:
            raw = dict(record["n"])
            node_id = str(raw.get(self.node_id_property, record["n"].element_id))
            name = str(raw.get(self.name_property, node_id))
            entity_type = raw.get("type")
            description = raw.get("description")
            source_chunks = list(raw.get("source_chunks", []))
            # Everything else goes into attributes (skip internal nodecanon fields)
            attrs = {
                k: v for k, v in raw.items()
                if k not in _INTERNAL_PROPS
                and k != self.node_id_property
                and k != self.name_property
            }
            nodes.append(
                KGNode(
                    id=node_id,
                    name=name,
                    type=str(entity_type) if entity_type else None,
                    description=str(description) if description else None,
                    source_chunks=source_chunks,
                    attributes=attrs,
                )
            )
        return nodes

    def _load_edges(
        self,
        session: Any,
        node_ids: set[str],
        node_label: str | None,
    ) -> list[KGEdge]:
        label_clause = f":{node_label}" if node_label else ""
        id_prop = self.node_id_property
        query = (
            f"MATCH (a{label_clause})-[r]->(b{label_clause}) "
            f"RETURN coalesce(a.{id_prop}, elementId(a)) AS src, "
            f"type(r) AS rel, "
            f"coalesce(b.{id_prop}, elementId(b)) AS tgt, "
            f"coalesce(r.weight, 1.0) AS weight, "
            f"properties(r) AS props"
        )
        edges: list[KGEdge] = []
        result = session.run(query)
        for record in result:
            src, tgt = str(record["src"]), str(record["tgt"])
            # Only include edges between loaded nodes
            if src not in node_ids or tgt not in node_ids:
                continue
            raw_props = dict(record["props"])
            raw_props.pop("weight", None)
            edges.append(
                KGEdge(
                    source_id=src,
                    target_id=tgt,
                    relation=str(record["rel"]),
                    weight=float(record["weight"]),
                    attributes=raw_props,
                )
            )
        return edges

    # ------------------------------------------------------------------
    # Write back to Neo4j
    # ------------------------------------------------------------------

    @classmethod
    def to_neo4j(
        cls,
        driver: Driver,
        result: ResolveResult,
        *,
        database: str | None = None,
        node_id_property: str = "id",
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Write a resolved graph back to a live Neo4j database.

        Strategy is non-destructive:

        - Canonical nodes are updated in place (``MERGE`` on id, ``SET`` props).
        - Alias nodes gain ``_is_alias: true``, ``_canonical_id`` properties, and
          an ``IS_ALIAS_OF`` relationship to their canonical.
        - Canonical nodes gain ``_is_canonical: true`` and ``_merged_from``.
        - No nodes or edges are deleted.

        Returns a dict with counts: ``{"nodes_upserted", "aliases_annotated",
        "edges_merged"}``.

        Parameters
        ----------
        driver:
            An open ``neo4j.GraphDatabase.driver(...)`` instance.
        result:
            The ``ResolveResult`` returned by ``Resolver().resolve(graph)``.
        database:
            Target Neo4j database (default: driver default).
        node_id_property:
            Property that uniquely identifies each node.
        batch_size:
            Number of statements per transaction.  Reduce for very large graphs
            to avoid memory pressure.
        """
        adapter = cls(node_id_property=node_id_property)
        graph = result.graph

        # Build alias → canonical map from merge records
        alias_to_canonical: dict[str, str] = {}
        for record in result.merge_records:
            for alias_id in record.merged_ids:
                alias_to_canonical[alias_id] = record.canonical_id

        stats = {"nodes_upserted": 0, "aliases_annotated": 0, "edges_merged": 0}

        with driver.session(database=database) as session:
            # 1. Upsert canonical nodes
            canonical_batches = _batch(list(graph.nodes), batch_size)
            for batch in canonical_batches:
                count = session.execute_write(
                    adapter._upsert_canonical_nodes_tx, batch, node_id_property
                )
                stats["nodes_upserted"] += count

            # 2. Annotate alias nodes and create IS_ALIAS_OF relationships
            if alias_to_canonical:
                alias_batches = _batch(list(alias_to_canonical.items()), batch_size)
                for batch in alias_batches:
                    count = session.execute_write(
                        adapter._annotate_aliases_tx, batch, node_id_property
                    )
                    stats["aliases_annotated"] += count

            # 3. Merge edges
            edge_batches = _batch(list(graph.edges), batch_size)
            for batch in edge_batches:
                count = session.execute_write(
                    adapter._merge_edges_tx, batch, node_id_property
                )
                stats["edges_merged"] += count

        return stats

    @staticmethod
    def _upsert_canonical_nodes_tx(
        tx: Any, nodes: list[KGNode], id_prop: str
    ) -> int:
        count = 0
        for node in nodes:
            props = Neo4jAdapter._node_props(node)
            props["_is_canonical"] = bool(node._merged_from)
            tx.run(
                f"MERGE (n {{{id_prop}: $id}}) SET n += $props",
                id=node.id,
                props=props,
            )
            count += 1
        return count

    @staticmethod
    def _annotate_aliases_tx(
        tx: Any, pairs: list[tuple[str, str]], id_prop: str
    ) -> int:
        count = 0
        for alias_id, canonical_id in pairs:
            tx.run(
                f"""
                MATCH (alias {{{id_prop}: $alias_id}})
                MATCH (canonical {{{id_prop}: $canonical_id}})
                SET alias._is_alias = true,
                    alias._canonical_id = $canonical_id
                MERGE (alias)-[:IS_ALIAS_OF]->(canonical)
                """,
                alias_id=alias_id,
                canonical_id=canonical_id,
            )
            count += 1
        return count

    @staticmethod
    def _merge_edges_tx(
        tx: Any, edges: list[KGEdge], id_prop: str
    ) -> int:
        count = 0
        for edge in edges:
            rel_type = _safe_label(edge.relation)
            props = {"weight": edge.weight, **edge.attributes}
            tx.run(
                f"""
                MATCH (a {{{id_prop}: $src}})
                MATCH (b {{{id_prop}: $tgt}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
                """,
                src=edge.source_id,
                tgt=edge.target_id,
                props=props,
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Cypher file export (offline, no live connection needed)
    # ------------------------------------------------------------------

    def load(self, source: Path | str | Any) -> KGGraph:
        raise NotImplementedError(
            "Use Neo4jAdapter.from_neo4j(driver) to load from a live database, "
            "or pass a KGGraph built with GraphBuilder / KGGraph.from_dicts()."
        )

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        """Write Cypher to a file (no live connection required)."""
        cypher = self.to_cypher(graph)
        if isinstance(destination, (str, Path)):
            Path(destination).write_text(cypher, encoding="utf-8")
        elif isinstance(destination, dict):
            destination["cypher"] = cypher
        else:
            destination.cypher = cypher

    def to_cypher(self, graph: KGGraph) -> str:
        """Return a Cypher script that recreates the graph in Neo4j.

        Uses MERGE on node id — idempotent, safe to run multiple times.
        """
        lines: list[str] = [
            "// Generated by nodecanon — https://github.com/rasinmuhammed/node-canon",
            "// Run in Neo4j Browser or cypher-shell",
            "",
        ]
        for node in graph.nodes:
            lines.append(self._node_statement(node))
        if graph.nodes and graph.edges:
            lines.append("")
        for edge in graph.edges:
            lines.append(self._edge_statement(edge))
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_props(node: KGNode) -> dict[str, Any]:
        props: dict[str, Any] = {"id": node.id, "name": node.name}
        if node.type is not None:
            props["type"] = node.type
        if node.description is not None:
            props["description"] = node.description
        if node.source_chunks:
            props["source_chunks"] = node.source_chunks
        if node._merged_from is not None:
            props["_merged_from"] = node._merged_from
        if node._merge_strategy is not None:
            props["_merge_strategy"] = node._merge_strategy
        if node._resolved_types is not None:
            props["_resolved_types"] = node._resolved_types
        props.update(node.attributes)
        return props

    def _node_statement(self, node: KGNode) -> str:
        label = _safe_label(node.type or "Entity")
        props = self._node_props(node)
        id_prop = self.node_id_property
        return (
            f"MERGE (n:{label} {{{id_prop}: {_cypher_str(node.id)}}}) "
            f"SET n += {_props_map(props)};"
        )

    @staticmethod
    def _edge_statement(edge: KGEdge) -> str:
        rel_type = _safe_label(edge.relation)
        edge_props: dict[str, Any] = {"weight": edge.weight}
        edge_props.update(edge.attributes)
        props_str = (" " + _props_map(edge_props)) if edge_props else ""
        return (
            f"MATCH (a {{id: {_cypher_str(edge.source_id)}}}), "
            f"(b {{id: {_cypher_str(edge.target_id)}}}) "
            f"MERGE (a)-[:{rel_type}{props_str}]->(b);"
        )


def _batch(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def to_cypher(graph: KGGraph) -> str:
    """Module-level convenience: ``Neo4jAdapter().to_cypher(graph)``."""
    return Neo4jAdapter().to_cypher(graph)

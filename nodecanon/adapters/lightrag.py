from __future__ import annotations

from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode

# LightRAG stores its graph as a NetworkX GraphML file at this path
# relative to the working directory passed to LightRAG.
_DEFAULT_GRAPHML = "graph_chunk_entity_relation.graphml"

# LightRAG node attribute names
_ENTITY_TYPE_ATTR = "entity_type"
_DESCRIPTION_ATTR = "description"
_SOURCE_ID_ATTR = "source_id"

# LightRAG edge attribute names
_WEIGHT_ATTR = "weight"
_RELATION_ATTR = "description"  # LightRAG stores relation text in "description"
_KEYWORDS_ATTR = "keywords"


class LightRAGAdapter(BaseAdapter):
    """Bidirectional adapter between LightRAG's GraphML output and KGGraph.

    LightRAG stores its internal entity-relation graph as a NetworkX GraphML
    file (``graph_chunk_entity_relation.graphml``) in the working directory
    passed to ``LightRAG(working_dir=...)``.

    Node convention in LightRAG GraphML
    ------------------------------------
    - Node **id** = entity name (often uppercased, e.g. ``"IBM"``)
    - ``entity_type`` → KGNode.type
    - ``description``  → KGNode.description
    - ``source_id``    → KGNode.source_chunks (comma-separated chunk IDs)

    Edge convention in LightRAG GraphML
    ------------------------------------
    - ``description`` → KGEdge.relation (relation text)
    - ``weight``      → KGEdge.weight
    - ``keywords``    → stored in KGEdge.attributes["keywords"]

    Loading
    -------
    ::

        from nodecanon.adapters.lightrag import LightRAGAdapter
        graph = LightRAGAdapter.from_working_dir("./my_lightrag_output/")

    Or via the base interface::

        graph = LightRAGAdapter().load("./my_lightrag_output/")

    Dumping
    -------
    ``dump()`` writes a new GraphML file that can be loaded back into a
    LightRAG instance via ``rag.chunk_entity_relation_graph``.
    """

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self, source: Path | str | Any) -> KGGraph:
        """Load from a LightRAG working directory or a direct GraphML path."""
        path = Path(source)
        if path.is_dir():
            return self.from_working_dir(path)
        return self.from_graphml(path)

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        """Write *graph* as a LightRAG-compatible GraphML file.

        If *destination* is a directory, writes to
        ``<destination>/graph_chunk_entity_relation.graphml``.
        Otherwise treats *destination* as the target file path.
        """
        dest = Path(destination)
        out_path = dest / _DEFAULT_GRAPHML if dest.is_dir() else dest
        self.to_graphml(graph, out_path)

    # ------------------------------------------------------------------
    # Public class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_working_dir(cls, directory: Path | str) -> KGGraph:
        """Load from a LightRAG ``working_dir``."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"LightRAG working directory not found: {directory!r}. "
                "Make sure you have run LightRAG and the path points to the "
                "directory passed as ``working_dir`` to LightRAG(...)."
            )
        graphml_path = directory / _DEFAULT_GRAPHML
        if not graphml_path.exists():
            raise FileNotFoundError(
                f"LightRAG graph file not found: {graphml_path!r}. "
                "Run LightRAG to completion before exporting. "
                f"Expected: {_DEFAULT_GRAPHML}"
            )
        return cls.from_graphml(graphml_path)

    @classmethod
    def from_graphml(cls, path: Path | str) -> KGGraph:
        """Convert a LightRAG GraphML file to a KGGraph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for LightRAGAdapter. "
                "Install it with: pip install networkx"
            ) from None

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"GraphML file not found: {path!r}. "
                "Check the path and ensure LightRAG has finished indexing."
            )

        G = nx.read_graphml(str(path))
        return cls._graphml_to_kggraph(G)

    @classmethod
    def to_graphml(cls, graph: KGGraph, path: Path | str) -> None:
        """Write a KGGraph as a LightRAG-compatible GraphML file."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for LightRAGAdapter. "
                "Install it with: pip install networkx"
            ) from None

        G = cls._kggraph_to_graphml(graph)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, str(path))

    # ------------------------------------------------------------------
    # Private conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _graphml_to_kggraph(G: Any) -> KGGraph:
        """Convert a networkx.Graph (read from LightRAG GraphML) to KGGraph."""
        _RESERVED = {
            _ENTITY_TYPE_ATTR,
            _DESCRIPTION_ATTR,
            _SOURCE_ID_ATTR,
        }

        nodes: list[KGNode] = []
        for node_id, data in G.nodes(data=True):
            name = str(node_id)  # LightRAG uses entity name as node ID
            entity_type: str | None = data.get(_ENTITY_TYPE_ATTR) or None
            description: str | None = data.get(_DESCRIPTION_ATTR) or None

            raw_source = data.get(_SOURCE_ID_ATTR, "")
            source_chunks = (
                [s.strip() for s in str(raw_source).split(",") if s.strip()]
                if raw_source
                else []
            )

            extra = {k: v for k, v in data.items() if k not in _RESERVED}

            nodes.append(
                KGNode(
                    id=str(node_id),
                    name=name,
                    type=entity_type,
                    description=description,
                    attributes=extra,
                    source_chunks=source_chunks,
                )
            )

        edges: list[KGEdge] = []
        _EDGE_RESERVED = {_WEIGHT_ATTR, _RELATION_ATTR}
        for src, tgt, data in G.edges(data=True):
            relation = str(data.get(_RELATION_ATTR, "RELATED_TO") or "RELATED_TO")
            try:
                weight = float(data.get(_WEIGHT_ATTR, 1.0))
            except (TypeError, ValueError):
                weight = 1.0
            extra = {k: v for k, v in data.items() if k not in _EDGE_RESERVED}
            edges.append(
                KGEdge(
                    source_id=str(src),
                    target_id=str(tgt),
                    relation=relation,
                    weight=weight,
                    attributes=extra,
                )
            )

        return KGGraph(nodes=nodes, edges=edges)

    @staticmethod
    def _kggraph_to_graphml(graph: KGGraph) -> Any:
        """Convert a KGGraph to a networkx.DiGraph with LightRAG's attribute schema."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for LightRAGAdapter. "
                "Install it with: pip install networkx"
            ) from None

        G: Any = nx.DiGraph()

        for node in graph.nodes:
            attrs: dict[str, Any] = {}
            if node.type is not None:
                attrs[_ENTITY_TYPE_ATTR] = node.type
            if node.description is not None:
                attrs[_DESCRIPTION_ATTR] = node.description
            if node.source_chunks:
                attrs[_SOURCE_ID_ATTR] = ",".join(node.source_chunks)
            attrs.update(node.attributes)
            G.add_node(node.id, **attrs)

        for edge in graph.edges:
            attrs = {
                _RELATION_ATTR: edge.relation,
                _WEIGHT_ATTR: edge.weight,
                **edge.attributes,
            }
            G.add_edge(edge.source_id, edge.target_id, **attrs)

        return G

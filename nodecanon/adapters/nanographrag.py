"""nano-graphrag adapter.

nano-graphrag (github.com/gusye1234/nano-graphrag) stores its entity-relation
graph in GraphML format using the same attribute schema as LightRAG:

  Node attributes : entity_type, description, source_id
  Edge attributes : weight, description (relation text), source_id

The file is written to ``<working_dir>/graph_<namespace>.graphml``.  The
default namespace is ``chunk_entity_relation``, which produces
``graph_chunk_entity_relation.graphml``.

Install nano-graphrag:
    pip install nano-graphrag

Usage::

    from nodecanon.adapters.nanographrag import NanoGraphRAGAdapter
    from nodecanon import Resolver

    # From a working directory
    graph = NanoGraphRAGAdapter.from_working_dir("./nano_output/")
    result = Resolver().resolve(graph)
    NanoGraphRAGAdapter.save(result.graph, "./nano_output/")

    # From a live GraphRAG instance (in-memory, no disk I/O)
    from nano_graphrag import GraphRAG
    rag = GraphRAG(working_dir="./nano_output/")
    await rag.ainsert(documents)

    graph = NanoGraphRAGAdapter.from_instance(rag)
    result = Resolver().resolve(graph)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.adapters.lightrag import LightRAGAdapter
from nodecanon.core.models import KGGraph

_DEFAULT_NAMESPACE = "chunk_entity_relation"


def _graphml_filename(namespace: str) -> str:
    return f"graph_{namespace}.graphml"


class NanoGraphRAGAdapter(BaseAdapter):
    """Bidirectional adapter between nano-graphrag and KGGraph.

    Parameters
    ----------
    namespace:
        The graph namespace used by nano-graphrag.  Determines the GraphML
        filename: ``graph_<namespace>.graphml``.  Default is
        ``"chunk_entity_relation"``.
    """

    def __init__(self, namespace: str = _DEFAULT_NAMESPACE) -> None:
        self.namespace = namespace

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self, source: Path | str | Any) -> KGGraph:
        """Load from a working directory path or a live GraphRAG instance."""
        if hasattr(source, "chunk_entity_relation_graph"):
            return self.from_instance(source)
        path = Path(source)
        if path.is_dir():
            return self.from_working_dir(path, namespace=self.namespace)
        return LightRAGAdapter.from_graphml(path)

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        """Write a resolved KGGraph back to nano-graphrag's GraphML format.

        If *destination* is a directory, writes to
        ``<destination>/graph_<namespace>.graphml``.
        """
        dest = Path(destination)
        out_path = (
            dest / _graphml_filename(self.namespace) if dest.is_dir() else dest
        )
        LightRAGAdapter.to_graphml(graph, out_path)

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_working_dir(
        cls,
        directory: Path | str,
        namespace: str = _DEFAULT_NAMESPACE,
    ) -> KGGraph:
        """Load from a nano-graphrag working directory.

        Parameters
        ----------
        directory:
            Path passed as ``working_dir`` to ``nano_graphrag.GraphRAG(...)``.
        namespace:
            Graph namespace.  Almost always the default ``"chunk_entity_relation"``.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"nano-graphrag working directory not found: {directory!r}. "
                "Make sure you have run GraphRAG indexing before loading."
            )
        graphml_path = directory / _graphml_filename(namespace)
        if not graphml_path.exists():
            # Auto-detect: scan for any graph_*.graphml in the directory
            candidates = list(directory.glob("graph_*.graphml"))
            if candidates:
                graphml_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"No GraphML file found in {directory!r}. "
                    f"Expected: {_graphml_filename(namespace)}. "
                    "Run nano-graphrag to completion before loading."
                )
        return LightRAGAdapter.from_graphml(graphml_path)

    @classmethod
    def from_instance(cls, rag_instance: Any) -> KGGraph:
        """Load directly from a live ``nano_graphrag.GraphRAG`` instance.

        Accesses the in-memory NetworkX graph without reading from disk.
        Call this after ``await rag.ainsert(...)`` has completed.

        Parameters
        ----------
        rag_instance:
            A ``nano_graphrag.GraphRAG`` object with a populated
            ``chunk_entity_relation_graph`` attribute.
        """
        graph_storage = getattr(rag_instance, "chunk_entity_relation_graph", None)
        if graph_storage is None:
            raise AttributeError(
                "The provided object has no 'chunk_entity_relation_graph' attribute. "
                "Pass a nano_graphrag.GraphRAG instance after indexing is complete, "
                "or use NanoGraphRAGAdapter.from_working_dir(path) instead."
            )
        nx_graph = getattr(graph_storage, "_graph", None)
        if nx_graph is None:
            raise AttributeError(
                "nano-graphrag graph storage has no '_graph' attribute. "
                "This may be an unsupported version of nano-graphrag. "
                "Try NanoGraphRAGAdapter.from_working_dir(path) instead."
            )
        return LightRAGAdapter._graphml_to_kggraph(nx_graph)

    @classmethod
    def save(
        cls,
        graph: KGGraph,
        directory: Path | str,
        namespace: str = _DEFAULT_NAMESPACE,
    ) -> None:
        """Write a resolved KGGraph back to a nano-graphrag working directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / _graphml_filename(namespace)
        LightRAGAdapter.to_graphml(graph, out_path)

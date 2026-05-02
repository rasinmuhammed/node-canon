from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from nodecanon.adapters.base import BaseAdapter
from nodecanon.core.models import KGEdge, KGGraph, KGNode

# GraphRAG parquet column names (as of graphrag 1.x / 2.x)
_ENTITY_REQUIRED = {"id", "title"}
_REL_REQUIRED = {"id", "source", "target"}

# Column aliases: each tuple lists acceptable column names in priority order.
_ENTITY_TYPE_COLS = ("type", "entity_type")
_ENTITY_DESC_COLS = ("description",)
_REL_WEIGHT_COLS = ("weight", "rank", "combined_degree")
_REL_RELATION_COLS = ("description", "relation", "relationship_type")


def _first_col(df: Any, candidates: tuple[str, ...]) -> str | None:
    """Return the first column name from *candidates* that exists in *df*."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


class GraphRAGAdapter(BaseAdapter):
    """Read Microsoft GraphRAG parquet output files into a KGGraph.

    Supported directory layouts (both common GraphRAG versions):

      <directory>/
        entities.parquet            (v1 layout)
        relationships.parquet

      <directory>/
        output/
          entities.parquet          (v2 layout, nested under output/)
          relationships.parquet
    """

    # ------------------------------------------------------------------
    # BaseAdapter interface
    # ------------------------------------------------------------------

    def load(self, source: Path | str | Any) -> KGGraph:
        return self.from_directory(Path(source))

    def dump(self, graph: KGGraph, destination: Path | str | Any) -> None:
        raise NotImplementedError(
            "GraphRAGAdapter.dump() is not yet implemented. "
            "Use NetworkXAdapter or Neo4jAdapter to export a KGGraph."
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(cls, directory: Path | str) -> KGGraph:
        """Load a KGGraph from a GraphRAG output directory.

        Searches for ``entities.parquet`` and ``relationships.parquet``
        in *directory* and its immediate ``output/`` subdirectory.

        Parameters
        ----------
        directory:
            Path to the GraphRAG output root (or the ``output/`` subdirectory
            itself — both are tried).

        Raises
        ------
        FileNotFoundError
            If neither ``entities.parquet`` nor ``relationships.parquet``
            can be located under *directory*.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for GraphRAGAdapter. "
                "Install it with: pip install pandas pyarrow"
            ) from None

        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(
                f"GraphRAG output directory not found: {directory!r}. "
                "Make sure you have run the GraphRAG indexing pipeline and "
                "the path points to a directory containing "
                "'entities.parquet' and 'relationships.parquet'."
            )

        entities_path = cls._find_parquet(directory, "entities.parquet")
        relationships_path = cls._find_parquet(directory, "relationships.parquet")

        entities_df = pd.read_parquet(entities_path)
        relationships_df = (
            pd.read_parquet(relationships_path)
            if relationships_path is not None
            else pd.DataFrame()
        )

        nodes = cls._parse_entities(entities_df)
        edges = cls._parse_relationships(relationships_df)
        return KGGraph(nodes=nodes, edges=edges)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_parquet(directory: Path, filename: str) -> Path | None:
        """Search *directory* and its ``output/`` subdirectory for *filename*."""
        candidates = [
            directory / filename,
            directory / "output" / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    @staticmethod
    def _parse_entities(df: Any) -> list[KGNode]:
        """Convert a GraphRAG entities DataFrame into KGNode objects."""
        if df.empty:
            return []

        missing = _ENTITY_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"GraphRAG entities parquet is missing required columns: {missing}. "
                f"Present columns: {list(df.columns)}"
            )

        type_col = _first_col(df, _ENTITY_TYPE_COLS)
        desc_col = _first_col(df, _ENTITY_DESC_COLS)

        # Columns that map to reserved KGNode fields — exclude from attributes
        reserved = {"id", "title", "short_id", "human_readable_id"}
        if type_col:
            reserved.add(type_col)
        if desc_col:
            reserved.add(desc_col)

        nodes: list[KGNode] = []
        for _, row in df.iterrows():
            node_id = str(row["id"])
            name = str(row["title"])

            entity_type: str | None = None
            if type_col and not _is_na(row.get(type_col)):
                entity_type = str(row[type_col]).strip() or None

            description: str | None = None
            if desc_col and not _is_na(row.get(desc_col)):
                description = str(row[desc_col]).strip() or None

            extra = {
                k: v for k, v in row.items() if k not in reserved and not _is_na(v)
            }

            nodes.append(
                KGNode(
                    id=node_id,
                    name=name,
                    type=entity_type,
                    description=description,
                    attributes=extra,
                )
            )
        return nodes

    @staticmethod
    def _parse_relationships(df: Any) -> list[KGEdge]:
        """Convert a GraphRAG relationships DataFrame into KGEdge objects."""
        if df.empty:
            return []

        missing = _REL_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"GraphRAG relationships parquet is missing required columns: {missing}. "
                f"Present columns: {list(df.columns)}"
            )

        weight_col = _first_col(df, _REL_WEIGHT_COLS)
        relation_col = _first_col(df, _REL_RELATION_COLS)

        reserved = {"id", "source", "target", "short_id", "human_readable_id"}
        if weight_col:
            reserved.add(weight_col)
        if relation_col:
            reserved.add(relation_col)

        edges: list[KGEdge] = []
        for _, row in df.iterrows():
            source_id = str(row["source"])
            target_id = str(row["target"])

            relation = "RELATED_TO"
            if relation_col and not _is_na(row.get(relation_col)):
                relation = str(row[relation_col]).strip() or "RELATED_TO"

            weight = 1.0
            if weight_col and not _is_na(row.get(weight_col)):
                with contextlib.suppress(TypeError, ValueError):
                    weight = float(row[weight_col])

            extra = {
                k: v for k, v in row.items() if k not in reserved and not _is_na(v)
            }

            edges.append(
                KGEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation,
                    weight=weight,
                    attributes=extra,
                )
            )
        return edges


def _is_na(value: Any) -> bool:
    """Return True if *value* is a pandas NA / NaN / None."""
    if value is None:
        return True
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False

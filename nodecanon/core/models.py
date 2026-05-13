from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KGNode:
    id: str
    name: str
    type: str | None = None
    description: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    source_chunks: list[str] = field(default_factory=list)
    _merged_from: list[str] | None = None
    _merge_evidence: dict[str, Any] | None = None
    _merge_strategy: str | None = None
    _resolved_types: list[str] | None = None  # all types from merged nodes (union)

    def __repr__(self) -> str:
        type_part = f", type={self.type!r}" if self.type else ""
        merged = f", merged_from={self._merged_from}" if self._merged_from else ""
        return f"KGNode(id={self.id!r}, name={self.name!r}{type_part}{merged})"


@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"KGEdge({self.source_id!r} --[{self.relation}]--> {self.target_id!r})"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _auto_id(name: str, existing: set[str]) -> str:
    base = _SLUG_RE.sub("_", name.lower()).strip("_") or "node"
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


@dataclass
class KGGraph:
    nodes: list[KGNode]
    edges: list[KGEdge]

    def __repr__(self) -> str:
        return f"KGGraph({len(self.nodes)} nodes, {len(self.edges)} edges)"

    def node_index(self) -> dict[str, KGNode]:
        return {n.id: n for n in self.nodes}

    def adjacency_index(self) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            idx[edge.source_id].append(edge.target_id)
            idx[edge.target_id].append(edge.source_id)
        return idx

    @classmethod
    def from_dicts(
        cls,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
    ) -> KGGraph:
        """Build a KGGraph from plain Python dicts.

        Node dicts accept: id, name, type, description, plus any extra keys
        which land in ``attributes``.  ``id`` is auto-generated from ``name``
        when omitted.

        Edge dicts accept: source / source_id, target / target_id,
        relation / type, weight (default 1.0).
        """
        _RESERVED = {"id", "name", "type", "description", "source_chunks"}
        existing_ids: set[str] = set()
        kg_nodes: list[KGNode] = []
        for raw in nodes:
            raw = dict(raw)
            name = str(raw.pop("name", raw.get("id", "")))
            node_id = str(raw.pop("id", _auto_id(name, existing_ids)))
            existing_ids.add(node_id)
            entity_type = raw.pop("type", None)
            description = raw.pop("description", None)
            source_chunks = list(raw.pop("source_chunks", []))
            attrs = {k: v for k, v in raw.items() if k not in _RESERVED}
            kg_nodes.append(
                KGNode(
                    id=node_id,
                    name=name,
                    type=str(entity_type) if entity_type is not None else None,
                    description=str(description) if description is not None else None,
                    source_chunks=source_chunks,
                    attributes=attrs,
                )
            )

        kg_edges: list[KGEdge] = []
        for raw in edges or []:
            src = str(raw.get("source", raw.get("source_id", "")))
            tgt = str(raw.get("target", raw.get("target_id", "")))
            rel = str(raw.get("relation", raw.get("type", "RELATED_TO")))
            weight = float(raw.get("weight", 1.0))
            kg_edges.append(KGEdge(source_id=src, target_id=tgt, relation=rel, weight=weight))

        return cls(nodes=kg_nodes, edges=kg_edges)

    # Alias for discoverability
    from_records = from_dicts


@dataclass
class ScoreVector:
    name_similarity: float
    semantic_similarity: float
    type_agreement: float
    neighbor_overlap: float
    description_similarity: float

    def weighted_sum(self, weights: dict[str, float] | None = None) -> float:
        w = weights or {
            "name_similarity": 0.30,
            "semantic_similarity": 0.25,
            "type_agreement": 0.20,
            "neighbor_overlap": 0.20,
            "description_similarity": 0.05,
        }
        return (
            self.name_similarity * w["name_similarity"]
            + self.semantic_similarity * w["semantic_similarity"]
            + self.type_agreement * w["type_agreement"]
            + self.neighbor_overlap * w["neighbor_overlap"]
            + self.description_similarity * w["description_similarity"]
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "name_similarity": self.name_similarity,
            "semantic_similarity": self.semantic_similarity,
            "type_agreement": self.type_agreement,
            "neighbor_overlap": self.neighbor_overlap,
            "description_similarity": self.description_similarity,
        }


@dataclass
class MergeRecord:
    canonical_id: str
    merged_ids: list[str]
    score: ScoreVector
    strategy: str  # "rule_based" | "llm_assisted" | "manual"
    original_nodes: list[KGNode] = field(default_factory=list)


@dataclass
class MergeConflict:
    node_id_a: str
    node_id_b: str
    score: ScoreVector
    conflict_reason: str

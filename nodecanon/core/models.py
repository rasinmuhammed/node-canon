from __future__ import annotations

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


@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class KGGraph:
    nodes: list[KGNode]
    edges: list[KGEdge]

    def node_index(self) -> dict[str, KGNode]:
        return {n.id: n for n in self.nodes}

    def adjacency_index(self) -> dict[str, list[str]]:
        idx: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            idx[edge.source_id].append(edge.target_id)
            idx[edge.target_id].append(edge.source_id)
        return idx


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
    strategy: str  # "rule_based" | "llm_assisted"


@dataclass
class MergeConflict:
    node_id_a: str
    node_id_b: str
    score: ScoreVector
    conflict_reason: str

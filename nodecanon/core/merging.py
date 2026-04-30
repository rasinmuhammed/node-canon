from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any

from nodecanon.core.blocking import TypeCompatibilityBlocker
from nodecanon.core.models import (
    KGEdge,
    KGGraph,
    KGNode,
    MergeConflict,
    MergeRecord,
    ScoreVector,
)

MergeStrategy = str  # "longest" | "union" | "newest" | "keep_all"


class ConflictDetector:
    """Detects incompatible type assignments before a merge is executed.

    Uses the same type compatibility rules as TypeCompatibilityBlocker so that
    any pair that slipped through blocking is still caught here.
    """

    def __init__(self) -> None:
        self._type_compat = TypeCompatibilityBlocker()

    def detect(
        self,
        node_a: KGNode,
        node_b: KGNode,
        score: ScoreVector,
    ) -> MergeConflict | None:
        if node_a.type is None or node_b.type is None:
            return None  # no type info — can't confirm a conflict
        if not self._type_compat.are_compatible(node_a.type, node_b.type):
            return MergeConflict(
                node_id_a=node_a.id,
                node_id_b=node_b.id,
                score=score,
                conflict_reason=(
                    f"Incompatible types: {node_a.type!r} vs {node_b.type!r}. "
                    "These entities are unlikely to be the same real-world concept. "
                    "Review manually before merging."
                ),
            )
        return None


class NodeMerger:
    """Merges a group of alias nodes into one canonical node.

    Never mutates input nodes — always returns a new KGNode with provenance.
    """

    def __init__(
        self,
        description_strategy: MergeStrategy = "longest",
        type_strategy: MergeStrategy = "union",
    ) -> None:
        self.description_strategy = description_strategy
        self.type_strategy = type_strategy

    def select_canonical(self, nodes: list[KGNode], graph: KGGraph) -> KGNode:
        """Returns the highest-degree node as canonical.

        Rationale: the most frequently cross-referenced node is the most
        reliably extracted one. Tiebreakers: description length, attribute count.
        """
        adjacency = graph.adjacency_index()
        return max(
            nodes,
            key=lambda n: (
                len(adjacency.get(n.id, [])),
                len(n.description or ""),
                len(n.attributes),
            ),
        )

    def merge(
        self,
        canonical: KGNode,
        aliases: list[KGNode],
        score: ScoreVector,
        strategy: str = "rule_based",
    ) -> tuple[KGNode, MergeRecord]:
        """Merge aliases into canonical, returning an updated node and provenance.

        The returned KGNode is a new object — the original canonical is untouched.
        """
        all_nodes = [canonical] + aliases
        merged = replace(
            canonical,
            description=self._merge_descriptions(all_nodes),
            type=self._merge_types(all_nodes),
            source_chunks=self._merge_source_chunks(all_nodes),
            attributes=self._merge_attributes(all_nodes),
            _merged_from=[n.id for n in all_nodes],
            _merge_evidence=score.to_dict(),
            _merge_strategy=strategy,
            _resolved_types=self._collect_types(all_nodes),
        )
        record = MergeRecord(
            canonical_id=canonical.id,
            merged_ids=[n.id for n in aliases],
            score=score,
            strategy=strategy,
        )
        return merged, record

    # ------------------------------------------------------------------
    # Attribute merge helpers
    # ------------------------------------------------------------------

    def _merge_descriptions(self, nodes: list[KGNode]) -> str | None:
        descs = [n.description for n in nodes if n.description]
        if not descs:
            return None
        return max(descs, key=len)  # "longest" strategy

    def _merge_types(self, nodes: list[KGNode]) -> str | None:
        """Keep canonical's type; full union is stored in _resolved_types."""
        types = list(dict.fromkeys(n.type for n in nodes if n.type))
        return types[0] if types else None

    def _merge_source_chunks(self, nodes: list[KGNode]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for n in nodes:
            for chunk in n.source_chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    result.append(chunk)
        return result

    def _merge_attributes(self, nodes: list[KGNode]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for n in nodes:
            for k, v in n.attributes.items():
                merged.setdefault(k, v)  # canonical's attributes win on conflict
        return merged

    def _collect_types(self, nodes: list[KGNode]) -> list[str] | None:
        types = list(dict.fromkeys(n.type for n in nodes if n.type))
        return types if types else None


class EdgeMerger:
    """Redirects alias edges to canonical node and deduplicates parallel edges."""

    def merge_edges(
        self,
        edges: list[KGEdge],
        alias_to_canonical: dict[str, str],
    ) -> list[KGEdge]:
        """Redirect alias IDs, drop self-loops, then deduplicate parallel edges."""
        redirected: list[KGEdge] = []
        for edge in edges:
            src = alias_to_canonical.get(edge.source_id, edge.source_id)
            tgt = alias_to_canonical.get(edge.target_id, edge.target_id)
            if src == tgt:
                continue  # self-loop produced by merge — not meaningful in a KG
            redirected.append(replace(edge, source_id=src, target_id=tgt))
        return self._deduplicate_parallel(redirected)

    def _deduplicate_parallel(self, edges: list[KGEdge]) -> list[KGEdge]:
        """Merge edges with identical (source, target, relation) by summing weight."""
        groups: dict[tuple[str, str, str], list[KGEdge]] = defaultdict(list)
        for edge in edges:
            groups[(edge.source_id, edge.target_id, edge.relation)].append(edge)

        result: list[KGEdge] = []
        for (src, tgt, rel), group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                merged_attrs: dict[str, Any] = {}
                for e in group:
                    merged_attrs.update(e.attributes)
                result.append(
                    KGEdge(
                        source_id=src,
                        target_id=tgt,
                        relation=rel,
                        weight=sum(e.weight for e in group),
                        attributes=merged_attrs,
                    )
                )
        return result

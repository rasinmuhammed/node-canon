from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from nodecanon.core.blocking import BaseBlocker, UnionBlocker
from nodecanon.core.matching import BaseMatcher, RuleBasedMatcher
from nodecanon.core.merging import ConflictDetector, EdgeMerger, NodeMerger
from nodecanon.core.models import (
    KGGraph,
    KGNode,
    MergeConflict,
    MergeRecord,
    ScoreVector,
)
from nodecanon.core.scoring import NodeScorer


class _UnionFind:
    """Disjoint-set union with path compression and union-by-rank.

    Used to compute transitive merge groups: if A matches B and B matches C,
    all three end up in the same canonical group without re-scoring.
    """

    def __init__(self, ids: list[str]) -> None:
        self._parent: dict[str, str] = {x: x for x in ids}
        self._rank: dict[str, int] = {x: 0 for x in ids}

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> dict[str, list[str]]:
        """Returns {root_id: [all_member_ids]} for every set."""
        result: dict[str, list[str]] = defaultdict(list)
        for node_id in self._parent:
            result[self.find(node_id)].append(node_id)
        return dict(result)


@dataclass
class ResolveResult:
    graph: KGGraph
    merge_records: list[MergeRecord] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)
    original_node_count: int = 0
    original_edge_count: int = 0

    def merge_report(self) -> str:
        aliases_absorbed = sum(len(r.merged_ids) for r in self.merge_records)
        edges_removed = max(0, self.original_edge_count - len(self.graph.edges))
        lines = [
            f"Merged {self.original_node_count} nodes into "
            f"{len(self.graph.nodes)} canonical nodes",
            f"Absorbed {aliases_absorbed} alias nodes",
            f"Removed {edges_removed} redundant edges",
            f"Flagged {len(self.conflicts)} conflicts for human review",
        ]
        return "\n".join(f"→ {line}" for line in lines)


class Resolver:
    """Orchestrates the full four-layer entity resolution pipeline.

    Layers: Block → Score → Match → Merge

    Parameters
    ----------
    blocker:
        Custom blocking strategy. Defaults to UnionBlocker with
        TokenOverlap + NGramFingerprint + TypeCompatibility.
    scorer:
        Custom NodeScorer. Defaults to all-MiniLM-L6-v2 embeddings.
    matcher:
        Custom match decision logic. Defaults to RuleBasedMatcher(0.75).
    node_merger, edge_merger, conflict_detector:
        Override individual merge-layer components if needed.
    """

    def __init__(
        self,
        blocker: BaseBlocker | None = None,
        scorer: NodeScorer | None = None,
        matcher: BaseMatcher | None = None,
        node_merger: NodeMerger | None = None,
        edge_merger: EdgeMerger | None = None,
        conflict_detector: ConflictDetector | None = None,
    ) -> None:
        self.blocker = blocker
        self.scorer = scorer or NodeScorer()
        self.matcher = matcher or RuleBasedMatcher()
        self.node_merger = node_merger or NodeMerger()
        self.edge_merger = edge_merger or EdgeMerger()
        self.conflict_detector = conflict_detector or ConflictDetector()

    def resolve(self, graph: KGGraph) -> ResolveResult:
        """Run the full resolution pipeline and return a ResolveResult.

        Does not modify the input graph — all merges produce new node objects.
        """
        self._validate(graph)

        blocker = self.blocker or self._build_default_blocker()
        self.scorer.fit(graph)

        # ---- Blocking -------------------------------------------------------
        candidates = blocker.candidate_pairs(graph)

        # ---- Scoring + conflict check + match decision ----------------------
        uf = _UnionFind([n.id for n in graph.nodes])
        pair_scores: dict[tuple[str, str], ScoreVector] = {}
        conflicts: list[MergeConflict] = []

        # Score all candidates first, sort by score descending so
        # highest-confidence merges are processed first (greedy approximation
        # of collective resolution — see CLAUDE.md for rationale).
        scored: list[tuple[KGNode, KGNode, ScoreVector]] = [
            (a, b, self.scorer.score(a, b, graph)) for a, b in candidates
        ]
        scored.sort(key=lambda t: t[2].weighted_sum(self.scorer.weights), reverse=True)

        for node_a, node_b, sv in scored:
            conflict = self.conflict_detector.detect(node_a, node_b, sv)
            if conflict is not None:
                conflicts.append(conflict)
                continue
            if self.matcher.is_match(node_a, node_b, sv):
                key = (min(node_a.id, node_b.id), max(node_a.id, node_b.id))
                pair_scores[key] = sv
                uf.union(node_a.id, node_b.id)

        # ---- Merge ----------------------------------------------------------
        node_idx = graph.node_index()
        merge_records: list[MergeRecord] = []
        resolved_nodes: list[KGNode] = []
        alias_to_canonical: dict[str, str] = {}

        for _root, member_ids in uf.groups().items():
            if len(member_ids) == 1:
                resolved_nodes.append(node_idx[member_ids[0]])
                continue

            members = [node_idx[mid] for mid in member_ids]
            canonical = self.node_merger.select_canonical(members, graph)
            aliases = [n for n in members if n.id != canonical.id]

            # Representative score: best pairwise score among canonical ↔ alias.
            # Indirect pairs (e.g. canonical never directly scored against alias C)
            # fall back to ScoreVector(0,...) — still safe, just less informative.
            best_sv = max(
                (
                    pair_scores.get(
                        (min(canonical.id, a.id), max(canonical.id, a.id)),
                        ScoreVector(0.0, 0.0, 0.0, 0.0, 0.0),
                    )
                    for a in aliases
                ),
                key=lambda sv: sv.weighted_sum(self.scorer.weights),
            )

            merged, record = self.node_merger.merge(
                canonical, aliases, best_sv, strategy="rule_based"
            )
            resolved_nodes.append(merged)
            merge_records.append(record)
            for alias in aliases:
                alias_to_canonical[alias.id] = canonical.id

        # ---- Rebuild graph --------------------------------------------------
        resolved_edges = self.edge_merger.merge_edges(
            list(graph.edges), alias_to_canonical
        )
        resolved_graph = KGGraph(nodes=resolved_nodes, edges=resolved_edges)

        return ResolveResult(
            graph=resolved_graph,
            merge_records=merge_records,
            conflicts=conflicts,
            original_node_count=len(graph.nodes),
            original_edge_count=len(graph.edges),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(graph: KGGraph) -> None:
        for node in graph.nodes:
            if not node.name:
                raise ValueError(
                    f"Node '{node.id}' has no name field. "
                    "All nodes must have a non-empty name for blocking to work. "
                    "Set a default name before calling resolve()."
                )

    def _build_default_blocker(self) -> UnionBlocker:
        from nodecanon.core.blocking import (
            NGramFingerprintBlocker,
            TokenOverlapBlocker,
            TypeCompatibilityBlocker,
        )

        return UnionBlocker(
            [
                TokenOverlapBlocker(),
                NGramFingerprintBlocker(),
                TypeCompatibilityBlocker(),
            ]
        )

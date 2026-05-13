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

    def __repr__(self) -> str:
        return (
            f"ResolveResult("
            f"{len(self.graph.nodes)} nodes, "
            f"{len(self.merge_records)} merges, "
            f"{len(self.conflicts)} conflicts)"
        )

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

    def explain(self, node_id: str) -> str:
        """Human-readable explanation of why a node was merged (or wasn't).

        Pass the canonical id or any alias id that was absorbed into a merge.
        """
        node_idx = self.graph.node_index()
        node = node_idx.get(node_id)

        record = next(
            (
                r
                for r in self.merge_records
                if r.canonical_id == node_id or node_id in r.merged_ids
            ),
            None,
        )
        if record is None:
            name = node.name if node else node_id
            return f"'{name}' ({node_id}) was not merged — it is already a canonical singleton."

        canonical = node_idx.get(record.canonical_id)
        canonical_name = canonical.name if canonical else record.canonical_id
        sv = record.score
        w: dict[str, float] = {
            "name_similarity": 0.30,
            "semantic_similarity": 0.25,
            "type_agreement": 0.20,
            "neighbor_overlap": 0.20,
            "description_similarity": 0.05,
        }

        merged_names = (
            [f'  · "{n.name}" (id: {n.id})' for n in record.original_nodes]
            if record.original_nodes
            else [f"  · {mid}" for mid in [record.canonical_id] + record.merged_ids]
        )

        lines = [
            f"Canonical node: {canonical_name!r} (id: {record.canonical_id})",
            "",
            f"Merged from {len(merged_names)} nodes:",
            *merged_names,
            "",
            "Merge evidence:",
            f"  name_similarity:        {sv.name_similarity:.3f}  (weight {w['name_similarity']})",
            f"  semantic_similarity:    {sv.semantic_similarity:.3f}  (weight {w['semantic_similarity']})",
            f"  type_agreement:         {sv.type_agreement:.3f}  (weight {w['type_agreement']})",
            f"  neighbor_overlap:       {sv.neighbor_overlap:.3f}  (weight {w['neighbor_overlap']})",
            f"  description_similarity: {sv.description_similarity:.3f}  (weight {w['description_similarity']})",
            f"  {'─' * 40}",
            f"  weighted score:         {sv.weighted_sum(w):.3f}",
            "",
            f"Merge strategy: {record.strategy}",
        ]
        return "\n".join(lines)

    def reject_merge(
        self,
        canonical_id: str,
        restore: list[str] | None = None,
    ) -> ResolveResult:
        """Undo a merge, restoring original nodes as separate graph members.

        Parameters
        ----------
        canonical_id:
            Id of the canonical node whose merge you want to undo.
        restore:
            Specific alias ids to restore. Defaults to all aliases (full revert).

        Returns a new ResolveResult — the original is not mutated.
        Edges remain on the canonical node; they cannot be split back automatically.
        """
        record = next(
            (r for r in self.merge_records if r.canonical_id == canonical_id), None
        )
        if record is None:
            raise ValueError(
                f"No merge record found for canonical_id={canonical_id!r}. "
                f"Available: {[r.canonical_id for r in self.merge_records]}"
            )
        if not record.original_nodes:
            raise ValueError(
                f"No original_nodes stored in merge record for {canonical_id!r}. "
                "Re-run resolve() to produce editable records."
            )

        restore_set = set(restore) if restore is not None else None
        to_restore = [
            n
            for n in record.original_nodes
            if n.id != canonical_id
            and (restore_set is None or n.id in restore_set)
        ]
        canonical_original = next(
            (n for n in record.original_nodes if n.id == canonical_id), None
        )

        current_idx = self.graph.node_index()
        new_nodes = [n for n in self.graph.nodes if n.id != canonical_id]
        new_nodes.append(canonical_original or current_idx[canonical_id])
        new_nodes.extend(to_restore)

        new_records = [r for r in self.merge_records if r.canonical_id != canonical_id]
        return ResolveResult(
            graph=KGGraph(nodes=new_nodes, edges=list(self.graph.edges)),
            merge_records=new_records,
            conflicts=list(self.conflicts),
            original_node_count=self.original_node_count,
            original_edge_count=self.original_edge_count,
        )

    def force_merge(self, *node_ids: str) -> ResolveResult:
        """Manually merge nodes, overriding the resolver's decision.

        Returns a new ResolveResult — the original is not mutated.
        """
        from nodecanon.core.merging import EdgeMerger, NodeMerger

        node_idx = self.graph.node_index()
        missing = [nid for nid in node_ids if nid not in node_idx]
        if missing:
            raise ValueError(
                f"Node ids not found in the resolved graph: {missing}. "
                f"Available (sample): {list(node_idx)[:8]}"
            )
        nodes_to_merge = [node_idx[nid] for nid in node_ids]
        if len(nodes_to_merge) < 2:
            raise ValueError("force_merge requires at least 2 node ids.")

        merger = NodeMerger()
        canonical = merger.select_canonical(nodes_to_merge, self.graph)
        aliases = [n for n in nodes_to_merge if n.id != canonical.id]
        dummy_score = ScoreVector(0.0, 0.0, 0.0, 0.0, 0.0)
        merged, record = merger.merge(canonical, aliases, dummy_score, strategy="manual")

        merged_id_set = {n.id for n in nodes_to_merge}
        alias_to_canonical = {a.id: canonical.id for a in aliases}
        remaining = [n for n in self.graph.nodes if n.id not in merged_id_set]
        remaining.append(merged)

        new_edges = EdgeMerger().merge_edges(list(self.graph.edges), alias_to_canonical)
        return ResolveResult(
            graph=KGGraph(nodes=remaining, edges=new_edges),
            merge_records=self.merge_records + [record],
            conflicts=list(self.conflicts),
            original_node_count=self.original_node_count,
            original_edge_count=self.original_edge_count,
        )

    def accept_conflict(self, index: int) -> ResolveResult:
        """Accept flagged conflict at position ``index`` and force-merge the pair.

        Returns a new ResolveResult with the conflict removed from the list.
        """
        if index < 0 or index >= len(self.conflicts):
            raise IndexError(
                f"Conflict index {index} is out of range — "
                f"{len(self.conflicts)} conflict(s) available (0-based)."
            )
        conflict = self.conflicts[index]
        merged = self.force_merge(conflict.node_id_a, conflict.node_id_b)
        return ResolveResult(
            graph=merged.graph,
            merge_records=merged.merge_records,
            conflicts=[c for i, c in enumerate(self.conflicts) if i != index],
            original_node_count=self.original_node_count,
            original_edge_count=self.original_edge_count,
        )


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
        """Run up to two resolution passes, stopping early if the graph converges.

        A second pass catches groups that only became neighbours after the first
        merge — e.g. "IBM" and "IBM Corp" share the Watson node as a common
        neighbour once "IBM Corporation" is folded in.  Does not modify the
        input graph.
        """
        self._validate(graph)

        original_node_count = len(graph.nodes)
        original_edge_count = len(graph.edges)
        all_merge_records: list[MergeRecord] = []
        all_conflicts: list[MergeConflict] = []
        current = graph

        for _ in range(2):
            prev_n = len(current.nodes)
            pass_result = self._resolve_pass(current)
            all_merge_records.extend(pass_result.merge_records)
            all_conflicts.extend(pass_result.conflicts)
            current = pass_result.graph
            if len(current.nodes) == prev_n:
                break  # converged — no merges happened, stop early

        return ResolveResult(
            graph=current,
            merge_records=all_merge_records,
            conflicts=all_conflicts,
            original_node_count=original_node_count,
            original_edge_count=original_edge_count,
        )

    def _resolve_pass(self, graph: KGGraph) -> ResolveResult:
        """Single Block → Score → Match → Merge pass."""
        blocker = self.blocker or self._build_default_blocker()
        self.scorer.fit(graph)

        # ---- Blocking -------------------------------------------------------
        candidates = blocker.candidate_pairs(graph)

        # ---- Scoring + conflict check + match decision ----------------------
        uf = _UnionFind([n.id for n in graph.nodes])
        pair_scores: dict[tuple[str, str], ScoreVector] = {}
        conflicts: list[MergeConflict] = []

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
            AbbreviationBlocker,
            NGramFingerprintBlocker,
            TokenOverlapBlocker,
            TypeCompatibilityBlocker,
        )

        return UnionBlocker(
            [
                TokenOverlapBlocker(),
                NGramFingerprintBlocker(),
                AbbreviationBlocker(),
                TypeCompatibilityBlocker(),
            ]
        )

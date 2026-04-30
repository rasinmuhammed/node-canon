from __future__ import annotations

from dataclasses import dataclass, field

from nodecanon.core.models import KGGraph, MergeConflict, MergeRecord


@dataclass
class MergeReport:
    original_node_count: int
    resolved_node_count: int
    merge_records: list[MergeRecord] = field(default_factory=list)
    conflicts: list[MergeConflict] = field(default_factory=list)

    @property
    def reduction_pct(self) -> float:
        if self.original_node_count == 0:
            return 0.0
        return (1 - self.resolved_node_count / self.original_node_count) * 100

    @property
    def aliases_absorbed(self) -> int:
        return sum(len(r.merged_ids) for r in self.merge_records)

    def __str__(self) -> str:
        lines = [
            f"→ {self.original_node_count} nodes → {self.resolved_node_count} "
            f"canonical nodes ({self.reduction_pct:.1f}% reduction)",
            f"→ {self.aliases_absorbed} alias nodes absorbed",
            f"→ {len(self.conflicts)} conflicts flagged for human review",
        ]
        return "\n".join(lines)


@dataclass
class QADeltaReport:
    baseline_accuracy: float
    resolved_accuracy: float
    dataset: str

    @property
    def delta(self) -> float:
        return self.resolved_accuracy - self.baseline_accuracy

    def __str__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return (
            f"QA accuracy on {self.dataset}: "
            f"{self.baseline_accuracy:.3f} → {self.resolved_accuracy:.3f} "
            f"({sign}{self.delta:.3f})"
        )


def evaluate(
    original: KGGraph,
    resolved: KGGraph,
    merge_records: list[MergeRecord],
    conflicts: list[MergeConflict],
) -> MergeReport:
    return MergeReport(
        original_node_count=len(original.nodes),
        resolved_node_count=len(resolved.nodes),
        merge_records=merge_records,
        conflicts=conflicts,
    )

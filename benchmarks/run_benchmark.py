"""
nodecanon synthetic benchmark.

Generates a ground-truth graph of canonical entities plus realistic variant
spellings, runs the resolver, then measures pairwise precision / recall / F1.

Usage:
    python benchmarks/run_benchmark.py          # full run (downloads model once)
    python benchmarks/run_benchmark.py --fast   # string-only, no embeddings

The dataset is self-contained — no external files needed.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import combinations

from nodecanon.core.models import KGEdge, KGGraph, KGNode
from nodecanon.core.resolver import Resolver

# ---------------------------------------------------------------------------
# Ground-truth entity clusters.
# Each entry: (cluster_id, type, [canonical_name, *variants], [anchor_keys])
#
# anchor_keys refer to entries in _ANCHORS below.  Anchors are independent
# nodes that are NOT duplicated — they provide topological signal (neighbour
# overlap) without polluting the ground truth.
# ---------------------------------------------------------------------------

_CLUSTERS: list[tuple[str, str, list[str], list[str]]] = [
    # ── organisations ────────────────────────────────────────────────────────
    (
        "ibm",
        "ORGANIZATION",
        ["IBM", "I.B.M.", "International Business Machines", "IBM Corp"],
        ["a_ginni", "a_watson", "a_armonk"],
    ),
    (
        "microsoft",
        "ORGANIZATION",
        ["Microsoft", "Microsoft Corp", "MSFT"],
        ["a_satya", "a_azure", "a_redmond"],
    ),
    (
        "openai",
        "ORGANIZATION",
        ["OpenAI", "Open AI", "Open AI Inc"],
        ["a_chatgpt", "a_gpt4", "a_sf"],
    ),
    (
        "anthropic",
        "ORGANIZATION",
        ["Anthropic", "Anthropic Inc", "Anthropic AI"],
        ["a_dario", "a_claude_prod", "a_sf"],
    ),
    (
        "google",
        "ORGANIZATION",
        ["Google", "Google LLC", "Alphabet Inc"],
        ["a_sundar", "a_search", "a_mv"],
    ),
    (
        "nvidia",
        "ORGANIZATION",
        ["NVIDIA", "Nvidia Corp", "NVDA"],
        ["a_h100", "a_cuda", "a_sc"],
    ),
    # ── people ───────────────────────────────────────────────────────────────
    (
        "sam_altman",
        "PERSON",
        ["Sam Altman", "Samuel Altman", "S. Altman"],
        ["a_openai_hq", "a_yc"],
    ),
    (
        "elon_musk",
        "PERSON",
        ["Elon Musk", "Elon R. Musk", "E. Musk"],
        ["a_tesla", "a_spacex"],
    ),
    (
        "jensen_huang",
        "PERSON",
        ["Jensen Huang", "Jen-Hsun Huang", "Jensen H. Huang"],
        ["a_nvidia_hq", "a_cuda"],
    ),
    # ── concepts ─────────────────────────────────────────────────────────────
    (
        "llm",
        "CONCEPT",
        ["large language model", "Large Language Models", "LLM"],
        ["a_transformer", "a_gpt4"],
    ),
    (
        "ai",
        "CONCEPT",
        ["artificial intelligence", "Artificial Intelligence", "A.I."],
        ["a_turing", "a_deepmind"],
    ),
    (
        "ml",
        "CONCEPT",
        ["machine learning", "Machine Learning", "ML"],
        ["a_sklearn", "a_gradient"],
    ),
]

# Anchor nodes — appear exactly once, never deduplicated, purely structural.
# No anchor name matches any cluster variant name (checked manually).
_ANCHORS: dict[str, tuple[str, str]] = {
    "a_ginni": ("Ginni Rometty", "PERSON"),
    "a_watson": ("Watson AI", "PRODUCT"),
    "a_armonk": ("Armonk, NY", "LOCATION"),
    "a_satya": ("Satya Nadella", "PERSON"),
    "a_azure": ("Azure Cloud", "PRODUCT"),
    "a_redmond": ("Redmond, WA", "LOCATION"),
    "a_chatgpt": ("ChatGPT", "PRODUCT"),
    "a_gpt4": ("GPT-4", "PRODUCT"),
    "a_sf": ("San Francisco", "LOCATION"),
    "a_dario": ("Dario Amodei", "PERSON"),
    "a_claude_prod": ("Claude (product)", "PRODUCT"),
    "a_sundar": ("Sundar Pichai", "PERSON"),
    "a_search": ("Google Search", "PRODUCT"),
    "a_mv": ("Mountain View, CA", "LOCATION"),
    "a_h100": ("H100 GPU", "PRODUCT"),
    "a_cuda": ("CUDA Toolkit", "PRODUCT"),
    "a_sc": ("Santa Clara, CA", "LOCATION"),
    "a_openai_hq": ("OpenAI HQ", "LOCATION"),
    "a_yc": ("Y Combinator", "ORGANIZATION"),
    "a_tesla": ("Tesla Inc.", "ORGANIZATION"),
    "a_spacex": ("SpaceX", "ORGANIZATION"),
    "a_nvidia_hq": ("NVIDIA Headquarters", "LOCATION"),
    "a_transformer": ("Transformer Architecture", "CONCEPT"),
    "a_turing": ("Turing Test", "CONCEPT"),
    "a_deepmind": ("DeepMind", "ORGANIZATION"),
    "a_sklearn": ("scikit-learn", "PRODUCT"),
    "a_gradient": ("Gradient Descent", "CONCEPT"),
}


@dataclass
class BenchmarkResult:
    n_nodes: int
    n_edges: int
    n_clusters: int
    n_true_pairs: int
    n_resolved_nodes: int
    true_positives: int
    false_positives: int
    false_negatives: int
    elapsed_seconds: float

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def print(self) -> None:
        reduction = (1 - self.n_resolved_nodes / self.n_nodes) * 100
        fp_note = (
            f"  ({self.false_positives} wrong merges)" if self.false_positives else ""
        )
        fn_note = f"  ({self.false_negatives} missed)" if self.false_negatives else ""
        print()
        print("─" * 54)
        print("  nodecanon benchmark results")
        print("─" * 54)
        print(f"  Input:     {self.n_nodes} nodes, {self.n_edges} edges")
        print(
            f"  Output:    {self.n_resolved_nodes} canonical nodes  "
            f"({reduction:.0f}% reduction)"
        )
        print(f"  Clusters:  {self.n_clusters} ground-truth entity groups")
        print(f"  Pairs:     {self.n_true_pairs} pairs that should merge")
        print()
        print(f"  Precision: {self.precision:.3f}{fp_note}")
        print(f"  Recall:    {self.recall:.3f}{fn_note}")
        print(f"  F1:        {self.f1:.3f}")
        print()
        print(f"  Time:      {self.elapsed_seconds:.1f}s")
        print("─" * 54)


def _build_graph() -> tuple[KGGraph, dict[str, str]]:
    """Build the benchmark graph and return (graph, node_id → cluster_id).

    Anchor nodes map to themselves (ground_truth[id] = id), so they're
    automatically excluded from duplicate-pair computation.
    """
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    ground_truth: dict[str, str] = {}

    # Anchor nodes first.
    for anchor_id, (name, entity_type) in _ANCHORS.items():
        nodes.append(KGNode(id=anchor_id, name=name, type=entity_type))
        ground_truth[anchor_id] = anchor_id  # singleton cluster

    # Cluster variant nodes.
    for cluster_id, entity_type, variants, anchor_keys in _CLUSTERS:
        for i, variant_name in enumerate(variants):
            node_id = f"{cluster_id}_{i}"
            nodes.append(KGNode(id=node_id, name=variant_name, type=entity_type))
            ground_truth[node_id] = cluster_id

            # Every variant in a cluster connects to the same anchors, giving
            # each pair a neighbour-Jaccard of 1.0.
            for anchor_id in anchor_keys:
                edges.append(
                    KGEdge(
                        source_id=node_id, target_id=anchor_id, relation="RELATED_TO"
                    )
                )

    return KGGraph(nodes=nodes, edges=edges), ground_truth


def _true_pairs(ground_truth: dict[str, str]) -> set[tuple[str, str]]:
    """All canonical (a, b) pairs that belong to the same non-singleton cluster."""
    cluster_members: dict[str, list[str]] = {}
    for node_id, cluster_id in ground_truth.items():
        if node_id != cluster_id:  # skip anchor singletons
            cluster_members.setdefault(cluster_id, []).append(node_id)
    pairs: set[tuple[str, str]] = set()
    for members in cluster_members.values():
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return pairs


def _resolved_pairs(result_graph: KGGraph) -> set[tuple[str, str]]:
    """All (a, b) pairs that the resolver merged (read from _merged_from)."""
    pairs: set[tuple[str, str]] = set()
    for node in result_graph.nodes:
        if node._merged_from and len(node._merged_from) > 1:
            for a, b in combinations(sorted(node._merged_from), 2):
                pairs.add((a, b))
    return pairs


def run(fast: bool = False) -> BenchmarkResult:
    graph, ground_truth = _build_graph()
    true_p = _true_pairs(ground_truth)

    if fast:
        from unittest.mock import patch

        import numpy as np

        from nodecanon.core.scoring import NodeScorer

        def _fast_fit(self: NodeScorer, g: KGGraph) -> None:
            rng = np.random.default_rng(42)
            for n in g.nodes:
                v = rng.standard_normal(384).astype(np.float32)
                v /= np.linalg.norm(v)
                self._name_emb_cache[n.id] = v
            self._adjacency_index = g.adjacency_index()
            self._node_index = g.node_index()

        with patch.object(NodeScorer, "fit", _fast_fit):
            t0 = time.perf_counter()
            result = Resolver().resolve(graph)
            elapsed = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        result = Resolver().resolve(graph)
        elapsed = time.perf_counter() - t0

    resolved = _resolved_pairs(result.graph)

    tp = len(true_p & resolved)
    fp = len(resolved - true_p)
    fn = len(true_p - resolved)

    return BenchmarkResult(
        n_nodes=len(graph.nodes),
        n_edges=len(graph.edges),
        n_clusters=len(_CLUSTERS),
        n_true_pairs=len(true_p),
        n_resolved_nodes=len(result.graph.nodes),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nodecanon benchmark")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip embedding model; use string + graph topology only.",
    )
    args = parser.parse_args()

    mode = "fast (string + topology)" if args.fast else "full (with embeddings)"
    print(f"\nRunning nodecanon benchmark — {mode} …")
    run(fast=args.fast).print()

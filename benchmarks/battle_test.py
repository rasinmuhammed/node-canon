"""
Large-scale battle test: performance at scale + real-world alias quality.

Two tests:

  1. FB15k-237 scale test (~14k nodes, 310k edges) — validates the resolver
     handles real-scale graphs without crashing. Uses fast mode (no embeddings).
     Note: FB15k-237 uses Freebase machine IDs, not natural language names —
     this is purely a throughput/stability test, not a quality test.

  2. Real-world alias quality test — real organization and person aliases from
     a curated offline dataset, optionally augmented with live Wikidata results.
     Builds a graph where each alias is a separate node and measures resolution
     quality (precision / recall / F1) against known ground truth.

Usage:
    python benchmarks/battle_test.py              # both tests
    python benchmarks/battle_test.py --fb15k      # scale test only
    python benchmarks/battle_test.py --aliases    # alias quality test only
    python benchmarks/battle_test.py --fb15k --sample 2000
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from itertools import combinations
from unittest.mock import patch

from nodecanon.core.matching import RuleBasedMatcher
from nodecanon.core.models import KGEdge, KGGraph, KGNode
from nodecanon.core.resolver import Resolver
from nodecanon.core.scoring import NodeScorer

# ---------------------------------------------------------------------------
# Fast-mode helpers
# ---------------------------------------------------------------------------

_FAST_WEIGHTS = {
    "name_similarity": 0.43,
    "semantic_similarity": 0.00,
    "type_agreement": 0.29,
    "neighbor_overlap": 0.29,
    "description_similarity": 0.00,
}


def _fast_fit(self: NodeScorer, g: KGGraph) -> None:
    self._adjacency_index = g.adjacency_index()
    self._node_index = g.node_index()


def _zero(_self: NodeScorer, _a: KGNode, _b: KGNode) -> float:
    return 0.0


def _make_fast_resolver(threshold: float = 0.72) -> Resolver:
    scorer = NodeScorer(weights=_FAST_WEIGHTS, cache_dir=None)
    matcher = RuleBasedMatcher(threshold=threshold, weights=_FAST_WEIGHTS)
    return Resolver(scorer=scorer, matcher=matcher)


# ---------------------------------------------------------------------------
# Test 1: FB15k-237 scale test
# ---------------------------------------------------------------------------

_FB15K_BASE = "https://raw.githubusercontent.com/thunlp/OpenKE/master/benchmarks/FB15K237"
_ENTITY_URL = f"{_FB15K_BASE}/entity2id.txt"
_TRAIN_URL = f"{_FB15K_BASE}/train2id.txt"
_VALID_URL = f"{_FB15K_BASE}/valid2id.txt"


@dataclass
class ScaleResult:
    n_nodes_in: int
    n_edges_in: int
    n_nodes_out: int
    n_edges_out: int
    n_merges: int
    elapsed_seconds: float

    def print(self) -> None:
        reduction = (1 - self.n_nodes_out / self.n_nodes_in) * 100 if self.n_nodes_in else 0
        print()
        print("─" * 62)
        print("  FB15k-237 scale test  (throughput / stability)")
        print("─" * 62)
        print(f"  Input:      {self.n_nodes_in:,} nodes,  {self.n_edges_in:,} edges")
        print(f"  Output:     {self.n_nodes_out:,} nodes,  {self.n_edges_out:,} edges  ({reduction:.0f}% reduction)")
        print(f"  Merges:     {self.n_merges:,} canonical groups formed")
        print(f"  Time:       {self.elapsed_seconds:.1f}s")
        print(f"  Throughput: {self.n_nodes_in / self.elapsed_seconds:,.0f} nodes/s")
        print()
        print("  ⚠  FB15k-237 uses Freebase machine IDs, not natural language.")
        print("     Reported merges may be spurious — use alias test for quality.")
        print("─" * 62)


def _fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "nodecanon-battle-test/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _load_fb15k(sample: int | None = None) -> KGGraph:
    print("  Downloading entity list...", end=" ", flush=True)
    entity_text = _fetch_text(_ENTITY_URL)
    lines = entity_text.strip().splitlines()
    n_total = int(lines[0])
    id_to_name: dict[int, str] = {}
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            name, eid = parts[0], int(parts[1])
            id_to_name[eid] = name
    print(f"{n_total:,} entities")

    if sample is not None and sample < n_total:
        id_to_name = dict(list(id_to_name.items())[:sample])
        print(f"  Sampled to {sample:,} entities")

    print("  Downloading triples (train + valid)...", end=" ", flush=True)
    edges: list[KGEdge] = []
    for url in (_TRAIN_URL, _VALID_URL):
        text = _fetch_text(url)
        triple_lines = text.strip().splitlines()
        for line in triple_lines[1:]:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            if h in id_to_name and t in id_to_name:
                edges.append(KGEdge(source_id=str(h), target_id=str(t), relation=str(r)))
    print(f"{len(edges):,} edges")

    nodes = [
        KGNode(id=str(eid), name=name, type="ENTITY") for eid, name in id_to_name.items()
    ]
    return KGGraph(nodes=nodes, edges=edges)


def run_fb15k(sample: int | None = None) -> ScaleResult:
    print("\nFB15k-237 scale test")
    print("━" * 62)
    graph = _load_fb15k(sample=sample)
    n_in = len(graph.nodes)
    e_in = len(graph.edges)

    resolver = _make_fast_resolver()
    print("  Resolving...", end=" ", flush=True)
    t0 = time.perf_counter()
    with (
        patch.object(NodeScorer, "fit", _fast_fit),
        patch.object(NodeScorer, "_semantic_similarity", _zero),
        patch.object(NodeScorer, "_description_similarity", _zero),
    ):
        result = resolver.resolve(graph)
    elapsed = time.perf_counter() - t0
    print(f"done ({elapsed:.1f}s)")

    return ScaleResult(
        n_nodes_in=n_in,
        n_edges_in=e_in,
        n_nodes_out=len(result.graph.nodes),
        n_edges_out=len(result.graph.edges),
        n_merges=len(result.merge_records),
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Test 2: Real-world alias quality test
# ---------------------------------------------------------------------------

# Curated real-world aliases — public knowledge, no synthetic generation.
# Format: (entity_type, canonical_name, [alias1, alias2, ...])
# Sources: Wikidata, Wikipedia, public corporate filings.
# Format: (entity_type, canonical_name, [aliases], [anchor_node_names])
# Anchor names are real distinct entities that serve as shared context.
# Each variant in a cluster connects to all anchors in that cluster.
_OFFLINE_ALIASES: list[tuple[str, str, list[str], list[str]]] = [
    # ── Organizations ────────────────────────────────────────────────────
    ("ORGANIZATION", "International Business Machines", [
        "IBM", "I.B.M.", "IBM Corp",
    ], ["Armonk New York", "Watson supercomputer", "Ginni Rometty"]),
    ("ORGANIZATION", "Microsoft Corporation", [
        "Microsoft", "MSFT", "Microsoft Corp",
    ], ["Redmond Washington", "Bill Gates", "Azure cloud platform"]),
    ("ORGANIZATION", "Alphabet Inc", [
        "Google", "Google LLC", "Google Inc",
    ], ["Mountain View California", "Larry Page", "Google Search engine"]),
    ("ORGANIZATION", "Meta Platforms", [
        "Meta", "Facebook", "Facebook Inc", "Meta Inc",
    ], ["Menlo Park California", "Mark Zuckerberg", "Instagram app"]),
    ("ORGANIZATION", "Apple Inc", [
        "Apple", "Apple Computer",
    ], ["Cupertino California", "Tim Cook", "iPhone smartphone"]),
    ("ORGANIZATION", "Amazon.com", [
        "Amazon", "Amazon Web Services",
    ], ["Seattle Washington", "Jeff Bezos", "Prime delivery"]),
    ("ORGANIZATION", "NVIDIA Corporation", [
        "NVIDIA", "Nvidia", "NVDA", "Nvidia Corp",
    ], ["Santa Clara California", "Jensen Huang", "CUDA toolkit"]),
    ("ORGANIZATION", "Tesla Inc", [
        "Tesla", "Tesla Motors",
    ], ["Austin Texas", "Elon Musk", "Model S electric car"]),
    ("ORGANIZATION", "JPMorgan Chase", [
        "JPMorgan", "JP Morgan", "J.P. Morgan",
    ], ["New York headquarters", "Jamie Dimon", "Chase Bank branch"]),
    ("ORGANIZATION", "Goldman Sachs", [
        "Goldman", "Goldman Sachs Group",
    ], ["Lower Manhattan office", "David Solomon", "investment banking"]),
    ("ORGANIZATION", "McKinsey & Company", [
        "McKinsey", "McKinsey and Company",
    ], ["management consulting firm", "Marvin Bower", "strategy report"]),
    ("ORGANIZATION", "Massachusetts Institute of Technology", [
        "MIT", "M.I.T.",
    ], ["Cambridge Massachusetts", "Noam Chomsky", "MIT Media Lab"]),
    ("ORGANIZATION", "World Health Organization", [
        "WHO", "W.H.O.",
    ], ["Geneva Switzerland", "Tedros Adhanom", "public health treaty"]),
    ("ORGANIZATION", "United Nations", [
        "UN", "U.N.",
    ], ["East River New York", "Security Council chamber", "Ban Ki-moon"]),
    # ── People ───────────────────────────────────────────────────────────
    ("PERSON", "Elon Musk", [
        "Elon R. Musk", "E. Musk",
    ], ["SpaceX rockets", "Tesla CEO", "PayPal founder"]),
    ("PERSON", "Jeff Bezos", [
        "Jeffrey Bezos", "Jeffrey P. Bezos", "J. Bezos",
    ], ["Amazon founder", "Blue Origin spacecraft", "Washington Post owner"]),
    ("PERSON", "Warren Buffett", [
        "Warren E. Buffett", "W. Buffett",
    ], ["Berkshire Hathaway chairman", "Omaha Nebraska investor", "Oracle of Omaha"]),
    ("PERSON", "Sundar Pichai", [
        "Sundar P.", "S. Pichai",
    ], ["Google CEO role", "Alphabet executive", "Chrome browser creator"]),
    ("PERSON", "Satya Nadella", [
        "S. Nadella",
    ], ["Microsoft CEO role", "Azure growth", "cloud-first strategy"]),
    # ── Concepts ─────────────────────────────────────────────────────────
    ("CONCEPT", "Artificial Intelligence", [
        "AI", "A.I.",
    ], ["Turing test benchmark", "neural network model", "Alan Turing paper"]),
    ("CONCEPT", "Machine Learning", [
        "ML",
    ], ["gradient descent optimization", "training dataset", "scikit-learn library"]),
    ("CONCEPT", "Large Language Model", [
        "LLM", "large language models",
    ], ["transformer architecture", "GPT-4 model", "token prediction"]),
    ("CONCEPT", "Natural Language Processing", [
        "NLP", "N.L.P.",
    ], ["text parsing pipeline", "named entity recognition", "spaCy library"]),
    ("CONCEPT", "Retrieval Augmented Generation", [
        "RAG",
    ], ["vector database store", "semantic search index", "LlamaIndex framework"]),
    # ── Locations ────────────────────────────────────────────────────────
    ("LOCATION", "United States of America", [
        "USA", "U.S.A.", "United States", "U.S.", "US",
    ], ["Washington DC capital", "Congress legislature", "Constitution document"]),
    ("LOCATION", "United Kingdom", [
        "UK", "U.K.", "Britain", "Great Britain",
    ], ["London capital city", "Parliament building", "pound sterling currency"]),
    ("LOCATION", "New York City", [
        "NYC", "New York", "N.Y.C.",
    ], ["Manhattan island borough", "Times Square landmark", "Statue of Liberty"]),
    ("LOCATION", "San Francisco", [
        "SF", "S.F.",
    ], ["Golden Gate Bridge", "Bay Area peninsula", "Silicon Valley tech hub"]),
]

_WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
_SPARQL_QUERY = """
SELECT DISTINCT ?canonical_name ?alias WHERE {
  ?item wdt:P31/wdt:P279* wd:Q4830453 .
  ?item rdfs:label ?canonical_name .
  ?item skos:altLabel ?alias .
  FILTER(LANG(?canonical_name) = "en")
  FILTER(LANG(?alias) = "en")
  FILTER(?canonical_name != ?alias)
  FILTER(STRLEN(?canonical_name) > 3)
  FILTER(STRLEN(?alias) > 1)
}
LIMIT 2000
"""


@dataclass
class QualityResult:
    n_entities: int
    n_total_nodes: int
    n_true_pairs: int
    true_positives: int
    false_positives: int
    false_negatives: int
    elapsed_seconds: float
    source: str

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
        fp_note = f"  ({self.false_positives} wrong)" if self.false_positives else ""
        fn_note = f"  ({self.false_negatives} missed)" if self.false_negatives else ""
        print()
        print("─" * 62)
        print(f"  Real-world alias quality test  [{self.source}]")
        print("─" * 62)
        print(f"  Entities:    {self.n_entities:,}")
        print(f"  Total nodes: {self.n_total_nodes:,}  (canonical + aliases)")
        print(f"  True pairs:  {self.n_true_pairs:,}")
        print()
        print(f"  Precision:   {self.precision:.3f}{fp_note}")
        print(f"  Recall:      {self.recall:.3f}{fn_note}")
        print(f"  F1:          {self.f1:.3f}")
        print()
        print(f"  Time:        {self.elapsed_seconds:.1f}s")
        print("─" * 62)


def _try_fetch_wikidata() -> dict[str, list[str]] | None:
    """Returns {canonical: [aliases]} or None on failure."""
    import time as _time

    for attempt in range(2):
        if attempt:
            print("  Retrying in 60s...", end=" ", flush=True)
            _time.sleep(62)
        try:
            encoded = urllib.parse.urlencode({"query": _SPARQL_QUERY, "format": "json"})
            url = f"{_WIKIDATA_SPARQL}?{encoded}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "nodecanon-battle-test/1.0 (https://github.com/rasinmuhammed/node-canon)",
                    "Accept": "application/sparql-results+json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result: dict[str, set[str]] = {}
            for binding in data["results"]["bindings"]:
                canonical = binding["canonical_name"]["value"].strip()
                alias = binding["alias"]["value"].strip()
                if canonical and alias and canonical.lower() != alias.lower():
                    result.setdefault(canonical, set()).add(alias)
            return {k: sorted(v) for k, v in result.items() if v}
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                print(f"  rate-limited (429)", end=" ", flush=True)
                continue
            print(f"  HTTP {exc.code}", end=" ", flush=True)
            return None
        except Exception as exc:
            print(f"  {exc}", end=" ", flush=True)
            return None
    return None


def _build_alias_graph(
    alias_data: list[tuple[str, str, list[str], list[str]]],
) -> tuple[KGGraph, dict[str, str]]:
    """Builds graph with shared real-named anchor nodes per cluster.

    Each cluster's anchors are distinct real entities (locations, people,
    products) so their embeddings won't bleed cross-cluster topology signal.
    Wikidata entries use generated anchor names when real ones aren't available.
    """
    nodes: list[KGNode] = []
    edges: list[KGEdge] = []
    ground_truth: dict[str, str] = {}

    for cluster_idx, (entity_type, canonical, aliases, anchor_names) in enumerate(
        alias_data
    ):
        anchor_ids = [f"anc_{cluster_idx}_{i}" for i in range(len(anchor_names))]
        for aid, aname in zip(anchor_ids, anchor_names):
            nodes.append(KGNode(id=aid, name=aname, type="CONCEPT"))

        c_id = f"c_{cluster_idx}"
        nodes.append(KGNode(id=c_id, name=canonical, type=entity_type))
        ground_truth[c_id] = canonical
        for aid in anchor_ids:
            edges.append(KGEdge(source_id=c_id, target_id=aid, relation="RELATED_TO"))

        for j, alias in enumerate(aliases):
            a_id = f"a_{cluster_idx}_{j}"
            nodes.append(KGNode(id=a_id, name=alias, type=entity_type))
            ground_truth[a_id] = canonical
            for aid in anchor_ids:
                edges.append(KGEdge(source_id=a_id, target_id=aid, relation="RELATED_TO"))

    return KGGraph(nodes=nodes, edges=edges), ground_truth


def _true_pairs(ground_truth: dict[str, str]) -> set[tuple[str, str]]:
    clusters: dict[str, list[str]] = {}
    for node_id, canonical in ground_truth.items():
        clusters.setdefault(canonical, []).append(node_id)
    pairs: set[tuple[str, str]] = set()
    for members in clusters.values():
        if len(members) < 2:
            continue
        for a, b in combinations(sorted(members), 2):
            pairs.add((a, b))
    return pairs


def _resolved_pairs(result_graph: KGGraph) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for node in result_graph.nodes:
        if node._merged_from and len(node._merged_from) > 1:
            for a, b in combinations(sorted(node._merged_from), 2):
                pairs.add((a, b))
    return pairs


def run_aliases(try_wikidata: bool = True, fast: bool = False) -> QualityResult:
    print("\nReal-world alias quality test")
    print("━" * 62)

    alias_data = list(_OFFLINE_ALIASES)
    source = "offline"

    if try_wikidata:
        print("  Querying Wikidata SPARQL...", end=" ", flush=True)
        wikidata = _try_fetch_wikidata()
        if wikidata:
            n_fetched = len(wikidata)
            print(f"{n_fetched:,} entities with aliases")
            for canonical, aliases in wikidata.items():
                # Wikidata entries don't have hand-curated anchors — generate
                # two opaque cluster-specific names so topology is correct
                # without embedding bleed (numbers differ between clusters).
                slug = canonical[:20].replace(" ", "_").lower()
                anchors = [f"{slug}_ctx_0", f"{slug}_ctx_1"]
                alias_data.append(("ORGANIZATION", canonical, aliases, anchors))
            source = f"offline + Wikidata ({n_fetched} entities)"
        else:
            print("unavailable — using offline dataset only")

    graph, ground_truth = _build_alias_graph(alias_data)
    true_p = _true_pairs(ground_truth)
    n_entities = len(alias_data)
    print(f"  Built graph: {len(graph.nodes):,} nodes, {len(true_p):,} true pairs")

    if fast:
        resolver = _make_fast_resolver(threshold=0.72)
        print("  Resolving (fast mode, no embeddings)...", end=" ", flush=True)
        ctx = (
            patch.object(NodeScorer, "fit", _fast_fit),
            patch.object(NodeScorer, "_semantic_similarity", _zero),
            patch.object(NodeScorer, "_description_similarity", _zero),
        )
        t0 = time.perf_counter()
        with ctx[0], ctx[1], ctx[2]:
            result = resolver.resolve(graph)
    else:
        print("  Resolving (full mode with embeddings)...")
        print("  Note: first run downloads all-MiniLM-L6-v2 (~90MB) if not cached.")
        resolver = Resolver()
        t0 = time.perf_counter()
        result = resolver.resolve(graph)
    elapsed = time.perf_counter() - t0
    print(f"  Done ({elapsed:.1f}s)")

    resolved = _resolved_pairs(result.graph)
    tp = len(true_p & resolved)
    fp = len(resolved - true_p)
    fn = len(true_p - resolved)

    if fn:
        missed_pairs = true_p - resolved
        node_idx = graph.node_index()
        missed_names = {
            (node_idx[a].name, node_idx[b].name)
            for a, b in list(missed_pairs)[:5]
        }
        print(f"  Missed pairs (sample): {missed_names}")

    return QualityResult(
        n_entities=n_entities,
        n_total_nodes=len(graph.nodes),
        n_true_pairs=len(true_p),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        elapsed_seconds=elapsed,
        source=source,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nodecanon battle test")
    parser.add_argument("--fb15k", action="store_true", help="FB15k-237 scale test")
    parser.add_argument(
        "--aliases", action="store_true", help="Real-world alias quality test"
    )
    parser.add_argument(
        "--no-wikidata",
        action="store_true",
        help="Skip live Wikidata fetch, use offline dataset only",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode (no embeddings) for alias quality test",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N entities from FB15k-237 (default: all ~14k)",
    )
    args = parser.parse_args()
    run_all = not args.fb15k and not args.aliases

    if args.fb15k or run_all:
        run_fb15k(sample=args.sample).print()

    if args.aliases or run_all:
        run_aliases(try_wikidata=not args.no_wikidata, fast=args.fast).print()

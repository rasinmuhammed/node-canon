"""
DBpedia real-world entity resolution benchmark.

Ground truth: DBpedia ``wikiPageRedirects`` — when Wikipedia redirects
"I.B.M." → "IBM" that redirect is an entity alias.  We download company
and person redirect pairs, filter to genuine name-variant aliases, build
a KGGraph where both names appear as separate nodes sharing real DBpedia
anchor connections (founders, parent companies), run nodecanon, and
measure precision / recall / F1.

Two benchmark tiers are reported:
  Name-only   — pairs where nodecanon must rely on string/semantic similarity
  With topology — pairs where shared DBpedia anchors provide neighborhood signal

Usage
-----
    python benchmarks/dbpedia_benchmark.py           # full run (downloads + embeddings)
    python benchmarks/dbpedia_benchmark.py --fast    # no embeddings, string signals only
    python benchmarks/dbpedia_benchmark.py --offline # use cached data only
    python benchmarks/dbpedia_benchmark.py --limit N # cap entity pairs (default 300)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

CACHE_DIR = Path(__file__).parent / "datasets" / "dbpedia"
SPARQL_ENDPOINT = "https://dbpedia.org/sparql"
_HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "nodecanon-benchmark/0.1",
}

# Minimum rapidfuzz token_sort_ratio to consider a redirect pair a genuine alias.
# Filters out "Pete Dreissigacker → Concept2" (founder redirected to company),
# album/product pages redirected to parent label, etc.
_MIN_ALIAS_SIMILARITY = 50

# Wikipedia "List of …" pages that redirect to an entity are not entity aliases.
_NOISE_PREFIXES = ("list of ", "category:", "template:", "file:")


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------


def _sparql(query: str, retries: int = 3) -> list[dict[str, Any]]:
    url = SPARQL_ENDPOINT + "?" + urllib.parse.urlencode(
        {"query": query, "format": "json"}
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                return data["results"]["bindings"]
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  SPARQL attempt {attempt + 1} failed ({exc}), retrying in 5s…")
            time.sleep(5)
    return []


def _label(binding: dict, key: str) -> str:
    return binding[key]["value"]


# ---------------------------------------------------------------------------
# Ground truth filtering
# ---------------------------------------------------------------------------


def _is_plausible_alias(canon: str, alias: str) -> bool:
    """True when the redirect looks like a genuine name variant for the same entity.

    Rejects redirects that are:
    - Founders / employees redirected to their company
    - Albums / products redirected to the parent label
    - Completely different strings (low token-level similarity)
    - Wikipedia list / category / template pages
    """
    a, b = canon.lower().strip(), alias.lower().strip()
    if any(a.startswith(p) or b.startswith(p) for p in _NOISE_PREFIXES):
        return False
    # Direct substring containment catches "ABCmouse" ↔ "ABCmouse.com Early…"
    if a in b or b in a:
        return True
    return fuzz.token_sort_ratio(canon, alias) >= _MIN_ALIAS_SIMILARITY


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def _fetch_redirect_pairs(limit: int) -> list[tuple[str, str]]:
    """Return (canonical_label, alias_label) pairs from DBpedia redirects."""
    pairs: list[tuple[str, str]] = []

    # Companies: ~2/3 of the limit
    company_q = f"""
    SELECT DISTINCT ?canoLabel ?redirLabel WHERE {{
      ?redir dbo:wikiPageRedirects ?canonical .
      ?canonical a dbo:Company .
      ?canonical rdfs:label ?canoLabel .
      ?redir rdfs:label ?redirLabel .
      FILTER(lang(?canoLabel) = "en" && lang(?redirLabel) = "en")
      FILTER(strlen(str(?redirLabel)) < 70)
      FILTER(strlen(str(?canoLabel)) < 70)
      FILTER(?canoLabel != ?redirLabel)
    }} LIMIT {limit * 2}
    """
    for row in _sparql(company_q):
        canon = _label(row, "canoLabel")
        alias = _label(row, "redirLabel")
        if _is_plausible_alias(canon, alias):
            pairs.append((canon, alias))
        if len(pairs) >= (limit * 2) // 3:
            break

    # Persons: ~1/3 of the limit
    person_q = f"""
    SELECT DISTINCT ?canoLabel ?redirLabel WHERE {{
      ?redir dbo:wikiPageRedirects ?canonical .
      ?canonical a dbo:Person .
      ?canonical rdfs:label ?canoLabel .
      ?redir rdfs:label ?redirLabel .
      FILTER(lang(?canoLabel) = "en" && lang(?redirLabel) = "en")
      FILTER(strlen(str(?redirLabel)) < 70)
      FILTER(strlen(str(?canoLabel)) < 70)
      FILTER(?canoLabel != ?redirLabel)
    }} LIMIT {limit}
    """
    person_pairs: list[tuple[str, str]] = []
    for row in _sparql(person_q):
        canon = _label(row, "canoLabel")
        alias = _label(row, "redirLabel")
        if _is_plausible_alias(canon, alias):
            person_pairs.append((canon, alias))
        if len(person_pairs) >= limit // 3:
            break
    pairs.extend(person_pairs)

    return pairs[:limit]


def _fetch_anchors(canonical_names: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Return entity-specific anchor connections from DBpedia.

    We intentionally use only entity-specific relations (founders, parent
    companies, employers, research fields) — NOT generic ones like location /
    birthPlace / nationality that would be shared by many entities and create
    false topology overlap across clusters.
    """
    if not canonical_names:
        return {}

    safe = [n.replace('"', '\\"') for n in canonical_names]
    values_clause = " ".join(f'"{n}"@en' for n in safe[:80])

    query = f"""
    SELECT ?canoLabel ?anchorLabel ?rel WHERE {{
      VALUES ?canoLabel {{ {values_clause} }}
      ?canonical rdfs:label ?canoLabel .
      {{
        ?canonical dbo:foundedBy ?anchor . BIND("foundedBy" AS ?rel)
      }} UNION {{
        ?canonical dbo:parentCompany ?anchor . BIND("parentCompany" AS ?rel)
      }} UNION {{
        ?canonical dbo:employer ?anchor . BIND("employer" AS ?rel)
      }} UNION {{
        ?canonical dbo:field ?anchor . BIND("field" AS ?rel)
      }}
      ?anchor rdfs:label ?anchorLabel .
      FILTER(lang(?anchorLabel) = "en")
      FILTER(strlen(str(?anchorLabel)) < 60)
    }} LIMIT {len(canonical_names) * 5}
    """
    rows = _sparql(query)
    result: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        canon = _label(row, "canoLabel")
        anchor = _label(row, "anchorLabel")
        rel = _label(row, "rel")
        result.setdefault(canon, []).append((anchor, rel))
    return result


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph(
    pairs: list[tuple[str, str]],
    anchors: dict[str, list[tuple[str, str]]],
) -> tuple[Any, list[tuple[str, str]], set[str]]:
    """Build a KGGraph containing canonical + alias nodes with shared edges.

    Anchor nodes are given cluster-scoped IDs (e.g. ``anc_ibm_thomas_watson``)
    so that anchor node merging doesn't pollute cross-cluster topology.
    Since neighbor_overlap uses node names, and founders like "Thomas J. Watson"
    are unique to IBM, the topology signal is clean even with name-based Jaccard.

    Returns (graph, ground_truth_pairs, ids_with_topology).
    """
    from nodecanon import GraphBuilder

    builder = GraphBuilder()
    ground_truth: list[tuple[str, str]] = []
    ids_with_topology: set[str] = set()

    for canonical_name, alias_name in pairs:
        canon_id = "c_" + canonical_name.lower().replace(" ", "_")[:40]
        alias_id = "a_" + alias_name.lower().replace(" ", "_")[:40]

        # Determine entity type hint from name
        entity_type = "ORGANIZATION" if any(
            t in canonical_name for t in ("Inc", "Corp", "Ltd", "Company", "Records",
                                           "Group", "Bank", "Fund", "Holdings")
        ) else "ENTITY"

        builder.add_node(canonical_name, id=canon_id, type=entity_type)
        builder.add_node(alias_name, id=alias_id, type=entity_type)
        ground_truth.append((canon_id, alias_id))

        # Cluster-scoped anchor nodes: unique ID per (canonical, anchor) pair
        cluster_key = canonical_name[:20].lower().replace(" ", "_")
        for anchor_name, rel in anchors.get(canonical_name, []):
            anchor_id = f"anc_{cluster_key}_{anchor_name.lower().replace(' ', '_')[:25]}"
            builder.add_node(anchor_name, id=anchor_id)
            builder.add_edge(canon_id, anchor_id, rel.upper())
            builder.add_edge(alias_id, anchor_id, rel.upper())
            ids_with_topology.add(canon_id)

    return builder.build(), ground_truth, ids_with_topology


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    result: Any,
    ground_truth: list[tuple[str, str]],
    ids_with_topology: set[str] | None = None,
) -> dict[str, Any]:
    alias_to_canonical: dict[str, str] = {}
    for record in result.merge_records:
        for aid in record.merged_ids:
            alias_to_canonical[aid] = record.canonical_id

    def _merged(canon_id: str, alias_id: str) -> bool:
        return (
            alias_to_canonical.get(alias_id) == canon_id
            or alias_to_canonical.get(canon_id) == alias_id
            or (
                alias_to_canonical.get(alias_id) is not None
                and alias_to_canonical.get(alias_id) == alias_to_canonical.get(canon_id)
            )
        )

    gt_set = {(min(c, a), max(c, a)) for c, a in ground_truth}

    tp = fp = fn = 0
    tp_topo = fn_topo = 0
    for canon_id, alias_id in ground_truth:
        has_topo = ids_with_topology is not None and canon_id in ids_with_topology
        if _merged(canon_id, alias_id):
            tp += 1
            if has_topo:
                tp_topo += 1
        else:
            fn += 1
            if has_topo:
                fn_topo += 1

    predicted_pairs: set[tuple[str, str]] = set()
    for record in result.merge_records:
        for aid in record.merged_ids:
            pair = (min(aid, record.canonical_id), max(aid, record.canonical_id))
            predicted_pairs.add(pair)

    for pair in predicted_pairs:
        if pair not in gt_set:
            fp += 1

    def _metrics(tp_: int, fp_: int, fn_: int) -> dict[str, float]:
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f, "tp": float(tp_), "fp": float(fp_), "fn": float(fn_)}

    m = _metrics(tp, fp, fn)
    m["topo"] = _metrics(tp_topo, 0, fn_topo)  # type: ignore[assignment]
    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DBpedia entity resolution benchmark")
    parser.add_argument("--fast", action="store_true", help="No embeddings (string signals only)")
    parser.add_argument("--offline", action="store_true", help="Use cached data only (no network)")
    parser.add_argument("--limit", type=int, default=300, help="Max entity pairs to download")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pairs_cache = CACHE_DIR / "redirect_pairs.json"
    anchors_cache = CACHE_DIR / "anchor_data.json"

    # ------------------------------------------------------------------
    # Load or download
    # ------------------------------------------------------------------
    if args.offline and pairs_cache.exists():
        print("Loading cached DBpedia data…")
        pairs: list[tuple[str, str]] = [tuple(p) for p in json.loads(pairs_cache.read_text())]  # type: ignore[misc]
        anchors: dict[str, list[tuple[str, str]]] = {
            k: [tuple(v) for v in vl]  # type: ignore[misc]
            for k, vl in json.loads(anchors_cache.read_text()).items()
        } if anchors_cache.exists() else {}
    else:
        print(f"Querying DBpedia SPARQL for up to {args.limit} redirect pairs…")
        pairs = _fetch_redirect_pairs(args.limit)
        print(f"  Got {len(pairs)} plausible alias pairs (after similarity filter).")

        canonical_names = list({c for c, _ in pairs})
        print(f"Fetching entity-specific anchors for {len(canonical_names)} canonical entities…")
        anchors = {}
        batch = 80
        for i in range(0, len(canonical_names), batch):
            chunk = canonical_names[i : i + batch]
            anchors.update(_fetch_anchors(chunk))
            print(f"  {min(i + batch, len(canonical_names))}/{len(canonical_names)}")
            time.sleep(1)

        pairs_cache.write_text(json.dumps(pairs))
        anchors_cache.write_text(json.dumps(anchors))
        print(f"  Cached to {CACHE_DIR}/")

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------
    print("\nBuilding KGGraph…")
    graph, ground_truth, ids_with_topology = _build_graph(pairs, anchors)
    anchor_nodes = sum(1 for n in graph.nodes if n.id.startswith("anc_"))
    topo_pairs = sum(1 for c, _ in ground_truth if c in ids_with_topology)
    print(f"  {len(graph.nodes)} nodes  ({len(pairs) * 2} entity, {anchor_nodes} anchor)")
    print(f"  {len(graph.edges)} edges")
    print(f"  {len(ground_truth)} ground-truth merge pairs")
    print(f"  {topo_pairs}/{len(pairs)} pairs have entity-specific topology signal")

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------
    from nodecanon import Resolver
    from nodecanon.core.matching import RuleBasedMatcher
    from nodecanon.core.scoring import NodeScorer

    if args.fast:
        print("\nResolving (fast — string + topology, no embeddings)…")
        weights = {
            "name_similarity": 0.43,
            "semantic_similarity": 0.00,
            "type_agreement": 0.29,
            "neighbor_overlap": 0.29,
            "description_similarity": 0.00,
        }
        scorer = NodeScorer(weights=weights, cache_dir=None)
        matcher = RuleBasedMatcher(threshold=0.72, weights=weights)
        resolver = Resolver(scorer=scorer, matcher=matcher)
    else:
        print("\nResolving (full — sentence-transformers)…")
        resolver = Resolver()

    t0 = time.perf_counter()
    result = resolver.resolve(graph)
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    metrics = _evaluate(result, ground_truth, ids_with_topology)
    topo_m: dict[str, float] = metrics["topo"]  # type: ignore[assignment]

    w = 62
    print(f"\n{'─' * w}")
    print("  DBpedia Real-World Benchmark  (nodecanon v0.1.0)")
    print(f"{'─' * w}")
    print(f"  Source       : DBpedia wikiPageRedirects (live SPARQL)")
    print(f"  Entity pairs : {len(ground_truth)}  (companies + persons)")
    print(f"  Graph size   : {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"  Mode         : {'fast (no embeddings)' if args.fast else 'full (sentence-transformers)'}")
    print(f"  Elapsed      : {elapsed:.1f}s")
    print()
    print(f"  {'Metric':<20} {'Overall':>10}  {'w/ topology':>12}")
    print(f"  {'─'*20} {'─'*10}  {'─'*12}")
    print(f"  {'Precision':<20} {metrics['precision']:>10.3f}  {topo_m['precision']:>12.3f}")
    print(f"  {'Recall':<20} {metrics['recall']:>10.3f}  {topo_m['recall']:>12.3f}")
    print(f"  {'F1':<20} {metrics['f1']:>10.3f}  {topo_m['f1']:>12.3f}")
    print()
    print(f"  TP={int(metrics['tp'])}  FP={int(metrics['fp'])}  FN={int(metrics['fn'])}")
    print(f"{'─' * w}")

    # Show a sample of what was missed
    if metrics["fn"] > 0:
        print("\n  Sample missed pairs (false negatives):")
        alias_to_canonical: dict[str, str] = {}
        for record in result.merge_records:
            for aid in record.merged_ids:
                alias_to_canonical[aid] = record.canonical_id

        shown = 0
        for (canon_id, alias_id), (canon_name, alias_name) in zip(ground_truth, pairs):
            merged = (
                alias_to_canonical.get(alias_id) == canon_id
                or alias_to_canonical.get(canon_id) == alias_id
            )
            if not merged and shown < 10:
                sim = fuzz.token_sort_ratio(canon_name, alias_name)
                has_topo = canon_id in ids_with_topology
                tag = "[topo]" if has_topo else "[name]"
                print(f"    {tag}  {canon_name!r}  ↔  {alias_name!r}  (sim={sim})")
                shown += 1
        if int(metrics["fn"]) > 10:
            print(f"    … and {int(metrics['fn']) - 10} more")


if __name__ == "__main__":
    main()

# nodecanon

Your knowledge graph extracted 847 entities. You should have 312.

The other 535 are the same real-world things written differently — "IBM", "I.B.M.", "International Business Machines", "IBM Corp" — because the LLM that extracted them had no memory of what it called the same company three chunks ago.

nodecanon fixes that.

```bash
pip install nodecanon
```

```python
from nodecanon import Resolver, GraphBuilder

graph = (
    GraphBuilder()
    .add_node("IBM", type="ORGANIZATION")
    .add_node("I.B.M.", type="ORGANIZATION")
    .add_node("International Business Machines", type="ORGANIZATION")
    .add_node("Watson AI", type="PRODUCT")
    .add_edge("IBM", "Watson AI", "MAKES")
    .add_edge("I.B.M.", "Watson AI", "MAKES")
    .add_edge("International Business Machines", "Watson AI", "MAKES")
    .build()
)

result = Resolver().resolve(graph)
print(result.merge_report())
```

```
→ Merged 4 nodes into 2 canonical nodes
→ Absorbed 2 alias nodes
→ Removed 2 redundant edges
→ Flagged 0 conflicts for human review
```

No LLM calls. No API keys. Runs locally in under two minutes on 10,000 nodes.

---

## Why this exists

Multi-hop reasoning over a knowledge graph only works if the graph is actually connected. When "IBM" and "I.B.M." are two separate nodes with no edge between them, your RAG pipeline cannot traverse that gap. It thinks they're strangers. Every query that crosses this invisible seam comes back wrong or incomplete.

This is not a GraphRAG bug. It is a fundamental consequence of how LLMs process text in chunks — each chunk names entities independently, with no awareness of how the same entity was named 3,000 tokens ago.

nodecanon is the post-processing step that reconnects the graph.

---

## How it works

Four layers, run in sequence:

### 1. Block — O(n), not O(n²)

At 10,000 nodes, all-pairs scoring is 50 million comparisons. Blocking cuts this to ~1–5% of pairs by generating only *plausible* candidates.

Three strategies, combined via union:

- **TokenOverlapBlocker** — pairs nodes that share at least one non-stopword token. Catches "IBM Corp" / "IBM Inc". Misses abbreviations.
- **NGramFingerprintBlocker** — pairs nodes with overlapping character trigrams after normalization. "IBM" and "I.B.M." both normalize to `ibm`, sharing trigram `ibm`. Catches abbreviation variants.
- **AbbreviationBlocker** — pairs a short token with a longer name when one looks like an abbreviation of the other, via three tests: initialism (`ML` → `Machine Learning`), consonant contraction (`NVDA` → `NVIDIA`), subsequence (`MSFT` → `Microsoft`).
- **TypeCompatibilityBlocker** — post-filter that removes type-incompatible pairs (`PERSON` + `ORGANIZATION` never reach scoring).

### 2. Score — five-component ScoreVector

For each candidate pair, a `ScoreVector` is computed rather than a single number. A vector preserves *why* two nodes are similar, which drives both the merge decision and the audit trail.

```python
ScoreVector(
    name_similarity        = 0.94,   # rapidfuzz WRatio + Jaro-Winkler on metaphone forms
    semantic_similarity    = 0.91,   # cosine similarity of all-MiniLM-L6-v2 embeddings
    type_agreement         = 1.00,   # 1.0 if compatible, 0.5 if unknown, 0.0 if incompatible
    neighbor_overlap       = 0.87,   # soft Jaccard of 1-hop neighbor sets via embeddings
    description_similarity = 0.83,   # cosine similarity of description embeddings
)
```

The `neighbor_overlap` component is what separates nodecanon from all classical ER tools. If "IBM" and "I.B.M." both connect to "Watson", "Ginni Rometty", and "Armonk NY" — even if their name similarity is moderate — their structural position in the graph is identical. They are the same entity.

### 3. Match — weighted threshold

The weighted sum is compared against a configurable threshold (default 0.75):

```
score = 0.30 × name + 0.25 × semantic + 0.20 × type + 0.20 × neighbor + 0.05 × description
```

Pairs scoring above the threshold merge. Pairs in an optional *ambiguous zone* (default 0.65–0.80) can route to an LLM for a binary yes/no call — off by default, only affects ~5–10% of candidates.

### 4. Merge — union-find, full provenance

Union-find ensures transitivity: if A matches B and B matches C, all three merge into one canonical node without re-scoring.

The most-connected node becomes canonical. Every merge is logged:

```python
node._merged_from    = ["ibm_001", "ibm_047", "ibm_203"]
node._merge_evidence = {"name_similarity": 0.94, "semantic_similarity": 0.91, ...}
node._merge_strategy = "rule_based"
node._resolved_types = ["ORGANIZATION", "COMPANY"]
```

Nothing is silently dropped.

---

## Installation

```bash
pip install nodecanon
```

For Microsoft GraphRAG integration (adds pandas + pyarrow):
```bash
pip install nodecanon[graphrag]
```

For LLM-assisted matching on ambiguous pairs:
```bash
pip install nodecanon[llm]   # installs openai + anthropic
```

All adapters:
```bash
pip install nodecanon[graphrag,llamaindex,lightrag,neo4j,llm]
```

---

## Building a graph

### From plain dicts

```python
from nodecanon import KGGraph

graph = KGGraph.from_dicts(
    nodes=[
        {"name": "IBM",                          "type": "ORGANIZATION"},
        {"name": "I.B.M.",                       "type": "ORGANIZATION"},
        {"name": "International Business Machines", "type": "ORGANIZATION"},
        {"name": "Watson AI",                    "type": "PRODUCT"},
    ],
    edges=[
        {"source": "IBM",    "target": "Watson AI", "relation": "MAKES"},
        {"source": "I.B.M.", "target": "Watson AI", "relation": "MAKES"},
    ],
)
```

- `id` is optional — auto-generated from the name when omitted (`"IBM Corp"` → id `"ibm_corp"`)
- Any extra fields land in `node.attributes` (`{"founded": 1911}` → `node.attributes["founded"]`)
- Edge keys accept `source` / `source_id` and `target` / `target_id` interchangeably

### Fluent builder

```python
from nodecanon import GraphBuilder

graph = (
    GraphBuilder()
    .add_node("IBM",   type="ORGANIZATION", founded=1911)
    .add_node("I.B.M.", type="ORGANIZATION")
    .add_node("Watson AI", type="PRODUCT")
    .add_edge("IBM",    "Watson AI", "MAKES")
    .add_edge("I.B.M.", "Watson AI", "MAKES")
    .build()
)
```

- `add_node` is idempotent — calling it twice with the same name is a no-op
- `add_edge` accepts node names or node ids; referenced nodes that don't exist yet are auto-created
- Keyword args on `add_node` go into `attributes`

### Directly (verbose, full control)

```python
from nodecanon import KGGraph, KGNode, KGEdge

graph = KGGraph(
    nodes=[
        KGNode(id="n1", name="IBM",    type="ORGANIZATION"),
        KGNode(id="n2", name="I.B.M.", type="ORGANIZATION"),
    ],
    edges=[
        KGEdge(source_id="n1", target_id="n2", relation="SAME_AS"),
    ],
)
```

---

## Resolving

```python
from nodecanon import Resolver

result = Resolver().resolve(graph)
```

### Configuration

```python
from nodecanon import Resolver
from nodecanon.core.scoring import NodeScorer
from nodecanon.core.matching import RuleBasedMatcher, LLMAssistedMatcher

# Custom score weights (must sum to 1.0 for interpretable thresholds)
scorer = NodeScorer(
    weights={
        "name_similarity":        0.35,
        "semantic_similarity":    0.30,
        "type_agreement":         0.20,
        "neighbor_overlap":       0.10,
        "description_similarity": 0.05,
    }
)

# Stricter threshold for high-precision use cases
matcher = RuleBasedMatcher(threshold=0.85)

# LLM-assisted mode: only calls the LLM for pairs in the ambiguous zone
llm_matcher = LLMAssistedMatcher(
    rule_matcher=RuleBasedMatcher(threshold=0.75),
    ambiguous_low=0.65,
    ambiguous_high=0.80,
    provider="anthropic",          # or "openai"
    model="claude-haiku-4-5-20251001",
)

resolver = Resolver(scorer=scorer, matcher=matcher)
result = resolver.resolve(graph)
```

### Disable the embedding model (fast mode)

For graphs where topology signal is strong enough, you can skip the sentence-transformer entirely:

```python
from nodecanon import Resolver
from nodecanon.core.scoring import NodeScorer
from nodecanon.core.matching import RuleBasedMatcher

fast_weights = {
    "name_similarity":        0.43,
    "semantic_similarity":    0.00,
    "type_agreement":         0.29,
    "neighbor_overlap":       0.29,
    "description_similarity": 0.00,
}
resolver = Resolver(
    scorer=NodeScorer(weights=fast_weights, cache_dir=None),
    matcher=RuleBasedMatcher(threshold=0.72, weights=fast_weights),
)
```

---

## Reading results

### Summary

```python
print(result.merge_report())
# → Merged 847 nodes into 312 canonical nodes
# → Absorbed 535 alias nodes
# → Removed 1,203 redundant edges
# → Flagged 14 conflicts for human review
```

### Iterate canonical nodes

```python
for node in result.graph.nodes:
    if node._merged_from:
        print(f"{node.name!r} absorbed: {node._merged_from}")
```

### Explain a specific merge

```python
print(result.explain("ibm_canonical_id"))
```

```
Canonical node: 'IBM' (id: n1)

Merged from 3 nodes:
  · "IBM" (id: n1)
  · "I.B.M." (id: n2)
  · "IBM Corporation" (id: n3)

Merge evidence:
  name_similarity:        0.890  (weight 0.3)
  semantic_similarity:    0.940  (weight 0.25)
  type_agreement:         1.000  (weight 0.2)
  neighbor_overlap:       1.000  (weight 0.2)
  description_similarity: 0.000  (weight 0.05)
  ────────────────────────────────────────
  weighted score:         0.921

Merge strategy: rule_based
```

### Review conflicts

```python
for i, conflict in enumerate(result.conflicts):
    print(f"[{i}] {conflict.node_id_a} vs {conflict.node_id_b}")
    print(f"     Reason: {conflict.conflict_reason}")
    print(f"     Score:  {conflict.score.weighted_sum():.3f}")
```

---

## Editing results after resolution

All editing methods return a **new** `ResolveResult` — the original is never mutated. You can chain corrections and compare paths.

### Reject a merge you disagree with

```python
# The resolver merged "Python" (language) with "Python" (snake) — undo it
corrected = result.reject_merge("python_canonical_id")

# Restore only specific aliases, not all of them
corrected = result.reject_merge("python_canonical_id", restore=["python_snake_id"])
```

After rejecting, the canonical node reverts to its pre-merge form and the specified aliases are re-added as independent nodes. Edges remain on the canonical — they cannot be split back automatically.

### Force a merge the resolver missed

```python
# The resolver didn't merge "Alphabet Inc" and "Google" — do it manually
corrected = result.force_merge("alphabet_id", "google_id")

# Three-way force merge
corrected = result.force_merge("id_a", "id_b", "id_c")
```

### Accept a flagged conflict

Conflicts are type-incompatible pairs the resolver flagged rather than merging. If you've reviewed one and want to merge it anyway:

```python
# See all conflicts
for i, c in enumerate(result.conflicts):
    print(f"[{i}] {c.node_id_a} + {c.node_id_b} — {c.conflict_reason}")

# Accept conflict at index 0
corrected = result.accept_conflict(0)
```

### Chain corrections

```python
final = (
    result
    .reject_merge("wrong_merge_id")
    .force_merge("alphabet_id", "google_id")
    .accept_conflict(0)
)
```

---

## Adapters

### Microsoft GraphRAG

```bash
pip install nodecanon[graphrag]
```

```python
from nodecanon.adapters.graphrag import GraphRAGAdapter

graph = GraphRAGAdapter.from_directory("./graphrag_output/")
result = Resolver().resolve(graph)
```

Reads `entities.parquet` and `relationships.parquet`. Supports both v1 and v2 GraphRAG output layouts.

### LlamaIndex PropertyGraphIndex

```bash
pip install nodecanon[llamaindex]
```

```python
from nodecanon.adapters.llamaindex import LlamaIndexAdapter

adapter = LlamaIndexAdapter()
graph = adapter.load(my_property_graph_index)

result = Resolver().resolve(graph)

# Write back to the index
adapter.save(result.graph, my_property_graph_index)
```

### LightRAG

```bash
pip install nodecanon[lightrag]
```

```python
from nodecanon.adapters.lightrag import LightRAGAdapter

graph = LightRAGAdapter.from_working_dir("./lightrag_data/")
result = Resolver().resolve(graph)
LightRAGAdapter.save(result.graph, "./lightrag_data/")
```

Reads `graph_chunk_entity_relation.graphml` from the LightRAG working directory.

### NetworkX

```python
from nodecanon.adapters.networkx import NetworkXAdapter
import networkx as nx

# Load from any NetworkX DiGraph
G = nx.read_graphml("my_graph.graphml")
graph = NetworkXAdapter.from_networkx(G)

result = Resolver().resolve(graph)

# Export back to NetworkX
G_resolved = NetworkXAdapter.to_networkx(result.graph)
```

### Neo4j (export only)

```bash
pip install nodecanon[neo4j]
```

```python
from pathlib import Path
from nodecanon.adapters.neo4j import Neo4jAdapter

Neo4jAdapter().dump(result.graph, Path("resolved.cypher"))
```

Generates idempotent `MERGE` statements. Load with:

```bash
cypher-shell -u neo4j -p password < resolved.cypher
```

---

## CLI

```bash
# Resolve a GraphRAG output directory
nodecanon resolve ./graphrag_output/ --output ./resolved/

# Inspect the resolved graph
nodecanon inspect ./resolved/

# Explain a specific merge decision
nodecanon explain <node_id> ./resolved/
```

---

## Benchmark

### Real-world: DBpedia entity aliases

Ground truth from DBpedia `wikiPageRedirects` — when Wikipedia redirects "I.B.M." → "IBM", that redirect is an entity alias.
**287 company and person pairs**, filtered to genuine name variants (similarity ≥ 50%).
Graph built from real DBpedia properties (founders, parent companies, employer relations) as topology anchors.

| Condition | Pairs | Precision | Recall | F1 |
|-----------|-------|-----------|--------|-----|
| **With topology** (shared DBpedia anchors) | 71 | **1.000** | **0.986** | **0.993** |
| Name-only, fast mode | 216 | 0.771 | 0.282 | 0.413 |
| Name-only, full mode | 216 | 0.930 | 0.230 | 0.369 |

The topology result is the headline: when your GraphRAG output has shared neighbors between duplicate nodes — which is the typical case when the same entity is mentioned across multiple text chunks — nodecanon achieves near-perfect precision and recall with no API calls.

The name-only rows cover structurally hard cases: subsidiary names ("Egmont Imagination" vs "Egmont Group"), different-language translations ("Royal Dutch" vs "Royal Netherlands"), and abbreviated forms without shared graph context. These are candidates for `LLMAssistedMatcher`.

```bash
python benchmarks/dbpedia_benchmark.py --fast     # downloads ~287 pairs from DBpedia, fast mode
python benchmarks/dbpedia_benchmark.py            # full mode with sentence-transformers
python benchmarks/dbpedia_benchmark.py --offline  # reuse cached data
```

### Synthetic benchmark

64 nodes (12 canonical entity clusters × realistic name variants), 93 edges.
Covers: IBM / IBM Corp (easy), Samuel Altman / S. Altman (medium), LLM / large language model (hard), NVDA / NVIDIA (abbreviation).

| Mode | Precision | Recall | F1 | Time |
|------|-----------|--------|-----|------|
| Fast (no embeddings) | **1.000** | **0.949** | **0.974** | < 0.1s |
| Full (sentence-transformers) | 1.000 | 0.949+ | 0.974+ | ~5s |

**Real-world alias test** (28 entity clusters, curated organization / person / concept aliases, topology-equipped):

| Precision | Recall | F1 |
|-----------|--------|-----|
| **0.990** | **0.783** | **0.874** |

```bash
python benchmarks/run_benchmark.py --fast          # instant, no download
python benchmarks/run_benchmark.py                 # full, downloads model once
python benchmarks/battle_test.py --aliases --no-wikidata   # curated real-world aliases
python benchmarks/battle_test.py --fb15k --sample 2000     # 14k-node scale test
```

---

## Data model reference

### KGNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier within the graph |
| `name` | `str` | Surface form of the entity name |
| `type` | `str \| None` | Entity type label (e.g. `"ORGANIZATION"`) |
| `description` | `str \| None` | Free-text description |
| `attributes` | `dict` | Any additional key-value metadata |
| `source_chunks` | `list[str]` | Source chunk IDs from the extraction pipeline |
| `_merged_from` | `list[str] \| None` | IDs of all nodes merged into this one (set on merge) |
| `_merge_evidence` | `dict \| None` | ScoreVector components that triggered the merge |
| `_merge_strategy` | `str \| None` | `"rule_based"`, `"llm_assisted"`, or `"manual"` |
| `_resolved_types` | `list[str] \| None` | All type labels from merged nodes (union) |

### KGEdge

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | ID of the source node |
| `target_id` | `str` | ID of the target node |
| `relation` | `str` | Relationship label |
| `weight` | `float` | Default 1.0; parallel edges sum their weights on merge |
| `attributes` | `dict` | Any additional metadata |

### ScoreVector

| Field | Type | Default weight |
|-------|------|----------------|
| `name_similarity` | `float` | 0.30 |
| `semantic_similarity` | `float` | 0.25 |
| `type_agreement` | `float` | 0.20 |
| `neighbor_overlap` | `float` | 0.20 |
| `description_similarity` | `float` | 0.05 |

Call `score.weighted_sum()` for the combined decision score. Pass a `weights` dict to override defaults.

### ResolveResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `KGGraph` | The resolved graph with canonical nodes |
| `merge_records` | `list[MergeRecord]` | One record per merged group |
| `conflicts` | `list[MergeConflict]` | Pairs flagged for human review |
| `original_node_count` | `int` | Node count before resolution |
| `original_edge_count` | `int` | Edge count before resolution |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `merge_report()` | `str` | Human-readable summary of what changed |
| `explain(node_id)` | `str` | Detailed breakdown of a merge decision |
| `reject_merge(canonical_id, restore=None)` | `ResolveResult` | Undo a merge |
| `force_merge(*node_ids)` | `ResolveResult` | Manually merge nodes |
| `accept_conflict(index)` | `ResolveResult` | Accept a flagged conflict and merge it |

---

## TypeCompatibilityBlocker — built-in type clusters

Unknown types (not in any cluster) default to compatible with everything — the scoring layer handles disambiguation. You can extend the map:

```python
from nodecanon.core.blocking import TypeCompatibilityBlocker, UnionBlocker
from nodecanon.core.blocking import TokenOverlapBlocker, NGramFingerprintBlocker, AbbreviationBlocker

custom_compat = {
    **TypeCompatibilityBlocker.DEFAULT_COMPATIBILITY,
    "DRUG": {"DRUG", "MEDICATION", "PHARMACEUTICAL", "COMPOUND"},
    "GENE": {"GENE", "PROTEIN", "BIOMARKER"},
}

resolver = Resolver(
    blocker=UnionBlocker([
        TokenOverlapBlocker(),
        NGramFingerprintBlocker(),
        AbbreviationBlocker(),
        TypeCompatibilityBlocker(compatibility_map=custom_compat),
    ])
)
```

Built-in clusters:

| Canonical | Compatible labels |
|-----------|-----------------|
| ORGANIZATION | COMPANY, CORP, CORPORATION, FIRM, INSTITUTION, STARTUP, AGENCY, ASSOCIATION, FOUNDATION, UNIVERSITY |
| PERSON | INDIVIDUAL, HUMAN, RESEARCHER, AUTHOR, SCIENTIST |
| LOCATION | PLACE, CITY, COUNTRY, REGION, GPE, AREA |
| PRODUCT | SOFTWARE, SERVICE, TOOL, SYSTEM, PLATFORM |
| EVENT | INCIDENT, OCCURRENCE |
| CONCEPT | IDEA, TOPIC, THEORY, METHOD, TECHNIQUE |

---

## Known limitations

**Acronym ↔ full name pairs** (e.g. `"IBM"` ↔ `"International Business Machines"`) require either strong graph topology overlap or the optional `LLMAssistedMatcher`. At the default threshold (0.75) with no shared neighbors, the weighted score peaks at ~0.72 — just below the merge threshold. If your graph has many such pairs, either lower the threshold, ensure edges are populated before resolving, or enable LLM-assisted matching for the ambiguous zone.

**Rebranding / informal names** (e.g. `"Google"` ↔ `"Alphabet"`, `"Britain"` ↔ `"United Kingdom"`) score low on name similarity and require semantic or topological evidence. These are the primary driver of missed recall in the real-world alias test.

**Short ambiguous acronyms** (`"WHO"`, `"UN"`, `"ML"`) can false-match unrelated entities if used in different domains in the same graph. The `TypeCompatibilityBlocker` and high type_agreement weight mitigate this, but verify results when your graph mixes domains.

**Very large graphs (>50k nodes)** may hit memory pressure on the embedding matrix. Use `cache_dir` to persist embeddings between runs, and `batch_size` in the resolver's scorer to control peak memory.

---

## What it does NOT do

- **Extract** knowledge graphs from text — that is GraphRAG's job
- **Require an API key** in default mode — sentence-transformers runs locally on CPU
- **Silently drop data** — every merge is logged with provenance; type conflicts surface as `MergeConflict`
- **Modify your original graph** — `resolve()` always returns a new graph
- **Require a GPU** — all-MiniLM-L6-v2 runs on CPU in ~50ms per sentence

---

## Performance targets

| Scale | Blocking | Scoring | Total |
|-------|----------|---------|-------|
| 1,000 nodes, 5,000 edges | < 0.5s | < 10s | < 15s |
| 10,000 nodes, 50,000 edges | < 5s | < 60s | < 2 min |

Memory: peak < 4 GB for 10,000 nodes on an 8 GB laptop.

---

## License

MIT

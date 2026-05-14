# nodecanon

[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/nodecanon/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-282%20passing-brightgreen)](tests/)
[![Typed](https://img.shields.io/badge/typed-py.typed-informational)](nodecanon/py.typed)

**Entity resolution and deduplication for LLM-extracted knowledge graphs.**

Your knowledge graph extracted 847 entities. You should have 312.

The other 535 are the same real-world things written differently: "IBM", "I.B.M.", "International Business Machines", "IBM Corp". The LLM that extracted them had no memory of what it called the same company three chunks ago.

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
Merged 4 nodes into 2 canonical nodes
Absorbed 2 alias nodes
Removed 2 redundant edges
Flagged 0 conflicts for human review
```

No LLM calls. No API keys. Runs locally in under two minutes on 10,000 nodes.

---

## Why this exists

Multi-hop reasoning over a knowledge graph only works if the graph is actually connected. When "IBM" and "I.B.M." are two separate nodes with no edge between them, your retrieval pipeline cannot traverse that gap. It treats them as strangers. Every query that crosses this invisible seam comes back wrong or incomplete.

This is not a bug in GraphRAG or LlamaIndex. It is a fundamental consequence of how LLMs process text in chunks: each chunk names entities independently, with no awareness of how the same entity was named 3,000 tokens earlier. The same problem has been independently reported across every major GraphRAG framework.

nodecanon is the post-processing step that reconnects the graph.

### What makes this problem specific

LLM-extracted knowledge graphs have three properties that make entity resolution harder than the standard case:

- **No fixed schema**: one node has a description, another has none; one has a type label, another has five different ones extracted across chunks
- **Graph-structured identity**: two nodes may be the same entity not because their attributes match, but because they connect to the same neighbors in the graph
- **Schema-free types**: "COMPANY", "ORGANIZATION", "FIRM", "CORP" all mean the same thing but look different to any string or embedding comparison

nodecanon is built specifically for this combination.

---

## How it works

Four layers run in sequence.

### 1. Block: O(n), not O(n²)

At 10,000 nodes, all-pairs scoring requires 50 million comparisons. Blocking cuts this to roughly 1-5% of pairs by generating only plausible candidates.

Four strategies combine via union:

- **TokenOverlapBlocker**: pairs nodes that share at least one non-stopword token. Catches "IBM Corp" / "IBM Inc". Misses pure abbreviations.
- **NGramFingerprintBlocker**: pairs nodes with overlapping character trigrams. "IBM" and "I.B.M." both normalize to `ibm`, sharing the trigram fingerprint. Catches abbreviation variants that token overlap misses.
- **AbbreviationBlocker**: pairs a short name with a longer name when the short one looks like an abbreviation. Three tests: initialism (`ML` from `Machine Learning`), consonant contraction (`NVDA` from `NVIDIA`), subsequence (`MSFT` from `Microsoft`).
- **TypeCompatibilityBlocker**: a filter, not a generator. Removes type-incompatible pairs from the union before scoring. `PERSON` + `ORGANIZATION` never reach the scorer.

### 2. Score: five-component ScoreVector

For each candidate pair, a `ScoreVector` is computed rather than a single number. The vector preserves *why* two nodes are similar, which drives both the merge decision and the audit trail.

```python
ScoreVector(
    name_similarity        = 0.94,   # rapidfuzz WRatio + Jaro-Winkler on metaphone forms
    semantic_similarity    = 0.91,   # cosine similarity of all-MiniLM-L6-v2 embeddings
    type_agreement         = 1.00,   # 1.0 if compatible, 0.5 if unknown, 0.0 if incompatible
    neighbor_overlap       = 0.87,   # soft Jaccard of 1-hop neighbor name sets
    description_similarity = 0.83,   # cosine similarity of description embeddings
)
```

The `neighbor_overlap` component is the key differentiator from classical ER. If "IBM" and "I.B.M." both connect to "Watson", "Ginni Rometty", and "Armonk NY", their structural position in the graph is identical even when their name similarity is moderate. Two nodes that occupy the same position in a graph are almost certainly the same entity.

When both nodes have zero neighbors, `neighbor_overlap` is 0.0, not 1.0. Absence of evidence is not evidence of match.

### 3. Match: weighted threshold

The weighted sum is compared against a configurable threshold (default 0.75):

```
score = 0.30 * name + 0.25 * semantic + 0.20 * type + 0.20 * neighbor + 0.05 * description
```

Pairs above the threshold merge. An optional ambiguous zone (default 0.65-0.80) can route uncertain pairs to an LLM for a binary yes/no call. Off by default, affects roughly 5-10% of candidates when enabled.

### 4. Merge: union-find, full provenance

Union-find ensures transitivity: if A matches B and B matches C, all three collapse into one canonical node without re-scoring.

The most-connected node becomes canonical. Every merge is logged on the resulting node:

```python
node._merged_from    = ["ibm_001", "ibm_047", "ibm_203"]
node._merge_evidence = {"name_similarity": 0.94, "neighbor_overlap": 0.87, ...}
node._merge_strategy = "rule_based"
node._resolved_types = ["ORGANIZATION", "COMPANY"]
```

Nothing is silently dropped.

---

## Installation

```bash
pip install nodecanon
```

For Microsoft GraphRAG integration (adds pandas and pyarrow):
```bash
pip install nodecanon[graphrag]
```

For LLM-assisted matching on ambiguous pairs:
```bash
pip install nodecanon[llm]   # installs openai + anthropic
```

For Neo4j full roundtrip (load from live instance, write back resolved):
```bash
pip install nodecanon[neo4j]
```

All adapters at once:
```bash
pip install nodecanon[graphrag,llamaindex,lightrag,neo4j,llm]
```

---

## Building a graph

### From plain dicts

The most common path when loading from a database or JSON file:

```python
from nodecanon import KGGraph

graph = KGGraph.from_dicts(
    nodes=[
        {"name": "IBM",                             "type": "ORGANIZATION"},
        {"name": "I.B.M.",                          "type": "ORGANIZATION"},
        {"name": "International Business Machines", "type": "ORGANIZATION"},
        {"name": "Watson AI",                       "type": "PRODUCT"},
    ],
    edges=[
        {"source": "IBM",    "target": "Watson AI", "relation": "MAKES"},
        {"source": "I.B.M.", "target": "Watson AI", "relation": "MAKES"},
    ],
)
```

- `id` is optional: auto-generated from the name when omitted (`"IBM Corp"` becomes id `"ibm_corp"`)
- Extra fields land in `node.attributes` (`{"founded": 1911}` becomes `node.attributes["founded"]`)
- Edge keys accept `source` / `source_id` and `target` / `target_id` interchangeably

### Fluent builder

```python
from nodecanon import GraphBuilder

graph = (
    GraphBuilder()
    .add_node("IBM",      type="ORGANIZATION", founded=1911)
    .add_node("I.B.M.",   type="ORGANIZATION")
    .add_node("Watson AI", type="PRODUCT")
    .add_edge("IBM",    "Watson AI", "MAKES")
    .add_edge("I.B.M.", "Watson AI", "MAKES")
    .build()
)
```

- `add_node` is idempotent: calling it twice with the same name is a no-op
- `add_edge` accepts node names or node ids; referenced nodes that do not exist yet are auto-created
- Keyword arguments on `add_node` go into `attributes`

### Direct construction

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

The first call downloads `all-MiniLM-L6-v2` (~90 MB) and caches it locally. Subsequent calls use the cached model.

### Persist embeddings across runs

On large graphs, re-embedding the same nodes on every run is wasteful. Pass `cache_dir` to reuse embeddings:

```python
from nodecanon import Resolver
from nodecanon.core.scoring import NodeScorer

resolver = Resolver(
    scorer=NodeScorer(cache_dir=".nodecanon/embeddings")
)
result = resolver.resolve(graph)
```

The cache is keyed by node content hash. If a node changes, its embedding is automatically recomputed.

### Custom weights and threshold

```python
from nodecanon import Resolver
from nodecanon.core.scoring import NodeScorer
from nodecanon.core.matching import RuleBasedMatcher

scorer = NodeScorer(
    weights={
        "name_similarity":        0.35,
        "semantic_similarity":    0.30,
        "type_agreement":         0.20,
        "neighbor_overlap":       0.10,
        "description_similarity": 0.05,
    }
)

# Stricter threshold for high-precision requirements
matcher = RuleBasedMatcher(threshold=0.85)

resolver = Resolver(scorer=scorer, matcher=matcher)
result = resolver.resolve(graph)
```

### LLM-assisted matching for ambiguous pairs

```python
from nodecanon.core.matching import LLMAssistedMatcher, RuleBasedMatcher

llm_matcher = LLMAssistedMatcher(
    rule_matcher=RuleBasedMatcher(threshold=0.75),
    ambiguous_low=0.65,
    ambiguous_high=0.80,
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
)

resolver = Resolver(matcher=llm_matcher)
result = resolver.resolve(graph)
```

The LLM is called only for pairs that fall in the ambiguous zone. Clear matches and clear non-matches are decided locally.

### Fast mode: no embeddings

For graphs where topology signal is strong and speed matters:

```python
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

Fast mode runs in under 0.1 seconds on 64 nodes. F1 on the synthetic benchmark: 0.974.

---

## Reading results

### Summary report

```python
print(result.merge_report())
# Merged 847 nodes into 312 canonical nodes
# Absorbed 535 alias nodes
# Removed 1,203 redundant edges
# Flagged 14 conflicts for human review
```

### Iterate canonical nodes

```python
for node in result.graph.nodes:
    if node._merged_from:
        print(f"{node.name!r} absorbed: {node._merged_from}")
```

### Explain a specific merge decision

```python
print(result.explain("ibm_canonical_id"))
```

```
Canonical node: 'IBM' (id: n1)

Merged from 3 nodes:
  . "IBM" (id: n1)
  . "I.B.M." (id: n2)
  . "IBM Corporation" (id: n3)

Merge evidence:
  name_similarity:        0.890  (weight 0.3)
  semantic_similarity:    0.940  (weight 0.25)
  type_agreement:         1.000  (weight 0.2)
  neighbor_overlap:       1.000  (weight 0.2)
  description_similarity: 0.000  (weight 0.05)
  weighted score:         0.921

Merge strategy: rule_based
```

### Review conflicts

Type-incompatible pairs are flagged as `MergeConflict` rather than silently merged:

```python
for i, conflict in enumerate(result.conflicts):
    print(f"[{i}] {conflict.node_id_a} vs {conflict.node_id_b}")
    print(f"     Reason: {conflict.conflict_reason}")
    print(f"     Score:  {conflict.score.weighted_sum():.3f}")
```

---

## Editing results after resolution

All editing methods return a new `ResolveResult`. The original is never mutated. Corrections can be chained.

### Reject a merge

```python
# The resolver merged "Python" (language) with "Python" (snake) -- undo it
corrected = result.reject_merge("python_canonical_id")

# Restore only specific aliases, not all of them
corrected = result.reject_merge("python_canonical_id", restore=["python_snake_id"])
```

After rejecting, the canonical node reverts to its pre-merge form and the restored aliases are re-added as independent nodes. Edges stay on the canonical and cannot be automatically split back.

### Force a merge

```python
# The resolver did not merge "Alphabet Inc" and "Google" -- do it manually
corrected = result.force_merge("alphabet_id", "google_id")

# Three-way force merge
corrected = result.force_merge("id_a", "id_b", "id_c")
```

### Accept a flagged conflict

```python
# See all conflicts
for i, c in enumerate(result.conflicts):
    print(f"[{i}] {c.node_id_a} + {c.node_id_b}: {c.conflict_reason}")

# Accept conflict at index 0 and merge the pair
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
from nodecanon import Resolver

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
from nodecanon import Resolver

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
from nodecanon import Resolver

graph = LightRAGAdapter.from_working_dir("./lightrag_data/")
result = Resolver().resolve(graph)
LightRAGAdapter.save(result.graph, "./lightrag_data/")
```

Reads `graph_chunk_entity_relation.graphml` from the LightRAG working directory.

### nano-graphrag

nano-graphrag stores its entity-relation graph in the same GraphML format as LightRAG. No extra install is needed beyond networkx (already a core dependency).

```python
from nodecanon.adapters.nanographrag import NanoGraphRAGAdapter
from nodecanon import Resolver

# From a working directory (after nano-graphrag has finished indexing)
graph = NanoGraphRAGAdapter.from_working_dir("./nano_output/")
result = Resolver().resolve(graph)
NanoGraphRAGAdapter.save(result.graph, "./nano_output/")

# From a live GraphRAG instance (in-memory, no disk I/O)
graph = NanoGraphRAGAdapter.from_instance(rag)
result = Resolver().resolve(graph)
```

### NetworkX

```python
from nodecanon.adapters.networkx import NetworkXAdapter
from nodecanon import Resolver
import networkx as nx

G = nx.read_graphml("my_graph.graphml")
graph = NetworkXAdapter.from_networkx(G)

result = Resolver().resolve(graph)

G_resolved = NetworkXAdapter.to_networkx(result.graph)
```

### Neo4j (full roundtrip)

```bash
pip install nodecanon[neo4j]
```

Load from a live Neo4j instance, resolve, and write back. The write-back is non-destructive: canonical nodes are updated in place, alias nodes gain `_is_alias: true` and an `IS_ALIAS_OF` relationship. Nothing is deleted.

```python
from neo4j import GraphDatabase
from nodecanon.adapters.neo4j import Neo4jAdapter
from nodecanon import Resolver

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# Load
graph = Neo4jAdapter.from_neo4j(driver, node_label="Entity")

# Resolve
result = Resolver().resolve(graph)

# Write back
stats = Neo4jAdapter.to_neo4j(driver, result)
print(stats)
# {"nodes_upserted": 312, "aliases_annotated": 535, "edges_merged": 1203}

driver.close()
```

Export to a Cypher file instead (no live connection required):

```python
from pathlib import Path
from nodecanon.adapters.neo4j import Neo4jAdapter

Neo4jAdapter().dump(result.graph, Path("resolved.cypher"))
```

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

Ground truth from DBpedia `wikiPageRedirects`. When Wikipedia redirects "I.B.M." to "IBM", that redirect is an entity alias. We download 287 company and person pairs filtered to genuine name variants (similarity >= 50%), build a graph from real DBpedia properties (founders, parent companies, employer relations) as topology anchors, and measure against that ground truth.

| Condition | Pairs | Precision | Recall | F1 |
|-----------|-------|-----------|--------|-----|
| With topology (shared DBpedia anchors) | 71 | **1.000** | **0.986** | **0.993** |
| Name-only, fast mode | 216 | 0.771 | 0.282 | 0.413 |
| Name-only, full mode | 216 | 0.930 | 0.230 | 0.369 |

When your GraphRAG output has shared neighbors between duplicate nodes (the typical case when the same entity is mentioned across multiple text chunks), nodecanon achieves near-perfect precision and recall with no API calls.

The name-only rows cover structurally hard cases: subsidiary names ("Egmont Imagination" vs "Egmont Group"), different-language translations ("Royal Dutch" vs "Royal Netherlands"), and short forms without shared graph context. These are candidates for `LLMAssistedMatcher`.

```bash
python benchmarks/dbpedia_benchmark.py --fast     # downloads from DBpedia, fast mode
python benchmarks/dbpedia_benchmark.py            # full mode with sentence-transformers
python benchmarks/dbpedia_benchmark.py --offline  # reuse cached data
```

### Synthetic benchmark

64 nodes across 12 canonical entity clusters with realistic name variants, 93 edges. Covers easy (IBM / IBM Corp), medium (Samuel Altman / S. Altman), hard (LLM / large language model), and abbreviation cases (NVDA / NVIDIA).

| Mode | Precision | Recall | F1 | Time |
|------|-----------|--------|-----|------|
| Fast (no embeddings) | **1.000** | **0.949** | **0.974** | < 0.1s |
| Full (sentence-transformers) | 1.000 | 0.949+ | 0.974+ | ~5s |

Curated real-world alias test (28 entity clusters, actual organization / person / concept aliases, topology-equipped):

| Precision | Recall | F1 |
|-----------|--------|-----|
| **0.990** | **0.783** | **0.874** |

```bash
python benchmarks/run_benchmark.py --fast
python benchmarks/run_benchmark.py
python benchmarks/battle_test.py --aliases --no-wikidata
python benchmarks/battle_test.py --fb15k --sample 2000
```

---

## FAQ

**Does nodecanon work if my graph has no edges?**

Yes. Name similarity and semantic similarity still fire. You will not get the topology signal (`neighbor_overlap` stays at 0.0), so pairs with similar names but no shared context are harder to merge confidently. Populate edges before resolving when possible.

**Why did it miss an obvious duplicate?**

Three common reasons. First, the pair may not have been blocked: check `AbbreviationBlocker` for acronym-to-full-name pairs without shared tokens. Second, the score may be below threshold: run `result.explain(node_id)` to see the component breakdown and decide whether to lower the threshold or use `force_merge`. Third, the types may be incompatible: the `TypeCompatibilityBlocker` removes them before scoring.

**What happens to edges when nodes merge?**

All edges from alias nodes redirect to the canonical node. If merging creates parallel edges (same source, target, and relation), they are deduplicated and their weights are summed.

**How do I run nodecanon on the same graph multiple times without re-embedding?**

Pass `cache_dir` to `NodeScorer`. Embeddings are cached by content hash and reused automatically on subsequent runs.

**Can I use a different embedding model?**

Yes. Subclass `NodeScorer` and override `_embed`. The default is `all-MiniLM-L6-v2` from sentence-transformers because it runs on CPU, downloads once, and is fast enough for production-scale graphs.

**Does it run offline?**

After the first run (which downloads the embedding model), yes. The model is cached by sentence-transformers in `~/.cache/torch/sentence_transformers/`. Set `cache_dir` on `NodeScorer` to also persist embeddings across graph runs.

**What is the recommended threshold for high-precision production use?**

0.85 with the default weights. This virtually eliminates false merges at the cost of lower recall on borderline pairs. Use `LLMAssistedMatcher` with `ambiguous_low=0.75, ambiguous_high=0.85` to recover ambiguous pairs via LLM at low cost.

---

## Data model reference

### KGNode

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier within the graph |
| `name` | `str` | Surface form of the entity name |
| `type` | `str or None` | Entity type label (e.g. `"ORGANIZATION"`) |
| `description` | `str or None` | Free-text description |
| `attributes` | `dict` | Any additional key-value metadata |
| `source_chunks` | `list[str]` | Source chunk IDs from the extraction pipeline |
| `_merged_from` | `list[str] or None` | IDs of all nodes merged into this one (set on merge) |
| `_merge_evidence` | `dict or None` | ScoreVector components that triggered the merge |
| `_merge_strategy` | `str or None` | `"rule_based"`, `"llm_assisted"`, or `"manual"` |
| `_resolved_types` | `list[str] or None` | All type labels from merged nodes (union) |

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

## TypeCompatibilityBlocker: built-in type clusters

Unknown types (not in any cluster) default to compatible with everything. The scoring layer handles disambiguation. You can extend the compatibility map:

```python
from nodecanon.core.blocking import TypeCompatibilityBlocker, UnionBlocker
from nodecanon.core.blocking import TokenOverlapBlocker, NGramFingerprintBlocker, AbbreviationBlocker

custom_compat = {
    **TypeCompatibilityBlocker.DEFAULT_COMPATIBILITY,
    "DRUG":     {"DRUG", "MEDICATION", "PHARMACEUTICAL", "COMPOUND"},
    "GENE":     {"GENE", "PROTEIN", "BIOMARKER"},
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

**Acronym to full name pairs** (e.g. `"IBM"` vs `"International Business Machines"`) require either strong graph topology overlap or `LLMAssistedMatcher`. At the default threshold with no shared neighbors, the weighted score peaks at roughly 0.72, just below the 0.75 merge threshold. If your graph has many such pairs, lower the threshold, ensure edges are populated before resolving, or enable LLM-assisted matching for the ambiguous zone.

**Rebranding and informal names** (e.g. `"Google"` vs `"Alphabet"`, `"Britain"` vs `"United Kingdom"`) score low on name similarity and need semantic or topological evidence. These are the primary driver of missed recall in the real-world alias benchmark.

**Short ambiguous acronyms** (`"WHO"`, `"UN"`, `"ML"`) can false-match unrelated entities if different domains share the same graph. The `TypeCompatibilityBlocker` and high type_agreement weight mitigate this, but verify results when your graph spans multiple domains.

**Very large graphs (>50k nodes)** may hit memory pressure on the embedding matrix. Use `cache_dir` to persist embeddings between runs, and `batch_size` on the scorer to control peak memory.

---

## What it does not do

- **Extract** knowledge graphs from text: that is GraphRAG's job
- **Require an API key** in default mode: sentence-transformers runs locally on CPU
- **Silently drop data**: every merge is logged with provenance; type conflicts surface as `MergeConflict`
- **Modify your original graph**: `resolve()` always returns a new graph
- **Require a GPU**: all-MiniLM-L6-v2 runs on CPU in roughly 50ms per sentence

---

## Performance targets

| Scale | Blocking | Scoring | Total |
|-------|----------|---------|-------|
| 1,000 nodes, 5,000 edges | < 0.5s | < 10s | < 15s |
| 10,000 nodes, 50,000 edges | < 5s | < 60s | < 2 min |

Memory: peak < 4 GB for 10,000 nodes on an 8 GB laptop.

---

## Contributing

Bug reports, feature requests, and pull requests are welcome at [github.com/rasinmuhammed/node-canon](https://github.com/rasinmuhammed/node-canon).

When filing a bug, include the output of `result.explain(node_id)` for any merge that behaved unexpectedly. The score breakdown makes root causes much easier to identify.

---

## License

MIT

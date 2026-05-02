# nodecanon

Your knowledge graph extracted 847 entities. You should have 312.

The other 535 are the same real-world things written differently — "IBM", "I.B.M.", "International Business Machines", "IBM Corp" — because the LLM that extracted them had no memory of what it called the same company three chunks ago.

nodecanon fixes that.

```
pip install nodecanon
```

```python
from nodecanon import Resolver
from nodecanon.adapters.graphrag import GraphRAGAdapter

graph = GraphRAGAdapter.from_directory("./graphrag_output/")
result = Resolver().resolve(graph)

print(result.merge_report())
```

```
→ Merged 847 nodes into 312 canonical nodes
→ Absorbed 535 alias nodes
→ Removed 1,203 redundant edges
→ Flagged 14 conflicts for human review
```

No LLM calls. No API keys. Runs locally in under two minutes on 10k nodes.

---

## Why this matters

Multi-hop reasoning over a knowledge graph only works if the graph is actually connected. When "IBM" and "I.B.M." are two separate nodes with no edge between them, your RAG pipeline cannot traverse that gap. It thinks they're strangers. Every query that crosses this invisible seam comes back wrong.

This is not a GraphRAG bug. It's a fundamental consequence of how LLMs process text in chunks — each chunk names entities independently, with no awareness of how the same entity was named 3,000 tokens ago.

nodecanon is the post-processing step that reconnects the graph.

---

## How it works

Four layers, run in sequence:

**1. Block** — Don't compare every pair. At 10k nodes that's 50 million comparisons. Instead, generate *candidate pairs* using overlapping tokens and character n-gram fingerprints. This catches both `"IBM Corp"/"IBM"` (token overlap) and `"IBM"/"I.B.M."` (n-gram fingerprint). O(n), not O(n²).

**2. Score** — For each candidate pair, compute a five-component `ScoreVector`:

```python
ScoreVector(
    name_similarity        = 0.94,   # rapidfuzz + phonetic
    semantic_similarity    = 0.91,   # all-MiniLM-L6-v2 cosine
    type_agreement         = 1.00,   # ORGANIZATION == ORGANIZATION
    neighbor_overlap       = 0.87,   # Jaccard of 1-hop neighbor names
    description_similarity = 0.83,   # embedding cosine on descriptions
)
```

The `neighbor_overlap` is the part that doesn't exist anywhere else. If "IBM" and "I.B.M." both connect to "Watson", "Ginni Rometty", and "Armonk NY" — even if their name similarity is moderate — their structural position in the graph is identical. They're the same node.

**3. Match** — Weighted sum against a configurable threshold (default 0.75). Pairs in the ambiguous zone can optionally route to an LLM for a binary yes/no call, but that's off by default.

**4. Merge** — Union-find for transitive groups (A matches B, B matches C → all three merge). The most-connected node becomes canonical. All aliases are logged with full provenance:

```python
node._merged_from      = ["ibm_001", "ibm_047", "ibm_203", "ibm_891"]
node._merge_evidence   = ScoreVector(0.94, 0.91, 1.0, 0.87, 0.83)
node._merge_strategy   = "rule_based"
node._resolved_types   = ["ORGANIZATION", "COMPANY"]
```

Every decision is auditable. Nothing is silently dropped.

---

## CLI

```bash
# Resolve a GraphRAG output directory
nodecanon resolve ./graphrag_output/ --output ./resolved/

# Inspect what changed
nodecanon inspect ./resolved/

# Understand a specific merge decision
nodecanon explain e1_canonical ./resolved/
```

```
Canonical node: IBM  (id: e1_canonical)
  type: ORGANIZATION

Merged from 4 nodes:
  [canonical] IBM (id: e1_canonical)
  [alias]     I.B.M. (id: chunk_047_ibm)
  [alias]     IBM Corporation (id: chunk_203_ibm)
  [alias]     International Business Machines (id: chunk_891_ibm)

Merge evidence:
  name_similarity:        0.94
  semantic_similarity:    0.91
  type_agreement:         1.00
  neighbor_overlap:       0.87
  description_similarity: 0.83
  ────────────────────────────────────
  weighted score:         0.91  (strategy: rule_based)
```

---

## Adapters

| Framework | Direction | Notes |
|-----------|-----------|-------|
| Microsoft GraphRAG | Load | Reads `entities.parquet` + `relationships.parquet`, v1 and v2 layouts |
| LlamaIndex PropertyGraphIndex | Load + Save | Works with any graph store that implements `get_triplets()` |
| LightRAG | Load + Save | Reads `graph_chunk_entity_relation.graphml` from working dir |
| NetworkX DiGraph | Load + Save | Universal interop layer |
| Neo4j | Save | Exports idempotent Cypher `MERGE` statements |

```python
# GraphRAG
from nodecanon.adapters.graphrag import GraphRAGAdapter
graph = GraphRAGAdapter.from_directory("./output/")

# LightRAG
from nodecanon.adapters.lightrag import LightRAGAdapter
graph = LightRAGAdapter.from_working_dir("./lightrag_data/")

# LlamaIndex
from nodecanon.adapters.llamaindex import LlamaIndexAdapter
graph = LlamaIndexAdapter().load(my_property_graph_index)

# Export to Neo4j
from nodecanon.adapters.neo4j import Neo4jAdapter
Neo4jAdapter().dump(result.graph, Path("resolved.cypher"))
```

---

## Benchmark

Synthetic dataset: 64 nodes (12 canonical entities + realistic variants + anchor context), 93 edges.
Variants include easy cases (IBM / I.B.M.), medium (Samuel Altman / S. Altman), and hard (LLM / large language model, NVDA / NVIDIA).

| Metric | Value |
|--------|-------|
| Precision | **1.000** — zero wrong merges at default threshold |
| Recall | 0.615 — misses hard abbreviations (NVDA, A.I., ML vs "machine learning") |
| F1 | 0.762 |
| Redundancy removed | 28% |
| Time (64 nodes, CPU) | 10s |

Recall improves significantly with LLM-assisted mode (`--llm`) for ambiguous pairs. Precision stays near 1.0 by design — a wrong merge is worse than a missed one.

Run it yourself:

```bash
python benchmarks/run_benchmark.py
```

---

## What it doesn't do

- **Extract** knowledge graphs from text — that's GraphRAG's job
- **Require an API key** in default mode — sentence-transformers runs locally
- **Silently drop data** — every merge is logged; unresolvable conflicts surface as `MergeConflict` for human review
- **Modify your original graph** — always returns a new resolved graph

---

## Installation

```bash
pip install nodecanon
```

Optional extras:

```bash
pip install nodecanon[llamaindex]   # LlamaIndex adapter
pip install nodecanon[llm]          # LLM-assisted matching (OpenAI or Anthropic)
pip install nodecanon[neo4j]        # Neo4j Cypher export
```

---

## License

MIT

# Changelog

## [0.1.0] — 2026-05-14

### Added
- Core entity resolution pipeline: blocking → scoring → matching → merging → resolver
- `TokenOverlapBlocker`, `NGramFingerprintBlocker`, `TypeCompatibilityBlocker`, `AbbreviationBlocker`, `UnionBlocker`
- `NodeScorer` with `ScoreVector` (name similarity, semantic similarity, type agreement, neighbor overlap, description similarity)
- `RuleBasedMatcher` and `LLMAssistedMatcher` (optional, OpenAI / Anthropic)
- `NodeMerger`, `EdgeMerger`, `ConflictDetector` with full provenance tracking
- `Resolver` orchestrator with embedding cache (`~/.nodecanon/embeddings.npz`)
- `ResolveResult` with `merge_report()`, `explain()`, `reject_merge()`, `force_merge()`, `accept_conflict()`
- `GraphBuilder` fluent API: `.add_node() / .add_edge() / .build()`
- `KGGraph.from_dicts()` for loading from JSON / database query results
- Adapters: `GraphRAGAdapter`, `LlamaIndexAdapter`, `LightRAGAdapter`, `NetworkXAdapter`, `Neo4jAdapter` (full roundtrip)
- CLI: `nodecanon resolve / inspect / explain`
- Evaluation module: `evaluate()`, `MergeReport`
- 282 tests, zero lint warnings

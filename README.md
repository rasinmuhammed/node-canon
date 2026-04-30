# nodecanon

**Entity resolution for LLM-extracted knowledge graphs.**

> nodecanon cleans the entity soup that LLMs leave behind in your knowledge graph.

```python
pip install nodecanon
```

```python
from nodecanon import Resolver
from nodecanon.adapters import GraphRAGAdapter

graph = GraphRAGAdapter().load("./graphrag_output/")
result = Resolver().resolve(graph)
print(result.merge_report())
# → Merged 847 nodes into 312 canonical nodes
```

## Status

Under active development. Core pipeline in progress.

## License

MIT

"""Tests for LlamaIndex adapter.

LlamaIndex is an optional dependency, so we stub its types with lightweight
dataclasses and patch the import guards — no actual llama-index install needed.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from nodecanon.adapters.llamaindex import (
    LlamaIndexAdapter,
    _entity_node_to_kgnode,
    _get_triplets,
    _kgnode_extra_props,
)
from nodecanon.core.models import KGNode

# ---------------------------------------------------------------------------
# Minimal stubs that mimic LlamaIndex's public API surface
# ---------------------------------------------------------------------------


@dataclass
class _EntityNode:
    name: str
    label: str = "ENTITY"
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.label}:{self.name}"


@dataclass
class _Relation:
    label: str
    source_id: str
    target_id: str
    properties: dict[str, Any] = field(default_factory=dict)


class _SimpleStore:
    """Minimal graph store stub with get_triplets()."""

    def __init__(self, triplets: list[tuple]) -> None:
        self._triplets = triplets

    def get_triplets(self) -> list[tuple]:
        return self._triplets


class _RelMapStore:
    """Minimal graph store stub with get_rel_map() (no get_triplets)."""

    def __init__(self, rel_map: dict) -> None:
        self._rel_map = rel_map

    def get_rel_map(self) -> dict:
        return self._rel_map


# ---------------------------------------------------------------------------
# Helpers to install fake llama_index into sys.modules
# ---------------------------------------------------------------------------


def _install_fake_llama_index() -> None:
    """Create a minimal llama_index module tree in sys.modules."""
    if "llama_index" in sys.modules:
        return

    li = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_gs = types.ModuleType("llama_index.core.graph_stores")
    li_gs_types = types.ModuleType("llama_index.core.graph_stores.types")
    li_gs_simple = types.ModuleType("llama_index.core.graph_stores.simple_labelled")
    li_idx = types.ModuleType("llama_index.core.indices")
    li_pg = types.ModuleType("llama_index.core.indices.property_graph")

    li_gs_types.EntityNode = _EntityNode  # type: ignore[attr-defined]
    li_gs_types.Relation = _Relation  # type: ignore[attr-defined]

    class _FakeSimpleStore(_SimpleStore):
        def upsert_nodes(self, _nodes: list) -> None:
            self._triplets = []

        def upsert_relations(self, rels: list) -> None:
            pass

    li_gs_simple.SimplePropertyGraphStore = _FakeSimpleStore  # type: ignore[attr-defined]

    class _FakePGI:
        def __init__(self, store: Any) -> None:
            self.property_graph_store = store

        @classmethod
        def from_existing(cls, property_graph_store: Any, **_kw: Any) -> _FakePGI:
            return cls(property_graph_store)

    li_pg.PropertyGraphIndex = _FakePGI  # type: ignore[attr-defined]

    for name, mod in [
        ("llama_index", li),
        ("llama_index.core", li_core),
        ("llama_index.core.graph_stores", li_gs),
        ("llama_index.core.graph_stores.types", li_gs_types),
        ("llama_index.core.graph_stores.simple_labelled", li_gs_simple),
        ("llama_index.core.indices", li_idx),
        ("llama_index.core.indices.property_graph", li_pg),
    ]:
        sys.modules[name] = mod


def _remove_fake_llama_index() -> None:
    prefixes = [
        "llama_index",
        "llama_index.core",
        "llama_index.core.graph_stores",
        "llama_index.core.graph_stores.types",
        "llama_index.core.graph_stores.simple_labelled",
        "llama_index.core.indices",
        "llama_index.core.indices.property_graph",
    ]
    for key in prefixes:
        sys.modules.pop(key, None)


@pytest.fixture(autouse=True)
def fake_llama_index():
    _install_fake_llama_index()
    yield
    _remove_fake_llama_index()


# ---------------------------------------------------------------------------
# Tests: _get_triplets helper
# ---------------------------------------------------------------------------


class TestGetTriplets:
    def test_uses_get_triplets_method(self) -> None:
        ibm = _EntityNode("IBM", "ORGANIZATION")
        watson = _EntityNode("Watson", "PRODUCT")
        rel = _Relation("PRODUCT", ibm.id, watson.id)
        store = _SimpleStore([(ibm, rel, watson)])
        result = _get_triplets(store)
        assert len(result) == 1
        assert result[0][0].name == "IBM"

    def test_falls_back_to_get_rel_map(self) -> None:
        ibm = _EntityNode("IBM", "ORGANIZATION")
        watson = _EntityNode("Watson", "PRODUCT")
        rel = _Relation("PRODUCT", ibm.id, watson.id)
        # get_rel_map uses string IDs as keys in LlamaIndex
        store = _RelMapStore({ibm.id: [(rel, watson)]})
        result = _get_triplets(store)
        assert len(result) == 1

    def test_store_without_either_method_raises(self) -> None:
        with pytest.raises(AttributeError, match="get_triplets"):

            class _BadStore:
                pass

            _get_triplets(_BadStore())


# ---------------------------------------------------------------------------
# Tests: _entity_node_to_kgnode helper
# ---------------------------------------------------------------------------


class TestEntityNodeToKGNode:
    def test_basic_fields(self) -> None:
        li_node = _EntityNode("IBM", "ORGANIZATION")
        node = _entity_node_to_kgnode(li_node)
        assert node.name == "IBM"
        assert node.type == "ORGANIZATION"

    def test_description_extracted_from_properties(self) -> None:
        li_node = _EntityNode("IBM", properties={"description": "A tech company."})
        node = _entity_node_to_kgnode(li_node)
        assert node.description == "A tech company."
        assert "description" not in node.attributes

    def test_extra_properties_go_to_attributes(self) -> None:
        li_node = _EntityNode("IBM", properties={"founded": 1911})
        node = _entity_node_to_kgnode(li_node)
        assert node.attributes.get("founded") == 1911


# ---------------------------------------------------------------------------
# Tests: from_graph_store
# ---------------------------------------------------------------------------


class TestFromGraphStore:
    def test_node_count(self) -> None:
        ibm = _EntityNode("IBM", "ORGANIZATION")
        ginni = _EntityNode("Ginni Rometty", "PERSON")
        rel = _Relation("CEO_OF", ibm.id, ginni.id)
        store = _SimpleStore([(ibm, rel, ginni)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert len(graph.nodes) == 2

    def test_edge_count(self) -> None:
        ibm = _EntityNode("IBM", "ORGANIZATION")
        ginni = _EntityNode("Ginni Rometty", "PERSON")
        rel = _Relation("CEO_OF", ibm.id, ginni.id)
        store = _SimpleStore([(ibm, rel, ginni)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert len(graph.edges) == 1

    def test_relation_label_preserved(self) -> None:
        a = _EntityNode("A")
        b = _EntityNode("B")
        rel = _Relation("OWNS", a.id, b.id)
        store = _SimpleStore([(a, rel, b)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert graph.edges[0].relation == "OWNS"

    def test_edge_weight_from_properties(self) -> None:
        a = _EntityNode("A")
        b = _EntityNode("B")
        rel = _Relation("KNOWS", a.id, b.id, properties={"weight": 3.5})
        store = _SimpleStore([(a, rel, b)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert graph.edges[0].weight == pytest.approx(3.5)

    def test_duplicate_nodes_deduplicated(self) -> None:
        ibm = _EntityNode("IBM", "ORGANIZATION")
        watson = _EntityNode("Watson", "PRODUCT")
        rel1 = _Relation("PRODUCT", ibm.id, watson.id)
        rel2 = _Relation("ALSO_PRODUCT", ibm.id, watson.id)
        store = _SimpleStore([(ibm, rel1, watson), (ibm, rel2, watson)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert len(graph.nodes) == 2  # IBM + Watson, not 4

    def test_string_relation_label_fallback(self) -> None:
        a = _EntityNode("A")
        b = _EntityNode("B")
        store = _SimpleStore([(a, "MENTIONS", b)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert graph.edges[0].relation == "MENTIONS"

    def test_none_relation_defaults_to_related_to(self) -> None:
        a = _EntityNode("A")
        b = _EntityNode("B")
        store = _SimpleStore([(a, None, b)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert graph.edges[0].relation == "RELATED_TO"

    def test_node_type_preserved(self) -> None:
        node = _EntityNode("IBM", "ORGANIZATION")
        store = _SimpleStore([(node, _Relation("R", node.id, node.id), node)])
        graph = LlamaIndexAdapter.from_graph_store(store)
        assert graph.node_index()[node.id].type == "ORGANIZATION"


# ---------------------------------------------------------------------------
# Tests: load() dispatch
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_with_store_returns_kggraph(self) -> None:
        a = _EntityNode("A")
        b = _EntityNode("B")
        rel = _Relation("R", a.id, b.id)
        store = _SimpleStore([(a, rel, b)])
        graph = LlamaIndexAdapter().load(store)
        assert len(graph.nodes) == 2

    def test_load_with_property_graph_index(self) -> None:
        from llama_index.core.indices.property_graph import PropertyGraphIndex

        a = _EntityNode("A")
        b = _EntityNode("B")
        rel = _Relation("R", a.id, b.id)
        store = _SimpleStore([(a, rel, b)])
        index = PropertyGraphIndex(store)
        graph = LlamaIndexAdapter().load(index)
        assert len(graph.nodes) == 2


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_raises_if_llamaindex_not_installed(self) -> None:
        _remove_fake_llama_index()
        try:
            with pytest.raises(ImportError, match="llama-index-core"):
                LlamaIndexAdapter.from_graph_store(_SimpleStore([]))
        finally:
            _install_fake_llama_index()


# ---------------------------------------------------------------------------
# Tests: _kgnode_extra_props helper
# ---------------------------------------------------------------------------


class TestKGNodeExtraProps:
    def test_description_included(self) -> None:
        node = KGNode(id="a", name="A", description="Desc")
        props = _kgnode_extra_props(node)
        assert props["description"] == "Desc"

    def test_provenance_fields_included(self) -> None:
        node = KGNode(
            id="a",
            name="A",
            _merged_from=["x", "y"],
            _merge_strategy="rule_based",
        )
        props = _kgnode_extra_props(node)
        assert props["_merged_from"] == ["x", "y"]
        assert props["_merge_strategy"] == "rule_based"

    def test_empty_node_returns_empty_dict(self) -> None:
        node = KGNode(id="a", name="A")
        props = _kgnode_extra_props(node)
        assert props == {}

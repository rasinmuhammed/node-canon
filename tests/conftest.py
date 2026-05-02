"""Shared fixtures for all tests."""

from __future__ import annotations

import pytest

from nodecanon.core.models import KGEdge, KGGraph, KGNode


@pytest.fixture
def simple_graph() -> KGGraph:
    """Minimal graph with two near-duplicate nodes and one edge."""
    nodes = [
        KGNode(id="n1", name="IBM", type="ORGANIZATION"),
        KGNode(id="n2", name="I.B.M.", type="COMPANY"),
        KGNode(id="n3", name="Ginni Rometty", type="PERSON"),
    ]
    edges = [
        KGEdge(source_id="n1", target_id="n3", relation="CEO_OF"),
        KGEdge(source_id="n2", target_id="n3", relation="CEO_OF"),
    ]
    return KGGraph(nodes=nodes, edges=edges)


@pytest.fixture
def no_duplicate_graph() -> KGGraph:
    """Graph with no duplicate nodes."""
    nodes = [
        KGNode(id="a1", name="Apple Inc", type="ORGANIZATION"),
        KGNode(id="a2", name="Google LLC", type="ORGANIZATION"),
        KGNode(id="a3", name="Sundar Pichai", type="PERSON"),
    ]
    edges = [
        KGEdge(source_id="a2", target_id="a3", relation="CEO_OF"),
    ]
    return KGGraph(nodes=nodes, edges=edges)


@pytest.fixture
def conflict_graph() -> KGGraph:
    """Graph where a PERSON and ORGANIZATION share the same name — should conflict."""
    nodes = [
        KGNode(id="c1", name="Apple", type="PERSON"),
        KGNode(id="c2", name="Apple", type="ORGANIZATION"),
    ]
    return KGGraph(nodes=nodes, edges=[])


@pytest.fixture
def large_graph() -> KGGraph:
    """Graph with many nodes for blocker/performance tests."""
    nodes = [
        KGNode(id=f"n{i}", name=f"Entity {i}", type="ORGANIZATION") for i in range(200)
    ]
    edges = [
        KGEdge(source_id=f"n{i}", target_id=f"n{i + 1}", relation="RELATED_TO")
        for i in range(199)
    ]
    return KGGraph(nodes=nodes, edges=edges)

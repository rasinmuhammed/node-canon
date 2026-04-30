from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from nodecanon.core.blocking import TypeCompatibilityBlocker
from nodecanon.core.models import KGGraph, KGNode, ScoreVector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Strip everything except letters and spaces before phonetic comparison.
_ALPHA_RE = re.compile(r"[^a-zA-Z ]")


class NodeScorer:
    """Computes a ScoreVector for a candidate pair of KGNodes.

    Call fit(graph) before scoring to batch-encode embeddings and cache the
    adjacency index. Without fit(), score() works lazily but is much slower.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.weights = weights
        self._encoder: SentenceTransformer | None = None
        self._name_emb_cache: dict[str, np.ndarray] = {}
        self._desc_emb_cache: dict[str, np.ndarray] = {}
        self._adjacency_index: dict[str, list[str]] | None = None
        self._node_index: dict[str, KGNode] | None = None
        self._type_compat = TypeCompatibilityBlocker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, graph: KGGraph) -> None:
        """Batch-encode all node names and non-empty descriptions.

        Must be called before resolve() for acceptable performance on large
        graphs. Encodes in batches of 64 to minimise memory overhead.
        """
        encoder = self._get_encoder()
        nodes = graph.nodes

        # Encode all names in one batch.
        names = [n.name for n in nodes]
        name_embs: np.ndarray = encoder.encode(
            names,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        for i, node in enumerate(nodes):
            self._name_emb_cache[node.id] = name_embs[i]

        # Encode only non-empty descriptions (saves compute and memory).
        desc_nodes = [(i, n) for i, n in enumerate(nodes) if n.description]
        if desc_nodes:
            idxs, dnodes = zip(*desc_nodes)
            desc_embs: np.ndarray = encoder.encode(
                [n.description for n in dnodes],
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for i, node in zip(idxs, dnodes):
                self._desc_emb_cache[node.id] = desc_embs[i]

        self._adjacency_index = graph.adjacency_index()
        self._node_index = graph.node_index()

    def score(
        self,
        node_a: KGNode,
        node_b: KGNode,
        graph: KGGraph,
    ) -> ScoreVector:
        adjacency = (
            self._adjacency_index
            if self._adjacency_index is not None
            else graph.adjacency_index()
        )
        node_idx = (
            self._node_index
            if self._node_index is not None
            else graph.node_index()
        )
        return ScoreVector(
            name_similarity=self._name_similarity(node_a.name, node_b.name),
            semantic_similarity=self._semantic_similarity(node_a, node_b),
            type_agreement=self._type_agreement(node_a, node_b),
            neighbor_overlap=self._neighbor_overlap(node_a, node_b, adjacency, node_idx),
            description_similarity=self._description_similarity(node_a, node_b),
        )

    # ------------------------------------------------------------------
    # Score components
    # ------------------------------------------------------------------

    def _name_similarity(self, a: str, b: str) -> float:
        """Combines rapidfuzz WRatio with Jaro-Winkler on metaphone forms.

        max() rather than a weighted average: if either string similarity OR
        phonetic similarity is high, the names are plausibly the same entity.
        Phonetic catches "IBM" / "I.B.M." where string similarity alone is ~0.6.
        """
        from rapidfuzz import fuzz
        import jellyfish

        string_score = fuzz.WRatio(a, b) / 100.0

        # Strip punctuation before phonetic encoding so "I.B.M." → "IBM".
        a_clean = _ALPHA_RE.sub("", a).strip()
        b_clean = _ALPHA_RE.sub("", b).strip()
        try:
            m_a = jellyfish.metaphone(a_clean) if a_clean else ""
            m_b = jellyfish.metaphone(b_clean) if b_clean else ""
            phonetic_score = (
                jellyfish.jaro_winkler_similarity(m_a, m_b)
                if m_a and m_b
                else 0.0
            )
        except Exception:
            phonetic_score = 0.0

        return max(string_score, phonetic_score)

    def _semantic_similarity(self, a: KGNode, b: KGNode) -> float:
        """Cosine similarity of name embeddings, clipped to [0, 1]."""
        emb_a = self._name_emb_cache.get(a.id)
        emb_b = self._name_emb_cache.get(b.id)

        if emb_a is None or emb_b is None:
            encoder = self._get_encoder()
            if emb_a is None:
                emb_a = encoder.encode(a.name, normalize_embeddings=True)
                self._name_emb_cache[a.id] = emb_a
            if emb_b is None:
                emb_b = encoder.encode(b.name, normalize_embeddings=True)
                self._name_emb_cache[b.id] = emb_b

        return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    def _type_agreement(self, a: KGNode, b: KGNode) -> float:
        """1.0 if types are compatible, 0.0 if incompatible, 0.5 if unknown.

        0.5 when either type is None: absence of type information is neutral,
        not a positive or negative signal.
        """
        if a.type is None or b.type is None:
            return 0.5
        if a.type.upper() == b.type.upper():
            return 1.0
        return 1.0 if self._type_compat.are_compatible(a.type, b.type) else 0.0

    def _neighbor_overlap(
        self,
        node_a: KGNode,
        node_b: KGNode,
        adjacency: dict[str, list[str]],
        node_index: dict[str, KGNode],
    ) -> float:
        """Jaccard similarity of 1-hop neighbor name sets (string, not embedding).

        Uses names not IDs: two duplicate nodes share neighbor NAMES even
        though they were assigned different IDs during extraction.

        Excludes the two candidate nodes from each other's neighbor sets to
        avoid circular similarity inflation when they are directly connected.

        Returns 0.0 when both nodes are isolated — absence of evidence is not
        evidence of similarity.
        """
        nbr_a = {
            node_index[nid].name.lower()
            for nid in adjacency.get(node_a.id, [])
            if nid != node_b.id and nid in node_index
        }
        nbr_b = {
            node_index[nid].name.lower()
            for nid in adjacency.get(node_b.id, [])
            if nid != node_a.id and nid in node_index
        }

        if not nbr_a and not nbr_b:
            return 0.0

        union = nbr_a | nbr_b
        if not union:
            return 0.0

        return len(nbr_a & nbr_b) / len(union)

    def _description_similarity(self, a: KGNode, b: KGNode) -> float:
        """Cosine similarity of description embeddings, clipped to [0, 1].

        Returns 0.0 if either node has no description — missing description
        is the absence of a signal, not evidence of dissimilarity.
        """
        if not a.description or not b.description:
            return 0.0

        emb_a = self._desc_emb_cache.get(a.id)
        emb_b = self._desc_emb_cache.get(b.id)

        if emb_a is None or emb_b is None:
            encoder = self._get_encoder()
            if emb_a is None:
                emb_a = encoder.encode(a.description, normalize_embeddings=True)
                self._desc_emb_cache[a.id] = emb_a
            if emb_b is None:
                emb_b = encoder.encode(b.description, normalize_embeddings=True)
                self._desc_emb_cache[b.id] = emb_b

        return float(np.clip(np.dot(emb_a, emb_b), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nodecanon.core.blocking import TypeCompatibilityBlocker
from nodecanon.core.models import KGGraph, KGNode, ScoreVector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Strip everything except letters and spaces before phonetic comparison.
_ALPHA_RE = re.compile(r"[^a-zA-Z ]")

_DEFAULT_CACHE_DIR = Path(".nodecanon")
_CACHE_FILE = "embeddings.npz"


def _content_hash(node: KGNode, kind: str) -> str:
    """Stable cache key for a node embedding.

    Changing a node's name or description automatically invalidates its
    entry — the old key just goes stale and gets overwritten on the next fit.
    """
    raw = f"{node.id}|{node.name}|{node.description or ''}|{kind}"
    return hashlib.md5(raw.encode()).hexdigest()


class NodeScorer:
    """Computes a ScoreVector for a candidate pair of KGNodes.

    Call fit(graph) before scoring to batch-encode embeddings and cache the
    adjacency index. Without fit(), score() works lazily but is much slower.

    Embeddings are persisted to `<cache_dir>/embeddings.npz` so repeated
    calls on the same graph skip re-encoding entirely. Pass cache_dir=None
    to disable disk persistence.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        weights: dict[str, float] | None = None,
        cache_dir: Path | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        self.embedding_model = embedding_model
        self.weights = weights
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._encoder: SentenceTransformer | None = None
        self._name_emb_cache: dict[str, np.ndarray] = {}
        self._desc_emb_cache: dict[str, np.ndarray] = {}
        self._adjacency_index: dict[str, list[str]] | None = None
        self._node_index: dict[str, KGNode] | None = None
        self._type_compat = TypeCompatibilityBlocker()
        # Disk cache maps content_hash → embedding vector.  Loaded once on
        # first fit(), written back only when new vectors were computed.
        self._disk_cache: dict[str, np.ndarray] = {}
        self._disk_cache_dirty = False
        if self.cache_dir is not None:
            self._load_disk_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, graph: KGGraph) -> None:
        """Batch-encode all node names and non-empty descriptions.

        Must be called before resolve() for acceptable performance on large
        graphs. Encodes in batches of 64 to minimise memory overhead.

        Already-cached vectors (keyed by content hash) are loaded from disk
        instead of re-encoded.  Only genuinely new or changed nodes hit the
        embedding model.  Results are written back to the disk cache after
        encoding.
        """
        nodes = graph.nodes

        # ---- Names --------------------------------------------------------
        name_miss: list[tuple[KGNode, str]] = []
        for node in nodes:
            key = _content_hash(node, "name")
            if key in self._disk_cache:
                self._name_emb_cache[node.id] = self._disk_cache[key]
            else:
                name_miss.append((node, key))

        if name_miss:
            encoder = self._get_encoder()
            miss_nodes, miss_keys = zip(*name_miss, strict=True)
            name_embs: np.ndarray = encoder.encode(
                [n.name for n in miss_nodes],
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for node, key, emb in zip(miss_nodes, miss_keys, name_embs, strict=True):
                self._name_emb_cache[node.id] = emb
                if self.cache_dir is not None:
                    self._disk_cache[key] = emb
            if self.cache_dir is not None:
                self._disk_cache_dirty = True

        # ---- Descriptions -------------------------------------------------
        desc_miss: list[tuple[KGNode, str]] = []
        for node in nodes:
            if not node.description:
                continue
            key = _content_hash(node, "desc")
            if key in self._disk_cache:
                self._desc_emb_cache[node.id] = self._disk_cache[key]
            else:
                desc_miss.append((node, key))

        if desc_miss:
            encoder = self._get_encoder()
            miss_nodes_d, miss_keys_d = zip(*desc_miss, strict=True)
            desc_embs: np.ndarray = encoder.encode(
                [n.description for n in miss_nodes_d],
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for node, key, emb in zip(
                miss_nodes_d, miss_keys_d, desc_embs, strict=True
            ):
                self._desc_emb_cache[node.id] = emb
                if self.cache_dir is not None:
                    self._disk_cache[key] = emb
            if self.cache_dir is not None:
                self._disk_cache_dirty = True

        self._adjacency_index = graph.adjacency_index()
        self._node_index = graph.node_index()

        if self._disk_cache_dirty and self.cache_dir is not None:
            self._save_disk_cache()
            self._disk_cache_dirty = False

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
            self._node_index if self._node_index is not None else graph.node_index()
        )
        return ScoreVector(
            name_similarity=self._name_similarity(node_a.name, node_b.name),
            semantic_similarity=self._semantic_similarity(node_a, node_b),
            type_agreement=self._type_agreement(node_a, node_b),
            neighbor_overlap=self._neighbor_overlap(
                node_a, node_b, adjacency, node_idx
            ),
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
        import jellyfish
        from rapidfuzz import fuzz

        string_score = fuzz.WRatio(a, b) / 100.0

        # Strip punctuation before phonetic encoding so "I.B.M." → "IBM".
        a_clean = _ALPHA_RE.sub("", a).strip()
        b_clean = _ALPHA_RE.sub("", b).strip()
        try:
            m_a = jellyfish.metaphone(a_clean) if a_clean else ""
            m_b = jellyfish.metaphone(b_clean) if b_clean else ""
            phonetic_score = (
                jellyfish.jaro_winkler_similarity(m_a, m_b) if m_a and m_b else 0.0
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

    def _cache_path(self) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / _CACHE_FILE

    def _load_disk_cache(self) -> None:
        path = self._cache_path()
        if not path.exists():
            return
        try:
            data = np.load(path)
            self._disk_cache = {k: data[k] for k in data.files}
        except Exception:
            # Corrupt or incompatible cache — start fresh rather than crashing.
            self._disk_cache = {}

    def _save_disk_cache(self) -> None:
        path = self._cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self._disk_cache)

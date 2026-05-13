from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import combinations

from nodecanon.core.models import KGGraph, KGNode

# Common words that add no discriminative value as blocking keys.
# Includes corporate suffixes so "Apple Inc" / "Google Inc" don't pair on "inc".
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "inc",
        "corp",
        "corporation",
        "llc",
        "ltd",
        "co",
        "group",
        "holdings",
    }
)

_MIN_TOKEN_LEN: int = 2
_PUNCT_RE = re.compile(r"[^a-z0-9]")


def _normalize(name: str) -> str:
    """Lowercase and strip all non-alphanumeric characters."""
    return _PUNCT_RE.sub("", name.lower())


def _tokenize(name: str) -> set[str]:
    """Split on non-word chars, lowercase, drop stopwords and short tokens."""
    return {
        t
        for t in re.split(r"\W+", name.lower())
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    }


def _char_ngrams(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def _build_inverted_index(
    graph: KGGraph,
    key_fn: Callable[[KGNode], list[str]],
) -> dict[str, list[KGNode]]:
    index: dict[str, list[KGNode]] = {}
    for node in graph.nodes:
        for key in key_fn(node):
            index.setdefault(key, []).append(node)
    return index


def _pairs_from_index(
    index: dict[str, list[KGNode]],
) -> list[tuple[KGNode, KGNode]]:
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[KGNode, KGNode]] = []
    for nodes in index.values():
        if len(nodes) < 2:
            continue
        for a, b in combinations(nodes, 2):
            key = (min(a.id, b.id), max(a.id, b.id))
            if key not in seen:
                seen.add(key)
                pairs.append((a, b))
    return pairs


class BaseBlocker(ABC):
    @abstractmethod
    def candidate_pairs(self, graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        raise NotImplementedError


class TokenOverlapBlocker(BaseBlocker):
    """Pairs nodes sharing at least min_shared_tokens non-stopword tokens."""

    def __init__(self, min_shared_tokens: int = 1) -> None:
        self.min_shared_tokens = min_shared_tokens

    def candidate_pairs(self, graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        index = _build_inverted_index(graph, lambda n: list(_tokenize(n.name)))

        if self.min_shared_tokens == 1:
            return _pairs_from_index(index)

        # Count how many distinct tokens each pair shares.
        pair_counts: dict[tuple[str, str], int] = {}
        pair_nodes: dict[tuple[str, str], tuple[KGNode, KGNode]] = {}
        for nodes in index.values():
            if len(nodes) < 2:
                continue
            for a, b in combinations(nodes, 2):
                key = (min(a.id, b.id), max(a.id, b.id))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                pair_nodes.setdefault(key, (a, b))

        return [
            v for k, v in pair_nodes.items() if pair_counts[k] >= self.min_shared_tokens
        ]


class NGramFingerprintBlocker(BaseBlocker):
    """Pairs nodes with overlapping character n-gram fingerprints.

    Catches abbreviation variants (IBM / I.B.M.) that token overlap misses:
    both normalize to "ibm", sharing the trigram "ibm".

    top_k: maximum n-grams indexed per node (sorted alphabetically).
    Set to 0 for unlimited. Limiting caps inverted-index bucket sizes for
    common n-grams in large graphs.
    """

    def __init__(self, n: int = 3, top_k: int = 5) -> None:
        self.n = n
        self.top_k = top_k

    def _fingerprint(self, name: str) -> list[str]:
        normalized = _normalize(name)
        if not normalized:
            return []
        if len(normalized) < self.n:
            grams = [normalized]
        else:
            grams = sorted(set(_char_ngrams(normalized, self.n)))
        return grams[: self.top_k] if self.top_k > 0 else grams

    def candidate_pairs(self, graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        index = _build_inverted_index(graph, lambda n: self._fingerprint(n.name))
        return _pairs_from_index(index)


def _strip_non_alpha(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s).upper()


def _is_subsequence(short: str, long_s: str) -> bool:
    """True if every character of `short` appears in order in `long_s`."""
    pos = 0
    for c in short:
        idx = long_s.find(c, pos)
        if idx == -1:
            return False
        pos = idx + 1
    return True


class AbbreviationBlocker(BaseBlocker):
    """Pairs a short token with a longer name when one looks like an abbreviation
    of the other — covering the three patterns that token/ngram blockers miss:

    1. Initialism   — "ML"   → "Machine Learning"  (first letters of words)
    2. Consonant    — "NVDA" → "NVIDIA"             (vowels stripped)
    3. Subsequence  — "MSFT" → "Microsoft"          (chars appear in order)

    max_abbrev_len: maximum alpha-character count of the short side.  6 covers
    most real-world abbreviations without generating too many spurious pairs.
    """

    _VOWELS = re.compile(r"[AEIOU]")

    def __init__(self, max_abbrev_len: int = 6) -> None:
        self.max_abbrev_len = max_abbrev_len

    def candidate_pairs(self, graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        short: list[tuple[KGNode, str]] = []
        long_: list[tuple[KGNode, str, str, str]] = []  # node, alpha, initials, consonants

        for node in graph.nodes:
            alpha = _strip_non_alpha(node.name)
            if not alpha:
                continue
            if len(alpha) <= self.max_abbrev_len:
                short.append((node, alpha))
            words = [w for w in re.split(r"\W+", node.name) if w]
            initials = "".join(w[0].upper() for w in words if w)
            consonants = self._VOWELS.sub("", alpha)
            long_.append((node, alpha, initials, consonants))

        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[KGNode, KGNode]] = []

        for abbrev_node, abbrev_alpha in short:
            for full_node, full_alpha, initials, consonants in long_:
                if abbrev_node.id == full_node.id:
                    continue
                if full_alpha == abbrev_alpha:
                    continue  # same string — token/ngram blockers already cover this
                if len(full_alpha) <= len(abbrev_alpha):
                    continue  # full form must be strictly longer
                if (
                    abbrev_alpha == initials
                    or abbrev_alpha == consonants
                    or _is_subsequence(abbrev_alpha, full_alpha)
                ):
                    key = (
                        min(abbrev_node.id, full_node.id),
                        max(abbrev_node.id, full_node.id),
                    )
                    if key not in seen:
                        seen.add(key)
                        pairs.append((abbrev_node, full_node))

        return pairs


class TypeCompatibilityBlocker(BaseBlocker):
    """Post-filter: removes type-incompatible pairs from the candidate set.

    candidate_pairs() always returns [] — this blocker generates no pairs on
    its own. It is applied by UnionBlocker as a filter after taking the union
    of other blockers' output.

    Unknown types (not in any cluster) default to compatible with everything;
    the scoring layer handles the disambiguation.
    """

    DEFAULT_COMPATIBILITY: dict[str, set[str]] = {
        "ORGANIZATION": {
            "ORGANIZATION",
            "COMPANY",
            "CORP",
            "CORPORATION",
            "FIRM",
            "INSTITUTION",
            "STARTUP",
            "AGENCY",
            "ASSOCIATION",
            "FOUNDATION",
            "UNIVERSITY",
        },
        "PERSON": {
            "PERSON",
            "INDIVIDUAL",
            "HUMAN",
            "RESEARCHER",
            "AUTHOR",
            "SCIENTIST",
        },
        "LOCATION": {"LOCATION", "PLACE", "CITY", "COUNTRY", "REGION", "GPE", "AREA"},
        "PRODUCT": {"PRODUCT", "SOFTWARE", "SERVICE", "TOOL", "SYSTEM", "PLATFORM"},
        "EVENT": {"EVENT", "INCIDENT", "OCCURRENCE"},
        "CONCEPT": {"CONCEPT", "IDEA", "TOPIC", "THEORY", "METHOD", "TECHNIQUE"},
    }

    def __init__(
        self,
        compatibility_map: dict[str, set[str]] | None = None,
        allow_null_type: bool = True,
    ) -> None:
        self.compatibility_map = compatibility_map or self.DEFAULT_COMPATIBILITY
        self.allow_null_type = allow_null_type
        self._known: frozenset[str] = frozenset(
            t for cluster in self.compatibility_map.values() for t in cluster
        )

    def candidate_pairs(self, _graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        return []

    def are_compatible(self, type_a: str | None, type_b: str | None) -> bool:
        if type_a is None or type_b is None:
            return self.allow_null_type
        a, b = type_a.upper(), type_b.upper()
        if a == b:
            return True
        for cluster in self.compatibility_map.values():
            if a in cluster and b in cluster:
                return True
        # Unknown type → conservative: let scoring decide, not blocking.
        return bool(a not in self._known or b not in self._known)


class UnionBlocker(BaseBlocker):
    """Takes union of candidate pairs from all generator blockers.

    Any TypeCompatibilityBlocker in the list is used as a post-filter rather
    than a generator: after unioning all other blockers' pairs, incompatible-
    type pairs are removed.
    """

    def __init__(self, blockers: list[BaseBlocker]) -> None:
        self.blockers = blockers
        self._type_filter: TypeCompatibilityBlocker | None = next(
            (b for b in blockers if isinstance(b, TypeCompatibilityBlocker)), None
        )
        self._generators: list[BaseBlocker] = [
            b for b in blockers if not isinstance(b, TypeCompatibilityBlocker)
        ]

    def candidate_pairs(self, graph: KGGraph) -> list[tuple[KGNode, KGNode]]:
        seen: set[tuple[str, str]] = set()
        pairs: list[tuple[KGNode, KGNode]] = []
        for blocker in self._generators:
            for a, b in blocker.candidate_pairs(graph):
                key = (min(a.id, b.id), max(a.id, b.id))
                if key not in seen:
                    seen.add(key)
                    pairs.append((a, b))
        if self._type_filter is not None:
            pairs = [
                (a, b)
                for a, b in pairs
                if self._type_filter.are_compatible(a.type, b.type)
            ]
        return pairs

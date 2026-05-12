"""Tests for blocking layer."""

from __future__ import annotations

from nodecanon.core.blocking import (
    AbbreviationBlocker,
    NGramFingerprintBlocker,
    TokenOverlapBlocker,
    TypeCompatibilityBlocker,
    UnionBlocker,
)
from nodecanon.core.models import KGGraph, KGNode


def _pair_ids(pairs: list[tuple[KGNode, KGNode]]) -> set[tuple[str, str]]:
    return {(min(a.id, b.id), max(a.id, b.id)) for a, b in pairs}


class TestTokenOverlapBlocker:
    def test_shared_token_produces_candidate(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM Corporation"),
                KGNode(id="n2", name="IBM Inc"),
                KGNode(id="n3", name="Microsoft"),
            ],
            edges=[],
        )
        ids = _pair_ids(TokenOverlapBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids  # share "ibm"
        assert ("n1", "n3") not in ids
        assert ("n2", "n3") not in ids

    def test_no_shared_tokens_skips_pair(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="Apple"), KGNode(id="n2", name="Zebra")],
            edges=[],
        )
        assert TokenOverlapBlocker().candidate_pairs(graph) == []

    def test_single_node_no_pairs(self) -> None:
        graph = KGGraph(nodes=[KGNode(id="n1", name="IBM")], edges=[])
        assert TokenOverlapBlocker().candidate_pairs(graph) == []

    def test_empty_graph(self) -> None:
        assert TokenOverlapBlocker().candidate_pairs(KGGraph(nodes=[], edges=[])) == []

    def test_stopwords_not_matched(self) -> None:
        # "inc" is a stopword — these two should NOT be candidates on that token alone
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="Apple Inc"),
                KGNode(id="n2", name="Google Inc"),
            ],
            edges=[],
        )
        ids = _pair_ids(TokenOverlapBlocker().candidate_pairs(graph))
        assert ("n1", "n2") not in ids

    def test_min_shared_tokens_respected(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="New York City"),
                KGNode(id="n2", name="New York"),
                KGNode(id="n3", name="New Haven"),
            ],
            edges=[],
        )
        ids = _pair_ids(TokenOverlapBlocker(min_shared_tokens=2).candidate_pairs(graph))
        assert ("n1", "n2") in ids  # share "new" + "york" = 2
        assert ("n1", "n3") not in ids  # share only "new" = 1
        assert ("n2", "n3") not in ids  # share only "new" = 1

    def test_no_duplicate_pairs(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM Watson Cloud"),
                KGNode(id="n2", name="IBM Watson"),
            ],
            edges=[],
        )
        pairs = TokenOverlapBlocker().candidate_pairs(graph)
        ids = [(min(a.id, b.id), max(a.id, b.id)) for a, b in pairs]
        # n1 and n2 share "ibm" AND "watson" — must appear only once in output
        assert len(ids) == len(set(ids))
        assert len(ids) == 1

    def test_abbreviation_not_caught_by_token_overlap(self) -> None:
        # "I.B.M." tokenises to single-char tokens (all filtered), "IBM" → {"ibm"}
        # They should NOT be candidates from TokenOverlapBlocker alone
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="IBM"), KGNode(id="n2", name="I.B.M.")],
            edges=[],
        )
        ids = _pair_ids(TokenOverlapBlocker().candidate_pairs(graph))
        assert ("n1", "n2") not in ids


class TestNGramFingerprintBlocker:
    def test_abbreviation_variant_is_candidate(self) -> None:
        # "IBM" and "I.B.M." both normalize to "ibm" → share trigram "ibm"
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="IBM"), KGNode(id="n2", name="I.B.M.")],
            edges=[],
        )
        ids = _pair_ids(NGramFingerprintBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_completely_different_names_not_candidate(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="Apple"), KGNode(id="n2", name="Zebra")],
            edges=[],
        )
        assert NGramFingerprintBlocker().candidate_pairs(graph) == []

    def test_punctuation_stripped_before_fingerprint(self) -> None:
        # "U.S.A." → "usa", "USA" → "usa": same fingerprint
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="U.S.A."), KGNode(id="n2", name="USA")],
            edges=[],
        )
        ids = _pair_ids(NGramFingerprintBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_name_shorter_than_n_handled(self) -> None:
        # "AB" normalized is "ab" (len 2 < n=3) — falls back to the whole string
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="AB"), KGNode(id="n2", name="AB")],
            edges=[],
        )
        ids = _pair_ids(NGramFingerprintBlocker(n=3).candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_empty_name_does_not_crash(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name=""), KGNode(id="n2", name="IBM")],
            edges=[],
        )
        NGramFingerprintBlocker().candidate_pairs(graph)  # must not raise

    def test_no_duplicate_pairs(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="IBM"), KGNode(id="n2", name="I.B.M.")],
            edges=[],
        )
        pairs = NGramFingerprintBlocker().candidate_pairs(graph)
        ids = [(min(a.id, b.id), max(a.id, b.id)) for a, b in pairs]
        assert len(ids) == len(set(ids))

    def test_token_overlap_misses_what_ngram_catches(self) -> None:
        # Confirms the two blockers are complementary
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="IBM"), KGNode(id="n2", name="I.B.M.")],
            edges=[],
        )
        token_ids = _pair_ids(TokenOverlapBlocker().candidate_pairs(graph))
        ngram_ids = _pair_ids(NGramFingerprintBlocker().candidate_pairs(graph))
        assert ("n1", "n2") not in token_ids
        assert ("n1", "n2") in ngram_ids


class TestTypeCompatibilityBlocker:
    def test_compatible_types_pass(self) -> None:
        blocker = TypeCompatibilityBlocker()
        assert blocker.are_compatible("ORGANIZATION", "COMPANY")

    def test_incompatible_types_blocked(self) -> None:
        blocker = TypeCompatibilityBlocker()
        assert not blocker.are_compatible("PERSON", "ORGANIZATION")

    def test_null_type_allowed_by_default(self) -> None:
        blocker = TypeCompatibilityBlocker()
        assert blocker.are_compatible(None, "ORGANIZATION")

    def test_null_type_blocked_when_disallowed(self) -> None:
        blocker = TypeCompatibilityBlocker(allow_null_type=False)
        assert not blocker.are_compatible(None, "ORGANIZATION")

    def test_same_type_always_compatible(self) -> None:
        blocker = TypeCompatibilityBlocker()
        assert blocker.are_compatible("PERSON", "PERSON")
        assert blocker.are_compatible("LOCATION", "LOCATION")

    def test_unknown_type_defaults_to_compatible(self) -> None:
        blocker = TypeCompatibilityBlocker()
        assert blocker.are_compatible("DEITY", "ORGANIZATION")
        assert blocker.are_compatible("DEITY", "ANOTHER_UNKNOWN")

    def test_custom_compatibility_map(self) -> None:
        custom = {"ANIMAL": {"ANIMAL", "MAMMAL", "BIRD"}}
        blocker = TypeCompatibilityBlocker(compatibility_map=custom)
        assert blocker.are_compatible("ANIMAL", "MAMMAL")
        # Types not in the custom map are unknown → compatible
        assert blocker.are_compatible("ORGANIZATION", "COMPANY")

    def test_candidate_pairs_always_empty(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="IBM"), KGNode(id="n2", name="I.B.M.")],
            edges=[],
        )
        assert TypeCompatibilityBlocker().candidate_pairs(graph) == []


class TestUnionBlocker:
    def test_union_is_superset_of_each_child(self) -> None:
        # n1/n2 share token "ibm"; n1/n3, n2/n3, n1/n2 share ngram "ibm"
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM Corporation"),
                KGNode(id="n2", name="IBM Inc"),
                KGNode(id="n3", name="I.B.M."),
            ],
            edges=[],
        )
        token_blocker = TokenOverlapBlocker()
        ngram_blocker = NGramFingerprintBlocker()
        union = UnionBlocker([token_blocker, ngram_blocker])

        token_ids = _pair_ids(token_blocker.candidate_pairs(graph))
        ngram_ids = _pair_ids(ngram_blocker.candidate_pairs(graph))
        union_ids = _pair_ids(union.candidate_pairs(graph))

        assert token_ids.issubset(union_ids)
        assert ngram_ids.issubset(union_ids)

    def test_type_filter_removes_incompatible_pairs(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM", type="ORGANIZATION"),
                KGNode(id="n2", name="IBM Inc", type="COMPANY"),
                KGNode(id="n3", name="IBM CEO", type="PERSON"),
            ],
            edges=[],
        )
        union = UnionBlocker([TokenOverlapBlocker(), TypeCompatibilityBlocker()])
        ids = _pair_ids(union.candidate_pairs(graph))
        assert ("n1", "n2") in ids  # ORGANIZATION + COMPANY → compatible
        assert ("n1", "n3") not in ids  # ORGANIZATION + PERSON → incompatible
        assert ("n2", "n3") not in ids  # COMPANY + PERSON → incompatible

    def test_no_duplicates_across_blockers(self) -> None:
        # Two identical blockers — every pair would be generated twice without dedup
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM Corporation"),
                KGNode(id="n2", name="IBM Inc"),
            ],
            edges=[],
        )
        union = UnionBlocker([TokenOverlapBlocker(), TokenOverlapBlocker()])
        pairs = union.candidate_pairs(graph)
        ids = [(min(a.id, b.id), max(a.id, b.id)) for a, b in pairs]
        assert len(ids) == len(set(ids))

    def test_empty_graph(self) -> None:
        union = UnionBlocker([TokenOverlapBlocker(), NGramFingerprintBlocker()])
        assert union.candidate_pairs(KGGraph(nodes=[], edges=[])) == []

    def test_null_types_pass_filter_by_default(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="IBM"),  # type=None
                KGNode(id="n2", name="IBM Inc"),  # type=None
            ],
            edges=[],
        )
        union = UnionBlocker([TokenOverlapBlocker(), TypeCompatibilityBlocker()])
        ids = _pair_ids(union.candidate_pairs(graph))
        assert ("n1", "n2") in ids


class TestAbbreviationBlocker:
    def test_initialism_ml_machine_learning(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="ML"),
                KGNode(id="n2", name="Machine Learning"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_initialism_llm_large_language_models(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="LLM"),
                KGNode(id="n2", name="Large Language Models"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_initialism_ai_artificial_intelligence(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="AI"),
                KGNode(id="n2", name="Artificial Intelligence"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_consonant_contraction_nvda_nvidia(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="NVDA"),
                KGNode(id="n2", name="NVIDIA"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_subsequence_msft_microsoft(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="MSFT"),
                KGNode(id="n2", name="Microsoft"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") in ids

    def test_no_self_pairs(self) -> None:
        graph = KGGraph(
            nodes=[KGNode(id="n1", name="ML")],
            edges=[],
        )
        assert AbbreviationBlocker().candidate_pairs(graph) == []

    def test_unrelated_short_names_not_paired(self) -> None:
        # "CAT" is not an initialism, consonant form, or subsequence of "ZEBRA"
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="CAT"),
                KGNode(id="n2", name="ZEBRA"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker().candidate_pairs(graph))
        assert ("n1", "n2") not in ids

    def test_max_abbrev_len_respected(self) -> None:
        # "TOOLONG" (7 chars) should not be treated as abbreviation side
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="TOOLONG"),
                KGNode(id="n2", name="Tool Long Name Here"),
            ],
            edges=[],
        )
        ids = _pair_ids(AbbreviationBlocker(max_abbrev_len=6).candidate_pairs(graph))
        assert ("n1", "n2") not in ids

    def test_no_duplicate_pairs(self) -> None:
        graph = KGGraph(
            nodes=[
                KGNode(id="n1", name="ML"),
                KGNode(id="n2", name="Machine Learning"),
                KGNode(id="n3", name="Machine Learning Ops"),
            ],
            edges=[],
        )
        pairs = AbbreviationBlocker().candidate_pairs(graph)
        ids = [(min(a.id, b.id), max(a.id, b.id)) for a, b in pairs]
        assert len(ids) == len(set(ids))

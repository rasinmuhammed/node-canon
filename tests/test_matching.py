"""Tests for matching layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nodecanon.core.matching import LLMAssistedMatcher, RuleBasedMatcher
from nodecanon.core.models import KGNode, ScoreVector


def _sv(
    name: float = 0.0,
    semantic: float = 0.0,
    type_: float = 0.0,
    neighbor: float = 0.0,
    desc: float = 0.0,
) -> ScoreVector:
    return ScoreVector(
        name_similarity=name,
        semantic_similarity=semantic,
        type_agreement=type_,
        neighbor_overlap=neighbor,
        description_similarity=desc,
    )


_NODE_A = KGNode(id="a", name="IBM", type="ORGANIZATION")
_NODE_B = KGNode(id="b", name="I.B.M.", type="COMPANY")


class TestRuleBasedMatcher:
    def test_above_threshold_is_match(self) -> None:
        matcher = RuleBasedMatcher(threshold=0.75)
        sv = _sv(name=1.0, semantic=1.0, type_=1.0, neighbor=1.0, desc=1.0)
        assert matcher.is_match(_NODE_A, _NODE_B, sv) is True

    def test_below_threshold_is_not_match(self) -> None:
        matcher = RuleBasedMatcher(threshold=0.75)
        sv = _sv()  # all zeros → weighted_sum = 0.0
        assert matcher.is_match(_NODE_A, _NODE_B, sv) is False

    def test_exactly_at_threshold_is_match(self) -> None:
        matcher = RuleBasedMatcher(threshold=0.75)
        # Build a ScoreVector whose default weighted_sum == exactly 0.75.
        # Default weights: name=0.30, sem=0.25, type=0.20, nbr=0.20, desc=0.05
        # Use name=1.0, type=1.0, rest=0 → 0.30 + 0.20 = 0.50 … not enough.
        # name=1.0, sem=1.0, type=0.5 → 0.30 + 0.25 + 0.10 = 0.65 … still not.
        # Easiest: use custom weights summing to threshold.
        sv = _sv(name=0.75)
        weights = {
            "name_similarity": 1.0,
            "semantic_similarity": 0.0,
            "type_agreement": 0.0,
            "neighbor_overlap": 0.0,
            "description_similarity": 0.0,
        }
        matcher = RuleBasedMatcher(threshold=0.75, weights=weights)
        assert matcher.is_match(_NODE_A, _NODE_B, sv) is True

    def test_just_below_threshold_not_match(self) -> None:
        weights = {
            "name_similarity": 1.0,
            "semantic_similarity": 0.0,
            "type_agreement": 0.0,
            "neighbor_overlap": 0.0,
            "description_similarity": 0.0,
        }
        matcher = RuleBasedMatcher(threshold=0.75, weights=weights)
        sv = _sv(name=0.749)
        assert matcher.is_match(_NODE_A, _NODE_B, sv) is False

    def test_custom_threshold_respected(self) -> None:
        high = RuleBasedMatcher(threshold=0.95)
        low = RuleBasedMatcher(threshold=0.50)
        sv = _sv(name=1.0, semantic=1.0, type_=1.0, neighbor=0.5)
        # default weighted_sum ≈ 0.30+0.25+0.20+0.10+0 = 0.85
        assert not high.is_match(_NODE_A, _NODE_B, sv)
        assert low.is_match(_NODE_A, _NODE_B, sv)

    def test_custom_weights_respected(self) -> None:
        # Weight neighbor_overlap at 1.0, everything else 0.
        weights = {
            "name_similarity": 0.0,
            "semantic_similarity": 0.0,
            "type_agreement": 0.0,
            "neighbor_overlap": 1.0,
            "description_similarity": 0.0,
        }
        matcher = RuleBasedMatcher(threshold=0.75, weights=weights)
        sv_high = _sv(neighbor=0.9)
        sv_low = _sv(neighbor=0.3)
        assert matcher.is_match(_NODE_A, _NODE_B, sv_high) is True
        assert matcher.is_match(_NODE_A, _NODE_B, sv_low) is False


class TestLLMAssistedMatcher:
    def _matcher(self, low: float = 0.65, high: float = 0.80) -> LLMAssistedMatcher:
        weights = {
            "name_similarity": 1.0,
            "semantic_similarity": 0.0,
            "type_agreement": 0.0,
            "neighbor_overlap": 0.0,
            "description_similarity": 0.0,
        }
        rule = RuleBasedMatcher(threshold=high, weights=weights)
        return LLMAssistedMatcher(rule, ambiguous_low=low, ambiguous_high=high)

    def test_above_zone_auto_match_no_llm(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.90)  # above ambiguous_high=0.80
        with patch.object(matcher, "_call_llm") as mock_llm:
            result = matcher.is_match(_NODE_A, _NODE_B, sv)
        assert result is True
        mock_llm.assert_not_called()

    def test_below_zone_auto_reject_no_llm(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.30)  # below ambiguous_low=0.65
        with patch.object(matcher, "_call_llm") as mock_llm:
            result = matcher.is_match(_NODE_A, _NODE_B, sv)
        assert result is False
        mock_llm.assert_not_called()

    def test_in_zone_calls_llm(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)  # 0.65 <= 0.72 < 0.80
        with patch.object(matcher, "_call_llm", return_value=True) as mock_llm:
            result = matcher.is_match(_NODE_A, _NODE_B, sv)
        assert result is True
        mock_llm.assert_called_once_with(_NODE_A, _NODE_B, sv)

    def test_llm_says_no(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)
        with patch.object(matcher, "_call_llm", return_value=False):
            result = matcher.is_match(_NODE_A, _NODE_B, sv)
        assert result is False

    def test_is_ambiguous(self) -> None:
        matcher = self._matcher(low=0.65, high=0.80)
        assert matcher._is_ambiguous(_sv(name=0.72)) is True
        assert matcher._is_ambiguous(_sv(name=0.64)) is False
        assert matcher._is_ambiguous(_sv(name=0.80)) is False  # high boundary excluded

    def test_invalid_zone_raises(self) -> None:
        rule = RuleBasedMatcher()
        with pytest.raises(ValueError, match="ambiguous_low"):
            LLMAssistedMatcher(rule, ambiguous_low=0.80, ambiguous_high=0.65)

    def test_prompt_contains_node_names(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)
        prompt = matcher._build_prompt(_NODE_A, _NODE_B, sv)
        assert "IBM" in prompt
        assert "I.B.M." in prompt

    def test_llm_yes_response_parsed(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)
        with patch.object(matcher, "_llm_request", return_value="yes"):
            assert matcher._call_llm(_NODE_A, _NODE_B, sv) is True

    def test_llm_no_response_parsed(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)
        with patch.object(matcher, "_llm_request", return_value="no"):
            assert matcher._call_llm(_NODE_A, _NODE_B, sv) is False

    def test_llm_response_case_insensitive(self) -> None:
        matcher = self._matcher()
        sv = _sv(name=0.72)
        with patch.object(matcher, "_llm_request", return_value="Yes"):
            assert matcher._call_llm(_NODE_A, _NODE_B, sv) is True
        with patch.object(matcher, "_llm_request", return_value="NO"):
            assert matcher._call_llm(_NODE_A, _NODE_B, sv) is False

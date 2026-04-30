from __future__ import annotations

from abc import ABC, abstractmethod

from nodecanon.core.models import KGNode, ScoreVector


class BaseMatcher(ABC):
    @abstractmethod
    def is_match(self, node_a: KGNode, node_b: KGNode, score: ScoreVector) -> bool:
        raise NotImplementedError


class RuleBasedMatcher(BaseMatcher):
    """Decides matches via weighted sum of ScoreVector against a threshold."""

    def __init__(
        self,
        threshold: float = 0.75,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.weights = weights

    def is_match(self, _node_a: KGNode, _node_b: KGNode, score: ScoreVector) -> bool:
        return score.weighted_sum(self.weights) >= self.threshold


class LLMAssistedMatcher(BaseMatcher):
    """Routes ambiguous candidate pairs to an LLM for a binary merge decision.

    Only pairs whose weighted score falls in [ambiguous_low, ambiguous_high)
    are sent to the LLM. Pairs above the zone are auto-matched; pairs below
    are auto-rejected. This limits LLM calls to ~5-10% of candidates.
    """

    _PROMPT_TEMPLATE = """\
Are these two knowledge graph entities the same real-world entity?

Entity A:
  Name: {name_a}
  Type: {type_a}
  Description: {desc_a}

Entity B:
  Name: {name_b}
  Type: {type_b}
  Description: {desc_b}

Similarity scores:
  Name similarity:        {name_sim:.2f}
  Semantic similarity:    {sem_sim:.2f}
  Type agreement:         {type_agr:.2f}
  Neighbor overlap:       {nbr_ovl:.2f}
  Description similarity: {desc_sim:.2f}
  Weighted score:         {weighted:.2f}  (threshold: {threshold:.2f})

Answer with ONLY "yes" or "no".\
"""

    def __init__(
        self,
        rule_matcher: RuleBasedMatcher,
        ambiguous_low: float = 0.65,
        ambiguous_high: float = 0.80,
        provider: str = "openai",
        model: str | None = None,
    ) -> None:
        if ambiguous_low >= ambiguous_high:
            raise ValueError(
                f"ambiguous_low ({ambiguous_low}) must be less than "
                f"ambiguous_high ({ambiguous_high})."
            )
        self.rule_matcher = rule_matcher
        self.ambiguous_low = ambiguous_low
        self.ambiguous_high = ambiguous_high
        self.provider = provider
        self.model = model

    def is_match(self, node_a: KGNode, node_b: KGNode, score: ScoreVector) -> bool:
        weighted = score.weighted_sum(self.rule_matcher.weights)
        if weighted >= self.ambiguous_high:
            return True
        if weighted < self.ambiguous_low:
            return False
        return self._call_llm(node_a, node_b, score)

    def _is_ambiguous(self, score: ScoreVector) -> bool:
        weighted = score.weighted_sum(self.rule_matcher.weights)
        return self.ambiguous_low <= weighted < self.ambiguous_high

    def _build_prompt(
        self, node_a: KGNode, node_b: KGNode, score: ScoreVector
    ) -> str:
        return self._PROMPT_TEMPLATE.format(
            name_a=node_a.name,
            type_a=node_a.type or "unknown",
            desc_a=node_a.description or "(none)",
            name_b=node_b.name,
            type_b=node_b.type or "unknown",
            desc_b=node_b.description or "(none)",
            name_sim=score.name_similarity,
            sem_sim=score.semantic_similarity,
            type_agr=score.type_agreement,
            nbr_ovl=score.neighbor_overlap,
            desc_sim=score.description_similarity,
            weighted=score.weighted_sum(self.rule_matcher.weights),
            threshold=self.rule_matcher.threshold,
        )

    def _call_llm(self, node_a: KGNode, node_b: KGNode, score: ScoreVector) -> bool:
        prompt = self._build_prompt(node_a, node_b, score)
        raw = self._llm_request(prompt)
        return raw.strip().lower().startswith("yes")

    def _llm_request(self, prompt: str) -> str:
        """Send prompt to the configured LLM provider and return raw response."""
        match self.provider:
            case "openai":
                return self._openai_request(prompt)
            case "anthropic":
                return self._anthropic_request(prompt)
            case _:
                raise ValueError(
                    f"Unknown LLM provider {self.provider!r}. "
                    f"Supported: 'openai', 'anthropic'."
                )

    def _openai_request(self, prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package is required for LLM-assisted matching. "
                "Install it with: pip install nodecanon[llm]"
            ) from None

        model = self.model or "gpt-4o-mini"
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    def _anthropic_request(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for LLM-assisted matching. "
                "Install it with: pip install nodecanon[llm]"
            ) from None

        model = self.model or "claude-haiku-4-5-20251001"
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

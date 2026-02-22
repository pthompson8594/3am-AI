"""
Decision Gate — 3AM
Runs after memory retrieval. Decides whether the LLM should:
  - answer    → respond from memory/training knowledge
  - search    → use the web_search tool
  - ask       → request clarification from the user

Uses a hybrid approach:
  1. Rule-based fast path for obvious cases (zero latency)
  2. Small LLM call for ambiguous cases, with logprob of the decision
     token used as the gate confidence score
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

import httpx

# ── Patterns for the rule-based fast path ────────────────────────────────────

_SEARCH_TRIGGERS = re.compile(
    r"^(search|look up|find|google|check|what('s| is) (the )?(latest|current|today|now|recent)|"
    r"what happened|news about|price of|weather|stock)",
    re.IGNORECASE,
)

_CHITCHAT_TRIGGERS = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|great|sure|cool|yes|no|yep|nope|got it)\b",
    re.IGNORECASE,
)

_ASK_TRIGGERS = re.compile(
    r"\b(what do you mean|can you clarify|what exactly|which one|be more specific)\b",
    re.IGNORECASE,
)

# How many chars of memory context counts as "enough to answer"
_MEMORY_CONTEXT_MIN_LEN = 80

# LLM gate prompt — kept minimal for speed
_GATE_PROMPT = """\
You are a routing agent. Given a user query and available memory context, \
decide the best action.

Memory context:
{memory_context}

User query: {query}

Respond with exactly one word: answer, search, or ask
- answer  → you have enough knowledge/context to respond directly
- search  → the query needs fresh or specific external information
- ask     → the query is too ambiguous to act on without clarification
"""


@dataclass
class GateDecision:
    action: str          # "answer" | "search" | "ask"
    confidence: float    # 0.0–1.0 (logprob of action token, or rule-based estimate)
    reason: str          # short human-readable explanation
    from_rules: bool = False  # True if fast-path rule fired


class DecisionGate:
    """
    Evaluates whether the LLM should answer, search, or ask before generating
    a full response.
    """

    def __init__(self, llm_url: str, model: str):
        self.llm_url = llm_url
        self.model = model

    async def evaluate(
        self,
        query: str,
        memory_context: str,
        client: httpx.AsyncClient,
        search_threshold: float = 0.5,
        ask_threshold: float = 0.3,
        enabled: bool = True,
    ) -> GateDecision:
        """
        Evaluate the query and return a GateDecision.

        search_threshold: gate confidence below this → override to 'search'
        ask_threshold: gate confidence below this + short memory → override to 'ask'
        enabled: if False, always returns answer with confidence 1.0
        """
        if not enabled:
            return GateDecision(action="answer", confidence=1.0,
                                reason="gate disabled", from_rules=True)

        # ── Fast path: rule-based ─────────────────────────────────────────────
        query_stripped = query.strip()

        if _CHITCHAT_TRIGGERS.match(query_stripped):
            return GateDecision(action="answer", confidence=0.95,
                                reason="chitchat pattern", from_rules=True)

        if _SEARCH_TRIGGERS.match(query_stripped):
            return GateDecision(action="search", confidence=0.90,
                                reason="explicit search trigger", from_rules=True)

        if _ASK_TRIGGERS.search(query_stripped):
            return GateDecision(action="ask", confidence=0.85,
                                reason="ambiguity trigger", from_rules=True)

        # Very short query + no useful memory → search
        if len(query_stripped) < 15 and len(memory_context) < _MEMORY_CONTEXT_MIN_LEN:
            return GateDecision(action="search", confidence=0.75,
                                reason="short query, sparse memory", from_rules=True)

        # ── LLM path: ambiguous cases ─────────────────────────────────────────
        ctx_snippet = memory_context[:500] if memory_context else "(none)"
        prompt = _GATE_PROMPT.format(memory_context=ctx_snippet, query=query_stripped)

        try:
            resp = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "stream": False,
                    "temperature": 0.0,
                    "logprobs": True,
                    "top_logprobs": 1,
                },
                timeout=15.0,
            )
            data = resp.json()
        except Exception:
            # Gate failure → default to answer (safe fallback)
            return GateDecision(action="answer", confidence=0.5,
                                reason="gate LLM call failed", from_rules=True)

        choice = data.get("choices", [{}])[0]
        raw_text = (choice.get("message", {}).get("content", "") or "").strip().lower()

        # Parse action
        if "search" in raw_text:
            action = "search"
        elif "ask" in raw_text:
            action = "ask"
        else:
            action = "answer"

        # Extract confidence from logprob of first meaningful token
        confidence = 0.5
        try:
            logprobs_content = choice.get("logprobs", {}).get("content", [])
            if logprobs_content:
                lp = logprobs_content[0].get("logprob", -0.693)  # default ≈ 0.5
                confidence = round(max(0.05, min(1.0, math.exp(lp))), 3)
        except (KeyError, TypeError, IndexError):
            pass

        # Apply sensitivity thresholds
        reason = f"LLM gate → {action} (conf={confidence})"
        if action == "answer" and confidence < search_threshold and len(memory_context) < _MEMORY_CONTEXT_MIN_LEN:
            action = "search"
            reason = f"confidence {confidence} < threshold {search_threshold}, sparse memory → search"
        elif action == "answer" and confidence < ask_threshold:
            action = "ask"
            reason = f"confidence {confidence} < ask threshold {ask_threshold} → ask"

        return GateDecision(action=action, confidence=confidence, reason=reason)

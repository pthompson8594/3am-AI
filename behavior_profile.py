"""
Behavior Profile — 3AM
Stores learned behavioral preferences for the LLM assistant and injects them
into the system prompt. Updated by the introspection loop based on experience
log patterns.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experience_log import ExperienceLog


_DEFAULTS = {
    "tool_usage_bias": "balanced",       # "conservative" | "balanced" | "aggressive"
    "verbosity": "medium",               # "brief" | "medium" | "detailed"
    "uncertainty_behavior": "hedge",     # "answer" | "hedge" | "search"
    "search_threshold": 0.5,            # gate confidence below this → search
    "ask_threshold": 0.3,               # gate confidence below this → ask
    "user_preferences": {},
}

# How much to nudge thresholds per update cycle
_THRESHOLD_STEP = 0.05
_THRESHOLD_MIN = 0.2
_THRESHOLD_MAX = 0.8


class BehaviorProfile:
    """
    Manages per-user behavior profile.
    Thread-safe file-backed JSON store.
    """

    def __init__(self, data_dir: Path):
        self._path = data_dir / "behavior_profile.json"
        self._lock = threading.Lock()
        self._data: dict = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        with self._lock:
            if self._path.exists():
                try:
                    with open(self._path) as f:
                        loaded = json.load(f)
                    # Merge with defaults so new keys always exist
                    self._data = {**_DEFAULTS, **loaded}
                except (json.JSONDecodeError, OSError):
                    self._data = dict(_DEFAULTS)
            else:
                self._data = dict(_DEFAULTS)

    def _save(self):
        """Must be called under self._lock."""
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError:
            pass

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, updates: dict):
        """Merge partial updates and persist."""
        allowed = set(_DEFAULTS.keys())
        with self._lock:
            for k, v in updates.items():
                if k in allowed:
                    self._data[k] = v
            self._save()

    # ── System prompt injection ───────────────────────────────────────────────

    def to_prompt_fragment(self) -> str:
        """
        Returns a concise string injected into the system prompt.
        Kept short to minimise token cost.
        """
        d = self.get()
        lines = []

        verbosity_map = {
            "brief": "Keep responses concise and to the point.",
            "medium": "Use balanced detail — neither overly brief nor verbose.",
            "detailed": "Provide thorough, detailed responses.",
        }
        ub_map = {
            "answer": "When uncertain, still attempt to answer directly.",
            "hedge": "When uncertain, acknowledge uncertainty before answering.",
            "search": "When uncertain, prefer using the web_search tool.",
        }

        verb_hint = verbosity_map.get(d.get("verbosity", "medium"), "")
        ub_hint = ub_map.get(d.get("uncertainty_behavior", "hedge"), "")

        if verb_hint:
            lines.append(verb_hint)
        if ub_hint:
            lines.append(ub_hint)

        prefs = d.get("user_preferences", {})
        if prefs:
            pref_str = "; ".join(f"{k}={v}" for k, v in prefs.items())
            lines.append(f"User preferences: {pref_str}.")

        return "\n".join(lines)

    def reset(self):
        """Reset profile to defaults and delete the persisted file."""
        with self._lock:
            self._data = dict(_DEFAULTS)
            try:
                if self._path.exists():
                    self._path.unlink()
            except OSError:
                pass

    # ── Gate threshold accessors ──────────────────────────────────────────────

    @property
    def search_threshold(self) -> float:
        return float(self._data.get("search_threshold", 0.5))

    @property
    def ask_threshold(self) -> float:
        return float(self._data.get("ask_threshold", 0.3))

    # ── Introspection-driven update ───────────────────────────────────────────

    def update_from_experience(self, experience_log: "ExperienceLog"):
        """
        Adjust gate thresholds and uncertainty behavior based on feedback patterns.
        Called by the hourly introspection cycle.
        """
        patterns = experience_log.get_feedback_patterns()

        low_conf_neg = patterns.get("low_conf_negative_rate", 0.0)
        search_helped = patterns.get("search_helped_rate", 0.5)
        hallucination_rate = patterns.get("hallucination_rate", 0.0)

        with self._lock:
            st = float(self._data.get("search_threshold", 0.5))
            at = float(self._data.get("ask_threshold", 0.3))

            # If low-confidence answers are getting thumbs-down → lower search threshold
            # (be more willing to search earlier)
            if low_conf_neg > 0.6:
                st = max(_THRESHOLD_MIN, st - _THRESHOLD_STEP)
            elif low_conf_neg < 0.2 and search_helped > 0.7:
                # Searching is working well → keep or slightly raise threshold
                # (don't over-search)
                st = min(_THRESHOLD_MAX, st + _THRESHOLD_STEP * 0.5)

            # High hallucination rate → be more humble, hedge more
            if hallucination_rate > 0.15:
                self._data["uncertainty_behavior"] = "hedge"
                st = max(_THRESHOLD_MIN, st - _THRESHOLD_STEP)

            self._data["search_threshold"] = round(st, 3)
            self._data["ask_threshold"] = round(at, 3)
            self._save()

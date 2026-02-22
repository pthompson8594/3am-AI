"""
Experience Log — 3AM
Tracks every LLM interaction: gate decision, confidence score, tool used,
and user feedback. Used by the introspection loop to evolve the behavior profile.
"""

from __future__ import annotations

import json
import uuid
import time
import threading
from pathlib import Path


class ExperienceLog:
    """
    Thin wrapper around the experience_log table in memory.db.
    The table is created by memory.py's _init_db(), so this class
    only needs the db_path to open connections.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()

    # ── Connection helpers (mirrors memory.py pattern) ────────────────────────

    def _get_conn(self):
        import sqlite3
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(
        self,
        message_id: str,
        gate_action: str | None = None,
        gate_confidence: float | None = None,
        response_confidence: float | None = None,
        tool_used: str | None = None,
    ) -> str:
        """
        Record a new interaction. Returns the log entry id.
        Call this immediately after a response is generated.
        Feedback is added later via add_feedback().
        """
        entry_id = f"exp_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO experience_log
                    (id, timestamp, message_id, gate_action, gate_confidence,
                     response_confidence, tool_used, feedback_tags, outcome_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, '[]', NULL)
                """,
                (entry_id, now, message_id, gate_action, gate_confidence,
                 response_confidence, tool_used),
            )
            conn.commit()
            conn.close()
        return entry_id

    def add_feedback(self, message_id: str, value: str, tags: list[str]) -> bool:
        """
        Attach user feedback to an existing log entry by message_id.
        value: 'positive' | 'negative'
        tags: list of tag strings e.g. ['hallucinated', 'too_verbose']
        Returns True if a matching entry was found and updated.
        """
        outcome = 1.0 if value == "positive" else 0.0
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """
                UPDATE experience_log
                SET user_feedback = ?, feedback_tags = ?, outcome_score = ?
                WHERE message_id = ?
                """,
                (value, json.dumps(tags), outcome, message_id),
            )
            updated = cur.rowcount > 0
            conn.commit()
            conn.close()
        return updated

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Aggregate feedback stats for the settings panel.
        Returns:
            {
              "total": int,
              "rated": int,
              "positive": int,
              "negative": int,
              "by_tag": {"hallucinated": int, ...},
              "avg_response_confidence": float | None,
              "avg_gate_confidence": float | None,
            }
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT user_feedback, feedback_tags, response_confidence, gate_confidence FROM experience_log"
        ).fetchall()
        conn.close()

        total = len(rows)
        positive = sum(1 for r in rows if r["user_feedback"] == "positive")
        negative = sum(1 for r in rows if r["user_feedback"] == "negative")
        rated = positive + negative

        by_tag: dict[str, int] = {}
        for r in rows:
            if r["feedback_tags"]:
                try:
                    for tag in json.loads(r["feedback_tags"]):
                        by_tag[tag] = by_tag.get(tag, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass

        rc_vals = [r["response_confidence"] for r in rows if r["response_confidence"] is not None]
        gc_vals = [r["gate_confidence"] for r in rows if r["gate_confidence"] is not None]

        return {
            "total": total,
            "rated": rated,
            "positive": positive,
            "negative": negative,
            "by_tag": by_tag,
            "avg_response_confidence": round(sum(rc_vals) / len(rc_vals), 3) if rc_vals else None,
            "avg_gate_confidence": round(sum(gc_vals) / len(gc_vals), 3) if gc_vals else None,
        }

    def get_recent(self, limit: int = 100) -> list[dict]:
        """
        Return the most recent entries for introspection analysis.
        Entries with feedback are prioritised (ordered by feedback presence desc, then timestamp desc).
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM experience_log
            ORDER BY (user_feedback IS NOT NULL) DESC, timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["feedback_tags"] = json.loads(d["feedback_tags"] or "[]")
            except (json.JSONDecodeError, TypeError):
                d["feedback_tags"] = []
            result.append(d)
        return result

    def get_analytics(self) -> dict:
        """
        Comprehensive analytics for the settings panel.
        Returns gate decision breakdown, confidence distribution,
        feedback patterns, and top tags in one query.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT gate_action, gate_confidence, response_confidence, "
            "user_feedback, feedback_tags FROM experience_log"
        ).fetchall()
        conn.close()

        total = len(rows)
        rated_rows = [r for r in rows if r["user_feedback"] is not None]
        positive = sum(1 for r in rated_rows if r["user_feedback"] == "positive")
        negative = len(rated_rows) - positive

        gate_counts = {"answer": 0, "search": 0, "ask": 0}
        for r in rows:
            if r["gate_action"] in gate_counts:
                gate_counts[r["gate_action"]] += 1

        conf_vals = [r["response_confidence"] for r in rows if r["response_confidence"] is not None]
        high   = sum(1 for v in conf_vals if v >= 0.65)
        medium = sum(1 for v in conf_vals if 0.40 <= v < 0.65)
        low    = sum(1 for v in conf_vals if v < 0.40)
        avg_rc = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None

        gc_vals = [r["gate_confidence"] for r in rows if r["gate_confidence"] is not None]
        avg_gc  = round(sum(gc_vals) / len(gc_vals), 3) if gc_vals else None

        patterns = self.get_feedback_patterns()

        by_tag: dict[str, int] = {}
        for r in rated_rows:
            try:
                for tag in json.loads(r["feedback_tags"] or "[]"):
                    by_tag[tag] = by_tag.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass
        top_tags = sorted(by_tag.items(), key=lambda x: x[1], reverse=True)[:4]

        return {
            "interactions": {
                "total": total,
                "rated": len(rated_rows),
                "positive": positive,
                "negative": negative,
            },
            "confidence": {
                "avg_response": avg_rc,
                "avg_gate": avg_gc,
                "high": high,
                "medium": medium,
                "low": low,
            },
            "gate_decisions": gate_counts,
            "feedback_patterns": {
                **patterns,
                "top_tags": top_tags,
            },
        }

    def get_feedback_patterns(self) -> dict:
        """
        Summarise patterns for the behavior profile update logic.
        Returns:
            {
              "low_conf_negative_rate": float,  # how often low-confidence responses get thumbs-down
              "search_helped_rate": float,       # how often gate_action=search got positive feedback
              "hallucination_rate": float,       # fraction of rated responses flagged hallucinated
            }
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT gate_action, gate_confidence, response_confidence, user_feedback, feedback_tags FROM experience_log WHERE user_feedback IS NOT NULL"
        ).fetchall()
        conn.close()

        if not rows:
            return {
                "low_conf_negative_rate": 0.0,
                "search_helped_rate": 0.5,
                "hallucination_rate": 0.0,
            }

        total = len(rows)

        low_conf = [r for r in rows if r["response_confidence"] is not None and r["response_confidence"] < 0.4]
        low_conf_negative = sum(1 for r in low_conf if r["user_feedback"] == "negative")
        low_conf_negative_rate = low_conf_negative / len(low_conf) if low_conf else 0.0

        search_rows = [r for r in rows if r["gate_action"] == "search"]
        search_positive = sum(1 for r in search_rows if r["user_feedback"] == "positive")
        search_helped_rate = search_positive / len(search_rows) if search_rows else 0.5

        hallucinated = 0
        for r in rows:
            try:
                tags = json.loads(r["feedback_tags"] or "[]")
                if "hallucinated" in tags:
                    hallucinated += 1
            except (json.JSONDecodeError, TypeError):
                pass
        hallucination_rate = hallucinated / total

        return {
            "low_conf_negative_rate": round(low_conf_negative_rate, 3),
            "search_helped_rate": round(search_helped_rate, 3),
            "hallucination_rate": round(hallucination_rate, 3),
        }

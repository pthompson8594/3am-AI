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

        agg = conn.execute("""
            SELECT
                COUNT(*)                                                             AS total,
                SUM(CASE WHEN user_feedback = 'positive' THEN 1 ELSE 0 END)        AS positive,
                SUM(CASE WHEN user_feedback = 'negative' THEN 1 ELSE 0 END)        AS negative,
                AVG(response_confidence)                                             AS avg_rc,
                AVG(gate_confidence)                                                 AS avg_gc
            FROM experience_log
        """).fetchone()

        # Tags column is JSON — must unpack in Python; fetch only rows that have tags
        tag_rows = conn.execute(
            "SELECT feedback_tags FROM experience_log WHERE feedback_tags IS NOT NULL"
        ).fetchall()

        conn.close()

        by_tag: dict[str, int] = {}
        for r in tag_rows:
            try:
                for tag in json.loads(r["feedback_tags"]):
                    by_tag[tag] = by_tag.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        positive = agg["positive"] or 0
        negative = agg["negative"] or 0
        return {
            "total": agg["total"] or 0,
            "rated": positive + negative,
            "positive": positive,
            "negative": negative,
            "by_tag": by_tag,
            "avg_response_confidence": round(agg["avg_rc"], 3) if agg["avg_rc"] is not None else None,
            "avg_gate_confidence": round(agg["avg_gc"], 3) if agg["avg_gc"] is not None else None,
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
        All aggregations done in SQL; only the feedback_tags JSON column
        still needs a small Python pass for tag counting.
        """
        conn = self._get_conn()

        # Totals and feedback counts
        totals = conn.execute("""
            SELECT
                COUNT(*)                                                        AS total,
                COUNT(user_feedback)                                            AS rated,
                SUM(CASE WHEN user_feedback = 'positive' THEN 1 ELSE 0 END)   AS positive,
                AVG(response_confidence)                                        AS avg_rc,
                AVG(gate_confidence)                                            AS avg_gc,
                SUM(CASE WHEN response_confidence >= 0.65 THEN 1 ELSE 0 END)  AS high_conf,
                SUM(CASE WHEN response_confidence >= 0.40
                          AND response_confidence < 0.65 THEN 1 ELSE 0 END)   AS mid_conf,
                SUM(CASE WHEN response_confidence < 0.40 THEN 1 ELSE 0 END)   AS low_conf
            FROM experience_log
        """).fetchone()

        # Gate action breakdown
        gate_rows = conn.execute("""
            SELECT gate_action, COUNT(*) AS n
            FROM experience_log
            WHERE gate_action IN ('answer','search','ask')
            GROUP BY gate_action
        """).fetchall()

        # Tags only (need Python to unpack JSON array column)
        tag_rows = conn.execute(
            "SELECT feedback_tags FROM experience_log WHERE user_feedback IS NOT NULL AND feedback_tags IS NOT NULL"
        ).fetchall()

        conn.close()

        total   = totals["total"] or 0
        rated   = totals["rated"] or 0
        positive = totals["positive"] or 0
        avg_rc  = round(totals["avg_rc"], 3) if totals["avg_rc"] is not None else None
        avg_gc  = round(totals["avg_gc"], 3) if totals["avg_gc"] is not None else None

        gate_counts = {"answer": 0, "search": 0, "ask": 0}
        for r in gate_rows:
            gate_counts[r["gate_action"]] = r["n"]

        by_tag: dict[str, int] = {}
        for r in tag_rows:
            try:
                for tag in json.loads(r["feedback_tags"] or "[]"):
                    by_tag[tag] = by_tag.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass
        top_tags = sorted(by_tag.items(), key=lambda x: x[1], reverse=True)[:4]

        patterns = self.get_feedback_patterns()

        return {
            "interactions": {
                "total": total,
                "rated": rated,
                "positive": positive,
                "negative": rated - positive,
            },
            "confidence": {
                "avg_response": avg_rc,
                "avg_gate": avg_gc,
                "high":   totals["high_conf"] or 0,
                "medium": totals["mid_conf"]  or 0,
                "low":    totals["low_conf"]  or 0,
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

        agg = conn.execute("""
            SELECT
                COUNT(*)                                                               AS total,
                SUM(CASE WHEN response_confidence < 0.4 THEN 1 ELSE 0 END)           AS low_conf_total,
                SUM(CASE WHEN response_confidence < 0.4
                          AND user_feedback = 'negative' THEN 1 ELSE 0 END)          AS low_conf_neg,
                SUM(CASE WHEN gate_action = 'search' THEN 1 ELSE 0 END)              AS search_total,
                SUM(CASE WHEN gate_action = 'search'
                          AND user_feedback = 'positive' THEN 1 ELSE 0 END)          AS search_pos
            FROM experience_log
            WHERE user_feedback IS NOT NULL
        """).fetchone()

        # Hallucination tag still needs JSON unpacking
        tag_rows = conn.execute(
            "SELECT feedback_tags FROM experience_log WHERE user_feedback IS NOT NULL AND feedback_tags IS NOT NULL"
        ).fetchall()

        conn.close()

        total = agg["total"] or 0
        if total == 0:
            return {
                "low_conf_negative_rate": 0.0,
                "search_helped_rate": 0.5,
                "hallucination_rate": 0.0,
            }

        low_conf_total = agg["low_conf_total"] or 0
        low_conf_neg   = agg["low_conf_neg"]   or 0
        search_total   = agg["search_total"]   or 0
        search_pos     = agg["search_pos"]     or 0

        low_conf_negative_rate = low_conf_neg / low_conf_total if low_conf_total else 0.0
        search_helped_rate     = search_pos   / search_total   if search_total   else 0.5

        hallucinated = 0
        for r in tag_rows:
            try:
                if "hallucinated" in json.loads(r["feedback_tags"] or "[]"):
                    hallucinated += 1
            except (json.JSONDecodeError, TypeError):
                pass
        hallucination_rate = hallucinated / total

        return {
            "low_conf_negative_rate": round(low_conf_negative_rate, 3),
            "search_helped_rate": round(search_helped_rate, 3),
            "hallucination_rate": round(hallucination_rate, 3),
        }

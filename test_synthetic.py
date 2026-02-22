#!/usr/bin/env python3
"""
Synthetic test suite for the 3AM feedback loop.

Simulates weeks of interactions to validate that the closed
    feedback → experience_log → behavior_profile
cycle behaves correctly without requiring a running server or LLM.

Usage:
    python test_synthetic.py              # full 3-week simulation
    python test_synthetic.py --week 1     # run only week 1
    python test_synthetic.py --clean      # delete test data dir and exit

Each "week" seeds a batch of experience_log entries, runs
BehaviorProfile.update_from_experience(), and asserts the profile changed
(or stayed stable) as expected.
"""

import argparse
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path

# ── Test data directory (isolated from real user data) ───────────────────────

TEST_DIR = Path.home() / ".local/share/3am/test_synthetic"


# ── Minimal DB bootstrap (mirrors memory.py _init_db, experience_log only) ───

def _bootstrap_db(db_path: Path):
    """Create the experience_log table if it doesn't already exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experience_log (
            id                  TEXT PRIMARY KEY,
            timestamp           REAL NOT NULL,
            message_id          TEXT NOT NULL,
            gate_action         TEXT,
            gate_confidence     REAL,
            response_confidence REAL,
            tool_used           TEXT,
            user_feedback       TEXT,
            feedback_tags       TEXT DEFAULT '[]',
            outcome_score       REAL
        )
    """)
    conn.commit()
    conn.close()


# ── Scenario definitions ──────────────────────────────────────────────────────

def _msg():
    return f"msg_{uuid.uuid4().hex[:10]}"


def scenario_week1():
    """
    Good week: high-confidence answers, mostly positive feedback.
    Expected: thresholds hold steady (no overreaction to good data).
    """
    entries = []
    # 25 high-confidence answers — all positive
    for _ in range(25):
        entries.append({
            "message_id": _msg(),
            "gate_action": "answer",
            "gate_confidence": 0.88,
            "response_confidence": 0.91,
            "tool_used": None,
            "feedback": "positive",
            "tags": [],
        })
    # 5 search actions that worked well
    for _ in range(5):
        entries.append({
            "message_id": _msg(),
            "gate_action": "search",
            "gate_confidence": 0.45,
            "response_confidence": 0.78,
            "tool_used": "web_search",
            "feedback": "positive",
            "tags": ["found_it"],
        })
    return entries


def scenario_week2():
    """
    Bad week: low-confidence answers, majority negative feedback.
    Expected: search_threshold lowers (system learns to search sooner).
    """
    entries = []
    # 20 low-confidence answers — mostly negative
    for i in range(20):
        entries.append({
            "message_id": _msg(),
            "gate_action": "answer",
            "gate_confidence": 0.38,
            "response_confidence": 0.31,   # < 0.4 threshold
            "tool_used": None,
            "feedback": "negative" if i < 14 else "positive",  # 70% negative
            "tags": ["incorrect"] if i < 14 else [],
        })
    # 10 unrated interactions (normal noise)
    for _ in range(10):
        entries.append({
            "message_id": _msg(),
            "gate_action": "answer",
            "gate_confidence": 0.55,
            "response_confidence": 0.65,
            "tool_used": None,
            "feedback": None,
            "tags": [],
        })
    return entries


def scenario_week3():
    """
    Hallucination week: frequent 'hallucinated' tags.
    Expected: uncertainty_behavior → 'hedge', search_threshold lowers further.
    """
    entries = []
    # 18 negative responses explicitly tagged 'hallucinated' (>15% of all 30)
    for _ in range(18):
        entries.append({
            "message_id": _msg(),
            "gate_action": "answer",
            "gate_confidence": 0.60,
            "response_confidence": 0.58,
            "tool_used": None,
            "feedback": "negative",
            "tags": ["hallucinated"],
        })
    # 12 unrated
    for _ in range(12):
        entries.append({
            "message_id": _msg(),
            "gate_action": "answer",
            "gate_confidence": 0.70,
            "response_confidence": 0.72,
            "tool_used": None,
            "feedback": None,
            "tags": [],
        })
    return entries


SCENARIOS = [
    {
        "label": "Good session (high confidence, positive feedback)",
        "entries_fn": scenario_week1,
        "assertions": lambda before, after: [
            # search_threshold must not drop on a good week
            (after["search_threshold"] >= before["search_threshold"] - 0.001,
             f"search_threshold should be stable or higher on a good week, "
             f"got {before['search_threshold']} → {after['search_threshold']}"),
        ],
    },
    {
        "label": "Bad session (low confidence, 70% negative feedback)",
        "entries_fn": scenario_week2,
        "assertions": lambda before, after: [
            (after["search_threshold"] < before["search_threshold"],
             f"search_threshold should have dropped, "
             f"got {before['search_threshold']} → {after['search_threshold']}"),
        ],
    },
    {
        "label": "Hallucination week (>15% 'hallucinated' tags)",
        "entries_fn": scenario_week3,
        "assertions": lambda before, after: [
            (after["uncertainty_behavior"] == "hedge",
             f"uncertainty_behavior should be 'hedge', got '{after['uncertainty_behavior']}'"),
            (after["search_threshold"] < before["search_threshold"],
             f"search_threshold should have dropped further, "
             f"got {before['search_threshold']} → {after['search_threshold']}"),
        ],
    },
]


# ── Seed helpers ──────────────────────────────────────────────────────────────

def seed_entries(log, entries):
    """Insert a list of scenario entry dicts into the experience log."""
    for e in entries:
        mid = e["message_id"]
        log.record(
            message_id=mid,
            gate_action=e.get("gate_action"),
            gate_confidence=e.get("gate_confidence"),
            response_confidence=e.get("response_confidence"),
            tool_used=e.get("tool_used"),
        )
        if e.get("feedback"):
            log.add_feedback(mid, e["feedback"], e.get("tags", []))


# ── Pretty printer ────────────────────────────────────────────────────────────

def _diff(key, before, after):
    b = before.get(key)
    a = after.get(key)
    if b == a:
        return f"  {key:<26} {b!s:<8} (no change)"
    arrow = "↑" if (isinstance(a, float) and isinstance(b, float) and a > b) else "↓" if (isinstance(a, float) and isinstance(b, float) and a < b) else "→"
    return f"  {key:<26} {b!s:<8} {arrow}  {a!s}"


def print_week_report(week_num, total_weeks, label, stats, patterns, before, after, results):
    w = "═" * 54
    print(f"\n{w}")
    print(f"  3AM Synthetic Test — Week {week_num} of {total_weeks}")
    print(f"  {label}")
    print(w)
    print(f"  Interactions : {stats['total']}  |  Rated : {stats['rated']}  |  "
          f"Positive : {stats['positive']}  |  Negative : {stats['negative']}")
    print("  Feedback patterns:")
    print(f"    low_conf_negative_rate  : {patterns['low_conf_negative_rate']:.3f}")
    print(f"    search_helped_rate      : {patterns['search_helped_rate']:.3f}")
    print(f"    hallucination_rate      : {patterns['hallucination_rate']:.3f}")
    print("  Behavior Profile changes:")
    for key in ("search_threshold", "ask_threshold", "uncertainty_behavior"):
        print(f"  {_diff(key, before, after)}")

    all_pass = True
    for ok, msg in results:
        status = "  PASS" if ok else "  FAIL"
        print(f"  {status} — {msg if not ok else 'assertion passed'}")
        if not ok:
            all_pass = False

    result_str = "PASS" if all_pass else "FAIL"
    print(f"  Result: {result_str}")
    return all_pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3AM synthetic feedback loop test")
    parser.add_argument("--week", type=int, choices=[1, 2, 3], help="Run only one week")
    parser.add_argument("--clean", action="store_true", help="Delete test data dir and exit")
    args = parser.parse_args()

    if args.clean:
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
            print(f"Cleaned: {TEST_DIR}")
        else:
            print("Nothing to clean.")
        return

    # ── Setup ──
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    db_path = TEST_DIR / "memory.db"
    _bootstrap_db(db_path)

    # Import after path setup so sys.path resolution works from project dir
    try:
        from experience_log import ExperienceLog
        from behavior_profile import BehaviorProfile
    except ImportError as exc:
        print(f"Import error: {exc}")
        print("Run this script from the 3am-AI/ directory.")
        sys.exit(1)

    # Fresh log + profile for this run
    # Wipe any leftover data from a previous run
    conn = sqlite3.connect(str(db_path))
    conn.execute("DELETE FROM experience_log")
    conn.commit()
    conn.close()

    log = ExperienceLog(db_path)
    bp = BehaviorProfile(TEST_DIR)
    bp.reset()  # start from defaults every run

    scenarios = SCENARIOS if args.week is None else [SCENARIOS[args.week - 1]]
    total_weeks = len(scenarios)
    week_results = []

    for i, scenario in enumerate(scenarios, start=1 if args.week is None else args.week):
        entries = scenario["entries_fn"]()
        seed_entries(log, entries)

        before = bp.get()
        bp.update_from_experience(log)
        after = bp.get()

        stats = log.get_stats()
        patterns = log.get_feedback_patterns()
        assertion_results = scenario["assertions"](before, after)

        ok = print_week_report(i, total_weeks, scenario["label"],
                               stats, patterns, before, after, assertion_results)
        week_results.append((i, ok))

    # ── Summary ──
    print("\n" + "─" * 54)
    summary = "  " + "   ".join(
        f"Week {w} {'PASS' if ok else 'FAIL'}"
        for w, ok in week_results
    )
    print(summary)

    all_passed = all(ok for _, ok in week_results)
    if all_passed:
        print("  All assertions passed.\n")
    else:
        print("  Some assertions FAILED — see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Memory System — SQLite + sqlite-vec storage with Torque Clustering.

Storage architecture:
- memory.db (SQLite, WAL mode) per user
- Embeddings live in a sqlite-vec virtual table (float[768], cosine distance)
- Embeddings are NOT loaded into RAM at startup — fetched on demand
- Dedup check: single sqlite-vec MATCH query instead of full-scan loop
- Conflict resolution: per-memory MATCH query instead of O(n²) loop
- Clustering: batch-fetches all embeddings once into numpy, then releases them

Includes Torque Clustering, deferred extraction, async clustering,
compact user profile generation, and conflict resolution.
"""

import asyncio
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable
import threading

if TYPE_CHECKING:
    from data_security import DataEncryptor

import httpx
import numpy as np
import sqlite_vec
from scipy.spatial.distance import cdist

DEFAULT_MEMORY_DIR = Path.home() / ".local/share/3am"
DEFAULT_CONFIG_DIR = Path.home() / ".config/3am"

CLUSTER_HEALTH_THRESHOLD = 0.35
MESSAGE_RETENTION_THRESHOLD = 0.35
MIN_MEMORIES_FOR_CLUSTERING = 5
NEW_CLUSTER_SIMILARITY_THRESHOLD = 0.5
RECLUSTER_THRESHOLD_RATIO = 0.5
CONFLICT_SIMILARITY_THRESHOLD = 0.75
MAX_CLUSTER_SIZE = 20          # clusters over this are split on the nightly pass
FULL_RECLUSTER_DAYS = 7        # full O(n²) recluster once per week

# Cosine distance equivalents (distance = 1 - similarity)
DEDUP_DISTANCE = 1.0 - 0.92          # 0.08 — identical fact, skip
CONFLICT_DIST_NEAR = DEDUP_DISTANCE  # 0.08 — closer than this = duplicate
CONFLICT_DIST_FAR  = 1.0 - CONFLICT_SIMILARITY_THRESHOLD  # 0.25 — further = different topic

DECAY_RATES = {
    5: 0.0005,
    4: 0.0005,
    3: 0.005,
    2: 0.025,
    1: 0.1,
}

# ── Three-universe memory constants ──────────────────────────────────────────
# episodic   — personal experiences, preferences, conversation facts
# declarative — world knowledge, ingested documents, research findings
# procedural  — learned patterns, behavioural rules, how-to knowledge
VALID_UNIVERSES = {"episodic", "declarative", "procedural"}

# Base decay rate multipliers per universe (applied on top of priority rates)
UNIVERSE_DECAY_MULTIPLIERS = {
    "episodic":    1.0,   # standard decay — personal memories fade naturally
    "declarative": 0.3,   # slow decay — reference knowledge persists
    "procedural":  0.1,   # very slow decay — only replaced by better patterns
}

# Context allocation bias per query type (episodic, declarative, procedural)
CONTEXT_BIAS = {
    "personal":    (0.55, 0.25, 0.20),
    "factual":     (0.17, 0.63, 0.20),  # rounds to 2+8+2=12 (0.20,0.60,0.20 rounded to 11)
    "procedural":  (0.20, 0.25, 0.55),
    "balanced":    (0.35, 0.40, 0.25),
}

# ── Memory lane constants ─────────────────────────────────────────────────
LANE_MIN_SIM  = 0.55   # below this = unrelated, no semantic lane
LANE_MAX_SIM  = 0.92   # above this = duplicate (already deduped before here)
LANE_MAX_K    = 8      # max neighbors to check per new memory
CAUSAL_WEIGHT = 0.7    # fixed weight for one-way causal lanes

# ── Prompts ──────────────────────────────────────────────────────────────────

CLUSTER_THEME_PROMPT = """These memories about a user are clustered together. Generate a short theme (2-5 words) that captures what they have in common.

Memories:
{memories}

Respond with JSON:
{{"theme": "<2-5 word theme describing the user>", "category": "<identity|preferences|activities|projects|interests|other>"}}"""

USER_PROFILE_PROMPT = """Build a maximally dense user profile from these facts. Compress aggressively: abbreviate, drop articles/spaces where unambiguous, use symbols. No sentences. Only include fields you have data for.

Facts:
{facts}

Example output (show how dense this should be):
Alex|SWE|Portland; Prefs:Neovim,ArchLinux,terse; Stack:Python,Rust,Nix; Projects:ML/LLM,OSS; Traits:direct,self-taught

Respond with JSON:
{{"profile": "<single line or 2 lines max, pack as many facts as possible>"}}"""

DURABLE_FACT_PROMPT = """Extract durable facts from this single exchange. Durable means: still true tomorrow, next week, or longer.

Skip entirely: system metrics (CPU%, memory%, disk usage percentage, uptime, current load), values that change minute-to-minute, tasks still in progress, real-time events.
Include: identity, preferences, skills, tools, installed software, hardware specs, projects, opinions, completed actions, persistent facts about the environment.

Priority:
5 - Remember FOREVER (name, identity, profession, location, core system facts)
4 - Remember for MONTHS (preferences, skills, installed tools, hardware specs)
3 - Remember for WEEKS (ongoing projects, patterns, recurring topics)

memory_type — classify each fact:
- "episodic"    : personal to the user (preferences, identity, experiences, "I/my/we" facts)
- "declarative" : general world/technical knowledge (how things work, external facts, tool docs)
- "procedural"  : learned patterns, behavioural rules, "when X do Y" knowledge

User: {user_message}
Assistant: {assistant_response}

Respond with JSON — up to 5 facts, priority 3 or higher only:
{{"facts": [{{"priority": <3-5>, "summary": "<concise durable fact>", "category": "<identity|preferences|activities|projects|interests|other>", "memory_type": "<episodic|declarative|procedural>"}}]}}
If nothing durable was learned: {{"facts": []}}"""

GROUPING_PROMPT = """Group these conversations by topic. Conversations covering the same subject should be in the same group. A group can be large if the conversations are all closely related.

Conversations:
{conversations}

Respond with JSON — every index must appear in exactly one group:
{{"groups": [[0, 1, 2], [3, 4], [5]]}}"""

EXTRACTION_PROMPT = """What facts about the user can we learn from these conversations? Extract ALL distinct learnable facts.

Rate each fact:
5 - Remember FOREVER (name, identity, profession, location)
4 - Remember for MONTHS (preferences, skills, hobbies, opinions)
3 - Remember for WEEKS (patterns, projects, recurring topics)
2 - Remember for DAYS (current tasks, temporary context)
1 - Skip (not worth storing)

Conversations:
{conversations}

Respond with JSON — up to 8 facts, omit priority-1:
{{"facts": [{{"priority": <2-5>, "summary": "<concise fact about user>", "category": "<identity|preferences|activities|projects|interests|other>"}}]}}"""


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry. embedding is None at runtime — lives in sqlite-vec."""
    id: str
    message: str
    response: str
    timestamp: float
    priority: int
    category: str
    summary: str
    cluster_id: Optional[str] = None
    embedding: Optional[list[float]] = None  # Not kept in RAM; only set transiently
    memory_type: str = "episodic"     # episodic | declarative | procedural
    source_type: str = "conversation" # conversation | ingestion | research
    access_count: int = 0             # incremented each retrieval; decay resistance
    last_accessed: float = 0.0        # timestamp of last retrieval


@dataclass
class MemoryCluster:
    """A cluster of related memories discovered by Torque Clustering."""
    id: str
    theme: str
    center_vector: list[float]   # Kept in RAM — small (~50 clusters), needed for retrieval scoring
    message_refs: list[str]
    priority: int
    last_update: float
    torque_mass: float = 0.0


# ── Embedding model ───────────────────────────────────────────────────────────

class EmbeddingModel:
    """Lazy-loaded sentence-transformer for summary embeddings (CPU only)."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    _kwargs = dict(
                        trust_remote_code=True,
                        device="cpu",
                        model_kwargs={"torch_dtype": torch.float16},
                    )
                    try:
                        # Use local cache — avoids HF Hub network call on every reload
                        self._model = SentenceTransformer(
                            self.model_name, local_files_only=True, **_kwargs
                        )
                    except Exception:
                        # First run: cache miss, download from HF Hub
                        self._model = SentenceTransformer(self.model_name, **_kwargs)

    def embed(self, text: str) -> list[float]:
        self._load_model()
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load_model()
        return self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=8
        ).tolist()

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Release model weights from RAM. Will reload on next embed() call."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            gc.collect()

# Shared singleton — loaded once, reused across all MemorySystem instances.
_SHARED_EMBEDDER: Optional[EmbeddingModel] = None

def get_shared_embedder() -> EmbeddingModel:
    global _SHARED_EMBEDDER
    if _SHARED_EMBEDDER is None:
        _SHARED_EMBEDDER = EmbeddingModel()
    return _SHARED_EMBEDDER


def _rrf_merge(list_a: list, list_b: list, k: int = 60) -> list:
    """Reciprocal Rank Fusion — combine two ranked lists into one."""
    scores: dict = {}
    for rank, item in enumerate(list_a):
        scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    for rank, item in enumerate(list_b):
        scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])


def _llm_json_content(response) -> str:
    """
    Extract JSON content from an LLM chat-completions response.

    Qwen3 sometimes puts its JSON *inside* a <think> block and leaves the main
    content field empty.  Strategy:
      1. Take the content field, strip any <think>…</think> wrapper → use if non-empty.
      2. If still empty, pull the raw text out of the think block itself.
      3. Return whatever we found (may still be empty — caller's json.loads will raise).
    """
    import re as _re
    raw = (response.json()["choices"][0]["message"]["content"] or "").strip()

    # Extract think-block text before stripping, in case we need it as fallback
    think_match = _re.search(r"<think>(.*?)</think>", raw, flags=_re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    # Strip think block from main content
    content = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()

    # If main content is empty, try the think block text
    if not content and think_text:
        content = think_text

    return content


def _classify_query(query: str) -> str:
    """
    Heuristic classifier for three-universe context allocation.
    Returns: "personal" | "factual" | "procedural" | "balanced"
    """
    q = query.lower()
    personal  = sum(1 for w in ("i ", "my ", " me ", " i'", " we ", "our ", "prefer", "like ", "feel ", "tell me about myself", "what do i") if w in q)
    factual   = sum(1 for w in ("how ", "what is", "explain", "define", "does ", "work ", "what are", "describe", "documentation") if w in q)
    procedural = sum(1 for w in ("when ", "should ", "pattern", "behavior", "behave", "rule ", "always ", "never ", "if i", "next time", "remember to") if w in q)

    mx = max(personal, factual, procedural)
    if mx == 0:
        return "balanced"
    if personal == mx and personal > factual and personal > procedural:
        return "personal"
    if factual == mx and factual > personal and factual > procedural:
        return "factual"
    if procedural == mx and procedural > personal and procedural > factual:
        return "procedural"
    return "balanced"


# ── Memory system ─────────────────────────────────────────────────────────────

class MemorySystem:
    """
    Persistent memory with Torque Clustering and SQLite + sqlite-vec storage.

    Storage layout (memory.db):
      memories      — id, summary, category, priority, timestamp, cluster_id, message, response
      clusters      — id, theme, priority, last_update, torque_mass, center_vector (BLOB), message_refs (JSON)
      vec_memories  — memory_id TEXT, embedding float[768] distance_metric=cosine  (sqlite-vec virtual table)
      meta          — key/value for user_profile and stats

    In-memory caches:
      self.messages  — MemoryEntry metadata only (embedding=None)
      self.clusters  — MemoryCluster with center_vector loaded (small, needed for retrieval)
    """

    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        data_dir: Optional[Path] = None,
        model: str = "Qwen3-14B-Q4_K_M",
        encryptor: Optional["DataEncryptor"] = None,
    ):
        self.llm_url = llm_url
        self.encryptor = encryptor
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_MEMORY_DIR
        self.db_file = self.data_dir / "memory.db"
        self.pending_file = self.data_dir / "pending_conversations.jsonl"
        self.model = model
        self.llm_model_id = self._load_model_id()
        self.embedder = get_shared_embedder()

        self.messages: dict[str, MemoryEntry] = {}
        self.clusters: dict[str, MemoryCluster] = {}
        self._clustering_dirty = False
        self._clustering_in_progress = False
        self._user_profile: Optional[str] = None
        self._profile_dirty = False
        self._viz_cache: Optional[dict] = None       # cached UMAP/PCA result
        self._viz_cache_key: tuple = (-1, -1)        # (memory_count, cluster_count) at last compute
        self._lanes_ready: Optional[bool] = None     # None = unknown, True/False cached after first check
        # LRU cache for query embeddings: {query_key: (embedding, timestamp)}
        # Avoids re-running the sentence-transformer on repeated/similar queries.
        self._embed_cache: dict = {}
        self._EMBED_CACHE_TTL = 60.0   # seconds
        self._EMBED_CACHE_MAX = 32     # max entries before oldest evicted
        self.stats = {
            "total_messages": 0,
            "active_clusters": 0,
            "last_cleanup": time.time(),
            "last_clustering": 0,
            "last_full_recluster": 0,
        }

        # Rolling dedup stats for saturation detection: list of {"extracted": N, "stored": M}
        self._recent_exchange_stats: list = []

        # Per-thread DB connections (WAL allows concurrent reads from multiple threads)
        self._local = threading.local()
        # Write lock — prevents two threads from racing on a write
        self._write_lock = threading.Lock()

        self._init_db()
        self._load_from_db()
        self._cleanup()

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_model_id(self) -> str:
        try:
            config_file = DEFAULT_CONFIG_DIR / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    return json.load(f).get("llm_model", "qwen3-14b")
        except Exception as e:
            print(f"[Memory] Config load error: {e}")
        return "qwen3-14b"

    def _load_adjustment_factor(self) -> float:
        """Read clustering_adjustment_factor from config.json (default 0.5)."""
        try:
            config_file = DEFAULT_CONFIG_DIR / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    return float(json.load(f).get("clustering_adjustment_factor", 0.5))
        except Exception:
            pass
        return 0.5

    # ── DB connection ──────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        """Return the per-thread SQLite connection, creating it if needed."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._local.conn = conn
        return self._local.conn

    # ── Schema + migration ────────────────────────────────────────────────────

    def _init_db(self):
        """Create schema if not present, migrate from JSON if DB is empty."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                summary      TEXT NOT NULL,
                category     TEXT NOT NULL DEFAULT 'general',
                priority     INTEGER NOT NULL DEFAULT 3,
                timestamp    REAL NOT NULL,
                cluster_id   TEXT,
                message      TEXT NOT NULL DEFAULT '',
                response     TEXT NOT NULL DEFAULT '',
                memory_type  TEXT NOT NULL DEFAULT 'episodic',
                source_type  TEXT NOT NULL DEFAULT 'conversation',
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS clusters (
                id           TEXT PRIMARY KEY,
                theme        TEXT NOT NULL,
                priority     INTEGER NOT NULL DEFAULT 3,
                last_update  REAL NOT NULL,
                torque_mass  REAL NOT NULL DEFAULT 0.0,
                center_vector BLOB NOT NULL,
                message_refs  TEXT NOT NULL DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

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
            );
        """)

        # sqlite-vec virtual table (extension already loaded by _get_conn)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                memory_id TEXT,
                embedding float[768] distance_metric=cosine
            )
        """)

        # Memory lanes — directed graph of memory relationships
        # link_type: 'semantic' (bidirectional) or 'causal' (one-way: source→research)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_links (
                source_id  TEXT NOT NULL,
                target_id  TEXT NOT NULL,
                weight     REAL NOT NULL DEFAULT 1.0,
                link_type  TEXT NOT NULL DEFAULT 'semantic',
                created_at REAL NOT NULL,
                PRIMARY KEY (source_id, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_links_source ON memory_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id);
        """)

        # FTS5 index for keyword-based (BM25) seed retrieval — standalone table.
        # No content= option: FTS5 content tables can't be queried in subquery
        # context (sqlite3 aliases the vtab as T and loses UNINDEXED column names).
        # Drop + recreate on every startup so schema changes are always applied.
        conn.execute("DROP TABLE IF EXISTS fts_memories")
        conn.execute("""
            CREATE VIRTUAL TABLE fts_memories USING fts5(
                memory_id UNINDEXED,
                summary,
                tokenize='porter ascii'
            )
        """)
        # Rebuild FTS index from memories table on every startup.
        # Fast (<1ms for hundreds of rows); ensures FTS is always in sync.
        conn.execute("""
            INSERT INTO fts_memories(memory_id, summary)
            SELECT id, summary FROM memories
        """)

        # Migrate existing DBs — add new columns if they don't exist yet
        for col, definition in [
            ("memory_type",  "TEXT NOT NULL DEFAULT 'episodic'"),
            ("source_type",  "TEXT NOT NULL DEFAULT 'conversation'"),
            ("access_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_accessed","REAL NOT NULL DEFAULT 0"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # column already exists

        conn.commit()

        # One-time migration from memory.json
        old_json = self.data_dir / "memory.json"
        if old_json.exists():
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if count == 0:
                self._migrate_from_json(old_json)

    def _migrate_from_json(self, json_file: Path):
        """Import all records from the old memory.json into SQLite."""
        print(f"[Memory] Migrating {json_file} → memory.db ...")
        try:
            with open(json_file) as f:
                data = json.load(f)

            conn = self._get_conn()
            migrated_memories = 0
            migrated_clusters = 0

            for entry_data in data.get("messages", {}).values():
                embedding = entry_data.get("embedding", [])
                conn.execute("""
                    INSERT OR IGNORE INTO memories
                    (id, summary, category, priority, timestamp, cluster_id, message, response)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry_data["id"],
                    entry_data.get("summary", ""),
                    entry_data.get("category", "general"),
                    entry_data.get("priority", 3),
                    entry_data.get("timestamp", time.time()),
                    entry_data.get("cluster_id"),
                    entry_data.get("message", "")[:500],
                    entry_data.get("response", "")[:500],
                ))
                if embedding:
                    emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    conn.execute(
                        "DELETE FROM vec_memories WHERE memory_id=?", (entry_data["id"],)
                    )
                    conn.execute(
                        "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                        (entry_data["id"], emb_bytes),
                    )
                migrated_memories += 1

            for cluster_data in data.get("clusters", {}).values():
                center = cluster_data.get("center_vector", [])
                if not center:
                    continue
                center_bytes = np.array(center, dtype=np.float32).tobytes()
                conn.execute("""
                    INSERT OR IGNORE INTO clusters
                    (id, theme, priority, last_update, torque_mass, center_vector, message_refs)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    cluster_data["id"],
                    cluster_data.get("theme", "General"),
                    cluster_data.get("priority", 3),
                    cluster_data.get("last_update", time.time()),
                    cluster_data.get("torque_mass", 0.0),
                    center_bytes,
                    json.dumps(cluster_data.get("message_refs", [])),
                ))
                migrated_clusters += 1

            user_profile = data.get("user_profile")
            if user_profile:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('user_profile', ?)",
                    (user_profile,),
                )

            conn.commit()
            print(f"[Memory] Migrated {migrated_memories} memories, {migrated_clusters} clusters")
            json_file.rename(json_file.with_suffix(".json.migrated"))

        except Exception as e:
            print(f"[Memory] Migration error: {e}")
            import traceback; traceback.print_exc()

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_from_db(self):
        """Populate in-memory caches from DB. Embeddings stay on disk."""
        conn = self._get_conn()

        enc = self.encryptor
        _d = (lambda s: enc.decrypt_str(s)) if (enc and enc.config.enabled) else (lambda s: s)

        for row in conn.execute(
            "SELECT id, summary, category, priority, timestamp, cluster_id, message, response, "
            "memory_type, source_type, access_count, last_accessed FROM memories"
        ):
            self.messages[row["id"]] = MemoryEntry(
                id=row["id"],
                summary=_d(row["summary"]),
                category=row["category"],
                priority=row["priority"],
                timestamp=row["timestamp"],
                cluster_id=row["cluster_id"],
                message=_d(row["message"]),
                response=_d(row["response"]),
                embedding=None,
                memory_type=row["memory_type"] or "episodic",
                source_type=row["source_type"] or "conversation",
                access_count=row["access_count"] or 0,
                last_accessed=row["last_accessed"] or 0.0,
            )

        for row in conn.execute(
            "SELECT id, theme, priority, last_update, torque_mass, center_vector, message_refs FROM clusters"
        ):
            center_vector = (
                np.frombuffer(row["center_vector"], dtype=np.float32).tolist()
                if row["center_vector"] else []
            )
            self.clusters[row["id"]] = MemoryCluster(
                id=row["id"],
                theme=_d(row["theme"]),
                center_vector=center_vector,
                message_refs=json.loads(row["message_refs"]),
                priority=row["priority"],
                last_update=row["last_update"],
                torque_mass=row["torque_mass"],
            )

        row = conn.execute("SELECT value FROM meta WHERE key='user_profile'").fetchone()
        self._user_profile = _d(row["value"]) if row and row["value"] else None

        row = conn.execute("SELECT value FROM meta WHERE key='stats'").fetchone()
        if row:
            try:
                self.stats.update(json.loads(row["value"]))
            except Exception:
                pass

        self.stats["total_messages"] = len(self.messages)
        self.stats["active_clusters"] = len(self.clusters)

        unclustered = sum(
            1 for m in self.messages.values()
            if m.cluster_id is None or m.cluster_id not in self.clusters
        )
        if unclustered > 0:
            print(f"[Memory] {unclustered} unclustered memories found")

        print(f"[Memory] Loaded {len(self.messages)} memories, {len(self.clusters)} clusters from DB")

    # ── Targeted DB writes ────────────────────────────────────────────────────

    def _save_memory(self, entry: MemoryEntry, embedding: list[float]):
        """Insert a new MemoryEntry + its embedding into DB."""
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        enc = self.encryptor
        _e = (lambda s: enc.encrypt_str(s)) if (enc and enc.config.enabled) else (lambda s: s)
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO memories
                (id, summary, category, priority, timestamp, cluster_id, message, response,
                 memory_type, source_type, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, _e(entry.summary), entry.category, entry.priority,
                entry.timestamp, entry.cluster_id,
                _e(entry.message[:500]), _e(entry.response[:500]),
                entry.memory_type, entry.source_type,
                entry.access_count, entry.last_accessed,
            ))
            conn.execute("DELETE FROM vec_memories WHERE memory_id=?", (entry.id,))
            conn.execute(
                "INSERT INTO vec_memories(memory_id, embedding) VALUES (?, ?)",
                (entry.id, emb_bytes),
            )
            conn.execute(
                "INSERT OR REPLACE INTO fts_memories(memory_id, summary) VALUES (?, ?)",
                (entry.id, entry.summary),  # FTS needs plain text, not encrypted
            )
            conn.commit()

    def _save_cluster(self, cluster: MemoryCluster):
        """Upsert a MemoryCluster to DB."""
        center_bytes = np.array(cluster.center_vector, dtype=np.float32).tobytes()
        enc = self.encryptor
        theme = enc.encrypt_str(cluster.theme) if (enc and enc.config.enabled) else cluster.theme
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("""
                INSERT OR REPLACE INTO clusters
                (id, theme, priority, last_update, torque_mass, center_vector, message_refs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster.id, theme, cluster.priority,
                cluster.last_update, cluster.torque_mass,
                center_bytes, json.dumps(cluster.message_refs),
            ))
            conn.commit()

    def _delete_memory(self, memory_id: str):
        """Remove a memory from both memories and vec_memories tables."""
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            conn.execute("DELETE FROM vec_memories WHERE memory_id=?", (memory_id,))
            conn.execute("DELETE FROM fts_memories WHERE memory_id=?", (memory_id,))
            conn.execute(
                "DELETE FROM memory_links WHERE source_id=? OR target_id=?",
                (memory_id, memory_id),
            )
            conn.commit()

    # ── Memory lane building ───────────────────────────────────────────────

    def _build_semantic_links(
        self, new_id: str, embedding: list[float], new_universe: str = "episodic"
    ) -> int:
        """
        Find the K nearest neighbors of a new memory and create semantic lanes.

        Same universe  → bidirectional (both rows).
        Cross-universe → one-way: new_id→neighbor only.
          This lets PPR walk FROM the new memory INTO other universes when semantically
          relevant, but prevents neighbors from "pulling back" across universes during
          their own PPR expansions.

        Returns the number of lanes created (not rows).
        """
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT vm.memory_id, vm.distance, m.memory_type
            FROM vec_memories vm
            JOIN memories m ON m.id = vm.memory_id
            WHERE vm.embedding MATCH ?
            AND vm.memory_id != ?
            ORDER BY vm.distance
            LIMIT ?
        """, (emb_bytes, new_id, LANE_MAX_K + 2)).fetchall()

        links = []
        now = time.time()
        for row in rows:
            sim = 1.0 - row["distance"]
            if LANE_MIN_SIM <= sim < LANE_MAX_SIM:
                neighbor_universe = row["memory_type"] or "episodic"
                same_universe = (neighbor_universe == new_universe)
                # Always add new_id → neighbor
                links.append((new_id, row["memory_id"], sim, "semantic", now))
                if same_universe:
                    # Add reverse only within the same universe
                    links.append((row["memory_id"], new_id, sim, "semantic", now))

        if links:
            with self._write_lock:
                conn.executemany("""
                    INSERT OR REPLACE INTO memory_links
                    (source_id, target_id, weight, link_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, links)
                conn.commit()
            self._lanes_ready = True

        # Count distinct lanes (a cross-universe link = 1 lane = 1 row)
        return len(links)

    def _build_causal_links(
        self, research_memory_id: str, source_memory_ids: list[str]
    ) -> int:
        """
        Build one-way causal lanes from source memories to a research finding.

        These are INCOMPLETE trade lanes: you can fly FROM the source memory TO
        the research finding, but there is no lane back. The research finding
        exists BECAUSE of the source, not the other way around.

        In PPR: A[research_idx][source_idx] is set; A[source_idx][research_idx] is not.
        This directional asymmetry is correct by construction — only source→research
        rows are inserted.
        """
        valid_sources = [
            sid for sid in source_memory_ids if sid in self.messages
        ]
        if not valid_sources or research_memory_id not in self.messages:
            return 0

        now = time.time()
        links = [
            (sid, research_memory_id, CAUSAL_WEIGHT, "causal", now)
            for sid in valid_sources
        ]

        with self._write_lock:
            conn = self._get_conn()
            conn.executemany("""
                INSERT OR REPLACE INTO memory_links
                (source_id, target_id, weight, link_type, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, links)
            conn.commit()

        return len(links)

    def _build_sequential_links(self, prop_ids: list[str]) -> int:
        """
        Build bidirectional sequential lanes between consecutive proposition IDs.

        These lanes encode reading order within a document section — when any
        proposition is retrieved, PPR walks both forward and backward along the
        chain, surfacing prerequisites and follow-ons automatically.

        Weight is fixed at 0.65 — strong enough to be surfaced by PPR but lower
        than high-similarity semantic lanes so they don't swamp unrelated queries.
        """
        if len(prop_ids) < 2:
            return 0

        now = time.time()
        links = []
        for i in range(len(prop_ids) - 1):
            a, b = prop_ids[i], prop_ids[i + 1]
            if a not in self.messages or b not in self.messages:
                continue
            links.append((a, b, 0.65, "sequential", now))
            links.append((b, a, 0.65, "sequential", now))

        if links:
            with self._write_lock:
                conn = self._get_conn()
                conn.executemany("""
                    INSERT OR REPLACE INTO memory_links
                    (source_id, target_id, weight, link_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, links)
                conn.commit()
            self._lanes_ready = True

        return len(links) // 2

    def _build_hierarchical_links(
        self, prop_ids: list[str], section_id: str, doc_id: str
    ) -> int:
        """
        Build hierarchical lanes: each proposition → its section summary,
        and the section summary → the document summary.

        These lanes let PPR walk up from a retrieved proposition to its section
        context, and from a section to the full document summary — similar to
        RAPTOR's tree traversal but driven by PPR rather than a separate index.

        Weights:
          prop → section  : 0.80  (high — section provides essential context)
          section → doc   : 0.70  (meaningful — document summary provides top-level framing)
        Both directions are stored so retrieval can start at any level.
        """
        now = time.time()
        links = []

        for pid in prop_ids:
            if pid not in self.messages or section_id not in self.messages:
                continue
            links.append((pid,        section_id, 0.80, "hierarchical", now))
            links.append((section_id, pid,        0.80, "hierarchical", now))

        if section_id in self.messages and doc_id in self.messages:
            links.append((section_id, doc_id, 0.70, "hierarchical", now))
            links.append((doc_id, section_id, 0.70, "hierarchical", now))

        if links:
            with self._write_lock:
                conn = self._get_conn()
                conn.executemany("""
                    INSERT OR REPLACE INTO memory_links
                    (source_id, target_id, weight, link_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, links)
                conn.commit()
            self._lanes_ready = True

        return len(links)

    async def store_document(self, ingestion_result: dict) -> dict:
        """
        Store a fully-extracted document (from ingest.ingest_document) into memory.

        Stores:
          - One doc-level summary memory
          - One section-level summary memory per section
          - All proposition memories

        Then builds:
          - Semantic lanes (existing _build_semantic_links, called inside _store_fact)
          - Sequential lanes between propositions within each section
          - Hierarchical lanes: proposition → section → document

        Returns a summary dict with counts.
        """
        doc_name = ingestion_result.get("doc_name", "document")
        doc_summary = ingestion_result.get("doc_summary", "").strip()
        sections = ingestion_result.get("sections", [])

        user_msg = f"[Ingested document: {doc_name}]"
        # Keep embedder loaded for the full batch — one load instead of N load/unload cycles
        self._suppress_embedder_unload = True

        # Store document-level summary
        doc_id = None
        if doc_summary:
            doc_id = await self._store_fact(
                {"summary": doc_summary[:200], "category": "document", "priority": 4,
                 "memory_type": "declarative", "source_type": "ingestion"},
                user_message=user_msg,
            )

        total_props = 0
        section_count = 0

        for section in sections:
            sec_name = section.get("name", "")
            sec_summary = section.get("summary", "").strip()
            propositions = section.get("propositions", [])

            # Store section-level summary
            section_id = None
            if sec_summary:
                sec_label = f"{doc_name} — {sec_name}: {sec_summary}"
                section_id = await self._store_fact(
                    {"summary": sec_label[:200], "category": "document_section", "priority": 3,
                     "memory_type": "declarative", "source_type": "ingestion"},
                    user_message=user_msg,
                )
                section_count += 1

            # Store propositions and collect their IDs for link building
            prop_ids = []
            for prop in propositions:
                pid = await self._store_fact(
                    {**prop, "memory_type": "declarative", "source_type": "ingestion"},
                    user_message=user_msg,
                )
                if pid:
                    prop_ids.append(pid)
                    total_props += 1

            # Build sequential lanes within this section
            if prop_ids:
                self._build_sequential_links(prop_ids)

            # Build hierarchical lanes for this section
            if prop_ids and section_id and doc_id:
                self._build_hierarchical_links(prop_ids, section_id, doc_id)
            elif prop_ids and doc_id and not section_id:
                # No section summary — link props directly to doc
                self._build_hierarchical_links(prop_ids, doc_id, doc_id)

        self._suppress_embedder_unload = False
        self._maybe_unload_embedder()

        return {
            "doc_name": doc_name,
            "doc_id": doc_id,
            "sections": section_count,
            "propositions": total_props,
        }

    def _save_profile(self, profile: str):
        """Persist the compact user profile to meta table."""
        enc = self.encryptor
        stored = enc.encrypt_str(profile) if (enc and enc.config.enabled) else profile
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('user_profile', ?)",
                (stored,),
            )
            conn.commit()

    def _save_stats(self):
        """Persist the stats dict to meta table."""
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('stats', ?)",
                (json.dumps(self.stats),),
            )
            conn.commit()

    # ── Embedding fetch helpers ───────────────────────────────────────────────

    def _fetch_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Fetch a single memory's embedding from DB as float32 numpy array."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM vec_memories WHERE memory_id=?", (memory_id,)
        ).fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def _fetch_all_embeddings(
        self, memory_ids: list[str]
    ) -> tuple[list[str], np.ndarray]:
        """
        Batch-fetch embeddings for the given memory IDs.
        Returns (ids_found, embeddings_array). Only IDs with stored embeddings are returned.
        """
        if not memory_ids:
            return [], np.array([])
        conn = self._get_conn()
        placeholders = ",".join("?" * len(memory_ids))
        rows = conn.execute(
            f"SELECT memory_id, embedding FROM vec_memories WHERE memory_id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        ids_out, embs_out = [], []
        for row in rows:
            ids_out.append(row[0])
            embs_out.append(np.frombuffer(row[1], dtype=np.float32))
        return ids_out, np.array(embs_out) if embs_out else np.array([])

    # ── Decay / health ────────────────────────────────────────────────────────

    def _calculate_retention(self, entry: MemoryEntry, current_time: float) -> float:
        age_hours = (current_time - entry.timestamp) / 3600
        base_rate = DECAY_RATES.get(entry.priority, DECAY_RATES[1])
        universe_mult = UNIVERSE_DECAY_MULTIPLIERS.get(entry.memory_type, 1.0)
        # Frequently-accessed memories resist decay — kept separate from Torque
        # cluster geometry so high-access memories don't warp semantic distances.
        access_resistance = 1.0 / (1.0 + entry.access_count)
        effective_rate = base_rate * universe_mult * access_resistance
        return math.exp(-effective_rate * age_hours)

    def _calculate_cluster_health(self, cluster: MemoryCluster, current_time: float) -> float:
        valid_refs = [r for r in cluster.message_refs if r in self.messages]
        if not valid_refs:
            return 0.0
        avg_decay = sum(
            self._calculate_retention(self.messages[r], current_time) for r in valid_refs
        ) / len(valid_refs)
        size_bonus = min(len(valid_refs) / 10, 1.0) * 0.1
        recent = 0.2 if any(
            (current_time - self.messages[r].timestamp) < 86400 for r in valid_refs
        ) else 0.0
        mass_bonus = min(cluster.torque_mass / 100, 0.2) if cluster.torque_mass > 0 else 0.0
        return avg_decay + size_bonus + recent + mass_bonus

    def _cleanup(self):
        """Remove decayed memories and empty/unhealthy clusters from DB and caches."""
        current_time = time.time()

        to_delete = [
            mid for mid, entry in self.messages.items()
            if self._calculate_retention(entry, current_time) <= MESSAGE_RETENTION_THRESHOLD
        ]
        for mid in to_delete:
            entry = self.messages.pop(mid)
            self._delete_memory(mid)
            if entry.cluster_id and entry.cluster_id in self.clusters:
                self.clusters[entry.cluster_id].message_refs = [
                    r for r in self.clusters[entry.cluster_id].message_refs if r != mid
                ]

        to_remove_clusters = []
        for cid, cluster in self.clusters.items():
            cluster.message_refs = [r for r in cluster.message_refs if r in self.messages]
            if (not cluster.message_refs or
                    self._calculate_cluster_health(cluster, current_time) <= CLUSTER_HEALTH_THRESHOLD):
                to_remove_clusters.append(cid)

        for cid in to_remove_clusters:
            self.clusters.pop(cid)
            with self._write_lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM clusters WHERE id=?", (cid,))
                conn.commit()

        # Recompute centroids for clusters that lost members
        if to_delete:
            for cluster in self.clusters.values():
                valid_ids = [r for r in cluster.message_refs if r in self.messages]
                if valid_ids and len(valid_ids) < len(cluster.message_refs) + len(to_delete):
                    _, embs = self._fetch_all_embeddings(valid_ids)
                    if len(embs) > 0:
                        cluster.center_vector = np.mean(embs, axis=0).tolist()
                        cluster.torque_mass = float(len(valid_ids))
                        self._save_cluster(cluster)

        self.stats["total_messages"] = len(self.messages)
        self.stats["active_clusters"] = len(self.clusters)
        self.stats["last_cleanup"] = current_time

        if to_delete or to_remove_clusters:
            print(f"[Memory] Cleanup: removed {len(to_delete)} memories, {len(to_remove_clusters)} clusters")

    # ── Cosine similarity (used for cluster scoring, not for search) ──────────

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        a, b = np.array(vec1), np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # ── Cluster theme generation ──────────────────────────────────────────────

    def _generate_cluster_theme(self, memory_ids: list[str]) -> str:
        """Sync theme generation (used by assign_unclustered_memories)."""
        summaries = [
            f"- {self.messages[mid].summary}"
            for mid in memory_ids[:10] if mid in self.messages
        ]
        if not summaries:
            return "General"
        prompt = CLUSTER_THEME_PROMPT.format(memories="\n".join(summaries))
        try:
            response = httpx.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            parsed = json.loads(_llm_json_content(response))
            return parsed.get("theme", "General")[:100]
        except Exception as e:
            print(f"[Memory] Theme generation error: {e}")
            best = max(
                (self.messages[mid] for mid in memory_ids if mid in self.messages),
                key=lambda m: (m.priority, m.timestamp),
                default=None,
            )
            return best.summary[:50] if best else "General"

    async def _generate_cluster_theme_async(self, memory_ids: list[str], http_client) -> str:
        """Async theme generation (used by run_torque_clustering_async)."""
        summaries = [
            f"- {self.messages[mid].summary}"
            for mid in memory_ids[:10] if mid in self.messages
        ]
        if not summaries:
            return "General"
        prompt = CLUSTER_THEME_PROMPT.format(memories="\n".join(summaries))
        try:
            response = await http_client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            parsed = json.loads(_llm_json_content(response))
            return parsed.get("theme", "General")[:100]
        except Exception as e:
            print(f"[Memory] Async theme error: {e}")
            best = max(
                (self.messages[mid] for mid in memory_ids if mid in self.messages),
                key=lambda m: (m.priority, m.timestamp),
                default=None,
            )
            return best.summary[:50] if best else "General"

    # ── Incremental cluster assignment ────────────────────────────────────────

    def assign_unclustered_memories(self) -> dict:
        """
        Assign unclustered memories to existing clusters.
        Fetches only the unclustered embeddings from DB — not all embeddings.
        """
        if not self.clusters:
            return {"status": "skipped", "reason": "no_clusters_exist"}

        unclustered_ids = [
            mid for mid, m in self.messages.items()
            if m.cluster_id is None or m.cluster_id not in self.clusters
        ]
        if not unclustered_ids:
            return {"status": "skipped", "reason": "no_unclustered_memories"}

        print(f"[Memory] Assigning {len(unclustered_ids)} unclustered memories...")

        ids, embeddings = self._fetch_all_embeddings(unclustered_ids)
        if len(ids) == 0:
            return {"status": "skipped", "reason": "no_embeddings_found"}

        assigned_count = 0
        orphans: list[tuple[str, np.ndarray]] = []  # (memory_id, embedding)

        updated_clusters: set[str] = set()

        for mid, embedding in zip(ids, embeddings):
            entry = self.messages[mid]
            best_cluster = None
            best_sim = NEW_CLUSTER_SIMILARITY_THRESHOLD

            for cluster in self.clusters.values():
                sim = self._cosine_similarity(embedding.tolist(), cluster.center_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

            if best_cluster:
                entry.cluster_id = best_cluster.id
                if mid not in best_cluster.message_refs:
                    best_cluster.message_refs.append(mid)
                best_cluster.last_update = time.time()
                updated_clusters.add(best_cluster.id)
                assigned_count += 1
            else:
                orphans.append((mid, embedding))

        # Batch update memories in DB
        with self._write_lock:
            conn = self._get_conn()
            for mid in ids:
                entry = self.messages.get(mid)
                if entry and entry.cluster_id:
                    conn.execute(
                        "UPDATE memories SET cluster_id=? WHERE id=?",
                        (entry.cluster_id, mid),
                    )
            conn.commit()

        # Recompute centroids for affected clusters
        for cid in updated_clusters:
            cluster = self.clusters[cid]
            valid_ids = [r for r in cluster.message_refs if r in self.messages]
            _, cluster_embs = self._fetch_all_embeddings(valid_ids)
            if len(cluster_embs) > 0:
                cluster.center_vector = np.mean(cluster_embs, axis=0).tolist()
                cluster.torque_mass = float(len(valid_ids))
            self._save_cluster(cluster)

        new_clusters_created = 0
        if len(orphans) >= 2:
            current_time = time.time()
            cluster_id = f"tc_{int(current_time * 1000)}_new"
            memory_ids = [mid for mid, _ in orphans]
            orphan_embs = np.array([emb for _, emb in orphans])
            centroid = np.mean(orphan_embs, axis=0).tolist()
            max_priority = max(self.messages[mid].priority for mid in memory_ids if mid in self.messages)
            theme = self._generate_cluster_theme(memory_ids)

            new_cluster = MemoryCluster(
                id=cluster_id,
                theme=theme,
                center_vector=centroid,
                message_refs=memory_ids,
                priority=max_priority,
                last_update=time.time(),
                torque_mass=float(len(memory_ids)),
            )
            self.clusters[cluster_id] = new_cluster
            self._save_cluster(new_cluster)

            with self._write_lock:
                conn = self._get_conn()
                for mid, _ in orphans:
                    self.messages[mid].cluster_id = cluster_id
                    conn.execute(
                        "UPDATE memories SET cluster_id=? WHERE id=?", (cluster_id, mid)
                    )
                conn.commit()

            assigned_count += len(orphans)
            new_clusters_created = 1

        elif len(orphans) == 1:
            mid, embedding = orphans[0]
            best_cluster = max(
                self.clusters.values(),
                key=lambda c: self._cosine_similarity(embedding.tolist(), c.center_vector),
            )
            self.messages[mid].cluster_id = best_cluster.id
            if mid not in best_cluster.message_refs:
                best_cluster.message_refs.append(mid)
            self._save_cluster(best_cluster)
            with self._write_lock:
                conn = self._get_conn()
                conn.execute(
                    "UPDATE memories SET cluster_id=? WHERE id=?", (best_cluster.id, mid)
                )
                conn.commit()
            assigned_count += 1

        self._clustering_dirty = False
        self.stats["active_clusters"] = len(self.clusters)

        result = {"status": "success", "assigned": assigned_count, "new_clusters": new_clusters_created}
        print(f"[Memory] Incremental assignment: {result}")
        return result

    # ── Context retrieval ─────────────────────────────────────────────────────

    def _embedder_loaded(self) -> bool:
        return self.embedder.is_loaded()

    def _maybe_unload_embedder(self):
        """Unload the embedding model after a write if configured to do so.
        Controlled by config key 'embedder_unload_after_write' (default True).
        Keeps idle RAM ~280MB; write spike is ~540MB for ~1-2s only.
        Suppressed during bulk operations (e.g. store_document) via _suppress_embedder_unload.
        """
        if getattr(self, "_suppress_embedder_unload", False):
            return
        try:
            config_file = DEFAULT_CONFIG_DIR / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    if not json.load(f).get("embedder_unload_after_write", True):
                        return
        except Exception:
            pass
        self.embedder.unload()

    def _embed_cached(self, text: str) -> list[float]:
        """Return embedding for text, using a short-TTL LRU cache to avoid
        redundant sentence-transformer calls on repeated queries."""
        key = text[:200]  # truncate long queries for the cache key
        now = time.time()
        entry = self._embed_cache.get(key)
        if entry and (now - entry[1]) < self._EMBED_CACHE_TTL:
            return entry[0]
        embedding = self.embedder.embed(text)
        # Evict oldest entry if at capacity
        if len(self._embed_cache) >= self._EMBED_CACHE_MAX:
            oldest = min(self._embed_cache, key=lambda k: self._embed_cache[k][1])
            del self._embed_cache[oldest]
        self._embed_cache[key] = (embedding, now)
        return embedding

    def get_relevant_context(self, query: str, max_clusters: int = 2) -> str:
        """
        Three-universe retrieval with dynamic context allocation.

        1. Classify query → personal | factual | procedural | balanced
        2. Seed: FTS5 BM25 + vec, merged with RRF
        3. PPR expand through the lane graph
        4. Split results by universe, apply CONTEXT_BIAS budget caps
        5. Increment access_count for recalled memories
        6. Format with per-universe sections

        Falls back to legacy cluster-centroid scoring if no seeds or lanes exist.
        """
        if not self.messages:
            return ""

        conn = self._get_conn()

        # Step 1: classify query → dynamic budget split
        query_type = _classify_query(query)
        ep_bias, dec_bias, proc_bias = CONTEXT_BIAS[query_type]
        total_budget = 12
        budgets = {
            "episodic":    max(1, round(ep_bias   * total_budget)),
            "declarative": max(1, round(dec_bias  * total_budget)),
            "procedural":  max(1, round(proc_bias * total_budget)),
        }

        # Step 2a: FTS5 BM25 keyword seeds
        fts_query = " OR ".join(
            f'"{w}"' for w in query.split()[:8] if len(w) > 2
        )
        fts_seeds: list[str] = []
        if fts_query:
            try:
                fts_rows = conn.execute("""
                    SELECT memory_id, rank FROM fts_memories
                    WHERE fts_memories MATCH ?
                    ORDER BY rank
                    LIMIT 5
                """, (fts_query,)).fetchall()
                fts_seeds = [r["memory_id"] for r in fts_rows if r["memory_id"] in self.messages]
            except Exception:
                pass  # malformed query — safe to ignore

        # Step 2b: vec seeds — only if embedder already resident in RAM
        vec_seeds: list[str] = []
        query_embedding: Optional[list[float]] = None
        if self._embedder_loaded():
            query_embedding = self._embed_cached(query)
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            vec_rows = conn.execute("""
                SELECT memory_id FROM vec_memories
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT 3
            """, (emb_bytes,)).fetchall()
            vec_seeds = [r["memory_id"] for r in vec_rows if r["memory_id"] in self.messages]

        # Step 2c: merge with RRF
        seed_ids = _rrf_merge(fts_seeds, vec_seeds, k=60)[:5]

        if not seed_ids:
            # Fall back: force-load embedder for a one-off vec search
            query_embedding = query_embedding or self._embed_cached(query)
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            seed_rows = conn.execute("""
                SELECT memory_id FROM vec_memories
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT 5
            """, (emb_bytes,)).fetchall()
            seed_ids = [r["memory_id"] for r in seed_rows if r["memory_id"] in self.messages]

        if not seed_ids:
            query_embedding = query_embedding or self._embed_cached(query)
            return self._legacy_cluster_context(query_embedding, max_clusters)

        # Step 3: PPR expand — fetch slightly more than total_budget to give each
        # universe enough candidates before applying per-universe caps.
        if self._lanes_ready is None:
            self._lanes_ready = conn.execute(
                "SELECT COUNT(*) FROM memory_links LIMIT 1"
            ).fetchone()[0] > 0

        if self._lanes_ready:
            ranked = self._ppr_expand(seed_ids, top_k=total_budget + 6)
        else:
            ranked = [(sid, 1.0) for sid in seed_ids]

        if not ranked:
            query_embedding = query_embedding or self._embed_cached(query)
            return self._legacy_cluster_context(query_embedding, max_clusters)

        # Step 4: split by universe, apply budget caps
        universe_buckets: dict[str, list[tuple[str, float]]] = {
            "episodic": [], "declarative": [], "procedural": []
        }
        for memory_id, score in ranked:
            entry = self.messages.get(memory_id)
            if not entry:
                continue
            u = entry.memory_type if entry.memory_type in VALID_UNIVERSES else "episodic"
            bucket = universe_buckets[u]
            if len(bucket) < budgets[u]:
                bucket.append((memory_id, score))

        # Step 5: increment access_count for all recalled memories
        recalled_ids = [mid for bucket in universe_buckets.values() for mid, _ in bucket]
        if recalled_ids:
            now = time.time()
            with self._write_lock:
                conn.executemany(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    [(now, mid) for mid in recalled_ids],
                )
                conn.commit()
            for mid in recalled_ids:
                entry = self.messages.get(mid)
                if entry:
                    entry.access_count += 1
                    entry.last_accessed = now

        # Step 6: format with per-universe sections
        section_labels = {
            "episodic":    "Personal",
            "declarative": "Knowledge",
            "procedural":  "Patterns",
        }
        context_parts = ["[MEMORY CONTEXT]"]
        has_content = False

        for universe in ("episodic", "declarative", "procedural"):
            bucket = universe_buckets[universe]
            if not bucket:
                continue
            has_content = True
            context_parts.append(f"\n## {section_labels[universe]}")
            seen_themes: set[str] = set()
            for memory_id, _score in bucket:
                entry = self.messages.get(memory_id)
                if not entry:
                    continue
                cluster = self.clusters.get(entry.cluster_id or "") if entry.cluster_id else None
                theme = cluster.theme if cluster else entry.category.title()
                if theme not in seen_themes:
                    context_parts.append(f"  [{theme}]")
                    seen_themes.add(theme)
                prefix = "[Research] " if entry.category == "research" else ""
                context_parts.append(f"  - {prefix}{entry.summary}")

        if not has_content:
            query_embedding = query_embedding or self._embed_cached(query)
            return self._legacy_cluster_context(query_embedding, max_clusters)

        context_parts.append("\n[Use this context naturally in your responses when relevant.]")
        return "\n".join(context_parts)

    def _ppr_expand(
        self,
        seed_ids: list[str],
        damping: float = 0.85,
        iterations: int = 15,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Personalized PageRank through the lane graph from seed memories.

        Semantic lanes are bidirectional — both directions stored in memory_links,
        so activation flows freely either way.
        Causal lanes are one-way — only source→research rows exist, so activation
        flows from user memories toward research findings but not back via the lane.
        This asymmetry is load-bearing: it models the Freelancer incomplete trade
        lane. You can fly source→research, but to return you fly under your own power.

        Returns list of (memory_id, score) sorted by score descending.
        """
        if not seed_ids:
            return []

        conn = self._get_conn()
        placeholders = ",".join("?" * len(seed_ids))

        # Step 1: get 1-hop neighborhood of seeds.
        # Semantic links store both A→B and B→A, so querying either side captures
        # the full undirected neighborhood. Causal links only store source→research,
        # so only the forward direction appears — the one-way lane is preserved.
        hop1_rows = conn.execute(f"""
            SELECT DISTINCT target_id FROM memory_links WHERE source_id IN ({placeholders})
            UNION
            SELECT DISTINCT source_id FROM memory_links WHERE target_id IN ({placeholders})
        """, seed_ids + seed_ids).fetchall()

        neighborhood = set(seed_ids)
        for r in hop1_rows:
            if r[0] in self.messages:
                neighborhood.add(r[0])

        nb_list = list(neighborhood)
        nb_placeholders = ",".join("?" * len(nb_list))

        # Step 2: fetch all outgoing edges within the neighborhood.
        # For semantic links both directions are present so PPR flows freely.
        # For causal links only source→research is present — activation flows
        # from the user memory toward the research finding, not back.
        rows = conn.execute(f"""
            SELECT source_id, target_id, weight FROM memory_links
            WHERE source_id IN ({nb_placeholders})
        """, nb_list).fetchall()

        if not rows:
            return [(sid, 1.0) for sid in seed_ids if sid in self.messages]

        # Build adjacency list — only include still-alive memories
        nodes_set: set[str] = set(seed_ids)
        adj: dict[str, list[tuple[str, float]]] = {}

        for source, target, weight in rows:
            if source in self.messages and target in self.messages:
                nodes_set.add(source)
                nodes_set.add(target)
                adj.setdefault(source, []).append((target, weight))

        if len(nodes_set) < 2:
            return [(sid, 1.0) for sid in seed_ids if sid in self.messages]

        node_list = list(nodes_set)
        idx = {n: i for i, n in enumerate(node_list)}
        n = len(node_list)

        # Column-stochastic transition matrix A[to][from]
        # Semantic links: both directions stored → A[b][a] and A[a][b] both set
        # Causal links: only source→research stored → A[research][source] set,
        #               A[source][research] = 0 (the one-way lane — correct by design)
        A = np.zeros((n, n))
        for source, neighbors in adj.items():
            si = idx[source]
            total = sum(w for _, w in neighbors)
            if total == 0:
                continue
            for target, w in neighbors:
                ti = idx.get(target)
                if ti is not None:
                    A[ti][si] = w / total

        # Personalization vector: uniform over seeds
        p = np.zeros(n)
        valid_seeds = [s for s in seed_ids if s in idx]
        if not valid_seeds:
            return []
        for sid in valid_seeds:
            p[idx[sid]] = 1.0 / len(valid_seeds)

        # Power iteration: r = (1 - d)*p + d*A*r
        r = p.copy()
        for _ in range(iterations):
            r = (1.0 - damping) * p + damping * (A @ r)

        results = [
            (node_list[i], float(r[i]))
            for i in range(n)
            if node_list[i] in self.messages and r[i] > 0
        ]
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def _legacy_cluster_context(
        self, query_embedding: list[float], max_clusters: int = 2
    ) -> str:
        """
        Original cluster-centroid retrieval — used as fallback before lanes exist.
        Scores clusters by cosine similarity of query to centroid (all in RAM).
        """
        if not self.clusters:
            return ""

        cluster_scores = []
        for cluster in self.clusters.values():
            similarity = self._cosine_similarity(query_embedding, cluster.center_vector)
            mass_weight = 1.0 + (cluster.torque_mass / 100.0) * 0.2
            weighted_score = similarity * mass_weight
            if similarity > 0.4:
                cluster_scores.append((cluster, weighted_score))

        cluster_scores.sort(key=lambda x: -x[1])
        top_clusters = cluster_scores[:max_clusters]

        if not top_clusters:
            return ""

        context_parts = ["[MEMORY CONTEXT - Things you know about this user:]"]
        for cluster, _ in top_clusters:
            valid_refs = [r for r in cluster.message_refs if r in self.messages]
            if not valid_refs:
                continue
            sorted_refs = sorted(
                valid_refs,
                key=lambda r: (self.messages[r].priority, self.messages[r].timestamp),
                reverse=True,
            )[:3]
            context_parts.append(f"\n## {cluster.theme}")
            for ref in sorted_refs:
                context_parts.append(f"- {self.messages[ref].summary}")

        context_parts.append("\n[Use this context naturally in your responses when relevant.]")
        return "\n".join(context_parts)

    # ── User profile ──────────────────────────────────────────────────────────

    def get_user_profile(self) -> str:
        return self._user_profile or ""

    def is_profile_dirty(self) -> bool:
        return self._profile_dirty

    async def regenerate_user_profile(self, http_client) -> str:
        high_priority = [m for m in self.messages.values() if m.priority >= 4]
        if not high_priority:
            return ""

        facts = "\n".join(f"- {m.summary}" for m in high_priority)
        prompt = USER_PROFILE_PROMPT.format(facts=facts)
        try:
            response = await http_client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 80,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            parsed = json.loads(_llm_json_content(response))
            profile = parsed.get("profile", "").strip()
        except Exception as e:
            print(f"[Memory] Profile generation error: {e}")
            profile = " | ".join(
                m.summary for m in sorted(high_priority, key=lambda m: -m.priority)[:6]
            )

        self._user_profile = profile
        self._profile_dirty = False
        self._save_profile(profile)
        print(f"[Memory] User profile updated: {profile[:80]}")
        return profile

    # ── Chat-time entry point ─────────────────────────────────────────────────

    async def queue_conversation(
        self,
        user_message: str,
        assistant_response: str,
        http_client,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        """
        Fast chat-time path: append to pending queue + extract durable facts immediately.
        Full batch extraction (cross-conversation patterns) deferred to sleep/introspection cycle.
        """
        try:
            self._append_pending({
                "id": f"pending_{time.time_ns()}",
                "timestamp": time.time(),
                "user": user_message,
                "assistant": assistant_response,
            })

            facts = await self._extract_durable_facts(user_message, assistant_response, http_client)
            stored = 0
            for fact in facts:
                mem_id = await self._store_fact(fact, user_message, assistant_response)
                if mem_id:
                    stored += 1

            # Track dedup stats for saturation detection
            self._recent_exchange_stats.append({"extracted": len(facts), "stored": stored})
            if len(self._recent_exchange_stats) > 10:
                self._recent_exchange_stats = self._recent_exchange_stats[-10:]

            if stored:
                print(f"[Memory] Stored {stored} durable fact(s) immediately")
                if on_status:
                    on_status(f"[Memory: {stored} fact(s) stored]")
            else:
                if on_status:
                    on_status("[Memory: queued for sleep processing]")

        except Exception as e:
            import traceback
            print(f"[Memory] Error in queue_conversation: {e}")
            traceback.print_exc()

    def saturation_rate(self, window: int = 3) -> float:
        """
        Fraction of recently extracted facts that were rejected as duplicates.
        Returns 0.0 if there is not enough data or too few facts to measure.
        A rate near 1.0 means the topic is mined out — nothing new to learn.
        """
        recent = self._recent_exchange_stats[-window:]
        if len(recent) < window:
            return 0.0
        total_extracted = sum(s["extracted"] for s in recent)
        total_stored = sum(s["stored"] for s in recent)
        if total_extracted < 3:  # too few facts to draw a conclusion
            return 0.0
        return 1.0 - (total_stored / total_extracted)

    def _append_pending(self, entry: dict):
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry)
            enc = self.encryptor
            if enc and enc.config.enabled:
                line = enc.encrypt_str(line)
            with open(self.pending_file, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[Memory] Pending write error: {e}")

    async def _extract_durable_facts(
        self, user_message: str, assistant_response: str, http_client
    ) -> list[dict]:
        """
        Extract durable facts from a single exchange. Fails safe (empty list on error).
        Filters out transient system metrics; returns priority 3-5 only.
        """
        prompt = DURABLE_FACT_PROMPT.format(
            user_message=user_message[:500],
            assistant_response=assistant_response[:800],
        )
        try:
            response = await http_client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 220,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=25.0,
            )
            parsed = json.loads(_llm_json_content(response))
            return [f for f in parsed.get("facts", []) if int(f.get("priority", 0)) >= 3]
        except Exception as e:
            print(f"[Memory] Durable fact extraction error: {e}")
            return []

    async def _store_fact(self, fact: dict, user_message: str = "", assistant_response: str = "") -> Optional[str]:
        """
        Store a single extracted fact.
        Dedup check via sqlite-vec MATCH (single query, no full-scan loop).
        Embedding stored in vec_memories; metadata in memories table.
        Returns the new memory ID, or None if the fact was deduped/skipped.
        """
        try:
            priority = max(1, min(5, int(fact.get("priority", 1))))
        except (TypeError, ValueError):
            priority = 2
        summary = fact.get("summary", "").strip()[:200]
        category = fact.get("category", "general")

        # Universe routing — source_type overrides LLM classification for ingestion/research
        raw_type = fact.get("memory_type", "episodic")
        source_type = fact.get("source_type", "conversation")
        if source_type in ("ingestion", "research"):
            memory_type = "declarative"
        elif raw_type in VALID_UNIVERSES:
            memory_type = raw_type
        else:
            memory_type = "episodic"

        if not summary or priority < 2:
            return None

        embedding = self.embedder.embed(summary)
        emb_bytes = np.array(embedding, dtype=np.float32).tobytes()

        # Dedup via sqlite-vec: if nearest neighbor distance < 0.08 → similarity > 0.92 → skip
        conn = self._get_conn()
        row = conn.execute("""
            SELECT memory_id, distance
            FROM vec_memories
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT 1
        """, (emb_bytes,)).fetchone()

        if row and row[1] < DEDUP_DISTANCE:
            print(f"[Memory] Skipping duplicate fact: {summary[:60]}")
            return None

        current_time = time.time()
        entry_id = f"msg_{time.time_ns()}"

        entry = MemoryEntry(
            id=entry_id,
            message=user_message[:500],
            response=assistant_response[:500],
            timestamp=current_time,
            priority=priority,
            category=category,
            summary=summary,
            cluster_id=None,
            embedding=None,
            memory_type=memory_type,
            source_type=source_type,
            access_count=0,
            last_accessed=0.0,
        )
        self.messages[entry_id] = entry
        self._save_memory(entry, embedding)
        self._build_semantic_links(entry_id, embedding, new_universe=memory_type)

        self._clustering_dirty = True
        if priority >= 4:
            self._profile_dirty = True

        # Temporary cluster assignment until next torque pass
        if self.clusters:
            best_cluster = None
            best_sim = 0.4
            for cluster in self.clusters.values():
                sim = self._cosine_similarity(embedding, cluster.center_vector)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster
            if best_cluster:
                entry.cluster_id = best_cluster.id
                if entry_id not in best_cluster.message_refs:
                    best_cluster.message_refs.append(entry_id)
                best_cluster.last_update = current_time
                if priority > best_cluster.priority:
                    best_cluster.priority = priority
                with self._write_lock:
                    conn = self._get_conn()
                    conn.execute(
                        "UPDATE memories SET cluster_id=? WHERE id=?",
                        (best_cluster.id, entry_id),
                    )
                    conn.commit()
                self._save_cluster(best_cluster)

        self.stats["total_messages"] = len(self.messages)
        self._maybe_unload_embedder()
        return entry_id

    async def add_research_finding(
        self,
        topic: str,
        fact: str,
        confidence: float,
        http_client=None,
        on_status=None,
        source_memory_ids: list[str] = None,
    ):
        """
        Store a research finding into the memory system.

        Research facts are treated as lower-priority memories (priority 2-3)
        so they don't outweigh things the user told us directly (priority 4-5).
        They're tagged category="research" so they're distinguishable.
        Dedup is handled by _store_fact via vector similarity.

        source_memory_ids: IDs of the user memories that triggered this research.
        One-way causal lanes are built from each source to this finding — the
        Freelancer incomplete trade lane. You can fly source→research via the lane,
        but to return you fly under your own power (semantic links only).
        """
        priority = 3 if confidence >= 0.8 else 2
        fact_dict = {
            "summary": fact,
            "priority": priority,
            "category": "research",
            "memory_type": "declarative",
            "source_type": "research",
        }
        memory_id = await self._store_fact(fact_dict, user_message=f"[Research] {topic}", assistant_response=fact)
        if memory_id and source_memory_ids:
            lane_count = self._build_causal_links(memory_id, source_memory_ids)
            if lane_count:
                print(f"[Memory] Built {lane_count} causal lane(s) to research finding")
        if on_status:
            on_status(f"[Memory] Stored research finding: {fact[:60]}...")
        return memory_id

    # ── Conflict resolution ───────────────────────────────────────────────────

    async def backfill_links(self) -> int:
        """
        Background job: build semantic lanes for memories that pre-date the lane system.

        Uses a persistent meta flag ('lanes_backfilled') so it won't re-run after
        completion, even if new memories have lanes before the first nightly cycle.
        On each call, finds memories with no outgoing links (true orphans) and only
        processes those — so a partial backfill is safely resumable.
        """
        conn = self._get_conn()

        # Persistent completion flag — set only after all orphans are resolved.
        if conn.execute("SELECT 1 FROM meta WHERE key='lanes_backfilled'").fetchone():
            return 0

        if not self.messages:
            return 0

        # Find memories that have no outgoing link rows (haven't been linked yet).
        orphan_rows = conn.execute("""
            SELECT m.id FROM memories m
            LEFT JOIN memory_links ml ON ml.source_id = m.id
            WHERE ml.source_id IS NULL
        """).fetchall()

        orphans = [r[0] for r in orphan_rows if r[0] in self.messages]

        if not orphans:
            # All memories already have links — mark done.
            with self._write_lock:
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('lanes_backfilled', '1')"
                )
                conn.commit()
            return 0

        before = conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
        print(f"[Memory] Building memory lanes for {len(orphans)} un-linked memories...")
        ids, embeddings = self._fetch_all_embeddings(orphans)

        for i, (mid, emb) in enumerate(zip(ids, embeddings)):
            universe = getattr(self.messages.get(mid), "memory_type", "episodic")
            self._build_semantic_links(mid, emb.tolist(), new_universe=universe)
            # Yield to the event loop every 10 memories to avoid blocking asyncio.
            if i % 10 == 9:
                await asyncio.sleep(0)

        after = conn.execute("SELECT COUNT(*) FROM memory_links").fetchone()[0]
        new_lanes = after - before
        print(f"[Memory] Backfill complete: {new_lanes} new lanes built across {len(ids)} memories")
        self._viz_cache = None  # Force viz to re-render with lanes

        with self._write_lock:
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('lanes_backfilled', '1')"
            )
            conn.commit()

        self._maybe_unload_embedder()
        return new_lanes

    def resolve_conflicts(self) -> dict:
        """
        Time-based conflict resolution — runs during introspection, no LLM needed.

        For each memory, queries sqlite-vec for similar memories in the conflict
        distance range [CONFLICT_DIST_NEAR, CONFLICT_DIST_FAR] (= similarity 0.75–0.92).
        Same-category pairs in this range: older fact is pruned, newer is kept.

        Much faster than the old O(n²) Python loop for large memory sets.
        """
        if len(self.messages) < 2:
            return {"pruned": 0, "details": []}

        to_prune: set[str] = set()
        pruned_details: list[str] = []
        conn = self._get_conn()

        for entry_id, entry in list(self.messages.items()):
            if entry_id in to_prune:
                continue

            emb_row = conn.execute(
                "SELECT embedding FROM vec_memories WHERE memory_id=?", (entry_id,)
            ).fetchone()
            if not emb_row or not emb_row[0]:
                continue

            emb_bytes = emb_row[0]

            # Fetch nearest neighbors in conflict range (up to 20 candidates)
            neighbors = conn.execute("""
                SELECT memory_id, distance
                FROM vec_memories
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT 20
            """, (emb_bytes,)).fetchall()

            for neighbor_id, distance in neighbors:
                if neighbor_id == entry_id or neighbor_id in to_prune:
                    continue
                if distance < CONFLICT_DIST_NEAR:
                    continue  # Duplicate — handled by dedup in _store_fact
                if distance > CONFLICT_DIST_FAR:
                    break     # Beyond conflict range (results are ordered by distance)

                neighbor = self.messages.get(neighbor_id)
                if not neighbor or neighbor.category != entry.category:
                    continue

                # Same category, conflict range — keep newer
                older = entry if entry.timestamp < neighbor.timestamp else neighbor
                newer = neighbor if older is entry else entry
                to_prune.add(older.id)
                pruned_details.append(f"'{older.summary[:50]}' → '{newer.summary[:50]}'")
                print(f"[Memory] Conflict: '{older.summary[:60]}' superseded by '{newer.summary[:60]}'")

        for mid in to_prune:
            entry = self.messages.pop(mid)
            if entry.cluster_id and entry.cluster_id in self.clusters:
                cluster = self.clusters[entry.cluster_id]
                cluster.message_refs = [r for r in cluster.message_refs if r != mid]
                self._save_cluster(cluster)
            self._delete_memory(mid)

        if to_prune:
            self.stats["total_messages"] = len(self.messages)
            self._clustering_dirty = True

        return {"pruned": len(to_prune), "details": pruned_details}

    # ── Sleep-time extraction ─────────────────────────────────────────────────

    async def process_pending_conversations(self, http_client) -> dict:
        """
        Two-pass sleep processor:
        Pass 1 — group pending conversations by topic.
        Pass 2 — extract multi-fact array per group.
        """
        if not self.pending_file.exists():
            return {"status": "skipped", "reason": "no_pending_file"}

        enc = self.encryptor
        pending = []
        with open(self.pending_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        if enc and enc.config.enabled:
                            line = enc.decrypt_str(line)
                        pending.append(json.loads(line))
                    except Exception:
                        pass

        if not pending:
            return {"status": "skipped", "reason": "empty_pending"}

        print(f"[Memory] Processing {len(pending)} pending conversations...")

        one_liners = "\n".join(
            f"[{i}] {p['user'][:80].replace(chr(10), ' ')}"
            for i, p in enumerate(pending)
        )
        groups = await self._group_conversations(one_liners, len(pending), http_client)

        total_facts = 0
        extraction_failures = 0
        for group_indices in groups:
            group_convs = [pending[i] for i in group_indices if i < len(pending)]
            if not group_convs:
                continue
            conv_text = "\n\n".join(
                f"[Conversation {j + 1}]\nUser: {c['user'][:400]}\nAssistant: {c['assistant'][:400]}"
                for j, c in enumerate(group_convs)
            )
            facts = await self._extract_facts_from_group(conv_text, http_client)
            if not facts:
                extraction_failures += 1
            ref_conv = group_convs[0]
            for fact in facts:
                await self._store_fact(fact, ref_conv["user"], ref_conv["assistant"])
                total_facts += 1

        # Only clear the pending file if extraction didn't completely fail
        # (all groups failed = server was down; keep conversations for next run)
        if extraction_failures < len(groups):
            self.pending_file.unlink()
        else:
            print("[Memory] All extraction groups failed — keeping pending file for next run.")

        result = {
            "status": "success",
            "conversations_processed": len(pending),
            "groups": len(groups),
            "facts_stored": total_facts,
        }
        print(f"[Memory] Pending processing complete: {result}")
        return result

    async def _group_conversations(
        self, one_liners: str, count: int, http_client
    ) -> list[list[int]]:
        if count <= 3:
            return [list(range(count))]
        prompt = GROUPING_PROMPT.format(conversations=one_liners)
        try:
            response = await http_client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 300,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            parsed = json.loads(_llm_json_content(response))
            groups = parsed.get("groups", [])
            covered = {i for g in groups for i in g}
            missing = [i for i in range(count) if i not in covered]
            if missing:
                groups.append(missing)
            return groups
        except Exception as e:
            print(f"[Memory] Grouping error: {e}")
            return [list(range(i, min(i + 4, count))) for i in range(0, count, 4)]

    async def _extract_facts_from_group(
        self, conv_text: str, http_client
    ) -> list[dict]:
        prompt = EXTRACTION_PROMPT.format(conversations=conv_text)

        def safe_priority(f):
            try:
                return int(f.get("priority", 1))
            except (TypeError, ValueError):
                return 1

        last_error = None
        for attempt in range(3):
            try:
                if attempt > 0:
                    await asyncio.sleep(15 * attempt)  # 15s then 30s before retries
                response = await http_client.post(
                    f"{self.llm_url}/v1/chat/completions",
                    json={
                        "model": self.llm_model_id,
                        "messages": [{"role": "user", "content": prompt + " /no_think"}],
                        "max_tokens": 500,
                        "temperature": 0.1,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=45.0,
                )
                parsed = json.loads(_llm_json_content(response))
                facts = parsed.get("facts", [])
                return [f for f in facts if safe_priority(f) >= 2][:8]
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                if any(x in err_str for x in ("disconnect", "connect", "transport", "network", "reset")):
                    print(f"[Memory] Extraction attempt {attempt + 1} failed (server busy): {e}")
                    continue
                break  # non-transport error — don't retry

        print(f"[Memory] Extraction error: {last_error}")
        return []

    # ── Torque Clustering ─────────────────────────────────────────────────────

    def _clustering_cpu_work(self, target_clusters: int, memory_id_subset: list = None) -> dict:
        """
        Pure CPU clustering work — safe to run in a thread pool executor.
        Fetches embeddings from DB, runs TorqueClustering, returns raw cluster data.
        No state mutation. Pass memory_id_subset to cluster only a specific set of facts.
        """
        from torque_clustering import TorqueClustering

        ids_to_cluster = memory_id_subset if memory_id_subset is not None else list(self.messages.keys())
        memory_ids, embeddings = self._fetch_all_embeddings(ids_to_cluster)

        if len(memory_ids) < MIN_MEMORIES_FOR_CLUSTERING:
            return {}

        print(f"[Memory] CPU clustering: {len(memory_ids)} memories → target {target_clusters} clusters")

        adjustment_factor = self._load_adjustment_factor()
        distance_matrix = cdist(embeddings, embeddings, metric="cosine")
        result = TorqueClustering(
            distance_matrix,
            K=target_clusters,
            isnoise=False,
            isfig=False,
            auto_config=False,
            use_std_adjustment=True,
            adjustment_factor=adjustment_factor,
        )

        cluster_labels = result[0]
        cluster_labels_with_noise = result[1]
        labels = cluster_labels_with_noise if len(cluster_labels_with_noise) > 0 else cluster_labels

        raw_clusters = {}
        for label in np.unique(labels):
            if label < 0:
                continue
            indices = np.where(labels == label)[0]
            ids = [memory_ids[i] for i in indices]
            if not ids:
                continue
            raw_clusters[int(label)] = {
                "memory_ids": ids,
                "centroid": np.mean(embeddings[indices], axis=0).tolist(),
                "mass": float(len(indices)),
            }

        return raw_clusters

    def _log_cluster_health(self):
        """Log cluster size distribution and warn if adjustment_factor tuning is needed."""
        if not self.clusters:
            return
        sizes = [len(c.message_refs) for c in self.clusters.values()]
        avg = sum(sizes) / len(sizes)
        factor = self._load_adjustment_factor()
        print(
            f"[Clustering] {len(self.messages)} facts → {len(sizes)} clusters | "
            f"avg {avg:.1f}/cluster | min {min(sizes)}, max {max(sizes)} | "
            f"adjustment_factor={factor}"
        )
        oversized = sum(1 for s in sizes if s > MAX_CLUSTER_SIZE)
        tiny = sum(1 for s in sizes if s < 3)
        if oversized:
            print(
                f"[Clustering] WARNING: {oversized} cluster(s) have >{MAX_CLUSTER_SIZE} facts "
                f"— consider increasing clustering_adjustment_factor in config.json (currently {factor})"
            )
        if tiny:
            print(
                f"[Clustering] WARNING: {tiny} cluster(s) have <3 facts "
                f"— consider decreasing clustering_adjustment_factor in config.json (currently {factor})"
            )

    async def _split_oversized_clusters(self, http_client) -> dict:
        """
        Split any cluster exceeding MAX_CLUSTER_SIZE by re-clustering only its facts.
        Much cheaper than a full recluster — O(k × cluster_size²) where k = oversized clusters.
        """
        from torque_clustering import TorqueClustering

        oversized = [
            c for c in self.clusters.values()
            if len(c.message_refs) > MAX_CLUSTER_SIZE
        ]
        if not oversized:
            return {"status": "skipped", "reason": "no_oversized_clusters"}

        print(f"[Memory] Splitting {len(oversized)} oversized cluster(s)...")
        loop = asyncio.get_event_loop()
        net_new = 0

        for cluster in oversized:
            memory_ids = [mid for mid in cluster.message_refs if mid in self.messages]
            if len(memory_ids) < MIN_MEMORIES_FOR_CLUSTERING:
                continue

            target_sub = max(2, len(memory_ids) // 4)

            try:
                raw = await loop.run_in_executor(
                    None, self._clustering_cpu_work, target_sub, memory_ids
                )
            except Exception as e:
                print(f"[Memory] Split error for cluster {cluster.id}: {e}")
                continue

            if not raw or len(raw) <= 1:
                continue

            current_time = time.time()
            new_sub_clusters: dict[str, MemoryCluster] = {}
            for label, cdata in raw.items():
                sub_ids = cdata["memory_ids"]
                if not sub_ids:
                    continue
                theme = await self._generate_cluster_theme_async(sub_ids, http_client)
                cid = f"tc_{int(current_time * 1000)}_{label}"
                sub_cluster = MemoryCluster(
                    id=cid,
                    theme=theme,
                    center_vector=cdata["centroid"],
                    message_refs=sub_ids,
                    priority=max(
                        self.messages[mid].priority
                        for mid in sub_ids
                        if mid in self.messages
                    ),
                    last_update=current_time,
                    torque_mass=cdata["mass"],
                )
                new_sub_clusters[cid] = sub_cluster

            if len(new_sub_clusters) <= 1:
                continue

            with self._write_lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM clusters WHERE id=?", (cluster.id,))
                for sub in new_sub_clusters.values():
                    center_bytes = np.array(sub.center_vector, dtype=np.float32).tobytes()
                    conn.execute(
                        "INSERT INTO clusters (id, theme, priority, last_update, torque_mass, center_vector, message_refs) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (sub.id, sub.theme, sub.priority, sub.last_update,
                         sub.torque_mass, center_bytes, json.dumps(sub.message_refs)),
                    )
                    for mid in sub.message_refs:
                        if mid in self.messages:
                            self.messages[mid].cluster_id = sub.id
                        conn.execute("UPDATE memories SET cluster_id=? WHERE id=?", (sub.id, mid))
                conn.commit()

            del self.clusters[cluster.id]
            self.clusters.update(new_sub_clusters)
            net_new += len(new_sub_clusters) - 1

        self.stats["active_clusters"] = len(self.clusters)
        self.stats["last_clustering"] = time.time()
        self._save_stats()
        self._log_cluster_health()

        return {
            "status": "success",
            "clusters_split": len(oversized),
            "net_new_clusters": net_new,
            "total_clusters": len(self.clusters),
        }

    async def run_torque_clustering_async(self, http_client, mode: str = "auto") -> dict:
        """
        Async clustering entry point. Mode controls the strategy:
          "auto"        — smart dispatch:
                          1. Full recluster if FULL_RECLUSTER_DAYS have passed since last full run
                          2. Split oversized clusters (>MAX_CLUSTER_SIZE facts) if any exist
                          3. Incremental: assign any unclustered facts to existing clusters
          "full"        — force a complete O(n²) recluster regardless of schedule
          "incremental" — only assign unclustered facts (fastest, no restructuring)

        CPU work (distance matrix + TorqueClustering) runs in thread pool executor.
        Theme generation runs async after the executor returns.
        """
        if len(self.messages) < MIN_MEMORIES_FOR_CLUSTERING:
            return {"status": "skipped", "reason": "not_enough_memories"}

        # ── Resolve "auto" mode ──────────────────────────────────────────────
        if mode == "auto":
            days_since_full = (time.time() - self.stats.get("last_full_recluster", 0)) / 86400
            if days_since_full >= FULL_RECLUSTER_DAYS:
                print(f"[Memory] Auto mode: {days_since_full:.1f} days since full recluster — running full")
                mode = "full"
            elif any(len(c.message_refs) > MAX_CLUSTER_SIZE for c in self.clusters.values()):
                print("[Memory] Auto mode: oversized clusters found — splitting")
                self._clustering_in_progress = True
                try:
                    return await self._split_oversized_clusters(http_client)
                finally:
                    self._clustering_in_progress = False
            else:
                self._clustering_in_progress = True
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.assign_unclustered_memories
                    )
                    self._log_cluster_health()
                    return result
                finally:
                    self._clustering_in_progress = False

        # ── Incremental-only mode ────────────────────────────────────────────
        if mode == "incremental":
            self._clustering_in_progress = True
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.assign_unclustered_memories
                )
                self._log_cluster_health()
                return result
            finally:
                self._clustering_in_progress = False

        # ── Full recluster (mode == "full") ──────────────────────────────────
        self._clustering_in_progress = True
        try:
            target_clusters = max(2, len(self.messages) // 4)
            print(f"[Memory] Running full Torque Clustering on {len(self.messages)} memories...")

            try:
                loop = asyncio.get_event_loop()
                raw_clusters = await loop.run_in_executor(
                    None, self._clustering_cpu_work, target_clusters
                )
            except Exception as e:
                print(f"[Memory] Torque clustering CPU work failed: {e}")
                import traceback; traceback.print_exc()
                return {"status": "error", "reason": str(e)}

            if not raw_clusters:
                return {"status": "skipped", "reason": "clustering_returned_no_clusters"}

            print(f"[Memory] CPU clustering found {len(raw_clusters)} clusters, generating themes...")

            current_time = time.time()
            new_clusters: dict[str, MemoryCluster] = {}

            for label, cluster_data in raw_clusters.items():
                theme = await self._generate_cluster_theme_async(cluster_data["memory_ids"], http_client)
                cluster_id = f"tc_{int(current_time * 1000)}_{label}"
                cluster = MemoryCluster(
                    id=cluster_id,
                    theme=theme,
                    center_vector=cluster_data["centroid"],
                    message_refs=cluster_data["memory_ids"],
                    priority=max(
                        self.messages[mid].priority
                        for mid in cluster_data["memory_ids"]
                        if mid in self.messages
                    ),
                    last_update=current_time,
                    torque_mass=cluster_data["mass"],
                )
                new_clusters[cluster_id] = cluster

                with self._write_lock:
                    conn = self._get_conn()
                    for mid in cluster_data["memory_ids"]:
                        if mid in self.messages:
                            self.messages[mid].cluster_id = cluster_id
                            conn.execute(
                                "UPDATE memories SET cluster_id=? WHERE id=?",
                                (cluster_id, mid),
                            )
                    conn.commit()

            # Replace all clusters in DB atomically
            with self._write_lock:
                conn = self._get_conn()
                conn.execute("DELETE FROM clusters")
                for cluster in new_clusters.values():
                    center_bytes = np.array(cluster.center_vector, dtype=np.float32).tobytes()
                    conn.execute("""
                        INSERT INTO clusters
                        (id, theme, priority, last_update, torque_mass, center_vector, message_refs)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        cluster.id, cluster.theme, cluster.priority,
                        cluster.last_update, cluster.torque_mass,
                        center_bytes, json.dumps(cluster.message_refs),
                    ))
                conn.commit()

            old_count = len(self.clusters)
            self.clusters = new_clusters
            self._clustering_dirty = False
            self.stats["active_clusters"] = len(self.clusters)
            self.stats["last_clustering"] = current_time
            self.stats["last_full_recluster"] = current_time
            self._save_stats()
            self._log_cluster_health()

            result = {
                "status": "success",
                "old_clusters": old_count,
                "new_clusters": len(new_clusters),
                "memories_clustered": sum(
                    len(c["memory_ids"]) for c in raw_clusters.values()
                ),
            }
            print(f"[Memory] Full recluster complete: {result}")
            return result

        finally:
            self._clustering_in_progress = False

    def needs_reclustering(self) -> bool:
        return self._clustering_dirty

    # ── Stats + visualization ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "total_messages": len(self.messages),
            "active_clusters": len(self.clusters),
            "memory_file": str(self.db_file),
            "file_size_kb": self.db_file.stat().st_size // 1024 if self.db_file.exists() else 0,
            "needs_reclustering": self._clustering_dirty,
            "last_clustering": self.stats.get("last_clustering", 0),
            "clustering_method": "torque_clustering",
            "storage": "sqlite+sqlite-vec",
        }

    # ── Visualization data ────────────────────────────────────────────────────

    def get_viz_data(self) -> dict:
        """
        Return 3D-projected memory positions for the star-map visualizer.
        Uses UMAP (umap-learn) if available, falls back to PCA via numpy SVD.
        Result is cached and only recomputed when memory or cluster count changes.
        Returns {"memories": [...], "clusters": [...]}.
        """
        from collections import defaultdict

        if not self.messages:
            return {"memories": [], "clusters": []}

        # Return cached result if nothing has changed
        current_key = (len(self.messages), len(self.clusters))
        if self._viz_cache is not None and current_key == self._viz_cache_key:
            return self._viz_cache

        all_ids = list(self.messages.keys())
        ids_found, embeddings = self._fetch_all_embeddings(all_ids)

        if len(ids_found) < 2:
            return {"memories": [], "clusters": []}

        coords = self._reduce_to_3d(embeddings)
        coord_map = {mid: coords[i] for i, mid in enumerate(ids_found)}

        memories_out = []
        for mid in ids_found:
            entry = self.messages.get(mid)
            if not entry:
                continue
            x, y, z = coord_map[mid]
            memories_out.append({
                "id": mid,
                "summary": entry.summary,
                "category": entry.category or "",
                "priority": entry.priority,
                "cluster_id": entry.cluster_id or "",
                "timestamp": entry.timestamp,
                "memory_type": entry.memory_type or "episodic",
                "source_type": entry.source_type or "conversation",
                "access_count": entry.access_count,
                "x": float(x),
                "y": float(y),
                "z": float(z),
            })

        # Compute cluster centroids from member projected positions
        cluster_member_coords: dict = defaultdict(list)
        for m in memories_out:
            if m["cluster_id"]:
                cluster_member_coords[m["cluster_id"]].append(
                    (m["x"], m["y"], m["z"])
                )

        clusters_out = []
        for cid, cluster in self.clusters.items():
            positions = cluster_member_coords.get(cid, [])
            if positions:
                cx = sum(p[0] for p in positions) / len(positions)
                cy = sum(p[1] for p in positions) / len(positions)
                cz = sum(p[2] for p in positions) / len(positions)
            else:
                cx, cy, cz = 0.0, 0.0, 0.0
            clusters_out.append({
                "id": cid,
                "theme": cluster.theme or "Unknown",
                "mass": float(cluster.torque_mass),
                "count": len(positions),
                "cx": cx,
                "cy": cy,
                "cz": cz,
            })

        # Fetch the strongest lanes for rendering.
        # Only include lanes where both endpoints have projected positions.
        # Cap at 200 to keep the payload manageable.
        memory_id_set = {m["id"] for m in memories_out}
        conn = self._get_conn()
        lane_rows = conn.execute("""
            SELECT source_id, target_id, weight, link_type
            FROM memory_links
            WHERE weight > 0.60
            ORDER BY weight DESC
            LIMIT 200
        """).fetchall()

        lanes_out = [
            {
                "source": r["source_id"],
                "target": r["target_id"],
                "weight": float(r["weight"]),
                "type": r["link_type"],
            }
            for r in lane_rows
            if r["source_id"] in memory_id_set and r["target_id"] in memory_id_set
        ]

        result = {"memories": memories_out, "clusters": clusters_out, "lanes": lanes_out}
        self._viz_cache = result
        self._viz_cache_key = current_key
        return result

    def _reduce_to_3d(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce high-dimensional embeddings to 3D coordinates.

        Tries UMAP first (best cluster separation); falls back to PCA via
        numpy SVD if umap-learn is not installed.
        Output is normalised to roughly [-50, 50] on each axis.
        """
        n = len(embeddings)

        if n < 4:
            # Too few points — spread on a small circle
            coords = np.zeros((n, 3))
            for i in range(n):
                angle = (i / max(n - 1, 1)) * 2 * math.pi
                coords[i] = [math.cos(angle) * 5, math.sin(angle) * 5, 0.0]
            return coords

        try:
            import umap as _umap
            reducer = _umap.UMAP(
                n_components=3,
                n_neighbors=min(15, n - 1),
                min_dist=0.15,
                random_state=42,
                verbose=False,
                low_memory=True,
            )
            coords = reducer.fit_transform(embeddings)
            print(f"[MemoryViz] UMAP 3D: {n} memories")
        except ImportError:
            mean = embeddings.mean(axis=0)
            centered = embeddings - mean
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            coords = centered @ Vt[:3].T
            print(f"[MemoryViz] PCA 3D: {n} memories (install umap-learn for better results)")

        # Normalise each axis to [-50, 50]
        for axis in range(3):
            col = coords[:, axis]
            span = col.max() - col.min()
            if span > 0:
                coords[:, axis] = (col - col.min()) / span * 100.0 - 50.0

        return coords

    # ── Export / Import / Delete ──────────────────────────────────────────────

    def export_all(self) -> dict:
        """Export all memories, clusters, and user profile as a portable JSON dict."""
        conn = self._get_conn()

        memories = []
        for row in conn.execute(
            "SELECT id, summary, category, priority, timestamp, cluster_id, message, response, "
            "memory_type, source_type, access_count, last_accessed FROM memories ORDER BY timestamp"
        ):
            memories.append(dict(row))

        clusters = []
        for row in conn.execute(
            "SELECT id, theme, priority, last_update, torque_mass, message_refs FROM clusters"
        ):
            c = dict(row)
            c["message_refs"] = json.loads(c.get("message_refs", "[]"))
            clusters.append(c)

        return {
            "version": "mk12",
            "exported_at": time.time(),
            "user_profile": self.get_user_profile() or "",
            "memories": memories,
            "clusters": clusters,
        }

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory and update its cluster's message_refs."""
        if memory_id not in self.messages:
            return False

        entry = self.messages[memory_id]
        cluster_id = entry.cluster_id

        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))

            if cluster_id and cluster_id in self.clusters:
                cluster = self.clusters[cluster_id]
                cluster.message_refs = [r for r in cluster.message_refs if r != memory_id]
                if cluster.message_refs:
                    conn.execute(
                        "UPDATE clusters SET message_refs = ? WHERE id = ?",
                        (json.dumps(cluster.message_refs), cluster_id),
                    )
                else:
                    conn.execute("DELETE FROM clusters WHERE id = ?", (cluster_id,))
                    del self.clusters[cluster_id]

            conn.commit()
            del self.messages[memory_id]

        self.stats["total_messages"] = max(0, self.stats["total_messages"] - 1)
        self.stats["active_clusters"] = len(self.clusters)
        return True

    def delete_all_memories(self):
        """Wipe all memories, clusters, embeddings, user profile, and experience log from the DB."""
        with self._write_lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM memories")
            conn.execute("DELETE FROM clusters")
            conn.execute("DELETE FROM vec_memories")
            conn.execute("DELETE FROM meta WHERE key = 'user_profile'")
            conn.execute("DELETE FROM experience_log")
            conn.commit()
            self.messages.clear()
            self.clusters.clear()

        self._user_profile = None
        self.stats["total_messages"] = 0
        self.stats["active_clusters"] = 0

    async def import_all(self, data: dict) -> dict:
        """
        Import memories from an export dict. Clears existing memories first.
        Re-generates embeddings from summaries (blocking — may take a moment for
        large exports). Clusters are dropped and rebuilt on the next nightly cycle.
        """
        self.delete_all_memories()

        memories = data.get("memories", [])
        user_profile = data.get("user_profile", "")

        valid = [m for m in memories if m.get("summary")]
        if not valid:
            return {"imported": 0, "failed": 0}

        summaries = [m["summary"] for m in valid]

        # Re-embed all summaries in a thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        try:
            embeddings = await loop.run_in_executor(None, self.embedder.embed_batch, summaries)
        except Exception as e:
            return {"error": f"Embedding failed: {e}", "imported": 0, "failed": len(valid)}

        imported = 0
        failed = 0

        with self._write_lock:
            conn = self._get_conn()
            for m, emb in zip(valid, embeddings):
                try:
                    mem_id = m["id"]
                    conn.execute(
                        "INSERT OR IGNORE INTO memories "
                        "(id, summary, category, priority, timestamp, cluster_id, message, response, "
                        "memory_type, source_type, access_count, last_accessed) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            mem_id,
                            m["summary"],
                            m.get("category", "general"),
                            m.get("priority", 3),
                            m.get("timestamp", time.time()),
                            None,  # cluster_id — rebuilt by next introspection
                            m.get("message", "")[:500],
                            m.get("response", "")[:500],
                            m.get("memory_type", "episodic"),
                            m.get("source_type", "conversation"),
                            m.get("access_count", 0),
                            m.get("last_accessed", 0.0),
                        ),
                    )
                    emb_bytes = np.array(emb, dtype=np.float32).tobytes()
                    conn.execute(
                        "INSERT OR IGNORE INTO vec_memories (memory_id, embedding) VALUES (?, ?)",
                        (mem_id, emb_bytes),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO fts_memories(memory_id, summary) VALUES (?, ?)",
                        (mem_id, m["summary"]),
                    )
                    self.messages[mem_id] = MemoryEntry(
                        id=mem_id,
                        summary=m["summary"],
                        category=m.get("category", "general"),
                        priority=m.get("priority", 3),
                        timestamp=m.get("timestamp", time.time()),
                        cluster_id=None,
                        message=m.get("message", "")[:500],
                        response=m.get("response", "")[:500],
                    )
                    imported += 1
                except Exception:
                    failed += 1

            if user_profile:
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('user_profile', ?)",
                    (user_profile,),
                )
                self._user_profile = user_profile

            conn.commit()

        self.stats["total_messages"] = imported
        self._clustering_dirty = True  # Trigger re-clustering on next nightly cycle
        return {"imported": imported, "failed": failed}

    def get_visualization(self) -> str:
        if not self.clusters:
            return "No memories stored yet."
        lines = [
            "=== MEMORY SYSTEM (Torque Clustering / SQLite) ===",
            f"Messages: {len(self.messages)} | Clusters: {len(self.clusters)}",
            f"Needs re-clustering: {self._clustering_dirty}",
            "=" * 50,
        ]
        for i, cluster in enumerate(
            sorted(self.clusters.values(), key=lambda c: c.torque_mass, reverse=True), 1
        ):
            valid_refs = [r for r in cluster.message_refs if r in self.messages]
            lines.append(f"\nCluster {i}: {cluster.theme}")
            lines.append(f"  Priority: {cluster.priority} | Messages: {len(valid_refs)} | Mass: {cluster.torque_mass:.1f}")
            for ref in valid_refs[-2:]:
                lines.append(f"  • {self.messages[ref].summary[:60]}...")
        return "\n".join(lines)

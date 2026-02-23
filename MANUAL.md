# 3AM — Complete Technical Manual

**Version 1.1.0** | Self-Evolving Local AI Assistant

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Configuration](#installation--configuration)
4. [Data Storage Layout](#data-storage-layout)
5. [Module Reference](#module-reference)
   - [server.py — Web Server & API](#serverpy--web-server--api)
   - [auth.py — Authentication System](#authpy--authentication-system)
   - [data_security.py — Encryption](#data_securitypy--encryption)
   - [memory.py — Memory System](#memorypy--memory-system)
   - [introspection.py — Background Processing](#introspectionpy--background-processing)
   - [decision_gate.py — Decision Gate](#decision_gatepy--decision-gate)
   - [experience_log.py — Experience Log](#experience_logpy--experience-log)
   - [behavior_profile.py — Behavior Profile](#behavior_profilepy--behavior-profile)
   - [tools.py — Tool System](#toolspy--tool-system)
   - [commands.py — Chat Commands](#commandspy--chat-commands)
   - [research.py — Research System](#researchpy--research-system)
   - [self_improve.py — Self-Improvement](#self_improvepy--self-improvement)
   - [scheduler.py — Scheduler](#schedulerpy--scheduler)
   - [torque_clustering/ — Clustering Algorithm](#torque_clustering--clustering-algorithm)
6. [Request Lifecycle](#request-lifecycle)
7. [Background Cycles](#background-cycles)
8. [Encryption Architecture](#encryption-architecture)
9. [API Reference](#api-reference)
10. [Chat Commands Reference](#chat-commands-reference)
11. [Configuration Reference](#configuration-reference)
12. [Dependencies](#dependencies)
13. [Known Limitations & Notes](#known-limitations--notes)

---

## Overview

3AM is a fully local AI assistant built on top of any llama.cpp-compatible model. It wraps a language model in a system that adds:

- **Persistent long-term memory** — facts extracted from conversations, stored in SQLite with vector embeddings
- **Torque Clustering** — physics-inspired algorithm that autonomously discovers topic clusters in memory
- **Decision Gate** — pre-response router that decides whether to answer, search, or ask for clarification
- **Fernet encryption at rest** — all sensitive user data encrypted with keys derived from login passwords
- **Adaptive behavior** — learns from feedback without changing model weights
- **Self-created tools** — proposes, generates, and installs new Python tools with user approval
- **Proactive research** — autonomously learns about user interests during idle periods
- **Multi-user support** — per-user isolated data stores, separate encryption keys

Everything runs locally. No data leaves the machine. No model weights are modified.

---

## Architecture

```
Browser (Vanilla JS)              FastAPI Server (server.py)
┌───────────────────────┐    ┌──────────────────────────────────────┐
│ Chat UI               │    │ WebSocket / SSE streaming             │
│ Memory star-map       │◄──►│ UserLLMCore (per-user instance)       │
│ Tool manager          │    │  ├─ MemorySystem                       │
│ Research panel        │    │  ├─ IntrospectionLoop                  │
│ Analytics panel       │    │  ├─ DecisionGate                       │
│ Settings              │    │  ├─ ExperienceLog                      │
└───────────────────────┘    │  ├─ BehaviorProfile                    │
                             │  ├─ ToolExecutor                        │
                             │  └─ CommandHandler                      │
                             └──────────────┬───────────────────────┘
                                            │
                          ┌─────────────────▼──────────────────┐
                          │ SQLite + sqlite-vec (memory.db)      │
                          │ memories · clusters · embeddings     │
                          │ experience_log · meta                │
                          └─────────────────────────────────────┘
                                            │
                          ┌─────────────────▼──────────────────┐
                          │ llama.cpp (llama-server)             │
                          │ Any GGUF model (Qwen3-14B default)  │
                          │ OpenAI-compatible API on :8080       │
                          └─────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Per-user `UserLLMCore` instances | Full isolation — memory, tools, settings, encryption keys |
| Embeddings in sqlite-vec | No RAM overhead at startup; fetched on demand |
| Cluster centroids in RAM | Small set (~50 clusters); needed for every context retrieval |
| Keys never persisted | Security — derived from password at login, cleared on logout |
| `/no_think` appended to prompts | Suppresses Qwen3 chain-of-thought for non-chat LLM calls |

---

## Installation & Configuration

### Quick Start

```bash
git clone https://github.com/pthompson8594/3am-AI.git
cd 3am-AI

# Quick test with a small model (~350MB)
./run-test.sh

# Full install (creates venv, systemd services)
./install.sh
```

Open `http://localhost:8000`, create an account, and start chatting.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLM_MODEL` | auto-detected | Model name (auto-detects from `/v1/models` if not set) |
| `GEMINI_API_KEY` | — | Required for Google Search web search provider |

### Config File

`~/.config/3am/config.json`:

```json
{
  "llm_server_url": "http://localhost:8080",
  "llm_model": "qwen3-14b",
  "introspection_schedule": "03:00",
  "introspection_check_interval": 3600,
  "allow_registration": true,
  "session_timeout_hours": 24,
  "clustering_adjustment_factor": 0.5,
  "gemini_api_key": "YOUR_KEY_HERE"
}
```

**`clustering_adjustment_factor`**: Controls how aggressively Torque Clustering splits clusters. Higher values → fewer, larger clusters. Lower values → more, smaller clusters. Default 0.5. After each run the log prints health stats — use these to tune.

---

## Data Storage Layout

### Server-level (shared)

```
~/.local/share/3am/
├── users.json          # All user accounts (password hashes, encryption salts)
├── sessions.json       # Active session tokens
└── users/
    └── {user_id}/      # Per-user isolated directory
        ├── memory.db           # SQLite: memories, clusters, embeddings, experience log
        ├── pending_conversations.jsonl  # Queued conversations (processed at 3 AM)
        ├── behavior_profile.json       # Learned behavioral preferences (encrypted)
        ├── research.json               # Research topics and findings (encrypted)
        ├── installed_tools.json        # Custom tool registry (encrypted)
        ├── settings.json               # User settings (search provider, gate config)
        ├── conversations/              # Full chat history (one JSON per conversation)
        │   └── {conversation_id}.json
        ├── error_journal.json          # Tool failure log for self-improvement
        ├── introspection_log.json      # Introspection run statistics
        ├── suggestions.json            # Self-improvement suggestions
        └── feedback.jsonl              # Legacy feedback log (append-only)

~/.config/3am/
└── config.json         # Server configuration
```

### memory.db Schema

```sql
-- Extracted facts about the user
CREATE TABLE memories (
    id          TEXT PRIMARY KEY,
    summary     TEXT NOT NULL,          -- encrypted
    category    TEXT NOT NULL,
    priority    INTEGER NOT NULL,       -- 1-5 (5=permanent identity facts)
    timestamp   REAL NOT NULL,
    cluster_id  TEXT,
    message     TEXT NOT NULL DEFAULT '', -- encrypted (first 500 chars of user message)
    response    TEXT NOT NULL DEFAULT ''  -- encrypted (first 500 chars of response)
);

-- Topic clusters discovered by Torque Clustering
CREATE TABLE clusters (
    id           TEXT PRIMARY KEY,
    theme        TEXT NOT NULL,         -- encrypted
    priority     INTEGER NOT NULL,
    last_update  REAL NOT NULL,
    torque_mass  REAL NOT NULL DEFAULT 0.0,
    center_vector BLOB NOT NULL,        -- float32 array, NOT encrypted (needed for search)
    message_refs  TEXT NOT NULL DEFAULT '[]'  -- JSON array of memory IDs
);

-- Vector embeddings (sqlite-vec extension)
CREATE VIRTUAL TABLE vec_memories USING vec0(
    memory_id TEXT,
    embedding float[768] distance_metric=cosine
);

-- Key-value store for user profile and stats
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,  -- 'user_profile', 'stats'
    value TEXT               -- encrypted
);

-- Per-response interaction log
CREATE TABLE experience_log (
    id                  TEXT PRIMARY KEY,
    timestamp           REAL NOT NULL,
    message_id          TEXT NOT NULL,
    gate_action         TEXT,           -- 'answer', 'search', 'ask'
    gate_confidence     REAL,
    response_confidence REAL,
    tool_used           TEXT,
    user_feedback       TEXT,           -- 'positive', 'negative', NULL
    feedback_tags       TEXT DEFAULT '[]',
    outcome_score       REAL
);
```

**What is and is not encrypted:**

| Data | Encrypted | Reason |
|------|-----------|--------|
| `memories.summary`, `.message`, `.response` | Yes | Sensitive user content |
| `clusters.theme` | Yes | Derived from user content |
| `meta.user_profile` | Yes | Compressed personal facts |
| `pending_conversations.jsonl` (each line) | Yes | Raw conversation text |
| `behavior_profile.json` | Yes | Behavioral preferences |
| `research.json` | Yes | Research findings |
| `installed_tools.json` | Yes | Tool code and schemas |
| `clusters.center_vector` | **No** | Required for vector similarity search |
| `vec_memories.embedding` | **No** | Required for vector similarity search |

---

## Module Reference

### server.py — Web Server & API

**Purpose:** FastAPI application. Entry point for all HTTP/WebSocket traffic. Owns per-user `UserLLMCore` instances.

**Key globals:**

| Name | Type | Description |
|------|------|-------------|
| `auth` | `AuthSystem` | Singleton — shared across all users |
| `scheduler` | `IntrospectionScheduler` | Singleton — 3 AM job scheduler |
| `user_cores` | `dict[str, UserLLMCore]` | Lazy-loaded per-user instances |
| `_user_last_message` | `dict[str, float]` | Rate limiting: last message timestamp |
| `_user_ws_connections` | `dict[str, WebSocket]` | Active WebSocket per user |

**`UserLLMCore`** — the per-user container:

```python
class UserLLMCore:
    user: User
    memory: MemorySystem
    introspection: IntrospectionLoop
    decision_gate: DecisionGate
    experience_log: ExperienceLog
    behavior_profile: BehaviorProfile
    tools: ToolExecutor
    commands: CommandHandler
    _encryptor: DataEncryptor
```

Instantiated on first login. Holds all per-user state. Never shared between users.

**Chat flow (`chat_stream`):**

1. Check for `?commands` — handled without LLM
2. Check if clustering in progress — return early with status message
3. Load or create conversation
4. Retrieve memory context (single embed + cosine similarity against cluster centroids)
5. Run Decision Gate — determines action: `answer`, `search`, or `ask`
6. Build system prompt: base + custom tools + learned behaviors + behavior profile + user profile + memory context
7. Stream LLM response, collecting tokens and logprobs
8. Handle tool calls if present; make follow-up LLM call
9. Compute response confidence from mean logprob: `exp(mean_logprob)`
10. Save conversation to disk
11. Queue conversation for memory processing (background)
12. Log to experience log
13. Emit `confidence` event to client

**Rate limiting:**
- 1 second minimum between messages per user
- Max 2 concurrent requests per user
- Applied to both HTTP and WebSocket endpoints

---

### auth.py — Authentication System

**Purpose:** User registration, login, session management, and encryption key lifecycle.

**`User` dataclass fields:**

| Field | Description |
|-------|-------------|
| `id` | Random hex32 unique identifier |
| `username` | 3–32 chars, alphanumeric + `_-` |
| `password_hash` | SHA-256 of (password + salt) — see note |
| `salt` | Random hex32 for password hashing |
| `encryption_salt` | Random hex32 for Fernet key derivation |
| `settings` | Dict of user preferences |

> **Note on password hashing:** Currently SHA-256 with salt. The code has a `TODO` to replace with bcrypt. The `encryption_salt` is separate from the auth `salt` so key derivation can be changed independently of auth hashing.

**`AuthSystem._user_keys`:**
- `dict[str, bytes]` — maps `user_id` → Fernet key
- Never written to disk
- Populated on login via `derive_key_from_password(password, encryption_salt)`
- Cleared on logout (if no other active sessions exist for the same user)
- Cleared implicitly on server restart

**Key operations:**

| Method | Description |
|--------|-------------|
| `register(username, password)` | Creates user, generates both salts, saves to `users.json` |
| `login(username, password)` | Verifies password, derives + caches Fernet key, creates session |
| `logout(token)` | Invalidates session, clears key if no other sessions |
| `validate_session(token)` | Returns User or None; updates last_activity |
| `get_user_key(user_id)` | Returns in-memory Fernet key or None |
| `change_password(user, old, new)` | Verifies old, sets new hash + new auth salt, invalidates all sessions |

Sessions expire after 24 hours (configurable). Stored in `sessions.json`.

---

### data_security.py — Encryption

**Purpose:** Fernet symmetric encryption. Key derivation from passwords.

**`DataEncryptor`:**

```python
DataEncryptor(user_key: Optional[bytes])
```

- `user_key=None` → encryption disabled (plaintext passthrough)
- `user_key=<bytes>` → Fernet encryption enabled

Core methods:

| Method | Description |
|--------|-------------|
| `encrypt_str(s)` / `decrypt_str(s)` | UTF-8 string encrypt/decrypt |
| `encrypt_file(path, data)` / `decrypt_file(path)` | JSON serialize → encrypt → write bytes |
| `encrypt(data)` / `decrypt(data)` | Raw bytes |

**Graceful fallback:** `decrypt()` catches `InvalidToken` and returns data unchanged. This allows existing plaintext data to be served without error — it will be re-encrypted on next write.

**`derive_key_from_password(password, salt)`:**

```
PBKDF2HMAC(SHA-256, 100,000 iterations, 32-byte output, salt)
→ base64url-encode → Fernet key
```

- `password`: user's plaintext login password
- `salt`: 16-byte random value stored in `users.json` as `encryption_salt`

**`generate_salt()`:** `os.urandom(16)` — 16 cryptographically random bytes.

**`SecureUserData`:** Thin wrapper providing `save(filename, data)` / `load(filename)` against the user's data directory with optional encryption. Used by `UserLLMCore` for path management.

---

### memory.py — Memory System

**Purpose:** All memory storage, retrieval, fact extraction, conflict resolution, and clustering.

**`EmbeddingModel`:**
- Model: `nomic-ai/nomic-embed-text-v1.5` (768-dim, CPU only)
- Lazy-loaded (first use triggers download)
- Thread-safe via lock
- Normalized embeddings (required for cosine similarity with `sqlite-vec`)

**`MemorySystem` initialization:**

1. Creates SQLite DB with WAL mode (concurrent reads allowed)
2. Creates `vec_memories` virtual table via sqlite-vec
3. Migrates from legacy `memory.json` if DB is empty and JSON exists
4. Loads all memory metadata and cluster centroids into RAM
5. Runs `_cleanup()` to remove decayed memories

**Memory priority system:**

| Priority | Retention | Examples |
|----------|-----------|---------|
| 5 | Permanent (decay rate 0.0005) | Name, profession, location |
| 4 | Months | Preferences, skills, hobbies |
| 3 | Weeks | Projects, patterns |
| 2 | Days | Current tasks, temporary context |
| 1 | Skip — not stored | Trivial information |

Retention formula: `exp(-decay_rate × age_hours)`. Memories below threshold 0.35 are deleted during `_cleanup()`.

**Chat-time path (`queue_conversation`):**

1. Append raw conversation to `pending_conversations.jsonl` (encrypted per-line)
2. Run urgency check (LLM call, 20s timeout) — looks for priority-5 identity facts
3. If urgent facts found → `_store_fact()` immediately; otherwise queued for 3 AM

**`_store_fact()` dedup logic:**

Uses `sqlite-vec MATCH` query for nearest neighbor. If cosine distance < 0.08 (similarity > 0.92) → duplicate, skip. Much faster than O(n²) Python loop.

**Context retrieval (`get_relevant_context`):**

1. Embed the user's query
2. Score each cluster by cosine similarity of query embedding to cluster centroid (in RAM — no DB read)
3. Apply mass weight: `score × (1 + torque_mass/100 × 0.2)`
4. Take top 2 clusters with similarity > 0.4
5. Return top 3 memories from each cluster

**Conflict resolution (`resolve_conflicts`):**

For each memory, queries `sqlite-vec` for neighbors in the conflict distance range (cosine similarity 0.75–0.92). Same-category pairs in this range: older fact is pruned. No LLM needed.

**`run_torque_clustering_async(mode)`:**

| Mode | When Used | What Happens |
|------|-----------|-------------|
| `"full"` | Weekly or manual | O(n²) full recluster via TorqueClustering |
| `"auto"` | Nightly 3 AM cycle | Full if >7 days since last; split oversized if any; else incremental |
| `"incremental"` | Lite re-cluster (3×/day) | Assign unclustered facts to existing clusters only — no LLM, no restructuring |

CPU-heavy work runs in `loop.run_in_executor(None, ...)` to keep the event loop live.

**Thread safety:** Per-thread SQLite connections via `threading.local()`. Write operations use `threading.Lock()`.

---

### introspection.py — Background Processing

**Purpose:** Manages all background processing loops. Coordinates the 3 AM cycle and hourly idle work.

**Constants:**

| Constant | Value | Purpose |
|----------|-------|---------|
| `IDLE_INTERVAL_SECONDS` | 3600 (1 hour) | How often idle loop runs |
| `LITE_RECLUSTER_INTERVAL` | 28800 (8 hours) | How often lite re-cluster runs |

**Two background loops:**

**1. Heavy memory cycle (`_idle_loop`)** — opt-in, runs when `ConsolidationConfig.enabled=True`:
- Triggered by scheduler at 3 AM
- Runs `_run_cycle_inner()`:
  1. Process pending conversations (fact extraction)
  2. Resolve conflicts (prune superseded facts)
  3. Torque Clustering (auto mode)
  4. Regenerate user profile if dirty
  5. Detect cluster conflicts
  6. Update cluster themes
  7. Summarize large clusters
  8. Update behavior profile from experience log

**2. Lightweight idle loop (`_light_idle_loop`)** — always running (daemon thread):
- Every hour: error analysis, research, self-improvement suggestions
- Every 8 hours: lite re-cluster (incremental mode, no LLM)
- Skips if heavy memory cycle is currently running

**`_run_lite_cluster_cycle()`:**
```python
if _in_progress: return          # Skip if 3 AM cycle is running
if not memory.needs_reclustering(): return  # Skip if nothing to do
run_torque_clustering_async(mode="incremental")
```

**`ErrorJournal`:**
Logs tool failures to `error_journal.json`. Keeps last 100 entries. Used by self-improvement to detect patterns.

---

### decision_gate.py — Decision Gate

**Purpose:** Pre-response router. Decides whether to answer, search, or ask for clarification before generating a full response.

**Two paths:**

**Fast path (rule-based, zero latency):**

| Pattern | Action | Confidence |
|---------|--------|------------|
| Chitchat (`hi`, `thanks`, etc.) | answer | 0.95 |
| Explicit search trigger (`search for`, `latest`, etc.) | search | 0.90 |
| Ambiguity trigger (`what do you mean`, etc.) | ask | 0.85 |
| Short query (<15 chars) + sparse memory (<80 chars) | search | 0.75 |

**LLM path (ambiguous cases):**

Small LLM call with `max_tokens=5`, `temperature=0`, `logprobs=True`. Prompt asks for one word: `answer`, `search`, or `ask`. Confidence = `exp(logprob of first token)`.

**Threshold application:**

```python
if action == "answer" and confidence < search_threshold and sparse_memory:
    action = "search"
elif action == "answer" and confidence < ask_threshold:
    action = "ask"
```

Thresholds come from `BehaviorProfile` and are learned from feedback patterns over time.

**Gate failure:** Defaults to `answer` with confidence 0.5 — safe fallback.

---

### experience_log.py — Experience Log

**Purpose:** Records every interaction for behavioral learning. Stored in `memory.db` `experience_log` table.

**Per-response record:**

| Field | Set By | Description |
|-------|--------|-------------|
| `gate_action` | Decision Gate | `answer` / `search` / `ask` |
| `gate_confidence` | Decision Gate | 0–1 |
| `response_confidence` | LLM logprobs | `exp(mean_logprob)` of all tokens |
| `tool_used` | Chat stream | Tool name or None |
| `user_feedback` | User | `positive` / `negative` via 👍👎 |
| `feedback_tags` | User | e.g. `["hallucinated", "too_verbose"]` |
| `outcome_score` | Computed | 1.0 positive, 0.0 negative |

**`get_feedback_patterns()`** — used by `BehaviorProfile.update_from_experience()`:

| Pattern | Description |
|---------|-------------|
| `low_conf_negative_rate` | Fraction of low-confidence responses that got thumbs-down |
| `search_helped_rate` | Fraction of search-routed responses that got thumbs-up |
| `hallucination_rate` | Fraction of rated responses tagged "hallucinated" |

---

### behavior_profile.py — Behavior Profile

**Purpose:** Stores and applies learned behavioral preferences. Updated nightly by introspection. Injected into system prompt on every chat request.

**Default profile:**

```json
{
  "tool_usage_bias": "balanced",
  "verbosity": "medium",
  "uncertainty_behavior": "hedge",
  "search_threshold": 0.5,
  "ask_threshold": 0.3,
  "user_preferences": {}
}
```

**Nightly update logic (`update_from_experience`):**

| Condition | Action |
|-----------|--------|
| `low_conf_negative_rate > 0.6` | Lower `search_threshold` by 0.05 (search earlier) |
| `low_conf_negative_rate < 0.2` AND `search_helped_rate > 0.7` | Slightly raise `search_threshold` (don't over-search) |
| `hallucination_rate > 0.15` | Set `uncertainty_behavior = "hedge"`, lower `search_threshold` |

Thresholds clamped to [0.2, 0.8].

**System prompt injection (`to_prompt_fragment`):**

Returns a short string like:
```
Keep responses concise and to the point.
When uncertain, acknowledge uncertainty before answering.
```

---

### tools.py — Tool System

**Purpose:** Built-in tools available to the LLM, custom tool registry, and tool execution.

**Built-in tools:**

| Tool | Description |
|------|-------------|
| `web_search` | Google (via Gemini API) or DuckDuckGo |
| `execute_command` | Shell command (dangerous commands require approval) |
| `create_file` | Write file (always requires user approval) |
| `read_file` | Read local file (up to 200 lines by default) |
| `read_url` | Fetch webpage content, extract text |
| `clipboard` | Read/write system clipboard (requires `xclip` or `xsel`) |
| `system_info` | CPU, memory, disk, battery, network stats via psutil |

**Approval flow for dangerous operations:**

```
LLM calls create_file or dangerous execute_command
    → ToolExecutor stores PendingApproval
    → Returns "[PENDING_APPROVAL:{id}]"
    → chat_stream emits approval_required event to frontend
    → User approves/denies in UI
    → POST /api/approve → execute_pending_approval()
```

Dangerous command patterns: `sudo`, `rm -rf`, `>`, `>>`, `cp`, `mv`, `mkdir`, `touch`, and more.

**`CustomToolRegistry`:**

Stores LLM-generated custom tools in `installed_tools.json` (encrypted). Each tool is a Python async function stored as source code and executed via `exec()` in an isolated namespace.

Safety scan blocks: `subprocess`, `socket`, `threading`, `multiprocessing`, `ctypes`, `eval`, `exec`, `os`, `sys`, `importlib`, and more.

Tool functions must be named `async def tool_{name}(**kwargs)`.

**Search providers:**

- `google` (default) — Gemini 2.5 Flash with Google Search grounding. Requires `GEMINI_API_KEY`.
- `duckduckgo` — `duckduckgo-search` Python package, no API key required.

---

### commands.py — Chat Commands

**Purpose:** Handles `?` prefix commands that bypass the LLM entirely.

All commands start with `?`. Unknown `?` messages are passed to the LLM as regular queries.

See [Chat Commands Reference](#chat-commands-reference) for the full list.

---

### research.py — Research System

**Purpose:** Proactive research on topics the user has discussed. Runs during idle hours.

**Key behaviors:**
- Topics discovered from conversation clusters automatically
- Manual topics via `?learn <topic>` or UI
- Daily quota to limit API usage (default: 5 searches/day)
- Research findings stored in `research.json` (encrypted) and written into long-term memory
- High-confidence findings (≥0.8) stored as priority-3 memories; others as priority-2
- Findings expire from `research.json` after 7 days (already in memory; log is just for review)
- Manually deleting an insight from the UI also removes its corresponding memory entry

**Four-layer decision gate — controls when topic scanning runs:**

| Gate | Condition | Effect |
|------|-----------|--------|
| 1 — Cluster recency | Cluster has no new memories in the last 14 days | Skip that cluster |
| 2 — Topic cooldown | `identify_topics()` was called within the last 5 days | Skip topic scan entirely |
| 3 — Poor results backoff | Last research on a topic had 0 useful insights or `search_quality == "poor"` | Back off that topic's cluster for 7 days |
| 4 — Experience signal | `low_conf_negative_rate > 0.25` in experience log | Halve the Gate 2 cooldown (2.5 days) |

Gates 1–3 are bypassed when the user manually queues a topic via `?learn`. Gate 2 countdown resets after each LLM call to `identify_topics()` regardless of whether new topics were found.

---

### self_improve.py — Self-Improvement System

**Purpose:** Analyzes error patterns and feedback to propose behavioral improvements, including custom tool proposals.

**Three-stage custom tool pipeline:**

| Stage | Command | What Happens |
|-------|---------|-------------|
| 1. Propose | `?propose-tool <desc>` or UI | LLM generates tool name, description, parameters |
| 2. Generate | `?approve-tool <id>` or UI | LLM generates Python code |
| 3. Install | `?install-tool <id>` or UI | Safety scan → `exec()` → registered live |

Tool status lifecycle: `proposal` → `code_ready` → `installed`

---

### scheduler.py — Scheduler

**Purpose:** Triggers the 3 AM introspection cycle.

**Behavior:**
- Checks every minute whether it's within 5 minutes of the scheduled time (default 03:00)
- Only runs once per calendar day
- Runs `quick_check()` first to confirm there's pending work
- Full cycle triggered via `IntrospectionScheduler.run_full_introspection()`

Note: The scheduler currently has callbacks set up but the `UserLLMCore`-level introspection for each user is actually driven by `IntrospectionLoop._idle_loop()` (started per-user on login). The scheduler is a global mechanism that would coordinate server-wide tasks.

---

### torque_clustering/ — Clustering Algorithm

**Purpose:** Autonomous cluster discovery based on gravitational torque balance.

Based on: **Yang & Lin, "Autonomous clustering by fast find of mass and distance peaks," IEEE TPAMI, 2025.**
Licensed under CC BY-NC-SA 4.0.

**How it works (simplified):**

1. Compute full pairwise cosine distance matrix between all memory embeddings
2. TorqueClustering identifies density peaks by computing "torque" — the rotational force imbalance between a point and its neighbors
3. Points are assigned to clusters based on torque equilibrium boundaries
4. No manual threshold for number of clusters — emerges from the data structure

**Integration in `memory.py`:**

```python
result = TorqueClustering(
    distance_matrix,    # scipy cdist cosine
    K=target_clusters,  # hint (n_memories // 4)
    isnoise=False,
    isfig=False,
    auto_config=False,
    use_std_adjustment=True,
    adjustment_factor=0.5,  # from config.json
)
```

Returns `cluster_labels` and `cluster_labels_with_noise`. Noise points (label < 0) are discarded.

---

## Request Lifecycle

### Normal Chat Request

```
User sends message
    │
    ├─ is_command()? → CommandHandler.handle() → return response
    │
    ├─ _clustering_in_progress? → return "clustering in progress" message
    │
    ├─ retrieve memory context
    │   embed(query) → score clusters → fetch top-3 memories from top-2 clusters
    │
    ├─ DecisionGate.evaluate()
    │   fast path: rule-based patterns
    │   slow path: LLM call with logprobs → confidence score
    │
    ├─ build system prompt
    │   SYSTEM_PROMPT_BASE
    │   + custom tool descriptions
    │   + learned behaviors (self-improve)
    │   + behavior profile fragment
    │   + user profile (priority 4/5 facts)
    │   + memory context
    │
    ├─ LLM stream call
    │   logprobs collected for confidence scoring
    │   tool_calls detected in streaming deltas
    │
    ├─ tool call? → execute → follow-up LLM call
    │   dangerous/file op? → PendingApproval → return, wait for user approval
    │
    ├─ compute response_confidence = exp(mean_logprob)
    │
    ├─ save conversation (disk)
    ├─ queue_conversation() → pending JSONL (background)
    │   urgency_check() → if priority-5 facts → store immediately
    │
    └─ experience_log.record() → experience_log table
```

### 3 AM Cycle

```
IntrospectionLoop._run_cycle_inner():
    1. process_pending_conversations()
       │  read pending JSONL (decrypt each line)
       │  LLM: group by topic
       │  LLM: extract facts per group (up to 8 per group)
       │  _store_fact() for each (dedup via sqlite-vec)
       │  delete pending file
       │
    2. resolve_conflicts()
       │  sqlite-vec MATCH for each memory → neighbors in conflict range
       │  same category + conflict range → prune older
       │
    3. run_torque_clustering_async(mode="auto")
       │  auto: full if weekly, split if oversized, else incremental
       │  CPU work in thread pool executor
       │  LLM: generate cluster themes (async after executor)
       │
    4. regenerate_user_profile() if dirty
       │  LLM: compress priority 4/5 facts into dense profile string
       │
    5. detect_conflicts() — LLM-based cluster-level check
    6. update_cluster_themes() — refresh stale themes
    7. summarize_cluster() — top 3 large clusters
    8. update_from_experience() — adjust behavior profile thresholds
```

---

## Background Cycles

### Timing Summary

| Cycle | Trigger | What Runs |
|-------|---------|-----------|
| Per-message | Every chat message | urgency check (priority-5 facts) |
| Hourly (idle) | Every 60 minutes | Error analysis, research, self-improve |
| 3× daily (lite recluster) | Every 8 hours | Incremental cluster assignment only |
| Nightly (3 AM) | Once per day at 03:00 | Full memory processing cycle |
| Weekly (full recluster) | Every 7 days | O(n²) Torque Clustering rebuild |

### Thread Architecture

Each `UserLLMCore` spawns two daemon threads on creation:
1. `_light_idle_loop` — lightweight hourly tasks
2. `_idle_loop` — heavy memory cycle (only if consolidation enabled)

These run in separate asyncio event loops (one per thread). The main FastAPI event loop is unaffected.

---

## Encryption Architecture

```
User password (plaintext, only in memory during login)
    │
    ▼ PBKDF2HMAC-SHA256 (100,000 iterations)
    │  key: 32 bytes
    │  salt: 16-byte random, stored in users.json as hex
    │
    ▼ base64url-encode
    │
    ▼ Fernet key (URL-safe base64, 32 bytes)
    │
    ├─ stored in AuthSystem._user_keys[user_id]
    │  (dict in process memory, never persisted)
    │
    └─ cleared when: logout, server restart, or session expiry with no remaining sessions

Fernet encryption:
    plaintext → AES-128-CBC encrypt → HMAC-SHA256 sign → base64url
    Fernet token includes IV, timestamp, and HMAC for authenticated encryption

Graceful migration:
    On decrypt: if InvalidToken → return data unchanged
    Old plaintext data served as-is; re-encrypted on next write
```

**Key properties:**
- Separate key per user — one user's data cannot decrypt another's
- Key never touches disk — server restart clears all keys (users re-derive on next login)
- Separate `encryption_salt` and auth `salt` — changing password doesn't invalidate old key (note: current implementation does not re-encrypt on password change — future improvement)
- Embedding vectors deliberately not encrypted — required for sqlite-vec cosine distance computation

---

## API Reference

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Register new user `{username, password}` |
| POST | `/api/auth/login` | Login `{username, password}` → sets session cookie |
| POST | `/api/auth/logout` | Invalidate session |
| GET | `/api/auth/me` | Get current user info |
| POST | `/api/auth/change-password` | Change password `{old_password, new_password}` |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | SSE streaming chat `{message, conversation_id?}` |
| WebSocket | `/ws/chat` | WebSocket chat (supports cancel, feedback) |

**SSE event types:**

| Type | Payload | Description |
|------|---------|-------------|
| `conversation_id` | `{id}` | Conversation UUID for this session |
| `status` | `{status}` | `memory`, `decision_gate`, `thinking`, `tool`, `introspection`, `research`, `clustering` |
| `gate_decision` | `{action, confidence}` | Decision Gate result |
| `token` | `{content}` | Streamed response token |
| `tool_call` | `{name, args}` | Tool being called |
| `tool_output` | `{name, output}` | Tool result |
| `approval_required` | `{approval}` | File operation needs user approval |
| `confidence` | `{value, message_id}` | Response confidence score |
| `done` | `{stats, message_id}` | Response complete with token stats |
| `error` | `{message}` | Error occurred |
| `cancelled` | — | Response cancelled (WebSocket) |

### Conversations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List all conversations |
| GET | `/api/conversations/{id}` | Get conversation |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| PATCH | `/api/conversations/{id}` | Rename `{title}` |

### Memory

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/memory` | Get clusters with torque stats |
| GET | `/api/memory/viz` | 3D positions for star-map (UMAP or PCA) |
| GET | `/api/memory/export` | Download all memories as JSON |
| POST | `/api/memory/import` | Import memories from export JSON |
| DELETE | `/api/memory` | Delete all memories |
| DELETE | `/api/memory/{id}` | Delete single memory |

### Introspection & Research

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/introspection/stats` | Introspection statistics |
| POST | `/api/introspection/trigger` | Manual trigger `?force_recluster=true` for full recluster |
| GET | `/api/research/status` | Research status and quota |
| POST | `/api/research/toggle` | Enable/disable research |
| GET | `/api/research/data` | All topics and insights |
| GET | `/api/research/findings` | Summary of findings |
| GET | `/api/research/findings/download` | Download findings as text |
| DELETE | `/api/research/topic/{idx}` | Delete research topic |
| DELETE | `/api/research/insight/{idx}` | Delete research insight |

### Settings & Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings` | Get all settings |
| PATCH | `/api/settings` | Update settings |
| GET | `/api/behavior-profile` | Get behavior profile |
| PATCH | `/api/behavior-profile` | Update behavior profile |
| POST | `/api/feedback` | Submit feedback `{message_id, value, tags}` |
| GET | `/api/feedback/stats` | Feedback statistics |
| GET | `/api/analytics` | Comprehensive analytics |

### Tools

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tools` | List installed + proposals |
| POST | `/api/tools/propose` | Stage 1: propose tool `{description}` |
| POST | `/api/tools/{id}/generate` | Stage 2: generate code |
| POST | `/api/tools/{id}/install` | Stage 3: safety-check and install |
| DELETE | `/api/tools/{id}` | Uninstall tool |
| POST | `/api/approve` | Approve/deny pending file operation `{approved}` |
| GET | `/api/suggestions` | Get self-improvement suggestions |
| POST | `/api/suggestions/{id}/approve` | Approve suggestion |
| POST | `/api/suggestions/{id}/dismiss` | Dismiss suggestion |

---

## Chat Commands Reference

All commands start with `?`. Unknown `?` prefixes are treated as regular messages.

### Feedback

| Command | Description |
|---------|-------------|
| `?+` | Rate last response positively |
| `?-` | Rate last response negatively |
| `?feedback <text>` | Add detailed feedback to last response |

### Memory

| Command | Description |
|---------|-------------|
| `?memory` | Show all clusters with priority stars and torque mass |
| `?introspect` | Show introspection run statistics |
| `?errors` | Show common tool error patterns |
| `?reflect` | Trigger a full introspection cycle manually |
| `?recluster` | Force full Torque Clustering rebuild |

### Consolidation

| Command | Description |
|---------|-------------|
| `?consolidate` | Show status |
| `?consolidate on` | Enable hourly memory consolidation |
| `?consolidate off` | Disable |

### Research

| Command | Description |
|---------|-------------|
| `?research` | Show status and quota |
| `?research on` | Enable proactive research |
| `?research off` | Disable |
| `?findings` | Show research findings summary |
| `?learn <topic>` | Queue a topic for research |

### Self-Improvement

| Command | Description |
|---------|-------------|
| `?improve` | Show status |
| `?improve on` | Enable self-improvement suggestions |
| `?improve off` | Disable |
| `?improve prompts on` | Allow prompt modifications |
| `?improve prompts off` | Disallow prompt modifications |
| `?suggestions` | List pending suggestions |
| `?approve <#>` | Approve suggestion |
| `?dismiss <#>` | Dismiss suggestion |
| `?implemented <#>` | Mark suggestion implemented |
| `?prompt` | Show active custom prompt additions |
| `?selfresearch` | Show self-research topics |
| `?selfresearch on/off` | Toggle self-research mode |

### Custom Tools

| Command | Description |
|---------|-------------|
| `?tools` | List all custom tools and proposals |
| `?propose-tool <desc>` | Stage 1: propose new tool concept |
| `?approve-tool <id>` | Stage 2: generate Python code |
| `?install-tool <id>` | Stage 3: safety-check and install |
| `?remove-tool <name>` | Uninstall a custom tool |

---

## Configuration Reference

### `~/.config/3am/config.json`

| Key | Default | Description |
|-----|---------|-------------|
| `llm_server_url` | `http://localhost:8080` | llama.cpp server |
| `llm_model` | `qwen3-14b` | Model name for LLM calls |
| `gemini_api_key` | `""` | API key for Google Search |
| `clustering_adjustment_factor` | `0.5` | Torque Clustering aggressiveness |
| `allow_registration` | `true` | Allow new user registration |
| `session_timeout_hours` | `24` | Session expiry |
| `introspection_schedule` | `"03:00"` | Nightly cycle time |
| `introspection_check_interval` | `3600` | Scheduler check interval (seconds) |

### Per-user `settings.json`

| Key | Default | Description |
|-----|---------|-------------|
| `search_provider` | `"google"` | `"google"` or `"duckduckgo"` |
| `decision_gate_enabled` | `true` | Enable/disable Decision Gate |
| `decision_gate_sensitivity` | `0.5` | Gate threshold (0=always search, 1=never) |
| `show_confidence` | `true` | Show confidence scores in UI |
| `show_feedback_buttons` | `true` | Show 👍👎 buttons |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | ≥0.109.0 | Web framework |
| uvicorn | ≥0.27.0 | ASGI server |
| httpx | ≥0.26.0 | Async HTTP client |
| pydantic | ≥2.5.0 | Request/response models |
| cryptography | ≥41.0.0 | Fernet encryption |
| torch | (CPU) | Required by sentence-transformers |
| sentence-transformers | ≥3.0.0 | Nomic embedding model |
| sqlite-vec | ≥0.1.6 | Vector similarity search in SQLite |
| numpy | ≥1.24.0 | Array operations |
| scipy | ≥1.10.0 | Cosine distance matrix |
| networkx | ≥3.0 | Used by Torque Clustering |
| google-genai | ≥1.0.0 | Gemini API for Google Search |
| duckduckgo-search | ≥7.0.0 | DuckDuckGo search |
| beautifulsoup4 | ≥4.12.0 | HTML parsing for `read_url` |
| psutil | ≥5.9.0 | System info tool |
| umap-learn | ≥0.5.0 | 3D memory map (optional, falls back to PCA) |
| apscheduler | ≥3.10.0 | Scheduler support |

---

## Known Limitations & Notes

**Password hashing:** Currently SHA-256 with salt. The code has a TODO to upgrade to bcrypt. For a single-machine personal deployment this is acceptable, but bcrypt should be implemented before any multi-machine or network-exposed deployment.

**Password change does not re-encrypt:** Changing a password updates the auth credentials but does not re-derive and re-encrypt the data with the new password's key. After a password change, the user must log in with the new password to get a new key, but existing encrypted data was written with the old key. The graceful fallback handles this — old data decrypts with the old key, and gets re-encrypted with the new key on next write. However, if the server restarts between password change and the next write, data written before the change may become inaccessible. This edge case should be addressed in a future update.

**Session persistence:** `sessions.json` stores session tokens. On server restart, valid sessions are reloaded — but encryption keys are not (they only live in process memory). Users with persisted sessions will need to log in again after a server restart to re-derive their encryption key.

**Single llama.cpp server:** All users share one `LLM_URL`. There is no per-user model routing. If the LLM server is slow or busy, all users experience the delay.

**Embedding model:** `nomic-ai/nomic-embed-text-v1.5` runs CPU-only. On first use it downloads the model (~270MB). Embedding is the main latency cost for context retrieval and fact storage.

**Torque Clustering `K` hint:** The `target_clusters` hint passed to TorqueClustering is `max(2, n_memories // 4)`. The algorithm may return a different number. The `adjustment_factor` in config.json is the main tuning lever.

**Conversation history not encrypted:** Full conversation JSON files in `conversations/` are stored as plaintext. Only facts extracted from conversations are encrypted. If this is a concern, the conversation storage layer should be encrypted in a future update.

**No HTTPS by default:** The server runs on plain HTTP. For network exposure, place behind a reverse proxy (nginx, Caddy) with TLS.

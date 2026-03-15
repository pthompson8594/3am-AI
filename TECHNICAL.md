# 3AM — Technical Reference

Web-based LLM assistant with **Torque Clustering memory system, SQLite + sqlite-vec storage, multi-user support, ChatGPT-like interface, self-created custom tools, decision gate, logprobs-based confidence scoring, feedback-driven behavior adaptation, Fernet field-level encryption at rest, and document ingestion with PPR-linked proposition storage**.

## What's New in v1.3.0 (current)

### Agentic Multi-Step Tool Loop

`chat_stream` in `server.py` now loops up to 10 times, each iteration calling `_stream_one_turn`. A turn returns either a finished response or a list of tool calls; if tool calls are returned, they are executed and the results are appended to the message list before the next iteration. Previously the handler did one round of tool calls and stopped.

`_stream_one_turn` is a new helper that owns all streaming logic for a single LLM request — collects tokens, detects think blocks, dispatches tool calls — so the agentic outer loop doesn't duplicate that logic.

### Think Block Handling

`_stream_one_turn` scans the token stream for `<think>` / `</think>` delimiters. Tokens inside a think block are emitted as `{"type": "thinking", "content": "..."}` SSE events instead of `{"type": "token"}`. The frontend renders them as collapsible "💭 Reasoning" sections using the same pattern as tool outputs.

### `cache_prompt: false` on All LLM Requests

Every request to llama-server now sends `"cache_prompt": false`. This stops llama-server from writing KV-cache checkpoints (~60 MB each) to disk, which was triggering OOM kills on memory-constrained machines. The freed RAM can be given back to `--ctx-size` for a larger context window.

### Immediate Cluster Assignment on Memory Write

After `_store_memory` writes a new conversation turn, it immediately calls `assign_unclustered_memories` via `run_in_executor`. New facts are cluster-assigned within seconds of being stored rather than waiting for the 3 AM cycle. The call is guarded by `_clustering_in_progress` so it skips silently if a full recluster is already running.

### LLM Status LED

A small dot in the chat header polls `GET /api/llm/health` every 10 seconds. Green if llama-server responds, red if not. The DOM is only updated on status change to avoid unnecessary repaints.

### `max_tokens` Increased: 2000 → 8000

Raised to give the model enough headroom for multi-step responses and extended reasoning output.

### `llm_core.py` Removed

The CLI entry point was dead code (not imported anywhere). Deleted to reduce clutter.

### `static/sw.js` Added (no-op)

An empty service-worker stub at `/sw.js` stops browsers from generating 404 errors when they probe for a service worker.

---

## What's New in v1.2.0

### Document Ingestion

A **[+]** button in the chat input opens an ingestion panel that accepts a local file or URL and stores it in one of two modes:

**Ephemeral mode** (default — "Learn this" unchecked):
- Raw text injected into `_ephemeral_context` on `UserLLMCore`
- Appended to every system prompt as a `[DOCUMENT CONTEXT — session only]` block
- Cleared automatically on new chat or via the ✕ badge in the UI
- No DB writes; zero cost beyond context tokens
- Truncated to 80K chars if the document is large

**Persistent mode** ("Learn this" checked):
- LLM reads the full document (up to 96K chars / ~24K tokens, leaving 8K headroom for output)
- Returns structured JSON: `doc_summary`, `sections[]` each with `name`, `summary`, `propositions[]`
- Each proposition stored via `_store_fact`: dedup check, embedding, vec table, FTS5 index
- Three link types built into the `memory_links` lane graph:
  - `semantic` — bidirectional similarity lanes (built inside `_store_fact`, existing mechanism)
  - `sequential` — bidirectional between consecutive propositions within each section (weight 0.65)
  - `hierarchical` — proposition ↔ section summary (0.80), section ↔ doc summary (0.70)
- Viz cache invalidated on persist so new nodes appear in the star-map immediately

**PPR retrieval benefit:** when any document proposition is retrieved as a seed, Personalized PageRank walks sequential lanes to surface prerequisites/follow-ons, and hierarchical lanes to surface the section/document context — equivalent to RAPTOR-style tree retrieval without a separate index.

**Supported formats:**

| Extension | Library |
|-----------|---------|
| `.txt`, `.md`, `.csv`, `.log`, `.rst` | stdlib |
| `.pdf` | `pymupdf` (preferred) → `pypdf` (fallback) |
| `.docx` | `python-docx` |

**New endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `POST /api/ingest/document` | Multipart: `file` or `url` field, `persist` bool. Returns mode, counts, message. |
| `DELETE /api/ingest/ephemeral` | Clear ephemeral document context for this session. |

**New file:** `ingest.py` — `extract_text()`, `ingest_document()`, `count_propositions()`.

**New `MemorySystem` methods:** `_build_sequential_links()`, `_build_hierarchical_links()`, `store_document()`.

**New optional dependencies:** `pymupdf>=1.24.0`, `pypdf>=4.0.0`, `python-docx>=1.1.0`.

---

## What's New in v1.1.0

### Fernet Field-Level Encryption

All sensitive user data is now encrypted at rest using **Fernet** (AES-128-CBC + HMAC-SHA256, from the Python `cryptography` library).

**Key management:**
- Key derived from login password via PBKDF2HMAC-SHA256 (100,000 iterations, per-user 16-byte salt)
- Key lives only in `AuthSystem._user_keys` (process memory) — never written to disk
- Cleared when the last session for that user is invalidated (logout or server restart)
- Per-user `encryption_salt` stored in `users.json` (plaintext — just random bytes, not the key)

**Encrypted fields:**
- SQLite `memories` table: `summary`, `message`, `response`
- SQLite `clusters` table: `theme`
- SQLite `meta` table: `user_profile` value
- `pending_conversations.jsonl`: each line encrypted individually
- `behavior_profile.json`, `research.json`, `installed_tools.json`: encrypted as binary blobs

**Not encrypted:** embedding vectors in `vec_memories` (required for sqlite-vec similarity search, not human-readable)

**Migration:** existing plaintext data is returned as-is via graceful `InvalidToken` fallback; it gets encrypted on the next write. No manual migration step needed.

### Lite Re-clustering 3× Per Day

The hourly idle loop now runs an **incremental recluster** every 8 hours — assigning any unclustered facts to existing clusters without a full rebuild:

- Mode: `run_torque_clustering_async(mode="incremental")` → `assign_unclustered_memories()`
- No LLM calls, no distance matrix rebuild, no theme regeneration
- New memories are available for context retrieval the same day they're stored
- Skipped automatically if the 3 AM memory cycle is in progress
- Full nightly Torque Clustering rebuild at 3 AM is unchanged

---

## What's New in MK13 (v1.0.0)

### Decision Gate
The LLM now evaluates what it knows *before* generating a response — after memory is loaded so it has full context.

**Flow:**
```
User Input
   ↓
Memory Retrieval  (yellow — existing)
   ↓
Decision Gate     (orange — NEW)
   ↓
answer | search | ask
   ↓
Response Generation (red)
```

**Hybrid evaluation:**
- Rule-based fast path for obvious cases (explicit search requests, chitchat, clear ambiguity)
- LLM call for ambiguous cases — one word response (`answer` / `search` / `ask`) with logprob-based confidence
- Adjustable via sensitivity slider in Settings

**When gate says `search`:** forces `web_search` tool via `tool_choice` — the LLM cannot skip it.
**When gate says `ask`:** injects a clarification instruction before generating.

---

### Logprobs-Based Confidence Scoring
Confidence is computed from the average logprob across all response tokens — `exp(avg_logprob)` gives a real probability, not a heuristic. Logprobs are requested per-call via `"logprobs": true` in the API payload — no server startup flag needed.

- `≥ 0.65` → green dot
- `0.40–0.65` → yellow dot
- `< 0.40` → red dot

Shown on each response via hover (toggleable in Settings).

---

### Feedback UI (👍 / 👎)
Hover-reveal buttons on every assistant response, consistent with the existing copy-button pattern.

- 👍 sends `positive` feedback immediately
- 👎 reveals tag buttons: **Wrong / Irrelevant / Hallucinated / Too verbose** → Submit
- Feedback is sent via WebSocket (`type: "feedback"`) and logged to the experience log
- Stats shown in Settings → Response Feedback

---

### Experience Log
New `experience_log` table in `memory.db`. Every interaction records:

```json
{
  "message_id": "uuid",
  "gate_action": "search",
  "gate_confidence": 0.72,
  "response_confidence": 0.61,
  "tool_used": "web_search",
  "user_feedback": "positive",
  "feedback_tags": [],
  "outcome_score": 1.0
}
```

Used by the nightly introspection cycle (3 AM) to update the Behavior Profile.

---

### Behavior Profile
New per-user `behavior_profile.json` storing learned behavioral preferences:

```json
{
  "tool_usage_bias": "balanced",
  "verbosity": "medium",
  "uncertainty_behavior": "hedge",
  "search_threshold": 0.5,
  "ask_threshold": 0.3
}
```

Injected into the system prompt. Updated nightly (3 AM) by the introspection loop alongside Torque Clustering, so a full day of interactions informs the update rather than a single session. Feedback patterns drive the changes:
- High rate of thumbs-down on low-confidence responses → lower `search_threshold` (search more eagerly)
- High hallucination flags → switch `uncertainty_behavior` to `hedge`

---

### Custom Tools Panel

The three-stage custom tool pipeline (`?propose-tool` / `?approve-tool` / `?install-tool`) is now also accessible via a slide-out drawer panel — no commands required.

Open it with the **🔧** button in the sidebar header. The panel shows all tools organised by state, with one-click actions at each stage:

| State | Card shows | Action |
|-------|-----------|--------|
| Proposal | Name + description | **Generate Code** |
| Code ready | Name + description + collapsible code viewer | **View Code** / **Install** |
| Installed | Name + description | **Remove** |

The **Propose New Tool** form at the bottom of the panel accepts a plain-English description and kicks off Stage 1 (LLM proposes name + parameter schema). The existing `?` commands still work as a fallback.

---

### Research Panel

Proactive research findings and the research topic queue are now accessible via the **🔬** button in the sidebar header — no commands required.

The panel opens as a slide-out drawer with two sections:

| Section | Shows | Action |
|---------|-------|--------|
| **Insights** | Fact found, topic label, colour-coded confidence badge | **Delete** |
| **Topics** | Topic name, reason, priority stars, ✓ if already researched | **Delete** |

Individual entries can be deleted directly from the panel — useful for removing a bad search result or a topic that's no longer relevant.

---

### Analytics in Settings

The **📊 Analytics** section in the Settings modal (replacing the old Memory Clusters view, which is now covered by the 3D star map) shows live performance data from the experience log:

| Metric | What it tells you |
|--------|------------------|
| Total / rated interactions | Volume and engagement rate |
| 👍 / 👎 feedback | User satisfaction at a glance |
| Gate decision bar | Visual split of answer / search / ask over all time |
| Avg response confidence | Whether the model is generally certain or hedging |
| Confidence hi / mid / lo counts | Distribution of green / yellow / red responses |
| Low-conf negative rate | Are low-confidence answers getting thumbs-down? |
| Search helped rate | Is web search producing good results? |
| Hallucination rate | How often responses get flagged as hallucinated |
| Top flags | Most common negative-feedback tags |
| Search threshold | Current learned gate threshold (from behavior profile) |
| Uncertainty behavior | Current learned hedging preference |

↺ Refresh button reloads all stats live. Shows "No data yet" gracefully when the experience log is empty.

---

### Synthetic Test Suite

`test_synthetic.py` validates the feedback → experience log → behavior profile closed loop without requiring a running server or LLM:

```bash
python3 test_synthetic.py          # full 3-week simulation
python3 test_synthetic.py --week 2 # run only one week
python3 test_synthetic.py --clean  # wipe test data and exit
```

Three scenario weeks, each asserting the profile changes (or stays stable) as expected:

| Week | Scenario | Asserts |
|------|----------|---------|
| 1 | High confidence, positive feedback | Thresholds hold steady — no overreaction to good data |
| 2 | Low confidence, 70% negative | `search_threshold` drops (system learns to search sooner) |
| 3 | >15% hallucination tags | `uncertainty_behavior` stays `hedge`, threshold drops further |

Runs in ~2 seconds. Writes to `~/.local/share/3am/test_synthetic/` (isolated from real user data). Useful as a regression check if `update_from_experience()` is ever refactored.

---

### Improved "Delete All Memories"

Delete All now performs a **full user reset** — not just memory tables:

| Data wiped | Before MK13 | MK13 |
|-----------|------------|------|
| memories / clusters / embeddings / user profile | ✓ | ✓ |
| `experience_log` table | ✗ | ✓ |
| `research.json` | ✗ | ✓ |
| `behavior_profile.json` | ✗ | ✓ (reset to defaults) |

---

### New Settings Panel Sections

Four new groups in the Settings modal:

| Section | Controls |
|---------|----------|
| Decision Gate | Enable/disable toggle, Sensitivity slider (0–1) |
| Response Feedback | Enable/disable toggle, live feedback stats |
| Confidence Display | Enable/disable toggle |
| 📊 Analytics | Live stats from experience log (replaces old Memory Clusters view) |

---

### New API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/feedback` | Submit feedback (REST fallback) |
| `GET /api/feedback/stats` | Aggregated feedback stats |
| `GET /api/behavior-profile` | Current behavior profile |
| `PATCH /api/behavior-profile` | Update profile fields |
| `GET /api/tools` | List all tools (installed / code-ready / proposals) |
| `POST /api/tools/propose` | Stage 1 — propose a new tool concept |
| `POST /api/tools/{id}/generate` | Stage 2 — generate Python code for a proposal |
| `POST /api/tools/{id}/install` | Stage 3 — safety-check and install |
| `DELETE /api/tools/{id}` | Remove an installed tool |
| `GET /api/research/data` | Full research data (topics + insights + stats) for panel UI |
| `DELETE /api/research/topic/{idx}` | Delete a research topic by list index |
| `DELETE /api/research/insight/{idx}` | Delete a research insight by list index |
| `GET /api/analytics` | Comprehensive experience log analytics + behavior profile state |

---

### New Files

| File | Purpose |
|------|---------|
| `decision_gate.py` | Hybrid rule+LLM gate, `GateDecision` dataclass |
| `experience_log.py` | SQLite-backed interaction log |
| `behavior_profile.py` | Learned behavioral profile, introspection integration |
| `test_synthetic.py` | Standalone 3-week closed-loop simulation (no server needed) |

---

## What's New in MK12

### Self-Made Tools

The LLM can now design, generate, safety-check, and install its own Python tools at runtime — without a service restart. This is a three-stage approval pipeline where you stay in control at every step.

**Three-stage pipeline:**

```
Stage 1 — Concept approval
  ?propose-tool <description>
  OR: ?approve <#> on a "new_tool" suggestion → auto-creates proposal stub
  → LLM proposes: name, description, parameter schema, prompt hint
  → No code yet. User reviews the concept.
  → Next: ?approve-tool <id>

Stage 2 — Code review
  ?approve-tool <id>
  → LLM generates: async Python function implementing the tool
  → Code is validated before you see it (syntax + structure checks)
  → User reads the full implementation before anything runs
  → Next: ?install-tool <id>

Stage 3 — Install
  ?install-tool <id>
  → Import safety scan (blocks subprocess, socket, os, sys, eval, exec, etc.)
  → exec() into an isolated empty namespace — no access to server internals
  → Tool schema injected into every LLM API call immediately (hot-reload)
  → Persisted to installed_tools.json — survives service restarts
```

**Code validation (before you even see the generated code):**

```
LLM returns code
      │
      ▼
1. Empty check           — rejects blank responses
      │
      ▼
2. Signature check       — "async def tool_{name}(" must be present
      │
      ▼
3. compile() check       — Python syntax, with line number on failure
      │
      ▼
4. ast.parse() check     — async function exists at module level (not a comment)
      │
      ▼
5. Body check            — rejects pass / ... placeholder bodies
      │
      ▼
   status = "code_ready" → shown to user for review
```

**Safety (at install time):**

Pattern-based import block stops dangerous patterns including: `import subprocess`, `import socket`, `import threading`, `import multiprocessing`, `import ctypes`, `import os`, `import sys`, `import importlib`, `eval(`, `exec(`, `compile(`, `globals(`, `locals(`, `__builtins__`, and more. Any violation rejects the install and sets the tool status to `rejected`.

**Model-agnostic:** any llama.cpp-served model (Qwen, Llama, Mistral, etc.) automatically gets the full tool pipeline. Swap models by changing `LLM_URL` — the auto-detect picks up whatever `/v1/models` reports. Installed custom tools persist across model swaps.

**New commands:**
- `?tools` — list all custom tools (installed / code-ready / proposals)
- `?propose-tool <description>` — Stage 1: LLM proposes concept
- `?approve-tool <id>` — Stage 2: LLM generates + validates Python code
- `?install-tool <id>` — Stage 3: safety-check and hot-install
- `?remove-tool <name>` — uninstall a custom tool

**Capability gap → tool pipeline:** negative user feedback is routed to `analyze_capability_gaps()`, which can auto-suggest a `new_tool`. Approving it with `?approve <#>` creates a proposal stub — `?approve-tool <id>` then expands it into a full tool.

## Recent UI/UX Additions (MK12 updates)

### WebSocket Streaming
- Replaced Server-Sent Events (SSE) with a persistent **WebSocket** connection (`/ws/chat`) as the primary transport — lower latency, bidirectional
- **Mid-stream cancellation** — Stop button sends `{"type":"cancel"}` over the socket; the server raises `CancelledError` at the next await and sends back a `cancelled` frame
- **Typing indicator** — three-dot bounce animation shown before the first token arrives
- **Server push** — background tasks (research, self-improvement, introspection) send status toasts to the browser in real time via `loop.create_task(ws.send_json(...))`
- Exponential backoff reconnect (1 s → 30 s max); code 4001 (unauthorized) suppresses reconnect
- **`thinking` event type** — `{"type": "thinking", "content": "..."}` frames carry model reasoning tokens (inside `<think>...</think>`); rendered as collapsible sections, not mixed into the response text

### Memory Management
- **Export** (`GET /api/memory/export`) — downloads all memories, clusters, and user profile as a portable JSON file
- **Import** (`POST /api/memory/import`) — clears existing memory, re-embeds imported summaries in the background, rebuilds from the JSON file
- **Delete all** (`DELETE /api/memory`) — wipes memories, clusters, embeddings, and user profile in one call
- **Delete single** (`DELETE /api/memory/{id}`) — removes one memory and updates cluster refs
- All four actions are accessible from the Settings modal

### Chat UI Improvements
- **Copy buttons** on code blocks (hover to reveal) and assistant messages
- **Smart auto-scroll** — only follows new tokens if you're within 120 px of the bottom; won't jump if you've scrolled up to read
- **Scroll-to-bottom FAB** — floating ↓ button appears when scrolled away from the bottom
- **Markdown tables** — pipe-delimited tables render as styled `<table>` elements with header detection
- **Inline markdown** — clickable links (`[text](url)`), ~~strikethrough~~, and full bold/italic support inside lists, headings, and table cells
- **Conversation sidebar search** — client-side filter that hides non-matching conversations as you type
- **Message timestamps** — shown on hover; formatted as "Today at HH:MM", "Yesterday at HH:MM", or "Mon DD at HH:MM"

### Visual Polish
- **Message slide-in animation** — new messages fade and slide up 8 px into position
- **Language badge** on code blocks — detected language (e.g. `python`, `bash`) shown in the header bar alongside the Copy button
- **Subtle background tinting** — user messages carry a faint green wash; assistant messages a faint blue wash, matching their border accents
- **Input focus glow** — the chat input container gets a soft blue `box-shadow` ring when focused

### 3D Memory Map
A live star-map visualisation of the Torque Clustering memory system, accessible via the **✦** button in the sidebar header. Opens as a slide-out drawer panel next to the sidebar.

**Solar system metaphor** — clusters are *suns*, individual memories are *planets/moons* orbiting them:

- **Suns** — each cluster centroid rendered as a large, brightly glowing point; size scales linearly with member count (6 px at 1 member → 18 px at the 20-member split threshold); colour is a brightened version of the cluster hue; pulses slowly
- **Planets/moons** — individual memories rendered as smaller pulsing stars that physically orbit their sun in real time; each memory orbits on its own uniquely tilted 3D plane (golden-angle seeded) at the same radius as its original UMAP position; Kepler-ish speeds mean inner planets orbit faster than outer ones; raycasting (hover/click) stays accurate because the actual position buffer is updated each frame
- **Orbital lines** — thin semi-transparent lines connect every orbiting memory back to its cluster sun in real time, stretching and sweeping as planets move
- **Nebulae** — each cluster surrounded by a coloured halo (inner sphere + outer wispy shell that breathes); radius proportional to `sqrt(torque_mass)`; opacity raised for a more visible presence
- **Floating labels** — cluster theme text projected from 3D space onto an HTML overlay; hidden by default, fade in when the mouse nears a cluster centroid (80 px screen-space threshold), fade out on leave
- **Orbit controls** — drag to rotate, scroll to zoom, right-drag to pan; auto-rotates gently until you interact
- **Hover tooltip** — mouse over any memory star → shows the memory summary and category/priority metadata
- **Click to inspect** — click a memory star, cluster label, or legend row → detail panel lists up to 10 member memories with priority-coloured dots
- **Legend panel** — right-hand strip lists all clusters sorted by mass (most important first) with member counts
- **Dimensionality reduction** — uses UMAP (`umap-learn`) for organic, semantically faithful 3D projection; falls back to PCA (numpy SVD) if not installed — no data leaves the machine
- **Data refresh** — the ↺ button re-fetches `/api/memory/viz` on demand; the projection is also re-computed automatically after each nightly introspection cycle
- **Resizable panel** — drag the right edge of the drawer to any width; a **⤢ maximize** button instantly expands the panel to fill the full screen beside the sidebar (⤡ to restore); the Three.js renderer resizes in real time so the 3D scene always fills the canvas

**New dependency** (`requirements.txt`):
```
umap-learn>=0.5.0   # optional but recommended; falls back to numpy PCA
```
**Installed automatically** by `install.sh` (runs `pip install -r requirements.txt`).
If you have an existing venv from a previous install, update it manually:
```bash
pip install umap-learn
```

**New endpoint:** `GET /api/memory/viz` — runs UMAP/PCA in a thread pool executor, returns `{memories: [{x,y,z,summary,priority,cluster_id,...}], clusters: [{cx,cy,cz,theme,mass,count,...}]}`.

## What Was New in MK11 (SQLite + sqlite-vec Storage)

- **SQLite + sqlite-vec Storage** — replaces the monolithic memory.json:
  - `memory.db` (WAL mode) holds memories, clusters, and metadata
  - Embeddings live in a `vec_memories` sqlite-vec virtual table (`float[768]`, cosine distance metric)
  - Embeddings are **not loaded into RAM at startup** — fetched on demand
  - Dedup check: single `MATCH ? LIMIT 1` query instead of a Python loop over all memories
  - Conflict resolution: per-memory `MATCH` query instead of O(n²) Python loop
  - Clustering: batch-fetches all embeddings into numpy for the algorithm, releases them after
  - ACID writes, WAL concurrent reads, per-thread connections, 5 s busy timeout
  - **Auto-migrates** existing `memory.json` on first run (renames to `memory.json.migrated`)

## What Was New in MK10 (Torque Clustering)

- **Torque Clustering Memory** - Physics-inspired autonomous clustering:
  - Uses gravitational torque principles to discover natural cluster boundaries
  - Automatically determines optimal number of clusters (no manual thresholds)
  - More coherent clusters = better use of limited context window
  - Cluster "mass" indicates importance/centrality of memories
  - Re-clustering runs during scheduled introspection (3 AM)

- **Async Torque Clustering** — non-blocking memory reorganization:
  - CPU-bound work (distance matrix, TorqueClustering algorithm) runs in a thread pool executor — event loop stays live during computation
  - Theme generation runs async after the executor returns, so LLM calls don't block a thread
  - `_clustering_in_progress` flag pauses incoming chat messages during reorganization with a clear status message rather than silently queueing behind a blocked loop
  - `try/finally` guarantees the flag always clears even on errors

- **Improved Context Retrieval**:
  - Retrieves entire coherent clusters rather than fragmented snippets
  - Weights results by both similarity AND cluster mass (importance)
  - Lower similarity threshold (0.4) since clusters are more coherent

- **Compact User Profile** - Always-present identity context:
  - Priority-4/5 memories synthesized into a maximally dense key-value profile
  - Example: `Alex|SWE|Portland; Prefs:Neovim,ArchLinux,terse; Projects:ML/LLM`
  - Injected at the top of every system prompt (~30-50 tokens, fixed cost)
  - Leaves the cluster retrieval slots free for query-relevant facts
  - Regenerated during introspection whenever high-priority memories change

- **Deferred Multi-Fact Extraction** ("sleep processing"):
  - Chat time is now a fast path: raw conversation appended to `pending_conversations.jsonl`, cheap urgency check only
  - Urgency check immediately stores priority-5 identity facts (name, job, location, major life change) so they're available in the next message
  - New facts get a temporary cluster assignment at chat time (nearest existing cluster) — retrieval stays current without waiting for 3 AM
  - Full extraction runs during introspection (3 AM):
    - **Pass 1 (grouping)**: LLM reads one-liners of all pending conversations and groups related ones together — dynamic window, so a 10-message programming discussion becomes one group
    - **Pass 2 (extraction)**: Each group processed with full multi-fact array extraction — up to 8 facts per group, each with its own summary-based embedding
  - Facts from the same topic cluster more accurately because they share an embedding context
  - Per-fact embeddings on summary text (not conversation text) give better cluster placement

- **Time-Based Conflict Resolution** — runs during introspection, no LLM needed:
  - Same-category memories with cosine similarity ≥ 0.75 are treated as conflicting versions of the same fact
  - Older fact is pruned; newer fact is kept — assumes most recent data is most current
  - Runs before re-clustering so stale facts don't end up in the new cluster map
  - Examples: "User lives in Portland" + "User lives in Seattle" → keeps Seattle; "Works as SWE" + "Works as senior SWE at Google" → keeps the newer one

## What Was New in MK8

- **Web Interface** - ChatGPT-like browser UI:
  - Chat interface with streaming responses
  - Conversation history sidebar
  - New chat / continue existing conversations
  - Mobile-responsive design

- **Multi-User System**:
  - User registration with password authentication
  - Separate memory/history per user
  - Session-based auth with secure cookies

- **Split Introspection Loops** — two separate background loops with different cadences:

  **Nightly memory cycle (3 AM, opt-in)**
  - Process deferred conversations → extract facts → resolve conflicts
  - Torque Clustering (autonomous cluster rebuild/split/assign)
  - User profile regeneration, cluster theme refresh, cluster summarization
  - **Behavior Profile update** — reads a full day of experience log data; one bad session is noise, thirty interactions is a pattern
  - Heavy, potentially disruptive — chat is paused if clustering is in progress

  **Hourly idle cycle (always on, self-limiting)**
  - Error pattern analysis → self-improvement suggestions
  - Proactive research → web search on user interests
  - Self-research → LLM proposes ways to improve its own behaviour
  - Logs "nothing to do" and exits immediately if research + self-improve are both disabled
  - Uses its own `httpx` client — does not block or share state with the memory cycle
  - Skipped automatically if the memory cycle happens to overlap

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (Vanilla JS / HTML)                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Chat UI              │  History Sidebar  │  Settings       ││
│  │  - Messages           │  - Conversations  │  - Research     ││
│  │  - Streaming          │  - Search         │  - Self-improve ││
│  │  - Tool outputs       │  - Delete/rename  │  - Decision Gate││
│  │  - Think blocks       │                   │  - Confidence   ││
│  │  - 👍/👎 Feedback     │                   │  - Feedback     ││
│  │  - Confidence badges  │                   │                 ││
│  │  - LLM status LED     │                   │                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Web Server (FastAPI)                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Auth       │  │  Chat API          │  │  Scheduled Tasks        │  │
│  │  - Login    │  │  - Stream          │  │  - 3 AM memory cycle    │  │
│  │  - Register │  │  - Agentic loop    │  │  - Hourly idle cycle    │  │
│  │  - Sessions │  │  - _stream_one_turn│  │  - 3 AM: behavior upd   │  │
│  │             │  │  - History/Tools   │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Core Logic (MK13)                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Decision   │  │  Memory +   │  │  Introspection          │  │
│  │  Gate (NEW) │  │  Torque     │  │  (scheduled)            │  │
│  │             │  │  Clustering │  │  + behavior profile upd │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Experience │  │  Behavior   │  │  Custom Tool Registry   │  │
│  │  Log  (NEW) │  │  Profile    │  │  (LLM-generated tools,  │  │
│  │             │  │  (NEW)      │  │   per-user)             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  LLM Server (llama.cpp — any model)                             │
│  - Default: Qwen3-14B (Q4_K_M quantization)                     │
│  - Swap via LLM_URL env var — auto-detected on startup          │
│  - Logprobs requested per-call in the API payload (MK13: enables confidence scoring) │
└─────────────────────────────────────────────────────────────────┘
```

## Data Storage

Per-user data stored in `~/.local/share/3am/users/{user_id}/`:

| File | Description |
|------|-------------|
| `memory.db` | SQLite database: memories, clusters, vec_memories (embeddings), meta, **experience_log** (MK13) |
| `pending_conversations.jsonl` | Raw conversations queued for sleep processing (deleted after each introspection cycle) |
| `conversations/` | Chat history (one file per conversation) |
| `research.json` | Research topics and insights |
| `suggestions.json` | Self-improvement suggestions + custom tool proposals |
| `installed_tools.json` | LLM-generated custom tools (code + schemas) |
| `custom_prompt.txt` | User-specific prompt additions |
| `behavior_profile.json` | **MK13:** Learned behavioral profile (verbosity, thresholds, uncertainty style) |
| `settings.json` | User preferences (includes MK13: gate enabled, sensitivity, confidence/feedback toggles) |

Global data in `~/.local/share/3am/`:

| File | Description |
|------|-------------|
| `users.json` | User accounts (hashed passwords) |
| `scheduler_state.json` | Introspection schedule state |

## Requirements

- Python 3.10+
- llama.cpp with `llama-server` (for GPU inference)
- **8GB+ `/tmp` space** - PyTorch CUDA dependencies require ~3GB during installation. On low-RAM systems, the default tmpfs may be too small. Fix by adding to `/etc/fstab`:
  ```
  tmpfs  /tmp  tmpfs  rw,nosuid,nodev,size=8G  0 0
  ```
  Then `sudo mount -o remount /tmp` or reboot.

## Quick Start

```bash
# Test mode (downloads small 0.5B model)
./run-test.sh

# Full install (uses venv, creates systemd services)
./install.sh
```

## Testing

```bash
# Run with small model for quick testing
./run-test.sh
# Opens http://localhost:8000
```

This downloads Qwen2.5-0.5B (~350MB) for fast testing. Responses are instant but less intelligent than the full model.

## Configuration

Edit `~/.config/3am/config.json`:

```json
{
  "llm_server_url": "http://localhost:8080",
  "introspection_schedule": "03:00",
  "introspection_check_interval": 3600,
  "allow_registration": true,
  "session_timeout_hours": 24,
  "clustering_adjustment_factor": 0.5
}
```

### Tuning `clustering_adjustment_factor`

Controls how aggressively Torque Clustering draws cluster boundaries. After each
recluster the log prints a health summary — use it to decide whether to adjust:

```
[Clustering] 200 facts → 18 clusters | avg 11.1/cluster | min 3, max 26 | adjustment_factor=0.5
[Clustering] WARNING: 2 cluster(s) have >20 facts — consider increasing clustering_adjustment_factor
```

| Value | Effect | When to use |
|-------|--------|-------------|
| `0.3` | Conservative — fewer, broader clusters | Very noisy / rapidly changing facts |
| `0.5` | Standard *(default)* | Most users |
| `0.7` | Aggressive — more, tighter clusters | Rich fact set, topics are highly distinct |

Increase by ~0.05 if you see persistent oversized clusters. Decrease if you see many
single-fact clusters. After changing, trigger a full recluster to see the effect:

```bash
curl -X POST "http://localhost:8000/api/introspection/trigger?force_recluster=true" \
  -b "session=<your-session-cookie>"
```

## API Endpoints

### Auth
- `POST /api/auth/register` - Create account
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Current user

### Chat
- `WS  /ws/chat` - Primary streaming channel (WebSocket, session-cookie auth)
- `POST /api/chat` - Send message (streaming SSE, fallback)
- `GET /api/conversations` - List conversations
- `GET /api/conversations/{id}` - Get conversation
- `DELETE /api/conversations/{id}` - Delete conversation
- `PATCH /api/conversations/{id}` - Rename conversation

### Memory & Introspection
- `GET /api/memory` - View memory clusters
- `GET /api/memory/export` - Export all memories as JSON
- `POST /api/memory/import` - Import memories from JSON (re-embeds summaries)
- `DELETE /api/memory` - Delete all memories (MK13: also wipes experience log, research, behavior profile)
- `DELETE /api/memory/{id}` - Delete a single memory
- `GET /api/memory/viz` - 3D projected positions for memory map (UMAP/PCA, thread-pooled)
- `GET /api/introspection/stats` - Introspection statistics
- `POST /api/introspection/trigger` - Manual trigger
- `GET /api/research/status` - Research enable/quota status
- `POST /api/research/toggle` - Enable or disable proactive research
- `GET /api/research/findings` - Research insights (formatted text summary)
- `GET /api/research/findings/download` - Download all findings as .txt
- `GET /api/research/data` - Full topics + insights + stats for Research Panel UI
- `DELETE /api/research/topic/{idx}` - Delete a research topic by index
- `DELETE /api/research/insight/{idx}` - Delete a research insight by index
- `GET /api/suggestions` - Self-improvement suggestions

### Custom Tools (MK13)
- `GET /api/tools` - List all tools (installed / code-ready / proposals)
- `POST /api/tools/propose` - Stage 1: propose tool concept (body: `{description}`)
- `POST /api/tools/{id}/generate` - Stage 2: generate Python code for proposal
- `POST /api/tools/{id}/install` - Stage 3: safety-check and install
- `DELETE /api/tools/{id}` - Uninstall a custom tool

### Feedback, Analytics & Behavior Profile (MK13)
- `POST /api/feedback` - Submit thumbs up/down feedback
- `GET /api/feedback/stats` - Aggregated feedback statistics
- `GET /api/analytics` - Comprehensive experience log analytics + behavior profile (for Settings panel)
- `GET /api/behavior-profile` - Current learned behavior profile
- `PATCH /api/behavior-profile` - Update profile fields

### Document Ingestion (v1.2)
- `POST /api/ingest/document` - Ingest file (multipart) or URL; `persist=false` → ephemeral context, `persist=true` → proposition extraction + memory storage
- `DELETE /api/ingest/ephemeral` - Clear ephemeral document context for this session

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI web server |
| `auth.py` | User authentication |
| `scheduler.py` | Scheduled introspection |
| `data_security.py` | Fernet encryption: `DataEncryptor`, `derive_key_from_password`, `SecureUserData` |
| `memory.py` | Memory system with SQLite + sqlite-vec and Torque Clustering (MK11) |
| `introspection.py` | Introspection loop with Torque re-clustering + behavior profile update (MK13) |
| `research.py` | Research system (from MK7) |
| `self_improve.py` | Self-improvement + tool proposal/code-gen/install pipeline (MK12) |
| `tools.py` | Built-in tools + `CustomToolRegistry` with hot-reload (MK12) |
| `decision_gate.py` | **MK13:** Hybrid rule+LLM decision gate (`GateDecision`) |
| `experience_log.py` | **MK13:** SQLite-backed interaction + feedback log |
| `behavior_profile.py` | **MK13:** Learned behavioral profile, introspection-driven updates |
| `ingest.py` | **v1.2:** Document text extraction + LLM proposition extraction for ingestion |
| `test_synthetic.py` | **MK13:** Standalone closed-loop simulation — 3-week feedback scenario, no server needed |
| `torque_clustering/` | Torque Clustering algorithm (MK10) |
| `static/` | Frontend assets (HTML, CSS, JS) |
| `static/index.html` | Main HTML page |
| `static/css/style.css` | Stylesheet |
| `static/js/app.js` | Frontend JavaScript |
| `static/js/memory-map.js` | Three.js 3D memory star-map (ES module, loads via importmap) |
| `static/js/neural-visualizer.js` | Canvas brain-activity visualizer |
| `static/sw.js` | No-op service worker stub (suppresses browser 404 probes) |

## Torque Clustering

MK10/MK11 uses Torque Clustering for memory organization. This physics-inspired algorithm:

1. **Treats memories as particles with mass** - density in embedding space = mass
2. **Calculates gravitational torque** between memory groups
3. **Merges groups based on torque balance** - like galaxy mergers
4. **Automatically finds optimal cluster count** - no manual threshold needed

Benefits for LLM context:
- Clusters are **semantically coherent** (natural boundaries)
- High-mass clusters contain **core user interests**
- Better context = fewer wasted tokens on irrelevant memories

### Tiered Clustering Strategy

Clustering runs on multiple cadences:

| Tier | When | Cost | What it does |
|------|------|------|--------------|
| **Daytime incremental** | Every 8 hours (3× daily, idle loop) | O(new × clusters) | Assigns new unclustered facts to nearest existing cluster — same-day availability |
| **Nightly incremental** | Most nights (no oversized clusters) | O(new × clusters) | Same as daytime pass, catches anything the day passes missed |
| **Split** | Any cluster exceeds 20 facts | O(k × cluster_size²) | Re-clusters only the oversized cluster(s), leaves everything else alone |
| **Full** | Once per week (every 7 days) | O(n²) | Complete rebuild — catches topic drift, merges/splits anything |

`?force_recluster=true` on the API endpoint always runs a full recluster immediately.

After each recluster the log prints a health line:
```
[Clustering] 200 facts → 18 clusters | avg 11.1/cluster | min 3, max 26 | adjustment_factor=0.5
```

Citation: Jie Yang and Chin-Teng Lin, "Autonomous clustering by fast find of mass and
distance peaks," IEEE TPAMI, 2025. DOI: 10.1109/TPAMI.2025.3535743

## Differences from MK8

| Feature | MK8 | MK13 |
|---------|-----|------|
| Clustering | Threshold-based (0.75) | Torque Clustering (autonomous) |
| Cluster count | Manual threshold | Auto-determined |
| Cluster quality | Variable | More coherent |
| Context usage | May waste tokens | Better utilization |
| Re-clustering | Per-message | Periodic (introspection) |
| Memory extraction | Single fact per chat, at chat time | Multi-fact array, deferred to sleep |
| Extraction window | One conversation | Grouped by topic (dynamic window) |
| Embeddings | Conversation text | Per-fact summary text |
| Identity context | In cluster retrieval (can miss) | Always-present compact user profile |
| Chat-time LLM calls | 1 (classify) | 1 (urgency check, often returns fast) |
| Clustering execution | Synchronous, blocks event loop | CPU in thread pool, themes async |
| Chat during clustering | Silently blocked | Paused with status message |
| Conflict resolution | None | Time-based: newer fact supersedes older (same category, similarity ≥ 0.75) |
| Retrieval staleness | Per-message clustering | Incremental assignment at chat time, full recluster at 3 AM |
| Storage | Monolithic JSON (embeddings in RAM always) | SQLite + sqlite-vec (embeddings on disk, in RAM only during clustering) |
| Dedup check | Loop all memories, numpy cosine | Single sqlite-vec MATCH query |
| Custom tools | None | LLM-generated, 3-stage approval, hot-reload, persisted per-user |
| Tool UI | N/A | Sidebar panel (🔧) + `?` command fallback |
| Tool validation | N/A | Syntax + AST + import safety check before install |
| Model portability | Fixed model config | Any llama.cpp model, auto-detected |
| Pre-response reasoning | None | Decision Gate (rule+LLM hybrid, logprob confidence) |
| Confidence scoring | None | Per-token logprobs → coloured dot on each response |
| Feedback | None | 👍/👎 hover buttons, tag-based negative feedback |
| Adaptive behavior | None | Behavior Profile updated nightly (3 AM) from a full day of feedback patterns |
| Research UI | `?research` commands | 🔬 sidebar panel — scrollable list with per-entry delete |
| Performance visibility | None | 📊 Analytics in Settings — gate distribution, confidence, hallucination rate, live |
| Test suite | None | `test_synthetic.py` — 3-week closed-loop simulation, no server needed |

## Differences from MK7

| Feature | MK7 | MK8/MK10 |
|---------|-----|----------|
| Interface | Terminal/Dropdown | Web browser |
| Users | Single user | Multi-user with auth |
| Memory | Shared | Per-user |
| Introspection | Every 5 min | Hourly check, 3 AM run |
| History | File-based | Database with UI |
| Deployment | Local only | Can run as service |

## Security

- Passwords hashed with SHA-256 (bcrypt planned for future)
- Session tokens with secure cookies
- Per-user data isolation
- Optional HTTPS (configure reverse proxy)
- **Encryption at rest** — all sensitive fields in SQLite and JSON sidefiles encrypted with Fernet; key derived from login password, never persisted to disk
- **Personal/single-machine use** — designed for a single owner; not hardened for hosting for others or multi-tenant deployments

## Running as a Service

```bash
# Create systemd service
sudo cp 3am.service /etc/systemd/system/
sudo systemctl enable 3am
sudo systemctl start 3am
```

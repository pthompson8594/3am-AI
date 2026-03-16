# Memory System

The memory system provides **persistent, learning memory** using a physics-inspired clustering algorithm, a directed lane graph for associative retrieval, and a three-universe partitioning model that separates personal knowledge from world knowledge and learned patterns.

---

## Three Memory Universes

Every stored fact belongs to one of three universes:

| Universe | What goes here | Decay multiplier |
|---|---|---|
| **Episodic** | Personal facts about the user — preferences, identity, experiences, "I/my/we" knowledge | 1.0× |
| **Declarative** | World and technical knowledge — ingested documents, research findings, external facts | 0.3× |
| **Procedural** | Learned behavioural patterns — "when X do Y" rules, interaction preferences, adaptive behaviour | 0.1× |

**Classification:**
- Conversation facts are classified by the LLM at extraction time (`memory_type` field per fact)
- Ingested documents are always routed to `declarative` + `source_type=ingestion`
- Research findings are always routed to `declarative` + `source_type=research`
- The source_type override is enforced in `_store_fact` regardless of LLM output

Universe multipliers slow decay for knowledge that should persist longer. A declarative memory at priority 3 decays 3× slower than an episodic memory at the same priority.

---

## Priority-Based Decay

Base decay rates by priority level, then modified by universe multiplier and access resistance:

| Priority | Base rate | Lifespan (episodic) |
|----------|-----------|---------------------|
| 5 | 0.0005 | Years (core identity, name) |
| 4 | 0.0005 | Months (preferences, skills) |
| 3 | 0.005  | Weeks (patterns, projects) |
| 2 | 0.025  | Days (current tasks) |
| 1 | 0.1    | Hours (small talk) |

Full retention formula:

```
effective_rate = base_rate × universe_multiplier × (1 / (1 + access_count))
retention = exp(-effective_rate × age_hours)
```

**Access-count decay resistance:** each time a memory is recalled, `access_count` is incremented. This reduces `effective_rate`, making frequently-used memories progressively harder to forget. This counter is kept separate from Torque cluster geometry — a high-access memory like "user's name" shouldn't warp the semantic distance matrix and get pulled into unrelated cluster recall.

---

## Torque Clustering

Memories are grouped into clusters using a physics-inspired algorithm. Each memory has a position (embedding vector) and a mass. The algorithm finds natural cluster boundaries by cutting connections with the highest gravitational torque — no manual threshold for cluster count.

Cluster health is scored by:
- Average member retention (decay)
- Cluster size bonus
- Recency bonus (any member touched in last 24 h)
- Torque mass bonus (large, tight clusters)

Low-health clusters are pruned. Oversized clusters are split. Nightly full re-clustering rebuilds the entire map.

---

## Memory Lanes (Hyperspace Lanes)

A directed link graph connects related memories. Three link types:

| Type | Direction | When built |
|---|---|---|
| `semantic` | same-universe: bidirectional; cross-universe: one-way (new→neighbor) | At every memory write |
| `causal` | one-way: user memory → research finding | When research findings are stored |
| `sequential` | bidirectional | Between consecutive propositions within an ingested section |
| `hierarchical` | bidirectional | Proposition ↔ section summary ↔ document summary |

**Universe-aware directionality:** cross-universe semantic links are one-way only. This lets PPR walk *from* an episodic memory *into* declarative knowledge (e.g. the user mentioned Python, so their profile can surface Python documentation), but prevents declarative nodes from pulling unrelated episodic memories into a retrieval path seeded by world knowledge.

Constants: `LANE_MIN_SIM=0.55`, `LANE_MAX_SIM=0.92`, `LANE_MAX_K=8`, `CAUSAL_WEIGHT=0.7`

---

## Retrieval: Dynamic Context Allocation

`get_relevant_context` runs in four stages:

**1. Query classification**

`_classify_query` uses keyword heuristics to classify the query as `personal | factual | procedural | balanced` — no LLM call, runs in microseconds.

**2. Seeding (hybrid FTS5 + vec)**

BM25 full-text and vector similarity run in parallel. Results are merged with Reciprocal Rank Fusion (k=60) into 5 seed memories. If the embedder is already resident, vec runs; otherwise only FTS5 runs (the embedder is lazy-loaded only for writes).

**3. PPR expansion**

Personalized PageRank propagates activation through the lane graph from the seeds, surfacing associated memories that pure similarity search would miss. Fetches `total_budget + 6` candidates to give each universe enough to fill its quota.

**4. Universe budget split**

Results are partitioned into three buckets and capped per-universe according to `CONTEXT_BIAS`:

| Query type | Episodic | Declarative | Procedural | Total |
|---|---|---|---|---|
| personal    | 7 | 3 | 2 | 12 |
| factual     | 2 | 8 | 2 | 12 |
| procedural  | 2 | 3 | 7 | 12 |
| balanced    | 4 | 5 | 3 | 12 |

Access counts are incremented for all recalled memories at this point.

**Output format** (three labelled sections):

```
[MEMORY CONTEXT]

## Personal
  [Theme]
  - fact

## Knowledge
  [Theme]
  - [Research] fact

## Patterns
  [Theme]
  - fact

[Use this context naturally in your responses when relevant.]
```

Falls back to legacy cluster-centroid retrieval if no lane graph exists yet.

---

## Storage Layout

Per-user SQLite database at `~/.local/share/3am/users/{user_id}/memory.db`:

| Table | What it holds |
|---|---|
| `memories` | id, summary, category, priority, timestamp, cluster_id, memory_type, source_type, access_count, last_accessed, message, response |
| `clusters` | id, theme, priority, torque_mass, center_vector, message_refs |
| `vec_memories` | memory_id, embedding float[768] cosine (sqlite-vec virtual table) |
| `fts_memories` | memory_id, summary (FTS5 full-text index) |
| `memory_links` | source_id, target_id, weight, link_type, created_at |
| `meta` | key/value store (user profile, stats, flags) |

WAL journal mode. Float32 cosine distance via sqlite-vec. All text fields encrypted at rest (Fernet, key derived from login password).

---

## Workflow Summary

**Every conversation turn:**
1. LLM extracts durable facts, classifying each as `episodic | declarative | procedural`
2. Each fact is embedded and stored with universe routing enforced by source_type
3. Semantic lanes built to K nearest neighbours (universe-aware directionality)
4. Immediate cluster assignment via `assign_unclustered_memories` (background executor)
5. Priority ≥ 4 facts mark the profile dirty for regeneration

**Every 8 hours (lite recluster):**
- `assign_unclustered_memories` runs — new facts visible for retrieval same day

**At 3 AM (full cycle):**
1. Pending conversations grouped by topic
2. Multi-fact extraction (up to 8 facts per group)
3. Conflict resolution (newer wins, similarity ≥ 0.75)
4. Full Torque Clustering rebuild
5. `backfill_links` for any pre-lane orphan memories
6. Behavior profile update
7. Compact user profile regenerated

**Every hour (idle):**
- Proactive research → findings stored as `declarative` with causal lanes to source memories

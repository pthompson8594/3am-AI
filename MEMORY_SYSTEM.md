# Memory System

The memory system provides **persistent, learning memory** using a physics-inspired approach called **Torque Clustering**.

## Core Components

- **MemoryEntry**: Individual memories storing user message, assistant response, embedding vector, priority (1-5), and cluster assignment
- **MemoryCluster**: Groups of semantically related memories with a theme, center vector, and "torque mass" indicating importance
- **EmbeddingModel**: Uses `nomic-embed-text-v1.5` to convert text into 768-dim vectors for similarity matching

## Priority-Based Decay

Memories decay at different rates based on priority:

| Priority | Decay Rate | Lifespan |
|----------|------------|----------|
| 5 | 0.0005 | Years (core identity, name, profession) |
| 4 | 0.0005 | Months (preferences, skills) |
| 3 | 0.005 | Weeks (patterns, projects) |
| 2 | 0.025 | Days (current tasks) |
| 1 | 0.1 | Hours (small talk) |

## Torque Clustering

Clusters form autonomously based on "gravitational torque" between embeddings—no manual thresholds for cluster count. New memories either join existing clusters (if similarity > 0.5) or trigger re-clustering when too many are unclustered.

## Workflow

1. **Store**: After each exchange, LLM classifies what was learned and assigns priority
2. **Embed**: Text → vector via sentence-transformers  
3. **Cluster**: Assign to existing cluster or mark for re-clustering
4. **Decay**: Periodic cleanup removes low-retention memories
5. **Retrieve**: Query by embedding similarity, return relevant clusters for context injection

Data persists to `~/.local/share/llm-unified/users/{user_id}/memory.db` (SQLite).

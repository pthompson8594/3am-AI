# 3AM

**A self-evolving local AI that learns while you sleep.**

3AM is a fully local AI assistant that doesn't just answer questions — it remembers you, researches your interests, builds its own tools, and adapts its behavior over time. Everything runs on your hardware. No API keys, no cloud, no one training on your conversations.

At 3 AM each night, while you're asleep, it processes the day's conversations into long-term memory, resolves conflicting facts, re-clusters its knowledge, updates its behavior based on your feedback, and researches topics you care about. You wake up to a smarter assistant than the one you went to bed with.

---

## Why This Exists

Cloud AI assistants are stateless. Every conversation starts from zero. They don't remember what you told them last week, they can't learn your preferences over time, and every word you type trains someone else's model.

3AM takes a different approach: a small local model (14B parameters) wrapped in a system that makes it punch above its weight. The model provides reasoning. Everything else — memory, research, tool creation, behavioral adaptation — is handled by the architecture around it. The result is a personal AI that gets better the longer you use it, without getting bigger.

---

## Key Features

### Persistent Memory with Torque Clustering
Conversations are extracted into discrete facts, embedded, and organized using a physics-inspired clustering algorithm based on gravitational torque. The system automatically discovers natural topic boundaries — no manual thresholds. High-importance clusters get priority in context retrieval, so the model remembers what matters most.

### Sleep Processing
During the day, conversations are saved cheaply. At 3 AM, the system runs a full processing cycle: grouping related conversations, extracting multiple facts per topic, resolving conflicting information (newer facts supersede older ones), and rebuilding the cluster map. This keeps the chat path fast while doing heavy work in the background.

### Self-Created Tools
When the system identifies something it can't do, it can propose, generate, and install new Python tools — with you approving at every step. A three-stage pipeline (concept → code review → install) ensures nothing runs without your sign-off. Safety scanning blocks dangerous imports. Tools persist across restarts and model swaps.

### Decision Gate
Before generating a response, the system evaluates whether it actually knows enough to answer. Based on memory context and logprob-based confidence, it decides: answer directly, search the web first, or ask for clarification. This isn't the model guessing — it's a real confidence signal from token probabilities.

### Adaptive Behavior
Every response is logged with its confidence score, gate decision, and your feedback (👍/👎 with optional tags like "hallucinated" or "too verbose"). Nightly, the system analyzes a full day of these patterns and adjusts its behavior profile — when to search, when to hedge, how verbose to be. It learns from its mistakes without changing a single weight.

### Proactive Research
The system autonomously researches topics you've discussed, bringing back current information during idle hours. Research topics require your approval before being added. Results are browsable and deletable through the UI.

### 3D Memory Visualization
A live star-map shows your memory space — clusters as suns, individual memories as orbiting planets. Useful for spotting cluster health, interesting for understanding how your AI organizes what it knows about you.

---

## Architecture

```
Browser (Vanilla JS)          FastAPI Server              Local LLM
┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────┐
│ Chat UI          │    │ WebSocket streaming  │    │ llama.cpp        │
│ Memory star-map  │◄──►│ Decision Gate        │◄──►│ Qwen3-14B       │
│ Tool manager     │    │ Memory retrieval     │    │ (or any model)   │
│ Research panel   │    │ Experience logging   │    │ Vulkan/CUDA/CPU  │
│ Analytics        │    │ Tool execution       │    └──────────────────┘
└──────────────────┘    └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ SQLite + sqlite-vec  │
                        │ memory.db per user   │
                        │ Torque Clustering    │
                        │ Experience log       │
                        │ Behavior profile     │
                        └──────────────────────┘
```

**Nightly cycle (3 AM):** Fact extraction → conflict resolution → Torque Clustering → behavior profile update → user profile regeneration

**Hourly cycle (idle):** Research → self-improvement suggestions → capability gap analysis

---

## Quick Start

```bash
git clone https://github.com/pthompson8594/3am-ai.git
cd 3am-ai

# Quick test with a small model (~350MB download)
./run-test.sh

# Full install (venv, systemd services, your model)
./install.sh
```

Open `http://localhost:8000`, create an account, and start talking. The system starts learning immediately — priority facts (your name, job, location) are captured in real time. Everything else processes overnight.

### Requirements

- Python 3.10+
- llama.cpp with `llama-server`
- Any GGUF model (tested with Qwen3-14B Q4_K_M)
- GPU recommended (Vulkan, CUDA, or Metal) but CPU works

---

## How It Works

**During conversation:**
1. Your message arrives
2. Relevant memory clusters are retrieved via sqlite-vec similarity search
3. Decision Gate evaluates: answer / search / ask for clarification
4. Response is generated with confidence scoring (logprobs)
5. Conversation is queued for overnight processing
6. Priority-5 facts (identity info) are stored immediately

**At 3 AM:**
1. Pending conversations are grouped by topic
2. Multi-fact extraction runs on each group (up to 8 facts per group)
3. Per-fact embeddings are generated from summaries, not raw conversation
4. Conflicting facts are resolved (newer wins, cosine similarity ≥ 0.75)
5. Torque Clustering rebuilds the memory map
6. Behavior profile updates from the day's feedback patterns
7. Compact user profile is regenerated

**Every hour (when idle):**
1. Proactive research on approved topics
2. Self-improvement analysis
3. Capability gap detection → tool proposals

---

## The Self-Made Tools Pipeline

```
You: "Can you convert this CSV to JSON?"
3AM: "I don't have a tool for that, but I can build one."

Stage 1 — Propose     You review the concept (name, description, parameters)
Stage 2 — Generate    You review the actual Python code
Stage 3 — Install     Safety scan runs, tool goes live immediately

No restart needed. Tool persists across sessions.
```

Safety scan blocks: `subprocess`, `socket`, `os`, `sys`, `eval`, `exec`, `threading`, `ctypes`, `importlib`, and more. Tools execute in an isolated namespace with no access to server internals.

---

## Configuration

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

The `clustering_adjustment_factor` controls how aggressively clusters split. After each recluster, the log shows health stats:
```
[Clustering] 200 facts → 18 clusters | avg 11.1/cluster | min 3, max 26
```

---

## Data Storage

Everything stays on your machine. Per-user data in `~/.local/share/llm-unified/users/{user_id}/`:

| File | What it holds |
|------|---------------|
| `memory.db` | Memories, clusters, embeddings (sqlite-vec), experience log |
| `behavior_profile.json` | Learned behavioral preferences |
| `conversations/` | Full chat history |
| `research.json` | Research topics and findings |
| `installed_tools.json` | Custom tools you've approved |

---

## Model Compatibility

3AM is model-agnostic. It works with any model served by llama.cpp. Tested with:

- **Qwen3-14B** (Q4_K_M) — recommended for ~16GB systems
- **Qwen2.5-0.5B** — fast testing, minimal hardware

Swap models by changing the server URL. The system auto-detects whatever `/v1/models` reports. Custom tools, memory, and behavior profiles persist across model swaps.

---

## Security & Privacy

- All data stored locally as plaintext (designed for single-machine personal use)
- Passwords hashed with bcrypt
- Session tokens with secure cookies
- Per-user data isolation
- No telemetry, no external API calls (except user-approved web searches)
- **Your conversations never leave your machine**

---

## Background

3AM grew out of a simple question: can a small local model, wrapped in the right architecture, deliver a personal AI experience that rivals cloud services — without the cloud?

The answer is: for one user's daily needs, mostly yes. The model weights are fixed at 14B parameters, but the effective intelligence of the *system* grows continuously through accumulated memory, learned behavior, self-created tools, and proactive research. Context is a multiplier on capability.

The memory system uses Torque Clustering, a physics-inspired algorithm that treats memories as particles with mass and discovers natural cluster boundaries through gravitational torque balance. Based on: Yang & Lin, "Autonomous clustering by fast find of mass and distance peaks," IEEE TPAMI, 2025.

---

## License

The project code is released under **MIT License**.

The `torque_clustering/` module is adapted from work by Jie Yang and is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial, attribution required). See [Yang & Lin, IEEE TPAMI 2025](https://doi.org/10.1109/TPAMI.2024.3393173) for the original work.

---

## Status

This is a working personal project, actively used and developed by one person. It's not production software. Expect rough edges, opinionated defaults, and documentation that's still catching up to the code.

If you run into issues, open one. If you build something cool with it, I'd like to hear about it.
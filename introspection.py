#!/usr/bin/env python3
"""
Introspection System — Self-reflection and memory improvement with Torque Clustering.

Runs background tasks when the LLM server is idle:
- Torque Clustering: Autonomous cluster discovery using physics-inspired algorithm
- Conversation summarization: Compress verbose old memories
- Conflict detection: Find and flag contradictory facts
- Error journaling: Track tool failures and learn from them
- Proactive research: Learn more about user interests (if enabled)

Key change from MK8: Uses Torque Clustering for autonomous cluster discovery
instead of threshold-based merging. This produces more coherent clusters
that make better use of the limited context window.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable, Any
import threading

import numpy as np
import httpx

from memory import MemorySystem, MemoryCluster, MemoryEntry
from research import ResearchSystem
from self_improve import SelfImproveSystem

if TYPE_CHECKING:
    from data_security import DataEncryptor


DATA_DIR = Path.home() / ".local/share/3am"
ERROR_LOG_FILE = DATA_DIR / "error_journal.json"
INTROSPECTION_LOG_FILE = DATA_DIR / "introspection_log.json"
CONSOLIDATION_CONFIG_FILE = DATA_DIR / "consolidation_config.json"

# How often the lightweight idle cycle runs (research + self-improve)
IDLE_INTERVAL_SECONDS = 3600  # 1 hour

# How often the lite re-cluster pass runs (assign unclustered facts to existing clusters)
LITE_RECLUSTER_INTERVAL = 8 * 3600  # 3× per day


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation (introspection background loop)."""
    enabled: bool = False
    interval_seconds: int = 3600  # 1 hour
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
        }
    
    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "ConsolidationConfig":
        try:
            path = config_file or CONSOLIDATION_CONFIG_FILE
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception:
            pass
        return cls()
    
    def save(self, config_file: Optional[Path] = None):
        try:
            path = config_file or CONSOLIDATION_CONFIG_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception:
            pass

SUMMARIZATION_PROMPT = """Summarize these related memories about the user into a single concise fact.
Keep only the most important, current information. If facts conflict, prefer the more recent one.

Memories:
{memories}

Respond with JSON:
{{"summary": "<one concise sentence about the user>", "confidence": <0.0-1.0>}}"""

CONFLICT_DETECTION_PROMPT = """Do these two facts about the user contradict each other?

Fact 1: {fact1}
Fact 2: {fact2}

Respond with JSON:
{{"conflicts": <true/false>, "explanation": "<why they conflict or don't>", "resolution": "<which to keep if they conflict, or 'both' if compatible>"}}"""

THEME_UPDATE_PROMPT = """These memories are clustered together. Generate a better theme/title that captures what they have in common.

Current theme: {current_theme}

Memories:
{memories}

Respond with JSON:
{{"theme": "<2-5 word theme>", "category": "<personal_info|preferences|projects|skills|relationships|habits|other>"}}"""


@dataclass
class ErrorEntry:
    """A logged error/failure for learning."""
    timestamp: float
    tool_name: str
    error_type: str
    error_message: str
    context: str
    resolved: bool = False
    resolution: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "context": self.context,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ErrorEntry":
        return cls(**data)


@dataclass
class IntrospectionStats:
    """Statistics about introspection activities."""
    total_runs: int = 0
    memories_summarized: int = 0
    torque_clustering_runs: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    errors_logged: int = 0
    last_run: float = 0
    
    def to_dict(self) -> dict:
        return {
            "total_runs": self.total_runs,
            "memories_summarized": self.memories_summarized,
            "torque_clustering_runs": self.torque_clustering_runs,
            "conflicts_detected": self.conflicts_detected,
            "conflicts_resolved": self.conflicts_resolved,
            "errors_logged": self.errors_logged,
            "last_run": self.last_run,
        }


class ErrorJournal:
    """Tracks tool failures and errors for learning."""
    
    def __init__(self):
        self.entries: list[ErrorEntry] = []
        self._load()
    
    def _load(self):
        try:
            if ERROR_LOG_FILE.exists():
                with open(ERROR_LOG_FILE) as f:
                    data = json.load(f)
                self.entries = [ErrorEntry.from_dict(e) for e in data.get("entries", [])]
        except Exception as e:
            print(f"[ErrorJournal] Load error: {e}")
    
    def _save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {"entries": [e.to_dict() for e in self.entries[-100:]]}  # Keep last 100
            with open(ERROR_LOG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[ErrorJournal] Save error: {e}")
    
    def log_error(self, tool_name: str, error_type: str, error_message: str, context: str = ""):
        """Log a tool error for later analysis."""
        entry = ErrorEntry(
            timestamp=time.time(),
            tool_name=tool_name,
            error_type=error_type,
            error_message=error_message,
            context=context[:500],
        )
        self.entries.append(entry)
        self._save()
    
    def get_common_errors(self) -> dict[str, int]:
        """Get count of errors by tool."""
        counts = {}
        for entry in self.entries:
            key = f"{entry.tool_name}:{entry.error_type}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_unresolved(self) -> list[ErrorEntry]:
        """Get unresolved errors."""
        return [e for e in self.entries if not e.resolved]


class IntrospectionLoop:
    """
    Runs self-reflection tasks during idle time.
    
    Tasks are run in order of priority:
    1. Conflict detection (highest priority - data integrity)
    2. Memory consolidation (merge clusters)
    3. Conversation summarization (compress old memories)
    4. Error journal review (learn from failures)
    5. Proactive research (if enabled) - learn about user interests
    """
    
    def __init__(
        self,
        memory: MemorySystem,
        llm_url: str = "http://localhost:8080",
        llm_model_id: str = "qwen3-14b",
        on_status: Optional[Callable[[str], None]] = None,
        web_search_fn: Optional[Callable] = None,
        config_file: Optional[Path] = None,
        encryptor: Optional["DataEncryptor"] = None,
    ):
        self.memory = memory
        self.llm_url = llm_url
        self.llm_model_id = llm_model_id
        self.on_status = on_status or (lambda x: None)
        self.error_journal = ErrorJournal()
        # Optional hooks set by UserLLMCore after construction
        self.experience_log = None   # ExperienceLog | None
        self.behavior_profile = None # BehaviorProfile | None
        self.stats = IntrospectionStats()
        self._running = False
        self._running_idle = False
        self._in_progress = False      # True while run_cycle() / memory cycle is executing
        self._idle_in_progress = False  # True while run_idle_cycle() is executing
        self._task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._last_lite_recluster: float = 0  # epoch time of last incremental recluster

        # Consolidation config (opt-in)
        self._config_file = config_file
        self.config = ConsolidationConfig.load(config_file)

        # Research system for proactive learning
        self.research = ResearchSystem(
            llm_url=llm_url,
            llm_model_id=llm_model_id,
            web_search_fn=web_search_fn,
            on_status=on_status,
            encryptor=encryptor,
        )
        
        # Self-improvement system for LLM-suggested upgrades
        self.self_improve = SelfImproveSystem(
            llm_url=llm_url,
            llm_model_id=llm_model_id,
            on_status=on_status,
            web_search_fn=web_search_fn
        )
        
        self._load_stats()

        # Auto-start heavy consolidation loop if enabled (opt-in)
        if self.config.enabled:
            self.start_background()

        # Always start the lightweight hourly idle loop
        # (self-limiting: returns early if nothing to do)
        self._start_idle_loop()
    
    def _load_stats(self):
        try:
            if INTROSPECTION_LOG_FILE.exists():
                with open(INTROSPECTION_LOG_FILE) as f:
                    data = json.load(f)
                for key, value in data.get("stats", {}).items():
                    if hasattr(self.stats, key):
                        setattr(self.stats, key, value)
        except Exception:
            pass
    
    def _save_stats(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(INTROSPECTION_LOG_FILE, "w") as f:
                json.dump({"stats": self.stats.to_dict()}, f, indent=2)
        except Exception:
            pass
    
    async def _llm_request(self, prompt: str) -> Optional[dict]:
        """Make a request to the LLM for introspection tasks."""
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=60.0)
            
            response = await self._client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 200,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                }
            )
            
            result = response.json()
            if "choices" not in result:
                self.on_status(f"[Introspection] LLM response missing 'choices': {result}")
                return None
            
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
            
        except Exception as e:
            self.on_status(f"[Introspection] LLM error: {e}")
            return None
    
    async def summarize_cluster(self, cluster: MemoryCluster) -> bool:
        """Summarize memories in a cluster into a more concise form."""
        valid_refs = [ref for ref in cluster.message_refs if ref in self.memory.messages]
        
        if len(valid_refs) < 3:
            return False
        
        memories_text = "\n".join([
            f"- {self.memory.messages[ref].summary}"
            for ref in valid_refs[-5:]
        ])
        
        prompt = SUMMARIZATION_PROMPT.format(memories=memories_text)
        result = await self._llm_request(prompt)
        
        if result and result.get("summary"):
            cluster.theme = result["summary"]
            cluster.last_update = time.time()
            self.stats.memories_summarized += len(valid_refs)
            self.memory._save_cluster(cluster)
            return True
        
        return False
    
    async def detect_conflicts(self) -> list[tuple[str, str, str]]:
        """Detect conflicting facts in memory."""
        conflicts = []
        clusters = list(self.memory.clusters.values())
        
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i+1:]:
                similarity = MemorySystem._cosine_similarity(
                    cluster1.center_vector,
                    cluster2.center_vector
                )
                
                if 0.6 < similarity < 0.85:
                    prompt = CONFLICT_DETECTION_PROMPT.format(
                        fact1=cluster1.theme,
                        fact2=cluster2.theme
                    )
                    
                    result = await self._llm_request(prompt)
                    
                    if result and result.get("conflicts"):
                        conflicts.append((
                            cluster1.id,
                            cluster2.id,
                            result.get("explanation", "Unknown conflict")
                        ))
                        self.stats.conflicts_detected += 1
        
        return conflicts
    
    # NOTE: merge_similar_clusters() has been removed in MK10
    # Torque Clustering handles cluster discovery autonomously
    # See memory.run_torque_clustering() for the new approach
    
    async def update_cluster_themes(self) -> int:
        """Update cluster themes to better reflect their contents."""
        updated = 0
        
        for cluster in self.memory.clusters.values():
            valid_refs = [ref for ref in cluster.message_refs if ref in self.memory.messages]
            
            if len(valid_refs) < 2:
                continue
            
            age_hours = (time.time() - cluster.last_update) / 3600
            if age_hours < 24:
                continue
            
            memories_text = "\n".join([
                f"- {self.memory.messages[ref].summary}"
                for ref in valid_refs[-5:]
            ])
            
            prompt = THEME_UPDATE_PROMPT.format(
                current_theme=cluster.theme,
                memories=memories_text
            )
            
            result = await self._llm_request(prompt)
            
            if result and result.get("theme") and result["theme"] != cluster.theme:
                cluster.theme = result["theme"]
                cluster.last_update = time.time()
                self.memory._save_cluster(cluster)
                updated += 1
        
        return updated
    
    async def analyze_errors(self) -> Optional[str]:
        """Analyze error patterns and generate insights."""
        common_errors = self.error_journal.get_common_errors()
        
        if not common_errors:
            return None
        
        error_summary = "\n".join([
            f"- {error}: {count} times"
            for error, count in list(common_errors.items())[:5]
        ])
        
        self.on_status(f"[Introspection] Common errors:\n{error_summary}")
        
        return error_summary
    
    async def run_idle_cycle(self) -> dict[str, Any]:
        """Lightweight hourly cycle: error analysis, research, self-improvement.

        Runs independently from the 3 AM memory cycle. Returns early with
        nothing_to_do=True if nothing is enabled or there is nothing to act on.
        Skipped if the heavy memory cycle is currently running.
        """
        if self._in_progress:
            self.on_status("[Introspection] Idle cycle skipped (memory cycle in progress)")
            return {"skipped": True}

        has_research = self.research.is_enabled()
        has_improve = self.self_improve.is_enabled()
        has_errors = bool(self.error_journal.get_common_errors())

        if not has_research and not has_improve and not has_errors:
            self.on_status("[Introspection] Idle check: nothing to do (all disabled)")
            return {"nothing_to_do": True}

        self.on_status("[Introspection] Idle cycle starting...")
        results: dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Error analysis
            if has_errors:
                self.on_status("[Introspection] Analyzing error patterns...")
                await self.analyze_errors()
                results["errors_analyzed"] = True

            # Proactive research
            if has_research:
                self.on_status("[Introspection] Running research cycle...")
                cluster_info = [
                    {
                        "theme": c.theme,
                        "priority": c.priority,
                        "message_count": len(c.message_refs),
                    }
                    for c in self.memory.clusters.values()
                ]
                research_results = await self.research.run_research_cycle(
                    cluster_info, client, memory_system=self.memory
                )
                results["research"] = research_results

            # Self-improvement analysis
            if has_improve:
                self.on_status("[Introspection] Analyzing for self-improvement...")

                error_patterns = self.error_journal.get_common_errors()
                if error_patterns:
                    suggestion = await self.self_improve.analyze_errors(error_patterns, client)
                    if suggestion:
                        results["new_suggestion"] = suggestion.title

                feedback = self._load_recent_feedback(limit=50)
                failed_queries = [
                    f.get("query", "")
                    for f in feedback
                    if f.get("rating", 0) < 0 and f.get("query")
                ]
                if failed_queries:
                    suggestion = await self.self_improve.analyze_capability_gaps(
                        failed_queries, client
                    )
                    if suggestion:
                        results["capability_gap"] = suggestion.title

                if self.self_improve.config.allow_self_research:
                    self.on_status("[Introspection] Self-research phase...")
                    unresearched = [
                        t for t in self.self_improve.self_research_topics if not t.researched
                    ]
                    if not unresearched:
                        await self.self_improve.generate_self_research_topics(
                            error_patterns, feedback, client
                        )
                    suggestion = await self.self_improve.research_self_improvement(client)
                    if suggestion:
                        results["self_research_suggestion"] = suggestion.title

        if not results:
            self.on_status("[Introspection] Idle check: nothing to do")
            results["nothing_to_do"] = True
        else:
            self.on_status(f"[Introspection] Idle cycle complete: {results}")

        return results

    async def run_cycle(self) -> dict[str, Any]:
        """Run one full introspection cycle (memory + idle). Used for manual triggers."""
        self._in_progress = True
        try:
            results = await self._run_cycle_inner()
            idle = await self.run_idle_cycle()
            results["idle"] = idle
            return results
        finally:
            self._in_progress = False

    async def _run_cycle_inner(self) -> dict[str, Any]:
        """Inner implementation of run_cycle (called with _in_progress already set)."""
        self.stats.total_runs += 1
        self.stats.last_run = time.time()

        results = {
            "torque_clustering": None,
            "conflicts": [],
            "themes_updated": 0,
            "clusters_summarized": 0,
            "errors_analyzed": False,
        }

        self.on_status("[Introspection] Starting cycle...")

        # FIRST: Process pending conversations from the day into memory facts
        # Must run before clustering so new facts are included in tonight's cluster rebuild
        if self.memory.pending_file.exists():
            self.on_status("[Introspection] Processing pending conversations...")
            try:
                if not self._client:
                    self._client = httpx.AsyncClient(timeout=60.0)
                pending_result = await self.memory.process_pending_conversations(self._client)
                results["pending_processed"] = pending_result
                self.on_status(
                    f"[Introspection] Pending: {pending_result.get('conversations_processed', 0)} convs → "
                    f"{pending_result.get('facts_stored', 0)} facts in {pending_result.get('groups', 0)} groups"
                )
            except Exception as e:
                self.on_status(f"[Introspection] Pending processing error: {e}")
                results["pending_processed"] = {"status": "error", "reason": str(e)}

        # Resolve conflicting facts — time-based, no LLM: newer supersedes older
        # Runs after new facts are extracted so tonight's updates can supersede stale ones,
        # and before clustering so pruned memories don't end up in the new clusters.
        if len(self.memory.messages) >= 2:
            self.on_status("[Introspection] Resolving memory conflicts...")
            conflict_result = self.memory.resolve_conflicts()
            results["conflicts_resolved"] = conflict_result
            if conflict_result["pruned"] > 0:
                self.stats.conflicts_resolved += conflict_result["pruned"]
                self.on_status(
                    f"[Introspection] Pruned {conflict_result['pruned']} superseded memories"
                )

        # PRIMARY: Run Torque Clustering — always runs after introspection.
        # "auto" mode handles weekly full recluster, nightly split of oversized clusters,
        # and incremental assignment. Never skips silently.
        self.on_status("[Introspection] Running Torque Clustering (auto mode)...")
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=60.0)
            clustering_result = await self.memory.run_torque_clustering_async(self._client, mode="auto")
            results["torque_clustering"] = clustering_result
            if clustering_result.get("status") == "success":
                self.stats.torque_clustering_runs += 1
            self.on_status(
                f"[Introspection] Torque clustering: {clustering_result.get('status')} — "
                f"{clustering_result.get('new_clusters', clustering_result.get('total_clusters', 0))} clusters"
            )
        except Exception as e:
            self.on_status(f"[Introspection] Torque clustering error: {e}")
            results["torque_clustering"] = {"status": "error", "reason": str(e)}
        
        # Regenerate compact user profile if priority-4/5 memories changed
        if self.memory.is_profile_dirty():
            self.on_status("[Introspection] Updating user profile...")
            try:
                if not self._client:
                    self._client = httpx.AsyncClient(timeout=60.0)
                profile = await self.memory.regenerate_user_profile(self._client)
                results["profile_updated"] = bool(profile)
            except Exception as e:
                self.on_status(f"[Introspection] Profile update error: {e}")
                results["profile_updated"] = False

        # Check for conflicts between clusters
        if len(self.memory.clusters) > 1:
            self.on_status("[Introspection] Checking for conflicts...")
            results["conflicts"] = await self.detect_conflicts()
        
        # Update cluster themes using LLM
        if self.memory.clusters:
            self.on_status("[Introspection] Updating cluster themes...")
            results["themes_updated"] = await self.update_cluster_themes()
        
        # Summarize large clusters
        for cluster in list(self.memory.clusters.values())[:3]:
            if len(cluster.message_refs) >= 5:
                self.on_status(f"[Introspection] Summarizing cluster: {cluster.theme[:30]}...")
                if await self.summarize_cluster(cluster):
                    results["clusters_summarized"] += 1
        
        # Note: error analysis, research, and self-improvement run in the
        # separate hourly idle cycle (run_idle_cycle) rather than here,
        # so the heavy nightly memory work stays fast and focused.

        # MK13: update behavior profile from a full day's experience log data.
        # Runs here (nightly) rather than hourly so a single bad session can't
        # swing thresholds before enough data accumulates to distinguish noise
        # from a real pattern.
        if self.experience_log is not None and self.behavior_profile is not None:
            try:
                self.on_status("[Introspection] Updating behavior profile from experience log...")
                self.behavior_profile.update_from_experience(self.experience_log)
                results["behavior_profile_updated"] = True
            except Exception as e:
                self.on_status(f"[Introspection] Behavior profile update failed: {e}")

        self._save_stats()

        self.on_status(f"[Introspection] Cycle complete: {results}")

        return results
    
    async def _idle_loop(self):
        """Background loop that runs the heavy memory cycle (opt-in, typically 3 AM)."""
        while self._running:
            await asyncio.sleep(self.config.interval_seconds)

            if self._running and self.config.enabled:
                try:
                    await self._run_cycle_inner()
                except Exception as e:
                    self.on_status(f"[Introspection] Error in memory cycle: {e}")

    async def _run_lite_cluster_cycle(self):
        """
        Assign any unclustered facts to existing clusters (incremental mode).
        Fast: no LLM calls, no full rebuild. Runs up to 3× per day.
        Skipped if the heavy 3 AM cycle is currently running.
        """
        if self._in_progress:
            return
        if not self.memory.needs_reclustering():
            return
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                result = await self.memory.run_torque_clustering_async(client, mode="incremental")
            self._last_lite_recluster = time.time()
            self.on_status(f"[Introspection] Lite recluster complete: {result.get('status', 'done')}")
        except Exception as e:
            self.on_status(f"[Introspection] Lite recluster error: {e}")

    async def _light_idle_loop(self):
        """Hourly loop for lightweight research, self-improvement, and lite re-clustering."""
        while self._running_idle:
            await asyncio.sleep(IDLE_INTERVAL_SECONDS)
            if not self._running_idle:
                break

            # Lite re-cluster: assign unclustered facts every 8 hours
            if time.time() - self._last_lite_recluster > LITE_RECLUSTER_INTERVAL:
                await self._run_lite_cluster_cycle()

            self._idle_in_progress = True
            try:
                await self.run_idle_cycle()
            except Exception as e:
                self.on_status(f"[Introspection] Idle cycle error: {e}")
            finally:
                self._idle_in_progress = False

    def _start_idle_loop(self):
        """Start the lightweight hourly idle loop in a background thread."""
        if self._running_idle:
            return
        self._running_idle = True

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._light_idle_loop())
            finally:
                loop.close()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        self.on_status(
            f"[Introspection] Hourly idle loop started "
            f"(every {IDLE_INTERVAL_SECONDS // 60} min)"
        )

    def start_background(self):
        """Start the introspection loop in the background."""
        if self._running:
            return
        
        self._running = True
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._idle_loop())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        interval_mins = self.config.interval_seconds // 60
        self.on_status(f"[Consolidation] Background loop started (every {interval_mins} min)")
    
    def stop(self):
        """Stop all introspection loops."""
        self._running = False
        self._running_idle = False
        if self._client:
            asyncio.create_task(self._client.aclose())
            self._client = None
        self.on_status("[Introspection] Stopped")
    
    def is_enabled(self) -> bool:
        """Check if consolidation is enabled."""
        return self.config.enabled
    
    def enable(self):
        """Enable memory consolidation and start background loop."""
        self.config.enabled = True
        self.config.save(self._config_file)
        if not self._running:
            self.start_background()
        self.on_status("[Consolidation] Enabled - will consolidate memories hourly")
    
    def disable(self):
        """Disable memory consolidation."""
        self.config.enabled = False
        self.config.save(self._config_file)
        self.on_status("[Consolidation] Disabled")
    
    def log_tool_error(self, tool_name: str, error_type: str, message: str, context: str = ""):
        """Log a tool error for future analysis."""
        self.error_journal.log_error(tool_name, error_type, message, context)
        self.stats.errors_logged += 1
    
    def _load_recent_feedback(self, limit: int = 20) -> list[dict]:
        """Load recent feedback entries from feedback.jsonl."""
        feedback_file = DATA_DIR / "feedback.jsonl"
        entries = []
        try:
            if feedback_file.exists():
                lines = feedback_file.read_text().strip().splitlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return entries

    def get_stats(self) -> dict:
        """Get introspection statistics."""
        return {
            **self.stats.to_dict(),
            "error_count": len(self.error_journal.entries),
            "common_errors": self.error_journal.get_common_errors(),
            "research": self.research.get_stats(),
            "self_improve": self.self_improve.get_stats(),
        }

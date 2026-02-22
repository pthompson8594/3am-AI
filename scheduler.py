#!/usr/bin/env python3
"""
Scheduler - Scheduled introspection and background tasks.

Instead of running introspection every 5 minutes, MK8 uses a smarter approach:
1. Hourly quick check - Is there work to do?
2. Scheduled run time (default 3 AM) - Run full introspection to completion
3. Manual trigger available via API

This reduces unnecessary LLM usage while ensuring maintenance happens.
"""

import asyncio
import json
import time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import threading

DATA_DIR = Path.home() / ".local/share/llm-unified"
SCHEDULER_STATE_FILE = DATA_DIR / "scheduler_state.json"


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    check_interval_seconds: int = 3600  # 1 hour
    scheduled_hour: int = 3  # 3 AM
    scheduled_minute: int = 0
    enabled: bool = True
    run_on_startup: bool = False  # Run missed schedule on startup
    
    def to_dict(self) -> dict:
        return {
            "check_interval_seconds": self.check_interval_seconds,
            "scheduled_hour": self.scheduled_hour,
            "scheduled_minute": self.scheduled_minute,
            "enabled": self.enabled,
            "run_on_startup": self.run_on_startup,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerConfig":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_scheduled_time(self) -> dt_time:
        """Get the scheduled time as a time object."""
        return dt_time(hour=self.scheduled_hour, minute=self.scheduled_minute)


@dataclass
class SchedulerState:
    """Persistent state for the scheduler."""
    last_check: float = 0
    last_full_run: float = 0
    last_full_run_date: str = ""  # YYYY-MM-DD
    pending_work: bool = False
    running: bool = False
    
    def to_dict(self) -> dict:
        return {
            "last_check": self.last_check,
            "last_full_run": self.last_full_run,
            "last_full_run_date": self.last_full_run_date,
            "pending_work": self.pending_work,
            "running": self.running,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SchedulerState":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class CheckResult:
    """Result of an hourly check."""
    has_work: bool
    memory_clusters_needing_merge: int = 0
    memory_clusters_needing_summarization: int = 0
    unresearched_topics: int = 0
    pending_errors_to_analyze: int = 0
    
    def to_dict(self) -> dict:
        return {
            "has_work": self.has_work,
            "memory_clusters_needing_merge": self.memory_clusters_needing_merge,
            "memory_clusters_needing_summarization": self.memory_clusters_needing_summarization,
            "unresearched_topics": self.unresearched_topics,
            "pending_errors_to_analyze": self.pending_errors_to_analyze,
        }


class IntrospectionScheduler:
    """
    Manages scheduled introspection runs.
    
    - Hourly check: Quick scan to see if work is needed
    - Scheduled run: Full introspection at configured time (default 3 AM)
    - Run to completion: Once started, completes all pending work
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        on_status: Optional[Callable[[str], None]] = None
    ):
        self.config = config or SchedulerConfig()
        self.state = SchedulerState()
        self.on_status = on_status or (lambda x: None)
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        
        # Callbacks for actual introspection work
        self._check_callback: Optional[Callable[[], CheckResult]] = None
        self._run_callback: Optional[Callable[[], Any]] = None
        
        self._load_state()
    
    def _load_state(self):
        """Load scheduler state from disk."""
        try:
            if SCHEDULER_STATE_FILE.exists():
                with open(SCHEDULER_STATE_FILE) as f:
                    data = json.load(f)
                self.state = SchedulerState.from_dict(data.get("state", {}))
                if "config" in data:
                    self.config = SchedulerConfig.from_dict(data["config"])
        except Exception as e:
            self.on_status(f"[Scheduler] Load error: {e}")
    
    def _save_state(self):
        """Save scheduler state to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "state": self.state.to_dict(),
                "config": self.config.to_dict(),
            }
            with open(SCHEDULER_STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.on_status(f"[Scheduler] Save error: {e}")
    
    def set_callbacks(
        self,
        check_callback: Callable[[], CheckResult],
        run_callback: Callable[[], Any]
    ):
        """
        Set the callbacks for introspection work.
        
        Args:
            check_callback: Quick check function that returns CheckResult
            run_callback: Full introspection function
        """
        self._check_callback = check_callback
        self._run_callback = run_callback
    
    def _is_scheduled_time(self) -> bool:
        """Check if current time is within the scheduled window."""
        now = datetime.now()
        scheduled = self.config.get_scheduled_time()
        
        # Check if we're within 5 minutes of scheduled time
        current_minutes = now.hour * 60 + now.minute
        scheduled_minutes = scheduled.hour * 60 + scheduled.minute
        
        return abs(current_minutes - scheduled_minutes) <= 5
    
    def _should_run_today(self) -> bool:
        """Check if we should run introspection today."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.state.last_full_run_date != today
    
    async def quick_check(self) -> CheckResult:
        """
        Perform a quick check to see if introspection work is needed.
        
        This is a lightweight check that doesn't use the LLM.
        """
        self.on_status("[Scheduler] Performing quick check...")
        self.state.last_check = time.time()
        
        if self._check_callback:
            try:
                result = self._check_callback()
                self.state.pending_work = result.has_work
                self._save_state()
                self.on_status(f"[Scheduler] Check complete: work_pending={result.has_work}")
                return result
            except Exception as e:
                self.on_status(f"[Scheduler] Check error: {e}")
        
        return CheckResult(has_work=False)
    
    async def run_full_introspection(self) -> dict:
        """
        Run full introspection cycle to completion.
        
        This runs all pending introspection tasks including:
        - Memory consolidation
        - Conflict detection
        - Theme updates
        - Error analysis
        - Research (if enabled)
        - Self-improvement analysis (if enabled)
        """
        if self.state.running:
            self.on_status("[Scheduler] Introspection already running")
            return {"status": "already_running"}
        
        self.state.running = True
        self._save_state()
        
        self.on_status("[Scheduler] Starting full introspection run...")
        
        results = {"status": "completed", "cycles": 0}
        
        try:
            if self._run_callback:
                # Run until no more work
                max_cycles = 10  # Safety limit
                cycle = 0
                
                while cycle < max_cycles:
                    cycle += 1
                    self.on_status(f"[Scheduler] Running cycle {cycle}...")
                    
                    cycle_result = await self._run_callback()
                    results[f"cycle_{cycle}"] = cycle_result
                    
                    # Check if more work needed
                    check = await self.quick_check()
                    if not check.has_work:
                        break
                
                results["cycles"] = cycle
            
            # Update state
            self.state.last_full_run = time.time()
            self.state.last_full_run_date = datetime.now().strftime("%Y-%m-%d")
            self.state.pending_work = False
            
        except Exception as e:
            self.on_status(f"[Scheduler] Run error: {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        finally:
            self.state.running = False
            self._save_state()
        
        self.on_status(f"[Scheduler] Introspection complete: {results['cycles']} cycles")
        return results
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        self.on_status("[Scheduler] Loop started")
        
        while self._running:
            try:
                now = datetime.now()
                
                # Hourly check
                time_since_check = time.time() - self.state.last_check
                if time_since_check >= self.config.check_interval_seconds:
                    await self.quick_check()

                # Check if it's scheduled run time — always refresh pending_work
                # first so a stale hourly interval can't prevent the scheduled run
                if self.config.enabled and self._is_scheduled_time() and self._should_run_today():
                    await self.quick_check()

                # Check if it's scheduled run time
                if (
                    self.config.enabled
                    and self._is_scheduled_time()
                    and self._should_run_today()
                    and self.state.pending_work
                ):
                    self.on_status(f"[Scheduler] Scheduled run time ({now.strftime('%H:%M')})")
                    await self.run_full_introspection()
                
                # Sleep for 1 minute before next check
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.on_status(f"[Scheduler] Loop error: {e}")
                await asyncio.sleep(60)
        
        self.on_status("[Scheduler] Loop stopped")
    
    def start(self):
        """Start the scheduler background loop."""
        if self._running:
            return
        
        self._running = True
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._scheduler_loop())
            finally:
                loop.close()
        
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        
        self.on_status("[Scheduler] Started")
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self.on_status("[Scheduler] Stopping...")
    
    def get_status(self) -> dict:
        """Get current scheduler status."""
        now = datetime.now()
        scheduled = self.config.get_scheduled_time()
        
        # Calculate next scheduled run
        next_run = now.replace(
            hour=scheduled.hour,
            minute=scheduled.minute,
            second=0,
            microsecond=0
        )
        if next_run <= now:
            next_run += timedelta(days=1)
        
        return {
            "enabled": self.config.enabled,
            "running": self.state.running,
            "pending_work": self.state.pending_work,
            "last_check": datetime.fromtimestamp(self.state.last_check).isoformat() if self.state.last_check else None,
            "last_full_run": datetime.fromtimestamp(self.state.last_full_run).isoformat() if self.state.last_full_run else None,
            "next_scheduled_run": next_run.isoformat(),
            "scheduled_time": f"{scheduled.hour:02d}:{scheduled.minute:02d}",
            "check_interval_hours": self.config.check_interval_seconds / 3600,
        }
    
    def update_schedule(self, hour: int, minute: int = 0):
        """Update the scheduled run time."""
        self.config.scheduled_hour = hour
        self.config.scheduled_minute = minute
        self._save_state()
        self.on_status(f"[Scheduler] Schedule updated to {hour:02d}:{minute:02d}")
    
    async def trigger_manual_run(self) -> dict:
        """Manually trigger a full introspection run."""
        self.on_status("[Scheduler] Manual run triggered")
        return await self.run_full_introspection()

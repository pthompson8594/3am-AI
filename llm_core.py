#!/usr/bin/env python3
"""
LLM Core - Shared logic for LLM terminal interfaces.
Provides ServerManager, LLMCore, tool calling, and history management.
Now includes persistent memory system with embeddings and clustering.
MK7: Added introspection system for self-reflection during idle time.
"""

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any
import re
import httpx

from memory import MemorySystem
from introspection import IntrospectionLoop, ErrorJournal


# Valid tool names for inline parsing
VALID_TOOL_NAMES = {"execute_command", "create_file", "web_search", "read_file", "read_url", "clipboard", "system_info"}


def parse_inline_tool_calls(content: str) -> tuple[str, list]:
    """
    Parse inline JSON tool calls from LLM response content.
    Returns (remaining_content, tool_calls_list).
    
    Handles formats like:
      {"name": "web_search", "arguments": {"query": "..."}}
      {"name":"execute_command","arguments":{"command":"ls"}}
    """
    if not content:
        return content, []
    
    tool_calls = []
    remaining = content
    
    # Pattern to match JSON objects that look like tool calls
    # Matches {"name": "...", "arguments": {...}}
    pattern = r'\{["\s]*name["\s]*:["\s]*([^"]+)["\s]*,["\s]*arguments["\s]*:\s*(\{[^}]+\})\s*\}'
    
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    for i, match in enumerate(matches):
        name = match.group(1).strip().strip('"')
        args_str = match.group(2)
        
        # Only parse if it's a valid tool name
        if name not in VALID_TOOL_NAMES:
            continue
        
        try:
            args = json.loads(args_str)
            tool_calls.append({
                "id": f"inline_call_{i}",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args)
                }
            })
            # Remove the matched JSON from content
            remaining = remaining.replace(match.group(0), "").strip()
        except json.JSONDecodeError:
            continue
    
    return remaining, tool_calls


PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DATA_DIR = Path.home() / ".local/share/3am"
HISTORY_FILE = DATA_DIR / "history"
CONVERSATIONS_FILE = DATA_DIR / "conversations.jsonl"
FEEDBACK_FILE = DATA_DIR / "feedback.jsonl"
SETTINGS_FILE = Path.home() / ".config/llm-terminal/settings.json"

# LLM Server URL - can be local or remote
# Set via environment variable or in settings.json
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080")
IDLE_TIMEOUT = 300  # 5 minutes

MAX_HISTORY = 1000

SYSTEM_PROMPT_BASE = """You are a helpful terminal assistant. You MUST use your tools to help the user.

YOUR TOOLS:
- web_search: Search the web for ANY information request. USE THIS FREQUENTLY.
- execute_command: Run shell commands in the terminal
- create_file: Create or write files (shows preview first)
- read_file: Read local files (code, configs, logs)
- read_url: Fetch content from a specific URL
- clipboard: Read/write the system clipboard
- system_info: Get CPU, memory, disk, battery, network stats

HOW TO CALL TOOLS:
You can call tools by outputting JSON in this format:
{"name": "tool_name", "arguments": {"arg1": "value1"}}

Examples:
- {"name": "web_search", "arguments": {"query": "current date"}}
- {"name": "execute_command", "arguments": {"command": "ls -la"}}
- {"name": "system_info", "arguments": {"info_type": "all"}}

IMPORTANT RULES:
1. When the user asks about ANYTHING you're not 100% certain about, USE web_search.
2. When the user mentions "search", "find", "look up", "what is", "tell me about" - USE web_search.
3. When asked for recommendations or suggestions - USE web_search to find current options.
4. Do NOT say "I cannot search" - you CAN search using the web_search tool.
5. Be concise in your responses.
6. Use any memory context provided to personalize responses - remember what you know about this user.
7. When tool results contradict your assumptions, TRUST THE TOOL RESULTS - especially for time-sensitive info like versions, dates, current events, and facts about yourself.

For system commands (pacman, yay, apt, systemctl), use execute_command.
/no_think"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command in the terminal. Do NOT use this for creating files - use create_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file. The user will see the full content and must approve before the file is written.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to create"
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content of the file"
                    },
                    "executable": {
                        "type": "boolean",
                        "description": "Whether to make the file executable (chmod +x)"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information using Google. Use for recent events, news, facts you're unsure about, or anything requiring up-to-date information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a local file. Use for viewing code, configs, logs, or any text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read (can be relative to current directory or absolute)"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 200)"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch and extract text content from a URL/webpage. Use when you need the full content of a specific webpage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clipboard",
            "description": "Read from or write to the system clipboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "write"],
                        "description": "Whether to read from or write to clipboard"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (only needed for write action)"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "system_info",
            "description": "Get system information like CPU usage, memory, disk space, battery status, network info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["all", "cpu", "memory", "disk", "battery", "network"],
                        "description": "What type of system info to retrieve (default: all)"
                    }
                },
                "required": []
            }
        }
    }
]

DANGEROUS_PATTERNS = [
    "sudo", "rm -rf", "mkfs", "dd if=", ":(){", "chmod -R 777", "> /dev/sd",
    "tee ", "> ", ">> ", "cat >", "echo >", "echo >>",
    "cp ", "mv ", "mkdir ", "touch ",
    "cat <<", "cat>", "printf >",
]


@dataclass
class Settings:
    """Application settings with defaults."""
    position: str = "top"
    width_percent: int = 75
    height_percent: int = 30
    opacity: float = 0.95
    animation_duration: int = 250
    idle_timeout: int = 300
    
    # Remote LLM server (empty = use local, manages server automatically)
    llm_server_url: str = ""
    
    # Colors
    color_background: str = "#0d1117"
    color_text: str = "#c9d1d9"
    color_user_host: str = "#3fb950"
    color_directory: str = "#58a6ff"
    color_folders: str = "#79c0ff"
    color_assistant: str = "#bc8cff"
    color_dim: str = "#8b949e"
    color_status_ready: str = "#3fb950"
    color_status_busy: str = "#d29922"
    color_status_error: str = "#f85149"
    color_status_stopped: str = "#8b949e"
    gemini_api_key: str = ""
    
    @classmethod
    def load(cls) -> "Settings":
        """Load settings from file or return defaults."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE) as f:
                    data = json.load(f)
                # Flatten colors dict if present
                if "colors" in data:
                    colors = data.pop("colors")
                    for key, value in colors.items():
                        data[f"color_{key}"] = value
                return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k.startswith("color_")})
        except Exception as e:
            print(f"Error loading settings: {e}")
        return cls()
    
    def save(self):
        """Save settings to file."""
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Structure colors in nested dict for readability
            data = {
                "position": self.position,
                "width_percent": self.width_percent,
                "height_percent": self.height_percent,
                "opacity": self.opacity,
                "animation_duration": self.animation_duration,
                "idle_timeout": self.idle_timeout,
                "llm_server_url": self.llm_server_url,
                "colors": {
                    "background": self.color_background,
                    "text": self.color_text,
                    "user_host": self.color_user_host,
                    "directory": self.color_directory,
                    "folders": self.color_folders,
                    "assistant": self.color_assistant,
                    "dim": self.color_dim,
                    "status_ready": self.color_status_ready,
                    "status_busy": self.color_status_busy,
                    "status_error": self.color_status_error,
                    "status_stopped": self.color_status_stopped,
                }
            }
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")


class ServerManager:
    """Manages the LLM server lifecycle."""
    
    def __init__(self, on_status_change: Optional[Callable[[str], None]] = None):
        self.process = None
        self._status = "stopped"
        self._idle_timer: Optional[threading.Timer] = None
        self.on_status_change = on_status_change
    
    @property
    def status(self) -> str:
        return self._status
    
    @status.setter
    def status(self, value: str):
        self._status = value
        if self.on_status_change:
            self.on_status_change(value)
    
    def _is_server_running(self) -> bool:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server.*--port"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_server_pids(self) -> list[int]:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "llama-server.*--port"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return [int(pid) for pid in result.stdout.strip().split('\n') if pid]
        except:
            pass
        return []
    
    def check_health(self) -> bool:
        """Check if server is healthy."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{LLM_URL}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    return data.get("status") == "ok"
        except:
            pass
        return False
    
    def start(self, callback: Optional[Callable[[str], None]] = None) -> bool:
        """Start the LLM server. Returns True if successful."""
        if self._is_server_running():
            self.status = "running"
            return True
        
        self.cancel_scheduled_stop()
        self.status = "starting"
        
        if callback:
            callback("Starting LLM server...")
        
        script = SCRIPTS_DIR / "start-llm-server.sh"
        if not script.exists():
            if callback:
                callback(f"Error: Server script not found: {script}")
            self.status = "error"
            return False
        
        self.process = subprocess.Popen(
            ["bash", str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to be ready
        for i in range(60):
            time.sleep(1)
            if self.check_health():
                self.status = "running"
                if callback:
                    callback("LLM server ready!")
                return True
            if callback:
                callback(f"Waiting for server... {i+1}s")
        
        if callback:
            callback("Server failed to start within 60 seconds")
        self.status = "error"
        return False
    
    def stop(self, callback: Optional[Callable[[str], None]] = None):
        """Stop the LLM server."""
        self.cancel_scheduled_stop()
        
        pids = self._get_server_pids()
        if not pids:
            self.status = "stopped"
            return
        
        if callback:
            callback("Stopping LLM server...")
        
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        
        time.sleep(0.5)
        
        # Force kill if still running
        pids = self._get_server_pids()
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        self.status = "stopped"
        if callback:
            callback("LLM server stopped.")
    
    def schedule_stop(self, seconds: int = IDLE_TIMEOUT):
        """Schedule server stop after idle timeout."""
        self.cancel_scheduled_stop()
        self._idle_timer = threading.Timer(seconds, self.stop)
        self._idle_timer.daemon = True
        self._idle_timer.start()
    
    def cancel_scheduled_stop(self):
        """Cancel any scheduled stop."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None


@dataclass
class ToolResult:
    """Result of a tool execution."""
    output: str
    success: bool
    needs_approval: bool = False
    approval_type: Optional[str] = None  # "command" or "file"
    approval_data: Optional[dict] = None


class OutputHandler:
    """Abstract output handler - implemented by CLI and GUI."""
    
    def print_text(self, text: str, color: Optional[str] = None):
        raise NotImplementedError
    
    def print_line(self, text: str = "", color: Optional[str] = None):
        self.print_text(text + "\n", color)
    
    def print_command_output(self, output: str, from_tool: bool = False):
        raise NotImplementedError
    
    def set_pending_tool_command(self, command: str):
        """Set the command for the next tool output (for collapsible display)."""
        pass  # Override in GUI implementation
    
    def print_prompt(self, user: str, host: str, directory: str, settings: Settings):
        raise NotImplementedError
    
    def print_assistant(self, text: str, settings: Settings):
        raise NotImplementedError
    
    def print_assistant_token(self, token: str, settings: Settings):
        """Print a single streamed token from the assistant."""
        # Default implementation just prints the token
        self.print_text(token, settings.color_assistant)
    
    def start_assistant_response(self, settings: Settings):
        """Called before streaming assistant response."""
        pass  # Override if needed
    
    def end_assistant_response(self, settings: Settings):
        """Called after streaming assistant response completes."""
        self.print_text("\n")  # Default: add newline
    
    def print_status(self, message: str, color: Optional[str] = None):
        raise NotImplementedError
    
    def show_spinner(self, message: str = "thinking"):
        raise NotImplementedError
    
    def hide_spinner(self):
        raise NotImplementedError
    
    def get_approval(self, prompt: str) -> str:
        """Get user approval. Returns 'y', 'n', or 'c' (cancel)."""
        raise NotImplementedError
    
    def show_file_preview(self, path: str, content: str, executable: bool) -> str:
        """Show file preview and get approval."""
        raise NotImplementedError
    
    def clear(self):
        raise NotImplementedError


class LLMCore:
    """Core LLM functionality shared between CLI and GUI."""
    
    def __init__(self, output: OutputHandler, settings: Optional[Settings] = None,
                 interactive_command_handler: Optional[Callable[[str], None]] = None):
        self.output = output
        self.settings = settings or Settings.load()
        
        # Determine LLM server URL (settings > env var > default)
        self.llm_url = self.settings.llm_server_url or os.environ.get("LLM_URL") or "http://localhost:8080"
        self.is_remote_server = self.settings.llm_server_url != ""
        
        # Only manage server lifecycle for local servers
        self.server = ServerManager(on_status_change=self._on_server_status) if not self.is_remote_server else None
        
        self.client = httpx.AsyncClient(timeout=300.0)
        self.conversation: list[dict] = []
        self.command_history: list[str] = []
        self.history_index: int = -1
        self.cwd = os.getcwd()
        self.last_command = ""
        self.last_output = ""
        self.pending_approval: Optional[dict] = None
        self._pending_interactive: Optional[dict] = None  # For package manager commands
        self._interactive_command_handler = interactive_command_handler  # Callback for sudo/interactive commands
        
        self.memory = MemorySystem(llm_url=self.llm_url)
        self._last_exchange: Optional[dict] = None  # For feedback tracking
        
        # MK7: Introspection system for self-reflection during idle time
        # We pass web_search as a callable so research can use it
        self.introspection = IntrospectionLoop(
            memory=self.memory,
            llm_url=self.llm_url,
            llm_model_id=self.model_id,
            on_status=lambda msg: print(msg),  # Can be overridden
            web_search_fn=self._web_search_for_research
        )
        self._introspection_started = False
        
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_history()
    
    def _on_server_status(self, status: str):
        """Called when server status changes."""
        pass  # Override in subclass if needed
    
    def _check_server_health(self) -> bool:
        """Check if LLM server is healthy and reachable."""
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.llm_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read())
                    return data.get("status") == "ok"
        except:
            pass
        return False
    
    def _load_history(self):
        """Load command history from file."""
        try:
            if HISTORY_FILE.exists():
                self.command_history = HISTORY_FILE.read_text().strip().split('\n')[-MAX_HISTORY:]
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def save_history(self):
        """Save command history to file."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            HISTORY_FILE.write_text('\n'.join(self.command_history[-MAX_HISTORY:]))
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def record_feedback(self, rating: int, comment: str = "") -> bool:
        """
        Record feedback on the last LLM response.
        rating: 1 (good), -1 (bad)
        comment: optional explanation of what went wrong
        """
        if not self._last_exchange:
            return False
        
        feedback = {
            "timestamp": time.time(),
            "query": self._last_exchange.get("query", ""),
            "response": self._last_exchange.get("response", ""),
            "rating": rating,
            "comment": comment,
        }
        
        try:
            with open(FEEDBACK_FILE, "a") as f:
                f.write(json.dumps(feedback) + "\n")
            return True
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def handle_feedback_command(self, cmd: str) -> Optional[str]:
        """
        Handle feedback commands. Returns message if handled, None otherwise.
        Commands: ?+, ?-, ?feedback <text>
        """
        cmd = cmd.strip()
        
        if cmd == "?+":
            if self.record_feedback(1):
                return "👍 Feedback recorded"
            return "No recent response to rate"
        
        if cmd == "?-":
            if self.record_feedback(-1):
                return "👎 Feedback recorded"
            return "No recent response to rate"
        
        if cmd.startswith("?feedback "):
            comment = cmd[10:].strip()
            if comment:
                if self.record_feedback(-1, comment):
                    return f"📝 Feedback recorded: {comment[:50]}..."
                return "No recent response to rate"
            return "Usage: ?feedback <what went wrong>"
        
        return None
    
    def handle_introspection_command(self, cmd: str) -> Optional[str]:
        """
        Handle introspection commands. Returns message if handled, None otherwise.
        Commands: ?introspect, ?memory, ?errors
        """
        cmd = cmd.strip()
        
        if cmd == "?introspect":
            stats = self.get_introspection_stats()
            lines = [
                "=== Introspection Stats ===",
                f"Total runs: {stats['total_runs']}",
                f"Memories summarized: {stats['memories_summarized']}",
                f"Clusters merged: {stats['clusters_merged']}",
                f"Conflicts detected: {stats['conflicts_detected']}",
                f"Errors logged: {stats['errors_logged']}",
            ]
            if stats.get('last_run'):
                age = (time.time() - stats['last_run']) / 60
                lines.append(f"Last run: {age:.1f} minutes ago")
            return "\n".join(lines)
        
        if cmd == "?memory":
            return self.memory.get_visualization()
        
        if cmd == "?errors":
            common = self.introspection.error_journal.get_common_errors()
            if not common:
                return "No errors logged"
            lines = ["=== Common Errors ==="]
            for error, count in list(common.items())[:10]:
                lines.append(f"  {error}: {count}x")
            return "\n".join(lines)
        
        if cmd == "?reflect":
            return "ASYNC_INTROSPECT"  # Signal to run async introspection
        
        if cmd == "?recluster":
            return "ASYNC_RECLUSTER"  # Signal to force full reclustering
        
        # Research commands
        if cmd == "?research on":
            self.introspection.research.enable()
            return "🔬 Research mode enabled (5 searches/day during idle time)"
        
        if cmd == "?research off":
            self.introspection.research.disable()
            return "🔬 Research mode disabled"
        
        if cmd == "?research" or cmd == "?research status":
            stats = self.introspection.research.get_stats()
            lines = [
                "=== Research Status ===",
                f"Enabled: {'✓' if stats['enabled'] else '✗'}",
                f"Quota: {stats['quota_remaining']}/{stats['daily_limit']} remaining today",
                f"Topics queued: {stats['total_topics']} ({stats['researched_topics']} researched)",
                f"Insights: {stats['total_insights']} ({stats['unshared_insights']} unshared)",
            ]
            return "\n".join(lines)
        
        if cmd == "?findings":
            return self.introspection.research.get_findings_summary()
        
        if cmd.startswith("?learn "):
            topic = cmd[7:].strip()
            if topic:
                self.introspection.research.add_manual_topic(topic)
                return f"📚 Queued for research: {topic}"
            return "Usage: ?learn <topic to research>"
        
        # Self-improvement commands
        if cmd == "?improve on":
            self.introspection.self_improve.enable()
            return "🧠 Self-improvement enabled - LLM will suggest its own upgrades"
        
        if cmd == "?improve off":
            self.introspection.self_improve.disable()
            return "🧠 Self-improvement disabled"
        
        if cmd == "?improve prompts on":
            self.introspection.self_improve.enable_prompt_changes()
            return "📝 Prompt modifications enabled - LLM can suggest system prompt changes"
        
        if cmd == "?improve prompts off":
            self.introspection.self_improve.disable_prompt_changes()
            return "📝 Prompt modifications disabled"
        
        if cmd == "?improve" or cmd == "?improve status":
            stats = self.introspection.self_improve.get_stats()
            lines = [
                "=== Self-Improvement Status ===",
                f"Enabled: {'✓' if stats['enabled'] else '✗'}",
                f"Prompt changes: {'✓' if stats['allow_prompt_changes'] else '✗'}",
                f"Suggestions: {stats['pending']} pending, {stats['implemented']} implemented",
            ]
            return "\n".join(lines)
        
        if cmd == "?suggestions":
            return self.introspection.self_improve.get_suggestions_summary()
        
        if cmd.startswith("?approve "):
            n = cmd[9:].strip()
            if self.introspection.self_improve.approve_suggestion(n):
                return "✓ Suggestion approved"
            return "Suggestion not found"
        
        if cmd.startswith("?dismiss "):
            n = cmd[9:].strip()
            if self.introspection.self_improve.dismiss_suggestion(n):
                return "✗ Suggestion dismissed"
            return "Suggestion not found"
        
        if cmd.startswith("?implemented "):
            n = cmd[13:].strip()
            if self.introspection.self_improve.mark_implemented(n):
                return "✓ Marked as implemented"
            return "Suggestion not found"
        
        if cmd == "?prompt":
            custom = self.introspection.self_improve.get_custom_prompt()
            if custom:
                return f"=== Custom Prompt Additions ===\n{custom}"
            return "No custom prompt additions yet."
        
        # Self-research commands
        if cmd == "?selfresearch on":
            self.introspection.self_improve.enable_self_research()
            return "🔬 Self-research enabled - LLM will research how to improve itself"
        
        if cmd == "?selfresearch off":
            self.introspection.self_improve.disable_self_research()
            return "🔬 Self-research disabled"
        
        if cmd == "?selfresearch":
            return self.introspection.self_improve.get_self_research_summary()
        
        return None
    
    def add_to_history(self, command: str):
        """Add command to history."""
        if command and (not self.command_history or self.command_history[-1] != command):
            self.command_history.append(command)
            self.history_index = -1
    
    def navigate_history(self, direction: int) -> Optional[str]:
        """Navigate command history. Returns command or None."""
        if not self.command_history:
            return None
        
        new_index = self.history_index + direction
        if 0 <= new_index < len(self.command_history):
            self.history_index = new_index
            return self.command_history[-(self.history_index + 1)]
        elif new_index < 0:
            self.history_index = -1
            return ""
        return None
    
    def _log_conversation(self, messages: list, response: str):
        """Log conversation to file."""
        try:
            with open(CONVERSATIONS_FILE, "a") as f:
                f.write(json.dumps({"messages": messages, "response": response}) + "\n")
        except Exception as e:
            print(f"Error logging conversation: {e}")
    
    def _is_dangerous(self, command: str) -> bool:
        """Check if command is potentially dangerous."""
        return any(p in command for p in DANGEROUS_PATTERNS)
    
    def change_directory(self, path: str) -> bool:
        """Change current directory. Returns True if successful."""
        try:
            expanded = os.path.expanduser(path) if path else os.path.expanduser("~")
            if not os.path.isabs(expanded):
                expanded = os.path.join(self.cwd, expanded)
            expanded = os.path.normpath(expanded)
            
            if os.path.isdir(expanded):
                self.cwd = expanded
                os.chdir(expanded)
                return True
            else:
                self.output.print_line(f"cd: no such directory: {path}", self.settings.color_status_error)
                return False
        except Exception as e:
            self.output.print_line(f"cd: {e}", self.settings.color_status_error)
            return False
    
    def _needs_interactive_terminal(self, command: str) -> bool:
        """Detect if a command needs an interactive terminal (PTY)."""
        # Commands that always need a TTY for password/authentication
        tty_required = ["sudo ", "su ", "passwd", "ssh ", "doas "]
        
        # Commands that use full-screen/ncurses UI
        fullscreen_cmds = [
            "vim", "nvim", "nano", "emacs", "vi ",
            "htop", "btop", "top", "less", "more", "man ",
            "ncdu", "mc", "ranger", "nnn",
            "tmux", "screen",
        ]
        
        # Package managers that prompt for confirmation
        pkg_managers = [
            "pacman -S", "pacman -R", "pacman -U", "pacman -Syu",
            "yay -S", "yay -R", "yay -Syu", "yay",
            "paru -S", "paru -R", "paru -Syu",
            "apt install", "apt remove", "apt upgrade", "apt full-upgrade",
            "dnf install", "dnf remove", "dnf upgrade",
            "zypper install", "zypper remove",
        ]
        
        # System commands that might need interaction
        system_cmds = ["systemctl", "reboot", "shutdown", "poweroff"]
        
        cmd_lower = command.lower()
        cmd_parts = command.split()
        first_word = cmd_parts[0] if cmd_parts else ""
        
        # Check TTY required commands
        for pattern in tty_required:
            if command.startswith(pattern):
                return True
        
        # Check full-screen commands (match first word or after sudo)
        for pattern in fullscreen_cmds:
            if first_word == pattern.strip() or (len(cmd_parts) > 1 and cmd_parts[1] == pattern.strip()):
                return True
        
        # Check package managers
        for pattern in pkg_managers:
            if pattern in command:
                return True
        
        # Check system commands
        for pattern in system_cmds:
            if pattern in command:
                return True
        
        return False
    
    async def execute_command(self, command: str, track: bool = True, from_llm: bool = False, 
                              force_interactive: bool = False) -> str:
        """Execute a shell command. Returns output.
        
        Args:
            command: The command to execute
            track: Whether to track as last command
            from_llm: Whether this was initiated by the LLM (tool call)
            force_interactive: Force use of interactive PTY (user typed !command)
        """
        needs_interactive = force_interactive or self._needs_interactive_terminal(command)
        
        # For interactive commands from LLM, don't execute - just suggest
        if needs_interactive and from_llm:
            self.output.print_line("")
            self.output.print_line(f"[Suggested command - press Enter to run]", self.settings.color_status_busy)
            self._suggested_command = command
            return f"[SUGGESTED_COMMAND:{command}]"
        
        # Check for dangerous commands only when LLM initiates them
        if from_llm and self._is_dangerous(command):
            approval = self.output.get_approval(f"Execute command: {command}")
            if approval == "c" or approval == "n":
                return "[Command cancelled]"
        
        # For interactive commands, use PTY handler if available
        display_command = command
        if needs_interactive:
            self.output.print_line(f"$ {display_command}", self.settings.color_dim)
            
            # Use the interactive command handler if available (PTY overlay in GUI)
            if self._interactive_command_handler:
                self.output.print_line("(running in interactive terminal...)", self.settings.color_dim)
                self._interactive_command_handler(command)
                return "[Running in interactive terminal]"
            
            # Fallback to external terminal
            self.output.print_line("(opening in external terminal...)", self.settings.color_dim)
            terminals = [
                ["konsole", "-e", "bash", "-c", f"{command}; echo ''; echo 'Press Enter to close...'; read"],
                ["gnome-terminal", "--", "bash", "-c", f"{command}; echo ''; echo 'Press Enter to close...'; read"],
                ["xterm", "-e", "bash", "-c", f"{command}; echo ''; echo 'Press Enter to close...'; read"],
            ]
            
            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd, start_new_session=True)
                    return "[Command opened in terminal]"
                except FileNotFoundError:
                    continue
            
            self.output.print_line("[Error: No terminal emulator found]", self.settings.color_status_error)
            return "[Error: No terminal emulator found (tried konsole, gnome-terminal, xterm)]"
        
        # For tool calls, set up collapsible output (don't print $ command)
        if from_llm:
            self.output.set_pending_tool_command(display_command)
        else:
            self.output.print_line(f"$ {display_command}", self.settings.color_dim)
        
        try:
            # Add color support for ls
            if command == "ls" or command.startswith("ls "):
                if "--color" not in command:
                    command = command.replace("ls", "ls --color=always", 1)
            
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
                env={**os.environ, "TERM": "xterm-256color"}
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode()
            
            if track:
                self.last_command = display_command
                self.last_output = output.strip()[:500]
            
            if output:
                self.output.print_command_output(output, from_tool=from_llm)
            
            has_error = proc.returncode != 0
            if has_error and not output:
                return f"[Command failed with exit code {proc.returncode}]"
            
            return output if output else "[Command completed]"
            
        except asyncio.TimeoutError:
            self.output.print_line("[Command timed out]", self.settings.color_status_error)
            return "[Command timed out]"
        except Exception as e:
            self.output.print_line(f"[Error: {e}]", self.settings.color_status_error)
            return f"[Error: {e}]"
    
    def execute_command_sync(self, command: str, track: bool = True) -> str:
        """Synchronous version of execute_command for CLI."""
        if self._is_dangerous(command):
            approval = self.output.get_approval(f"Execute dangerous command: {command}")
            if approval == "c" or approval == "n":
                return "[Command cancelled]"
        
        self.output.print_line(f"$ {command}", self.settings.color_dim)
        
        try:
            if command == "ls" or command.startswith("ls "):
                if "--color" not in command:
                    command = command.replace("ls", "ls --color=always", 1)
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.cwd,
                env={**os.environ, "TERM": "xterm-256color"}
            )
            output = result.stdout + result.stderr
            
            if track:
                self.last_command = command
                self.last_output = output.strip()[:500]
            
            if output:
                self.output.print_command_output(output)
            
            has_error = result.returncode != 0
            if has_error and not output:
                return f"[Command failed with exit code {result.returncode}]"
            
            return output if output else "[Command completed]"
            
        except subprocess.TimeoutExpired:
            self.output.print_line("[Command timed out after 60 seconds]", self.settings.color_status_error)
            return "[Command timed out]"
        except Exception as e:
            self.output.print_line(f"[Error: {e}]", self.settings.color_status_error)
            return f"[Error: {e}]"
    
    def create_file(self, path: str, content: str, executable: bool = False) -> str:
        """Create a file with user approval."""
        approval = self.output.show_file_preview(path, content, executable)
        
        if approval != "y":
            return "[File creation cancelled]"
        
        try:
            expanded = os.path.expanduser(path)
            if not os.path.isabs(expanded):
                expanded = os.path.join(self.cwd, expanded)
            
            os.makedirs(os.path.dirname(expanded) or ".", exist_ok=True)
            
            with open(expanded, "w") as f:
                f.write(content)
            
            if executable:
                os.chmod(expanded, 0o755)
            
            self.output.print_line(f"[created] {expanded}", self.settings.color_status_ready)
            return f"[File created: {expanded}]"
            
        except Exception as e:
            self.output.print_line(f"[Error: {e}]", self.settings.color_status_error)
            return f"[Error: {e}]"
    
    async def web_search(self, query: str) -> str:
        """Search web using Gemini with Google Search grounding."""
        try:
            from google import genai
            
            api_key = os.environ.get("GEMINI_API_KEY") or getattr(self.settings, 'gemini_api_key', '')
            if not api_key:
                return "[Error: No Gemini API key configured. Set GEMINI_API_KEY env var or add to settings.]"
            
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config={"tools": [{"google_search": {}}]}
            )
            
            text = response.text
            sources = []
            
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            for chunk in metadata.grounding_chunks:
                                if hasattr(chunk, 'web') and chunk.web:
                                    sources.append(f"  - {chunk.web.title}: {chunk.web.uri}")
            except Exception:
                pass
            
            if sources:
                return f"{text}\n\nSources:\n" + "\n".join(sources[:5])
            return text
            
        except ImportError:
            self.log_tool_error("web_search", "ImportError", "google-genai not installed")
            return "[Error: google-genai package not installed. Run: pip install google-genai]"
        except Exception as e:
            self.log_tool_error("web_search", type(e).__name__, str(e), query)
            return f"[Web search error: {e}]"
    
    def read_local_file(self, path: str, max_lines: int = 200) -> str:
        """Read contents of a local file."""
        try:
            expanded = os.path.expanduser(path)
            if not os.path.isabs(expanded):
                expanded = os.path.join(self.cwd, expanded)
            expanded = os.path.normpath(expanded)
            
            if not os.path.exists(expanded):
                return f"[Error: File not found: {expanded}]"
            
            if not os.path.isfile(expanded):
                return f"[Error: Not a file: {expanded}]"
            
            size = os.path.getsize(expanded)
            if size > 1024 * 1024:
                return f"[Error: File too large ({size // 1024}KB). Max 1MB.]"
            
            with open(expanded, 'r', errors='replace') as f:
                lines = f.readlines()
            
            total = len(lines)
            if total > max_lines:
                lines = lines[:max_lines]
                content = ''.join(lines)
                return f"{content}\n\n[Showing {max_lines} of {total} lines]"
            
            return ''.join(lines)
            
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    async def read_url(self, url: str) -> str:
        """Fetch and extract text from a URL."""
        try:
            from bs4 import BeautifulSoup
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
                })
                resp.raise_for_status()
                html = resp.text
            
            soup = BeautifulSoup(html, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if len(text) > 8000:
                text = text[:8000] + "\n\n[Content truncated...]"
            
            return text
            
        except ImportError:
            return "[Error: beautifulsoup4 not installed. Run: pip install beautifulsoup4]"
        except httpx.HTTPStatusError as e:
            return f"[Error: HTTP {e.response.status_code}]"
        except Exception as e:
            return f"[Error fetching URL: {e}]"
    
    def clipboard_action(self, action: str, content: str = "") -> str:
        """Read from or write to system clipboard."""
        try:
            if action == "read":
                result = subprocess.run(
                    ['xclip', '-selection', 'clipboard', '-o'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode != 0:
                    result = subprocess.run(
                        ['xsel', '--clipboard', '--output'],
                        capture_output=True, text=True, timeout=5
                    )
                
                if result.returncode == 0:
                    clip = result.stdout
                    if len(clip) > 4000:
                        clip = clip[:4000] + "\n\n[Clipboard truncated...]"
                    return clip if clip else "[Clipboard is empty]"
                return "[Error: Could not read clipboard. Install xclip or xsel.]"
                
            elif action == "write":
                if not content:
                    return "[Error: No content provided to write]"
                
                result = subprocess.run(
                    ['xclip', '-selection', 'clipboard'],
                    input=content, text=True, timeout=5
                )
                if result.returncode != 0:
                    result = subprocess.run(
                        ['xsel', '--clipboard', '--input'],
                        input=content, text=True, timeout=5
                    )
                
                if result.returncode == 0:
                    return f"[Copied {len(content)} characters to clipboard]"
                return "[Error: Could not write to clipboard. Install xclip or xsel.]"
            
            return f"[Error: Unknown action '{action}'. Use 'read' or 'write'.]"
            
        except subprocess.TimeoutExpired:
            return "[Error: Clipboard operation timed out]"
        except Exception as e:
            return f"[Clipboard error: {e}]"
    
    def get_system_info(self, info_type: str = "all") -> str:
        """Get system information."""
        try:
            import psutil
            
            info = []
            
            if info_type in ("all", "cpu"):
                cpu_percent = psutil.cpu_percent(interval=0.5)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()
                freq_str = f"{cpu_freq.current:.0f}MHz" if cpu_freq else "N/A"
                info.append(f"CPU: {cpu_percent}% ({cpu_count} cores @ {freq_str})")
            
            if info_type in ("all", "memory"):
                mem = psutil.virtual_memory()
                used_gb = mem.used / (1024**3)
                total_gb = mem.total / (1024**3)
                info.append(f"Memory: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent}%)")
            
            if info_type in ("all", "disk"):
                disk = psutil.disk_usage('/')
                used_gb = disk.used / (1024**3)
                total_gb = disk.total / (1024**3)
                free_gb = disk.free / (1024**3)
                info.append(f"Disk (/): {used_gb:.0f}GB / {total_gb:.0f}GB ({disk.percent}% used, {free_gb:.0f}GB free)")
            
            if info_type in ("all", "battery"):
                battery = psutil.sensors_battery()
                if battery:
                    status = "Charging" if battery.power_plugged else "Discharging"
                    time_left = ""
                    if battery.secsleft > 0:
                        hours = battery.secsleft // 3600
                        mins = (battery.secsleft % 3600) // 60
                        time_left = f", {hours}h {mins}m remaining"
                    info.append(f"Battery: {battery.percent:.0f}% ({status}{time_left})")
                elif info_type == "battery":
                    info.append("Battery: Not available (desktop)")
            
            if info_type in ("all", "network"):
                net = psutil.net_io_counters()
                sent_mb = net.bytes_sent / (1024**2)
                recv_mb = net.bytes_recv / (1024**2)
                info.append(f"Network: ↑{sent_mb:.0f}MB ↓{recv_mb:.0f}MB")
            
            return "\n".join(info) if info else "[No system info available]"
            
        except ImportError:
            return "[Error: psutil not installed. Run: pip install psutil]"
        except Exception as e:
            return f"[System info error: {e}]"
    
    def _get_os_info(self) -> str:
        """Get OS/distro information."""
        try:
            info = {}
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release") as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            info[key] = value.strip('"')
            
            distro = info.get("PRETTY_NAME", info.get("NAME", "Linux"))
            return distro
        except:
            return "Linux"
    
    def _get_directory_context(self) -> str:
        """Get current directory context for LLM."""
        try:
            # OS info
            os_info = self._get_os_info()
            context = f"OS: {os_info}\n"
            
            entries = sorted(os.listdir(self.cwd))[:30]
            items = []
            for name in entries:
                path = os.path.join(self.cwd, name)
                try:
                    if os.path.isdir(path):
                        items.append(f"{name}/")
                    else:
                        size = os.path.getsize(path)
                        if size < 1024:
                            items.append(f"{name} ({size}B)")
                        elif size < 1024 * 1024:
                            items.append(f"{name} ({size // 1024}KB)")
                        else:
                            items.append(f"{name} ({size // (1024 * 1024)}MB)")
                except:
                    items.append(name)
            
            context += f"Current directory: {self.cwd}\nContents: {', '.join(items)}"
            
            if len(os.listdir(self.cwd)) > 30:
                context += f"\n... and {len(os.listdir(self.cwd)) - 30} more files"
            
            if self.last_command:
                context += f"\n\nLast command: {self.last_command}\nOutput: {self.last_output}"
            
            return context
            
        except Exception as e:
            return f"Current directory: {self.cwd}"
    
    async def _stream_response(self, messages: list, include_tools: bool = True) -> tuple[str, list]:
        """Stream LLM response, returning (content, tool_calls).
        
        For tool calls, we need non-streaming to get the full tool call JSON.
        For regular responses, we stream tokens to the UI.
        """
        request_body = {
            "model": "qwen3-32b",
            "messages": messages,
            "max_tokens": 2000,
            "stream": True
        }
        if include_tools:
            request_body["tools"] = TOOLS
        
        content = ""
        tool_calls = []
        started_output = False
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.llm_url}/v1/chat/completions",
                json=request_body
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    
                    # Check for tool calls
                    if delta.get("tool_calls"):
                        # Tool call detected - we need to collect it
                        for tc in delta["tool_calls"]:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                            
                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    
                    # Stream content tokens
                    token = delta.get("content", "")
                    if token:
                        # Hide spinner and start output on first token
                        if not started_output:
                            self.output.hide_spinner()
                            self.output.start_assistant_response(self.settings)
                            started_output = True
                        
                        self.output.print_assistant_token(token, self.settings)
                        content += token
        
        except httpx.ConnectError:
            if started_output:
                self.output.end_assistant_response(self.settings)
            raise
        
        if started_output:
            self.output.end_assistant_response(self.settings)
        
        return content, tool_calls
    
    async def _non_streaming_request(self, messages: list, include_tools: bool = True) -> dict:
        """Make a non-streaming request, returning the full response."""
        request_body = {
            "model": "qwen3-32b",
            "messages": messages,
            "max_tokens": 2000,
            "stream": False
        }
        if include_tools:
            request_body["tools"] = TOOLS
        
        resp = await self.client.post(
            f"{self.llm_url}/v1/chat/completions",
            json=request_body
        )
        return resp.json()
    
    def _build_system_prompt(self, query: str) -> str:
        """Build system prompt with memory context and custom additions."""
        prompt = SYSTEM_PROMPT_BASE
        
        # Add custom prompt additions (from self-improvement)
        custom_prompt = self.introspection.self_improve.get_custom_prompt()
        if custom_prompt:
            prompt += f"\n\n[LEARNED BEHAVIORS]\n{custom_prompt}"
        
        # Add custom tool examples
        tool_examples = self.introspection.self_improve.get_tool_examples()
        if tool_examples:
            prompt += f"\n\n[ADDITIONAL TOOL EXAMPLES]\n{tool_examples}"
        
        # Add memory context
        memory_context = self.memory.get_relevant_context(query)
        if memory_context:
            prompt += f"\n\n{memory_context}"
        
        return prompt
    
    async def query_llm(self, query: str) -> Optional[str]:
        """Send query to LLM and handle response."""
        # Ensure server is running (only for local servers)
        if self.server:
            self.server.cancel_scheduled_stop()
            if not self._check_server_health():
                if not self.server.start(callback=lambda msg: self.output.print_status(msg, self.settings.color_status_busy)):
                    self.output.print_line("Failed to start LLM server.", self.settings.color_status_error)
                    return None
        else:
            # Remote server - just check if it's reachable
            if not self._check_server_health():
                self.output.print_line(f"[Error: Cannot reach LLM server at {self.llm_url}]", self.settings.color_status_error)
                return None
        
        # Build context with memory
        context = self._get_directory_context()
        contextual_input = f"[Context: {context}]\n\nUser: {query}"
        
        self.conversation.append({"role": "user", "content": query})
        
        system_prompt = self._build_system_prompt(query)
        messages = [{"role": "system", "content": system_prompt}] + self.conversation[-10:]
        messages[-1] = {"role": "user", "content": contextual_input}
        
        self.output.show_spinner("thinking")
        
        # First request: use non-streaming to properly capture tool calls
        try:
            result = await self._non_streaming_request(messages, include_tools=True)
        except httpx.ConnectError:
            self.output.hide_spinner()
            self.output.print_line("[Error: Cannot connect to LLM server]", self.settings.color_status_error)
            return None
        except Exception as e:
            self.output.hide_spinner()
            self.output.print_line(f"[Error: {e}]", self.settings.color_status_error)
            return None
        
        self.output.hide_spinner()
        
        # Process response
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        # Fallback: parse inline JSON tool calls if no proper tool_calls
        if not tool_calls and content:
            remaining_content, inline_calls = parse_inline_tool_calls(content)
            if inline_calls:
                tool_calls = inline_calls
                content = remaining_content
        
        # Handle tool calls
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                tool_call_id = tc.get("id", "call_1")
                
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except:
                    continue
                
                if name == "execute_command":
                    output = await self.execute_command(args.get("command", ""), from_llm=True)
                elif name == "create_file":
                    output = self.create_file(
                        args.get("path", ""),
                        args.get("content", ""),
                        args.get("executable", False)
                    )
                elif name == "web_search":
                    query = args.get("query", "")
                    self.output.set_pending_tool_command(f"web_search: {query}")
                    output = await self.web_search(query)
                    self.output.print_command_output(output, from_tool=True)
                elif name == "read_file":
                    path = args.get("path", "")
                    self.output.set_pending_tool_command(f"read_file: {path}")
                    output = self.read_local_file(path, args.get("max_lines", 200))
                    self.output.print_command_output(output, from_tool=True)
                elif name == "read_url":
                    url = args.get("url", "")
                    self.output.set_pending_tool_command(f"read_url: {url}")
                    output = await self.read_url(url)
                    self.output.print_command_output(output, from_tool=True)
                elif name == "clipboard":
                    action = args.get("action", "read")
                    self.output.set_pending_tool_command(f"clipboard: {action}")
                    output = self.clipboard_action(action, args.get("content", ""))
                    self.output.print_command_output(output, from_tool=True)
                elif name == "system_info":
                    info_type = args.get("info_type", "all")
                    self.output.set_pending_tool_command(f"system_info: {info_type}")
                    output = self.get_system_info(info_type)
                    self.output.print_command_output(output, from_tool=True)
                else:
                    output = f"[Unknown tool: {name}]"
                
                # Add to messages and get follow-up
                messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": func.get("arguments", "{}")}
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": output
                })
                
                # Get follow-up response (streamed for better UX)
                self.output.show_spinner("thinking")
                try:
                    content, _ = await self._stream_response(messages, include_tools=False)
                except Exception as e:
                    self.output.hide_spinner()
                    self.output.print_line(f"[Error: {e}]", self.settings.color_status_error)
                    return None
                
                self.output.hide_spinner()
        else:
            # No tool calls - display the content (wasn't streamed)
            if content:
                self.output.print_assistant(content, self.settings)
        
        # Log response and store in memory
        if content:
            self.conversation.append({"role": "assistant", "content": content})
            self._log_conversation(messages, content)
            
            # Store last exchange for feedback
            self._last_exchange = {"query": query, "response": content}
            
            # Store in memory (fire-and-forget in background thread)
            self._store_memory_background(query, content)
            

        
        return content
    
    def _store_memory_background(self, user_message: str, assistant_response: str):
        """Fire-and-forget memory storage in background thread."""
        import concurrent.futures
        
        def do_store():
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                client = httpx.AsyncClient(timeout=30.0)
                loop.run_until_complete(
                    self.memory.classify_and_store(
                        user_message,
                        assistant_response,
                        client,
                        on_status=lambda msg: None
                    )
                )
                loop.run_until_complete(client.aclose())
                loop.close()
            except Exception:
                pass
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(do_store)
        executor.shutdown(wait=False)
    
    async def close(self):
        """Cleanup resources."""
        self.save_history()
        self.introspection.stop()
        await self.client.aclose()
    
    def start_introspection(self, interval_seconds: int = 300):
        """Start the introspection background loop."""
        if not self._introspection_started:
            self.introspection.start_background(interval_seconds)
            self._introspection_started = True
    
    async def run_introspection_cycle(self) -> dict:
        """Manually trigger an introspection cycle."""
        return await self.introspection.run_cycle()
    
    def log_tool_error(self, tool_name: str, error_type: str, message: str, context: str = ""):
        """Log a tool error for introspection analysis."""
        self.introspection.log_tool_error(tool_name, error_type, message, context)
    
    def get_introspection_stats(self) -> dict:
        """Get introspection statistics."""
        return self.introspection.get_stats()
    
    async def _web_search_for_research(self, query: str) -> str:
        """Web search wrapper for research system to use."""
        return await self.web_search(query)
    
    def get_startup_status(self) -> Optional[str]:
        """
        Get a startup status message showing activity while away.
        Called once when terminal starts, not during conversation.
        """
        messages = []
        
        # Check for research insights
        if self.introspection.research.is_enabled():
            stats = self.introspection.research.get_stats()
            unshared = stats.get('unshared_insights', 0)
            if unshared > 0:
                messages.append(f"{unshared} new insights - ?findings")
        
        # Check for improvement suggestions
        if self.introspection.self_improve.is_enabled():
            pending = len(self.introspection.self_improve.get_pending_suggestions())
            if pending > 0:
                messages.append(f"{pending} suggestion{'s' if pending > 1 else ''} - ?suggestions")
        
        if messages:
            return f"({', '.join(messages)})"
        
        return None

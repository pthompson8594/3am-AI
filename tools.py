#!/usr/bin/env python3
"""
Tools - LLM tool implementations for MK8.

All tools the LLM can call:
- execute_command: Run shell commands
- create_file: Create/write files
- web_search: Search the web via Google
- read_file: Read local files
- read_url: Fetch webpage content
- clipboard: Read/write clipboard
- system_info: Get system stats
"""

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from data_security import DataEncryptor
from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup


# Tool definitions for the LLM API
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

# Valid tool names for inline parsing
VALID_TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


class CustomToolRegistry:
    """Runtime registry for LLM-generated custom tools. One instance per user."""

    # Patterns that must not appear in generated tool code
    UNSAFE_PATTERNS = [
        "import subprocess", "import socket", "import threading",
        "import multiprocessing", "import ctypes", "import pty", "import tty",
        "import signal", "from subprocess", "from socket", "from threading",
        "from multiprocessing", "from ctypes", "__import__(", "eval(", "exec(",
        "compile(", "import os\n", "import os ", "import sys", "import importlib",
        "importlib.", "builtins.", "__builtins__", "globals(", "locals(",
    ]

    def __init__(self, data_dir: Path, encryptor: Optional["DataEncryptor"] = None):
        self.data_dir = data_dir
        self._encryptor = encryptor
        # { name: {schema, fn, code, prompt_addition, installed_at} }
        self._tools: dict[str, dict] = {}
        self._load()

    def _tools_file(self) -> Path:
        return self.data_dir / "installed_tools.json"

    def _load(self):
        """Load and exec persisted tools on startup."""
        path = self._tools_file()
        if not path.exists():
            return
        try:
            if self._encryptor and self._encryptor.config.enabled:
                data = self._encryptor.decrypt_file(path)
            else:
                with open(path) as f:
                    data = json.load(f)
            for entry in data.get("tools", []):
                try:
                    self._exec_and_register(
                        name=entry["name"],
                        schema=entry["schema"],
                        code=entry["code"],
                        prompt_addition=entry.get("prompt_addition", ""),
                        installed_at=entry.get("installed_at"),
                        persist=False,
                    )
                except Exception as e:
                    print(f"[CustomToolRegistry] Failed to load '{entry.get('name')}': {e}")
        except Exception as e:
            print(f"[CustomToolRegistry] Load error: {e}")

    def _save(self):
        """Persist all registered tools to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tools_list = [
            {
                "name": name,
                "schema": entry["schema"],
                "code": entry["code"],
                "prompt_addition": entry["prompt_addition"],
                "installed_at": entry.get("installed_at", time.time()),
            }
            for name, entry in self._tools.items()
        ]
        data = {"tools": tools_list, "last_update": time.time()}
        if self._encryptor and self._encryptor.config.enabled:
            self._encryptor.encrypt_file(self._tools_file(), data)
        else:
            with open(self._tools_file(), "w") as f:
                json.dump(data, f, indent=2)

    def check_code_safety(self, code: str) -> list[str]:
        """Scan code for dangerous patterns. Returns list of violations."""
        return [
            f"forbidden: '{p.strip()}'"
            for p in self.UNSAFE_PATTERNS
            if p in code
        ]

    def _exec_and_register(
        self, name: str, schema: dict, code: str, prompt_addition: str,
        installed_at: float = None, persist: bool = True
    ):
        """Exec tool code in an isolated namespace and register it."""
        namespace: dict = {}
        exec(compile(code, f"<tool:{name}>", "exec"), namespace)  # noqa: S102
        fn = namespace.get(f"tool_{name}")
        if fn is None:
            raise ValueError(f"Code must define 'async def tool_{name}(**kwargs)'")
        if not asyncio.iscoroutinefunction(fn):
            raise ValueError(f"'tool_{name}' must be async")
        self._tools[name] = {
            "schema": schema,
            "fn": fn,
            "code": code,
            "prompt_addition": prompt_addition,
            "installed_at": installed_at or time.time(),
        }
        if persist:
            self._save()

    def register(self, name: str, schema: dict, code: str, prompt_addition: str):
        """Public registration (caller must run check_code_safety first)."""
        self._exec_and_register(name, schema, code, prompt_addition, persist=True)

    async def execute(self, name: str, args: dict) -> str:
        """Run a registered custom tool. Returns result or error string."""
        entry = self._tools.get(name)
        if not entry:
            return f"[Error: Unknown custom tool '{name}']"
        try:
            result = await entry["fn"](**args)
            return str(result) if result is not None else "[No output]"
        except Exception as e:
            return f"[Error in custom tool '{name}': {e}]"

    def get_schemas(self) -> list[dict]:
        """OpenAI-format schemas for all registered custom tools."""
        return [entry["schema"] for entry in self._tools.values()]

    def get_active_tools(self) -> list[dict]:
        """TOOLS + custom schemas — pass this to LLM API calls."""
        return TOOLS + self.get_schemas()

    def get_prompt_additions(self) -> str:
        """Joined prompt bullet lines for all installed tools."""
        return "\n".join(
            entry["prompt_addition"]
            for entry in self._tools.values()
            if entry.get("prompt_addition")
        )

    def unregister(self, name: str):
        """Remove a tool from the live registry and from disk."""
        if name in self._tools:
            del self._tools[name]
            self._save()

    def list_names(self) -> list[str]:
        return list(self._tools.keys())


# Dangerous command patterns that need approval
DANGEROUS_PATTERNS = [
    "sudo", "rm -rf", "mkfs", "dd if=", ":(){", "chmod -R 777", "> /dev/sd",
    "tee ", "> ", ">> ", "cat >", "echo >", "echo >>",
    "cp ", "mv ", "mkdir ", "touch ",
    "cat <<", "cat>", "printf >",
]


class PendingApproval:
    """Represents a pending file operation awaiting user approval."""
    
    def __init__(self, action: str, path: str, content: str = "", executable: bool = False):
        self.id = str(id(self))
        self.action = action  # "create_file", "delete_file", etc.
        self.path = path
        self.content = content
        self.executable = executable
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "path": self.path,
            "content": self.content,
            "executable": self.executable,
        }


def parse_inline_tool_calls(content: str) -> tuple[str, list]:
    """
    Parse inline JSON tool calls from LLM response content.
    Returns (remaining_content, tool_calls_list).
    """
    if not content:
        return content, []
    
    tool_calls = []
    remaining = content
    
    pattern = r'\{["\s]*name["\s]*:["\s]*([^"]+)["\s]*,["\s]*arguments["\s]*:\s*(\{[^}]+\})\s*\}'
    
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    for i, match in enumerate(matches):
        name = match.group(1).strip().strip('"')
        args_str = match.group(2)
        
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
            remaining = remaining.replace(match.group(0), "").strip()
        except json.JSONDecodeError:
            continue
    
    return remaining, tool_calls


def is_dangerous_command(command: str) -> bool:
    """Check if a command is potentially dangerous."""
    cmd_lower = command.lower()
    return any(pattern.lower() in cmd_lower for pattern in DANGEROUS_PATTERNS)


class ToolExecutor:
    """Executes tools for the LLM."""

    def __init__(
        self,
        working_dir: str = None,
        on_approval_needed: Optional[Callable] = None,
        on_tool_output: Optional[Callable] = None,
        search_provider: str = "google",
        data_dir: Optional[Path] = None,
        encryptor: Optional["DataEncryptor"] = None,
    ):
        self.working_dir = working_dir or os.getcwd()
        self.on_approval_needed = on_approval_needed
        self.on_tool_output = on_tool_output
        self.search_provider = search_provider  # "google" or "duckduckgo"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.last_command = ""
        self.last_output = ""
        self.pending_approval: Optional[PendingApproval] = None
        self.custom_registry = CustomToolRegistry(
            data_dir or Path.home() / ".config" / "3am",
            encryptor=encryptor,
        )

    def get_active_tools(self) -> list[dict]:
        """Return TOOLS + installed custom tool schemas for LLM API calls."""
        return self.custom_registry.get_active_tools()

    async def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        try:
            if tool_name == "execute_command":
                return await self.execute_command(arguments.get("command", ""))
            elif tool_name == "create_file":
                return await self.create_file(
                    arguments.get("path", ""),
                    arguments.get("content", ""),
                    arguments.get("executable", False)
                )
            elif tool_name == "web_search":
                return await self.web_search(arguments.get("query", ""))
            elif tool_name == "read_file":
                return self.read_file(
                    arguments.get("path", ""),
                    arguments.get("max_lines", 200)
                )
            elif tool_name == "read_url":
                return await self.read_url(arguments.get("url", ""))
            elif tool_name == "clipboard":
                return self.clipboard_action(
                    arguments.get("action", "read"),
                    arguments.get("content", "")
                )
            elif tool_name == "system_info":
                return self.get_system_info(arguments.get("info_type", "all"))
            else:
                if tool_name in self.custom_registry.list_names():
                    return await self.custom_registry.execute(tool_name, arguments)
                return f"[Error: Unknown tool '{tool_name}']"
        except Exception as e:
            return f"[Error executing {tool_name}: {e}]"
    
    async def execute_command(self, command: str) -> str:
        """Execute a shell command."""
        if not command:
            return "[Error: No command provided]"
        
        # Check if dangerous - requires user approval
        if is_dangerous_command(command):
            self.pending_approval = PendingApproval(
                action="execute_command",
                path=command,  # Store command in path field for display
                content=f"This command may modify files or system:\n\n$ {command}",
                executable=False
            )
            return f"[PENDING_APPROVAL:{self.pending_approval.id}]"
        
        self.last_command = command
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_dir
            )
            
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            
            # Truncate long output
            if len(output) > 8000:
                output = output[:8000] + "\n\n[Output truncated...]"
            
            self.last_output = output
            
            if self.on_tool_output:
                self.on_tool_output(f"$ {command}\n{output}")
            
            return output if output else "[Command completed with no output]"
            
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out after 30 seconds]"
        except Exception as e:
            return f"[Error: {e}]"
    
    async def create_file(self, path: str, content: str, executable: bool = False) -> str:
        """Queue a file creation for user approval. Returns pending status."""
        if not path:
            return "[Error: No path provided]"
        
        # Expand path
        if path.startswith("~"):
            path = os.path.expanduser(path)
        elif not path.startswith("/"):
            path = os.path.join(self.working_dir, path)
        
        # Create pending approval instead of writing immediately
        self.pending_approval = PendingApproval(
            action="create_file",
            path=path,
            content=content,
            executable=executable
        )
        
        return f"[PENDING_APPROVAL:{self.pending_approval.id}]"
    
    def execute_pending_approval(self) -> str:
        """Execute the pending operation after user approval."""
        if not self.pending_approval:
            return "[Error: No pending approval]"
        
        approval = self.pending_approval
        self.pending_approval = None
        
        try:
            if approval.action == "create_file":
                # Create parent directories if needed
                os.makedirs(os.path.dirname(approval.path) or ".", exist_ok=True)
                
                with open(approval.path, "w") as f:
                    f.write(approval.content)
                
                if approval.executable:
                    os.chmod(approval.path, 0o755)
                
                result = f"[Created file: {approval.path}]"
                if approval.executable:
                    result += " (executable)"
                
                if self.on_tool_output:
                    self.on_tool_output(result)
                
                return result
            
            elif approval.action == "execute_command":
                # Execute the approved command
                command = approval.path  # Command was stored in path field
                self.last_command = command
                
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.working_dir
                )
                
                output = result.stdout
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr}"
                if result.returncode != 0:
                    output += f"\n[exit code: {result.returncode}]"
                
                if len(output) > 8000:
                    output = output[:8000] + "\n\n[Output truncated...]"
                
                self.last_output = output
                
                if self.on_tool_output:
                    self.on_tool_output(f"$ {command}\n{output}")
                
                return output if output else "[Command completed with no output]"
            
            else:
                return f"[Error: Unknown action '{approval.action}']"
                
        except subprocess.TimeoutExpired:
            return "[Error: Command timed out after 30 seconds]"
        except Exception as e:
            return f"[Error: {e}]"
    
    def cancel_pending_approval(self) -> str:
        """Cancel the pending file operation."""
        if not self.pending_approval:
            return "[Error: No pending approval]"
        
        path = self.pending_approval.path
        self.pending_approval = None
        return f"[File creation cancelled: {path}]"
    
    async def web_search(self, query: str) -> str:
        """Search the web using configured provider (Google or DuckDuckGo)."""
        if not query:
            return "[Error: No search query provided]"
        
        if self.search_provider == "duckduckgo":
            return await self._web_search_duckduckgo(query)
        else:
            return await self._web_search_google(query)
    
    async def _web_search_duckduckgo(self, query: str) -> str:
        """Search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    title = r.get('title', 'No title')
                    body = r.get('body', '')
                    href = r.get('href', '')
                    results.append(f"**{title}**\n{body}\nURL: {href}")
            
            if self.on_tool_output:
                self.on_tool_output(f"🔍 {query} (DuckDuckGo)")
            
            if results:
                return "\n\n---\n\n".join(results)
            return "[No results found]"
            
        except ImportError:
            return "[Error: duckduckgo-search package not installed. Run: pip install duckduckgo-search]"
        except Exception as e:
            return f"[DuckDuckGo search error: {e}]"
    
    async def _web_search_google(self, query: str) -> str:
        """Search using Gemini with Google Search grounding."""
        try:
            from google import genai
            import json
            from pathlib import Path
            
            # Load API key from config
            config_file = Path.home() / '.config/3am/config.json'
            api_key = os.environ.get("GEMINI_API_KEY", "")
            
            if config_file.exists() and not api_key:
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    api_key = config.get('gemini_api_key', '')
                except Exception:
                    pass
            
            if not api_key:
                return "[Error: No Gemini API key configured. Set GEMINI_API_KEY env var or add to config.]"
            
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=query,
                config={"tools": [{"google_search": {}}]}
            )
            
            text = response.text
            sources = []
            
            # Extract sources if available
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
            
            if self.on_tool_output:
                self.on_tool_output(f"🔍 {query} (Google)")
            
            if sources:
                return f"{text}\n\nSources:\n" + "\n".join(sources[:5])
            return text
            
        except ImportError:
            return "[Error: google-genai package not installed. Run: pip install google-genai]"
        except Exception as e:
            return f"[Web search error: {e}]"
    
    def read_file(self, path: str, max_lines: int = 200) -> str:
        """Read a local file."""
        if not path:
            return "[Error: No path provided]"
        
        # Expand path
        if path.startswith("~"):
            path = os.path.expanduser(path)
        elif not path.startswith("/"):
            path = os.path.join(self.working_dir, path)
        
        try:
            if not os.path.exists(path):
                return f"[Error: File not found: {path}]"
            
            if os.path.isdir(path):
                entries = sorted(os.listdir(path))[:50]
                return f"Directory: {path}\n" + "\n".join(entries)
            
            with open(path, "r") as f:
                lines = f.readlines()
            
            if len(lines) > max_lines:
                content = "".join(lines[:max_lines])
                content += f"\n\n[Showing first {max_lines} of {len(lines)} lines]"
            else:
                content = "".join(lines)
            
            if self.on_tool_output:
                self.on_tool_output(f"📄 {path}")
            
            return content
            
        except Exception as e:
            return f"[Error reading file: {e}]"
    
    async def read_url(self, url: str) -> str:
        """Fetch and extract text from a URL."""
        if not url:
            return "[Error: No URL provided]"
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            }
            
            response = await self.client.get(url, headers=headers, follow_redirects=True, timeout=15)
            
            if response.status_code != 200:
                return f"[Error: HTTP {response.status_code}]"
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            text = "\n".join(lines)
            
            if len(text) > 6000:
                text = text[:6000] + "\n\n[Content truncated...]"
            
            if self.on_tool_output:
                self.on_tool_output(f"🌐 {url}")
            
            return text
            
        except httpx.TimeoutException:
            return "[Error: Request timed out]"
        except Exception as e:
            return f"[Error fetching URL: {e}]"
    
    def clipboard_action(self, action: str, content: str = "") -> str:
        """Read from or write to the system clipboard."""
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
    
    async def close(self):
        """Cleanup resources."""
        await self.client.aclose()

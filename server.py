#!/usr/bin/env python3
"""
Web Server - FastAPI backend for the LLM assistant.

Provides:
- REST API for chat, history, memory, etc.
- Server-Sent Events (SSE) for streaming responses
- Session-based authentication
- Tool calling and special commands
- Per-user memory, research, and self-improvement
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Request, Response, Cookie, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from auth import AuthSystem, AuthError, User
from scheduler import IntrospectionScheduler, SchedulerConfig, CheckResult
from data_security import SecureUserData, DataEncryptor
from tools import ToolExecutor, parse_inline_tool_calls, PendingApproval
from commands import CommandHandler, CommandResult
from memory import MemorySystem
from introspection import IntrospectionLoop
from experience_log import ExperienceLog
from decision_gate import DecisionGate
from behavior_profile import BehaviorProfile


# Configuration
DATA_DIR = Path.home() / ".local/share/3am"
STATIC_DIR = Path(__file__).parent / "static"
LLM_URL = os.environ.get("LLM_URL", "http://localhost:8080")
LLM_MODEL = os.environ.get("LLM_MODEL", None)
CONFIG_FILE = Path.home() / ".config/3am/config.json"


def _load_server_config() -> dict:
    """Load server config from ~/.config/3am/config.json."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
    except Exception as e:
        print(f"[Config] Error loading config: {e}")
    return {}


# Auto-detect model from llama.cpp server if not specified
def get_llm_model() -> str:
    """Auto-detect model from llama.cpp server."""
    if LLM_MODEL:
        return LLM_MODEL
    
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{LLM_URL}/v1/models")
            models = response.json().get("models", [])
            if models:
                # Use the first model's name (or full path if available)
                model_name = models[0].get("model") or models[0].get("name")
                print(f"[Config] Auto-detected model: {model_name}")
                return model_name
    except Exception as e:
        print(f"[Config] Failed to auto-detect model: {e}")
    
    # Fallback
    return "Qwen3-14B-Q4_K_M.gguf"

# Get model name (will auto-detect if not set)
_DETECTED_MODEL = get_llm_model()

# Global instances
auth = AuthSystem()
scheduler = IntrospectionScheduler()

# Per-user LLM cores (lazy loaded)
user_cores: dict[str, "UserLLMCore"] = {}

# Per-user rate limiting for /api/chat
_user_last_message: dict[str, float] = {}
_user_active_requests: dict[str, int] = {}
CHAT_RATE_LIMIT_SECONDS = 1    # min seconds between messages per user
CHAT_MAX_CONCURRENT = 2        # max simultaneous requests per user

# Per-user active WebSocket connections (for server-push and cancel)
_user_ws_connections: dict[str, WebSocket] = {}


async def _push_to_ws(user_id: str, payload: dict):
    """Push a JSON payload to the user's active WebSocket if connected."""
    ws = _user_ws_connections.get(user_id)
    if ws:
        try:
            await ws.send_json(payload)
        except Exception:
            pass


# System prompt
SYSTEM_PROMPT_BASE = """You are a helpful assistant with access to tools.

YOUR TOOLS:
- web_search: Search the web for information
- execute_command: Run shell commands
- create_file: Create or write files
- read_file: Read local files
- read_url: Fetch webpage content
- clipboard: Read/write clipboard
- system_info: Get system stats

HOW TO CALL TOOLS:
Output JSON in this format:
{"name": "tool_name", "arguments": {"arg1": "value1"}}

Examples:
- {"name": "web_search", "arguments": {"query": "weather today"}}
- {"name": "execute_command", "arguments": {"command": "ls -la"}}

RULES:
1. Use web_search when you're not certain about current information.
2. Be concise in responses.
3. Trust tool results over your assumptions.
/no_think"""


# --- Pydantic Models ---

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ConversationRename(BaseModel):
    title: str

class SettingsUpdate(BaseModel):
    research_enabled: Optional[bool] = None
    self_improve_enabled: Optional[bool] = None
    search_provider: Optional[str] = None  # "google" or "duckduckgo"
    decision_gate_enabled: Optional[bool] = None
    decision_gate_sensitivity: Optional[float] = None  # 0.0–1.0
    show_confidence: Optional[bool] = None
    show_feedback_buttons: Optional[bool] = None

class FeedbackRequest(BaseModel):
    message_id: str
    value: str   # "positive" | "negative"
    tags: list[str] = []

class BehaviorProfileUpdate(BaseModel):
    tool_usage_bias: Optional[str] = None
    verbosity: Optional[str] = None
    uncertainty_behavior: Optional[str] = None
    search_threshold: Optional[float] = None
    ask_threshold: Optional[float] = None

class PasswordChange(BaseModel):
    old_password: str
    new_password: str

class ApprovalResponse(BaseModel):
    approved: bool


# --- User LLM Core (per-user instance) ---

class UserLLMCore:
    """Per-user LLM core with isolated memory, tools, and settings."""
    
    def __init__(self, user: User):
        self.user = user
        self.user_data = SecureUserData(user.id)
        self.llm_url = LLM_URL
        self.model = _DETECTED_MODEL
        self.client = httpx.AsyncClient(timeout=120.0)
        self.conversations: dict[str, list] = {}
        
        # Ensure user directory exists
        self.user_data.base_path.mkdir(parents=True, exist_ok=True)

        # Build encryptor from the in-memory key (None if user not yet logged in via auth)
        _enc_key = auth.get_user_key(user.id)
        self._encryptor = DataEncryptor(_enc_key)
        
        # Set per-user data directories for modules that use global paths
        import introspection as intro_mod
        import research as research_mod
        import self_improve as si_mod
        
        intro_mod.DATA_DIR = self.user_data.base_path
        intro_mod.ERROR_LOG_FILE = self.user_data.base_path / "error_journal.json"
        intro_mod.INTROSPECTION_LOG_FILE = self.user_data.base_path / "introspection_log.json"
        
        research_mod.DATA_DIR = self.user_data.base_path
        research_mod.RESEARCH_FILE = self.user_data.base_path / "research.json"
        research_mod.RESEARCH_CONFIG_FILE = self.user_data.base_path / "research_config.json"
        
        si_mod.DATA_DIR = self.user_data.base_path
        si_mod.SUGGESTIONS_FILE = self.user_data.base_path / "suggestions.json"
        si_mod.SELF_IMPROVE_CONFIG_FILE = self.user_data.base_path / "self_improve_config.json"
        si_mod.CUSTOM_PROMPT_FILE = self.user_data.base_path / "custom_prompt.txt"
        si_mod.TOOL_EXAMPLES_FILE = self.user_data.base_path / "tool_examples.txt"
        
        # Per-user memory system
        self.memory = MemorySystem(
            llm_url=LLM_URL,
            data_dir=self.user_data.base_path,
            model=_DETECTED_MODEL,
            encryptor=self._encryptor,
        )
        
        # Per-user introspection
        self.introspection = IntrospectionLoop(
            memory=self.memory,
            llm_url=LLM_URL,
            llm_model_id=_DETECTED_MODEL,
            on_status=self._make_status_callback(),
            web_search_fn=self._web_search_for_research,
            encryptor=self._encryptor,
        )
        
        # Load user settings
        self.search_provider = self._load_search_provider()
        mk13_settings = self._load_mk13_settings()
        self.decision_gate_enabled: bool = mk13_settings.get("decision_gate_enabled", True)
        self.decision_gate_sensitivity: float = mk13_settings.get("decision_gate_sensitivity", 0.5)
        self.show_confidence: bool = mk13_settings.get("show_confidence", True)
        self.show_feedback_buttons: bool = mk13_settings.get("show_feedback_buttons", True)

        # MK13: Experience log (reads/writes to memory.db experience_log table)
        self.experience_log = ExperienceLog(self.user_data.base_path / "memory.db")

        # MK13: Decision gate
        self.decision_gate = DecisionGate(llm_url=LLM_URL, model=_DETECTED_MODEL)

        # MK13: Behavior profile (learned behavioral prefs injected into system prompt)
        self.behavior_profile = BehaviorProfile(self.user_data.base_path, encryptor=self._encryptor)

        # MK13: Wire experience_log and behavior_profile into introspection loop
        self.introspection.experience_log = self.experience_log
        self.introspection.behavior_profile = self.behavior_profile

        # Gate 4: wire experience_log into research for low-confidence signal
        self.introspection.research.set_experience_log(self.experience_log)

        # Tool executor (data_dir enables per-user custom tool registry)
        self.tools = ToolExecutor(
            working_dir=str(Path.home()),
            search_provider=self.search_provider,
            data_dir=self.user_data.base_path,
            encryptor=self._encryptor,
        )

        # Command handler
        self.commands = CommandHandler(
            memory=self.memory,
            introspection=self.introspection,
            on_feedback=self._save_feedback,
            tools=self.tools,
            client=self.client,
        )
        
        self._load_conversations()
    
    def _load_search_provider(self) -> str:
        """Load search provider from user settings."""
        try:
            settings_file = self.user_data.base_path / "settings.json"
            if settings_file.exists():
                with open(settings_file) as f:
                    settings = json.load(f)
                return settings.get("search_provider", "google")
        except Exception:
            pass
        return "google"
    
    def set_search_provider(self, provider: str):
        """Set and persist search provider setting."""
        if provider not in ("google", "duckduckgo"):
            return
        self.search_provider = provider
        self.tools.search_provider = provider
        
        # Persist to settings file
        try:
            settings_file = self.user_data.base_path / "settings.json"
            settings = {}
            if settings_file.exists():
                with open(settings_file) as f:
                    settings = json.load(f)
            settings["search_provider"] = provider
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass
    
    def _load_mk13_settings(self) -> dict:
        """Load MK13-specific settings from user settings.json."""
        try:
            settings_file = self.user_data.base_path / "settings.json"
            if settings_file.exists():
                with open(settings_file) as f:
                    s = json.load(f)
                return {
                    "decision_gate_enabled": s.get("decision_gate_enabled", True),
                    "decision_gate_sensitivity": float(s.get("decision_gate_sensitivity", 0.5)),
                    "show_confidence": s.get("show_confidence", True),
                    "show_feedback_buttons": s.get("show_feedback_buttons", True),
                }
        except Exception:
            pass
        return {}

    def _save_mk13_settings(self):
        """Persist MK13-specific settings to user settings.json."""
        try:
            settings_file = self.user_data.base_path / "settings.json"
            settings = {}
            if settings_file.exists():
                with open(settings_file) as f:
                    settings = json.load(f)
            settings["decision_gate_enabled"] = self.decision_gate_enabled
            settings["decision_gate_sensitivity"] = self.decision_gate_sensitivity
            settings["show_confidence"] = self.show_confidence
            settings["show_feedback_buttons"] = self.show_feedback_buttons
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _make_status_callback(self):
        """Build an on_status callback that logs and pushes to the user's WebSocket."""
        user_id = self.user.id
        username = self.user.username

        def on_status(msg: str):
            print(f"[{username}] {msg}")
            ws = _user_ws_connections.get(user_id)
            if ws:
                import asyncio as _asyncio
                try:
                    loop = _asyncio.get_running_loop()
                    loop.create_task(
                        ws.send_json({"type": "server_push", "content": msg, "push_type": "status"})
                    )
                except RuntimeError:
                    pass  # No running loop (called from sync context)

        return on_status

    async def _web_search_for_research(self, query: str) -> str:
        """Web search wrapper for research system."""
        return await self.tools.web_search(query)
    
    def _save_feedback(self, query: str, rating: int, comment: str):
        """Save user feedback."""
        try:
            feedback_file = self.user_data.base_path / "feedback.jsonl"
            with open(feedback_file, "a") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "query": query[:200],
                    "rating": rating,
                    "comment": comment
                }) + "\n")
        except Exception:
            pass
    
    def _load_conversations(self):
        """Load user's conversations from disk."""
        convos_dir = self.user_data.base_path / "conversations"
        if convos_dir.exists():
            for f in convos_dir.glob("*.json"):
                try:
                    with open(f) as file:
                        data = json.load(file)
                        self.conversations[f.stem] = data.get("messages", [])
                except Exception:
                    pass
    
    def _save_conversation(self, conversation_id: str):
        """Save a conversation to disk."""
        convos_dir = self.user_data.base_path / "conversations"
        convos_dir.mkdir(parents=True, exist_ok=True)
        
        messages = self.conversations.get(conversation_id, [])
        data = {
            "id": conversation_id,
            "messages": messages,
            "updated_at": time.time(),
            "title": self._generate_title(messages),
        }
        
        with open(convos_dir / f"{conversation_id}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def _generate_title(self, messages: list) -> str:
        """Generate a title from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content.startswith("?"):
                    continue
                return content[:50] + ("..." if len(content) > 50 else "")
        return "New conversation"
    
    def get_conversation_list(self) -> list[dict]:
        """Get list of all conversations with metadata."""
        convos_dir = self.user_data.base_path / "conversations"
        result = []
        
        if convos_dir.exists():
            for f in sorted(convos_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
                try:
                    with open(f) as file:
                        data = json.load(file)
                        result.append({
                            "id": f.stem,
                            "title": data.get("title", "Untitled"),
                            "updated_at": data.get("updated_at", 0),
                            "message_count": len(data.get("messages", [])),
                        })
                except Exception:
                    pass
        
        return result
    
    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        """Get a specific conversation."""
        convos_dir = self.user_data.base_path / "conversations"
        path = convos_dir / f"{conversation_id}.json"
        
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        convos_dir = self.user_data.base_path / "conversations"
        path = convos_dir / f"{conversation_id}.json"
        
        if path.exists():
            path.unlink()
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
            return True
        return False
    
    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        """Rename a conversation."""
        convos_dir = self.user_data.base_path / "conversations"
        path = convos_dir / f"{conversation_id}.json"
        
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            data["title"] = title
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return True
        return False
    
    async def chat_stream(self, message: str, conversation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream a chat response with tool support."""
        
        # Pause chat while memory clustering runs (typically 3 AM introspection)
        if self.memory._clustering_in_progress:
            yield f"data: {json.dumps({'type': 'status', 'status': 'clustering'})}\n\n"
            yield f"data: {json.dumps({'type': 'content', 'content': 'Memory reorganization is in progress — your memories are being clustered right now. This usually takes under a minute. Please try again in a moment.'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Check for special commands first
        if self.commands.is_command(message):
            cmd_result = await self.commands.handle(message)
            if cmd_result.handled:
                yield f"data: {json.dumps({'type': 'command_response', 'content': cmd_result.response})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
        
        # Create or get conversation
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            self.conversations[conversation_id] = []
        elif conversation_id not in self.conversations:
            data = self.get_conversation(conversation_id)
            if data:
                self.conversations[conversation_id] = data.get("messages", [])
            else:
                self.conversations[conversation_id] = []
        
        messages = self.conversations[conversation_id]
        messages.append({"role": "user", "content": message})
        
        # Yield conversation ID
        yield f"data: {json.dumps({'type': 'conversation_id', 'id': conversation_id})}\n\n"
        
        # Generate a message_id for this response (used by feedback and confidence)
        message_id = str(uuid.uuid4())

        # Signal memory loading status (or background cycle if one is running)
        if self.introspection._in_progress:
            yield f"data: {json.dumps({'type': 'status', 'status': 'introspection'})}\n\n"
        elif self.introspection._idle_in_progress:
            yield f"data: {json.dumps({'type': 'status', 'status': 'research'})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'status', 'status': 'memory'})}\n\n"

        # Retrieve memory context once (shared by decision gate and system prompt)
        memory_ctx = self._get_memory_context(message)

        # MK13: Decision gate — evaluates after memory loaded
        gate_action = "answer"
        gate_confidence = 1.0
        if self.decision_gate_enabled:
            yield f"data: {json.dumps({'type': 'status', 'status': 'decision_gate'})}\n\n"
            try:
                bp = self.behavior_profile.get()
                gate = await self.decision_gate.evaluate(
                    query=message,
                    memory_context=memory_ctx,
                    client=self.client,
                    search_threshold=bp.get("search_threshold", 0.5),
                    ask_threshold=bp.get("ask_threshold", 0.3),
                    enabled=True,
                )
                gate_action = gate.action
                gate_confidence = gate.confidence
                yield f"data: {json.dumps({'type': 'gate_decision', 'action': gate_action, 'confidence': gate_confidence})}\n\n"
            except Exception:
                gate_action = "answer"
                gate_confidence = 0.5

        # Build system prompt (memory already retrieved above)
        system_prompt = self._build_system_prompt(message, memory_ctx=memory_ctx)
        request_messages = [{"role": "system", "content": system_prompt}] + messages[-20:]

        # Signal LLM thinking status
        yield f"data: {json.dumps({'type': 'status', 'status': 'thinking'})}\n\n"
        
        # Make LLM request with streaming
        content = ""
        tool_calls = []
        token_count = 0
        generation_start = None
        logprob_values: list[float] = []  # MK13: accumulate for confidence scoring

        # MK13: if gate says "ask", inject clarification instruction
        if gate_action == "ask":
            request_messages.append({
                "role": "system",
                "content": "The user's query is ambiguous. Ask a clarifying question before attempting to answer.",
            })

        # MK13: if gate says "search", force web_search via tool_choice
        gate_forced_search = gate_action == "search"

        active_tools = self.tools.get_active_tools()
        stream_request = {
            "model": self.model,
            "messages": request_messages,
            "max_tokens": 2000,
            "tools": active_tools,
            "stream": True,
        }
        # MK13: llama-server rejects logprobs + tools + stream simultaneously
        if not active_tools:
            stream_request["logprobs"] = True
        if gate_forced_search:
            stream_request["tool_choice"] = {"type": "function", "function": {"name": "web_search"}}

        try:
            async with self.client.stream(
                "POST",
                f"{self.llm_url}/v1/chat/completions",
                json=stream_request,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        choice_chunk = chunk.get("choices", [{}])[0]
                        delta = choice_chunk.get("delta", {})

                        # Handle content tokens
                        token = delta.get("content", "")
                        if token:
                            if generation_start is None:
                                generation_start = time.time()
                            token_count += 1
                            content += token
                            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                        # MK13: collect logprob for each token
                        chunk_logprobs = choice_chunk.get("logprobs", {})
                        if chunk_logprobs and isinstance(chunk_logprobs, dict):
                            for lp_entry in (chunk_logprobs.get("content") or []):
                                lp = lp_entry.get("logprob")
                                if lp is not None:
                                    logprob_values.append(float(lp))

                        # Handle tool calls
                        tc_delta = delta.get("tool_calls", [])
                        for tc in tc_delta:
                            idx = tc.get("index", 0)
                            while len(tool_calls) <= idx:
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})

                            if tc.get("id"):
                                tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                tool_calls[idx]["function"]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                tool_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
        
        # Filter out empty/incomplete tool calls
        tool_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name")]
        
        # Check for inline tool calls
        if not tool_calls and content:
            remaining, inline_calls = parse_inline_tool_calls(content)
            if inline_calls:
                tool_calls = inline_calls
                content = remaining
        
        # Handle tool calls
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                tool_name = func.get("name", "")
                tool_call_id = tc.get("id", "call_1")
                
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    continue
                
                yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_name, 'args': args})}\n\n"
                
                # Signal tool execution status
                yield f"data: {json.dumps({'type': 'status', 'status': 'tool'})}\n\n"
                
                # Execute tool
                tool_output = await self.tools.execute(tool_name, args)

                # Log tool errors to error journal for self-improvement analysis
                if tool_output.startswith("[Error"):
                    self.introspection.error_journal.log_error(
                        tool_name=tool_name,
                        error_type="tool_error",
                        error_message=tool_output[:200],
                        context=message[:200],
                    )

                # Check if this requires user approval (file operations)
                if tool_output.startswith("[PENDING_APPROVAL:") and self.tools.pending_approval:
                    pending = self.tools.pending_approval
                    yield f"data: {json.dumps({'type': 'approval_required', 'approval': pending.to_dict()})}\n\n"
                    # Return early - frontend will handle approval and call /api/approve endpoint
                    return
                
                yield f"data: {json.dumps({'type': 'tool_output', 'name': tool_name, 'output': tool_output[:2000]})}\n\n"
                
                # Add to messages for follow-up
                request_messages.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": func.get("arguments", "{}")}
                    }]
                })
                request_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_output
                })
            
            # Signal thinking status for follow-up response
            yield f"data: {json.dumps({'type': 'status', 'status': 'thinking'})}\n\n"
            
            # Get follow-up response
            try:
                followup_tools = self.tools.get_active_tools()
                followup_request = {
                    "model": self.model,
                    "messages": request_messages,
                    "max_tokens": 2000,
                    "tools": followup_tools,
                    "stream": True,
                }
                # MK13: llama-server rejects logprobs + tools + stream simultaneously
                if not followup_tools:
                    followup_request["logprobs"] = True
                async with self.client.stream(
                    "POST",
                    f"{self.llm_url}/v1/chat/completions",
                    json=followup_request,
                ) as resp:
                    content = ""
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            choice_chunk = chunk.get("choices", [{}])[0]
                            token = choice_chunk.get("delta", {}).get("content", "")
                            if token:
                                if generation_start is None:
                                    generation_start = time.time()
                                token_count += 1
                                content += token
                                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                            # MK13: collect logprobs from follow-up too
                            chunk_logprobs = choice_chunk.get("logprobs", {})
                            if chunk_logprobs and isinstance(chunk_logprobs, dict):
                                for lp_entry in (chunk_logprobs.get("content") or []):
                                    lp = lp_entry.get("logprob")
                                    if lp is not None:
                                        logprob_values.append(float(lp))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return
        
        # MK13: compute response confidence from accumulated logprobs
        import math as _math
        response_confidence: float = 0.5
        if logprob_values:
            avg_lp = sum(logprob_values) / len(logprob_values)
            response_confidence = round(max(0.05, min(1.0, _math.exp(avg_lp))), 3)

        # Save response
        tool_used = tool_calls[0].get("function", {}).get("name") if tool_calls else None
        if content:
            messages.append({"role": "assistant", "content": content})
            self._save_conversation(conversation_id)

            # Store in memory (background)
            asyncio.create_task(self._store_memory(message, content))

            self.commands.set_last_exchange(message, content)

        # MK13: log to experience log (fire-and-forget)
        try:
            self.experience_log.record(
                message_id=message_id,
                gate_action=gate_action,
                gate_confidence=gate_confidence,
                response_confidence=response_confidence,
                tool_used=tool_used,
            )
        except Exception:
            pass

        # MK13: emit confidence to client (only if setting enabled)
        if self.show_confidence:
            yield f"data: {json.dumps({'type': 'confidence', 'value': response_confidence, 'message_id': message_id})}\n\n"

        # Calculate generation stats
        stats = {}
        if generation_start is not None and token_count > 0:
            elapsed = time.time() - generation_start
            if elapsed > 0:
                stats = {
                    "tokens": token_count,
                    "elapsed": round(elapsed, 2),
                    "tokens_per_sec": round(token_count / elapsed, 1),
                }

        yield f"data: {json.dumps({'type': 'done', 'stats': stats, 'message_id': message_id})}\n\n"
    
    async def _store_memory(self, user_message: str, assistant_response: str):
        """Store conversation in memory system."""
        try:
            await self.memory.queue_conversation(
                user_message,
                assistant_response,
                self.client,
                on_status=lambda x: None
            )
        except Exception as e:
            print(f"[Memory] Error storing memory: {e}")
    
    def _get_memory_context(self, query: str) -> str:
        """Return raw memory context string for a query (used by decision gate and system prompt)."""
        return self.memory.get_relevant_context(query) or ""

    def _build_system_prompt(self, query: str, memory_ctx: str = "") -> str:
        """Build system prompt with memory context (includes research findings via memory)."""
        prompt = SYSTEM_PROMPT_BASE

        # Add custom-installed tool descriptions so LLM knows about them
        custom_tool_lines = self.tools.custom_registry.get_prompt_additions()
        if custom_tool_lines:
            prompt += f"\n\nCUSTOM TOOLS (user-installed):\n{custom_tool_lines}"

        # Add custom prompt from self-improvement
        custom = self.introspection.self_improve.get_custom_prompt()
        if custom:
            prompt += f"\n\n[LEARNED BEHAVIORS]\n{custom}"

        # MK13: Behavior profile fragment (verbosity, uncertainty style, etc.)
        bp_fragment = self.behavior_profile.to_prompt_fragment()
        if bp_fragment:
            prompt += f"\n\n[BEHAVIOR PROFILE]\n{bp_fragment}"

        # Add compact user profile (priority-4/5 facts, always present)
        profile = self.memory.get_user_profile()
        if profile:
            prompt += f"\n\n[USER PROFILE]\n{profile}"

        # Add memory context (retrieved once before this call; passed in to avoid double-fetch)
        if not memory_ctx:
            memory_ctx = self._get_memory_context(query)
        if memory_ctx:
            prompt += f"\n\n{memory_ctx}"

        return prompt
    
    async def close(self):
        """Cleanup resources."""
        await self.client.aclose()
        await self.tools.close()
        self.introspection.stop()


def get_user_core(user: User) -> UserLLMCore:
    """Get or create LLM core for a user."""
    if user.id not in user_cores:
        user_cores[user.id] = UserLLMCore(user)
    return user_cores[user.id]


# --- Dependency: Get current user from session ---

async def get_current_user(session_token: Optional[str] = Cookie(None, alias="session")) -> User:
    """Dependency to get current authenticated user."""
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = auth.validate_session(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return user


async def get_optional_user(session_token: Optional[str] = Cookie(None, alias="session")) -> Optional[User]:
    """Dependency to get current user if logged in."""
    if not session_token:
        return None
    return auth.validate_session(session_token)


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("[Server] Starting up...")
    scheduler.start()

    # Start autonomous session if configured
    _auto_session = None
    _server_config = _load_server_config()
    if _server_config.get("autonomous_mode"):
        from autonomous import AutonomousSession
        _auto_session = AutonomousSession(
            auth_system=auth,
            core_factory=get_user_core,
            config=_server_config,
        )
        await _auto_session.start()

    yield

    print("[Server] Shutting down...")
    if _auto_session:
        await _auto_session.stop()
    scheduler.stop()
    for core in user_cores.values():
        await core.close()


# --- App ---

app = FastAPI(
    title="3AM",
    description="Self-evolving local AI with persistent memory, decision gate, confidence scoring, and experience log",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth Endpoints ---

@app.post("/api/auth/register")
async def register(request: RegisterRequest, response: Response):
    """Register a new user."""
    try:
        user = auth.register(request.username, request.password)
        user_obj, token = auth.login(request.username, request.password)
        
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=86400,
            samesite="lax",
        )
        
        return {"user": user.to_public_dict(), "message": "Registration successful"}
    
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/auth/login")
async def login(request: LoginRequest, response: Response):
    """Login a user."""
    try:
        user, token = auth.login(request.username, request.password)
        
        response.set_cookie(
            key="session",
            value=token,
            httponly=True,
            max_age=86400,
            samesite="lax",
        )
        
        return {"user": user.to_public_dict(), "message": "Login successful"}
    
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/auth/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(None, alias="session")):
    """Logout current user."""
    if session_token:
        auth.logout(session_token)
    
    response.delete_cookie("session")
    return {"message": "Logged out"}


@app.get("/api/auth/me")
async def get_me(user: User = Depends(get_current_user)):
    """Get current user info."""
    return {"user": user.to_public_dict()}


@app.post("/api/auth/change-password")
async def change_password(request: PasswordChange, response: Response, user: User = Depends(get_current_user)):
    """Change user password."""
    try:
        auth.change_password(user, request.old_password, request.new_password)
        response.delete_cookie("session")
        return {"message": "Password changed. Please log in again."}
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Chat Endpoints ---

@app.post("/api/chat")
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    """Send a chat message and stream the response."""
    # Rate limiting: max 1 message per CHAT_RATE_LIMIT_SECONDS, max CHAT_MAX_CONCURRENT active
    now = time.time()
    last = _user_last_message.get(user.id, 0.0)
    wait = CHAT_RATE_LIMIT_SECONDS - (now - last)
    print(f"[RateLimit] user={user.id} now={now:.3f} last={last:.3f} elapsed={now-last:.3f}s wait={wait:.3f}s", flush=True)
    if wait > 0:
        print(f"[RateLimit] 429 → user={user.id} must wait {int(wait)+1}s", flush=True)
        raise HTTPException(status_code=429, detail=f"Rate limited — wait {int(wait)+1}s before sending another message")
    active = _user_active_requests.get(user.id, 0)
    if active >= CHAT_MAX_CONCURRENT:
        raise HTTPException(status_code=429, detail="Too many concurrent requests — wait for your current message to complete")

    _user_last_message[user.id] = now
    _user_active_requests[user.id] = active + 1

    core = get_user_core(user)

    async def stream_and_release():
        try:
            async for chunk in core.chat_stream(request.message, request.conversation_id):
                yield chunk
        finally:
            _user_active_requests[user.id] = max(0, _user_active_requests.get(user.id, 1) - 1)

    return StreamingResponse(
        stream_and_release(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for chat — supports mid-stream cancellation and server-push."""
    session_token = websocket.cookies.get("session")
    user = auth.validate_session(session_token) if session_token else None
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()
    _user_ws_connections[user.id] = websocket
    core = get_user_core(user)
    active_task: Optional[asyncio.Task] = None

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            msg_type = data.get("type")

            if msg_type == "cancel":
                if active_task and not active_task.done():
                    active_task.cancel()
                continue

            # MK13: handle inline feedback via WebSocket
            if msg_type == "feedback":
                try:
                    core.experience_log.add_feedback(
                        message_id=data.get("message_id", ""),
                        value=data.get("value", ""),
                        tags=data.get("tags", []),
                    )
                except Exception:
                    pass
                continue

            if msg_type != "message":
                continue

            message = data.get("content", "").strip()
            conversation_id = data.get("conversation_id")
            if not message:
                continue

            # Rate limiting
            now = time.time()
            last = _user_last_message.get(user.id, 0.0)
            wait = CHAT_RATE_LIMIT_SECONDS - (now - last)
            if wait > 0:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Rate limited — wait {int(wait)+1}s before sending another message",
                })
                continue

            if active_task and not active_task.done():
                await websocket.send_json({"type": "error", "message": "Already processing a message"})
                continue

            _user_last_message[user.id] = now

            async def stream_response(msg=message, conv_id=conversation_id):
                try:
                    async for chunk in core.chat_stream(msg, conv_id):
                        if chunk.startswith("data: "):
                            json_str = chunk[6:].strip()
                            if json_str and json_str != "[DONE]":
                                try:
                                    await websocket.send_json(json.loads(json_str))
                                except Exception:
                                    pass
                except asyncio.CancelledError:
                    try:
                        await websocket.send_json({"type": "cancelled"})
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        await websocket.send_json({"type": "error", "message": str(e)})
                    except Exception:
                        pass

            active_task = asyncio.create_task(stream_response())

    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
        _user_ws_connections.pop(user.id, None)


@app.get("/api/conversations")
async def list_conversations(user: User = Depends(get_current_user)):
    """List all conversations for current user."""
    core = get_user_core(user)
    return {"conversations": core.get_conversation_list()}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user: User = Depends(get_current_user)):
    """Get a specific conversation."""
    core = get_user_core(user)
    conversation = core.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return conversation


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user: User = Depends(get_current_user)):
    """Delete a conversation."""
    core = get_user_core(user)
    
    if not core.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted"}


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: ConversationRename, user: User = Depends(get_current_user)):
    """Rename a conversation."""
    core = get_user_core(user)
    
    if not core.rename_conversation(conversation_id, request.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation renamed"}


# --- File Approval Endpoints ---

@app.post("/api/approve")
async def handle_file_approval(request: ApprovalResponse, user: User = Depends(get_current_user)):
    """Handle user's approval/denial of a file operation."""
    core = get_user_core(user)
    
    if not core.tools.pending_approval:
        raise HTTPException(status_code=400, detail="No pending approval")
    
    if request.approved:
        result = core.tools.execute_pending_approval()
    else:
        result = core.tools.cancel_pending_approval()
    
    return {"message": result, "approved": request.approved}


# --- Memory Endpoints ---

@app.get("/api/memory")
async def get_memory(user: User = Depends(get_current_user)):
    """Get user's memory clusters with torque clustering stats."""
    core = get_user_core(user)
    clusters = []
    
    # Sort by torque mass (importance) descending
    sorted_clusters = sorted(
        core.memory.clusters.values(),
        key=lambda c: c.torque_mass,
        reverse=True
    )
    
    for c in sorted_clusters:
        clusters.append({
            "id": c.id,
            "theme": c.theme,
            "priority": c.priority,
            "message_count": len(c.message_refs),
            "mass": round(c.torque_mass, 1),  # MK10: torque mass
        })
    
    # Memory system stats
    stats = core.memory.get_stats()
    
    return {
        "clusters": clusters,
        "stats": {
            "total_messages": stats["total_messages"],
            "total_clusters": stats["active_clusters"],
            "needs_reclustering": stats["needs_reclustering"],
            "last_clustering": stats["last_clustering"],
            "clustering_method": "torque_clustering",
        }
    }


# --- Memory Management Endpoints ---

@app.get("/api/memory/export")
async def export_memory(user: User = Depends(get_current_user)):
    """Export all memories as a downloadable JSON file."""
    core = get_user_core(user)
    data = core.memory.export_all()
    filename = f"memory_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/memory/import")
async def import_memory(request: Request, user: User = Depends(get_current_user)):
    """Import memories from an export JSON. Clears existing memories first."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict) or "memories" not in body:
        raise HTTPException(
            status_code=400,
            detail='Invalid export format — expected {"memories": [...], ...}',
        )

    core = get_user_core(user)
    result = await core.memory.import_all(body)
    return result


@app.delete("/api/memory")
async def delete_all_memories(user: User = Depends(get_current_user)):
    """Delete all memories, experience log, research findings, and behavior profile for the current user."""
    core = get_user_core(user)
    core.memory.delete_all_memories()

    # Wipe research findings file
    research_file = core.user_data.base_path / "research.json"
    if research_file.exists():
        research_file.unlink()

    # Reset behavior profile to defaults
    core.behavior_profile.reset()

    return {"message": "All memories deleted"}


@app.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str, user: User = Depends(get_current_user)):
    """Delete a single memory by ID."""
    core = get_user_core(user)
    if not core.memory.delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory deleted"}


@app.get("/api/memory/viz")
async def get_memory_viz(user: User = Depends(get_current_user)):
    """Return 3D-projected memory positions for the star-map visualizer.

    CPU-bound (UMAP or PCA) — runs in a thread pool so the event loop stays live.
    """
    import asyncio
    core = get_user_core(user)
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, core.memory.get_viz_data)
    return data


# --- Introspection Endpoints ---

@app.get("/api/introspection/stats")
async def get_introspection_stats(user: User = Depends(get_current_user)):
    """Get introspection statistics."""
    core = get_user_core(user)
    return core.introspection.get_stats()


@app.post("/api/introspection/trigger")
async def trigger_introspection(
    force_recluster: bool = False,
    user: User = Depends(get_current_user),
):
    """Manually trigger introspection.
    Pass ?force_recluster=true to run a full Torque Clustering pass immediately,
    bypassing the weekly schedule and split-only logic.
    """
    core = get_user_core(user)
    if force_recluster:
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            cluster_result = await core.memory.run_torque_clustering_async(client, mode="full")
        return {"torque_clustering": cluster_result}
    result = await core.introspection.run_cycle()
    return result


# --- Research Endpoints ---

@app.get("/api/research/status")
async def get_research_status(user: User = Depends(get_current_user)):
    """Get research status."""
    core = get_user_core(user)
    return core.introspection.research.get_stats()


@app.post("/api/research/toggle")
async def toggle_research(user: User = Depends(get_current_user)):
    """Toggle research mode."""
    core = get_user_core(user)
    if core.introspection.research.is_enabled():
        core.introspection.research.disable()
    else:
        core.introspection.research.enable()
    return {"enabled": core.introspection.research.is_enabled()}


@app.get("/api/research/findings")
async def get_research_findings(user: User = Depends(get_current_user)):
    """Get research findings."""
    core = get_user_core(user)
    return {"summary": core.introspection.research.get_findings_summary()}


@app.get("/api/research/findings/download")
async def download_research_findings(user: User = Depends(get_current_user)):
    """Download all research findings as a text file."""
    from fastapi.responses import PlainTextResponse
    from datetime import datetime
    
    core = get_user_core(user)
    content = core.introspection.research.get_all_findings()
    
    filename = f"research_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    return PlainTextResponse(
        content=content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@app.get("/api/research/data")
async def get_research_data(user: User = Depends(get_current_user)):
    """Get all research topics and insights for the panel UI."""
    core = get_user_core(user)
    research = core.introspection.research
    return {
        "topics": [t.to_dict() for t in research.topics],
        "insights": [i.to_dict() for i in research.insights],
        "stats": research.get_stats(),
    }


@app.delete("/api/research/topic/{idx}")
async def delete_research_topic(idx: int, user: User = Depends(get_current_user)):
    """Delete a research topic by list index."""
    core = get_user_core(user)
    if not core.introspection.research.delete_topic(idx):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Topic index out of range")
    return {"message": "Deleted"}


@app.delete("/api/research/insight/{idx}")
async def delete_research_insight(idx: int, user: User = Depends(get_current_user)):
    """Delete a research insight and its corresponding memory entry."""
    core = get_user_core(user)
    research = core.introspection.research

    # Capture memory_id before the insight is removed
    insights = research.insights
    if idx < 0 or idx >= len(insights):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Insight index out of range")

    memory_id = insights[idx].memory_id
    research.delete_insight(idx)

    # Cascade: remove the corresponding memory entry if one was recorded
    if memory_id:
        core.memory.delete_memory(memory_id)

    return {"message": "Deleted"}


# --- Self-Improvement Endpoints ---

@app.get("/api/suggestions")
async def get_suggestions(user: User = Depends(get_current_user)):
    """Get self-improvement suggestions."""
    core = get_user_core(user)
    pending = core.introspection.self_improve.get_pending_suggestions()
    return {"suggestions": [s.to_dict() for s in pending]}


@app.post("/api/suggestions/{suggestion_id}/approve")
async def approve_suggestion(suggestion_id: str, user: User = Depends(get_current_user)):
    """Approve a suggestion."""
    core = get_user_core(user)
    if core.introspection.self_improve.approve_suggestion(suggestion_id):
        return {"message": "Suggestion approved"}
    raise HTTPException(status_code=404, detail="Suggestion not found")


@app.post("/api/suggestions/{suggestion_id}/dismiss")
async def dismiss_suggestion(suggestion_id: str, user: User = Depends(get_current_user)):
    """Dismiss a suggestion."""
    core = get_user_core(user)
    if core.introspection.self_improve.dismiss_suggestion(suggestion_id):
        return {"message": "Suggestion dismissed"}
    raise HTTPException(status_code=404, detail="Suggestion not found")


# --- Settings Endpoints ---

@app.get("/api/settings")
async def get_settings(user: User = Depends(get_current_user)):
    """Get user settings."""
    core = get_user_core(user)
    return {
        "settings": user.settings,
        "research_enabled": core.introspection.research.is_enabled(),
        "self_improve_enabled": core.introspection.self_improve.is_enabled(),
        "search_provider": core.search_provider,
        # MK13 settings
        "decision_gate_enabled": core.decision_gate_enabled,
        "decision_gate_sensitivity": core.decision_gate_sensitivity,
        "show_confidence": core.show_confidence,
        "show_feedback_buttons": core.show_feedback_buttons,
    }


@app.patch("/api/settings")
async def update_settings(request: SettingsUpdate, user: User = Depends(get_current_user)):
    """Update user settings."""
    core = get_user_core(user)

    if request.research_enabled is not None:
        if request.research_enabled:
            core.introspection.research.enable()
        else:
            core.introspection.research.disable()

    if request.self_improve_enabled is not None:
        if request.self_improve_enabled:
            core.introspection.self_improve.enable()
        else:
            core.introspection.self_improve.disable()

    if request.search_provider is not None:
        core.set_search_provider(request.search_provider)

    # MK13 settings
    mk13_changed = False
    if request.decision_gate_enabled is not None:
        core.decision_gate_enabled = request.decision_gate_enabled
        mk13_changed = True
    if request.decision_gate_sensitivity is not None:
        sens = max(0.0, min(1.0, request.decision_gate_sensitivity))
        core.decision_gate_sensitivity = sens
        # Mirror sensitivity to behavior profile thresholds
        core.behavior_profile.update({
            "search_threshold": round(sens, 2),
            "ask_threshold": round(sens * 0.6, 2),
        })
        mk13_changed = True
    if request.show_confidence is not None:
        core.show_confidence = request.show_confidence
        mk13_changed = True
    if request.show_feedback_buttons is not None:
        core.show_feedback_buttons = request.show_feedback_buttons
        mk13_changed = True
    if mk13_changed:
        core._save_mk13_settings()

    return {"message": "Settings updated"}


# --- MK13: Feedback & Behavior Profile Endpoints ---

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, user: User = Depends(get_current_user)):
    """Submit feedback for a response (REST fallback — WebSocket is preferred)."""
    core = get_user_core(user)
    found = core.experience_log.add_feedback(
        message_id=request.message_id,
        value=request.value,
        tags=request.tags,
    )
    if not found:
        raise HTTPException(status_code=404, detail="Message ID not found in experience log")
    return {"message": "Feedback recorded"}


@app.get("/api/feedback/stats")
async def get_feedback_stats(user: User = Depends(get_current_user)):
    """Get aggregated feedback statistics for the settings panel."""
    core = get_user_core(user)
    return core.experience_log.get_stats()


@app.get("/api/analytics")
async def get_analytics(user: User = Depends(get_current_user)):
    """Comprehensive analytics for the settings panel."""
    core = get_user_core(user)
    data = core.experience_log.get_analytics()
    data["behavior_profile"] = core.behavior_profile.get()
    return data


@app.get("/api/behavior-profile")
async def get_behavior_profile(user: User = Depends(get_current_user)):
    """Get current behavior profile."""
    core = get_user_core(user)
    return core.behavior_profile.get()


@app.patch("/api/behavior-profile")
async def update_behavior_profile(request: BehaviorProfileUpdate, user: User = Depends(get_current_user)):
    """Update behavior profile fields."""
    core = get_user_core(user)
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    core.behavior_profile.update(updates)
    return {"message": "Behavior profile updated", "profile": core.behavior_profile.get()}


# --- Custom Tools API ---

class ToolProposeRequest(BaseModel):
    description: str


@app.get("/api/tools")
async def list_tools(user: User = Depends(get_current_user)):
    """List all custom tools (installed + proposals)."""
    core = get_user_core(user)
    tools = core.introspection.self_improve.proposed_tools
    return {
        "installed": [t.to_dict() for t in tools if t.status == "installed"],
        "code_ready": [t.to_dict() for t in tools if t.status == "code_ready"],
        "proposals": [t.to_dict() for t in tools if t.status == "proposal"],
    }


@app.post("/api/tools/propose")
async def propose_tool(request: ToolProposeRequest, user: User = Depends(get_current_user)):
    """Stage 1 — propose a new tool concept (name + schema, no code)."""
    core = get_user_core(user)
    tool = await core.introspection.self_improve.generate_tool_proposal(
        request.description, core.client
    )
    if not tool:
        raise HTTPException(
            status_code=409,
            detail="Could not generate proposal — a tool with that name may already exist. Try rephrasing."
        )
    return tool.to_dict()


@app.post("/api/tools/{tool_id}/generate")
async def generate_tool_code(tool_id: str, user: User = Depends(get_current_user)):
    """Stage 2 — generate Python code for a proposal."""
    core = get_user_core(user)
    si = core.introspection.self_improve
    tool = si._find_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    if tool.status not in ("proposal",):
        raise HTTPException(status_code=409, detail=f"Tool is in state '{tool.status}', expected 'proposal'")
    result = await si.generate_tool_code(tool_id, core.client)
    if not result:
        raise HTTPException(status_code=500, detail="Code generation failed — try again")
    return result.to_dict()


@app.post("/api/tools/{tool_id}/install")
async def install_tool(tool_id: str, user: User = Depends(get_current_user)):
    """Stage 3 — safety-check and install a code-ready tool."""
    core = get_user_core(user)
    ok, message = core.introspection.self_improve.install_tool(
        tool_id, core.tools.custom_registry
    )
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"message": message}


@app.delete("/api/tools/{tool_id}")
async def remove_tool(tool_id: str, user: User = Depends(get_current_user)):
    """Uninstall a custom tool by its ID."""
    core = get_user_core(user)
    si = core.introspection.self_improve
    tool = si._find_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    ok, message = si.remove_tool(tool.name, core.tools.custom_registry)
    if not ok:
        raise HTTPException(status_code=409, detail=message)
    return {"message": message}


class ToolRetryProposalRequest(BaseModel):
    feedback: str


class ToolRetryCodeRequest(BaseModel):
    feedback: str
    explanation: str


@app.post("/api/tools/{tool_id}/explain")
async def explain_tool_code(tool_id: str, user: User = Depends(get_current_user)):
    """Return a plain-text pseudo-code explanation of a code_ready tool."""
    core = get_user_core(user)
    si = core.introspection.self_improve
    tool = si._find_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    if tool.status != "code_ready" or not tool.code:
        raise HTTPException(status_code=409, detail=f"Tool '{tool.name}' has no code to explain (status: {tool.status})")
    explanation = await si.explain_tool_code(tool_id, core.client)
    if not explanation:
        raise HTTPException(status_code=500, detail="Could not generate explanation — try again")
    return {"explanation": explanation}


@app.post("/api/tools/{tool_id}/retry-proposal")
async def retry_tool_proposal(tool_id: str, request: ToolRetryProposalRequest, user: User = Depends(get_current_user)):
    """Revise an existing proposal in-place based on user feedback."""
    core = get_user_core(user)
    si = core.introspection.self_improve
    tool = si._find_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    if tool.status != "proposal":
        raise HTTPException(status_code=409, detail=f"Tool is in state '{tool.status}', expected 'proposal'")
    result = await si.retry_tool_proposal(tool_id, request.feedback, core.client)
    if not result:
        raise HTTPException(status_code=500, detail="Could not revise proposal — try again")
    return result.to_dict()


@app.post("/api/tools/{tool_id}/retry-code")
async def retry_tool_code(tool_id: str, request: ToolRetryCodeRequest, user: User = Depends(get_current_user)):
    """Regenerate code for a code_ready tool using user feedback."""
    core = get_user_core(user)
    si = core.introspection.self_improve
    tool = si._find_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    if tool.status != "code_ready":
        raise HTTPException(status_code=409, detail=f"Tool is in state '{tool.status}', expected 'code_ready'")
    result = await si.retry_tool_code(tool_id, request.feedback, request.explanation, core.client)
    if not result:
        raise HTTPException(status_code=500, detail="Code regeneration failed — try again")
    return result.to_dict()


# --- Static Files & Frontend ---

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text()
    
    return """
<!DOCTYPE html>
<html><head><title>3AM</title></head>
<body style="background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:2rem;">
<h1>3AM</h1>
<p>Server running. <a href="/docs" style="color:#58a6ff;">API docs</a></p>
</body></html>
"""


# --- Run ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

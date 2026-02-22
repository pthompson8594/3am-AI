#!/usr/bin/env python3
"""
Self-Improvement System - LLM suggests its own upgrades.

The LLM can:
- Identify limitations and problems with tools
- Suggest new tools or improvements
- Propose adjustments to its own system prompt
- Queue suggestions for user approval
- Apply approved changes (prompt additions, tool examples)

This system is OPT-IN and requires explicit user approval for all changes.
"""

import ast
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
import httpx


DATA_DIR = Path.home() / ".local/share/3am"
SUGGESTIONS_FILE = DATA_DIR / "suggestions.json"
SELF_IMPROVE_CONFIG_FILE = DATA_DIR / "self_improve_config.json"
CUSTOM_PROMPT_FILE = DATA_DIR / "custom_prompt.txt"
TOOL_EXAMPLES_FILE = DATA_DIR / "tool_examples.txt"


ANALYZE_ERRORS_PROMPT = """Review these error patterns from tool usage:

{errors}

Based on these patterns, suggest ONE specific improvement. This could be:
- A fix to how I use an existing tool
- A new tool that would help
- A change to my behavior

Respond with JSON:
{{
  "suggestion_type": "tool_fix" | "new_tool" | "behavior_change",
  "title": "<short title>",
  "description": "<detailed explanation of the problem and proposed solution>",
  "implementation_hint": "<how this could be implemented>",
  "priority": <1-5, higher = more impactful>
}}

Only suggest if there's a clear pattern. If errors are random/minor, respond:
{{"suggestion_type": "none"}}"""


ANALYZE_CAPABILITIES_PROMPT = """Review these queries where I couldn't fully help the user:

{queries}

Do I seem to be missing a capability that would help? Consider:
- Tools I wish I had
- Knowledge gaps I could fill
- Behaviors that could be improved

Respond with JSON:
{{
  "suggestion_type": "new_tool" | "knowledge_gap" | "behavior_change",
  "title": "<short title>",
  "description": "<what I'm missing and how it would help>",
  "implementation_hint": "<how this could be added>",
  "priority": <1-5>
}}

Only suggest if there's a clear need. Otherwise respond:
{{"suggestion_type": "none"}}"""


PROMPT_IMPROVEMENT_PROMPT = """Based on my interactions, I want to improve my system prompt.

Current custom additions:
{current_additions}

Recent feedback patterns:
{feedback}

Suggest ONE addition or modification to improve my responses. This could be:
- A reminder about user preferences
- A tool usage tip I learned
- A behavior adjustment

Respond with JSON:
{{
  "action": "add" | "modify" | "none",
  "content": "<the text to add/modify>",
  "reason": "<why this would help>"
}}"""


SELF_REFLECTION_PROMPT = """I am an LLM assistant. I want to identify areas where I could improve.

My recent error patterns:
{errors}

Recent negative feedback:
{feedback}

My current capabilities:
- Web search, file reading, command execution, clipboard, system info
- Memory system that learns about the user
- Tool calling for various tasks

Based on this, what should I research to become better? Think about:
- Technical skills I'm lacking
- Response quality issues
- Tool usage problems
- Understanding gaps

Respond with JSON:
{{
  "self_research_topics": [
    {{
      "question": "<specific research question about how to improve>",
      "reason": "<why this would help me be better>",
      "priority": <1-5>
    }}
  ]
}}

Generate 1-3 focused, actionable research questions. If I'm doing well, respond:
{{"self_research_topics": []}}"""


APPLY_SELF_RESEARCH_PROMPT = """I researched this topic to improve myself:

Topic: {topic}
Research findings:
{findings}

Based on what I learned, what specific change should I make to my behavior or system prompt?

Respond with JSON:
{{
  "has_actionable_insight": true | false,
  "prompt_addition": "<text to add to my system prompt, or empty>",
  "behavior_note": "<what I learned about how to behave better>",
  "summary": "<one sentence summary of the improvement>"
}}"""


TOOL_PROPOSAL_PROMPT = """You are designing a new Python tool for an LLM assistant.

User request: {description}

Design this tool. Respond with JSON only:
{{
  "name": "<snake_case name, start with a verb, e.g. fetch_stock_price>",
  "description": "<one sentence: what this tool does and when to use it>",
  "parameters": {{
    "type": "object",
    "properties": {{
      "<param1>": {{
        "type": "<string|integer|number|boolean>",
        "description": "<what this parameter is>"
      }}
    }},
    "required": ["<param1>"]
  }},
  "prompt_addition": "<one bullet line for the system prompt, e.g. '- fetch_stock_price: Get real-time stock price for a ticker symbol'>"
}}

Rules:
- Name must be snake_case, 2-4 words, verb-first
- Parameters must cover exactly what the tool needs
- Do NOT include code or implementation details
- Do NOT propose tools that require subprocess, socket, threading, or OS-level access"""


TOOL_CODE_PROMPT = """You are implementing a Python async function for an LLM assistant tool.

Tool name: {name}
Description: {description}
Parameters schema:
{parameters_json}

Write the complete Python implementation. Respond with JSON only:
{{
  "code": "<complete Python source as a single string with \\n for newlines>"
}}

Requirements:
1. Function signature MUST be: async def tool_{name}(**kwargs) -> str:
   - Extract parameters from kwargs using kwargs.get()
   - Return a plain string result (or "[Error: ...]" on failure)
2. Allowed imports: json, re, math, datetime, pathlib, collections, itertools,
   functools, typing, os.path, urllib.parse, base64, hashlib, csv, io, textwrap,
   string, random. Use httpx for HTTP requests.
3. FORBIDDEN: subprocess, socket, threading, multiprocessing, ctypes, pty, tty,
   signal, os (bare), sys, importlib. No eval(), exec(), compile(), __import__().
4. Keep self-contained; handle exceptions; return "[Error: ...]" on failure.
5. Use httpx.AsyncClient for any HTTP calls.

Example:
import httpx

async def tool_example(**kwargs) -> str:
    query = kwargs.get("query", "")
    if not query:
        return "[Error: query is required]"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"https://api.example.com/{{query}}")
            return resp.text[:2000]
    except Exception as e:
        return f"[Error: {{e}}]"
"""


@dataclass
class ProposedTool:
    """A tool the LLM has proposed to create for itself."""
    id: str                          # "tool_<timestamp_ms>"
    name: str                        # snake_case; "" for stubs created from Suggestions
    description: str
    parameters: dict                 # JSON Schema object
    prompt_addition: str             # One bullet line for system prompt
    code: str                        # Empty until Stage 2 code generation
    status: str                      # "proposal" | "code_ready" | "installed" | "rejected"
    source_suggestion_id: str        # Suggestion.id that spawned this, or ""
    created_at: float
    code_generated_at: Optional[float] = None
    installed_at: Optional[float] = None
    rejection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "prompt_addition": self.prompt_addition,
            "code": self.code,
            "status": self.status,
            "source_suggestion_id": self.source_suggestion_id,
            "created_at": self.created_at,
            "code_generated_at": self.code_generated_at,
            "installed_at": self.installed_at,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProposedTool":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            prompt_addition=data.get("prompt_addition", ""),
            code=data.get("code", ""),
            status=data["status"],
            source_suggestion_id=data.get("source_suggestion_id", ""),
            created_at=data["created_at"],
            code_generated_at=data.get("code_generated_at"),
            installed_at=data.get("installed_at"),
            rejection_reason=data.get("rejection_reason", ""),
        )


@dataclass
class SelfResearchTopic:
    """A topic the LLM wants to research about itself."""
    question: str
    reason: str
    priority: int
    created_at: float
    researched: bool = False
    findings: str = ""
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reason": self.reason,
            "priority": self.priority,
            "created_at": self.created_at,
            "researched": self.researched,
            "findings": self.findings,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SelfResearchTopic":
        return cls(**data)


@dataclass
class SelfImproveConfig:
    """Configuration for self-improvement system."""
    enabled: bool = False
    allow_prompt_changes: bool = False  # Extra safety for prompt modifications
    allow_self_research: bool = False   # Allow LLM to research how to improve itself
    max_suggestions: int = 10
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "allow_prompt_changes": self.allow_prompt_changes,
            "allow_self_research": self.allow_self_research,
            "max_suggestions": self.max_suggestions,
        }
    
    @classmethod
    def load(cls) -> "SelfImproveConfig":
        try:
            if SELF_IMPROVE_CONFIG_FILE.exists():
                with open(SELF_IMPROVE_CONFIG_FILE) as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        except Exception:
            pass
        return cls()
    
    def save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(SELF_IMPROVE_CONFIG_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception:
            pass


@dataclass
class Suggestion:
    """A self-improvement suggestion."""
    id: str
    suggestion_type: str  # tool_fix, new_tool, behavior_change, knowledge_gap, prompt_change
    title: str
    description: str
    implementation_hint: str
    priority: int
    created_at: float
    status: str = "pending"  # pending, approved, dismissed, implemented
    approved_at: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "suggestion_type": self.suggestion_type,
            "title": self.title,
            "description": self.description,
            "implementation_hint": self.implementation_hint,
            "priority": self.priority,
            "created_at": self.created_at,
            "status": self.status,
            "approved_at": self.approved_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Suggestion":
        return cls(**data)


class SelfImproveSystem:
    """
    Self-improvement system that lets the LLM suggest its own upgrades.
    
    - Analyzes error patterns to suggest tool fixes
    - Identifies missing capabilities
    - Proposes system prompt adjustments
    - Can research how to improve itself
    - All changes require user approval
    """
    
    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        llm_model_id: str = "qwen3-14b",
        on_status: Optional[Callable[[str], None]] = None,
        web_search_fn: Optional[Callable] = None
    ):
        self.llm_url = llm_url
        self.llm_model_id = llm_model_id
        self.on_status = on_status or (lambda x: None)
        self.web_search_fn = web_search_fn
        self.config = SelfImproveConfig.load()
        self.suggestions: list[Suggestion] = []
        self.self_research_topics: list[SelfResearchTopic] = []
        self.proposed_tools: list[ProposedTool] = []
        self._load()

    def _load(self):
        try:
            if SUGGESTIONS_FILE.exists():
                with open(SUGGESTIONS_FILE) as f:
                    data = json.load(f)
                self.suggestions = [Suggestion.from_dict(s) for s in data.get("suggestions", [])]
                self.self_research_topics = [
                    SelfResearchTopic.from_dict(t)
                    for t in data.get("self_research_topics", [])
                ]
                self.proposed_tools = [
                    ProposedTool.from_dict(t)
                    for t in data.get("proposed_tools", [])
                ]
        except Exception as e:
            self.on_status(f"[SelfImprove] Load error: {e}")

    def _save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "suggestions": [s.to_dict() for s in self.suggestions[-50:]],
                "self_research_topics": [t.to_dict() for t in self.self_research_topics[-20:]],
                "proposed_tools": [t.to_dict() for t in self.proposed_tools],
                "last_update": time.time(),
            }
            with open(SUGGESTIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.on_status(f"[SelfImprove] Save error: {e}")
    
    def is_enabled(self) -> bool:
        return self.config.enabled
    
    def enable(self):
        self.config.enabled = True
        self.config.save()
        self.on_status("[SelfImprove] Enabled - LLM will suggest improvements")
    
    def disable(self):
        self.config.enabled = False
        self.config.save()
        self.on_status("[SelfImprove] Disabled")
    
    def enable_prompt_changes(self):
        """Allow LLM to suggest prompt modifications (extra opt-in)."""
        self.config.allow_prompt_changes = True
        self.config.save()
        self.on_status("[SelfImprove] Prompt changes enabled")
    
    def disable_prompt_changes(self):
        self.config.allow_prompt_changes = False
        self.config.save()
        self.on_status("[SelfImprove] Prompt changes disabled")
    
    def enable_self_research(self):
        """Allow LLM to research how to improve itself."""
        self.config.allow_self_research = True
        self.config.save()
        self.on_status("[SelfImprove] Self-research enabled - LLM can research how to improve")
    
    def disable_self_research(self):
        self.config.allow_self_research = False
        self.config.save()
        self.on_status("[SelfImprove] Self-research disabled")
    
    async def _llm_request(self, prompt: str, client: httpx.AsyncClient) -> Optional[dict]:
        """Make a request to the LLM."""
        try:
            response = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 400,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                }
            )
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            self.on_status(f"[SelfImprove] LLM error: {e}")
            return None
    
    async def analyze_errors(self, error_patterns: dict[str, int], client: httpx.AsyncClient) -> Optional[Suggestion]:
        """Analyze error patterns and suggest improvements."""
        if not error_patterns:
            return None
        
        errors_text = "\n".join([
            f"- {error}: occurred {count} times"
            for error, count in list(error_patterns.items())[:10]
        ])
        
        prompt = ANALYZE_ERRORS_PROMPT.format(errors=errors_text)
        result = await self._llm_request(prompt, client)
        
        if not result or result.get("suggestion_type") == "none":
            return None
        
        suggestion = Suggestion(
            id=f"sug_{int(time.time() * 1000)}",
            suggestion_type=result.get("suggestion_type", "tool_fix"),
            title=result.get("title", "Improvement suggestion"),
            description=result.get("description", ""),
            implementation_hint=result.get("implementation_hint", ""),
            priority=result.get("priority", 3),
            created_at=time.time(),
        )
        
        self.suggestions.append(suggestion)
        self._save()
        self.on_status(f"[SelfImprove] New suggestion: {suggestion.title}")
        
        return suggestion
    
    async def analyze_capability_gaps(self, failed_queries: list[str], client: httpx.AsyncClient) -> Optional[Suggestion]:
        """Analyze queries where the LLM struggled."""
        if not failed_queries:
            return None
        
        queries_text = "\n".join([f"- {q[:100]}" for q in failed_queries[:5]])
        prompt = ANALYZE_CAPABILITIES_PROMPT.format(queries=queries_text)
        result = await self._llm_request(prompt, client)
        
        if not result or result.get("suggestion_type") == "none":
            return None
        
        suggestion = Suggestion(
            id=f"sug_{int(time.time() * 1000)}",
            suggestion_type=result.get("suggestion_type", "new_tool"),
            title=result.get("title", "Capability suggestion"),
            description=result.get("description", ""),
            implementation_hint=result.get("implementation_hint", ""),
            priority=result.get("priority", 3),
            created_at=time.time(),
        )
        
        self.suggestions.append(suggestion)
        self._save()
        self.on_status(f"[SelfImprove] New suggestion: {suggestion.title}")
        
        return suggestion
    
    async def suggest_prompt_improvement(self, feedback_patterns: list[dict], client: httpx.AsyncClient) -> Optional[Suggestion]:
        """Suggest improvements to the system prompt."""
        if not self.config.allow_prompt_changes:
            return None
        
        current_additions = self.get_custom_prompt()
        feedback_text = "\n".join([
            f"- {f.get('comment', 'No comment')} (rating: {f.get('rating', 0)})"
            for f in feedback_patterns[:5]
        ])
        
        prompt = PROMPT_IMPROVEMENT_PROMPT.format(
            current_additions=current_additions or "(none)",
            feedback=feedback_text or "(no feedback)"
        )
        
        result = await self._llm_request(prompt, client)
        
        if not result or result.get("action") == "none":
            return None
        
        suggestion = Suggestion(
            id=f"sug_{int(time.time() * 1000)}",
            suggestion_type="prompt_change",
            title="System prompt improvement",
            description=result.get("reason", ""),
            implementation_hint=result.get("content", ""),
            priority=4,
            created_at=time.time(),
        )
        
        self.suggestions.append(suggestion)
        self._save()
        self.on_status(f"[SelfImprove] New prompt suggestion: {suggestion.title}")
        
        return suggestion
    
    def get_pending_suggestions(self) -> list[Suggestion]:
        """Get all pending suggestions."""
        return [s for s in self.suggestions if s.status == "pending"]
    
    def get_approved_suggestions(self) -> list[Suggestion]:
        """Get approved suggestions awaiting implementation."""
        return [s for s in self.suggestions if s.status == "approved"]
    
    def approve_suggestion(self, suggestion_id: str) -> bool:
        """Approve a suggestion."""
        for s in self.suggestions:
            if s.id == suggestion_id or suggestion_id.isdigit() and self.suggestions.index(s) == int(suggestion_id) - 1:
                s.status = "approved"
                s.approved_at = time.time()

                # If it's a prompt change, apply it immediately
                if s.suggestion_type == "prompt_change" and self.config.allow_prompt_changes:
                    self._apply_prompt_change(s.implementation_hint)
                    s.status = "implemented"

                # For new_tool suggestions, create a ProposedTool stub for Stage 1
                elif s.suggestion_type == "new_tool":
                    description = s.description
                    if s.implementation_hint:
                        description = f"{s.description}. Hint: {s.implementation_hint}"
                    stub = ProposedTool(
                        id=f"tool_{int(time.time() * 1000)}",
                        name="",          # filled when user runs ?approve-tool
                        description=description,
                        parameters={},
                        prompt_addition="",
                        code="",
                        status="proposal",
                        source_suggestion_id=s.id,
                        created_at=time.time(),
                    )
                    self.proposed_tools.append(stub)
                    s.status = "implemented"
                    self.on_status(f"[SelfImprove] Tool stub created: {stub.id} — use ?approve-tool {stub.id}")

                self._save()
                return True
        return False
    
    def dismiss_suggestion(self, suggestion_id: str) -> bool:
        """Dismiss a suggestion."""
        for s in self.suggestions:
            if s.id == suggestion_id or suggestion_id.isdigit() and self.suggestions.index(s) == int(suggestion_id) - 1:
                s.status = "dismissed"
                self._save()
                return True
        return False
    
    def mark_implemented(self, suggestion_id: str) -> bool:
        """Mark a suggestion as implemented."""
        for s in self.suggestions:
            if s.id == suggestion_id or suggestion_id.isdigit() and self.suggestions.index(s) == int(suggestion_id) - 1:
                s.status = "implemented"
                self._save()
                return True
        return False
    
    def _apply_prompt_change(self, content: str):
        """Apply a prompt addition."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Append to custom prompt file
            with open(CUSTOM_PROMPT_FILE, "a") as f:
                f.write(f"\n# Added {datetime.now().isoformat()}\n")
                f.write(content + "\n")
            
            self.on_status(f"[SelfImprove] Applied prompt addition")
        except Exception as e:
            self.on_status(f"[SelfImprove] Error applying prompt: {e}")
    
    def get_custom_prompt(self) -> str:
        """Get the custom prompt additions."""
        try:
            if CUSTOM_PROMPT_FILE.exists():
                return CUSTOM_PROMPT_FILE.read_text()
        except Exception:
            pass
        return ""
    
    def add_tool_example(self, example: str):
        """Add a tool usage example."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(TOOL_EXAMPLES_FILE, "a") as f:
                f.write(f"\n{example}\n")
            self.on_status("[SelfImprove] Added tool example")
        except Exception as e:
            self.on_status(f"[SelfImprove] Error adding example: {e}")
    
    def get_tool_examples(self) -> str:
        """Get custom tool examples."""
        try:
            if TOOL_EXAMPLES_FILE.exists():
                return TOOL_EXAMPLES_FILE.read_text()
        except Exception:
            pass
        return ""
    
    def get_suggestions_summary(self) -> str:
        """Get a formatted summary of suggestions."""
        pending = self.get_pending_suggestions()
        
        if not pending:
            return "No pending suggestions."
        
        lines = ["=== Improvement Suggestions ===", ""]
        
        for i, s in enumerate(pending, 1):
            type_emoji = {
                "tool_fix": "🔧",
                "new_tool": "🆕",
                "behavior_change": "💭",
                "knowledge_gap": "📚",
                "prompt_change": "📝",
                "self_research": "🔬",
            }.get(s.suggestion_type, "💡")
            
            lines.append(f"{type_emoji} Suggestion #{i}: {s.title}")
            lines.append(f"   {s.description[:200]}...")
            lines.append(f"   Priority: {'★' * s.priority}{'☆' * (5 - s.priority)}")
            lines.append("")
        
        lines.append("Commands: ?approve <#>, ?dismiss <#>, ?implemented <#>")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get self-improvement statistics."""
        return {
            "enabled": self.config.enabled,
            "allow_prompt_changes": self.config.allow_prompt_changes,
            "allow_self_research": self.config.allow_self_research,
            "total_suggestions": len(self.suggestions),
            "pending": len([s for s in self.suggestions if s.status == "pending"]),
            "approved": len([s for s in self.suggestions if s.status == "approved"]),
            "implemented": len([s for s in self.suggestions if s.status == "implemented"]),
            "dismissed": len([s for s in self.suggestions if s.status == "dismissed"]),
            "self_research_topics": len(self.self_research_topics),
            "researched_topics": len([t for t in self.self_research_topics if t.researched]),
        }
    
    def get_startup_message(self) -> Optional[str]:
        """Get startup message about pending suggestions."""
        pending = len(self.get_pending_suggestions())
        if pending > 0:
            return f"({pending} improvement suggestion{'s' if pending > 1 else ''} - use ?suggestions to view)"
        return None
    
    async def generate_self_research_topics(
        self, 
        error_patterns: dict[str, int],
        feedback: list[dict],
        client: httpx.AsyncClient
    ) -> list[SelfResearchTopic]:
        """Have the LLM generate questions about how it could improve."""
        if not self.config.allow_self_research:
            return []
        
        errors_text = "\n".join([
            f"- {error}: {count} times"
            for error, count in list(error_patterns.items())[:10]
        ]) or "(no errors)"
        
        feedback_text = "\n".join([
            f"- {f.get('comment', 'negative rating')}"
            for f in feedback if f.get('rating', 0) < 0
        ][:5]) or "(no negative feedback)"
        
        prompt = SELF_REFLECTION_PROMPT.format(
            errors=errors_text,
            feedback=feedback_text
        )
        
        result = await self._llm_request(prompt, client)
        
        if not result or not result.get("self_research_topics"):
            return []
        
        new_topics = []
        existing_questions = {t.question.lower() for t in self.self_research_topics}
        
        for topic_data in result["self_research_topics"]:
            question = topic_data.get("question", "").strip()
            if not question or question.lower() in existing_questions:
                continue
            
            topic = SelfResearchTopic(
                question=question,
                reason=topic_data.get("reason", ""),
                priority=topic_data.get("priority", 3),
                created_at=time.time(),
            )
            new_topics.append(topic)
            self.self_research_topics.append(topic)
        
        if new_topics:
            self._save()
            self.on_status(f"[SelfImprove] Generated {len(new_topics)} self-research questions")
        
        return new_topics
    
    async def research_self_improvement(self, client: httpx.AsyncClient) -> Optional[Suggestion]:
        """Research a self-improvement topic and generate a suggestion."""
        if not self.config.allow_self_research or not self.web_search_fn:
            return None
        
        # Find unresearched topic with highest priority
        unresearched = [t for t in self.self_research_topics if not t.researched]
        if not unresearched:
            return None
        
        unresearched.sort(key=lambda t: t.priority, reverse=True)
        topic = unresearched[0]
        
        self.on_status(f"[SelfImprove] Researching: {topic.question}")
        
        # Do web search
        try:
            search_results = await self.web_search_fn(topic.question)
        except Exception as e:
            self.on_status(f"[SelfImprove] Search error: {e}")
            return None
        
        if not search_results or search_results.startswith("[Error"):
            return None
        
        topic.researched = True
        topic.findings = search_results[:2000]
        
        # Ask LLM to apply what it learned
        prompt = APPLY_SELF_RESEARCH_PROMPT.format(
            topic=topic.question,
            findings=search_results[:3000]
        )
        
        result = await self._llm_request(prompt, client)
        
        if not result or not result.get("has_actionable_insight"):
            self._save()
            return None
        
        # Create suggestion from research
        suggestion = Suggestion(
            id=f"sug_{int(time.time() * 1000)}",
            suggestion_type="self_research",
            title=f"Self-improvement: {result.get('summary', topic.question)[:50]}",
            description=result.get("behavior_note", ""),
            implementation_hint=result.get("prompt_addition", ""),
            priority=4,
            created_at=time.time(),
        )
        
        self.suggestions.append(suggestion)
        self._save()
        
        self.on_status(f"[SelfImprove] Generated suggestion from self-research")
        
        return suggestion
    
    def get_self_research_summary(self) -> str:
        """Get summary of self-research topics."""
        if not self.self_research_topics:
            return "No self-research topics yet."

        lines = ["=== Self-Research Topics ===", ""]

        for i, t in enumerate(self.self_research_topics[-10:], 1):
            status = "✓" if t.researched else "○"
            lines.append(f"{status} {i}. {t.question}")
            lines.append(f"   Reason: {t.reason[:60]}...")
            lines.append("")

        return "\n".join(lines)

    # ── Custom Tool Creation ───────────────────────────────────────────────────

    def _find_tool(self, tool_id: str) -> Optional[ProposedTool]:
        """Find a ProposedTool by id."""
        for t in self.proposed_tools:
            if t.id == tool_id:
                return t
        return None

    async def generate_tool_proposal(
        self, description: str, client: httpx.AsyncClient
    ) -> Optional[ProposedTool]:
        """Stage 1: LLM proposes name + schema (no code). Returns ProposedTool(status='proposal')."""
        prompt = TOOL_PROPOSAL_PROMPT.format(description=description)
        try:
            response = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 600,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                }
            )
            data = json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            self.on_status(f"[SelfImprove] Tool proposal LLM error: {e}")
            return None

        name = data.get("name", "").strip().lower().replace("-", "_")
        if not name or not name.replace("_", "").isalnum():
            self.on_status(f"[SelfImprove] Tool proposal: invalid name '{name}'")
            return None

        # Reject name collisions
        taken = {t.name for t in self.proposed_tools if t.status in ("proposal", "code_ready", "installed")}
        if name in taken:
            self.on_status(f"[SelfImprove] Tool '{name}' already exists in proposals")
            return None

        tool = ProposedTool(
            id=f"tool_{int(time.time() * 1000)}",
            name=name,
            description=data.get("description", ""),
            parameters=data.get("parameters", {"type": "object", "properties": {}, "required": []}),
            prompt_addition=data.get("prompt_addition", f"- {name}: {data.get('description', '')}"),
            code="",
            status="proposal",
            source_suggestion_id="",
            created_at=time.time(),
        )
        self.proposed_tools.append(tool)
        self._save()
        self.on_status(f"[SelfImprove] Tool proposal created: {tool.name} ({tool.id})")
        return tool

    async def generate_tool_code(
        self, tool_id: str, client: httpx.AsyncClient
    ) -> Optional[ProposedTool]:
        """Stage 2: LLM generates Python code for an approved proposal."""
        tool = self._find_tool(tool_id)
        if not tool:
            self.on_status(f"[SelfImprove] Tool '{tool_id}' not found")
            return None
        if tool.status != "proposal":
            self.on_status(f"[SelfImprove] Tool '{tool.name}' is '{tool.status}', expected 'proposal'")
            return None

        prompt = TOOL_CODE_PROMPT.format(
            name=tool.name,
            description=tool.description,
            parameters_json=json.dumps(tool.parameters, indent=2),
        )
        try:
            response = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 1200,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                }
            )
            data = json.loads(response.json()["choices"][0]["message"]["content"])
        except Exception as e:
            self.on_status(f"[SelfImprove] Tool code LLM error: {e}")
            return None

        code = data.get("code", "").strip()
        if not code:
            self.on_status("[SelfImprove] LLM returned empty code")
            return None

        # --- Structural + syntax validation ---
        fn_sig = f"async def tool_{tool.name}("
        if fn_sig not in code:
            self.on_status(f"[SelfImprove] Code missing '{fn_sig}'")
            return None

        # Syntax check via compile() — catches SyntaxError before user sees the code
        try:
            compile(code, f"<tool:{tool.name}>", "exec")
        except SyntaxError as e:
            self.on_status(f"[SelfImprove] Syntax error in generated code: {e}")
            return None

        # AST check — confirm the function exists at module level, is async, has a real body
        try:
            tree = ast.parse(code)
        except Exception as e:
            self.on_status(f"[SelfImprove] AST parse failed: {e}")
            return None

        fn_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == f"tool_{tool.name}":
                fn_node = node
                break

        if fn_node is None:
            self.on_status(f"[SelfImprove] AST: async function 'tool_{tool.name}' not found")
            return None

        # Body must have more than just a pass/... placeholder
        real_stmts = [
            s for s in fn_node.body
            if not isinstance(s, (ast.Pass, ast.Expr))
            or (isinstance(s, ast.Expr) and not isinstance(s.value, ast.Constant))
        ]
        if not real_stmts and len(fn_node.body) <= 1:
            self.on_status(f"[SelfImprove] Generated function 'tool_{tool.name}' has empty body")
            return None
        # --- End validation ---

        tool.code = code
        tool.status = "code_ready"
        tool.code_generated_at = time.time()
        self._save()
        self.on_status(f"[SelfImprove] Code generated for: {tool.name}")
        return tool

    def install_tool(self, tool_id: str, registry) -> tuple[bool, str]:
        """
        Run safety check and install a code-ready tool into the registry.
        registry is a CustomToolRegistry instance from tools.py.
        Returns (success, message).
        """
        tool = self._find_tool(tool_id)
        if not tool:
            return False, f"Tool '{tool_id}' not found"
        if tool.status != "code_ready":
            return False, f"Tool '{tool.name}' is in state '{tool.status}', expected 'code_ready'"

        # Safety check
        violations = registry.check_code_safety(tool.code)
        if violations:
            tool.status = "rejected"
            tool.rejection_reason = f"Safety violations: {', '.join(violations)}"
            self._save()
            return False, f"Safety check failed: {', '.join(violations)}"

        # Name collision with built-in tools
        from tools import TOOLS as _STATIC_TOOLS
        static_names = {t["function"]["name"] for t in _STATIC_TOOLS}
        if tool.name in static_names:
            return False, f"Name '{tool.name}' conflicts with a built-in tool"

        try:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            registry.register(
                name=tool.name,
                schema=schema,
                code=tool.code,
                prompt_addition=tool.prompt_addition,
            )
        except Exception as e:
            tool.status = "rejected"
            tool.rejection_reason = f"Registration failed: {e}"
            self._save()
            return False, f"Registration failed: {e}"

        tool.status = "installed"
        tool.installed_at = time.time()
        self._save()
        return True, f"Tool '{tool.name}' installed and active"

    def remove_tool(self, tool_name: str, registry) -> tuple[bool, str]:
        """Uninstall a custom tool from the live registry."""
        tool = next(
            (t for t in self.proposed_tools if t.name == tool_name and t.status == "installed"),
            None
        )
        if not tool:
            return False, f"No installed tool named '{tool_name}'"
        registry.unregister(tool_name)
        tool.status = "rejected"
        tool.rejection_reason = "Manually removed"
        self._save()
        return True, f"Tool '{tool_name}' removed"

    def get_tools_summary(self) -> str:
        """Formatted list of custom tools by status."""
        installed = [t for t in self.proposed_tools if t.status == "installed"]
        code_ready = [t for t in self.proposed_tools if t.status == "code_ready"]
        proposals = [t for t in self.proposed_tools if t.status == "proposal"]

        lines = ["=== Custom Tools ===", ""]

        if installed:
            lines.append(f"Installed ({len(installed)}):")
            for t in installed:
                lines.append(f"  ● {t.name}: {t.description}")
        else:
            lines.append("No custom tools installed yet.")

        if code_ready:
            lines.append(f"\nCode ready — review and install ({len(code_ready)}):")
            for t in code_ready:
                lines.append(f"  ⬡ [{t.id}] {t.name}: {t.description}")
                lines.append(f"    → ?install-tool {t.id}")

        if proposals:
            lines.append(f"\nProposals — generate code ({len(proposals)}):")
            for t in proposals:
                name_str = t.name if t.name else "(pending design)"
                lines.append(f"  ○ [{t.id}] {name_str}: {t.description[:60]}")
                lines.append(f"    → ?approve-tool {t.id}")

        lines.append("")
        lines.append("Commands: ?propose-tool <desc>, ?approve-tool <id>, ?install-tool <id>, ?remove-tool <name>")
        return "\n".join(lines)

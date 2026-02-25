#!/usr/bin/env python3
"""
Autonomous Session — Self-triggering loop for 3AM.

Runs a background loop that:
1. Registers and logs in a dedicated autonomous user at startup
   (seeds the Fernet key so memories encrypt/decrypt correctly)
2. Every N minutes generates its own prompt and processes it through
   the full chat pipeline (memory, decision gate, LLM, tools, memory storage)
3. Writes results into a persistent daily conversation visible in the UI
   — log in as the autonomous user from the browser to watch in real time

Config keys (in ~/.config/3am/config.json):
    autonomous_mode               : true to enable
    autonomous_username           : username for the bot account (default: "3am_auto")
    autonomous_password           : password — REQUIRED, must be 8+ chars
    autonomous_interval_seconds   : seconds between cycles (default: 300)
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_ENDING_INSTRUCTIONS = """\
End your response with exactly one of:
NEXT: <the question you want to investigate next>
DONE: <one sentence — what you concluded and why this thread is complete>

Use DONE only when you have genuinely exhausted this topic and continuing \
would mean repeating yourself. Otherwise use NEXT."""

_COLD_START_PROMPT = """\
You are an autonomous AI assistant running on a local machine. No user is present. \
You have no memory yet.

Your task: decide what to think about first. Generate a question worth investigating, \
then answer it as thoroughly as you can, then decide what that answer makes you \
curious about next.

""" + _ENDING_INSTRUCTIONS + """

Begin."""

_MEMORY_RESUME_PROMPT = """\
You are an autonomous AI assistant in an ongoing self-directed session. \
No user is present. Your memory from previous sessions is available above.

Review what you have been exploring and continue the investigation. \
Choose the most interesting thread to develop further, explain your reasoning, \
and work through it carefully.

""" + _ENDING_INSTRUCTIONS

_CONTINUATION_PROMPT = """\
You are an autonomous AI assistant in an ongoing self-directed session. \
No user is present. Your memory from previous cycles is available above.

Continue your investigation. Your next question to explore:
{next_question}

Think through it carefully, then:

""" + _ENDING_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

_LOOP_WINDOW = 4       # How many recent NEXT questions to examine
_LOOP_THRESHOLD = 0.35  # Jaccard similarity above this → stuck

_STOPWORDS = {
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are',
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'with', 'can',
    'do', 'does', 'did', 'will', 'would', 'should', 'could', 'current',
    'currently', 'this', 'that', 'there', 'their', 'i', 'my', 'the', 'use',
}


# ---------------------------------------------------------------------------
# AutonomousSession
# ---------------------------------------------------------------------------

class AutonomousSession:
    """
    Manages the autonomous self-triggering loop.

    Parameters
    ----------
    auth_system : AuthSystem
        The shared AuthSystem singleton from server.py.
    core_factory : callable
        A function (user) -> UserLLMCore.  Passed in to avoid circular imports
        (server.py defines UserLLMCore and also imports this module).
    config : dict
        Server config dict (already loaded by the caller).
    """

    def __init__(self, auth_system, core_factory, config: dict):
        self._auth = auth_system
        self._core_factory = core_factory
        self._username: str = config.get("autonomous_username", "3am_auto")
        self._password: Optional[str] = config.get("autonomous_password")
        self._interval: int = int(config.get("autonomous_interval_seconds", 300))

        self._core = None
        self._task: Optional[asyncio.Task] = None
        self._conversation_id: Optional[str] = None
        self._next_question: Optional[str] = None
        self._cycle_count: int = 0
        self._running: bool = False
        self._recent_questions: list = []   # rolling window for loop detection
        self._loop_reflection: Optional[str] = None  # injected into next prompt if set
        self._state_cache: dict = {}        # single backing dict for autonomous_state.json

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self):
        """Register/login the autonomous user and launch the background loop."""
        if not self._password:
            print("[Autonomous] autonomous_password not set in config — skipping.")
            return

        user = self._setup_user()
        if not user:
            print("[Autonomous] Could not set up autonomous user — skipping.")
            return

        self._core = self._core_factory(user)

        # Restore state from a previous run (next_question, cycle_count)
        self._load_state()

        # Get or create a stable conversation ID for today
        self._conversation_id = self._get_or_create_conversation_id()

        self._running = True
        self._task = asyncio.create_task(self._loop())
        print(
            f"[Autonomous] Started. User: {self._username} | "
            f"Interval: {self._interval}s | "
            f"Conversation: {self._conversation_id}"
        )

    async def stop(self):
        """Cancel the background loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # User setup
    # ------------------------------------------------------------------

    def _setup_user(self):
        """
        Register the autonomous user if they don't exist, then login to
        seed the Fernet encryption key into auth._user_keys.
        Returns the User object, or None on failure.
        """
        # Try to register — silently ignore "already exists" errors
        try:
            self._auth.register(self._username, self._password)
            print(f"[Autonomous] Registered new autonomous user: {self._username}")
        except Exception:
            pass  # User already exists — that's fine

        # Login to derive and cache the encryption key
        try:
            user, _token = self._auth.login(self._username, self._password)
            print(f"[Autonomous] Logged in as: {self._username}")
            return user
        except Exception as e:
            print(f"[Autonomous] Login failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _get_or_create_conversation_id(self) -> str:
        """
        Return the conversation ID for today, persisted across restarts.
        One conversation per day keeps the UI sidebar tidy.
        Stored inside autonomous_state.json under the "conversation" key.
        """
        if not self._core:
            return str(uuid.uuid4())

        today = datetime.now().strftime("%Y-%m-%d")
        conv = self._state_cache.get("conversation", {})
        if conv.get("date") == today:
            conv_id = conv["conversation_id"]
            conv_data = self._core.get_conversation(conv_id)
            if conv_data:
                self._core.conversations[conv_id] = conv_data.get("messages", [])
            else:
                self._core.conversations[conv_id] = []
            return conv_id

        # New day or missing — create a fresh conversation and persist via _save_state
        conv_id = str(uuid.uuid4())
        self._core.conversations[conv_id] = []
        self._state_cache["conversation"] = {"date": today, "conversation_id": conv_id}
        self._save_state()
        return conv_id

    def _save_conversation_id(self, conv_id: str):
        """Persist the current conversation ID (called if chat_stream assigns one)."""
        if not self._core:
            return
        today = datetime.now().strftime("%Y-%m-%d")
        self._state_cache["conversation"] = {"date": today, "conversation_id": conv_id}
        self._save_state()

    # ------------------------------------------------------------------
    # State persistence (next_question, cycle_count across restarts)
    # ------------------------------------------------------------------

    def _load_state(self):
        if not self._core:
            return
        state_file = self._core.user_data.base_path / "autonomous_state.json"
        try:
            if state_file.exists():
                state = json.loads(state_file.read_text())
                self._state_cache = state  # keep full dict for conversation sub-key
                self._next_question = state.get("next_question")
                self._cycle_count = int(state.get("cycle_count", 0))
                self._recent_questions = state.get("recent_questions", [])
                if self._next_question:
                    print(
                        f"[Autonomous] Resuming from cycle {self._cycle_count}. "
                        f"Next question: {self._next_question[:80]}"
                    )
        except Exception:
            pass

    def _save_state(self):
        if not self._core:
            return
        state_file = self._core.user_data.base_path / "autonomous_state.json"
        try:
            self._state_cache.update({
                "next_question": self._next_question,
                "cycle_count": self._cycle_count,
                "last_cycle": time.time(),
                "recent_questions": self._recent_questions[-10:],
            })
            state_file.write_text(json.dumps(self._state_cache))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _has_memory(self) -> bool:
        """True if the autonomous user has any stored memories."""
        if not self._core:
            return False
        try:
            return len(self._core.memory.messages) > 0
        except Exception:
            return False

    def _build_prompt(self) -> str:
        # Consume any pending self-reflection note — injected once then cleared
        reflection_block = ""
        if self._loop_reflection:
            reflection_block = f"\n[NOTE TO SELF]\n{self._loop_reflection}\n[/NOTE TO SELF]\n"
            self._loop_reflection = None

        if self._next_question:
            base = _CONTINUATION_PROMPT.format(next_question=self._next_question)
            return base + reflection_block if reflection_block else base
        if self._has_memory():
            return _MEMORY_RESUME_PROMPT + reflection_block
        return _COLD_START_PROMPT

    # ------------------------------------------------------------------
    # NEXT: extraction
    # ------------------------------------------------------------------

    def _extract_next(self, response: str) -> Optional[str]:
        """
        Find the last line starting with 'NEXT:' in the response and return
        the question text after it.  Case-insensitive.
        """
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped.upper().startswith("NEXT:"):
                question = stripped[5:].strip()
                if question:
                    return question
        return None

    def _extract_done(self, response: str) -> Optional[str]:
        """
        Find the last line starting with 'DONE:' in the response.
        Returns the conclusion text, or bare 'DONE' if no text follows.
        """
        for line in reversed(response.splitlines()):
            stripped = line.strip()
            if stripped.upper().startswith("DONE"):
                conclusion = stripped[4:].lstrip(":").strip()
                return conclusion or "Thread complete."
        return None

    # ------------------------------------------------------------------
    # Cycle execution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Loop detection
    # ------------------------------------------------------------------

    def _question_tokens(self, question: str) -> set:
        """Meaningful words from a question, stopwords removed."""
        import re
        words = re.findall(r'\b[a-z]{3,}\b', question.lower())
        return {w for w in words if w not in _STOPWORDS}

    def _jaccard(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _is_looping(self) -> bool:
        """True if the last _LOOP_WINDOW questions are too similar to each other."""
        if len(self._recent_questions) < _LOOP_WINDOW:
            return False
        recent = self._recent_questions[-_LOOP_WINDOW:]
        token_sets = [self._question_tokens(q) for q in recent]
        similarities = [
            self._jaccard(token_sets[i], token_sets[i + 1])
            for i in range(len(token_sets) - 1)
        ]
        return (sum(similarities) / len(similarities)) >= _LOOP_THRESHOLD

    async def _self_reflect_on_loop(self) -> Optional[str]:
        """
        Ask the model to look at its own recent questions, spot the pattern,
        and write a note to itself about what to do differently.
        """
        recent = self._recent_questions[-_LOOP_WINDOW:]
        questions_text = "\n".join(f"- {q}" for q in recent)
        try:
            payload = {
                "model": self._core.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"You are an autonomous AI assistant reviewing your own "
                            f"recent investigation questions:\n\n{questions_text}\n\n"
                            "These questions appear to be circling the same topic "
                            "without making real progress. In 2-3 sentences: identify "
                            "exactly what pattern you have fallen into and what you "
                            "should do differently next. Be specific and honest. "
                            "Write this as a note to yourself."
                        ),
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.7,
                "stream": False,
            }
            resp = await self._core.client.post(
                f"{self._core.llm_url}/v1/chat/completions",
                json=payload,
            )
            reflection = resp.json()["choices"][0]["message"]["content"].strip()
            if reflection:
                print(f"[Autonomous] Loop detected — reflection: {reflection[:120]}")
                return reflection
        except Exception as e:
            print(f"[Autonomous] Self-reflection error: {e}")
        return None

    async def _fallback_next_question(self, response: str) -> Optional[str]:
        """
        If the model forgot to include a NEXT: line, make a small direct LLM
        call to generate one from the response rather than falling back to
        cold start.
        """
        try:
            import httpx as _httpx
            payload = {
                "model": self._core.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"You just wrote the following as part of an autonomous "
                            f"self-directed investigation:\n\n{response[-1000:]}\n\n"
                            "What single question should you investigate next? "
                            "Reply with only the question, nothing else."
                        ),
                    }
                ],
                "max_tokens": 60,
                "temperature": 0.7,
                "stream": False,
            }
            resp = await self._core.client.post(
                f"{self._core.llm_url}/v1/chat/completions",
                json=payload,
            )
            data = resp.json()
            question = data["choices"][0]["message"]["content"].strip()
            if question:
                print(f"[Autonomous] Fallback NEXT generated: {question[:80]}")
                return question
        except Exception as e:
            print(f"[Autonomous] Fallback NEXT error: {e}")
        return None

    async def _run_cycle(self):
        """Drive one full autonomous cycle through the existing chat pipeline."""
        self._cycle_count += 1
        prompt = self._build_prompt()

        print(f"[Autonomous] Cycle {self._cycle_count} starting...")

        full_response = ""
        try:
            async for event_str in self._core.chat_stream(
                prompt, conversation_id=self._conversation_id
            ):
                if not event_str.startswith("data: "):
                    continue
                raw = event_str[6:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    evt = json.loads(raw)
                    evt_type = evt.get("type", "")

                    if evt_type == "token":
                        full_response += evt.get("content", "")

                    elif evt_type == "conversation_id":
                        # chat_stream echoes back (or assigns) the conversation ID
                        assigned_id = evt.get("id")
                        if assigned_id and assigned_id != self._conversation_id:
                            self._conversation_id = assigned_id
                            self._save_conversation_id(assigned_id)

                    elif evt_type == "error":
                        print(
                            f"[Autonomous] Cycle {self._cycle_count} LLM error: "
                            f"{evt.get('message')}"
                        )

                except json.JSONDecodeError:
                    continue

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Autonomous] Cycle {self._cycle_count} exception: {e}")
            return

        # Extract the next question — or detect a voluntary DONE exit
        # NEXT takes priority: only treat as DONE if no NEXT: line is present
        self._next_question = self._extract_next(full_response)
        if not self._next_question:
            done_conclusion = self._extract_done(full_response)
            if done_conclusion:
                print(f"[Autonomous] Cycle {self._cycle_count} — DONE: {done_conclusion[:100]}")
                self._next_question = None
                self._save_state()
                return

        # Fallback: if the model forgot to include NEXT:, ask it directly
        if not self._next_question and full_response:
            self._next_question = await self._fallback_next_question(full_response)

        # Track question history and check for loops
        if self._next_question:
            self._recent_questions.append(self._next_question)
            if len(self._recent_questions) > 10:
                self._recent_questions = self._recent_questions[-10:]
            if self._is_looping():
                self._loop_reflection = await self._self_reflect_on_loop()

        # Check for topic saturation via memory dedup rate (only if loop reflection not already set)
        if not self._loop_reflection and self._core:
            sat = self._core.memory.saturation_rate()
            if sat >= 0.7:
                self._loop_reflection = (
                    f"The memory system has rejected {sat:.0%} of the facts you "
                    "generated recently as already known — this topic is thoroughly mined. "
                    "You have extracted everything worth knowing here. "
                    "Pivot to a completely different subject."
                )
                print(f"[Autonomous] Topic saturated ({sat:.0%} dedup rate) — injecting pivot signal.")

        next_preview = (
            self._next_question[:80] if self._next_question else "(none — will cold start next cycle)"
        )
        tokens = len(full_response.split())
        print(
            f"[Autonomous] Cycle {self._cycle_count} done. "
            f"~{tokens} words. Next: {next_preview}"
        )

        self._save_state()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self):
        """Background task: run cycles at the configured interval."""
        # Wait for llama-server to finish loading before the first cycle
        await asyncio.sleep(60)

        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Autonomous] Unexpected loop error: {e}")

            if not self._running:
                break

            print(f"[Autonomous] Next cycle in {self._interval}s.")
            await asyncio.sleep(self._interval)

        print("[Autonomous] Loop stopped.")

#!/usr/bin/env python3
"""
Research System - Proactive learning during idle time.

The LLM can research topics the user is interested in and build up
a knowledge base of interesting facts to share in future conversations.

Rate limited to respect Gemini free tier (5 research requests per day by default).
Must be explicitly enabled via config or command.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Callable, Any
import httpx


DATA_DIR = Path.home() / ".local/share/llm-unified"
RESEARCH_FILE = DATA_DIR / "research.json"
RESEARCH_CONFIG_FILE = DATA_DIR / "research_config.json"

TOPIC_EXTRACTION_PROMPT = """Based on these memory clusters about the user, identify 1-3 topics they seem genuinely interested in that would benefit from deeper research.

Memory clusters:
{clusters}

For each topic, explain:
1. Why they seem interested (based on conversation patterns)
2. What specific aspect would be worth researching
3. A PRECISE search query to learn more

SEARCH QUERY GUIDELINES:
- Be SPECIFIC and unambiguous - avoid words with multiple meanings
- Include context words to narrow results (e.g., "Python programming" not just "Python")
- Use quotes for exact phrases when needed
- Add domain-specific terms to filter results (e.g., "machine learning neural networks" not just "networks")
- Bad: "current weather" (ambiguous - electrical current?)
- Good: "weather forecast accuracy prediction methods"
- Bad: "memory" (RAM? human memory? programming?)
- Good: "LLM context window memory management techniques"

Respond with JSON:
{{
  "topics": [
    {{
      "name": "descriptive topic name (be specific)",
      "reason": "why user is interested",
      "search_query": "specific unambiguous search query with context terms",
      "priority": <1-5, higher = more interested>
    }}
  ]
}}"""

RESEARCH_DIGEST_PROMPT = """You researched this topic for the user. Extract ONLY facts that are directly relevant to the topic.

Topic: {topic}
Search results:
{results}

CRITICAL RULES:
1. ONLY extract facts that are DIRECTLY about "{topic}" - ignore unrelated content
2. If search results seem off-topic or contaminated with unrelated info, return fewer insights or none
3. Be skeptical - if a "fact" seems unrelated to {topic}, skip it
4. Quality over quantity - 1-2 highly relevant insights beats 4 tangential ones

Create a digest of 1-4 key insights. Focus on:
- Things the user probably doesn't know about THIS SPECIFIC TOPIC
- Practical/useful information directly related to {topic}
- Surprising or counterintuitive facts about {topic}
- Recent developments in this area

For confidence scoring:
- 0.9-1.0: Directly about the topic, from reliable source, verifiable
- 0.7-0.8: Related to topic, reasonable confidence
- 0.5-0.6: Tangentially related, uncertain
- Below 0.5: Skip this insight entirely

Respond with JSON:
{{
  "insights": [
    {{
      "fact": "the interesting fact",
      "why_interesting": "why user would care",
      "confidence": <0.5-1.0>,
      "relevance_check": "brief explanation of how this relates to {topic}"
    }}
  ],
  "summary": "one sentence summary of what you learned",
  "search_quality": "good/mixed/poor - were results relevant to the topic?"
}}"""


@dataclass
class ResearchConfig:
    """Configuration for the research system."""
    enabled: bool = False
    daily_limit: int = 5
    min_cluster_priority: int = 3  # Only research topics from high-priority clusters
    
    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "daily_limit": self.daily_limit,
            "min_cluster_priority": self.min_cluster_priority,
        }
    
    @classmethod
    def load(cls) -> "ResearchConfig":
        try:
            if RESEARCH_CONFIG_FILE.exists():
                with open(RESEARCH_CONFIG_FILE) as f:
                    data = json.load(f)
                return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
        except Exception:
            pass
        return cls()
    
    def save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESEARCH_CONFIG_FILE, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception:
            pass


@dataclass
class ResearchTopic:
    """A topic queued for research."""
    name: str
    search_query: str
    reason: str
    priority: int
    added_at: float
    researched: bool = False
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "search_query": self.search_query,
            "reason": self.reason,
            "priority": self.priority,
            "added_at": self.added_at,
            "researched": self.researched,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ResearchTopic":
        return cls(**data)


@dataclass
class Insight:
    """A researched insight to share with the user."""
    topic: str
    fact: str
    why_interesting: str
    confidence: float
    researched_at: float
    shared: bool = False
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "fact": self.fact,
            "why_interesting": self.why_interesting,
            "confidence": self.confidence,
            "researched_at": self.researched_at,
            "shared": self.shared,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Insight":
        return cls(**data)


@dataclass 
class DailyUsage:
    """Track daily research usage for rate limiting."""
    date: str
    requests_used: int = 0
    
    def to_dict(self) -> dict:
        return {"date": self.date, "requests_used": self.requests_used}
    
    @classmethod
    def from_dict(cls, data: dict) -> "DailyUsage":
        return cls(**data)


class ResearchSystem:
    """
    Proactive learning system that researches user interests during idle time.
    
    - Identifies topics from memory clusters
    - Queues topics for research
    - Uses web search to learn more
    - Stores insights to share later
    - Respects rate limits (default: 5/day)
    """
    
    def __init__(
        self,
        llm_url: str = "http://localhost:8080",
        llm_model_id: str = "qwen3-14b",
        web_search_fn: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None
    ):
        self.llm_url = llm_url
        self.llm_model_id = llm_model_id
        self.web_search_fn = web_search_fn
        self.on_status = on_status or (lambda x: None)
        
        self.config = ResearchConfig.load()
        self.topics: list[ResearchTopic] = []
        self.insights: list[Insight] = []
        self.usage = DailyUsage(date=str(date.today()))
        
        self._load()
    
    def _load(self):
        try:
            if RESEARCH_FILE.exists():
                with open(RESEARCH_FILE) as f:
                    data = json.load(f)
                
                self.topics = [ResearchTopic.from_dict(t) for t in data.get("topics", [])]
                self.insights = [Insight.from_dict(i) for i in data.get("insights", [])]
                
                usage_data = data.get("usage", {})
                if usage_data.get("date") == str(date.today()):
                    self.usage = DailyUsage.from_dict(usage_data)
                else:
                    self.usage = DailyUsage(date=str(date.today()))
                    
        except Exception as e:
            self.on_status(f"[Research] Load error: {e}")
    
    def _save(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            data = {
                "topics": [t.to_dict() for t in self.topics[-50:]],  # Keep last 50 topics
                "insights": [i.to_dict() for i in self.insights[-100:]],  # Keep last 100 insights
                "usage": self.usage.to_dict(),
                "last_update": time.time(),
            }
            
            with open(RESEARCH_FILE, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.on_status(f"[Research] Save error: {e}")
    
    def is_enabled(self) -> bool:
        """Check if research mode is enabled."""
        return self.config.enabled
    
    def enable(self):
        """Enable research mode."""
        self.config.enabled = True
        self.config.save()
        self.on_status("[Research] Enabled - will research during idle time")
    
    def disable(self):
        """Disable research mode."""
        self.config.enabled = False
        self.config.save()
        self.on_status("[Research] Disabled")
    
    def get_remaining_quota(self) -> int:
        """Get remaining research requests for today."""
        if self.usage.date != str(date.today()):
            self.usage = DailyUsage(date=str(date.today()))
        return max(0, self.config.daily_limit - self.usage.requests_used)
    
    def _use_quota(self) -> bool:
        """Use one quota slot. Returns False if no quota left."""
        if self.usage.date != str(date.today()):
            self.usage = DailyUsage(date=str(date.today()))
        
        if self.usage.requests_used >= self.config.daily_limit:
            return False
        
        self.usage.requests_used += 1
        self._save()
        return True
    
    async def _llm_request(self, prompt: str, client: httpx.AsyncClient) -> Optional[dict]:
        """Make a request to the LLM."""
        try:
            response = await client.post(
                f"{self.llm_url}/v1/chat/completions",
                json={
                    "model": self.llm_model_id,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                }
            )
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
            
        except Exception as e:
            self.on_status(f"[Research] LLM error: {e}")
            return None
    
    async def identify_topics(self, memory_clusters: list[dict], client: httpx.AsyncClient) -> list[ResearchTopic]:
        """Identify interesting topics from memory clusters."""
        if not memory_clusters:
            return []
        
        # Filter to high-priority clusters
        relevant = [c for c in memory_clusters if c.get("priority", 0) >= self.config.min_cluster_priority]
        
        if not relevant:
            return []
        
        clusters_text = "\n".join([
            f"- {c.get('theme', 'Unknown')} (priority: {c.get('priority', 0)}, messages: {c.get('message_count', 0)})"
            for c in relevant[:10]
        ])
        
        prompt = TOPIC_EXTRACTION_PROMPT.format(clusters=clusters_text)
        result = await self._llm_request(prompt, client)
        
        if not result or "topics" not in result:
            return []
        
        new_topics = []
        existing_names = {t.name.lower() for t in self.topics}
        
        for topic_data in result["topics"]:
            name = topic_data.get("name", "").strip()
            if name.lower() in existing_names:
                continue
            
            topic = ResearchTopic(
                name=name,
                search_query=topic_data.get("search_query", name),
                reason=topic_data.get("reason", ""),
                priority=topic_data.get("priority", 3),
                added_at=time.time(),
            )
            new_topics.append(topic)
            self.topics.append(topic)
        
        if new_topics:
            self._save()
            self.on_status(f"[Research] Identified {len(new_topics)} new topics to research")
        
        return new_topics
    
    async def research_topic(self, topic: ResearchTopic, client: httpx.AsyncClient, memory_system=None) -> list[Insight]:
        """Research a single topic using web search."""
        if not self.web_search_fn:
            self.on_status("[Research] No web search function configured")
            return []
        
        if not self._use_quota():
            self.on_status(f"[Research] Daily quota exhausted ({self.config.daily_limit}/day)")
            return []
        
        self.on_status(f"[Research] Researching: {topic.name}")
        
        # Do web search
        try:
            search_results = await self.web_search_fn(topic.search_query)
        except Exception as e:
            self.on_status(f"[Research] Search error: {e}")
            return []
        
        if not search_results or search_results.startswith("[Error"):
            self.on_status(f"[Research] Search failed: {search_results[:100]}")
            return []
        
        # Digest the results
        prompt = RESEARCH_DIGEST_PROMPT.format(
            topic=topic.name,
            results=search_results[:4000]
        )
        
        result = await self._llm_request(prompt, client)
        
        if not result or "insights" not in result:
            return []
        
        new_insights = []
        search_quality = result.get("search_quality", "unknown")
        
        for insight_data in result["insights"]:
            confidence = insight_data.get("confidence", 0.5)
            
            # Filter out low-confidence insights
            if confidence < 0.6:
                self.on_status(f"[Research] Skipping low-confidence insight ({confidence:.1f})")
                continue
            
            fact = insight_data.get("fact", "").strip()
            if not fact or len(fact) < 20:
                continue
            
            insight = Insight(
                topic=topic.name,
                fact=fact,
                why_interesting=insight_data.get("why_interesting", ""),
                confidence=confidence,
                researched_at=time.time(),
            )
            new_insights.append(insight)
            self.insights.append(insight)
            
            # Store finding into memory for future retrieval
            if memory_system:
                await memory_system.add_research_finding(
                    topic=topic.name,
                    fact=fact,
                    confidence=confidence,
                    http_client=client,
                    on_status=self.on_status,
                )
        
        if search_quality == "poor":
            self.on_status(f"[Research] Warning: Search quality was poor for '{topic.name}'")
        
        topic.researched = True
        self._save()
        
        self.on_status(f"[Research] Found {len(new_insights)} insights about {topic.name}")
        
        return new_insights
    
    async def run_research_cycle(self, memory_clusters: list[dict], client: httpx.AsyncClient, memory_system=None) -> dict:
        """Run one research cycle. Called during introspection."""
        results = {
            "topics_identified": 0,
            "topics_researched": 0,
            "insights_found": 0,
            "quota_remaining": self.get_remaining_quota(),
        }
        
        if not self.config.enabled:
            return results
        
        if self.get_remaining_quota() <= 0:
            self.on_status("[Research] Daily quota exhausted, skipping")
            return results
        
        # Identify new topics if we don't have unresearched ones
        unresearched = [t for t in self.topics if not t.researched]
        
        if not unresearched:
            new_topics = await self.identify_topics(memory_clusters, client)
            results["topics_identified"] = len(new_topics)
            unresearched = new_topics
        
        # Research the highest priority unresearched topic
        if unresearched:
            unresearched.sort(key=lambda t: t.priority, reverse=True)
            topic = unresearched[0]
            
            insights = await self.research_topic(topic, client, memory_system=memory_system)
            results["topics_researched"] = 1
            results["insights_found"] = len(insights)
        
        results["quota_remaining"] = self.get_remaining_quota()
        
        return results
    
    def get_unshared_insights(self, topic: Optional[str] = None, limit: int = 3) -> list[Insight]:
        """Get insights that haven't been shared with the user yet."""
        unshared = [i for i in self.insights if not i.shared]
        
        if topic:
            unshared = [i for i in unshared if topic.lower() in i.topic.lower()]
        
        # Sort by confidence and recency
        unshared.sort(key=lambda i: (i.confidence, i.researched_at), reverse=True)
        
        return unshared[:limit]
    
    def mark_insight_shared(self, insight: Insight):
        """Mark an insight as shared."""
        insight.shared = True
        self._save()
    
    def get_relevant_insight(self, query: str) -> Optional[Insight]:
        """Get a relevant unshared insight for a query."""
        query_lower = query.lower()
        
        for insight in self.insights:
            if insight.shared:
                continue
            
            # Simple keyword matching
            topic_words = insight.topic.lower().split()
            if any(word in query_lower for word in topic_words if len(word) > 3):
                return insight
        
        return None
    
    def add_manual_topic(self, topic_name: str, search_query: Optional[str] = None):
        """Manually add a topic to research."""
        topic = ResearchTopic(
            name=topic_name,
            search_query=search_query or topic_name,
            reason="Manually requested by user",
            priority=5,
            added_at=time.time(),
        )
        self.topics.append(topic)
        self._save()
        self.on_status(f"[Research] Queued topic: {topic_name}")
    
    def delete_topic(self, idx: int) -> bool:
        """Remove topic at list index and persist."""
        if 0 <= idx < len(self.topics):
            self.topics.pop(idx)
            self._save()
            return True
        return False

    def delete_insight(self, idx: int) -> bool:
        """Remove insight at list index and persist."""
        if 0 <= idx < len(self.insights):
            self.insights.pop(idx)
            self._save()
            return True
        return False

    def get_stats(self) -> dict:
        """Get research system statistics."""
        return {
            "enabled": self.config.enabled,
            "daily_limit": self.config.daily_limit,
            "quota_used_today": self.usage.requests_used,
            "quota_remaining": self.get_remaining_quota(),
            "total_topics": len(self.topics),
            "researched_topics": len([t for t in self.topics if t.researched]),
            "total_insights": len(self.insights),
            "unshared_insights": len([i for i in self.insights if not i.shared]),
        }
    
    def get_findings_summary(self) -> str:
        """Get a summary of research findings."""
        if not self.insights:
            return "No research findings yet."
        
        lines = [
            "=== Research Findings ===",
            f"Total insights: {len(self.insights)}",
            f"Unshared: {len([i for i in self.insights if not i.shared])}",
            "",
        ]
        
        # Group by topic
        by_topic = {}
        for insight in self.insights:
            if insight.topic not in by_topic:
                by_topic[insight.topic] = []
            by_topic[insight.topic].append(insight)
        
        for topic, insights in list(by_topic.items())[:5]:
            lines.append(f"📚 {topic}")
            for insight in insights[:2]:
                shared_mark = "✓" if insight.shared else "○"
                lines.append(f"  {shared_mark} {insight.fact[:80]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_all_findings(self) -> str:
        """Get complete list of all research findings for export."""
        if not self.insights:
            return "No research findings yet."
        
        from datetime import datetime
        
        lines = [
            "=" * 60,
            "RESEARCH FINDINGS - COMPLETE EXPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total insights: {len(self.insights)}",
            f"Shared: {len([i for i in self.insights if i.shared])}",
            f"Unshared: {len([i for i in self.insights if not i.shared])}",
            "=" * 60,
            "",
        ]
        
        # Group by topic
        by_topic = {}
        for insight in self.insights:
            if insight.topic not in by_topic:
                by_topic[insight.topic] = []
            by_topic[insight.topic].append(insight)
        
        for topic, insights in by_topic.items():
            lines.append("-" * 40)
            lines.append(f"📚 {topic} ({len(insights)} insights)")
            lines.append("-" * 40)
            
            for i, insight in enumerate(insights, 1):
                shared_mark = "✓ SHARED" if insight.shared else "○ UNSHARED"
                research_date = datetime.fromtimestamp(insight.researched_at).strftime('%Y-%m-%d %H:%M')
                
                lines.append(f"\n[{i}] {shared_mark} - {research_date}")
                lines.append(f"Fact: {insight.fact}")
                lines.append(f"Why interesting: {insight.why_interesting}")
                lines.append(f"Confidence: {insight.confidence}")
            
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("END OF EXPORT")
        lines.append("=" * 60)
        
        return "\n".join(lines)

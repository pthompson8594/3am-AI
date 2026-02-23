#!/usr/bin/env python3
"""
Commands — Special command handling for 3AM.

Handles all ?commands that bypass the LLM:
- ?+ / ?- / ?feedback - Rate responses
- ?introspect - View introspection stats
- ?memory - View memory clusters
- ?errors - View error patterns
- ?reflect - Trigger manual reflection (incremental)
- ?recluster - Force full memory reclustering
- ?analyze - Manually trigger self-improvement analysis cycle
- ?research on/off - Toggle research mode
- ?research - View research status
- ?findings - View research insights
- ?learn <topic> - Queue topic for research
- ?consolidate on/off - Toggle memory consolidation
- ?consolidate - View consolidation status
- ?improve on/off - Toggle self-improvement
- ?improve prompts on/off - Toggle prompt changes
- ?suggestions - View improvement suggestions
- ?approve <#> - Approve suggestion #
- ?dismiss <#> - Dismiss suggestion #
- ?implemented <#> - Mark suggestion implemented
- ?prompt - View custom prompt additions
- ?selfresearch on/off - Toggle self-research
- ?selfresearch - View self-research topics
- ?tools - List custom tools
- ?propose-tool <desc> - Start Stage 1: propose new tool concept
- ?approve-tool <id> - Stage 2: generate Python code for proposal
- ?install-tool <id> - Stage 3: safety-check and install tool live
- ?remove-tool <name> - Uninstall a custom tool
- ?help - Show all commands
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable, Any

from memory import MemorySystem
from introspection import IntrospectionLoop
from research import ResearchSystem
from self_improve import SelfImproveSystem


@dataclass
class CommandResult:
    """Result of a command execution."""
    handled: bool  # Was this a command?
    response: str  # Response to show user
    is_error: bool = False


class CommandHandler:
    """Handles special ? commands."""

    def __init__(
        self,
        memory: MemorySystem,
        introspection: IntrospectionLoop,
        on_feedback: Optional[Callable[[str, int, str], None]] = None,
        tools=None,   # ToolExecutor instance (for custom_registry)
        client=None,  # httpx.AsyncClient for LLM calls (tool proposal/code gen)
    ):
        self.memory = memory
        self.introspection = introspection
        self.research = introspection.research
        self.self_improve = introspection.self_improve
        self.on_feedback = on_feedback
        self.tools = tools
        self.client = client

        # Store last exchange for feedback
        self._last_query = ""
        self._last_response = ""
    
    def set_last_exchange(self, query: str, response: str):
        """Store the last query/response for feedback."""
        self._last_query = query
        self._last_response = response
    
    def is_command(self, message: str) -> bool:
        """Check if a message is a special command."""
        return message.strip().startswith("?")
    
    async def handle(self, message: str) -> CommandResult:
        """Handle a command. Returns CommandResult."""
        message = message.strip()
        
        if not message.startswith("?"):
            return CommandResult(handled=False, response="")
        
        # Parse command and args
        parts = message[1:].split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        # Route to handler
        handlers = {
            "+": self._handle_positive_feedback,
            "-": self._handle_negative_feedback,
            "feedback": self._handle_feedback,
            "help": self._handle_help,
            "introspect": self._handle_introspect,
            "memory": self._handle_memory,
            "errors": self._handle_errors,
            "reflect": self._handle_reflect,
            "recluster": self._handle_recluster,
            "analyze": self._handle_analyze,
            "consolidate": self._handle_consolidate,
            "research": self._handle_research,
            "findings": self._handle_findings,
            "learn": self._handle_learn,
            "improve": self._handle_improve,
            "suggestions": self._handle_suggestions,
            "approve": self._handle_approve,
            "dismiss": self._handle_dismiss,
            "implemented": self._handle_implemented,
            "prompt": self._handle_prompt,
            "selfresearch": self._handle_selfresearch,
            # MK12: custom tool pipeline
            "tools": self._handle_tools,
            "propose-tool": self._handle_propose_tool,
            "approve-tool": self._handle_approve_tool,
            "install-tool": self._handle_install_tool,
            "remove-tool": self._handle_remove_tool,
        }

        ASYNC_COMMANDS = {"reflect", "recluster", "analyze", "propose-tool", "approve-tool"}

        handler = handlers.get(cmd)
        if handler:
            try:
                response = await handler(args) if cmd in ASYNC_COMMANDS else handler(args)
                return CommandResult(handled=True, response=response)
            except Exception as e:
                return CommandResult(handled=True, response=f"Error: {e}", is_error=True)
        
        # Unknown command - might be a question starting with ?
        return CommandResult(handled=False, response="")
    
    def _handle_positive_feedback(self, args: str) -> str:
        """Handle ?+ command."""
        if self.on_feedback and self._last_query:
            self.on_feedback(self._last_query, 1, args or "Good response")
            return "✓ Thanks for the positive feedback!"
        return "No previous response to rate."
    
    def _handle_negative_feedback(self, args: str) -> str:
        """Handle ?- command."""
        if self.on_feedback and self._last_query:
            self.on_feedback(self._last_query, -1, args or "Could be better")
            return "✓ Thanks for the feedback. I'll try to improve."
        return "No previous response to rate."
    
    def _handle_feedback(self, args: str) -> str:
        """Handle ?feedback <comment> command."""
        if not args:
            return "Usage: ?feedback <your comment about the last response>"
        if self.on_feedback and self._last_query:
            self.on_feedback(self._last_query, 0, args)
            return f"✓ Feedback recorded: {args}"
        return "No previous response to give feedback on."
    
    def _handle_help(self, args: str) -> str:
        """Show all available commands."""
        return """**Available Commands**

**Feedback**
- `?+` - Rate last response positively
- `?-` - Rate last response negatively  
- `?feedback <text>` - Add detailed feedback

**Memory & Introspection**
- `?memory` - View memory clusters
- `?introspect` - View introspection statistics
- `?errors` - View common error patterns
- `?reflect` - Trigger introspection (incremental cluster updates)
- `?recluster` - Force full memory reclustering
- `?analyze` - Run self-improvement analysis now (errors, feedback, suggestions)

**Memory Consolidation** (automatic cleanup)
- `?consolidate` - View consolidation status
- `?consolidate on/off` - Enable/disable hourly consolidation

**Research** (proactive learning)
- `?research` - View research status/quota
- `?research on/off` - Enable/disable research
- `?findings` - View research discoveries
- `?learn <topic>` - Queue a topic for research

**Self-Improvement**
- `?improve` - View self-improvement status
- `?improve on/off` - Enable/disable suggestions
- `?improve prompts on/off` - Allow prompt modifications
- `?suggestions` - View pending suggestions
- `?approve <#>` - Approve suggestion #
- `?dismiss <#>` - Dismiss suggestion #
- `?implemented <#>` - Mark as implemented
- `?prompt` - View custom prompt additions

**Self-Research** (meta improvement)
- `?selfresearch` - View self-research topics
- `?selfresearch on/off` - Toggle self-research mode

**Custom Tools** (LLM-created tools, MK12)
- `?tools` - List all custom tools and proposals
- `?propose-tool <desc>` - Propose a new tool concept (Stage 1)
- `?approve-tool <id>` - Generate Python code for proposal (Stage 2)
- `?install-tool <id>` - Safety-check and install tool live (Stage 3)
- `?remove-tool <name>` - Uninstall a custom tool
"""
    
    def _handle_introspect(self, args: str) -> str:
        """Show introspection statistics."""
        stats = self.introspection.get_stats()
        
        lines = [
            "**Introspection Statistics**",
            "",
            f"Total runs: {stats.get('total_runs', 0)}",
            f"Memories summarized: {stats.get('memories_summarized', 0)}",
            f"Clustering runs: {stats.get('torque_clustering_runs', 0)}",
            f"Conflicts detected: {stats.get('conflicts_detected', 0)}",
            f"Conflicts resolved: {stats.get('conflicts_resolved', 0)}",
            f"Errors logged: {stats.get('errors_logged', 0)}",
        ]
        
        if stats.get('last_run'):
            from datetime import datetime
            last_run = datetime.fromtimestamp(stats['last_run']).strftime("%Y-%m-%d %H:%M")
            lines.append(f"Last run: {last_run}")
        
        return "\n".join(lines)
    
    def _handle_memory(self, args: str) -> str:
        """Show memory clusters with torque clustering stats."""
        clusters = list(self.memory.clusters.values())
        stats = self.memory.get_stats()
        
        if not clusters:
            return "No memory clusters yet. Start chatting to build memories!"
        
        # Header with stats
        lines = [
            f"**📊 Memory Clusters** (Torque Clustering)",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Total: {stats['total_messages']} memories in {len(clusters)} clusters",
        ]
        
        # Last clustering time
        if stats.get('last_clustering', 0) > 0:
            from datetime import datetime
            last_time = datetime.fromtimestamp(stats['last_clustering']).strftime("%Y-%m-%d %H:%M")
            lines.append(f"Last clustered: {last_time}")
        
        if stats.get('needs_reclustering', False):
            lines.append("⚠️ Re-clustering pending (next introspection)")
        
        lines.append("")
        
        # Sort by torque mass (importance)
        clusters.sort(key=lambda c: c.torque_mass, reverse=True)
        
        for i, cluster in enumerate(clusters[:10], 1):
            msg_count = len(cluster.message_refs)
            priority_stars = "★" * cluster.priority + "☆" * (5 - cluster.priority)
            mass_bar = "●" * min(int(cluster.torque_mass / 5), 10)  # Visual mass indicator
            
            lines.append(f"{i}. **{cluster.theme}**")
            lines.append(f"   {priority_stars} | {msg_count} memories | mass: {cluster.torque_mass:.1f} {mass_bar}")
        
        if len(clusters) > 10:
            lines.append(f"\n... and {len(clusters) - 10} more clusters")
        
        return "\n".join(lines)
    
    def _handle_errors(self, args: str) -> str:
        """Show common error patterns."""
        common_errors = self.introspection.error_journal.get_common_errors()
        
        if not common_errors:
            return "No errors logged yet. Tools are working well!"
        
        lines = [
            "**Common Error Patterns**",
            ""
        ]
        
        for error, count in list(common_errors.items())[:10]:
            lines.append(f"- `{error}`: {count} occurrence(s)")
        
        return "\n".join(lines)
    
    async def _handle_reflect(self, args: str) -> str:
        """Trigger manual introspection cycle (includes Torque Clustering)."""
        results = await self.introspection.run_cycle()
        
        lines = ["**Introspection Cycle Complete**", ""]
        
        # MK10: Show torque clustering results first
        tc = results.get("torque_clustering")
        if tc:
            if tc.get("status") == "success":
                lines.append(f"🌀 Torque Clustering: {tc.get('new_clusters', 0)} clusters from {tc.get('memories_clustered', 0)} memories")
            elif tc.get("status") == "skipped":
                lines.append(f"🌀 Torque Clustering: skipped ({tc.get('reason', 'unknown')})")
            elif tc.get("status") == "error":
                lines.append(f"🌀 Torque Clustering: error - {tc.get('reason', 'unknown')}")
        
        if results.get("conflicts"):
            lines.append(f"⚠ Found {len(results['conflicts'])} conflict(s)")
        if results.get("themes_updated"):
            lines.append(f"📝 Updated {results['themes_updated']} theme(s)")
        if results.get("clusters_summarized"):
            lines.append(f"📋 Summarized {results['clusters_summarized']} cluster(s)")
        
        if len(lines) == 2:
            lines.append("No actions needed - memory is in good shape!")
        
        return "\n".join(lines)
    
    async def _handle_recluster(self, args: str) -> str:
        """Force full reclustering of all memories."""
        result = await self.memory.run_torque_clustering_async(self.introspection._client, mode="full")
        
        if result.get("status") == "success":
            lines = [
                "**Memory Reclustering Complete**",
                "",
                f"🌀 Created {result.get('new_clusters', 0)} clusters from {result.get('memories_clustered', 0)} memories",
            ]
            
            # Show the new clusters
            for cluster in self.memory.clusters.values():
                lines.append(f"  • {cluster.theme} ({len(cluster.message_refs)} members)")
            
            return "\n".join(lines)
        elif result.get("status") == "skipped":
            return f"Reclustering skipped: {result.get('reason', 'unknown')}"
        else:
            return f"Reclustering error: {result.get('reason', 'unknown')}"
    
    async def _handle_analyze(self, args: str) -> str:
        """Manually trigger the self-improvement analysis cycle."""
        results = await self.introspection.run_idle_cycle()

        if results.get("skipped"):
            return "Analysis skipped — memory cycle is currently running. Try again in a moment."

        if results.get("nothing_to_do"):
            lines = [
                "**Analysis Complete** — nothing to do.",
                "",
                "Make sure self-improvement is enabled (`?improve on`) to generate suggestions.",
                "Research must also be enabled (`?research on`) to run research cycles.",
            ]
            return "\n".join(lines)

        lines = ["**Self-Improvement Analysis Complete**", ""]

        if results.get("errors_analyzed"):
            lines.append("🔍 Error patterns analyzed")

        research = results.get("research", {})
        if research and not research.get("skipped"):
            topics_done = research.get("topics_researched", 0)
            insights = research.get("insights_gathered", 0)
            if topics_done or insights:
                lines.append(f"🔬 Research: {topics_done} topic(s) researched, {insights} insight(s) gathered")

        if results.get("new_suggestion"):
            lines.append(f"💡 New suggestion from error analysis: **{results['new_suggestion']}**")

        if results.get("capability_gap"):
            lines.append(f"💡 New suggestion from feedback: **{results['capability_gap']}**")

        if results.get("self_research_suggestion"):
            lines.append(f"💡 Self-research suggestion: **{results['self_research_suggestion']}**")

        if len(lines) == 2:
            lines.append("Cycle ran but produced no new suggestions.")

        lines.append("")
        lines.append("Use `?suggestions` to view pending suggestions.")
        return "\n".join(lines)

    def _handle_consolidate(self, args: str) -> str:
        """Handle consolidation commands."""
        args = args.lower().strip()
        
        if args == "on":
            self.introspection.enable()
            return "✓ Memory consolidation enabled. Will merge duplicates and clean up hourly."
        elif args == "off":
            self.introspection.disable()
            return "✓ Memory consolidation disabled."
        else:
            enabled = self.introspection.is_enabled()
            status = "**enabled**" if enabled else "disabled"
            running = "running" if self.introspection._running else "stopped"
            stats = self.introspection.get_stats()
            
            lines = [
                f"**Memory Consolidation**: {status}",
                f"Background loop: {running}",
                f"Interval: {self.introspection.config.interval_seconds // 60} minutes",
                "",
                f"Total runs: {stats.get('total_runs', 0)}",
                f"Clusters merged: {stats.get('clusters_merged', 0)}",
                f"Memories summarized: {stats.get('memories_summarized', 0)}",
                f"Conflicts detected: {stats.get('conflicts_detected', 0)}",
                "",
                "Commands: `?consolidate on`, `?consolidate off`, `?reflect`"
            ]
            
            return "\n".join(lines)
    
    def _handle_research(self, args: str) -> str:
        """Handle research commands."""
        args = args.lower().strip()
        
        if args == "on":
            self.research.enable()
            return "✓ Research mode enabled. I'll learn about your interests during idle time."
        elif args == "off":
            self.research.disable()
            return "✓ Research mode disabled."
        else:
            stats = self.research.get_stats()
            status = "**enabled**" if stats["enabled"] else "disabled"
            
            lines = [
                f"**Research Status**: {status}",
                "",
                f"Daily quota: {stats['quota_used_today']}/{stats['daily_limit']} used",
                f"Remaining today: {stats['quota_remaining']}",
                f"Topics identified: {stats['total_topics']}",
                f"Topics researched: {stats['researched_topics']}",
                f"Insights gathered: {stats['total_insights']}",
                f"Unshared insights: {stats['unshared_insights']}",
                "",
                "Commands: `?research on`, `?research off`, `?findings`, `?learn <topic>`"
            ]
            
            return "\n".join(lines)
    
    def _handle_findings(self, args: str) -> str:
        """Show research findings."""
        if args.strip().lower() == "all":
            return "__DOWNLOAD_FINDINGS__"
        return self.research.get_findings_summary()
    
    def _handle_learn(self, args: str) -> str:
        """Queue a topic for research."""
        if not args:
            return "Usage: `?learn <topic to research>`\n\nExample: `?learn Python async best practices`"
        
        self.research.add_manual_topic(args)
        return f"✓ Queued for research: **{args}**\n\nI'll research this during the next idle period."
    
    def _handle_improve(self, args: str) -> str:
        """Handle self-improvement commands."""
        args = args.lower().strip()
        
        if args == "on":
            self.self_improve.enable()
            return "✓ Self-improvement enabled. I'll analyze my behavior and suggest improvements."
        elif args == "off":
            self.self_improve.disable()
            return "✓ Self-improvement disabled."
        elif args == "prompts on":
            self.self_improve.enable_prompt_changes()
            return "✓ Prompt modifications enabled. I can now suggest changes to my own instructions."
        elif args == "prompts off":
            self.self_improve.disable_prompt_changes()
            return "✓ Prompt modifications disabled."
        else:
            stats = self.self_improve.get_stats()
            status = "**enabled**" if stats["enabled"] else "disabled"
            prompts = "allowed" if stats["allow_prompt_changes"] else "not allowed"
            
            lines = [
                f"**Self-Improvement**: {status}",
                f"Prompt changes: {prompts}",
                "",
                f"Total suggestions: {stats['total_suggestions']}",
                f"Pending: {stats['pending']}",
                f"Approved: {stats['approved']}",
                f"Implemented: {stats['implemented']}",
                f"Dismissed: {stats['dismissed']}",
                "",
                "Commands: `?improve on/off`, `?improve prompts on/off`, `?suggestions`"
            ]
            
            return "\n".join(lines)
    
    def _handle_suggestions(self, args: str) -> str:
        """Show pending suggestions."""
        return self.self_improve.get_suggestions_summary()
    
    def _handle_approve(self, args: str) -> str:
        """Approve a suggestion."""
        if not args:
            return "Usage: `?approve <#>` — e.g. `?approve 1`"
        
        if self.self_improve.approve_suggestion(args):
            return f"✓ Suggestion #{args} approved!"
        return f"Could not find suggestion #{args}"
    
    def _handle_dismiss(self, args: str) -> str:
        """Dismiss a suggestion."""
        if not args:
            return "Usage: `?dismiss <#>` — e.g. `?dismiss 1`"
        
        if self.self_improve.dismiss_suggestion(args):
            return f"✓ Suggestion #{args} dismissed."
        return f"Could not find suggestion #{args}"
    
    def _handle_implemented(self, args: str) -> str:
        """Mark a suggestion as implemented."""
        if not args:
            return "Usage: `?implemented <#>` — e.g. `?implemented 1`"
        
        if self.self_improve.mark_implemented(args):
            return f"✓ Suggestion #{args} marked as implemented."
        return f"Could not find suggestion #{args}"
    
    def _handle_prompt(self, args: str) -> str:
        """Show custom prompt additions."""
        custom_prompt = self.self_improve.get_custom_prompt()
        
        if not custom_prompt:
            return "No custom prompt additions yet.\n\nEnable self-improvement (`?improve on`) and approve suggestions to add custom behaviors."
        
        return f"**Custom Prompt Additions**\n\n```\n{custom_prompt}\n```"
    
    def _handle_selfresearch(self, args: str) -> str:
        """Handle self-research commands."""
        args = args.lower().strip()

        if args == "on":
            if not self.self_improve.is_enabled():
                return "Enable self-improvement first with `?improve on`"
            self.self_improve.enable_self_research()
            return "✓ Self-research enabled. I'll research ways to improve myself!"
        elif args == "off":
            self.self_improve.disable_self_research()
            return "✓ Self-research disabled."
        else:
            return self.self_improve.get_self_research_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # MK12: Custom Tool Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_tools(self, args: str) -> str:
        """?tools — list all custom tools and proposals."""
        return self.self_improve.get_tools_summary()

    async def _handle_propose_tool(self, args: str) -> str:
        """?propose-tool <description> — Stage 1: LLM proposes name + schema."""
        if not args:
            return "Usage: `?propose-tool <what you want the tool to do>`"
        if not self.client:
            return "[Error: LLM client not available]"

        tool = await self.self_improve.generate_tool_proposal(args, self.client)
        if not tool:
            return (
                "Could not generate a tool proposal. A tool with that name may "
                "already exist, or the name generated was invalid. Try rephrasing "
                "the description."
            )

        params_json = json.dumps(tool.parameters, indent=2)
        lines = [
            f"**Tool Proposal: {tool.name}**",
            f"ID: `{tool.id}`",
            "",
            f"**Description:** {tool.description}",
            f"**Prompt hint:** {tool.prompt_addition}",
            "",
            "**Parameters:**",
            f"```json\n{params_json}\n```",
            "",
            f"Next step: `?approve-tool {tool.id}` to generate Python code",
        ]
        return "\n".join(lines)

    async def _handle_approve_tool(self, args: str) -> str:
        """?approve-tool <id> — Stage 2: generate code for a proposal."""
        if not args:
            return "Usage: `?approve-tool <id>`  (use `?tools` to list)"
        if not self.client:
            return "[Error: LLM client not available]"

        tool = self.self_improve._find_tool(args)
        if not tool:
            return f"No tool found with id `{args}`. Use `?tools` to list."

        # Stub from ?approve on a new_tool suggestion (no name designed yet)
        if tool.name == "":
            if not tool.description:
                return (
                    "This stub has no description. Delete it and use "
                    "`?propose-tool <description>` directly."
                )
            desc = tool.description
            # Remove the nameless stub so we can create a proper proposal
            self.self_improve.proposed_tools = [
                t for t in self.self_improve.proposed_tools if t.id != tool.id
            ]
            self.self_improve._save()

            tool = await self.self_improve.generate_tool_proposal(desc, self.client)
            if not tool:
                return (
                    "Could not generate a tool concept from the suggestion. "
                    "Try `?propose-tool <description>` manually."
                )

        # If still at proposal stage, generate code now
        if tool.status == "proposal":
            result = await self.self_improve.generate_tool_code(tool.id, self.client)
        elif tool.status == "code_ready":
            result = tool  # Already has code; just show it again
        else:
            return (
                f"Tool `{tool.name}` is in state `{tool.status}`. "
                "Use `?tools` to see available actions."
            )

        if not result:
            return f"Could not generate code for `{tool.name}`. Try `?approve-tool {tool.id}` again."

        lines = [
            f"**Generated Code: {result.name}**",
            f"ID: `{result.id}`",
            "",
            "```python",
            result.code,
            "```",
            "",
            "Review the code above carefully, then:",
            f"`?install-tool {result.id}` — run safety check and install live",
        ]
        return "\n".join(lines)

    def _handle_install_tool(self, args: str) -> str:
        """?install-tool <id> — Stage 3: safety-check and install."""
        if not args:
            return "Usage: `?install-tool <id>`  (use `?tools` to list)"
        if not self.tools:
            return "[Error: ToolExecutor not available]"

        ok, msg = self.self_improve.install_tool(args, self.tools.custom_registry)
        if ok:
            return f"✓ {msg}\n\nThe tool is now available in this session and will persist across restarts."
        return f"✗ Install failed: {msg}"

    def _handle_remove_tool(self, args: str) -> str:
        """?remove-tool <name> — uninstall a custom tool."""
        if not args:
            return "Usage: `?remove-tool <tool_name>`  (use `?tools` to see installed names)"
        if not self.tools:
            return "[Error: ToolExecutor not available]"

        ok, msg = self.self_improve.remove_tool(args.strip(), self.tools.custom_registry)
        if ok:
            return f"✓ {msg}"
        return f"✗ {msg}"

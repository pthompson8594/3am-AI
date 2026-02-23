# Changelog

## 1.1.1

### ?analyze command
- New `?analyze` command manually triggers the hourly self-improvement analysis cycle on demand
- Runs error pattern analysis, proactive research, capability gap detection, and self-research (if enabled)
- Previously this cycle only ran automatically once per hour in the background idle loop
- Reports what was found: new suggestions, research insights, and analyzed error patterns

### Tool pipeline UI fixes
- Added Reject/Delete buttons to `code_ready` and `proposal` tool cards in the tools panel
- Generate Code button now shows `↺` on failure instead of silently reverting to idle state

### Tool pipeline bug fixes
- Fixed `remove_tool()` to handle proposals and code-ready tools, not just installed ones
- Fixed `generate_tool_code()` crash when LLM returns a response with no `choices` key
- Fixed syntax error ("unexpected character after line continuation character") caused by LLM double-escaping newlines in JSON code strings

## 1.1.0

### Fernet field-level encryption
- All sensitive user data at rest is now encrypted using Fernet (AES-128-CBC + HMAC-SHA256)
- Encryption key derived from login password via PBKDF2HMAC-SHA256 (100k iterations, per-user salt)
- Key lives in memory only — never written to disk, cleared on logout/server restart
- Encrypted fields: SQLite text columns (`summary`, `message`, `response`, `theme`, `user_profile`), pending conversation JSONL lines, and JSON sidefiles (`behavior_profile.json`, `research.json`, `installed_tools.json`)
- Embedding vectors stay plaintext (required for sqlite-vec similarity search)
- Graceful migration: existing plaintext data is returned as-is (re-encrypted on next write)
- Per-user `encryption_salt` added to `users.json`; existing accounts get a salt on next login

### Lite re-clustering 3× per day
- Unclustered facts are now assigned to existing clusters every 8 hours (instead of waiting until 3 AM)
- Uses the existing incremental mode (`assign_unclustered_memories`) — no LLM calls, no full rebuild
- New memories are available for context retrieval the same day they are stored
- Full nightly Torque Clustering rebuild at 3 AM is unchanged

## 1.0.0 (initial release)
- Torque Clustering memory system
- SQLite + sqlite-vec storage
- Self-created tools with three-stage approval
- Decision Gate with logprob confidence
- Feedback-driven behavior adaptation
- Proactive research system
- 3D memory visualization
- WebSocket streaming
- Multi-user support
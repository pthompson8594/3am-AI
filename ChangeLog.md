# Changelog

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
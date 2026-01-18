# Persistent Conversation Storage

This document explains how conversation persistence works in the Marketing AI system.

## Overview

The system now saves all user conversations to a **SQLite database** for persistence across restarts.

## How It Works

### Architecture

```
Browser (JavaScript)          Backend (Python)              SQLite Database
================              ================              ===============

User sends message
  session_id='user_abc123'
          |
          v
                          → API receives request
                          → RAG processes query
                          → ConversationManager
                              ├─ Saves to memory (fast)
                              └─ Saves to database (persistent)
                                      |
                                      v
                                                            conversations.db
                                                            ├─ sessions table
                                                            └─ messages table
```

### Database Schema

**sessions table:**
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    message_count INTEGER DEFAULT 0
)
```

**messages table:**
```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,              -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
```

## Storage Location

**Local Development:**
```
marketing-ai/
  conversations/
    conversations.db       ← SQLite database file
```

**Docker Container:**
```
Container: /app/conversations/conversations.db
Host:      ./conversations/conversations.db
```

The `conversations/` directory is mounted as a volume in [docker-compose.yml:13](docker-compose.yml#L13), so data persists even when containers are stopped/restarted.

## Configuration

### Enable/Disable Persistence

In [src/rag_system.py:568](src/rag_system.py#L568), the RAGSystem accepts config:

```python
config = {
    'use_persistent_storage': True,  # Enable database storage
    'db_path': 'conversations/conversations.db',  # Database location
    'max_history': 10  # Max conversation turns to keep in memory
}

rag = RAGSystem(config)
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `use_persistent_storage` | `True` | Enable/disable database storage |
| `db_path` | `conversations/conversations.db` | Database file path |
| `max_history` | `10` | Max conversation turns in memory cache |

## Features

### 1. Automatic Save on Every Message

Every message (user or assistant) is automatically saved to the database:

```python
# User asks question
conversation_manager.add_message(
    session_id='user_abc123',
    role='user',
    content='What products do you have?'
)
# ✓ Saved to memory
# ✓ Saved to database

# Assistant responds
conversation_manager.add_message(
    session_id='user_abc123',
    role='assistant',
    content='We have headphones, cables...'
)
# ✓ Saved to memory
# ✓ Saved to database
```

### 2. Load Recent Sessions on Startup

When the application starts, it automatically loads the **100 most recent sessions** from the database into memory:

```python
# Happens automatically in __init__
self._load_recent_sessions(limit=100)
```

This ensures fast access to recent conversations without hitting the database on every request.

### 3. Load Specific Session on Demand

If a user returns with an old session_id not in memory, it's loaded from the database:

```python
messages = conversation_manager.get_session_from_db('user_old_session')
# ✓ Loads from database
# ✓ Caches in memory for fast access
```

### 4. Clear Session (Memory + Database)

```python
conversation_manager.clear_session('user_abc123')
# ✓ Clears from memory
# ✓ Deletes from database
```

## Benefits

### Before (In-Memory Only)

```
❌ Server restart → All conversations lost
❌ Container restart → All conversations lost
❌ Deployment → All conversations lost
❌ No conversation history
```

### After (With Persistent Storage)

```
✅ Server restart → Conversations preserved
✅ Container restart → Conversations preserved
✅ Deployment → Conversations preserved
✅ Full conversation history available
✅ Can analyze user interactions over time
```

## Performance

### Hybrid Approach (Memory + Database)

The system uses a **hybrid architecture** for best performance:

1. **Write:** Saves to both memory (instant) and database (persistent)
2. **Read:** Reads from memory (fast) first, falls back to database if needed
3. **Startup:** Loads 100 most recent sessions into memory

```
API Request
    ↓
Check memory cache (< 1ms)
    ↓
Found? → Return instantly ✓
    ↓
Not found? → Load from database (~ 10ms)
    ↓
Cache in memory for next time
    ↓
Return result
```

## Backup & Maintenance

### Backup Database

```bash
# Copy database file
cp conversations/conversations.db conversations/backup_$(date +%Y%m%d).db

# Or use SQLite dump
sqlite3 conversations/conversations.db .dump > backup.sql
```

### Query Database Directly

```bash
# Open database
sqlite3 conversations/conversations.db

# View all sessions
SELECT * FROM sessions ORDER BY last_activity DESC LIMIT 10;

# View messages for a session
SELECT role, content, timestamp
FROM messages
WHERE session_id = 'user_abc123'
ORDER BY timestamp;

# Count total messages
SELECT COUNT(*) FROM messages;

# Most active users
SELECT session_id, COUNT(*) as msg_count
FROM messages
GROUP BY session_id
ORDER BY msg_count DESC
LIMIT 10;
```

### Clean Old Data

```bash
sqlite3 conversations/conversations.db

-- Delete sessions older than 30 days
DELETE FROM messages
WHERE session_id IN (
    SELECT session_id FROM sessions
    WHERE datetime(last_activity) < datetime('now', '-30 days')
);

DELETE FROM sessions
WHERE datetime(last_activity) < datetime('now', '-30 days');

-- Vacuum to reclaim space
VACUUM;
```

## Docker Volume Management

### View Conversation Data

```bash
# From host
ls -lh conversations/

# From container
docker exec -it marketing-ai ls -lh /app/conversations/
```

### Backup from Docker

```bash
# Copy database from container to host
docker cp marketing-ai:/app/conversations/conversations.db ./backup_conversations.db
```

### Restore Database

```bash
# Copy backup to container
docker cp backup_conversations.db marketing-ai:/app/conversations/conversations.db

# Restart container
docker compose restart
```

## Monitoring

### Database Size

```bash
# Check database size
du -h conversations/conversations.db

# Check table sizes
sqlite3 conversations/conversations.db "
SELECT
    'sessions' as table_name, COUNT(*) as rows FROM sessions
UNION ALL
SELECT
    'messages', COUNT(*) FROM messages;
"
```

### Performance Metrics

The system logs database operations:

```
INFO: Database initialized at conversations/conversations.db
INFO: Loaded 42 sessions from database
```

## Troubleshooting

### Database Locked Error

```
Error: database is locked
```

**Fix:** SQLite doesn't handle concurrent writes well. The system uses quick connections (open → write → close) to minimize locking.

### Database Corrupted

```
Error: database disk image is malformed
```

**Fix:**
```bash
# Backup first
cp conversations/conversations.db conversations/corrupted_backup.db

# Try to recover
sqlite3 conversations/conversations.db .dump > dump.sql
rm conversations/conversations.db
sqlite3 conversations/conversations.db < dump.sql
```

### Missing conversations/ Directory

```
Error: unable to open database file
```

**Fix:**
```bash
# Create directory
mkdir -p conversations

# Restart
docker compose restart
```

## Future Enhancements

Potential improvements:

1. **PostgreSQL Support:** For production with high concurrency
2. **Conversation Analytics:** Dashboard showing usage patterns
3. **Export Conversations:** Download user conversations as JSON/CSV
4. **Search Conversations:** Full-text search across all messages
5. **Conversation Branching:** Support for multiple conversation threads per user

## Related Files

- [src/rag_system.py:126-383](src/rag_system.py#L126-L383) - ConversationManager class
- [docker-compose.yml:13](docker-compose.yml#L13) - Volume mount configuration
- [Dockerfile:38](Dockerfile#L38) - Directory creation

## Summary

**What changed:**
- Added SQLite database storage for conversations
- Conversations persist across restarts
- Hybrid memory + database architecture for performance
- Automatic backup via Docker volume

**What stayed the same:**
- User experience (no changes to UI)
- Session ID generation (still in browser)
- API interface (no changes to endpoints)
- Performance (still fast with memory cache)

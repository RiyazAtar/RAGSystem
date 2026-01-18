# Session Management Guide

This document explains how user sessions work in the Marketing AI system and how conversations persist across browser sessions.

## Overview

The system uses **localStorage** (browser storage) + **SQLite database** (server storage) to maintain user sessions and conversation history.

## How It Works

### Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  First Visit:                                                    │
│    1. Check localStorage for session ID                         │
│    2. Not found → Generate new ID: 'user_abc123'                │
│    3. Save to localStorage                                       │
│                                                                   │
│  Return Visit (page refresh):                                    │
│    1. Check localStorage for session ID                         │
│    2. Found → Restore: 'user_abc123' ✓                          │
│    3. Continue same conversation!                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    (sessionId sent with each message)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER (Backend)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. Receive message with sessionId='user_abc123'                │
│  2. Check memory cache                                           │
│     ├─ Found → Use cached conversation                          │
│     └─ Not found → Load from database                           │
│  3. Process message                                              │
│  4. Save to:                                                     │
│     ├─ Memory cache (fast reads)                                │
│     └─ SQLite database (persistent)                             │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Session Lifecycle

### 1. First Time User

```
User opens app
    ↓
JavaScript checks: localStorage.getItem('marketing_ai_session_id')
    ↓
Result: null (not found)
    ↓
Generate: sessionId = 'user_k7m2x9w1q'
    ↓
Store: localStorage.setItem('marketing_ai_session_id', 'user_k7m2x9w1q')
    ↓
Display: "Session: user_k7m2x9w1q"
    ↓
User starts chatting
    ↓
Messages saved to database with session_id='user_k7m2x9w1q'
```

### 2. Returning User (Page Refresh)

```
User refreshes page
    ↓
JavaScript checks: localStorage.getItem('marketing_ai_session_id')
    ↓
Result: 'user_k7m2x9w1q' (found!)
    ↓
Restore: sessionId = 'user_k7m2x9w1q'
    ↓
Display: "Session: user_k7m2x9w1q"
    ↓
User continues chatting
    ↓
Backend loads previous messages from database
    ↓
Conversation continues seamlessly ✓
```

### 3. Same User, Different Browser

```
Chrome browser:      sessionId = 'user_abc123'
Firefox browser:     sessionId = 'user_xyz789'  ← Different!

Why? localStorage is per-browser, per-origin
```

**Solution:** User needs to manually copy session ID between browsers (or we could add login system).

## Code Implementation

### Frontend: Session Storage ([ui/index.html:459-467](ui/index.html#L459-L467))

```javascript
// Get or create session ID (persists across page refreshes)
let sessionId = localStorage.getItem('marketing_ai_session_id');
if (!sessionId) {
    // First time user - generate new session ID
    sessionId = 'user_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('marketing_ai_session_id', sessionId);
    console.log('New session created:', sessionId);
} else {
    console.log('Returning user, session restored:', sessionId);
}
```

### Backend: Database Storage ([src/rag_system.py:270-294](src/rag_system.py#L270-L294))

```python
def _save_message_to_db(self, session_id: str, role: str, content: str, timestamp: str):
    """Save a single message to database"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Insert or update session metadata
    cursor.execute("""
        INSERT INTO sessions (session_id, created_at, last_activity, message_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(session_id) DO UPDATE SET
            last_activity = ?,
            message_count = message_count + 1
    """, (session_id, timestamp, timestamp, timestamp))

    # Insert message
    cursor.execute("""
        INSERT INTO messages (session_id, role, content, timestamp)
        VALUES (?, ?, ?, ?)
    """, (session_id, role, content, timestamp))

    conn.commit()
    conn.close()
```

## User Actions

### Clear Chat (Keep Session)

**Button:** "Clear Chat" (red button)

```javascript
// Clears messages from UI and database, but KEEPS same session ID
clearChat()
  ↓
DELETE /conversation/{sessionId}  // Backend deletes from DB
  ↓
sessionId stays: 'user_k7m2x9w1q'  // Same session ID!
  ↓
Chat UI cleared but session continues
```

**Use case:** User wants to start fresh conversation but stay in same session.

### New Session (New ID)

**Button:** "New Session" (orange button)

```javascript
// Creates completely new session with new ID
newSession()
  ↓
Generate new ID: 'user_p3j8m2r5t'
  ↓
localStorage.setItem('marketing_ai_session_id', 'user_p3j8m2r5t')
  ↓
Display updated: "Session: user_p3j8m2r5t"
  ↓
Fresh start with new session ID
```

**Use case:** User wants to start completely fresh with new session ID (old conversations still in database).

## localStorage Explained

### What is localStorage?

**localStorage** is a browser feature that stores data persistently on the user's computer.

```javascript
// Store data
localStorage.setItem('key', 'value');

// Retrieve data
let value = localStorage.getItem('key');  // Returns 'value'

// Remove data
localStorage.removeItem('key');

// Data persists across:
✓ Page refreshes
✓ Browser closes/reopens
✓ Days, weeks, months (until cleared)

// Data is separate per:
- Domain (example.com vs other.com)
- Browser (Chrome vs Firefox)
- Device (laptop vs phone)
```

### Viewing localStorage in Browser

**Chrome/Firefox Developer Tools:**
1. Open browser
2. Press `F12` (or Right-click → Inspect)
3. Go to "Application" tab (Chrome) or "Storage" tab (Firefox)
4. Click "Local Storage"
5. Click your domain
6. See: `marketing_ai_session_id: user_k7m2x9w1q`

### Clearing localStorage

**Via Browser:**
- Settings → Privacy → Clear Browsing Data → Cookies and Site Data

**Via Developer Console:**
```javascript
localStorage.clear()  // Clears all data
// or
localStorage.removeItem('marketing_ai_session_id')  // Remove just session
```

## Session Persistence Comparison

### Before (Original Implementation)

```
User visits → Generate 'user_abc123'
User chats → Messages stored with 'user_abc123'
User refreshes page → Generate 'user_xyz789' ← NEW ID!
❌ Lost previous conversation
❌ Cannot continue chat
```

### After (With localStorage)

```
User visits → Check localStorage → Not found → Generate 'user_abc123'
User chats → Messages stored with 'user_abc123'
              └→ Also saved to localStorage
User refreshes page → Check localStorage → Found 'user_abc123' ✓
✓ Session restored
✓ Can continue conversation
✓ Backend loads history from database
```

## Example Scenarios

### Scenario 1: Daily User

```
Day 1:
- User opens app
- Session created: user_abc123
- Chats about headphones
- Closes browser

Day 2:
- User opens app
- localStorage finds: user_abc123 ✓
- Continues same session
- Previous conversation loaded from database
- "Tell me more about those headphones?" ← Context maintained!
```

### Scenario 2: Multiple Tabs

```
Tab 1:
- Opens app
- Session: user_abc123
- Asks about cables

Tab 2 (same browser):
- Opens app
- localStorage: user_abc123 (same!)
- SHARES session with Tab 1
- Can see conversation from Tab 1
```

### Scenario 3: Different Devices

```
Laptop (Chrome):
- Session: user_abc123
- Chats about products

Phone (Safari):
- Session: user_xyz789 ← Different device, different session
- Cannot see laptop conversation
- Would need to manually copy session ID or implement login
```

## Security Considerations

### Current Implementation

```
✓ Session IDs are random and unpredictable
✓ No sensitive data in localStorage (just session ID)
✓ Database on server is not accessible from browser
✓ No passwords or personal info stored
```

### Limitations

```
⚠ Anyone with access to browser can see session ID
⚠ No authentication (anyone can use app)
⚠ localStorage can be cleared by user
⚠ No cross-device sync without login
```

### Production Recommendations

For production use, consider:
1. **User Authentication:** Login system with JWT tokens
2. **Session Expiration:** Auto-expire old sessions after X days
3. **Device Management:** Track and manage multiple devices per user
4. **Encryption:** Encrypt sensitive data in database
5. **Rate Limiting:** Prevent abuse by limiting requests per session

## Database Schema

**sessions table:**
```sql
session_id       | created_at           | last_activity        | message_count
-----------------------------------------------------------------------------
user_abc123      | 2025-01-17 10:00:00 | 2025-01-17 10:30:00 | 24
user_xyz789      | 2025-01-17 09:00:00 | 2025-01-17 09:15:00 | 8
```

**messages table:**
```sql
id | session_id   | role      | content                    | timestamp
--------------------------------------------------------------------------
1  | user_abc123  | user      | What headphones do you..   | 2025-01-17 10:00:00
2  | user_abc123  | assistant | We have great options...   | 2025-01-17 10:00:05
3  | user_abc123  | user      | Tell me more about boAt    | 2025-01-17 10:05:00
```

## Troubleshooting

### User Says: "I lost my conversation history"

**Possible causes:**
1. User cleared browser data/cookies
2. User switched browsers
3. User switched devices
4. localStorage was cleared
5. User clicked "New Session"

**Solution:** Session ID is stored in database, but user needs the session ID to access it.

### User Says: "I see someone else's conversation"

**This should NOT happen** - each browser gets unique session ID.

**Debug:**
1. Check localStorage in browser: `localStorage.getItem('marketing_ai_session_id')`
2. Check database: `SELECT * FROM sessions WHERE session_id = 'user_xxx'`
3. Verify messages table has correct session_id foreign key

### User Says: "My session ID keeps changing"

**Possible causes:**
1. Browser in incognito/private mode (localStorage cleared on close)
2. Browser blocking localStorage (privacy settings)
3. JavaScript error preventing localStorage save

**Debug:** Open browser console and check for errors

## Future Enhancements

### 1. User Accounts (Login System)

```
Instead of random session IDs:
- User logs in with email/password
- Session tied to user account
- Works across all devices
- Can view conversation history
```

### 2. Session Sharing

```
- Generate shareable link
- Friend opens link → sees conversation (read-only)
- Useful for getting help/support
```

### 3. Session Management UI

```
- List all your sessions
- Switch between sessions
- Delete old sessions
- Export conversations
```

### 4. Cross-Device Sync

```
- Login on any device
- Access all conversations
- Real-time sync across devices
```

## Summary

**Current System:**
- ✅ Sessions persist across page refreshes (localStorage)
- ✅ Conversations saved to database (SQLite)
- ✅ Users can continue conversations after returning
- ✅ Simple, no login required
- ⚠ Limited to single browser/device
- ⚠ No cross-device sync

**How Users Continue Conversations:**
1. Open app → localStorage restores session ID
2. Send message → Backend loads history from database
3. Continue chatting → Context maintained

**Key Files:**
- [ui/index.html:459-467](ui/index.html#L459-L467) - localStorage session management
- [src/rag_system.py:270-294](src/rag_system.py#L270-L294) - Database persistence
- [PERSISTENT_STORAGE.md](PERSISTENT_STORAGE.md) - Database documentation

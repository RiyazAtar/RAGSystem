#!/bin/bash

# Marketing AI - View Conversations Script
# Query the SQLite database to view conversation history

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

DB_PATH="conversations/conversations.db"

if [ ! -f "$DB_PATH" ]; then
    echo "Error: Database not found at $DB_PATH"
    echo "Make sure the application has been run at least once."
    exit 1
fi

echo "=========================================="
echo "Marketing AI - Conversation Viewer"
echo "=========================================="
echo ""

# Main menu
while true; do
    echo "Select an option:"
    echo "  1) List all sessions"
    echo "  2) View session details"
    echo "  3) Show statistics"
    echo "  4) Search messages"
    echo "  5) Exit"
    echo ""
    read -p "Enter choice [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "Recent Sessions:"
            echo "----------------------------------------"
            sqlite3 "$DB_PATH" <<EOF
.headers on
.mode column
SELECT
    session_id,
    message_count as msgs,
    datetime(created_at) as created,
    datetime(last_activity) as last_active
FROM sessions
ORDER BY last_activity DESC
LIMIT 20;
EOF
            echo ""
            ;;
        2)
            echo ""
            read -p "Enter session_id: " session_id
            echo ""
            echo "Messages for session: $session_id"
            echo "----------------------------------------"
            sqlite3 "$DB_PATH" <<EOF
.headers on
.mode box
SELECT
    role,
    substr(content, 1, 80) as message,
    datetime(timestamp) as time
FROM messages
WHERE session_id = '$session_id'
ORDER BY timestamp;
EOF
            echo ""
            ;;
        3)
            echo ""
            echo "Database Statistics:"
            echo "----------------------------------------"
            sqlite3 "$DB_PATH" <<EOF
SELECT 'Total Sessions' as metric, COUNT(*) as value FROM sessions
UNION ALL
SELECT 'Total Messages', COUNT(*) FROM messages
UNION ALL
SELECT 'Avg Messages/Session', CAST(AVG(message_count) AS INTEGER) FROM sessions
UNION ALL
SELECT 'Most Active Session', session_id FROM sessions ORDER BY message_count DESC LIMIT 1;
EOF
            echo ""
            ;;
        4)
            echo ""
            read -p "Enter search term: " search_term
            echo ""
            echo "Search Results:"
            echo "----------------------------------------"
            sqlite3 "$DB_PATH" <<EOF
.headers on
.mode box
SELECT
    session_id,
    role,
    substr(content, 1, 100) as message_preview,
    datetime(timestamp) as time
FROM messages
WHERE content LIKE '%$search_term%'
ORDER BY timestamp DESC
LIMIT 20;
EOF
            echo ""
            ;;
        5)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1-5."
            echo ""
            ;;
    esac
done

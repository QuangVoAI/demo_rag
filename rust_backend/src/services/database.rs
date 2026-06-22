//! SQLite Database Service — Chat History Persistence.
//!
//! Tables:
//!   sessions(id, title, created_at, updated_at)
//!   messages(id, session_id, role, content, created_at)

use rusqlite::{params, Connection};
use serde::Serialize;
use std::sync::Mutex;

use tracing::info;

/// Thread-safe wrapper around SQLite connection.
pub struct Database {
    conn: Mutex<Connection>,
}

#[derive(Debug, Serialize, Clone)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatMessage {
    pub id: String,
    pub session_id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

impl Database {
    /// Mở (hoặc tạo mới) file SQLite và khởi tạo schema.
    pub fn new(db_path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(db_path)?;

        // WAL mode cho concurrent reads
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;

        // Create tables
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, created_at);",
        )?;

        info!("✅ SQLite database initialized: {}", db_path);

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // ═══════════════════════════════════════════════════════
    // Sessions
    // ═══════════════════════════════════════════════════════

    /// Tạo session mới. Title sẽ được cập nhật sau khi có câu hỏi đầu tiên.
    pub fn create_session(&self, id: &str) -> Result<Session, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title) VALUES (?1, '')",
            params![id],
        )?;

        // Return the created session
        let session = conn.query_row(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?1",
            params![id],
            |row| {
                Ok(Session {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            },
        )?;

        Ok(session)
    }

    /// Cập nhật title của session (lấy từ vài chữ đầu câu hỏi).
    pub fn update_session_title(
        &self,
        session_id: &str,
        title: &str,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE sessions SET title = ?1, updated_at = datetime('now') WHERE id = ?2",
            params![title, session_id],
        )?;
        Ok(())
    }

    /// Lấy danh sách tất cả sessions, mới nhất lên đầu.
    pub fn list_sessions(&self) -> Result<Vec<Session>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC",
        )?;

        let sessions = stmt
            .query_map([], |row| {
                Ok(Session {
                    id: row.get(0)?,
                    title: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(sessions)
    }

    /// Xoá session và tất cả messages liên quan.
    pub fn delete_session(&self, session_id: &str) -> Result<bool, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            params![session_id],
        )?;
        let affected = conn.execute("DELETE FROM sessions WHERE id = ?1", params![session_id])?;
        Ok(affected > 0)
    }

    // ═══════════════════════════════════════════════════════
    // Messages
    // ═══════════════════════════════════════════════════════

    /// Lưu một tin nhắn mới.
    pub fn save_message(
        &self,
        session_id: &str,
        role: &str,
        content: &str,
    ) -> Result<ChatMessage, rusqlite::Error> {
        let msg_id = uuid::Uuid::new_v4().to_string();
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO messages (id, session_id, role, content) VALUES (?1, ?2, ?3, ?4)",
            params![msg_id, session_id, role, content],
        )?;

        // Cập nhật updated_at của session
        conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?1",
            params![session_id],
        )?;

        let msg = conn.query_row(
            "SELECT id, session_id, role, content, created_at FROM messages WHERE id = ?1",
            params![msg_id],
            |row| {
                Ok(ChatMessage {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        )?;

        Ok(msg)
    }

    /// Lấy tất cả messages của một session.
    pub fn get_messages(&self, session_id: &str) -> Result<Vec<ChatMessage>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, role, content, created_at
             FROM messages WHERE session_id = ?1
             ORDER BY created_at ASC",
        )?;

        let messages = stmt
            .query_map(params![session_id], |row| {
                Ok(ChatMessage {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(messages)
    }

    /// Lấy N messages gần nhất, trả về theo thứ tự thời gian tăng dần.
    pub fn get_recent_messages(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Result<Vec<ChatMessage>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, session_id, role, content, created_at
             FROM (
                SELECT id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = ?1
                ORDER BY created_at DESC
                LIMIT ?2
             )
             ORDER BY created_at ASC",
        )?;

        let messages = stmt
            .query_map(params![session_id, limit as i64], |row| {
                Ok(ChatMessage {
                    id: row.get(0)?,
                    session_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    created_at: row.get(4)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(messages)
    }
}

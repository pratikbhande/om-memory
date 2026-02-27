import os
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
import aiosqlite

from om_memory.storage.base import StorageBackend
from om_memory.models import Message, Observation, Priority

class SQLiteStorage(StorageBackend):
    """
    SQLite storage using aiosqlite for async, sqlite3 for sync.
    Default backend.
    """
    
    def __init__(self, db_path: str = None):
        if not db_path:
            db_path = os.environ.get("OM_DATABASE_URL")
        
        if not db_path:
            # Default to ~/.om_memory/om_memory.db
            home = Path.home()
            om_dir = home / ".om_memory"
            om_dir.mkdir(exist_ok=True)
            db_path = str(om_dir / "om_memory.db")
            
        self.db_path = db_path
        
    # --- Lifecycle ---
        
    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            self._create_tables(conn)
            
    async def ainitialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await self._acreate_tables(db)
            
    def close(self) -> None:
        pass
        
    async def aclose(self) -> None:
        pass
        
    def _create_tables(self, conn: sqlite3.Connection):
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            token_count INTEGER,
            metadata TEXT
        )
        ''')
        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, timestamp)")
        
        cur.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            observation_date TEXT,
            referenced_date TEXT,
            relative_date TEXT,
            priority TEXT,
            content TEXT,
            source_message_ids TEXT,
            token_count INTEGER
        )
        ''')
        cur.execute("CREATE INDEX IF NOT EXISTS idx_observations_thread ON observations(thread_id)")
        conn.commit()

    async def _acreate_tables(self, db: aiosqlite.Connection):
        await db.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            token_count INTEGER,
            metadata TEXT
        )
        ''')
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, timestamp)")
        await db.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            observation_date TEXT,
            referenced_date TEXT,
            relative_date TEXT,
            priority TEXT,
            content TEXT,
            source_message_ids TEXT,
            token_count INTEGER
        )
        ''')
        await db.execute("CREATE INDEX IF NOT EXISTS idx_observations_thread ON observations(thread_id)")
        await db.commit()

    # --- Sync Methods ---
    
    def save_messages(self, messages: list[Message]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for msg in messages:
                cur.execute(
                    "INSERT INTO messages (id, thread_id, role, content, timestamp, token_count, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (msg.id, msg.thread_id, msg.role, msg.content, msg.timestamp.isoformat(), msg.token_count, json.dumps(msg.metadata))
                )
            conn.commit()
            
    def get_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        query = "SELECT id, thread_id, role, content, timestamp, token_count, metadata FROM messages WHERE thread_id = ? ORDER BY timestamp ASC"
        if limit:
            query = f"SELECT * FROM ({query} LIMIT {limit}) ORDER BY timestamp ASC"
            
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, (thread_id,))
            rows = cur.fetchall()
            
        msgs = []
        for row in rows:
            msgs.append(Message(
                id=row[0],
                thread_id=row[1],
                role=row[2],
                content=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                token_count=row[5],
                metadata=json.loads(row[6])
            ))
        return msgs

    def delete_messages(self, message_ids: list[str]) -> None:
        if not message_ids: return
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            q = f"DELETE FROM messages WHERE id IN ({','.join(['?']*len(message_ids))})"
            cur.execute(q, message_ids)
            conn.commit()

    def save_observations(self, observations: list[Observation]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for obs in observations:
                cur.execute(
                    """INSERT INTO observations 
                    (id, thread_id, observation_date, referenced_date, relative_date, priority, content, source_message_ids, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.thread_id, obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count)
                )
            conn.commit()

    def get_observations(self, thread_id: str) -> list[Observation]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM observations WHERE thread_id = ? ORDER BY observation_date ASC", (thread_id,))
            rows = cur.fetchall()
            
        return [self._row_to_obs(row) for row in rows]
        
    def _row_to_obs(self, row) -> Observation:
        return Observation(
            id=row[0],
            thread_id=row[1],
            observation_date=datetime.fromisoformat(row[2]),
            referenced_date=datetime.fromisoformat(row[3]) if row[3] else None,
            relative_date=row[4],
            priority=Priority(row[5]),
            content=row[6],
            source_message_ids=json.loads(row[7]),
            token_count=row[8]
        )

    def update_observations(self, observations: list[Observation]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            for obs in observations:
                cur.execute(
                    """UPDATE observations 
                    SET observation_date=?, referenced_date=?, relative_date=?, priority=?, content=?, source_message_ids=?, token_count=?
                    WHERE id=?""",
                    (obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count, obs.id)
                )
            conn.commit()

    def delete_observations(self, observation_ids: list[str]) -> None:
        if not observation_ids: return
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            q = f"DELETE FROM observations WHERE id IN ({','.join(['?']*len(observation_ids))})"
            cur.execute(q, observation_ids)
            conn.commit()
            
    def replace_observations(self, thread_id: str, observations: list[Observation]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM observations WHERE thread_id = ?", (thread_id,))
            for obs in observations:
                cur.execute(
                    """INSERT INTO observations 
                    (id, thread_id, observation_date, referenced_date, relative_date, priority, content, source_message_ids, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.thread_id, obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count)
                )
            conn.commit()

    # --- Async Methods ---
    
    async def asave_messages(self, messages: list[Message]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for msg in messages:
                await db.execute(
                    "INSERT INTO messages (id, thread_id, role, content, timestamp, token_count, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (msg.id, msg.thread_id, msg.role, msg.content, msg.timestamp.isoformat(), msg.token_count, json.dumps(msg.metadata))
                )
            await db.commit()
            
    async def aget_messages(self, thread_id: str, limit: int = None) -> list[Message]:
        query = "SELECT id, thread_id, role, content, timestamp, token_count, metadata FROM messages WHERE thread_id = ? ORDER BY timestamp ASC"
        if limit:
            query = f"SELECT * FROM ({query} LIMIT {limit}) ORDER BY timestamp ASC"
            
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, (thread_id,)) as cursor:
                rows = await cursor.fetchall()
                
        msgs = []
        for row in rows:
            msgs.append(Message(
                id=row[0],
                thread_id=row[1],
                role=row[2],
                content=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                token_count=row[5],
                metadata=json.loads(row[6])
            ))
        return msgs

    async def adelete_messages(self, message_ids: list[str]) -> None:
        if not message_ids: return
        async with aiosqlite.connect(self.db_path) as db:
            q = f"DELETE FROM messages WHERE id IN ({','.join(['?']*len(message_ids))})"
            await db.execute(q, message_ids)
            await db.commit()

    async def asave_observations(self, observations: list[Observation]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for obs in observations:
                await db.execute(
                    """INSERT INTO observations 
                    (id, thread_id, observation_date, referenced_date, relative_date, priority, content, source_message_ids, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.thread_id, obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count)
                )
            await db.commit()

    async def aget_observations(self, thread_id: str) -> list[Observation]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT * FROM observations WHERE thread_id = ? ORDER BY observation_date ASC", (thread_id,)) as cursor:
                rows = await cursor.fetchall()
            
        return [self._row_to_obs(row) for row in rows]

    async def aupdate_observations(self, observations: list[Observation]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for obs in observations:
                await db.execute(
                    """UPDATE observations 
                    SET observation_date=?, referenced_date=?, relative_date=?, priority=?, content=?, source_message_ids=?, token_count=?
                    WHERE id=?""",
                    (obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count, obs.id)
                )
            await db.commit()

    async def adelete_observations(self, observation_ids: list[str]) -> None:
        if not observation_ids: return
        async with aiosqlite.connect(self.db_path) as db:
            q = f"DELETE FROM observations WHERE id IN ({','.join(['?']*len(observation_ids))})"
            await db.execute(q, observation_ids)
            await db.commit()
            
    async def areplace_observations(self, thread_id: str, observations: list[Observation]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM observations WHERE thread_id = ?", (thread_id,))
            for obs in observations:
                await db.execute(
                    """INSERT INTO observations 
                    (id, thread_id, observation_date, referenced_date, relative_date, priority, content, source_message_ids, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (obs.id, obs.thread_id, obs.observation_date.isoformat(), 
                     obs.referenced_date.isoformat() if obs.referenced_date else None, 
                     obs.relative_date, obs.priority.value, obs.content, 
                     json.dumps(obs.source_message_ids), obs.token_count)
                )
            await db.commit()

"""SQLite storage for processed documents."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union


class Database:
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------ #
    @contextmanager
    def _connection(self):
        """Open a short-lived connection (safe across Streamlit's threads)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename         TEXT NOT NULL,
                    processed_at     TEXT NOT NULL,
                    pages            INTEGER,
                    document_type    TEXT,
                    confidence       REAL,
                    extracted_fields TEXT,
                    text             TEXT
                )
                """
            )

    # ------------------------------------------------------------------ #
    def save_document(self, result: Dict) -> int:
        """Insert a processed document and return its new row id."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO documents
                    (filename, processed_at, pages, document_type,
                     confidence, extracted_fields, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["filename"],
                    result["processed_at"],
                    result.get("pages"),
                    result.get("document_type"),
                    result.get("confidence"),
                    json.dumps(result.get("extracted_fields", {}), ensure_ascii=False),
                    result.get("text", ""),
                ),
            )
            return cursor.lastrowid

    def get_all_documents(self) -> List[Dict]:
        """Summary rows (newest first) for the history table; excludes full text."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, processed_at, pages, document_type,
                       confidence, extracted_fields
                FROM documents
                ORDER BY id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Full record for one document, with extracted_fields parsed to a dict."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["extracted_fields"] = json.loads(record["extracted_fields"] or "{}")
        return record

    def delete_document(self, doc_id: int) -> bool:
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0
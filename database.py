import numpy
import psycopg
from xmlchunker import Chunk
from typing import List, Tuple
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


class VectorDBStore:
    def __init__(
        self,
        url: str,
        table: str = "chunks",
        text_dim: int = 768
    ):
        self.url = url
        self.table = table
        self.dim = text_dim
        self._ensure_schema()

    def _ensure_schema(self):
        with psycopg.connect(self.url, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table} (
                        chunk_id TEXT PRIMARY KEY,
                        page INTEGER NOT NULL,
                        character_offset INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        text_embedding VECTOR({self.dim}) NOT NULL
                    );
                    """
                )
                cursor.execute(
                    f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS(
                            SELECT 1 FROM pg_class info
                            JOIN pg_namespace name on name.oid = info.relnamespace
                            WHERE info.relname = '{self.table}_embedding_index'
                        ) THEN
                            CREATE INDEX {self.table}_embedding_index
                            ON {self.table}
                            USING ivfflat (text_embedding vector_cosine_ops)
                            WITH (lists = 100);
                        END IF;
                    END$$;
                    """
                )

    def upsert_batch(
        self,
        batch_rows: List[Tuple[str, int, int, str, list]]
    ) -> None:
        if not batch_rows:
            return
        with psycopg.connect(self.url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table}
                        (chunk_id, page, character_offset, content, text_embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        page = EXCLUDED.page,
                        character_offset = EXCLUDED.character_offset,
                        content = EXCLUDED.content,
                        text_embedding = EXCLUDED.text_embedding;
                    """,
                    batch_rows
                )
            connection.commit()

    def search(
        self,
        embedded_query: numpy.ndarray,
        top_k: int = 6
    ) -> List[Tuple[Chunk, float]]:
        query_vector = embedded_query.tolist()
        with psycopg.connect(self.url, row_factory=dict_row) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT chunk_id, page, character_offset, content,
                        (text_embedding <=> %s::vector(768)) AS cosine_distance
                    FROM {self.table}
                    ORDER BY text_embedding <=> (%s)::vector(768)
                    LIMIT %s;
                    """,
                    (query_vector, query_vector, top_k)
                )
                rows = cursor.fetchall()

        out: List[Tuple[Chunk, float]] = []
        for r in rows:
            _chunk = Chunk(
                id=r["chunk_id"],
                content=r["content"],
                metadata={
                    "page": r["page"],
                    "character_offset": r["character_offset"]
                }
            )
            similarity = float(1.0 - r["cosine_distance"])
            out.append((_chunk, similarity))
        return out

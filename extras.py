# Created by AG on 31-08-2025

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
import numpy as np
from typing import Optional, Tuple

np.set_printoptions(threshold=np.inf,
                    precision=6,
                    suppress=True,
                    linewidth=180
                    )


def fetch_content_vector_by_id(
    conn_url: str,
    table: str,
    row_id: str,
    id_col: str = "chunk_id",
    text_col: str = "content",
    vector_col: str = "text_embedding",
    as_column_matrix: bool = False,
) -> Optional[Tuple[str, np.ndarray]]:

    with psycopg.connect(conn_url, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {text_col} AS content, {vector_col} AS embedding
                FROM {table}
                WHERE {id_col} = %s
                LIMIT 1;
                """,
                (row_id,)
            )
            row = cur.fetchone()

    if not row:
        return None

    content = row["content"]
    embedding = np.array(row["embedding"], dtype=float)

    if as_column_matrix:
        embedding = embedding.reshape(-1, 1)

    print(f"\nID: {row_id}")
    print(f"Text:\n{content}")
    print("Embedding shape:", embedding.shape)
    print("Embedding (full vector):")
    print(embedding)

    return content, embedding


if __name__ == "__main__":
    CONNECTION_URL = "postgres://aalbatrossguy:pgadmin%40123@localhost:5432/vector_db"
    TABLE_NAME = "chunks"
    ROW_ID = "P001 -O01200"

    fetch_content_vector_by_id(
        CONNECTION_URL,
        TABLE_NAME,
        ROW_ID,
        as_column_matrix=False
    )

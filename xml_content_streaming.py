# Created by AG on 04-08-2025

import os
import sys
import re
import click
import numpy
import psycopg
import xml.etree.ElementTree as ETree
from groq import Groq
from psycopg.rows import dict_row
from dataclasses import dataclass
from pgvector.psycopg import register_vector
from typing import List, Tuple, Dict, Iterable, Generator, Optional
from sentence_transformers import SentenceTransformer

@dataclass 
class Chunk:
    id: str
    content: str
    metadata: Dict[str, int]
    
def _generate_text_from_xml(
        xml_file_path: str
    ) -> Generator[str, None, None]:
        for _, element in ETree.iterparse(
            xml_file_path,
            events=("end", )
            ):
                if element.tag.rsplit("}", 1)[-1].lower() == "content":
                    body_text = []
                    for text in element.itertext():
                        if text and (string := text.strip()):
                            body_text.append(string)
                    if body_text:
                        yield " ".join(body_text)
                element.clear()
                
def _normalise_xml_content(string: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
    

def iterate_content_chunks(
        xml_file_path: str,
        max_characters: int = 500,
        intersections: int = 150
    ) -> Generator[Chunk, None, None]:
        block_index = 0
        for raw_content in _generate_text_from_xml(xml_file_path):
            pure_text = _normalise_xml_content(raw_content)
            if not pure_text:
                block_index += 1
                continue
            n = len(pure_text)
            if max_characters <= 0:
                max_characters = 500
                
            if intersections < 0 or intersections >= max_characters:
                intersections = max(0, max_characters//5)
                
            chunk_step = max_characters - intersections if max_characters > intersections else max_characters
            
            index = 0
            while index < n:
                end = index + max_characters
                _slice = pure_text[index : end]
                chunk_id = f"Chunk-{block_index:06d} -Offset-{index: 08d}"
                yield Chunk(
                        id=chunk_id,
                        content=_slice.strip(),
                        metadata={
                            "page": block_index,
                            "character_offset": index
                        }
                )
                if end >= n:
                    break
                index += chunk_step
            block_index += 1
            

class VectorDBStore:
    def __init__(
        self,
        url: str,
        table: str = "chunks",
        text_dim: int = 384
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
                        (text_embedding <=> %s::vector) AS cosine_distance
                    FROM {self.table}
                    ORDER BY text_embedding <=> %s::vector
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
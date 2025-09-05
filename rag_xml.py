from openai import APIConnectionError, APIStatusError, RateLimitError, APITimeoutError
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import math
import os
import click
import numpy
import time
import psycopg
import requests
from groq import Groq
from nomic import embed
from dotenv import load_dotenv
from psycopg.rows import dict_row
# from transformers import AutoTokenizer
from pgvector.psycopg import register_vector
from typing import List, Tuple, Optional, Iterable
# from sentence_transformers import SentenceTransformer
from xmlchunker import Chunk, XMLChunker

load_dotenv()


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


class Embed:
    def __init__(
        self,
        model: str = "nomic-embed-text-v1.5",
        dim: int = 768,
        api_key: Optional[str] = None,
        request_timeout: int = 60,
        max_retries: int = 5,
        backoff_base: float = 0.8,
        backoff_factor: float = 1.8,
        l2_normalize: bool = True,
        task_type: str = "search_document",
        long_text_mode: str = "truncate",
        inference_mode: str = "remote"
    ):
        self.model = model
        self.dim = dim
        self.api_key = api_key or os.getenv("NOMICAI_API_KEY")
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.l2_normalize = l2_normalize
        self.task_type = task_type
        self.long_text_mode = long_text_mode
        self.inference_mode = inference_mode

        if not self.api_key:
            raise RuntimeError("NOMIC_API_KEY not set")

        os.environ.setdefault("NOMIC_API_KEY", self.api_key)

    def encode_embed(self, texts: List[str], batch_size: int = 64) -> numpy.ndarray:
        """
        Returns float32 ndarray of shape [len(texts), self.dim].
        """
        if not texts:
            return numpy.zeros((0, self.dim), dtype=np.float32)

        out: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tries = 0
            while True:
                tries += 1
                try:
                    resp = embed.text(
                        texts=batch,
                        model=self.model,
                        task_type=self.task_type,
                        inference_mode="remote",
                        long_text_mode="truncate",
                        dimensionality=self.dim,
                    )
                    out.extend(resp["embeddings"])
                    break

                except Exception:
                    if tries >= self.max_retries:
                        raise
                    time.sleep(self._sleep_for(tries))
                    continue

        arr = numpy.asarray(out, dtype=np.float32)

        if self.l2_normalize and arr.size:
            norms = numpy.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms

        return arr

    def encode_embed_stream(
        self,
        texts_iter: Iterable[str],
        batch_size: int = 32
    ):
        batch: List[str] = []
        for t in texts_iter:
            batch.append(t)
            if len(batch) == batch_size:
                yield self.encode_embed(batch, batch_size=batch_size)
                batch.clear()
        if batch:
            yield self.encode_embed(batch, batch_size=len(batch))

    def count_tokens(self, text: str) -> int:
        # very rough token budgeter for chunk sizing
        return max(1, len(text.split()))

    def _sleep_for(self, attempt: int) -> float:
        return self.backoff_base * (self.backoff_factor ** (attempt - 1)) + (0.05 * numpy.random.rand())


class RAGRetrieverModel:
    def __init__(
        self,
        vectorStore: VectorDBStore,
        embedder: Embed
    ):
        self.vectorStore = vectorStore
        self.embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 6
    ) -> List[Tuple[Chunk, float]]:
        query = self.embedder.encode_embed([query])[0]
        print(query)
        return self.vectorStore.search(embedded_query=query, top_k=top_k)


def create_retrieve_context(
    ChunkBlock: List[Tuple[Chunk, float]],
    max_characters=1000
) -> str:
    buffer, used, total = [], set(), 0
    for character, score in ChunkBlock:
        if character.id in used:
            continue
        line = f"[Chunk_id: {character.id} page: {character.metadata.get('page')}] {
            character.content.strip()}\n"
        if total + len(line) > max_characters:
            break
        buffer.append(line)
        used.add(character.id)
        total += len(line)
    return "".join(buffer)


def answer_pass_threshold(
    ChunkBlocks,
    min_similarity=0.14,
    min_support=1,
    support_similarity=0.06
) -> bool:
    if not ChunkBlocks:
        return False
    top_similarity = ChunkBlocks[0][1]
    if top_similarity >= 0.40:
        return True
    if top_similarity < min_similarity:
        return False
    supporter = sum(1 for _, s in ChunkBlocks if s >= support_similarity)
    return supporter >= min_support


fallback = "Don't know the answer"

PROMPT = """You are a strict RAG assistant.
RULES:
- Answer ONLY using the provided CONTEXT.
- If the answer is not fully contained in CONTEXT, reply exactly:
  "I don't know from the answer."
- Keep answers concise and cite chunk IDs like [Chunk_id: id] inline for claims
"""

CONTEXT_WITH_QUESTION = """QUESTION:
{question}

CONTEXT (verbatim excerpts):
{context}
"""


def groq() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key)


def chat(
    client: Groq,
    user_prompt: str,
    model="qwen/qwen3-32b",
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {
                "role": "system", "content": PROMPT
            },
            {
                "role": "user", "content": user_prompt
            }
        ],
        reasoning_format="hidden"
    )
    return response.choices[0].message.content.strip()


def get_response(
    query: str,
    retriever: RAGRetrieverModel,
    client: Groq,
    min_similarity=0.05,
    min_support=1,
    support_similarity=0.02
) -> str:
    ChunkBlocks = retriever.search(query, top_k=6)

    if not answer_pass_threshold(
        ChunkBlocks,
        min_similarity,
        min_support,
        support_similarity
    ):
        print('could not pass answer_pass_threshold')
        return fallback

    context = create_retrieve_context(ChunkBlocks, max_characters=1000)
    response = chat(
        client,
        CONTEXT_WITH_QUESTION.format(question=query, context=context)
    )
    print(f"\n\nCONTEXT = {context}")
    print(f"UNFILTERED RESPONSE = {response}\n\n")
    if fallback in response:
        print('fallback encountered')
        return fallback
    if "[Chunk_id:" not in response:
        print('Chunk_id not found in response')
        return fallback
    return response


class XMLIngestor:
    def __init__(
        self,
        db: VectorDBStore,
        embedder: Embed,
        semantic: bool = True,
        soft_target_token: int = 140,
        max_tokens: int = 200,
        intersections: int = 5,
        similarity_floor: float = 0.20,
        batch_size_embed: int = 8,
        upsert_batch_size: int = 150,
        on_flush: Optional[callable] = None
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.is_semantic = semantic
        self.target_tokens = soft_target_token
        self.max_tokens = max_tokens
        self.intersections = intersections
        self.similarity_floor = similarity_floor
        self.upsert_batch_size = upsert_batch_size
        self.batch_size_embed = batch_size_embed
        self.on_flush = on_flush

        self._buffered_chunks: List["Chunk"] = []
        self._buffered_texts: List[str] = []
        self._total: int = 0

    def ingest(
        self,
        xml_file_path: str
    ) -> int:
        self._clear_buffers()

        iterate_chunk = self._build_iterator(xml_file_path)
        for chunk in iterate_chunk:
            self._buffered_chunks.append(chunk)
            self._buffered_texts.append(chunk.content)
            if len(self._buffered_chunks) >= self.upsert_batch_size:
                self._flush_buffers()

        self._flush_buffers()
        return self._total

    def _build_iterator(
        self,
        xml_file_path: str
    ) -> Iterable["Chunk"]:
        # if self.is_semantic:
        chunker = XMLChunker(
            embedder=self.embedder,
            soft_target_token=self.target_tokens,
            max_tokens=self.max_tokens,
            intersections=self.intersections,
            similarity_floor=self.similarity_floor,
            batch_size_embed=self.batch_size_embed
        )
        return chunker.iterate_chunks(xml_file_path)

        # else:
        # Add code for fallback to legacy regex based chunking

    # def _flush_buffers(self) -> None:
    #     if not self._buffered_chunks:
    #         return
    #
    #     embedded_chunks = self.embedder.encode_embed(
    #         text=self._buffered_texts,
    #         batch_size=self.batch_size_embed
    #     )
    #
    #     rows = [
    #         (
    #             chunk.id,
    #             int(chunk.metadata.get("page", 0)),
    #             int(chunk.metadata.get("character_offset", 0)),
    #             chunk.content,
    #             embedded_chunks[i].tolist(),
    #         )
    #         for i, chunk in enumerate(self._buffered_chunks)
    #     ]
    #
    #     self.db.upsert_batch(rows)
    #
    #     flushed = len(self._buffered_chunks)
    #     self._total += flushed
    #     self._buffered_chunks.clear()
    #     self._buffered_texts.clear()
    #
    #     if self.on_flush:
    #         self.on_flush(flushed, self._total)

    def _flush_buffers(self) -> None:
        if not self._buffered_chunks:
            return

        # stream-embed the buffered texts in small batches
        offset = 0
        for emb_batch in self.embedder.encode_embed_stream(
            self._buffered_texts,
            batch_size=self.batch_size_embed
        ):
            n = emb_batch.shape[0]
            slice_chunks = self._buffered_chunks[offset:offset + n]

            rows = [
                (
                    ch.id,
                    int(ch.metadata.get("page", 0)),
                    int(ch.metadata.get("character_offset", 0)),
                    ch.content,
                    emb_batch[i].tolist(),
                )
                for i, ch in enumerate(slice_chunks)
            ]

            # upsert immediately per embedded batch (keeps memory flat)
            self.db.upsert_batch(rows)
            offset += n

        flushed = len(self._buffered_chunks)
        self._total += flushed
        self._buffered_chunks.clear()
        self._buffered_texts.clear()

        if self.on_flush:
            self.on_flush(flushed, self._total)

    def _clear_buffers(self) -> None:
        self._buffered_chunks.clear()
        self._buffered_texts.clear()
        self._total = 0


@click.command()
@click.argument("xml_file_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("question", type=str)
@click.option("--database-url", default="postgres://aalbatrossguy:pgadmin%40123@localhost:5432/vector_db")
@click.option("--is-semantic/--no-semantic", default=True)
@click.option("--target-tokens", default=160)
@click.option("--max-tokens", default=220)
@click.option("--intersections", default=1)
@click.option("--similarity-floor", default=0.20)
@click.option("--min-similarity", default=0.14)
@click.option("--support-similarity", default=0.06)
@click.option("--min-support", default=2)
@click.option("--skip-indexing", is_flag=True, default=False)
@click.option("--upsert-batch-size", default=1000)
@click.option("--embed-batch-size", default=128)
def script(
    xml_file_path,
    question,
    database_url,
    is_semantic,
    target_tokens,
    max_tokens,
    intersections,
    similarity_floor,
    min_similarity,
    support_similarity,
    min_support,
    skip_indexing,
    upsert_batch_size,
    embed_batch_size,
):
    embedder = Embed()
    store = VectorDBStore(url=database_url, text_dim=embedder.dim)

    if not skip_indexing:
        def _progress_log(flushed, current_status):
            click.echo(f"Flush status: {flushed}, chunks flushed so far: {
                       current_status}", err=True)

        ingestor = XMLIngestor(
            db=store,
            embedder=embedder,
            semantic=is_semantic,
            soft_target_token=target_tokens,
            max_tokens=max_tokens,
            intersections=intersections,
            similarity_floor=similarity_floor,
            batch_size_embed=embed_batch_size,
            on_flush=_progress_log
        )
        total = ingestor.ingest(xml_file_path)
        click.echo(f"Ingested chunks: {total}")

    retriever = RAGRetrieverModel(vectorStore=store, embedder=embedder)
    client = groq()
    response = get_response(
        query=question,
        retriever=retriever,
        client=client,
        min_similarity=min_similarity,
        min_support=min_support,
        support_similarity=support_similarity
    )
    click.echo(f"AI said: {response}")


if __name__ == "__main__":
    script()

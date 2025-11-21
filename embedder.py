import os
import time
import numpy
from nomic import embed
from typing import List, Optional, Iterable


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

    def encode_embed(
            self,
            texts: List[str],
            batch_size: int = 64
    ) -> numpy.ndarray:
        if not texts:
            return numpy.zeros((0, self.dim), dtype=numpy.float32)

        text_embeds: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tries = 0
            while True:
                tries += 1
                try:
                    response = embed.text(
                        texts=batch,
                        model=self.model,
                        task_type=self.task_type,
                        inference_mode="remote",
                        long_text_mode="truncate",
                        dimensionality=self.dim,
                    )
                    text_embeds.extend(response["embeddings"])
                    break

                except Exception:
                    if tries >= self.max_retries:
                        raise
                    time.sleep(self._sleep_for(tries))
                    continue

        embedded_texts = numpy.asarray(text_embeds, dtype=numpy.float32)

        if self.l2_normalize and embedded_texts.size:
            normalise = numpy.linalg.norm(
                embedded_texts,
                axis=1,
                keepdims=True
            ) + 1e-12
            embedded_texts /= normalise

        return embedded_texts

    def encode_embed_stream(
        self,
        texts: Iterable[str],
        batch_size: int = 32
    ):
        batch: List[str] = []
        for text in texts:
            batch.append(text)
            if len(batch) == batch_size:
                yield self.encode_embed(batch, batch_size=batch_size)
                batch.clear()
        if batch:
            yield self.encode_embed(batch, batch_size=len(batch))

    def encode_embed_single_sentence(self, text: str):
        return self.encode_embed([text], batch_size=1)[0]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))

    def _sleep_for(self, attempt: int) -> float:
        return self.backoff_base * (self.backoff_factor ** (attempt - 1)) + (0.05 * numpy.random.rand())

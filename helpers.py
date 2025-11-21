import numpy
import hashlib
from embedder import Embed
from xmlchunker import Chunk
from typing import List, Tuple


def hash_sha256(
    path: str,
    buffer_size: int = 1024 * 1024
) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            buffer = file.read(buffer_size)
            if not buffer:
                break
            hasher.update(buffer)

    return hasher.hexdigest()


def mmr_select(
    candidates: List[Tuple[Chunk, float]],
    embedder: Embed,
    top_k: int = 8,
    _lambda: float = 0.7
) -> List[Tuple[Chunk, float]]:
    if not candidates:
        return []

    max_cap = min(len(candidates), 50)
    _pool = candidates[:max_cap]
    texts = [chunk[0].content for chunk in _pool]
    embedded_texts = embedder.encode_embed(texts, batch_size=32)
    base = numpy.asarray([chunk[1] for chunk in _pool], dtype=numpy.float32)
    if base.size:
        base = (base - base.min()) / (base.max() - base.min() + 1e-9)

    selected: List[int] = []
    remaining: List[int] = list(range(max_cap))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: float(base[i]))
        else:
            def mmr(i: int) -> float:
                sim_to_sel = 0.0
                for j in selected:
                    sim_to_sel = max(
                        sim_to_sel,
                        float(numpy.dot(embedded_texts[i], embedded_texts[j]))
                    )
                return float(_lambda * base[i] - (1.0 - _lambda) * sim_to_sel)
            best = max(remaining, key=mmr)
        selected.append(best)
        remaining.remove(best)

    return [_pool[i] for i in selected]

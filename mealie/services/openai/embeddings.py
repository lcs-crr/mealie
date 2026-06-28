"""Vector helpers for semantic food matching.

Pure functions (no OpenAI/SQLAlchemy imports) so they're cheap to unit test. Embeddings are
stored L2-normalized, which makes cosine similarity a plain dot product.
"""

import math


def normalize_vector(vector: list[float]) -> list[float]:
    """Return the L2-normalized vector. A zero vector is returned unchanged."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        return list(vector)
    return [component / norm for component in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity for already-normalized vectors (falls back to full cosine otherwise)."""
    return dot(a, b)


def best_match[T](query: list[float], candidates: list[tuple[T, list[float]]]) -> tuple[T, float] | None:
    """Return the (candidate, score) with the highest cosine similarity, or None if empty."""
    best: tuple[T, float] | None = None
    for item, vector in candidates:
        score = cosine_similarity(query, vector)
        if best is None or score > best[1]:
            best = (item, score)
    return best


def top_k_matches[T](query: list[float], candidates: list[tuple[T, list[float]]], k: int) -> list[tuple[T, float]]:
    """Return up to k (candidate, score) pairs sorted by descending cosine similarity."""
    scored = [(item, cosine_similarity(query, vector)) for item, vector in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]

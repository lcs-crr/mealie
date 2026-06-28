import math

from mealie.db.models._model_utils.vector import Vector
from mealie.services.openai.embeddings import (
    cosine_similarity,
    normalize_vector,
    top_k_matches,
)


def test_vector_roundtrip():
    vector = [0.1, -0.2, 0.3, 0.0, 1.5]
    col = Vector()
    packed = col.process_bind_param(vector, None)
    assert isinstance(packed, bytes)
    restored = col.process_result_value(packed, None)
    assert len(restored) == len(vector)
    # float32 round-trip is lossy, so compare with tolerance rather than equality
    assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(vector, restored, strict=True))


def test_vector_none():
    col = Vector()
    assert col.process_bind_param(None, None) is None
    assert col.process_result_value(None, None) is None


def test_normalize_vector_unit_length():
    normalized = normalize_vector([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(c * c for c in normalized)), 1.0, rel_tol=1e-9)
    assert math.isclose(normalized[0], 0.6) and math.isclose(normalized[1], 0.8)


def test_normalize_zero_vector():
    assert normalize_vector([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_cosine_similarity_of_normalized():
    a = normalize_vector([1.0, 1.0])
    assert math.isclose(cosine_similarity(a, a), 1.0, rel_tol=1e-9)
    b = normalize_vector([1.0, -1.0])
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)


def test_top_k_matches_orders_by_similarity():
    query = normalize_vector([1.0, 0.0])
    candidates = [
        ("orthogonal", normalize_vector([0.0, 1.0])),
        ("close", normalize_vector([0.9, 0.1])),
        ("exact", normalize_vector([1.0, 0.0])),
    ]
    ranked = top_k_matches(query, candidates, 2)
    assert [item for item, _ in ranked] == ["exact", "close"]
    assert ranked[0][1] >= ranked[1][1]

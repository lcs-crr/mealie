import struct

from sqlalchemy.types import LargeBinary, TypeDecorator


class Vector(TypeDecorator):
    """Platform-independent embedding vector type.

    Stores a ``list[float]`` as packed little-endian float32 bytes in a ``LargeBinary``
    column. Behaves identically on SQLite and PostgreSQL — vectors live in Mealie's
    existing database, with no extension (pgvector / sqlite-vec) or separate vector store.

    Similarity search is performed in Python (brute-force cosine); at Mealie's food-list
    sizes this is sub-millisecond, so no native ANN index is required.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: list[float] | None, dialect) -> bytes | None:
        if value is None:
            return None
        return struct.pack(f"<{len(value)}f", *value)

    def process_result_value(self, value: bytes | None, dialect) -> list[float] | None:
        if value is None:
            return None
        count = len(value) // 4
        return list(struct.unpack(f"<{count}f", value))

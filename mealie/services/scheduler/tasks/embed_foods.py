from mealie.core import root_logger
from mealie.core.config import get_app_settings
from mealie.db.db_setup import session_context
from mealie.repos.all_repositories import get_repositories
from mealie.schema.response.pagination import PaginationQuery
from mealie.services.openai.food_embedding_service import FoodEmbeddingService

logger = root_logger.get_logger(__name__)


def embed_missing_foods() -> None:
    """Backfill semantic embeddings for foods that are missing or stale, per group.

    Covers foods that predate the feature and any interactive embed that failed (e.g. the provider
    was temporarily unavailable). Best-effort and idempotent. No-op when no embedding model is set.
    """
    if not get_app_settings().OPENAI_EMBEDDINGS_ENABLED:
        return

    with session_context() as session:
        groups = get_repositories(session, group_id=None).groups.page_all(PaginationQuery(page=1, per_page=-1)).items

        for group in groups:
            repos = get_repositories(session, group_id=group.id)
            embedded = FoodEmbeddingService(repos).embed_missing()
            if embedded:
                logger.info(f"Embedded {embedded} food(s) for group {group.id}")

from pydantic import UUID4
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from mealie.core import root_logger
from mealie.core.config import get_app_settings
from mealie.db.models.recipe.ingredient import IngredientFoodEmbeddingModel, IngredientFoodModel
from mealie.repos.repository_factory import AllRepositories

from .openai import OpenAIService

logger = root_logger.get_logger(__name__)


class FoodEmbeddingService:
    """Computes and stores semantic embeddings for foods.

    All public methods are best-effort: any failure (no provider, endpoint down, model without an
    embeddings route) is logged and swallowed so food create/update never fails because of it.
    """

    def __init__(self, repos: AllRepositories) -> None:
        self.repos = repos
        self.session = repos.session
        self.settings = get_app_settings()
        self._openai: OpenAIService | None = None

    @property
    def enabled(self) -> bool:
        return self.settings.OPENAI_EMBEDDINGS_ENABLED

    def _openai_service(self) -> OpenAIService:
        if self._openai is None:
            self._openai = OpenAIService(self.repos)
        return self._openai

    def _is_current(self, row: IngredientFoodEmbeddingModel | None, source_text: str) -> bool:
        return bool(
            row is not None
            and row.source_text == source_text
            and row.model == self.settings.OPENAI_EMBEDDING_MODEL
            and row.dimensions == self.settings.OPENAI_EMBEDDING_DIMENSIONS
        )

    def _store(self, food: IngredientFoodModel, source_text: str, vector: list[float]) -> None:
        existing: IngredientFoodEmbeddingModel | None = food.embedding
        if existing is not None:
            existing.model = self.settings.OPENAI_EMBEDDING_MODEL
            existing.dimensions = self.settings.OPENAI_EMBEDDING_DIMENSIONS
            existing.source_text = source_text
            existing.embedding = vector
        else:
            food.embedding = IngredientFoodEmbeddingModel(
                session=self.session,
                food_id=food.id,
                model=self.settings.OPENAI_EMBEDDING_MODEL,
                dimensions=self.settings.OPENAI_EMBEDDING_DIMENSIONS,
                source_text=source_text,
                embedding=vector,
            )

    def embed_food(self, food_id: UUID4, *, force: bool = False) -> None:
        """Best-effort: (re)embed a single food by id if its embedding is missing or stale."""
        if not self.enabled:
            return
        try:
            stmt = (
                select(IngredientFoodModel)
                .options(joinedload(IngredientFoodModel.embedding))
                .filter(IngredientFoodModel.id == food_id)
            )
            food = self.session.execute(stmt).scalars().one_or_none()
            if food is None or not food.name:
                return

            if not force and self._is_current(food.embedding, food.name):
                return

            vector = self._openai_service().get_embeddings_sync([food.name])[0]
            self._store(food, food.name, vector)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.warning(f"Failed to embed food {food_id} ({e.__class__.__name__}: {e})")

    def embed_missing(self, *, batch_size: int = 50, max_foods: int = 500) -> int:
        """Best-effort backfill: embed foods with a missing or stale embedding, in batches.

        Returns the number of foods embedded. Safe to call repeatedly (idempotent once current).
        """
        if not self.enabled:
            return 0

        embedded = 0
        try:
            stmt = (
                select(IngredientFoodModel)
                .options(joinedload(IngredientFoodModel.embedding))
                .filter(IngredientFoodModel.name.isnot(None))
            )
            foods = self.session.execute(stmt).scalars().all()
            stale = [f for f in foods if f.name and not self._is_current(f.embedding, f.name)][:max_foods]

            for start in range(0, len(stale), batch_size):
                chunk = stale[start : start + batch_size]
                names = [f.name for f in chunk]
                vectors = self._openai_service().get_embeddings_sync(names)
                for food, vector in zip(chunk, vectors, strict=True):
                    self._store(food, food.name, vector)
                self.session.commit()
                embedded += len(chunk)
        except Exception as e:
            self.session.rollback()
            logger.warning(f"Food embedding backfill failed ({e.__class__.__name__}: {e})")

        return embedded

from pydantic import UUID4, BaseModel
from sqlalchemy import select

from mealie.db.models.recipe.ingredient import IngredientFoodModel
from mealie.schema.recipe.recipe_ingredient import IngredientFood

from .repository_generic import GroupRepositoryGeneric


class RepositoryFood(GroupRepositoryGeneric[IngredientFood, IngredientFoodModel]):
    def _embed_food(self, food_id: UUID4) -> None:
        """Best-effort semantic embedding for the interactive create/update path.

        Bulk paths (create_many/update_many, e.g. seeding/imports) are intentionally left to the
        backfill scheduled task so a large import doesn't fire one embedding request per food.
        """
        # Imported lazily to avoid a circular import (OpenAIService imports repositories).
        from mealie.services.openai.food_embedding_service import FoodEmbeddingService

        from .all_repositories import get_repositories

        repos = get_repositories(self.session, group_id=self.group_id, household_id=self.household_id)
        service = FoodEmbeddingService(repos)
        if service.enabled:
            service.embed_food(food_id)

    def create(self, data: IngredientFood | BaseModel | dict) -> IngredientFood:
        food = super().create(data)
        self._embed_food(food.id)
        return food

    def update(self, match_value: str | int | UUID4, new_data: dict | BaseModel) -> IngredientFood:
        food = super().update(match_value, new_data)
        self._embed_food(food.id)
        return food

    def _get_food(self, id: UUID4) -> IngredientFoodModel:
        stmt = select(self.model).filter_by(**self._filter_builder(**{"id": id}))
        return self.session.execute(stmt).scalars().one()

    def merge(self, from_food: UUID4, to_food: UUID4) -> IngredientFood | None:
        from_model = self._get_food(from_food)
        to_model = self._get_food(to_food)

        to_model.ingredients += from_model.ingredients

        try:
            self.session.delete(from_model)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

        return self.get_one(to_food)

"""add ingredient food embeddings

Revision ID: a1b2c3d4e5f6
Revises: 2187537c52b8
Create Date: 2026-06-27 09:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

import mealie.db.migration_types

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision: str | None = "2187537c52b8"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade():
    op.create_table(
        "ingredient_food_embeddings",
        sa.Column("id", mealie.db.migration_types.GUID(), nullable=False),
        sa.Column("food_id", mealie.db.migration_types.GUID(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("source_text", sa.String(), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", mealie.db.migration_types.NaiveDateTime(), nullable=True),
        sa.Column("update_at", mealie.db.migration_types.NaiveDateTime(), nullable=True),
        sa.ForeignKeyConstraint(["food_id"], ["ingredient_foods.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("food_id", name="ingredient_food_embeddings_food_id_key"),
    )
    with op.batch_alter_table("ingredient_food_embeddings", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_ingredient_food_embeddings_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_ingredient_food_embeddings_food_id"), ["food_id"], unique=True)


def downgrade():
    with op.batch_alter_table("ingredient_food_embeddings", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_ingredient_food_embeddings_food_id"))
        batch_op.drop_index(batch_op.f("ix_ingredient_food_embeddings_created_at"))
    op.drop_table("ingredient_food_embeddings")

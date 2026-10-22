"""Add the cardiovascular ordinal category columns.

``chest_pain_type`` and ``st_slope`` are the dataset's ordinal category
codes. They are optional: the cardiovascular model uses them only when both
are present, and falls back to its base feature set otherwise.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5b2d8c1a704"
down_revision: Union[str, None] = "d4a1c7e5b902"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("chest_pain_type", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("st_slope", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.drop_column("st_slope")
        batch_op.drop_column("chest_pain_type")

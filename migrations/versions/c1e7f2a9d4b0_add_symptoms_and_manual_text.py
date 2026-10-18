"""Add persisted symptom context to health records."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1e7f2a9d4b0"
down_revision: Union[str, None] = "b8f80484c6df"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("symptoms", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("manual_text", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.drop_column("manual_text")
        batch_op.drop_column("symptoms")

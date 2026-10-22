"""Add extraction, assessment and source columns to health records.

Stores the parameters extracted from an uploaded report, a normalized
assessment snapshot used by the history endpoints, and the origin of a
record ("manual" or "report").
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4a1c7e5b902"
down_revision: Union[str, None] = "c1e7f2a9d4b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source", sa.String(), nullable=False, server_default="manual"))
        batch_op.add_column(sa.Column("extracted_parameters", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("assessment", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("homa_ir", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("health_records", schema=None) as batch_op:
        batch_op.drop_column("homa_ir")
        batch_op.drop_column("assessment")
        batch_op.drop_column("extracted_parameters")
        batch_op.drop_column("source")

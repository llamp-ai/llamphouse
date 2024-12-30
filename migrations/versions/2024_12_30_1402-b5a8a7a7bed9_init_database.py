"""init database

Revision ID: b5a8a7a7bed9
Revises: 
Create Date: 2024-12-30 14:02:49.320080+07:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b5a8a7a7bed9'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('threads',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('tool_resources', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_threads_id'), 'threads', ['id'], unique=False)
    op.create_table('messages',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('thread_id', sa.String(), nullable=False),
    sa.Column('status', sa.Enum('in_progress', 'incomplete', 'completed', name='message_status_enum'), server_default='in_progress', nullable=False),
    sa.Column('incomplete_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('role', sa.Enum('assistant', 'user', name='role_enum'), nullable=False),
    sa.Column('content', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('assistant_id', sa.String(), nullable=True),
    sa.Column('run_id', sa.String(), nullable=True),
    sa.Column('attachments', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('completed_at', sa.Integer(), nullable=True),
    sa.Column('incomplete_at', sa.Integer(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['thread_id'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)
    op.create_table('runs',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('status', sa.Enum('queued', 'in_progress', 'requires_action', 'cancelling', 'cancelled', 'failed', 'completed', 'incomplete', 'expired', name='run_status_enum'), server_default='queued', nullable=False),
    sa.Column('required_action', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('last_error', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('incomplete_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('model', sa.String(), nullable=False),
    sa.Column('instructions', sa.Text(), nullable=False),
    sa.Column('tools', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('temperature', sa.Float(), server_default='1.0', nullable=False),
    sa.Column('top_p', sa.Float(), server_default='1.0', nullable=False),
    sa.Column('max_prompt_tokens', sa.Integer(), nullable=True),
    sa.Column('max_completion_tokens', sa.Integer(), nullable=True),
    sa.Column('truncation_strategy', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('tool_choice', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('parallel_tool_calls', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('response_format', postgresql.JSONB(astext_type=sa.Text()), server_default='"auto"', nullable=False),
    sa.Column('thread_id', sa.String(), nullable=False),
    sa.Column('assistant_id', sa.String(), nullable=False),
    sa.Column('expires_at', sa.Integer(), nullable=True),
    sa.Column('started_at', sa.Integer(), nullable=True),
    sa.Column('cancelled_at', sa.Integer(), nullable=True),
    sa.Column('failed_at', sa.Integer(), nullable=True),
    sa.Column('completed_at', sa.Integer(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['thread_id'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_runs_id'), 'runs', ['id'], unique=False)
    op.create_table('run_steps',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('object', sa.String(), nullable=False),
    sa.Column('assistant_id', sa.String(), nullable=False),
    sa.Column('thread_id', sa.String(), nullable=False),
    sa.Column('run_id', sa.String(), nullable=False),
    sa.Column('type', sa.Enum('message_creation', 'tool_calls', name='run_step_type_enum'), nullable=False),
    sa.Column('status', sa.Enum('in_progress', 'cancelled', 'failed', 'completed', 'expired', name='run_step_status_enum'), nullable=False),
    sa.Column('step_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('meta', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('usage', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('last_error', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('expired_at', sa.Integer(), nullable=True),
    sa.Column('cancelled_at', sa.Integer(), nullable=True),
    sa.Column('failed_at', sa.Integer(), nullable=True),
    sa.Column('completed_at', sa.Integer(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
    sa.ForeignKeyConstraint(['run_id'], ['runs.id'], ),
    sa.ForeignKeyConstraint(['thread_id'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_run_steps_id'), 'run_steps', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_run_steps_id'), table_name='run_steps')
    op.drop_table('run_steps')
    op.drop_index(op.f('ix_runs_id'), table_name='runs')
    op.drop_table('runs')
    op.drop_index(op.f('ix_messages_id'), table_name='messages')
    op.drop_table('messages')
    op.drop_index(op.f('ix_threads_id'), table_name='threads')
    op.drop_table('threads')
    # ### end Alembic commands ###
from sqlalchemy import (
    Column, 
    String, 
    Text, 
    Integer, 
    ForeignKey, 
    DateTime,
    Float,
    Boolean,
    Enum
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import json

Base = declarative_base()


role_enum = Enum(
    'assistant', 'user', 
    name='role_enum'
)
message_status_enum = Enum(
    'in_progress', 'incomplete', 'completed', 
    name='message_status_enum'
)
run_status_enum = Enum(
    'queued', 'in_progress', 'requires_action', 'cancelling', 
    'cancelled', 'failed', 'completed', 'incomplete', 'expired', 
    name='run_status_enum'
)
step_type_enum = Enum(
    'message_creation', 'tool_calls', 
    name='run_step_type_enum'
)
run_step_status_enum = Enum(
    'in_progress', 'cancelled', 'failed', 'completed', 'expired', 
    name='run_step_status_enum'
)

class Thread(Base):
    __tablename__ = 'threads'

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    tool_resources = Column(JSONB)
    meta = Column(JSONB)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())
    
    messages = relationship("Message", back_populates="thread")
    runs = relationship("Run", back_populates="thread")
    run_steps = relationship("RunStep", back_populates="thread")


class Message(Base):
    __tablename__ = 'messages'

    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey('threads.id'), nullable=False)
    status = Column(message_status_enum, nullable=False, server_default='in_progress')
    incomplete_details = Column(JSONB)
    role = Column(role_enum, nullable=False)
    content = Column(JSONB, nullable=False)
    assistant_id = Column(String)
    run_id = Column(String)
    attachments = Column(JSONB)
    meta = Column(JSONB)
    completed_at = Column(Integer)
    incomplete_at = Column(Integer)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())

    thread = relationship("Thread", back_populates="messages")

    def to_json(self):
        return {
            "id": self.id,
            "object": "thread.message",
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "assistant_id": self.assistant_id,
            "run_id": self.run_id,
            "attachments": self.attachments,
            "meta": self.meta,
            "completed_at": int(self.completed_at) if self.completed_at else None,
            "incomplete_at": int(self.incomplete_at) if self.incomplete_at else None,
            "status": self.status,
            "incomplete_details": self.incomplete_details
        }
    
    def __repr__(self):
        return f"<Message {self.id}>"
    
    def __str__(self):
        return json.dumps(self.to_json())

class Run(Base):
    __tablename__ = 'runs'

    id = Column(String, primary_key=True, index=True)
    status = Column(run_status_enum, nullable=False, server_default='queued')
    required_action = Column(JSONB)
    last_error = Column(JSONB)
    incomplete_details = Column(JSONB)
    model = Column(String, nullable=False)
    instructions = Column(Text, nullable=False)
    tools = Column(JSONB)
    meta = Column(JSONB)
    usage = Column(JSONB)
    temperature = Column(Float, nullable=False, server_default="1.0")
    top_p = Column(Float, nullable=False, server_default="1.0")
    max_prompt_tokens = Column(Integer)
    max_completion_tokens = Column(Integer)
    truncation_strategy = Column(JSONB)
    tool_choice = Column(JSONB)
    parallel_tool_calls = Column(Boolean, nullable=False, server_default="false")
    response_format = Column(JSONB, nullable=False, server_default='"auto"')
    thread_id = Column(String, ForeignKey('threads.id'), nullable=False)
    assistant_id = Column(String, nullable=False)
    expires_at = Column(Integer)
    started_at = Column(Integer)
    cancelled_at = Column(Integer)
    failed_at = Column(Integer)
    completed_at = Column(Integer)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())
    
    thread = relationship("Thread", back_populates="runs")
    run_steps = relationship("RunStep", back_populates="run")

class RunStep(Base):
    __tablename__ = 'run_steps'

    id = Column(String, primary_key=True, index=True)
    object = Column(String, nullable=False, default="thread.run.step")
    created_at = Column(Integer, nullable=False)
    assistant_id = Column(String, nullable=False)
    thread_id = Column(String, ForeignKey('threads.id'), nullable=False)
    run_id = Column(String, ForeignKey('runs.id'), nullable=False)
    type = Column(step_type_enum, nullable=False)
    status = Column(run_step_status_enum, nullable=False)
    step_details = Column(JSONB)

    meta = Column(JSONB)
    usage = Column(JSONB)
    last_error = Column(JSONB)
    expired_at = Column(Integer)
    cancelled_at = Column(Integer)
    failed_at = Column(Integer)
    completed_at = Column(Integer)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())

    thread = relationship("Thread", back_populates="run_steps")
    run = relationship("Run", back_populates="run_steps")
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
from sqlalchemy.dialects.sqlite import JSON
import json
import os
from ..streaming.event import Event

Base = declarative_base()

# Use JSONB if PostgreSQL, else use JSON (e.g., for SQLite)
if os.getenv("DATABASE_URL", "").startswith("sqlite"):
    JSONType = JSON
else:
    JSONType = JSONB

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
    tool_resources = Column(JSONType)
    meta = Column(JSONType)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    messages = relationship("Message", back_populates="thread")
    runs = relationship("Run", back_populates="thread")
    run_steps = relationship("RunStep", back_populates="thread")

    def to_dict(self):
        return {
            "id": self.id,
            "object": "thread",
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "name": self.name,
            "tool_resources": self.tool_resources,
            "metadata": self.meta,
            "updated_at": int(self.updated_at.timestamp()) if self.updated_at else None
        }

    def __repr__(self):
        return f"<Thread {self.id}>"
    
    def __str__(self):
        return json.dumps(self.to_dict())
    
    def to_event(self, event: str) -> Event:
        return Event(event=event, data=str(self))

class Message(Base):
    __tablename__ = 'messages'

    id = Column(String, primary_key=True, index=True)
    thread_id = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    status = Column(message_status_enum, nullable=False, server_default='in_progress')
    incomplete_details = Column(JSONType)
    role = Column(role_enum, nullable=False)
    content = Column(JSONType, nullable=False)
    assistant_id = Column(String)
    run_id = Column(String)
    attachments = Column(JSONType)
    meta = Column(JSONType)
    completed_at = Column(Integer)
    incomplete_at = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    thread = relationship("Thread", back_populates="messages")

    def to_dict(self):
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
            "metadata": self.meta,
            "completed_at": int(self.completed_at) if self.completed_at else None,
            "incomplete_at": int(self.incomplete_at) if self.incomplete_at else None,
            "status": self.status,
            "incomplete_details": self.incomplete_details,
            "updated_at": int(self.updated_at.timestamp()) if self.updated_at else None
        }
    
    def __repr__(self):
        return f"<Message {self.id}>"
    
    def __str__(self):
        return json.dumps(self.to_dict())
    
    def to_event(self, event: str) -> Event:
        return Event(event=event, data=str(self))

class Run(Base):
    __tablename__ = 'runs'

    id = Column(String, primary_key=True, index=True)
    status = Column(run_status_enum, nullable=False, server_default='queued')
    required_action = Column(JSONType)
    last_error = Column(JSONType)
    incomplete_details = Column(JSONType)
    model = Column(String, nullable=False)
    instructions = Column(Text, nullable=False)
    tools = Column(JSONType)
    meta = Column(JSONType)
    usage = Column(JSONType)
    temperature = Column(Float, nullable=False, server_default="1.0")
    top_p = Column(Float, nullable=False, server_default="1.0")
    max_prompt_tokens = Column(Integer)
    max_completion_tokens = Column(Integer)
    truncation_strategy = Column(JSONType)
    tool_choice = Column(JSONType)
    parallel_tool_calls = Column(Boolean, nullable=False, server_default="false")
    response_format = Column(JSONType, nullable=False, server_default='"auto"')
    thread_id = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    assistant_id = Column(String, nullable=False)
    expires_at = Column(Integer)
    started_at = Column(Integer)
    cancelled_at = Column(Integer)
    failed_at = Column(Integer)
    completed_at = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    reasoning_effort = Column(String, nullable=False, server_default='medium')
    
    thread = relationship("Thread", back_populates="runs")
    run_steps = relationship("RunStep", back_populates="run")

    def to_dict(self):
        return {
            "id": self.id,
            "object": "thread.run",
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "status": self.status,
            "required_action": self.required_action,
            "last_error": self.last_error,
            "incomplete_details": self.incomplete_details,
            "model": self.model,
            "instructions": self.instructions,
            "tools": self.tools,
            "metadata": self.meta,
            "usage": self.usage,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "truncation_strategy": self.truncation_strategy,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "response_format": self.response_format,
            "reasoning_effort": self.reasoning_effort,
            "thread_id": self.thread_id,
            "assistant_id": self.assistant_id,
            "expires_at": int(self.expires_at) if self.expires_at else None,
            "started_at": int(self.started_at) if self.started_at else None,
            "cancelled_at": int(self.cancelled_at) if self.cancelled_at else None,
            "failed_at": int(self.failed_at) if self.failed_at else None,
            "completed_at": int(self.completed_at) if self.completed_at else None,
            "updated_at": int(self.updated_at.timestamp()) if self.updated_at else None,
        }
    
    def __repr__(self):
        return f"<Run {self.id}>"
    
    def __str__(self):
        return json.dumps(self.to_dict())
    
    def to_event(self, event: str) -> Event:
        return Event(event=event, data=str(self))

class RunStep(Base):
    __tablename__ = 'run_steps'

    id = Column(String, primary_key=True, index=True)
    object = Column(String, nullable=False, default="thread.run.step")
    assistant_id = Column(String, nullable=False)
    thread_id = Column(String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    type = Column(step_type_enum, nullable=False)
    status = Column(run_step_status_enum, nullable=False)
    step_details = Column(JSONType)

    meta = Column(JSONType)
    usage = Column(JSONType)
    last_error = Column(JSONType)
    expired_at = Column(Integer)
    cancelled_at = Column(Integer)
    failed_at = Column(Integer)
    completed_at = Column(Integer)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    thread = relationship("Thread", back_populates="run_steps")
    run = relationship("Run", back_populates="run_steps")

    def to_dict(self):
        return {
            "id": self.id,
            "object": self.object,
            "assistant_id": self.assistant_id,
            "thread_id": self.thread_id,
            "run_id": self.run_id,
            "type": self.type,
            "status": self.status,
            "step_details": self.step_details,
            "metadata": self.meta,
            "usage": self.usage,
            "last_error": self.last_error,
            "expired_at": int(self.expired_at) if self.expired_at else None,
            "cancelled_at": int(self.cancelled_at) if self.cancelled_at else None,
            "failed_at": int(self.failed_at) if self.failed_at else None,
            "updated_at": int(self.updated_at.timestamp()) if self.updated_at else None,
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
            "completed_at": int(self.completed_at) if self.completed_at else None
        }
    
    def __repr__(self):
        return f"<RunStep {self.id}>"
    
    def __str__(self):
        return json.dumps(self.to_dict())
    
    def to_event(self, event: str) -> Event:
        return Event(event=event, data=str(self))
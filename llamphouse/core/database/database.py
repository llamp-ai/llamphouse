import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from llamphouse.core.assistant import Assistant
from .models import Thread, Message, Run, RunStep
from ..types import thread, message, run
from ..types.enum import run_status, run_step_status
from dotenv import load_dotenv
import uuid

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/llamphouse")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, bind=engine)
db = SessionLocal()

def insert_thread(threads: thread.CreateThreadRequest):
    try:
        custom_thread_id = None
        if threads.metadata:
            custom_thread_id = threads.metadata.get("thread_id")
        if custom_thread_id:
            thread_id = custom_thread_id
        else:
            thread_id = str(uuid.uuid4())
        item = Thread(
            id=thread_id,
            name=thread_id,
            tool_resources=threads.tool_resources,
            meta=threads.metadata
        )
        db.add(item)
        db.commit()
        return item
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return None
    
def insert_message(thread_id: str, message: message.CreateMessageRequest):
    try:
        custom_message_id = None
        if message.metadata:
            custom_message_id = message.metadata.get("message_id")
        if custom_message_id:
            message_id = custom_message_id
        else:
            message_id = str(uuid.uuid4())
        item = Message(
            id=message_id,
            role=message.role,
            content=message.content,
            attachments=message.attachments,
            meta=message.metadata or {},
            thread_id=thread_id
        )
        db.add(item)
        db.commit()
        return item
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return None
    
def insert_run(thread_id: str, run: run.RunCreateRequest, assistant: Assistant):
    try:
        custom_run_id = None
        if run.metadata:
            custom_run_id = run.metadata.get("run_id")
        if custom_run_id:
            run_id = custom_run_id
        else:
            run_id = str(uuid.uuid4())
        item = Run(
            id=run_id,
            thread_id=thread_id,
            assistant_id=run.assistant_id,
            model=run.model or assistant.model,
            instructions=run.instructions or assistant.instructions,
            tools=run.tools or assistant.tools,
            meta=run.metadata or {},
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
        )
        db.add(item)
        db.commit()
        return item
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return None

def insert_run_step(run_id: str, assistant_id: str, thread_id: str, step_type: str, step_details: dict, status: str = run_step_status.IN_PROGRESS):
    try:
        run_step_id = str(uuid.uuid4())
        item = RunStep(
            id=run_step_id,
            object="thread.run.step",
            assistant_id=assistant_id,
            thread_id=thread_id,
            run_id=run_id,
            type=step_type,
            status=status,
            step_details=step_details
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return item
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return None
    
def get_run_by_id(run_id: str):
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        return run
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_thread_by_id(thread_id: str):
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        return thread
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_message_by_id(message_id: str):
    try:
        message = db.query(Message).filter(Message.id == message_id).first()
        return message
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_messages_by_thread_id(
        thread_id: str, 
        limit: int = 20, 
        order: str = "desc", 
        after: str = None, 
        before: str = None
    ) -> list[Message]:
    try:
        query = db.query(Message).filter(Message.thread_id == thread_id)
        if order == "asc":
            query = query.order_by(Message.created_at.asc())
        else:
            query = query.order_by(Message.created_at.desc())
        if after:
            query = query.filter(Message.id > after)
        if before:
            query = query.filter(Message.id < before)
        return query.limit(limit).all()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
def get_runs_by_thread_id(
        thread_id: str, 
        limit: int = 20, 
        order: str = "desc", 
        after: str = None, 
        before: str = None
    ) -> list[Message]:
    try:
        query = db.query(Run).filter(Run.thread_id == thread_id)
        if order == "asc":
            query = query.order_by(Run.created_at.asc())
        else:
            query = query.order_by(Run.created_at.desc())
        if after:
            query = query.filter(Run.id > after)
        if before:
            query = query.filter(Run.id < before)
        return query.limit(limit).all()
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_pending_runs():
    try:
        pending_runs = db.query(Run).filter(Run.status == run_status.QUEUED).all()
        return pending_runs
    except Exception as e:
        print(f"An error occurred while fetching pending runs: {e}")
        return []
    
def get_pending_run():
    try:
        pending_runs = db.query(Run).filter(Run.status == run_status.QUEUED).with_for_update().first()
        return pending_runs
    except Exception as e:
        print(f"An error occurred while fetching pending runs: {e}")
        return None

def list_run_steps(thread_id: str, run_id: str):
    try:
        run_steps = db.query(RunStep).filter(RunStep.run_id == run_id, RunStep.thread_id == thread_id)
        return run_steps
    except Exception as e:
        print(f"An error occurred while fetching run steps: {e}")
        return []

def update_thread_metadata(thread_id: str, metadata: dict):
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if thread:
            thread.meta = metadata
            db.commit()
            return thread
        return None
    except Exception as e:
        db.rollback()
        print(f"An error occurred while updating thread metadata: {e}")
        return None

def update_message_metadata(thread_id: str, message_id: str, metadata: dict):
    try:
        message = db.query(Message).filter(Message.thread_id == thread_id, Message.id == message_id).first()
        
        if message:
            message.meta = metadata or {}
            db.commit()
            return message
        else:
            return None
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return None

def update_run(run: Run):
    try:
        db.merge(run)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"An error occurred while updating the run: {e}")
        return False
    
def update_run_metadata(thread_id: str, run_id: str, metadata: dict):
    try:
        run = db.query(Run).filter(Run.thread_id == thread_id, Run.id == run_id).first()
        if run:
            run.meta = metadata
            db.commit()
            return run
        return None
    except Exception as e:
        db.rollback()
        print(f"An error occurred while updating thread metadata: {e}")
        return None
    
def update_run_status(run_id: str, status: str, error: dict = None):
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = status
            run.last_error = error
            db.commit()
            return run
        return None
    except Exception as e:
        db.rollback()
        print(f"An error occurred while updating the run: {e}")
        return None
    
def update_run_step_status(run_step_id: str, status: str, output = None, error: str = None):
    try:
        run_step = db.query(RunStep).filter(RunStep.id == run_step_id).first()
        if run_step:
            run_step.status = status
            run_step.last_error = error
            if output:
                run_step.step_details["tool_calls"][0]["function"]["output"] = output
            db.commit()
            return run_step
        return None
    except Exception as e:
        db.rollback()
        print(f"An error occurred while updating the run step: {e}")
        return None

def delete_thread_by_id(thread_id: str):
    try:
        thread = db.query(Thread).filter(Thread.id == thread_id).first()
        if thread:
            db.delete(thread)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return False

def delete_messages_by_thread_id(thread_id: str):
    try:
        db.query(Message).filter(Message.thread_id == thread_id).delete()
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return False

def delete_message_by_id(message_id: str):
    try:
        message = db.query(Message).filter(Message.id == message_id).first()
        if message:
            db.delete(message)
            db.commit()
            return True
        return False
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return False

def delete_message(thread_id: str, message_id: str):
    try:
        message = db.query(Message).filter(
            Message.thread_id == thread_id, Message.id == message_id).first()
        
        if message:
            db.delete(message)
            db.commit()
            return True
        else:
            return False
    except Exception as e:
        db.rollback()
        print(f"An error occurred: {e}")
        return False
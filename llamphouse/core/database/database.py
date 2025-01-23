import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Thread, Message, Run, RunStep
from ..types import thread, message, run
from ..types.enum import run_status, run_step_status
from dotenv import load_dotenv
import uuid

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/llamphouse")
engine = create_engine(DATABASE_URL)
SessionGlobal = sessionmaker(autocommit=False, bind=engine)

class DatabaseManager:
    def __init__(self, db_session: Session = None):
        self.db = db_session if db_session else SessionGlobal()

    def insert_thread(self, threads: thread.CreateThreadRequest):
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
            self.db.add(item)
            self.db.commit()
            return item
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return None
        
    def insert_message(self, thread_id: str, message: message.CreateMessageRequest):
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
            self.db.add(item)
            self.db.commit()
            return item
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return None
        
    def insert_run(self, thread_id: str, run: run.RunCreateRequest, assistant):
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
            self.db.add(item)
            self.db.commit()
            return item
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return None

    def insert_run_step(self, run_id: str, assistant_id: str, thread_id: str, step_type: str, step_details: dict, status: str = run_step_status.IN_PROGRESS):
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
            self.db.add(item)
            self.db.commit()
            self.db.refresh(item)
            return item
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return None
        
    def get_run_by_id(self, run_id: str):
        try:
            run = self.db.query(Run).filter(Run.id == run_id).first()
            return run
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_thread_by_id(self, thread_id: str):
        try:
            thread = self.db.query(Thread).filter(Thread.id == thread_id).first()
            return thread
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_message_by_id(self, message_id: str):
        try:
            message = self.db.query(Message).filter(Message.id == message_id).first()
            return message
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_messages_by_thread_id(
            self,
            thread_id: str, 
            limit: int = 20, 
            order: str = "desc", 
            after: str = None, 
            before: str = None
        ) -> list[Message]:
        try:
            query = self.db.query(Message).filter(Message.thread_id == thread_id)
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
            self,
            thread_id: str, 
            limit: int = 20, 
            order: str = "desc", 
            after: str = None, 
            before: str = None
        ) -> list[Message]:
        try:
            query = self.db.query(Run).filter(Run.thread_id == thread_id)
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

    def get_pending_runs(self):
        try:
            pending_runs = self.db.query(Run).filter(Run.status == run_status.QUEUED).all()
            return pending_runs
        except Exception as e:
            print(f"An error occurred while fetching pending runs: {e}")
            return []
        
    def get_pending_run(self):
        try:
            pending_runs = self.db.query(Run).filter(Run.status == run_status.QUEUED).with_for_update().first()
            return pending_runs
        except Exception as e:
            print(f"An error occurred while fetching pending runs: {e}")
            return None

    def list_run_steps(self, thread_id: str, run_id: str):
        try:
            run_steps = self.db.query(RunStep).filter(RunStep.run_id == run_id, RunStep.thread_id == thread_id)
            return run_steps
        except Exception as e:
            print(f"An error occurred while fetching run steps: {e}")
            return []

    def update_thread_metadata(self, thread_id: str, metadata: dict):
        try:
            thread = self.db.query(Thread).filter(Thread.id == thread_id).first()
            if thread:
                thread.meta = metadata
                self.db.commit()
                return thread
            return None
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating thread metadata: {e}")
            return None

    def update_message_metadata(self, thread_id: str, message_id: str, metadata: dict):
        try:
            message = self.db.query(Message).filter(Message.thread_id == thread_id, Message.id == message_id).first()
            
            if message:
                message.meta = metadata or {}
                self.db.commit()
                return message
            else:
                return None
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return None

    def update_thread(self, thread: Thread):
        try:
            self.db.merge(thread)
            self.db.commit()
            return thread
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating the thread: {e}")
            return None 

    def update_message(self, message: Message):
        try:
            self.db.merge(message)
            self.db.commit()
            return message
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating the message: {e}")
            return None

    def update_run(self, run: Run):
        try:
            self.db.merge(run)
            self.db.commit()
            return run
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating the run: {e}")
            return None
        
    def update_run_metadata(self, thread_id: str, run_id: str, metadata: dict):
        try:
            run = self.db.query(Run).filter(Run.thread_id == thread_id, Run.id == run_id).first()
            if run:
                run.meta = metadata
                self.db.commit()
                return run
            return None
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating thread metadata: {e}")
            return None
        
    def update_run_status(self, run_id: str, status: str, error: dict = None):
        try:
            run = self.db.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = status
                run.last_error = error
                self.db.commit()
                return run
            return None
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating the run: {e}")
            return None
        
    def update_run_step_status(self, run_step_id: str, status: str, output = None, error: str = None):
        try:
            run_step = self.db.query(RunStep).filter(RunStep.id == run_step_id).first()
            if run_step:
                run_step.status = status
                run_step.last_error = error
                if output:
                    run_step.step_details["tool_calls"][0]["function"]["output"] = output
                self.db.commit()
                return run_step
            return None
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred while updating the run step: {e}")
            return None

    def delete_thread_by_id(self, thread_id: str):
        try:
            thread = self.db.query(Thread).filter(Thread.id == thread_id).first()
            if thread:
                self.db.delete(thread)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return False

    def delete_messages_by_thread_id(self, thread_id: str):
        try:
            self.db.query(Message).filter(Message.thread_id == thread_id).delete()
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return False

    def delete_message_by_id(self, message_id: str):
        try:
            message = self.db.query(Message).filter(Message.id == message_id).first()
            if message:
                self.db.delete(message)
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return False

    def delete_message(self, thread_id: str, message_id: str):
        try:
            message = self.db.query(Message).filter(
                Message.thread_id == thread_id, Message.id == message_id).first()
            
            if message:
                self.db.delete(message)
                self.db.commit()
                return True
            else:
                return False
        except Exception as e:
            self.db.rollback()
            print(f"An error occurred: {e}")
            return False
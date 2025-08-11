from .database.database import DatabaseManager, SessionLocal
from typing import Dict, Optional
from .types.message import Attachment, CreateMessageRequest, MessageObject
from .types.run_step import ToolCallsStepDetails
from .types.run import ToolOutput, RunObject
from .types.enum import run_step_status, run_status
import uuid
import json
import asyncio

class Context:
    def __init__(self, assistant, assistant_id: str, run_id: str, run, thread_id: str = None, queue: asyncio.Queue = None, db_session = None):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant = assistant
        self.db = DatabaseManager(db_session=db_session or SessionLocal())
        self.thread = self._get_thread_by_id(thread_id)
        self.messages = self._list_messages_by_thread_id(thread_id)
        self.run: RunObject = run
        self.__queue = queue
        
    def insert_message(self, content: str, attachment: Attachment = None, metadata: Dict[str, str] = {}, role: str = "assistant"):
        messageRequest = CreateMessageRequest(
            role=role,
            content=content,
            attachment=attachment,
            metadata=metadata
        )
        new_message = self.db.insert_message(self.thread_id, messageRequest)
        step_details = self._message_step_details(new_message.id)
        self.db.insert_run_step(run_id=self.run_id, assistant_id=self.assistant_id, thread_id=self.thread_id, step_type="message_creation", step_details=step_details, status=run_step_status.COMPLETED)

        # Update context.message
        self.messages = self._list_messages_by_thread_id(self.thread_id)
        return new_message
    
    def insert_tool_calls_step(self, step_details: ToolCallsStepDetails, output: Optional[ToolOutput] = None):
        status = run_step_status.COMPLETED if output else run_step_status.IN_PROGRESS
        run_step = self.db.insert_run_step(
            run_id=self.run_id,
            assistant_id=self.assistant_id,
            thread_id=self.thread_id,
            step_type="tool_calls",
            step_details=step_details,
            status=status
        )

        if output:
            self.db.insert_tool_output(run_step, output)
        else:
            self.db.update_run_status(self.run_id, run_status.REQUIRES_ACTION)

        return run_step
    
    def update_thread_details(self, **kwargs):
        if not self.thread:
            raise ValueError("Thread object is not initialized.")

        for key, value in kwargs.items():
            if hasattr(self.thread, key):
                setattr(self.thread, key, value)
            else:
                raise AttributeError(f"Thread object has no attribute '{key}'")
        try:
            updated_thread = self.db.update_thread(self.thread)
            return updated_thread
        except Exception as e:
            raise Exception(f"Failed to update thread in the database: {e}")

    def update_message_details(self, message_id: str, **kwargs):
        message = next((msg for msg in self.messages if msg["id"] == message_id), None)
        if not message:
            raise ValueError(f"Message with ID '{message_id}' not found in thread.")

        for key, value in kwargs.items():
            if key in message:
                message[key] = value
            else:
                raise AttributeError(f"Message object has no attribute '{key}'")

        try:
            self.db.update_message(message)
            self.messages = self._list_messages_by_thread_id(self.thread_id)
            return message
        except Exception as e:
            raise Exception(f"Failed to update message: {e}")

    def update_run_details(self, **kwargs):
        if not self.run:
            raise ValueError("Run object is not initialized.")

        for key, value in kwargs.items():
            if hasattr(self.run, key):
                setattr(self.run, key, value)
            else:
                raise AttributeError(f"Run object has no attribute '{key}'")
        try:
            updated_run = self.db.update_run(self.run)
            return updated_run
        except Exception as e:
            raise Exception(f"Failed to update run in the database: {e}")

    def _get_thread_by_id(self, thread_id):
        thread = self.db.get_thread_by_id(thread_id)
        if not thread:
            print(f"Thread with ID {thread_id} not found.")
        return thread

    def _list_messages_by_thread_id(self, thread_id):
        messages = self.db.list_messages_by_thread_id(thread_id, order="asc", limit=1000)
        if not messages:
            print(f"No messages found in thread {thread_id}.")
        return [MessageObject.from_db_message(msg) for msg in messages]
    
    def _get_function_from_tools(self, function_name: str):
        for tool in self.assistant.tools:
            if tool['type'] == 'function' and tool['function']['name'] == function_name:
                function_name = tool['function']['name']
                return getattr(self.assistant, function_name)
        return None

    def _message_step_details(self, message_id: str):
        return {
            "type": "message_creation",
            "message_creation": {
                "message_id": message_id
            }
        }
    
    def _function_call_step_details(self, function_name: str, args: tuple, kwargs: dict, output: str = None):
        return {
            "type": "tool_calls",
            "tool_calls": [{
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": {
                        "args": args,
                        "kwargs": kwargs
                    },
                    "output": output
                }
            }]
        }
    
    async def create_message_stream(self, stream):
        has_started = False
        message = None
        for chunk in stream:
            # print(f"Chunk object type:", chunk)
            if not has_started:
                has_started = True
                if chunk.object == 'chat.completion.chunk':
                    message = self.create_message('')
                    print(f"Created message:", message)
                    await self.__queue.put({"event": "thread.message.created", "data": json.dumps({
                        "id": message.id,
                        "object": "thread.message",
                        "created_at": int(message.created_at.timestamp()),
                        "thread_id": self.thread_id,
                        "role": "assistant",
                        "content": [
                            {
                            "type": "text",
                            "text": {
                                "value": "",
                                "annotations": []
                            }
                            }
                        ],
                        "assistant_id": self.assistant_id,
                        "run_id": self.run_id,
                        "attachments": [],
                        "metadata": {}
                    })})
                else:
                    print(f"Stream object type {chunk.object} not recognized.")
            else:
                if chunk.object == 'chat.completion.chunk':
                    if chunk.choices[0].delta.content == None or not chunk.choices[0].finish_reason == None:
                        await self.__queue.put({"event": "thread.message.completed", "data": json.dumps({
                            "id": message.id,
                            "object": "thread.message",
                            "created_at": int(message.created_at.timestamp()),
                            "thread_id": self.thread_id,
                            "role": "assistant",
                            "content": [
                                {
                                "type": "text",
                                "text": {
                                    "value": None,
                                    "annotations": []
                                }
                                }
                            ],
                            "assistant_id": self.assistant_id,
                            "run_id": self.run_id,
                            "attachments": [],
                            "metadata": {}
                        })})

                    # message.content.text.value += chunk.choices[0].delta.content
                        
                    await self.__queue.put({"event": "thread.message.delta", "data": json.dumps({
                        "id": message.id,
                        "object": "thread.message.delta",
                        "delta": {
                            "content": [
                                {
                                    "index": chunk.choices[0].index,
                                    "type": "text",
                                    "text": { "value": chunk.choices[0].delta.content, "annotations": [] }
                                }
                            ]
                        }
                    })})
                else:
                    print(f"Stream object type {chunk.object} not recognized.")
            # print(chunk)
            # await self.create_message(chunk)

        await self.__queue.put(None)
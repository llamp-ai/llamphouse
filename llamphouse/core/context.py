from .database import database as db
from typing import Dict
from .types.message import Attachment, CreateMessageRequest
from .types.enum import run_step_status
import uuid
import json
import asyncio

class Context:
    def __init__(self, assistant, assistant_id: str, run_id: str, run, thread_id: str = None, queue: asyncio.Queue = None):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant = assistant
        self.thread = self._get_thread_by_id(thread_id)
        self.messages = self._get_messages_by_thread_id(thread_id)
        self.run = run
        self.__queue = queue
        
    def create_message(self, content: str, attachment: Attachment = None, metadata: Dict[str, str] = {}):
        messageRequest = CreateMessageRequest(
            role="assistant",
            content=content,
            attachment=attachment,
            metadata=metadata
        )
        new_message = db.insert_message(self.thread_id, messageRequest)
        step_details = self._message_step_details(new_message.id)
        db.insert_run_step(run_id=self.run_id, assistant_id=self.assistant_id, thread_id=self.thread_id, step_type="message_creation", step_details=step_details, status=run_step_status.COMPLETED)

        # Update context.message
        self.messages = self._get_messages_by_thread_id(self.thread_id)
        return new_message
    
    async def call_function(self, function_name: str, *args, **kwargs):
        function = self._get_function_from_tools(function_name)
        if not function:
            raise ValueError(f"Function {function_name} not found in assistant's tools.")
        
        step_details = self._function_call_step_details(function_name, args, kwargs)
        step = db.insert_run_step(run_id=self.run_id, assistant_id=self.assistant_id, thread_id=self.thread_id, step_type="tool_calls", step_details=step_details, status=run_step_status.IN_PROGRESS)
        try:
            result = await function(*args, **kwargs)
            db.update_run_step_status(run_step_id=step.id, status=run_step_status.COMPLETED, output=result)
            return result
        except Exception as e:
            db.update_run_step_status(run_step_id=step.id, status=run_step_status.FAILED, error=str(e))
            raise e

    def _get_thread_by_id(self, thread_id):
        thread = db.get_thread_by_id(thread_id)
        if not thread:
            print(f"Thread with ID {thread_id} not found.")
        return thread

    def _get_messages_by_thread_id(self, thread_id):
        messages = db.get_messages_by_thread_id(thread_id)
        if not messages:
            print(f"No messages found in thread {thread_id}.")
        return messages
    
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
import queue
from .database.database import DatabaseManager, SessionLocal
from typing import Dict, Optional
from .types.message import Attachment, CreateMessageRequest, MessageObject
from .types.run_step import ToolCallsStepDetails
from .types.run import ToolOutput, RunObject
from .types.enum import run_step_status, run_status, event_type, message_status
import uuid
import asyncio
from .streaming.openai_event_handler import OpenAIEventHandler
from .streaming.event_queue.base_event_queue import BaseEventQueue

class Context:
    def __init__(self, assistant, assistant_id: str, run_id: str, run, thread_id: str = None, queue: BaseEventQueue = None, db_session = None, loop = None):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant = assistant
        self.db = DatabaseManager(db_session=db_session or SessionLocal())
        self.thread = self._get_thread_by_id(thread_id)
        self.messages = self._list_messages_by_thread_id(thread_id)
        self.run: RunObject = run
        self.__queue: BaseEventQueue = queue
        self.__loop = loop
        
    def insert_message(self, content: str, attachment: Attachment = None, metadata: Dict[str, str] = {}, role: str = "assistant"):
        messageRequest = CreateMessageRequest(
            role=role,
            content=content,
            attachment=attachment,
            metadata=metadata
        )
        new_message = self.db.insert_message(self.thread_id, messageRequest, status=message_status.COMPLETED)

        # Send events to the queue
        self.__queue.add(new_message.to_event(event_type.MESSAGE_CREATED))
        self.__queue.add(new_message.to_event(event_type.MESSAGE_IN_PROGRESS))
        self.__queue.add(new_message.to_event(event_type.MESSAGE_COMPLETED))

        step_details = self._message_step_details(new_message.id)
        new_step_detail = self.db.insert_run_step(run_id=self.run_id, assistant_id=self.assistant_id, thread_id=self.thread_id, step_type="message_creation", step_details=step_details, status=run_step_status.COMPLETED)

        # Send events to the queue
        self.__queue.add(new_step_detail.to_event(event_type.RUN_STEP_CREATED))
        self.__queue.add(new_step_detail.to_event(event_type.RUN_STEP_IN_PROGRESS))
        self.__queue.add(new_step_detail.to_event(event_type.RUN_STEP_COMPLETED))

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
        messages = self.db.list_messages_by_thread_id(thread_id, order="asc")
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
    
    def _send_event(self, event):
        if self.__loop:
            asyncio.run_coroutine_threadsafe(self.__queue.async_q.put(event), self.__loop)
        else:
            self.__queue.sync_q.put(event)

    def send_completion_event(self, event):
        pass

    
    def handle_completion_stream(self, stream):
        full_text = ""
        full_tool_calls = {}
        active_tool = None

        event_handler = OpenAIEventHandler(self._send_event, self.assistant_id, self.thread_id, self.run_id)

        # message = self.insert_message("")

        # create_event = {
        #     "event": event_type.MESSAGE_CREATED,
        #     "data": {
        #         "id": message.id,
        #         "object": "thread.message",
        #         "created_at": message.created_at.isoformat(),
        #         "thread_id": self.thread_id,
        #         "role": "assistant",
        #         "content": [],
        #         "assistant_id": self.assistant_id,
        #         "run_id": self.run_id,
        #         "attachments": [],
        #         "metadata": {}
        #     }
        # }
        # self._send_event(create_event)

        # index = 0  # Tracks position of text parts (we're assuming 1-part text only for now)

        for chunk in stream:
            event_handler.handle_event(chunk)
            
            # if not chunk.choices or not chunk.choices[0].delta:
            #     print("No choices or delta in chunk, skipping...", chunk, flush=True)
            #     delta_event = {
            #             "event": event_type.MESSAGE_DELTA,
            #             "data": {
            #                 "id": message.id,
            #                 "object": "thread.message.created",
            #                 "thread_id": self.thread_id,
            #                 "role": "assistant",
            #                 "content": [
            #                     {
            #                         "type": "text",
            #                         "text": {
            #                             "value": "",
            #                             "annotations": []
            #                         }
            #                     }
            #                 ],
            #                 "assistant_id": self.assistant_id,
            #                 "run_id": self.run_id,
            #                 "attachments": [],
            #                 "metadata": {}
            #             }
            #         }
            #     self._send_event(delta_event)
            # else:
            #     print(f"Processing chunk", flush=True)
            #     delta = chunk.choices[0].delta
            #     finish_reason = chunk.choices[0].finish_reason
            #     if delta.content: # text response
            #         full_text += delta.content
            #         print(f"Received chunk", flush=True)

            #         delta_event = {
            #             "event": event_type.MESSAGE_DELTA,
            #             "data": {
            #                 "id": message.id,
            #                 "object": "thread.message.delta",
            #                 "delta": {
            #                     "content": [
            #                         {
            #                             "index": index,
            #                             "type": "text",
            #                             "text": {
            #                                 "value": delta.content,
            #                                 "annotations": []
            #                             }
            #                         }
            #                     ]
            #                 }
            #             }
            #         }
            #         self._send_event(delta_event)
            #         index += 1

            #     if delta.tool_calls: # tool call response
            #         # print(f"Received tool calls", delta.tool_calls, flush=True)
            #         for tool_call in delta.tool_calls:
            #             function_name = tool_call.function.name
            #             if function_name:
            #                 active_tool = function_name

            #             arguments = tool_call.function.arguments
            #             print(f"Function name: {active_tool}, Arguments: {arguments}", flush=True)
            #             if active_tool not in full_tool_calls:
            #                 full_tool_calls[active_tool] = arguments
            #             else:
            #                 full_tool_calls[active_tool] += arguments

            #             delta_event = {
            #                 "event": event_type.RUN_STEP_DELTA,
            #                 "data": {
            #                     "id": message.id,
            #                     "object": "thread.run.step.delta",
            #                     "delta": {
            #                         "step_details": {
            #                             "type": "tool_calls",
            #                             "tool_calls": [{
            #                                 "index": index,
            #                                 "id": str(uuid.uuid4()),
            #                                 "type": "function",
            #                                 "function": {
            #                                     "name": active_tool,
            #                                     "arguments": arguments
            #                                 }
            #                             }]
            #                         }
            #                     }
            #                 }
            #             }
            #             # print(f"Sending tool call delta event: {delta_event}", flush=True)
            #             self._send_event(delta_event)
                    
            #         index += 1


            #     # If stream ends
            #     if finish_reason == "stop" or finish_reason == "tool_calls":
            #         final_event = {
            #             "event": event_type.DONE,
            #             "data": {
            #                 "id": self.run_id,
            #                 "thread_id": self.thread_id,
            #                 "assistant_id": self.assistant_id,
            #                 "status": "completed"
            #             }
            #         }
            #         self._send_event(final_event)
            #         break
        # self.db.insert_message()
        # message.content = full_text
        # self.db.update_message(message)

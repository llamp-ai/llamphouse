import asyncio
import uuid
from typing import Dict, Optional
from .types.message import Attachment, CreateMessageRequest, MessageObject, ModifyMessageRequest
from .types.run_step import ToolCallsStepDetails, CreateRunStepRequest
from .types.run import ToolOutput, RunObject, ModifyRunRequest
from .types.thread import ModifyThreadRequest
from .types.enum import run_step_status, run_status, event_type, message_status
from .streaming.openai_event_handler import OpenAIEventHandler
from .streaming.event_queue.base_event_queue import BaseEventQueue
from .data_stores.base_data_store import BaseDataStore

class Context:
    def __init__(
            self, 
            assistant, 
            assistant_id: str, 
            run_id: str,
            run: RunObject,
            thread_id: str = None, 
            queue: Optional[BaseEventQueue] = None, 
            data_store: Optional[BaseDataStore] = None, 
            loop = None
    ):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant = assistant
        self.data_store = data_store
        self.thread = None
        self.messages: list[MessageObject] = []
        self.run: RunObject = run
        self.__queue = queue
        self.__loop = loop

    @classmethod
    async def create(cls, **kwargs) -> "Context":
        self = cls(**kwargs)
        self.thread = await self._get_thread_by_id(self.thread_id)
        self.messages = await self._list_messages_by_thread_id(self.thread_id)
        return self
        
    async def insert_message(self, content: str, attachment: Attachment = None, metadata: Optional[Dict[str, str]] = None, role: str = "assistant"):
        metadata = metadata or {}
        message_request = CreateMessageRequest(role=role, content=content, attachments=attachment, metadata=metadata)
        new_message = await self.data_store.insert_message(
            thread_id=self.thread_id,
            message=message_request,
            status=message_status.COMPLETED,
            event_queue=self.__queue,
        )
        step_details = self._message_step_details(new_message.id)
        await self.data_store.insert_run_step(
            thread_id=self.thread_id,
            run_id=self.run_id,
            step=CreateRunStepRequest(
                assistant_id=self.assistant_id,
                step_details=step_details,
                metadata={},
            ),
            event_queue=self.__queue,
        )
        self.messages = await self._list_messages_by_thread_id(self.thread_id)
        return new_message
    
    async def insert_tool_calls_step(self, step_details: ToolCallsStepDetails, output: Optional[ToolOutput] = None):
        status = run_step_status.COMPLETED if output else run_step_status.IN_PROGRESS
        run_step = await self.data_store.insert_run_step(
            run_id=self.run_id,
            thread_id=self.thread_id,
            step=CreateRunStepRequest(
                assistant_id=self.assistant_id,
                step_details=step_details,
                metadata={},
            ),
            status=status,
            event_queue=self.__queue,
        )

        if output:
            await self.data_store.submit_tool_outputs_to_run(self.thread_id, self.run_id, [output])
        else:
            await self.data_store.update_run_status(self.thread_id, self.run_id, run_status.REQUIRES_ACTION)

        return run_step
    
    async def update_thread_details(self, modifications: Dict[str, any]):
        if not self.thread:
            raise ValueError("Thread object is not initialized.")
        try:
            req = ModifyThreadRequest(**modifications)
            updated_thread = await self.data_store.update_thread(self.thread_id, req)
            if updated_thread:
                self.thread = updated_thread
            return updated_thread
        except Exception as e:
            raise Exception(f"Failed to update thread in the data_store: {e}")

    async def update_message_details(self, message_id: str, modifications: Dict[str, any]):
        try:
            req = ModifyMessageRequest(**modifications)
            updated_message = await self.data_store.update_message(self.thread_id, message_id, req)
            self.messages = await self._list_messages_by_thread_id(self.thread_id)
            return updated_message
        except Exception as e:
            raise Exception(f"Failed to update message via data_store: {e}")

    async def update_run_details(self, modifications: Dict[str, any]):
        if not self.run:
            raise ValueError("Run object is not initialized.")

        req = ModifyRunRequest(**modifications)
        try:
            updated_run = await self.data_store.update_run(self.thread_id, self.run_id, req)
            if updated_run:
                self.run = updated_run
            return updated_run
        except Exception as e:
            raise Exception(f"Failed to update run in the data_store: {e}")

    async def _get_thread_by_id(self, thread_id):
        if not thread_id:
            return None
        thread = await self.data_store.get_thread_by_id(thread_id)
        if not thread:
            print(f"Thread with ID {thread_id} not found.")
        return thread

    async def _list_messages_by_thread_id(self, thread_id):
        if not thread_id:
            return []
        resp = await self.data_store.list_messages(thread_id=thread_id, limit=100, order="asc", after=None, before=None)
        if not resp or not resp.data:
            print(f"No messages found in thread {thread_id}.")
        return resp.data if resp else []
    
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
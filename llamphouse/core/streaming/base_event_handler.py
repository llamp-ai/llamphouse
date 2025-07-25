from abc import ABC, abstractmethod
from ast import arg
from ..types.enum import event_type
from datetime import datetime

class BaseEventHandler(ABC):
    def __init__(self, send_event):
        self.send_event = send_event
        self.index = 0

    # @abstractmethod
    def handle_event(self, event):
        # This method should be overridden by subclasses
        # raise NotImplementedError("Subclasses must implement handle_event method")
        pass
    
    def send_event_message_created(self, assistant_id, thread_id, run_id, message_id):
        self.send_event({
            "event": event_type.MESSAGE_CREATED,
            "data": {
                "id": message_id,
                "object": "thread.message",
                "created_at": datetime.now().isoformat(),
                "thread_id": thread_id,
                "role": "assistant",
                "content": [],
                "assistant_id": assistant_id,
                "run_id": run_id,
                "attachments": [],
                "metadata": {}
            }
        })

    def send_event_message_in_progress(self, assistant_id, thread_id, run_id, message_id):
        self.send_event({
            "event": event_type.MESSAGE_IN_PROGRESS,
            "data": {
                "id": message_id,
                "object": "thread.message",
                "created_at": datetime.now().isoformat(),
                "thread_id": thread_id,
                "role": "assistant",
                "content": [],
                "assistant_id": assistant_id,
                "run_id": run_id,
                "attachments": [],
                "metadata": {}
            }
        })

    def send_event_message_delta(self, message_id, text):
        self.send_event({
            "event": event_type.MESSAGE_DELTA,
            "data": {
                "id": message_id,
                "object": "thread.message.delta",
                "delta": {
                    "content": [
                        {
                            "index": self.index,
                            "type": "text",
                            "text": {
                                "value": text,
                                "annotations": []
                            }
                        }
                    ]
                }
            }
        })

    def send_event_message_completed(self, assistant_id, thread_id, run_id, message_id, text):
        self.send_event({
            "event": event_type.MESSAGE_COMPLETED,
            "data": {
                "id": message_id,
                "object": "thread.message",
                "created_at": datetime.now().isoformat(),
                "thread_id": thread_id,
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": {
                            "value": text,
                            "annotations": []
                        }
                    }
                ],
                "assistant_id": assistant_id,
                "run_id": run_id,
                "attachments": [],
                "metadata": {}
            }
        })

    def send_event_done(self, assistant_id, thread_id, run_id):
        self.send_event({
            "event": event_type.DONE,
            "data": {
                "id": run_id,
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "status": "completed"
            }
        })
    
    def send_event_run_step_tool_created(self, assistant_id, thread_id, run_id, step_id, tool_call_id, function_name):
        self.send_event({
            "event": event_type.RUN_STEP_CREATED,
            "data": {
                "id": step_id,
                "object": "thread.run.step",
                "created_at": datetime.now().isoformat(),
                "run_id": run_id,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "type": "tool_calls",
                "status": "completed",
                "cancelled_at": None,
                "completed_at": None,
                "expired_at": None,
                "failed_at": None,
                "last_error": None,
                "step_details": {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "function": {
                                "name": function_name,
                                "arguments": "",
                                "output": None
                            },
                            "id": tool_call_id,
                            "type": "function"
                        }
                    ]
                },
                "usage": None
            }
        })

    def send_event_run_step_tool_delta(self, step_id, tool_call_id, function_name, arguments):
        self.send_event({
            "event": event_type.RUN_STEP_DELTA,
            "data": {
                "id": step_id,
                "object": "thread.run.step.delta",
                "delta": {
                    "step_details": {
                        "type": "tool_calls",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": arguments,
                                    "output": None
                                }
                            }
                        ]
                    }
                }
            }
        })

    def send_event_run_step_completed(self, assistant_id, thread_id, run_id, step_id, tool_call_id, function_name, arguments):
        self.send_event({
            "event": event_type.RUN_STEP_COMPLETED,
            "data": {
                "id": step_id,
                "object": "thread.run.step",
                "created_at": datetime.now().isoformat(),
                "run_id": run_id,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "type": "tool_calls",
                "status": "completed",
                "cancelled_at": None,
                "completed_at": datetime.now().isoformat(),
                "expired_at": None,
                "failed_at": None,
                "last_error": None,
                "step_details": {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "function": {
                                "name": function_name,
                                "arguments": arguments,
                                "output": None
                            },
                            "id": tool_call_id,
                            "type": "function"
                        }
                    ]
                },
                "usage": None
            }
        })
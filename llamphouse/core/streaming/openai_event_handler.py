import uuid
from .base_event_handler import BaseEventHandler

class OpenAIEventHandler(BaseEventHandler):
    def __init__(self, send_event, assistant_id, thread_id, run_id):
        super().__init__(send_event)
        self.started = False
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.content = ""
        self.run_step_id = None
        self.tool_call_name = None
        self.tool_call_id = None

    def handle_event(self, event):
        # Check if the event has a object that is not an empty string
        if (hasattr(event, 'object') and not event.object) or (hasattr(event, 'choices') and not event.choices):
            return
        
        # Process the other events
        # we will assume that each event has a maximum of one choice
        choice = event.choices[0]

        if choice.finish_reason:
            if choice.finish_reason == "stop":
                self.send_event_message_completed(self.assistant_id, self.thread_id, self.run_id, event.id, self.content)
            elif choice.finish_reason == "tool_calls":
                self.send_event_run_step_completed(
                    self.assistant_id,
                    self.thread_id,
                    self.run_id,
                    self.run_step_id,
                    self.tool_call_id,
                    self.tool_call_name,
                    self.content
                )
            else:
                print(f"OpenAIEventHandler: Unhandled finish reason: {choice.finish_reason}", flush=True)
            
            self.send_event_done(self.assistant_id, self.thread_id, self.run_id)
            
        
        elif choice.delta.tool_calls:
            if not self.started:
                self.started = True
                self.content = ""
                self.run_step_id = str(uuid.uuid4())
                self.tool_call_name = choice.delta.tool_calls[0].function.name
                self.tool_call_id = choice.delta.tool_calls[0].id
                self.send_event_run_step_tool_created(
                    self.assistant_id,
                    self.thread_id,
                    self.run_id,
                    self.run_step_id,
                    self.tool_call_id,
                    self.tool_call_name
                )
            
            if len(choice.delta.tool_calls[0].function.arguments) > 0:
                self.content += choice.delta.tool_calls[0].function.arguments
                self.send_event_run_step_tool_delta(
                    self.run_step_id,
                    None,
                    None,
                    choice.delta.tool_calls[0].function.arguments
                )

        elif choice.delta.content:
            if not self.started:
                self.started = True
                self.content = ""
                self.send_event_message_created(
                    self.assistant_id,
                    self.thread_id,
                    self.run_id,
                    event.id
                )
                self.send_event_message_in_progress(
                    self.assistant_id,
                    self.thread_id,
                    self.run_id,
                    event.id
                )

            if len(choice.delta.content) > 0:
                self.content += choice.delta.content
                self.send_event_message_delta(event.id, choice.delta.content)


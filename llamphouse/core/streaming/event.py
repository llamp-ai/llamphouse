import json

class Event:
    def __init__(self, event: str, data: str):
        self.event = event
        self.data = data

    def to_sse(self) -> str:
        return f"event: {self.event}\ndata: {self.data}\n\n"

# Define specific event classes for each event type

class DoneEvent(Event):
    """Occurs when a stream ends."""
    def __init__(self):
        super().__init__(event="done", data="[DONE]")

class ErrorEvent(Event):
    """Occurs when an error occurs. This can happen due to an internal server error or a timeout."""
    def __init__(self, data: dict):
        super().__init__(event="error", data=json.dumps(data))

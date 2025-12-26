from llamphouse.core import LLAMPHouse, Assistant
from dotenv import load_dotenv
from llamphouse.core.context import Context
from openai import OpenAI
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from typing import Any, Callable, Dict
from datetime import datetime, timezone

load_dotenv(override=True)

open_client = OpenAI()
SYSTEM_PROMPT = "You are a helpful assistant. When asked about time/date, always call the get_now tool."

def get_current_time(_: dict[str, Any] | None = None) -> str:
    """Returns the current UTC datetime as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return current datetime (UTC) as ISO-8601 string.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]

TOOL_REGISTRY: Dict[str, Callable] = {"get_current_time": get_current_time}

# Create a custom assistant
class CustomAssistant(Assistant):

    def __init__(self, assistant_id: str):
        super().__init__(id=assistant_id, tools=TOOL_SCHEMAS)

    async def run(self, context: Context):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in context.messages:
            if m.content and hasattr(m.content[0], "text") and m.content[0].text:
                messages.append({"role": m.role, "content": m.content[0].text})
        
        for _ in range(3):
            response = open_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            messages.append(message)

            # If the model wants to reply to user, stop here
            if not message.tool_calls:
                await context.insert_message(role="assistant", content=message.content)
                return

            # If the Model wants to use tools
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                if func_name in TOOL_REGISTRY:
                    result = TOOL_REGISTRY[func_name]()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

        # If loop exceeds limit without final answer
        await context.insert_message(
            role="assistant", 
            content="I'm sorry, I needed too many steps to process this request."
        )

def main():
    # Create an instance of the custom assistant
    my_assistant = CustomAssistant("my-assistant")

    # data store choice
    data_store = InMemoryDataStore() # PostgresDataStore() or InMemoryDataStore() for in-memory testing

    # Create a new LLAMPHouse instance
    llamphouse = LLAMPHouse(assistants=[my_assistant], data_store=data_store)
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

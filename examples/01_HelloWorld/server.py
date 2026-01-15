from llamphouse.core import LLAMPHouse, Assistant
from dotenv import load_dotenv
from llamphouse.core.context import Context
from openai import OpenAI
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.data_stores.retention import RetentionPolicy

load_dotenv(override=True)

open_client = OpenAI()

# Create a custom assistant
class CustomAssistant(Assistant):

    async def run(self, context: Context):
        # transform the assistant messages to chat messages
        messages = [{"role": message.role, "content": message.content[0].text} for message in context.messages]
        
        # send the messages to the OpenAI API
        result = open_client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini"
        )

        # add the assistant messages to the thread
        await context.insert_message(role="assistant", content=result.choices[0].message.content)

        # no need to return anything, the run will stop here

def main():
    # Create an instance of the custom assistant
    my_assistant = CustomAssistant("my-assistant")

    # data store choice
    data_store = InMemoryDataStore() # PostgresDataStore() or InMemoryDataStore() for in-memory testing

    # retention policy
    retention_policy = RetentionPolicy(ttl_days=30, interval_seconds=24*60*60, batch_size=100, dry_run=False, enabled=True)

    # Create a new LLAMPHouse instance
    llamphouse = LLAMPHouse(assistants=[my_assistant], data_store=data_store, retention_policy=retention_policy)
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

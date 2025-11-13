import os
from openai import OpenAI
import threading
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["POOL_SIZE"] = "5"

from sqlalchemy import create_engine
from llamphouse.core.database.models import Base  # Import your Base object

def setup_database():
    # Create an engine using the DATABASE_URL environment variable
    engine = create_engine(os.environ["DATABASE_URL"])
    # Create all tables based on the current models
    Base.metadata.create_all(engine)

from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.context import Context

setup_database()

open_client = OpenAI()

def test_add_assistant():
    class CustomAssistant(Assistant):
        def run(self, context: Context):
            pass

    my_assistant = CustomAssistant("my-assistant")
    llamphouse = LLAMPHouse(assistants=[my_assistant])

    # Start the server in a separate thread
    def start_server():
        llamphouse.ignite(host="127.0.0.1", port=8000)

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    assert "my-assistant" in [a.name for a in llamphouse.assistants]
    assert "my-assistant" in [a.id for a in llamphouse.assistants]
    assert llamphouse.assistants[0].tools == []
    assert llamphouse.assistants[0].description == None
    assert llamphouse.assistants[0].model == ""
    assert llamphouse.assistants[0].temperature == 0.7
    assert llamphouse.assistants[0].top_p == 1.0
    assert llamphouse.assistants[0].instructions == ""
    assert llamphouse.assistants[0].object == "assistant"

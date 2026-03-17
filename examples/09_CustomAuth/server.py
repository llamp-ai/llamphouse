from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import Request
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.auth.base_auth import BaseAuth, AuthResult
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter

load_dotenv(override=True)

openai_client = AsyncOpenAI()


class CustomAuth(BaseAuth):
    """Only accept requests whose Authorization header carries 'secret_key'.

    Routes under /compass/* are public and skip authentication.
    """

    def authenticate(self, request: Request) -> AuthResult:
        # Allow compass routes through without authentication
        if request.url.path.startswith("/compass"):
            return AuthResult(authenticated=True, identity={"user": "anonymous"})

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return AuthResult(message="Missing Authorization header.")

        token = auth_header.removeprefix("Bearer ")
        if token == "secret_key":
            return AuthResult(authenticated=True, identity={"user": "demo"})

        return AuthResult(message="Invalid API key.")


class CustomAgent(Agent):
    async def run(self, context: Context):
        # 1. Send a status update while we prepare
        context.emit("status", {"message": "Authenticated — generating response..."})

        # 2. Build conversation history
        messages = [
            {"role": message.role, "content": message.text}
            for message in context.messages
        ]

        # 3. Stream the response from OpenAI
        stream = await openai_client.chat.completions.create(
            messages=messages,
            model="gpt-5-mini",
            stream=True,
        )

        # 4. Pipe through LLAMPHouse — tokens forwarded to client in real time
        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)

        # 5. Persist the complete response
        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    agent = CustomAgent(
        id="my-assistant",
        name="Custom Auth Agent",
        description="A streaming assistant protected by a custom authenticator.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        adapters=[A2AAdapter()],
        authenticator=CustomAuth(),
        data_store=InMemoryDataStore(),
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()

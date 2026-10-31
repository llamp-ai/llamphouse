"""The concierge over three protocols: the OpenAI Responses (+ Conversations) and Assistants APIs with the
official `openai` client, and A2A v1.0 over JSON-RPC with httpx.

    python client.py                 # all three
    python client.py responses       # or: assistants, a2a

The server: `llamphouse up --agents agents.yaml --config llamphouse.yaml` (LLAMPHOUSE_URL points elsewhere
than :8000).
"""

import json
import os
import warnings
import sys
import time
import uuid

import httpx
from openai import OpenAI

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
KEY = os.environ.get("LLAMPHOUSE_API_KEY", "unused")  # sent as `Authorization: Bearer`; checked when auth is on
LOCAL_TIME = {"type": "function", "name": "local_time", "description": "The user's local time.",
              "parameters": {"type": "object", "properties": {}}}


def responses() -> None:
    print("== Responses API: a conversation is a thread, a response a stretch of a run")
    client = OpenAI(base_url=f"{URL}/v1", api_key=KEY)
    print("models:", [m.id for m in client.models.list()])  # the agents

    first = client.responses.create(model="concierge", input="Hi!")
    print(f"> Hi!\n{first.output_text}")  # the run asked: the response completes with the question
    second = client.responses.create(model="concierge", input="Ada", previous_response_id=first.id)
    print(f"> Ada\n{second.output_text}")  # the answer: the same run carries on

    conversation = client.conversations.create(metadata={"app": "adapters-example"})  # a new thread
    for text in ("Hello", "Grace"):
        said = client.responses.create(model="concierge", conversation=conversation.id, input=text).output_text
        print(f"> {text}\n{said}")
    asked = client.responses.create(model="concierge", conversation=conversation.id, input="What time is it?",
                                    tools=[LOCAL_TIME])  # the client's tools are the run's client tools
    call = next(o for o in asked.output if o.type == "function_call")
    print(f"> What time is it?\n[the run calls the client's tool: {call.name}({call.arguments})]")
    stream = client.responses.create(model="concierge", conversation=conversation.id, tools=[LOCAL_TIME], stream=True,
                                     input=[{"type": "function_call_output", "call_id": call.call_id,
                                             "output": time.strftime("%H:%M")}])
    for event in stream:  # the same run carries on, streamed
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    items = client.conversations.items.list(conversation.id, order="asc")
    print(f"\n[conversation {conversation.id}: {len(items.data)} items]\n")


def assistants() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # OpenAI's, for its own Assistants API
    print("== Assistants API: an assistant is an agent; a question is requires_action")
    client = OpenAI(base_url=f"{URL}/v1", api_key=KEY)
    print("assistants:", [a.id for a in client.beta.assistants.list()])
    thread = client.beta.threads.create(messages=[{"role": "user", "content": "Good morning"}])
    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id="concierge", poll_interval_ms=100)
    print(f"> Good morning\n[{run.status}: {run.required_action.type}]")  # wait_for_input: ctx.ask
    # the answer to its question goes to the run's signals, as "input" (not part of the openai client)
    httpx.post(f"{URL}/v1/threads/{thread.id}/runs/{run.id}/signals", json={"name": "input", "payload": "Alan"},
               headers={"Authorization": f"Bearer {KEY}"}).raise_for_status()
    run = client.beta.threads.runs.poll(run.id, thread_id=thread.id, poll_interval_ms=100)
    last = client.beta.threads.messages.list(thread_id=thread.id, limit=1).data[0]
    print(f"> Alan\n{last.content[0].text.value}\n[run {run.status}]\n")


def a2a() -> None:
    print("== A2A v1.0: a context is a thread, a task a run")
    with httpx.Client(base_url=URL, headers={"A2A-Version": "1.0", "Authorization": f"Bearer {KEY}"}) as http:
        card = http.get("/a2a/concierge/.well-known/agent-card.json").raise_for_status().json()
        print(f"card: {card['name']} - {card['description']}")
        print(f"      {json.dumps(card.get('supportedInterfaces') or card.get('url'))}")

        def send(text: str, **ids: str) -> dict:
            message = {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"text": text}], **ids}
            body = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage", "params": {"message": message}}
            reply = http.post("/a2a/concierge", json=body, timeout=30).raise_for_status().json()
            task = reply["result"]["task"]
            said = task["status"]["message"]["parts"][0]["text"]
            print(f"> {text}\n{said}\n[task {task['id']}: {task['status']['state']}]")
            return task

        task = send("Hi there")  # TASK_STATE_INPUT_REQUIRED: the question
        send("Lin", taskId=task["id"], contextId=task["contextId"])  # the answer, to that task
        send("Thanks!", contextId=task["contextId"])  # a new task in the same context: it remembers
    print()


def main() -> None:
    for name in sys.argv[1:] or ["responses", "assistants", "a2a"]:
        {"responses": responses, "assistants": assistants, "a2a": a2a}[name]()
    print(f"Every conversation above is a thread of the concierge: {URL}/compass")


if __name__ == "__main__":
    main()

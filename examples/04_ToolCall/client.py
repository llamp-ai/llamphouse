from openai import OpenAI

QUESTION = "What is the current date and time in UTC?"

def main():
    client = OpenAI(base_url="http://127.0.0.1:8000", api_key="secret_key")
    
    # Create new thread
    thread = client.beta.threads.create()
    print(f"Created thread: {thread.id}")

    client.beta.threads.messages.create(thread.id, role="user", content=QUESTION)

    print(f"User: {QUESTION}")

    client.beta.threads.runs.create_and_poll(assistant_id="my-assistant", thread_id=thread.id)

    messages = client.beta.threads.messages.list(thread.id)
    latest = next((m for m in messages.data if m.role == "assistant"), None)

    if latest and latest.content:
        print(latest.content[0].text)
    else:
        print("No assistant response.")

if __name__ == "__main__":
    main()
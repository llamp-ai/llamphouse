from openai import OpenAI

def main():
    client = OpenAI(api_key="123", base_url="http://127.0.0.1:8000")
    
    # List assistant in glowstick
    my_assistants = client.beta.assistants.list()
    # print(my_assistants)
    # # Create new thread and message in db
    empty_thread = client.beta.threads.create()
    print(f"Created thread: {empty_thread.id}")
    new_message = client.beta.threads.messages.create(empty_thread.id, role="user", content="How are you")
    print(f"Added message: {new_message.id}")

    # Create new run in specific thread & assistant
    print("Starting new run")
    new_run =client.beta.threads.runs.create_and_poll(assistant_id="Chatbot1", thread_id=empty_thread.id, temperature=1)
    print(f"Run finished: {new_run}")

if __name__ == "__main__":
    main()

from openai import OpenAI

def main():
    client = OpenAI(base_url="http://127.0.0.1:8000", api_key="secret_key")
    
    # Create new thread
    thread = client.beta.threads.create()
    print(f"Created thread: {thread.id}")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Add user message to thread
        client.beta.threads.messages.create(thread.id, role="user", content=user_input)

        # Run the thread
        client.beta.threads.runs.create_and_poll(assistant_id="my-assistant", thread_id=thread.id)

        # Get the messages in the thread
        messages = client.beta.threads.messages.list(thread.id)

        latest_assistant_message = []
        for message in messages.data:  # Check from most recent
            if message.role == "user":
                break
            else: 
                latest_assistant_message.insert(0, message.content[0].text)

        # Display the latest assistant message
        if latest_assistant_message:
            for msg in latest_assistant_message:
                print(f"Chatbot: {msg}")
        else:
            print("No assistant response yet.")

if __name__ == "__main__":
    main()
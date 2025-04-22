from llamphouse.core import LLAMPHouse, Assistant
from dotenv import load_dotenv
from llamphouse.core.context import Context
from openai import OpenAI

load_dotenv(override=True)

open_client = OpenAI()

class CustomAssistant(Assistant):
    def run(self, context: Context):
        messages = [{"role": message.role, "content": message.content[0].text} for message in context.messages]
        
        stream = open_client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            stream=True
        )
        context.handle_completion_stream(stream)


def main():
    # Create an instance of the custom assistant
    my_assistant = CustomAssistant("my-assistant")

    # Create a new LLAMPHouse instance
    llamphouse = LLAMPHouse(assistants=[my_assistant])
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()
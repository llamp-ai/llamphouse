from llamphouse.core import LLAMPHouse, Assistant
from dotenv import load_dotenv
from llamphouse.core.context import Context
from openai import OpenAI
from llamphouse.core.auth.base_auth import BaseAuth

load_dotenv(override=True)

open_client = OpenAI()

class CustomAuth(BaseAuth):
    def authenticate(self, api_key: str):
        if api_key == "secret_key":
            return True
        return False

# Create a custom assistant
class CustomAssistant(Assistant):

    def run(self, context: Context):
        # transform the assistant messages to chat messages
        messages = [{"role": message.role, "content": message.content[0].text} for message in context.messages]
        
        # send the messages to the OpenAI API
        result = open_client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini"
        )

        # add the assistant messages to the thread
        context.insert_message(role="assistant", content=result.choices[0].message.content)

        # no need to return anything, the run will stop here

def main():
    # Create an instance of the custom assistant
    my_assistant = CustomAssistant("my-assistant")

    # Create a new LLAMPHouse instance
    llamphouse = LLAMPHouse(assistants=[my_assistant], authenticator=CustomAuth())
    
    # Start the LLAMPHouse server
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

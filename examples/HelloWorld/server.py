from llamphouse.core import LLAMPHouse, Assistant
from typing import List, Optional
from dotenv import load_dotenv
from llamphouse.core.context import Context

load_dotenv(override=True)

class CustomAssistant(Assistant):
    def __init__(
        self, 
        name: str, 
        model: str, 
        temperature: float = 0.7, 
        top_p: float = 1.0, 
        instructions: Optional[str] = None,
        id: Optional[str] = None, 
        description: Optional[str] = None, 
        tools: Optional[List[str]] = None
    ):
        super().__init__(
            name=name, 
            model=model, 
            temperature=temperature, 
            top_p=top_p,
            instructions=instructions,
            id=id, 
            description=description, 
            tools=tools
        )

    def run(self, context: Context, *args, **kwargs):
        print(f"thread: {context.thread_id} ")
        print(f"assistant: {context.assistant_id} ")

assistant1 = CustomAssistant(
    id="Chatbot1",
    name="ChatBot", 
    model="gpt-3.5-turbo", 
    description="A general-purpose chatbot", 
    temperature=0.8, 
    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
    top_p=0.9, 
    # tools=["text-generation"]
)
assistant2 = CustomAssistant(
    id="HelpBot1",
    name="HelpBot", 
    model="gpt-4", 
    description="A help desk assistant", 
    temperature=0.7, 
    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
    top_p=1.0, 
    # tools=["text-generation", "code_interpreter"]
)
assistant3 = CustomAssistant(
    id="CustomerBot1",
    name="CustomerBot", 
    model="gpt-4", 
    description="A help desk assistant", 
    temperature=0.7, 
    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
    top_p=1.0, 
    # tools=["text-generation", "code_interpreter"]
)

def main():
    llamphouse = LLAMPHouse(assistants=[assistant1, assistant2, assistant3], api_key="123")
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()

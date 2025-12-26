from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from dotenv import load_dotenv

load_dotenv(override=True)

def main():
    client = OpenAI(api_key="secret_key", base_url="http://127.0.0.1:8000")
    
    thread = client.beta.threads.create()
    print(f"Created thread: {thread.id}")

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `x + 1 = 4`. Can you help me?"
        )

    class EventHandler(AssistantEventHandler): 
        # @override
        # def on_event(self, event) -> None:
        #     print(event.event, flush=True)

        # @override
        # def on_message_done(self, message):
        #     print(message, flush=True)

        # @override
        # def on_run_step_done(self, run_step):
        #     print(run_step, flush=True)

        # @override
        # def on_text_created(self, text) -> None:
        #     print("\nassistant > ", end="", flush=True)
            
        @override
        def on_text_delta(self, delta, snapshot):
            print(delta.value, end="", flush=True)

        @override
        def on_message_done(self, message):
            print()    
            
        # def on_tool_call_created(self, tool_call):
        #     print(f"\nassistant > {tool_call.type}\n", flush=True)
        
        # def on_tool_call_delta(self, delta, snapshot):
        #     if delta.type == 'code_interpreter':
        #         if delta.code_interpreter.input:
        #             print(delta.code_interpreter.input, end="", flush=True)
        #         if delta.code_interpreter.outputs:
        #             print(f"\n\noutput >", flush=True)
        #             for output in delta.code_interpreter.outputs:
        #                 if output.type == "logs":
        #                     print(f"\n{output.logs}", flush=True)

    try:
        with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id="my-assistant",
            instructions="Please address the user as Jane Doe.",
            event_handler=EventHandler(),
            ) as stream:
            print("Starting stream...")
            stream.until_done()
    except Exception as e:
        print("Error run stream:", e)

if __name__ == "__main__":
    main()
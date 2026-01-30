from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from opentelemetry import propagate
from opentelemetry.trace import Status, StatusCode
from dotenv import load_dotenv
from llamphouse.core.tracing import setup_tracing, get_tracer, span_context

load_dotenv(override=True)
setup_tracing()
tracer = get_tracer("llamphouse.client")

def main():
    with span_context(tracer, "llamphouse.client.run") as span:
        carrier = {}
        propagate.inject(carrier)
        
        with tracer.start_as_current_span("llamphouse.client.openai_request"):
            client = OpenAI(
                api_key="secret_key",
                base_url="http://127.0.0.1:8000",
                default_headers=carrier,
            )

        # with tracer.start_as_current_span("llamphouse.client.thread_create"):
        #     carrier = {}
        #     propagate.inject(carrier)
        thread = client.beta.threads.create()

        span.set_attribute("session.id", thread.id)
        print(f"Created thread: {thread.id}")

        user_text = "I need to solve the equation `x + 1 = 4`. Can you help me?"
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_text,
            )
        span.set_attribute("gen_ai.request.model", "gpt-4o-mini")
        span.set_attribute("input.value", user_text)

        chunks = []

        class EventHandler(AssistantEventHandler): 
                
            @override
            def on_text_delta(self, delta, snapshot):
                chunks.append(delta.value)
                print(delta.value, end="", flush=True)

            @override
            def on_message_done(self, message):
                print()    

        try:
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id="my-assistant",
                instructions="Please address the user as Jane Doe.",
                event_handler=EventHandler(),
            ) as stream:
                print("Starting stream...")
                stream.until_done()
            
            final_text = "".join(chunks)
            span.set_attribute("output.value", final_text)
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            print("Error run stream:", e)

if __name__ == "__main__":
    main()
import asyncio
import logging
import inspect
import json
from typing import Optional, Sequence, Tuple
from opentelemetry import propagate, context as otel_context
from opentelemetry.trace import Status, StatusCode
from ..tracing import get_tracer, span_context
from ..types.enum import run_status, event_type
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..context import Context
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..streaming.event import DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore
from ..queue.base_queue import BaseQueue
from ..queue.types import QueueMessage
from ..queue.exceptions import QueueRateLimitError, QueueRetryExceeded

logger = logging.getLogger("llamphouse.worker")
tracer = get_tracer("llamphouse.worker")

class AsyncWorker(BaseWorker):
    def __init__(self, time_out: float = 30.0):

        self.time_out = time_out
        self.task: Optional[asyncio.Task] = None
        self._running = True

    async def process_run_queue(self, data_store: BaseDataStore, run_queue: BaseQueue, assistants: Sequence[Assistant], fastapi_state):
        assistant_ids = [assistant.id for assistant in assistants] or None

        while self._running:
            try:
                run_object = None
                output_queue = None
                token = None
                span = None
                item: Optional[Tuple[str, QueueMessage]] = await run_queue.dequeue(assistant_ids=assistant_ids, timeout=None)
                if not item:
                    continue
                    
                receipt, message = item
                ctx = otel_context.get_current()
                carrier = (message.metadata or {}).get("traceparent", {})
                if carrier:
                    ctx = propagate.extract(carrier)
                    token = otel_context.attach(ctx)

                run_id, thread_id, assistant_id = message.run_id, message.thread_id, message.assistant_id
                with span_context(
                    tracer,
                    "llamphouse.worker.run",
                    context=ctx, 
                    attributes={
                        "run.id": run_id,
                        "assistant.id": assistant_id,
                        "queue.attempt": message.attempts,
                        "gen_ai.system": "llamphouse",
                    },
                ) as span:
                    try:
                        input_payload = {
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "assistant_id": assistant_id,
                            "attempt": message.attempts,
                        }
                        span.set_attribute("input.value", json.dumps(input_payload, ensure_ascii=True, default=str))

                        # Resolve assistant
                        assistant = next((a for a in assistants if a.id == assistant_id), None)
                        if not assistant:
                            await run_queue.ack(receipt)
                            span.set_status(Status(StatusCode.ERROR))
                            span.add_event("assistant.not_found")
                            span.set_attribute(
                                "output.value",
                                json.dumps({"status": "failed", "reason": "assistant.not_found"}, ensure_ascii=True),
                            )
                            logger.error("Assistant %s not found for run %s", assistant_id, run_id)
                            if thread_id and run_id:
                                await data_store.update_run_status(thread_id, run_id, run_status.FAILED, {
                                    "code": "server_error", "message": "Assistant not found"
                                })
                            continue

                        # Resolve event queue (if streaming)
                        task_key = f"{assistant_id}:{thread_id}"
                        output_queue = fastapi_state.event_queues.get(task_key)

                        # Mark IN_PROGRESS 
                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.IN_PROGRESS)

                        # Build context
                        run_object = await data_store.get_run_by_id(thread_id, run_id)
                        if not run_object:
                            span.set_status(Status(StatusCode.ERROR))
                            span.add_event("run.not_found")
                            span.set_attribute(
                                "output.value",
                                json.dumps({"status": "failed", "reason": "run.not_found"}, ensure_ascii=True),
                            )
                            await run_queue.ack(receipt)
                            continue

                        span.set_attribute("gen_ai.operation.name", "chat")
                        span.set_attribute("session.id", thread_id)
                        span.set_attribute("run.id", run_id)
                        span.set_attribute("assistant.id", assistant_id)

                        if run_object.model:
                            span.set_attribute("gen_ai.request.model", run_object.model)
                        if run_object.temperature is not None:
                            span.set_attribute("gen_ai.request.temperature", float(run_object.temperature))
                        if run_object.top_p is not None:
                            span.set_attribute("gen_ai.request.top_p", float(run_object.top_p))
                        if run_object.max_prompt_tokens is not None:
                            span.set_attribute("gen_ai.request.max_prompt_tokens", int(run_object.max_prompt_tokens))
                        if run_object.max_completion_tokens is not None:
                            span.set_attribute("gen_ai.request.max_completion_tokens", int(run_object.max_completion_tokens))
                        if run_object.tool_choice is not None:
                            span.set_attribute("gen_ai.request.tool_choice", str(run_object.tool_choice))
                        if run_object.parallel_tool_calls is not None:
                            span.set_attribute("gen_ai.request.parallel_tool_calls", bool(run_object.parallel_tool_calls))
                        if run_object.response_format is not None:
                            span.set_attribute("gen_ai.request.response_format", str(run_object.response_format))

                        if output_queue:
                            await output_queue.add(run_object.to_event(event_type.RUN_IN_PROGRESS))

                        context = await Context.create(
                            assistant=assistant,
                            assistant_id=assistant_id,
                            run_id=run_id,
                            run=run_object,
                            thread_id=thread_id,
                            queue=output_queue,
                            data_store=data_store,
                            loop=asyncio.get_running_loop(),
                            traceparent=carrier,
                        )

                        if inspect.iscoroutinefunction(assistant.run):
                            await asyncio.wait_for(assistant.run(context), timeout=self.time_out)
                        else:
                            await asyncio.wait_for(asyncio.to_thread(assistant.run, context), timeout=self.time_out)
                        
                        output_payload = {"status": "completed", "run_id": run_id}
                        span.set_attribute("output.value", json.dumps(output_payload, ensure_ascii=True, default=str))
                        if run_object and run_object.model:
                            span.set_attribute("gen_ai.response.model", run_object.model)
                        span.set_attribute("gen_ai.response.status", "completed")
                        span.set_status(Status(StatusCode.OK))

                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.COMPLETED)
                        if output_queue:
                            run_object = await data_store.get_run_by_id(thread_id, run_id)
                            if run_object:
                                await output_queue.add(run_object.to_event(event_type.RUN_COMPLETED))
                                await output_queue.add(DoneEvent())
                        await run_queue.ack(receipt)

                    except QueueRateLimitError as e:
                        error = {"code": "rate_limit_exceeded", "message": str(e)}
                        span.record_exception(e)
                        span.add_event("queue.rate_limit_exceeded", {
                            "error.code": error["code"],
                            "error.message": error["message"],
                        })
                        output_payload = {"status": "failed", "error": error}
                        span.set_attribute("output.value", json.dumps(output_payload, ensure_ascii=True))
                        span.set_attribute("gen_ai.response.status", "failed")
                        span.set_status(Status(StatusCode.ERROR))
                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)
                        await run_queue.ack(receipt)

                    except QueueRetryExceeded as e:
                        error = {"code": "max_retry_exceeded", "message": str(e)}
                        span.record_exception(e)
                        span.add_event("queue.retry_exceeded", {
                            "error.code": error["code"],
                            "error.message": error["message"],
                        })
                        output_payload = {"status": "failed", "error": error}
                        span.set_attribute("output.value", json.dumps(output_payload, ensure_ascii=True))
                        span.set_attribute("gen_ai.response.status", "failed")
                        span.set_status(Status(StatusCode.ERROR))
                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)                    
                        await run_queue.ack(receipt)

                    except asyncio.TimeoutError as e:
                        error = {"code": "server_error", "message": "Run timeout"}
                        span.record_exception(e)
                        span.add_event("queue.timeout", {
                            "error.code": error["code"],
                            "error.message": error["message"],
                            "timeout_s": self.time_out,
                        })
                        output_payload = {"status": "expired", "error": error}
                        span.set_attribute("output.value", json.dumps(output_payload, ensure_ascii=True))
                        span.set_attribute("gen_ai.response.status", "expired")
                        span.set_status(Status(StatusCode.ERROR))
                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.EXPIRED, error)
                        if output_queue and run_object:
                            await output_queue.add(run_object.to_event(event_type.RUN_EXPIRED))
                            await output_queue.add(ErrorEvent(error))
                        await run_queue.ack(receipt)

                    except Exception as e:
                        error = {"code": "server_error", "message": str(e)}
                        span.record_exception(e)
                        span.add_event("worker.exception", {
                            "error.code": error["code"],
                            "error.message": error["message"],
                        })
                        output_payload = {"status": "failed", "error": error}
                        span.set_attribute("output.value", json.dumps(output_payload, ensure_ascii=True))
                        span.set_attribute("gen_ai.response.status", "failed")
                        span.set_status(Status(StatusCode.ERROR))
                        if thread_id and run_id:
                            await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)
                        if output_queue and run_object:
                            await output_queue.add(run_object.to_event(event_type.RUN_FAILED))
                            await output_queue.add(ErrorEvent(error))
                        if message.attempts < run_queue.retry_policy.max_attempts:
                            await run_queue.requeue(receipt, message)
                        else:
                            await run_queue.ack(receipt)
                        logger.exception("Error executing run %s", run_id)
            
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in process_run_queue loop")
                await asyncio.sleep(1.0)                   
            finally:
                if token is not None:
                    otel_context.detach(token)              

    def start(self, data_store: BaseDataStore, run_queue: BaseQueue, **kwargs):
        logger.info("Starting async worker...")
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})
        self.loop = kwargs.get("loop")
        if not self.loop:
            raise ValueError("loop is required")
        
        self.task = self.loop.create_task(
            self.process_run_queue(
                data_store=data_store,
                run_queue=run_queue,
                assistants=self.assistants,
                fastapi_state=self.fastapi_state,
            )
        )

    def stop(self):
        logger.info("Stopping async worker...")
        self._running = False
        if self.task:
            self.task.cancel()
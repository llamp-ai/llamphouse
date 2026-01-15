import asyncio
import uuid
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.run_step import (
    CreateRunStepRequest,
    MessageCreation,
    MessageCreationStepDetails,
)
from llamphouse.core.database.models import Thread, Message, Run, RunStep


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


async def main() -> None:
    load_dotenv()
    store = PostgresDataStore()
    try:
        assistant = AssistantObject(
            id=_uid("asst"),
            model="gpt-4",
            created_at=datetime.now(timezone.utc),
            instructions="seed",
            metadata={},
        )

        for i in range(3):
            thread = await store.insert_thread(
                CreateThreadRequest(
                    metadata={"seed": True, "idx": i},
                    tool_resources={},
                    messages=[],
                )
            )

            msg = await store.insert_message(
                thread.id,
                CreateMessageRequest(
                    role="user",
                    content=f"seed message {i}",
                    metadata={"message_id": _uid("msg")},
                ),
            )

            run = await store.insert_run(
                thread.id,
                RunCreateRequest(
                    assistant_id=assistant.id,
                    metadata={"run_id": _uid("run")},
                ),
                assistant,
            )

            step_req = CreateRunStepRequest(
                assistant_id=assistant.id,
                metadata={"step_id": _uid("step")},
                step_details=MessageCreationStepDetails(
                    type="message_creation",
                    message_creation=MessageCreation(message_id=msg.id),
                ),
            )
            step = await store.insert_run_step(thread.id, run.id, step_req)

            # Make records look old so dry_run will detect them.
            old_ts = datetime.now(timezone.utc) - timedelta(days=365 * 5)
            store.session.query(Thread).filter(Thread.id == thread.id).update({"created_at": old_ts})
            store.session.query(Message).filter(Message.id == msg.id).update({"created_at": old_ts})
            store.session.query(Run).filter(Run.id == run.id).update({"created_at": old_ts})
            if step is not None:
                store.session.query(RunStep).filter(RunStep.id == step.id).update({"created_at": old_ts})
            store.session.commit()

            print(f"seeded old data: thread={thread.id} message={msg.id} run={run.id}")

    finally:
        session = getattr(store, "session", None)
        if session is not None:
            session.close()


if __name__ == "__main__":
    asyncio.run(main())

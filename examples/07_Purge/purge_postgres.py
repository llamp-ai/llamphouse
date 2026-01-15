import asyncio

from dotenv import load_dotenv

import seed_postgres
from llamphouse.core.data_stores.postgres_store import PostgresDataStore
from llamphouse.core.data_stores.retention import RetentionPolicy

YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

async def run_purge(store: PostgresDataStore, dry_run: bool, ttl_days: int, batch_size: int | None) -> None:
    policy = RetentionPolicy(
            ttl_days=ttl_days,    
            dry_run=dry_run,       
            batch_size=batch_size,
            log_fn=print,
    )
    stats = await store.purge_expired(policy)
    if dry_run:
        mode = f"{YELLOW}DRY_RUN{RESET}"
    else:
        mode = f"{RED}DELETE{RESET}"
    print(f"{mode} stats: {stats}")


async def main() -> None:
    load_dotenv()

    # Mock old data using seed_postgres.
    await seed_postgres.main()

    store = PostgresDataStore()
    try:
        await run_purge(store, dry_run=True, ttl_days=365 * 3, batch_size=500)
        await run_purge(store, dry_run=False, ttl_days=365 * 3, batch_size=500)
        await run_purge(store, dry_run=True, ttl_days=365 * 3, batch_size=500)
    finally:
        session = getattr(store, "session", None)
        if session is not None:
            session.close()


if __name__ == "__main__":
    asyncio.run(main())

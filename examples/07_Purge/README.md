# Postgres Purge Example

This example demonstrates how to purge old data from a Postgres data store **outside** the LLAMPHouse server using the retention policy.

## Prerequisites

- Python 3.10+
- PostgreSQL running
- `DATABASE_URL` configured

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Create your `.env` from `.env.sample`:

   ```bash
   cp .env.sample .env
   ```
3. Run migrations (from repo root that contains `migrations/`):

   ```bash
   cd ../..
   alembic upgrade head
   cd examples/07_Purge
   ```

## Purge Flow (dry-run → delete → verify)

`purge_postgres.py` will:

1) seed old data,
2) run dry‑run (count only),
3) delete expired data,
4) run dry‑run again to confirm.

   ```bash
   python purge_postgres.py
   ```

## Adjust Retention Settings

Inside `purge_postgres.py`:

- `ttl_days`: how long to keep data
- `dry_run`: `True` = count only, `False` = delete
- `batch_size`: delete in batches (`None` = no limit)

  ```python
  RetentionPolicy(
      ttl_days=365 * 3,
      dry_run=True,
      batch_size=500,
  )
  ```

## Notes

- This example uses `PostgresDataStore` directly (no server required).
- For production, you can disable the seed step and just run purge on a schedule.

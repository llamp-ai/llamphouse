# Tests

This directory contains unit, contract, integration, and optional test suites for LLAMPHouse.

## Layout

- tests/unit
  Fast, deterministic unit tests (models, queues, auth middleware).
- tests/contract
  Contract tests for shared interfaces (data_store backends).
- tests/integration
  API and streaming integration tests.
  - tests/integration/api
  - tests/integration/streaming
- tests/optional
  Optional suites (e2e, schemathesis).

## Requirements

- Use the project-supported Python version.
- Install optional test dependencies:
  - pip install -r tests/requirements.txt
- httpx is used for in-process ASGI tests (skipped if missing).

## Environment

- DATABASE_URL
  Enables postgres-backed contract tests when set to a Postgres URL.
  Example: tests/.env.sample

Note: tests/conftest.py calls load_dotenv() with default settings.
Put a .env in the repo root or export variables in your shell.

## Running

From the repo root (llamphouse/):

- Run everything:

  ```bash
  python -m pytest
  ```
- Run by suite:

  ```bash
  python -m pytest tests/unit
  python -m pytest tests/contract
  python -m pytest tests/integration
  python -m pytest tests/integration/streaming
  python -m pytest tests/optionalsh
  ```
- Run by marker (see pytest.ini):

  ```bash
  python -m pytest -m unit
  python -m pytest -m contract
  python -m pytest -m integration
  python -m pytest -m streaming
  ```

If you add custom markers (e.g., postgres, e2e, optional), register them in pytest.ini to avoid warnings.

## Postgres contract tests

- Set DATABASE_URL to a valid Postgres connection string.
- Postgres tests are skipped automatically when `DATABASE_URL` is missing or not Postgres.

## Integration server

Integration API tests start a local LLAMPHouse server on `127.0.0.1:8085` (see tests/conftest.py).
Make sure the port is available, or update it in conftest.py.

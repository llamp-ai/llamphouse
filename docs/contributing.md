# Contributing

Contributions are welcome! Whether it's bug fixes, new features, documentation improvements, or example additions — we appreciate your help.

## Getting started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/llamphouse.git
cd llamphouse
```

### 2. Set up the development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Run the tests

```bash
# Run all tests (unit + contract + integration)
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/unit/ -v
uv run pytest tests/contract/ -v
uv run pytest tests/integration/ -v

# Postgres-only tests (requires DATABASE_URL and migrated schema)
LLAMPHOUSE_TRACING_ENABLED=false uv run pytest -m postgres

# Data-store contract parity tests
LLAMPHOUSE_TRACING_ENABLED=false uv run pytest tests/contract/data_store -q
```

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes

3. Run the test suite to ensure nothing is broken:
   ```bash
   uv run pytest tests/ -v
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "Add amazing feature"
   ```

5. Push and open a pull request:
   ```bash
   git push origin feature/amazing-feature
   ```

## Project structure

```
llamphouse/
├── llamphouse/llamphouse/     # Core package
│   ├── core/
│   │   ├── adapters/          # Protocol adapters (A2A, Assistants API, Compass)
│   │   ├── auth/              # Authentication
│   │   ├── config_stores/     # Config store backends
│   │   ├── data_stores/       # Data store backends (in-memory, Postgres)
│   │   ├── queues/            # Run queue backends
│   │   ├── streaming/         # Streaming infrastructure
│   │   ├── tracing/           # OpenTelemetry tracing
│   │   ├── types/             # Type definitions
│   │   ├── assistant.py       # Agent base class
│   │   ├── context.py         # Context object
│   │   └── llamphouse.py      # Main LLAMPHouse class
│   └── spotlight/             # Compass dashboard frontend
├── migrations/                # Alembic database migrations
├── tests/                     # Test suite
├── examples/                  # Runnable examples
├── docs/                      # Documentation (this site)
└── docker/                    # Docker Compose configs
```

## Database migrations

If your change modifies the database schema:

```bash
# Start a local Postgres
docker run --rm -d --name postgres \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=llamphouse \
  -p 5432:5432 postgres

# Set connection string
export DATABASE_URL=postgresql://postgres:password@localhost:5432/llamphouse

# If port 5432 is already used locally, map another host port instead:
docker run --rm -d --name postgres-5433 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=llamphouse \
  -p 5433:5432 postgres
export DATABASE_URL=postgresql://postgres:password@localhost:5433/llamphouse

# Create a new migration
uv run alembic revision --autogenerate -m "description of change"

# Apply migrations
uv run alembic upgrade head
```

When changing data-store behavior, keep `InMemoryDataStore` and
`PostgresDataStore` interchangeable. Add or update contract tests under
`tests/contract/data_store` for any field or helper method that affects
threads, messages, runs, run steps, attachments, pagination, lifecycle
timestamps, `stream`, `provider_config`, `config_values`, or `usage`.

## Building

```bash
uv run python -m build
```

## Code style

- Use type hints where practical
- Follow existing patterns in the codebase
- Keep agent logic simple and focused
- Write tests for new features

## Reporting issues

- Use the [GitHub Issues](https://github.com/llamp-ai/llamphouse/issues) page
- Tag bugs with `bug` and feature requests with `enhancement`
- Include reproduction steps and expected vs. actual behavior

## Contact

Project Admin: Pieter van der Deen — [pieter@stack-wise.co.uk](mailto:pieter@stack-wise.co.uk)

## License

See [LICENSE](https://github.com/llamp-ai/llamphouse/blob/main/LICENSE) for details.

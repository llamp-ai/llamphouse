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
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/contract/ -v
python -m pytest tests/integration/ -v

# Postgres-only tests (requires DATABASE_URL)
python -m pytest -m postgres
```

## Development workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. Make your changes

3. Run the test suite to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
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
  -p 5432:5432 postgres
docker exec -it postgres psql -U postgres -c 'CREATE DATABASE llamphouse;'

# Set connection string
export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/llamphouse

# Create a new migration
alembic revision --autogenerate -m "description of change"

# Apply migrations
alembic upgrade head
```

## Building

```bash
python -m build
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

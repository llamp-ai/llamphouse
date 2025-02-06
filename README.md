> [!NOTE]  
> This package is still under development. Always use the latest version for better stability.

# Introduction 

When building production grade LLM apps, there are many frameworks available to work something out in Python code. However, how can we deploy this code? When looking for tutorials, most people suggest using FastAPI to deploy a server with an endpoint. But in this step, scaling can become an issue. That is where LLAMPHouse comes in.

![stack](docs/img/stack.png)

This packages creates a reliable self-hosted server that mimics the OpenAI Assistant behavior. However, you can fully customize the run behavior yourself using your favorite framework.

![assistant API](docs/img/assistant_api.png)

## Installation

### Local
1. Clone the repository
1. Install the library `pip install .`

## Build
This is only required if you want to push the package to PyPi.
1. `python setup.py sdist bdist_wheel`
1. `git tag -a v1.0.0 -m "Release version 1.0.0"`
1. `git push`

## Testing
1. Build/Install the latest solution locally
1. Run the test: `pytest`

## Database

To run a local database:
1. `docker run --rm -d --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 postgres`
1. `docker exec -it postgres psql -U postgres -c 'CREATE DATABASE llamphouse;'`

To create a new database revision: `alembic revision --autogenerate -m "Added account table"`

To upgrade the database with the latest revision: `alembic upgrade head`

To downgrade back to the base version: `alembic downgrade base`

## Included API endpoints

- Assistants
    - ~~Create~~  ->  created in code
    - [x] List
    - [x] Retrieve
    - ~~Modify~~  ->  only in code
    - ~~Delete~~  ->  only in code
- Threads
    - [x] Create
    - [x] Retrieve
    - [x] Modify
    - [x] Delete
- Messages
    - [x] Create
    - [x] List
    - [x] Retrieve
    - [x] Modify
    - [x] Delete
- Runs
    - [x] Create
    - [x] Create thread and run
    - [x] List
    - [x] Retrieve
    - [x] Modify
    - [x] Submit tool outputs
    - [x] Cancel
- Run steps
    - [x] List
    - [x] Retrieve
- Vector stores
    - [ ] Create  ->  depends on implementation
    - [ ] List
    - [ ] Retrieve
    - [ ] Modify
    - [ ] Delete  ->  depends on implementation
- Vector store files
    - [ ] Create
    - [ ] List
    - [ ] Retrieve
    - [ ] Delete
- Vector store file batches
    - [ ] Create
    - [ ] Retrieve
    - [ ] Cancel
    - [ ] List
- Streaming
    - [ ] Message delta
    - [ ] Run step object
    - [ ] Assistant stream
    
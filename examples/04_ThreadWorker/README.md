# Custom Authenticator Example

This example demonstrates how to use the Thread Worker to manage and execute runs concurrently in the server. The Thread Worker allows you to offload runs to separate threads, improving the performance and responsiveness of your application by taking advantage of multi-threading capabilities.

By default, an Async Worker is used to handle the runs. This is using AsyncIO tasks to handle the runs. Depending on your use case, you can switch between both workers.

## Prerequisites

- Python 3.x installed
- PostgreSQL server with a database

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/llamp-ai/llamphouse.git
    cd llamphouse/examples/04_ThreadWorker
    ```

2. Install any required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up the environment variables by creating a [.env](.env.sample) file with your keys.

4. Run Alembic migrations on the database:
    ```sh
    alembic upgrade head
    ```

## Running the Server

1. Navigate to the example directory:
    ```sh
    cd llamphouse/examples/04_ThreadWorker
    ```

2. Start the server:
    ```sh
    python server.py
    ```

## Running the Client

1. Open a new terminal and navigate to the example directory:
    ```sh
    cd llamphouse/examples/04_ThreadWorker
    ```

2. Run the client:
    ```sh
    python client.py
    ```
    
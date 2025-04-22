# Custom Authenticator Example

This example demonstrates how to use LLamphouse as a local server that supports OpenAI's Assistant API streaming. It allows user to send a message to an assistant and receive real-time, incremental responses—just like the official OpenAI streaming.

## Prerequisites

- Python 3.x installed
- PostgreSQL server with a database

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/llamp-ai/llamphouse.git
    cd llamphouse/examples/05_Streaming
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
    cd llamphouse/examples/05_Streaming
    ```

2. Start the server:
    ```sh
    python server.py
    ```

## Running the Client

1. Open a new terminal and navigate to the example directory:
    ```sh
    cd llamphouse/examples/05_Streaming
    ```

2. Run the client:
    ```sh
    python client.py
    ```
    
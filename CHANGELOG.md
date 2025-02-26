# Changelog

## [Unreleased]

- Add tracing and monitoring
- Streaming
- LangChain utilities
- LangGraph utilities
- Cleanup error messages
- Cleanup db object to OpenAI object (see types/message.py from_db_message)
- Fix completed_at, failed_at, expired_at, ... times on the run
- Fix fastapi deprecated events
- Easier setting and getting metadata values

## [0.0.7] - 19/02/2025

- Change order messages in context (from desc to asc)
- Add worker as object in server init
- Add authenticator as object in server init
- Add ability to create custom authentication verification
- Make Assistant init with only a unique name
- Change context create_message to insert_message
- Update and add examples

## [0.0.6] - 03/02/2025

- Implement Graceful shutdown for both worker types.
- Implement pool size customization by env variable and check with maximum available from database connection
- Enhanced `Context` class with the ability to create new tool call step.
- Add remaining runs endpoints: Submit tool output, Cancel run.
- Add remaining run_steps endpoints: List run_step, Retrieve run_step.

## [0.0.5] - 29/01/2025

- Fixed issue with created_at field causing incorrect ordering by using a separate session for each FastAPI request.
- Ensured sessions are properly closed after each request.
- Moved session maker to the class initialization part to manage sessions more effectively.

## [0.0.4] - 23/01/2025

- Enhanced `Context` class with the ability to update thread, messages, and run details.
- Create DatabaseManagement class to handle database interact function
- Enable thread worker count customization
- Enable worker's task timeout customization

## [0.0.3] - 14/01/2025

- Generate new thread, message or run id based on metadata

## [0.0.2] - 13/01/2025

- Add api key authentication
- Fix metadata type declaration

## [0.0.1] - 30/12/2024

- Add initial API server
- Create threads and messages

## [0.0.0] - 16/12/2024

- Start of the project: December 16
"""The file store: files' bytes, by reference from messages (a stored message never holds file bytes).

    base.py       BaseFileStore, FileRecord and the errors
    memory.py     InMemoryFileStore - in this process
    postgres.py   PostgresFileStore - records and bytes in Postgres (object storage later, same contract)
    service.py    FileService - what the engine uses: put, claim, clean-up, with the file.* system events

The runtime file names one: `file_store: {postgres: {dsn: ...}}`, or your own `{my_pkg:S3Store: {...}}`.
"""

from .base import BaseFileStore, FileAccessError, FileNotFound, FileRecord, FileUnavailable, file_id
from .memory import InMemoryFileStore
from .service import FileService

__all__ = ["BaseFileStore", "InMemoryFileStore", "FileService", "FileRecord", "FileNotFound", "FileAccessError",
           "FileUnavailable", "file_id"]

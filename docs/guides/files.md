# Files

Documents, images, audio: files travel in messages as `FilePart`s, by reference. A message never holds the bytes
themselves: they live in the **file store**, once per content (by `sha256`), and the message holds a `files://`
reference with the file's facts (`mime_type`, `name`, `size`, `sha256`).

## In agent code

### Sending a file

Give a `FilePart` its bytes in `data`; the runtime stores them before the message is written, fills in `uri`,
`size` and `sha256`, and drops `data`:

```python
from llamphouse.sdk import Agent, FilePart


class Reporter(Agent):
    """Writes a report."""

    async def run(self, ctx) -> None:
        csv = "city,per_day\nLisbon,140\nBangkok,95\n"
        await ctx.send("Here is the comparison.",
                       FilePart(mime_type="text/csv", name="comparison.csv", data=csv.encode()))
```

The same works inside an `ArtifactPart` (a named result, with parts) and in tool results. A `FilePart` with an
outside URL (`uri="https://..."`) stays as it is: the runtime doesn't fetch it.

### Reading files

The `Files` capability reads the bytes of a file part:

```python
from llamphouse.sdk import Agent, FilePart, Files, requires


@requires(Files)
class Reader(Agent):
    """Reads the files it gets."""

    async def run(self, ctx) -> None:
        files = ctx.require(Files)
        sizes = []
        for part in ctx.input.parts:
            if isinstance(part, FilePart):
                data = await files.get(part)
                sizes.append(f"{part.name}: {len(data)} bytes")
        await ctx.send("; ".join(sizes) or "no files")
```

`Files.get(part)` reads a file the run may read: one on its thread (or on the thread a fork starts from), on a
thread its ancestors or it created, or one it stored itself. Anything else raises `PermissionError`.

`Files.put(data, mime_type=..., name=...)` stores bytes and returns the part, without sending it yet (to put
in a message later, or in state). It is recorded like a step: a run that executes again gets the same reference
back. The file belongs to nobody until a message references it.

`@requires(Files)` makes the runtime refuse to start the agent where there is no file store; without it,
`ctx.capability(Files)` returns `None` where files aren't offered. The agent file's `permissions.files` limits
an instance: `read` (get only), `none` (no `Files` at all), or `read-write` (the default).

## Over HTTP

Two ways to get a file into a message:

**Upload first, reference later**, for large files or to upload while the user is still typing:

```bash
curl -s 'localhost:8000/v2/files?name=notes.txt' -H 'content-type: text/plain' --data-binary @notes.txt
# {"kind": "file", "mime_type": "text/plain", "name": "notes.txt", "uri": "files://f_0f33d50856e44c2f",
#  "size": 11, "sha256": "12998c01...", "metadata": {}, "data": null}

curl -s localhost:8000/v2/threads/$THREAD/runs -H 'content-type: application/json' -d '{
  "parts": [{"kind": "text", "text": "Summarize this"},
            {"kind": "file", "mime_type": "text/plain", "name": "notes.txt", "uri": "files://f_0f33d50856e44c2f"}]}'
```

**Inline**, the bytes in the message, base64-encoded:

```json
{"parts": [{"kind": "file", "mime_type": "text/plain", "name": "b.txt", "data": {"$bytes": "YWJj"}}]}
```

Either way the message is written with a reference. Adapters do the same for their protocols' files (A2A `raw`
parts, for one).

**Downloading**: `GET /v2/files/{id}` returns the bytes with their content type and
`Content-Disposition: inline; filename="..."`. The id is the last part of the `files://` URI.

## Who may read a file

- A file referenced by a message belongs to that message's thread: whoever may read the thread may read it.
- An upload that no message references yet is readable only by its uploader.
- Under tenant access, a file of another tenant's thread answers `404`, like the thread.

## Lifetime

A file belongs to the first thread whose message references it. An upload (or a file an agent stored with
`Files.put`) that no message references expires after `upload_ttl` seconds (an hour by default:
`file_store: {postgres: {dsn: ..., upload_ttl: 7200}}`). Worker processes delete expired
uploads every few minutes; `await engine.cleanup()` does it now and returns how many it deleted.

Files never change: a new version is a new file in a new message.

## File stores

| Name | Class | Options |
|---|---|---|
| `in_memory` | `InMemoryFileStore` | `upload_ttl` |
| `postgres` | `PostgresFileStore` | `dsn`, `migrate`, `upload_ttl` |
| `module:Class` | your own `BaseFileStore` | yours |

With `LLAMPHOUSE_DATABASE_URL` set and no `file_store` in the runtime file, the Postgres file store is used.
The Postgres store keeps the bytes in the database (`llamphouse_blobs`, once per sha256) and the records in
`llamphouse_files`. For object storage (S3, Azure Blob), implement `BaseFileStore` from
`llamphouse.runtime.file_store` and name it in the runtime file:

```yaml
file_store: {my_pkg.stores:S3FileStore: {bucket: "${FILES_BUCKET}", upload_ttl: 3600}}
```

A store only stores bytes; access checks and events are the runtime's. The contract tests in
`tests/contract/file_store` run against any store.

## See also

- [HTTP API](../http-api.md#files)
- [Example 05_ResearchDesk](../../examples/05_ResearchDesk): an artifact with a CSV file

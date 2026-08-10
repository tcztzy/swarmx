# Offline Reference Library

SwarmX keeps subjective and objective knowledge separate. `swarmx-mem` owns
curated Memory pages that a user or confirmed Agent may create, edit, delete,
diff, and restore. Its current organization uses linked Markdown, but that
organization is not a separate product concept. The `swarmx.ref` server owns no
knowledge: it is a read-only MCP
view over one local offline ZIM archive, normally a Kiwix/OpenZIM Wikipedia
edition. Reference results never become Memory unless a separate, confirmed
Memory mutation explicitly writes a curated page.

## Source and engine

The user supplies an existing `.zim` path. SwarmX does not download an edition,
run a network server, or scan folders. The Python module uses the official
openZIM `python-libzim` binding, whose wheels bundle the native reader on the
supported CPython desktop platforms. This avoids a second `kiwix-serve`
process and avoids inventing a SwarmX encyclopedia format.

The source archive remains the only authority. `swarmx.ref` exposes exactly one
private stdio MCP tool, `swarmx_reference`, with three operations:

```json
{ "request": { "operation": "status" } }
{ "request": { "operation": "search", "query": "distributed systems", "limit": 10 } }
{ "request": { "operation": "get", "path": "A/Distributed_system", "maxChars": 16000 } }
```

`status` returns bounded edition metadata and the file name/size, not an
arbitrary filesystem listing. `search` uses the archive full-text index and
falls back to its suggestion index when full-text search is unavailable. `get`
reads one path, rejects compressed items above the safe limit, converts HTML to
plain text, removes script/style/template content, and returns at most 32,000
characters. Request keys, query/path lengths, result counts, item bytes, and
output text are all bounded. There are no create, update, delete, download,
import, or Memory-promotion operations.

## Package and launch

Reference lives at `swarmx.ref` inside the one standard root `swarmx` package.
The worker, RSI, and Reference implementations share the same direct project
dependencies and locked environment; there is no namespace distribution,
entry-point discovery, uv member workspace, or `ref` dependency group. The
TypeScript host launches the explicitly owned module and validates the exact MCP
server name, tool list, and stdio transport.

Use the locked project during development:

```shell
uv run --locked python -m swarmx.ref.server \
  --zim /path/to/wikipedia.zim --stdio
```

Desktop Main enables the Agent-facing `ReferenceLibrary` projection only when
both launch paths are explicit:

```shell
SWARMX_REFERENCE_PYTHON=/absolute/path/to/.venv/bin/python \
SWARMX_REFERENCE_ZIM=/absolute/path/to/wikipedia.zim swarmx
```

The interpreter must contain the locked `swarmx` distribution. Main passes
neither ambient credentials nor the surrounding environment to the child. If
either setting is absent, direct Agents receive no Reference tool; external ACP
Harnesses never receive it. Search/get calls remain visible as ordinary tool
calls and `status` lets the Agent inspect the exact edition metadata in use.

The raw private MCP tool is not an external Harness tool and grants no Provider,
Session, audit, scheduling, or persistence authority. A TypeScript host must
still validate its own public/IPC boundary and explicitly report when the
Reference module or configured archive is unavailable.

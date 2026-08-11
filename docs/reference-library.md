# Reference Library

SwarmX keeps subjective and objective knowledge separate. `swarmx-mem` owns
curated Memory pages that a user or confirmed Agent may create, edit, delete,
diff, and restore. `swarmx.ref` owns no knowledge: it is a read-only MCP view
over explicitly configured objective sources. Reference results never become
Memory unless a separate, confirmed Memory mutation writes a curated page.

## Sources

`swarmx.ref` supports three independently configured sources:

- `zim` opens one existing local `.zim` archive with the official OpenZIM
  `python-libzim` binding. SwarmX does not download an edition or scan folders.
- `web` sends an explicitly selected query to one configured SearXNG JSON
  endpoint. It returns bounded titles, URLs, and snippets. A later `get` may
  read only a result already cached by that server process; it never fetches an
  arbitrary result URL.
- `zotero` reads Zotero Desktop's fixed `http://127.0.0.1:23119/api/` endpoint.
  It searches top-level items and returns bounded bibliographic metadata. It
  does not use the Connector write API, attachment file routes, or full text.

Web Search is opt-in twice: the endpoint must be configured at launch and every
search/get request must specify `"source": "web"`. An omitted source preserves
the existing ZIM behavior and fails when no ZIM is configured. Zotero likewise
must be selected explicitly. This prevents an ordinary offline query from being
silently sent to a network service or local personal library.

The private stdio server reserves stdout exclusively for JSON-RPC. Native
`libzim` diagnostics emitted while an archive is opened are routed to stderr
before the MCP transport starts, including diagnostics from language indexes
without a stemming implementation.

## Tool contract

`swarmx.ref` exposes exactly one private stdio MCP tool,
`swarmx_reference`, with three operations:

```json
{ "request": { "operation": "status" } }
{ "request": { "operation": "search", "source": "web", "query": "distributed systems", "limit": 10 } }
{ "request": { "operation": "get", "source": "zotero", "path": "ABCD2345", "maxChars": 16000 } }
```

`status` returns bounded metadata for every configured source and may be
filtered by source. `search` returns source-qualified matches; `get` reads one
source-native path or cached Web result. Existing ZIM search/get requests may
omit `source`. ZIM search uses the archive full-text index and falls back to its
suggestion index. ZIM `get` rejects compressed items above the safe limit,
converts HTML to plain text, removes script/style/template content, and returns
at most 32,000 characters. Request keys, query/path lengths, result counts,
network response bytes, item bytes, and output text are bounded. There are no
create, update, delete, download, import, arbitrary URL-fetch, attachment, or
Memory-promotion operations.

## Package and launch

Reference lives at `swarmx.ref` inside the one standard root `swarmx` package.
The worker, RSI, and Reference implementations share the same direct project
dependencies and locked environment; there is no namespace distribution,
entry-point discovery, uv member workspace, or `ref` dependency group. The
TypeScript host launches the explicitly owned module and validates the exact
MCP server name, tool list, and stdio transport.

Use one or more source flags with the locked project during development:

```shell
uv run --locked python -m swarmx.ref.server \
  --zim /path/to/wikipedia.zim \
  --web-search-url https://search.example.com \
  --zotero --stdio
```

Desktop Main enables the Agent-facing `ReferenceLibrary` projection only when
`SWARMX_REFERENCE_PYTHON` and at least one source setting are explicit:

```shell
SWARMX_REFERENCE_PYTHON=/absolute/path/to/.venv/bin/python \
SWARMX_REFERENCE_ZIM=/absolute/path/to/wikipedia.zim \
SWARMX_REFERENCE_WEB_SEARCH_URL=https://search.example.com \
SWARMX_REFERENCE_ZOTERO=1 swarmx
```

The interpreter must contain the locked `swarmx` distribution. Main passes no
ambient credentials or surrounding environment to the child. If no source is
configured, direct Agents receive no Reference tool; external ACP Harnesses
never receive it. Calls remain visible as ordinary tool calls and `status` lets
the Agent inspect the exact configured sources.

The raw private MCP tool is not an external Harness tool and grants no Provider,
Session, audit, scheduling, persistence, Project-shell, or Zotero-write
authority. A TypeScript host must still validate its own public/IPC boundary and
explicitly report when the Reference module or selected source is unavailable.

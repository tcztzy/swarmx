# SwarmX

Run local and ACP-compatible coding agents in one desktop workspace. SwarmX also provides a reusable TypeScript orchestration core and CLI.

<p align="center">
  <a href="docs/assets/swarmx-demo.mp4">
    <img src="docs/assets/swarmx-demo.gif" alt="SwarmX desktop running an agent task" width="900" />
  </a>
</p>

<p align="center"><sub>Pick a Harness and Model, run a task, and inspect live agent work. Click for MP4.</sub></p>

## Install

### macOS app

Download the latest DMG from [GitHub Releases](https://github.com/tcztzy/swarmx/releases/latest):

- `arm64` for Apple silicon Macs
- `x64` for Intel Macs

Open the DMG and drag SwarmX to Applications.

### npm

Requires Node.js 22.13 or newer.

```shell
npm install swarmx
npx swarmx
```

`npx swarmx` opens Desktop. Installation itself never launches the app; the first launch may finish downloading the Electron runtime.

For a global command:

```shell
npm install --global swarmx
swarmx
```

## First run

If no compatible Model is configured, use **Connect model provider** in the main workspace; it opens the Add Provider form directly. Add an OpenAI-, Anthropic-, DeepSeek-, OpenCode Go-, OpenRouter-, or Ollama-compatible connection, then choose a Harness and Model in the composer. Custom Providers use one exact Base URL and key. Official OpenRouter `/api` and `/api/v1` URLs normalize to one Provider with Anthropic, OpenAI Chat, and OpenAI Responses routes. OpenCode Go loads its live model list, routes models through their documented APIs plus narrow runtime-verified compatibility exceptions, and can keep additional plaintext backup keys with local usage counters and quota failover. A newly discovered Go model without a known native route uses the Provider's selected preferred protocol until SwarmX is updated with its official route.

Provider credentials are stored as plaintext in the editable `~/.swarmx/provider-auth.json` file with restrictive permissions. The renderer never reads the file or receives plaintext credentials. The current file format is `schemaVersion: 2`; older encrypted auth files and legacy `local_keychain` Provider references are intentionally not migrated.

## CLI

Passing a CLI argument keeps the existing terminal workflow:

```shell
npx swarmx doctor
npx swarmx send "Explain this repository" --model gpt-5.6-sol
npx swarmx serve --port 8000
npx swarmx cli --help
```

Use `npx swarmx desktop` as an explicit Desktop alias.

Session history uses append-only JSONL grouped by working directory under
`~/.swarmx/projects/`; sessions without Project context use `__recents__`.
Each directory has a rebuildable index. Older `.json` Session files and
migration backups are unsupported and are not read.

## Develop from source

```shell
git clone https://github.com/tcztzy/swarmx.git
cd swarmx
corepack enable
pnpm install
pnpm --filter @swarmx/desktop dev
```

Validation:

```shell
pnpm lint
pnpm test
pnpm -r build
pnpm test:python
pnpm test:mem
```

SwarmX is intentionally one polyglot workspace. Root `package.json` / pnpm own
the TypeScript product, root `pyproject.toml` / `uv.lock` own one standard
Python `swarmx` package, and root `Cargo.toml` / `Cargo.lock` discover Rust
`crates/*` modules:

- `src/swarmx/rsi` — DSPy/GEPA private MCP implementation
- `src/swarmx/ref` — read-only offline ZIM Reference MCP implementation
- `src/swarmx/worker.py` — durable Python worker
- `crates/swarmx-mem` — versioned subjective Memory MCP module

The Python worker and private MCP servers use the same locked package environment:

```shell
uv run --locked python -m swarmx.rsi.server --version-json
uv run --locked python -m swarmx.ref.server --help
```

Create local macOS DMG and ZIP packages with:

```shell
pnpm --filter @swarmx/desktop dist:mac
```

## Packages

- `swarmx` — Desktop-first npm launcher with CLI compatibility
- `@swarmx/codex` — repository-owned Codex App Server DSH plugin and launch module
- `@swarmx/desktop` — Electron app and reusable renderer shell
- `@swarmx/core` — the v4 DSH plugin runtime: request-Fiber execution services, Provider/Harness/Swarm registries, Harness transports, sessions, and platform contracts
- `@swarmx/cli` — terminal commands and OpenAI-compatible server
- `@swarmx/acp-server` — ACP server adapter
- `@swarmx/runtime` — runtime detection, Doctor, and repair planning

## Documentation

- [Current product specification](SPEC.md)
- [Codex module](docs/codex-module.md)
- [Product vision](docs/vision.md)
- [Roadmap](ROADMAP.md)
- [Architecture and design](DESIGNS.md)
- [Full documentation](docs/index.md)
- [Extensions and Custom Agents](docs/extensions-custom-agents.md)
- [Multimedia attachments and previews](docs/multimedia.md)

## License

[MIT](LICENSE)

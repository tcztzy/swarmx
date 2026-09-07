# SwarmX

Local-first research desktop with recursive Swarms and native Agents.

```text
Electron / Browser → assistant-ui + AG-UI → Host → ACP → Swarm → ACP → … → upstream ACP adapter
                                            ↑ ACP / A2A (external)
                                            └ MCP → ProductServices → ACP
```

Codex ACP is the default. The Host uses `@agentclientprotocol/codex-acp`,
`@agentclientprotocol/claude-agent-acp`, `hermes acp` and `openclaw acp` over stdio.
Adapters load lazily; startup failure never selects another Agent. DSH, ZCode and Kimi are deferred.
The lockfile pins the packaged adapters and applies the reviewed fixes in `patches/`. Native login is
needed for conversation, not for configuring an environment or inspecting research objects.

Install Node.js and pnpm matching `package.json`, Rust stable with a C/C++ linker (the bundled
Typst compiler is built with the committed Cargo lockfile), and Docker Engine or Docker Desktop
for Python execution. macOS also requires the Xcode command-line tools. Linux and macOS are the
CI targets; Windows has not been validated.

```sh
pnpm install --frozen-lockfile
pnpm dev
```

`pnpm dev` starts Vite HMR for the renderer. React/CSS edits update the open window; compatible
React edits preserve component state. Main-process, Host and workspace-package source changes
trigger an incremental TypeScript build and restart Electron after a successful build. A failed
build stays visible in the terminal and the watcher waits for the next edit. Main-process restarts
end active runs; saved conversations and their permissions remain on disk. Ctrl+C stops development.
Use `pnpm start` for the normal static production build and launch.

For Chrome or another browser, run `pnpm web` and open the one-use loopback URL printed in the
terminal. The URL expires after one minute and becomes a local HttpOnly session cookie. Set
`SWARMX_WORKSPACE=/absolute/research/path` to select an existing directory and `SWARMX_HOME` to
choose the private data directory (default `~/.swarmx`). Keep the launch URL private.

Start with a conversation. **Assets / 科研资产** opens files, images and optional source editing
beside that conversation. **Observe / 观测与溯源** groups react-o11y traces, recorded scientific
runs and the RO-Crate graph. Select the entire bottom-left workspace row to open full-page
**Settings / 设置** for language, permissions and environment setup. English and Simplified Chinese
are supported; the Host restores your language choice across restarts and workspace changes.

The bundled recipe uses the official `quay.io/jupyter/datascience-notebook` image pinned to a
multi-architecture digest. Docker selects its native amd64 or arm64 variant, and SwarmX records
the resolved image, platform and installed Python packages. The base also contains R and Julia;
SwarmX's notebook/figure execution currently uses Python. Setup needs network access; notebook and
figure code runs without network in an immutable image. Missing Docker or setup failures are
shown and never execute that code on the host. See [workbench operation and publication
readiness](docs/product-readiness.md) for boundaries, exports and acceptance evidence.

Use `SWARMX_AGENT=claude|hermes|openclaw` or the Agent selector. Native setup and external
ACP/A2A access: [Agent platform](docs/runtime-platform.md).

[Memory](docs/memory.md) provides shared semantic memory in private OKF Markdown. Agents use the
`memory` product tool to retrieve and curate research knowledge across sessions; native runtimes
keep their own conversation histories. Existing vaults move to the current storage path on startup.

```sh
pnpm typecheck
pnpm test
pnpm build
pnpm lint
pnpm docs:check
```

The real scientific integration test imports a dataset, executes and revises a figure, verifies
immutable bytes and RO-Crate references, and tests container confinement and cancellation:

```sh
docker build --tag swarmx-research:validation apps/desktop/resources/python
SWARMX_TEST_DOCKER_IMAGE=swarmx-research:validation pnpm vitest run apps/desktop/tests/research-environment.test.ts
pnpm paper:demo
```

Build libraries with `pnpm build:lib` before running Vitest directly on a fresh checkout.
The release workflow prepares a source archive, SHA256 checksums and a draft GitHub release;
it does not currently package signed desktop installers. [CITATION.cff](CITATION.cff) contains
software citation metadata. Submission requires a citable release and a reproducible scientific-use
example from that release. Claims of researcher productivity or comparative benefit require
separate evaluation; passing tests does not establish them.

[SPEC.md](SPEC.md): contract. [DESIGNS.md](DESIGNS.md): ownership and boundaries.
[CODEBASE.md](CODEBASE.md): source map.

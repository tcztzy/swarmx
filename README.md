# SwarmX

SwarmX is a local-first research work system with native Agents, recursive delegation and
a scientific workbench. Persistent goals share budgets, deadlines and acceptance criteria.

Start with the [product direction](docs/product-direction.md) for the vision and ownership,
[product readiness](docs/product-readiness.md) for the usable and tested scope, and
[roadmap](ROADMAP.md) for the next implementation milestones.

```text
Electron → assistant-ui + AG-UI → IPC → Host → Swarm → Native Agent
                                        ↑ ACP / A2A (external)
                                        └ MCP → ProductServices → Swarm
```

SwarmX orchestrates external Agents and contains no builtin Agent runtime. Codex App Server
is the default; use its native login/configuration. To use an independent ACP agent, configure
`SWARMX_ACP_AGENT` and explicitly admit `acp` in Host policy; see [ACP setup](docs/acp.md).
Builds generate App Server declarations with the official CLI pinned in the development
dependencies; `CODEX_PATH` can explicitly select a different installed executable.
Claude and DSH use their official Agent SDK and SDK Client. Hermes uses its installed TUI Gateway;
OpenClaw uses the official Gateway Client with an explicit address and credentials. DSH provides
independent executions and Host log viewing; it does not resume tasks across processes.
Integrations load lazily; startup failure never selects another Agent. Native login is
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

The Host uses its starting directory for execution. Set `SWARMX_CWD=/absolute/research/path`
to select another directory at launch, and `SWARMX_HOME` to choose the private data directory
(default `~/.swarmx`).

Independent projects can keep their own programs, dependencies and native skills in
that directory. See [running an independent project](docs/domain-projects.md) for desktop/ACP launch
commands, project-local resources, output ownership and execution-log export. Its historical
Pi examples are retained as evidence; see the migration note before using them.

Start with a conversation. **Assets / 科研资产** opens files, images and optional source editing
beside that conversation. **Observe / 观测与溯源** groups react-o11y traces, recorded scientific
runs and the RO-Crate graph. Use the bottom-left settings control to open full-page
**Settings / 设置** for language, permissions and environment setup. English and Simplified Chinese
are supported; the Host restores your language choice across restarts.

The bundled recipe uses the official `quay.io/jupyter/datascience-notebook` image pinned to a
multi-architecture digest. Docker selects its native amd64 or arm64 variant, and SwarmX records
the resolved image, platform and installed Python packages. The base also contains R and Julia;
SwarmX's notebook/figure execution currently uses Python. Setup needs network access; notebook and
figure code runs without network in an immutable image. Missing Docker or setup failures are
shown and never execute that code on the host. See [workbench operation](docs/product-readiness.md)
for boundaries, exports and acceptance evidence.

Use `SWARMX_AGENT=codex|claude|dsh|hermes|openclaw|acp` or the Agent selector. Native setup and external
ACP/A2A access: [Agent platform](docs/runtime-platform.md).

[Memory](docs/memory.md) retains research knowledge across sessions in private OKF Markdown.
[Work management](docs/work-management.md) describes persistent goals, Agent choices, runtime
limits and acceptance. Native runtimes keep their own conversation histories.

See [verification commands](docs/product-readiness.md#verification) for engineering and Docker checks.
Release tags use `v<version>` and must match `apps/desktop/package.json`. SwarmX workspace
packages, the bundled runtime and SwarmX-owned protocol identities use the same release version.
`pnpm package:mac` builds a macOS DMG for the current machine's architecture. The release
workflow builds Apple Silicon and Intel installers on GitHub's macOS runners, then publishes
them with a source archive and SHA256 checksums. See [macOS packaging and releases](docs/macos-release.md)
for local validation, tag publishing and signing limitations. [CITATION.cff](CITATION.cff) contains
software citation metadata.

Manuscript sources, Zotero references and publication evidence live in the separate `swarmx-paper`
project. Its paper revisions do not change this repository's software release archive or checksum.
Software builds, tests and releases work independently of the paper project.

[SPEC.md](SPEC.md): contract. [DESIGNS.md](DESIGNS.md): ownership and boundaries.
[CODEBASE.md](CODEBASE.md): source map.

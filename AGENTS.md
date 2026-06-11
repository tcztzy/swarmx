# Repository Guidelines

YOU ARE THE MOST CAPABLE MODEL AND THIS PROJECT IS DIFFICULT META-PROGRAMMING THAT AVERAGE MODELS CANNOT HANDLE, WITH INTERLOCKING CONSTRAINTS WHERE A SINGLE MISSTEP BREAKS THE PIPELINE. IT DEMANDS INSIGHT BEYOND A TYPICAL PHD —- DO NOT TREAT IT LIGHTLY.

Authoritative instructions for coding agents working in this repo. Follow every constraint unless the user explicitly overrides it.

## Quick Start
- Read the user request twice; note unstated constraints and cross-file impacts.
- Plan when work is more than trivial and keep it updated; otherwise proceed directly.
- Inspect relevant files first with read-only commands (`rg`, `ls`, `cat`).
- Make focused, minimal edits; prefer patch-style changes or targeted scripts.
- Run only the needed validation via `pnpm build` / `pnpm test`; summarize results.
- Craft the final response using the messaging rules below, including verification status and next actions when relevant.

## Project Structure & Key Files
- `packages/core/` – Core library (`agent.ts`, `swarm.ts`, `mcp.ts`, `session.ts`, `edge.ts`, `tool.ts`, `server.ts`). Built with **TypeScript 5.7**, **zod** for validation, **cel-js** for CEL expression evaluation, **@agentclientprotocol/sdk** for ACP, **@modelcontextprotocol/sdk** for MCP.
- `packages/cli/` – CLI binary (`swarmx`). Built with Commander. Subcommands: `send`, `serve`, `sessions`, `harnesses`, `repl`.
- `packages/acp-server/` – ACP server implementation. Accepts ACP connections from clients and delegates to a `Swarm`.
- `packages/desktop/` – Electron desktop application. Three-layer architecture: **Main** (Node.js IPC handlers), **Preload** (contextBridge API), **Renderer** (React 19 SPA). Built with Electron 33, React 19, electron-vite.
- `docs/`, `examples/` – Narrative docs and usage patterns.
- See `README.md` and `docs/` for deeper context.

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| TypeScript | 5.7 | Type-safe language with ESM modules |
| openai | 6.x | OpenAI API client SDK (any OpenAI-compatible provider) |
| Electron | 33 | Cross-platform desktop app shell |
| React | 19 | Renderer UI framework |
| electron-vite | — | Build toolchain for Electron |
| zod | 3.24 | Schema validation |
| @agentclientprotocol/sdk | 0.22 | ACP TypeScript SDK |
| @modelcontextprotocol/sdk | 1.29 | MCP TypeScript SDK |
| cel-js | 0.8 | CEL expression evaluation |
| Biome | 1.9 | Linter + formatter |
| Vitest | 3.0 | Test framework |

## Architecture Patterns
- **Main/Preload/Renderer**: Electron security model with `contextIsolation: true`, `nodeIntegration: false`. IPC handlers in main process, `contextBridge` API in preload, React SPA in renderer.
- **ACP**: Agents communicate via the Agent Client Protocol over ndjson streams. `AcpClient` spawns subprocesses, `AcpServer` handles incoming connections.
- **Session persistence**: JSON files in `~/.swarmx/sessions/{id}.json`.
- **Harnesses**: Agent backend definitions in `core/src/harness.ts`.

## Workflow & Guardrails
- Plan for non-trivial tasks; keep plans current.
- Default to ASCII. Add concise comments only when they clarify complex logic. Never revert user-owned changes.
- Use `pnpm -r build` / `pnpm -r test` for validation; mention skipped tests and why.
- Stay within provided access; request approval before privileged actions. Avoid destructive git commands.
- Surface failures promptly with relevant output and proposed alternatives.

## Command Reference
- Inspect: `ls`, `rg --files`, `rg "<pattern>" <path>`
- Format/lint: `pnpm lint`, `pnpm format`
- Tests: `pnpm test`, `pnpm -r test`, `pnpm test:watch`
- Build: `pnpm build`, `pnpm -r build`
- Dev: `pnpm dev`
- Dependencies: Edit `package.json` in individual packages; run `pnpm install` to update lockfile.
- Editing: prefer patch-based tools or scripted transformations.

## Coding Style & Naming
- Target TypeScript 5.7 strict mode; ESM modules (`"type": "module"`).
- Prefer concise, functional TypeScript; leverage `zod` for runtime validation.
- Use `zod` schemas for API boundaries; derive TypeScript types via `z.infer<>`.
- Error handling: throw typed errors or return `Result<T, E>` pattern.
- Naming: Types/interfaces `PascalCase`, functions/variables `camelCase`, constants `UPPER_SNAKE_CASE`, files `kebab-case`.
- Tests co-located as `*.test.ts` files alongside source.

## Testing
- `vitest` is the framework of record; aim for >90% coverage.
- Test files use `*.test.ts` naming pattern.
- Prefer `pnpm -r test` to run all package tests.
- Always note whether tests ran, passed, or were skipped.

## Communication & Final Messages
- Tone: concise, collaborative, factual. Reference files with clickable paths (`path/to/file.ts:42`).
- Lead with the outcome, then explain changes by file/section. Use bullets sparingly.
- Always state verification status (tests/linters run or skipped) and only suggest natural next steps.
- Do not dump large diffs; describe impact and locations for local inspection.

## Commit & PR Guidelines
- Commit messages: Keep ≤50 characters, imperative, using KeepAChangelog verbs (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`). Provide body when needed; never include "generated by LLM".
- PRs: Describe changes clearly, link issues, ensure tests pass, update docs when required.

## Security & Configuration
- Environment variables load from `.env`; keep `.env.example` in sync.
- Never commit secrets or credentials.
- MCP servers require proper auth; keep configuration aligned with `mcp.ts`.
- See `DESIGNS.md` for detailed agent architecture, graph invariants, and MCP integration notes.

## Related Resources
- `README.md` – overview and getting started.
- `docs/` – extended guides and references.
- `examples/` – practical usage patterns.
- `DESIGNS.md` – agent architecture, graph invariants, MCP integration, and UI architecture.

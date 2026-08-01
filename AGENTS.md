# Repository Guidelines

Authoritative instructions for coding agents working in this repository. Follow
them unless the user explicitly overrides a rule.

## Workflow

- Read the request carefully and identify cross-file effects before editing.
- Plan non-trivial work and keep the plan current.
- Inspect relevant files with `rg`, `rg --files`, `ls`, and focused reads.
- Preserve user-owned changes and avoid destructive Git commands.
- Make focused, minimal edits with patch-style tools.
- Validate in proportion to risk and report what ran or was skipped.
- Keep secrets, credentials, generated output, and local artifacts out of Git.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `packages/core/` | Agents, `SwarmConfig` execution, ACP/MCP, Sessions, Projects, and reusable contracts |
| `packages/runtime/` | Runtime detection, Doctor reports, and repair planning |
| `packages/acp-server/` | ACP server backed by a `Swarm` and Core Sessions |
| `packages/cli/` | Commander CLI: `doctor`, `send`, `eval-run`, `serve`, `sessions`, `harnesses`, and `repl` |
| `packages/desktop/` | Electron Main, Preload, Renderer, and host integrations |
| `packages/swarmx/` | Desktop-first npm launcher and CLI compatibility |
| `docs/` | Current user, feature, protocol, and product-direction documentation |
| `evals/` | Inspect evaluation adapter and fixtures |

Dependency versions are authoritative in package manifests and lockfiles.

## Architecture Invariants

- **Main/Preload/Renderer:** Renderer has no direct Node.js authority. Privileged
  work goes through a narrow typed Preload bridge to authorized Main handlers.
- **Identity:** Model, Provider, Harness, Agent, Extension, and Project remain
  separate concepts. Agent identity is `harnessId:modelId`.
- **Workflows:** `SwarmConfig` is the only persisted workflow format.
- **ACP:** External Harnesses own their native tools, authentication, sessions,
  and permissions. Do not inject duplicate SwarmX tools.
- **Sessions:** Canonical history is append-only JSONL under
  `~/.swarmx/sessions/`; the metadata index is rebuildable.
- **Boundaries:** Use zod at persistence, IPC, protocol, plugin, and public input
  boundaries. Renderer imports must use browser-safe Core subpaths.
- **Side effects:** Discovery and planning are read-only. Installation, repair,
  trust changes, authority changes, and destructive operations are explicit.
- **Secrets:** Persist references in settings metadata and never expose plaintext
  secrets to the Renderer, traces, telemetry, or logs. The dedicated,
  user-editable `~/.swarmx/provider-auth.json` file is an explicit product
  exception: it stores Provider credentials as plaintext with restrictive file
  permissions so users can inspect and edit it directly.
- **Provider auth compatibility:** The current Provider auth format is
  `schemaVersion: 2` with `local_auth_file` references. Older encrypted auth
  documents and legacy `local_keychain` Provider references are intentionally
  not migrated; affected Providers must be configured again in the current
  format.

See `SPEC.md` for product requirements and `DESIGNS.md` for architecture.

## Commands

```shell
pnpm lint
pnpm test
pnpm test:coverage
pnpm -r build
pnpm run ci:node
pnpm dev
```

Use focused package tests during implementation and the relevant broader gate
before completion. Edit a package's `package.json` for dependencies and run
`pnpm install` when the lockfile must change.

## TypeScript Style

- Use strict TypeScript and ESM modules.
- Prefer concise functional code without clever compression.
- Use zod schemas for runtime boundaries and derive types with `z.infer<>`.
- Keep errors actionable; do not swallow failures or silently widen authority.
- Name types and interfaces `PascalCase`, values `camelCase`, constants
  `UPPER_SNAKE_CASE`, and files `kebab-case`.
- Add comments only when they clarify a non-obvious invariant.
- Let Biome own formatting and import organization.

## Tests

- Vitest is the JavaScript/TypeScript test framework.
- Name tests `*.test.ts` or `*.test.tsx` and follow the package's existing
  placement.
- Test public behavior, boundary rejection, cancellation, persistence, and
  security-sensitive failure paths.
- Coverage thresholds and audited files are defined in repository configuration;
  do not duplicate their values in documentation.
- Always state whether tests, lint, and builds passed or were skipped.

## Documentation

- Keep `SPEC.md` short and limited to durable product requirements.
- Keep `ROADMAP.md` limited to unfinished work.
- Put architecture decisions in `DESIGNS.md` and feature guidance in `docs/`.
- Use Git history and release notes for completed tasks, incidents, and old
  research; do not rebuild ledgers in current documentation.
- Avoid hard-coding dependency versions outside manifests unless a compatibility
  rule specifically requires an exact version.

## Git and Delivery

- Default to the current branch and preserve unrelated work. Do not create or
  switch branches unless requested.
- If the user asks for a commit, use an imperative Keep a Changelog verb and
  keep the subject at most 50 characters.
- Never include secrets or claim tests passed when they did not run.
- In the final response, lead with the outcome, cite changed files, state
  verification status, and mention only natural next steps.

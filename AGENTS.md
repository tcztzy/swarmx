# Agent Guidelines

Rules for coding agents working in this repository.

## Principles

**Minimalism**

Implement required behavior with the smallest coherent design. Delete, consolidate, reuse. Never add abstractions, layers, dependencies, options, or parallel paths without a concrete requirement. Remove speculative flexibility and unused code immediately.

**No compatibility patches**

Change internal modules, private types, and implementations directly. Update all in-repo callers atomically. Remove obsolete adapters, aliases, and fallback paths. Only preserve compatibility for user-facing behavior: UI, public APIs, CLI commands, configuration formats, persisted data.

**Documentation-and-test-driven**

Before implementation:
1. State or update the contract in documentation
2. Express acceptance criteria in focused tests
3. Implement only enough code to satisfy both
4. Keep documentation, tests, and code synchronized

Bug fixes require a regression test. Documentation updates when behavior, boundaries, or flows change.

## Workflow

1. Read the request. Identify cross-file effects before editing.
2. Read `CODEBASE.md` and relevant `docs/codebase/*.md` maps before changing source.
3. Plan non-trivial work. Keep the plan current.
4. Define contract in documentation and tests before or alongside implementation.
5. Inspect relevant files with `rg`, `rg --files`, `ls`, focused reads.
6. Make focused, minimal edits with patch-style tools.
7. Validate in proportion to risk. Report what ran or was skipped.
8. Update code maps for new or moved files. Run `pnpm docs:check` before handoff.

## Code Style

**TypeScript**

Strict mode. ESM modules. Zod at boundaries. Derive types with `z.infer<>`. Keep errors actionable. Never swallow failures or silently widen authority.

**Naming**

Types `PascalCase`, values `camelCase`, constants `UPPER_SNAKE_CASE`, files `kebab-case`.

**Comments**

Only when clarifying non-obvious invariants. Let code speak.

**Formatting**

Biome owns it. Don't fight the formatter.

## Tests

Framework: Vitest. Name: `*.test.ts` or `*.test.tsx`.

Test: public behavior, boundary rejection, cancellation, persistence, security-sensitive failures.

Always state whether tests, lint, builds passed or were skipped. Never claim tests passed without running them.

## Documentation

- `SPEC.md` — durable product requirements (keep short)
- `ROADMAP.md` — unfinished work only
- `DESIGNS.md` — architecture decisions
- `docs/` — feature guidance

Use Git history for completed tasks, incidents, old research. Don't rebuild ledgers in current docs. Don't hard-code dependency versions outside manifests.

## Git

Preserve user changes. Avoid destructive commands. Default to current branch.

Commit messages: imperative Keep a Changelog verb, ≤50 chars. Never include secrets. Never claim tests passed when they didn't run.

## Add a Package Only When It Creates a Real Ownership Boundary

If the code isn't published separately, consumed by external projects, or owned by a different team with different release cadence, it's not a package. It's a directory. Monorepo packages are not free: each requires `package.json`, `tsconfig.json`, `vitest.config.ts`, build orchestration, and mental overhead. Create one only when it enforces a genuine boundary.

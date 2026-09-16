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

## Behavioral Acceptance

- Bind each acceptance conclusion to its scope and candidate: record HEAD and the staged/unstaged diff identity. On re-acceptance, reconcile previous unresolved findings with current evidence as closed, still failing, or unverified. Do not carry a passing conclusion across changed candidates.
- Derive cases from the existing contract across affected implementations, not just the latest reported symptom. For lifecycle changes, trace the actual runtime's directly affected transitions, including asynchronous preparation inside SDK calls, dispatch, acknowledgements, queued work, terminal events and cleanup. Use this to identify missing cases; avoid exhaustive combinations or unrelated audits.
- Place deterministic pauses at the relevant asynchronous boundaries. For cancellation before dispatch, release the pause after Stop returns and assert zero subsequent provider/tool calls as well as a cancelled outcome. A cancelled status alone does not prove execution stopped. For completion, verify accepted follow-up work finishes before resources are released.
- Verify native behavior against the installed SDK or matching upstream source. Prefer the real SDK with a local provider substitute for lifecycle regressions. A mock must preserve the relevant ordering, ownership and persistence constraints; using a real SDK alone does not prove the tested timing covers the failure.
- Promote confirmed regressions into the normal test suite when fixing them. Demonstrate failure before the fix and success after it. Preserve the behavioral assertion; justify fixture changes with native-contract evidence. A reproduction only in ignored `runs/` is not a durable regression guard.
- Report previous findings closed, new blockers and unverified scope separately. Distinguish a targeted fix passing from the whole candidate passing. A green suite proves its assertions, not missing contract coverage; neither test counts nor speculative, unreachable scenarios justify the acceptance decision.

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

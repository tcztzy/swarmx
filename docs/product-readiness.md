# Product readiness

This document maps implemented behavior to reproducible checks and deployment limits.
See [product direction](product-direction.md) for the objective and [ROADMAP](../ROADMAP.md)
for unfinished work.

Readiness is assessed for a particular source revision and its local changes. Existing tests and
historical experiment archives describe their own evidence scope; they do not establish that a
changed candidate passed. Report current checks, skipped environments and unresolved failures.

## Generic work and evidence contract

SwarmX retains native conversation, Work, Memory, execution evidence, evaluation RO-Crate export
and read-only graph views. Scientific data models, notebook/figure execution and artifact stores
belong to domain applications. No Docker or Rust compiler is required by the SwarmX build.

Settings and Host grants are validated. Native modes retain their harness semantics; SwarmX does
not promise a cross-harness filesystem sandbox. Journal evidence remains directory-scoped.
Domain operations use generic authorized Agent/tool calls. Work requires observed execution
evidence scoped to its directory and runtime; artifact identities remain unverified domain claims
until independently assessed. Memory labels external URI references unverified. See
[domain integration](domain-projects.md).

Linux/macOS × Node 22/24 quality checks validate the selected candidate. Native macOS packaging
has its own platform checks; no Windows or Linux installer claim is made. Language, native
lifecycle, grant intersection, independent acceptance, cancellation and persisted evidence retain
their regression tests. Scientific receiver checks do not replace SwarmX Host-boundary tests.

## Work-management checks

The current [Host work-management entry point](work-management.md) persists goals and criteria,
selects admitted configurations, reserves shared budget and records independent feedback. Its
desktop Long-term work panel creates cycles and goals, edits budgets and criteria, starts or stops
work, handles native confirmations and records user acceptance or corrections. Reloading reads durable
state; unresolved external outcomes and invoices require separate explicit reconciliation.
The delegation skill loads current private combination assessments and cited execution statistics at
each preparation, so later reviewed feedback can inform subsequent calls without a second knowledge store.
ProductServices tests run restricted reviews with a local provider substitute: late
acceptance triggers another review, creates a cited Memory entry available to later selection,
and charges its work cycle. Restart replays unpublished feedback, while acknowledged feedback
does not trigger duplicate reviews. Recovery tests keep unknown execution outcomes separate from
invoice settlement and require an explicit outcome confirmation before unresolved work can retry.
Registered learning resources additionally support a fixed baseline/candidate behavior evaluator;
see [learning resources](learning-resources.md) for its report and revision requirements.

Deterministic tests cover
concurrent reservation, repeated/cumulative usage, unknown outcomes, late correction, review
recursion, criteria-version conflict and candidate regression.
Use the real Host/ProductServices boundaries with local native-provider substitutes where feasible;
test doubles must preserve the asynchronous ordering and ownership relevant to each case.

## Verification

Run the standard engineering checks from the repository root:

```sh
pnpm typecheck
pnpm test
pnpm build
pnpm lint
pnpm docs:check
```

Build libraries with `pnpm build:lib` before invoking Vitest directly on a fresh checkout.
Live Agent evaluations require explicit opt-in, configured credentials and a declared budget.
Record them as unrun when they were not executed. The comparison design is in
[product direction](product-direction.md#how-progress-is-measured); passing tests does not
establish productivity or learning gains.

## Evidence and deployment scope

Publication planning and evaluation protocols live in the separate `swarmx-paper` project.
Their frozen research examples retain their original identities as integration evidence. New
continuous benefit evaluations retain separate manifests, acceptance reports and resource records.

The product exports selected evaluation evidence as Attached RO-Crate ZIPs containing the declared
payload files. Hashes do not establish scientific truth or that a behavioral judgment is correct.
macOS packaging is described in [macOS releases](macos-release.md); one architecture's successful
build does not verify another. Live Harness coverage and independent security assessment require
their own execution evidence. Domain scientific workflows require separately configured runtime
integration and acceptance; recorded provenance alone does not establish domain correctness.

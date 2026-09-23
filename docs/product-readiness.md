# Product readiness

This document maps implemented behavior to reproducible checks and deployment limits.
See [product direction](product-direction.md) for the objective and [ROADMAP](../ROADMAP.md)
for unfinished work.

Readiness is assessed for a particular source revision and its local changes. Existing tests and
historical experiment archives describe their own evidence scope; they do not establish that a
changed candidate passed. Report current checks, skipped environments and unresolved failures.

## Research workbench contract

The conversation uses assistant-ui. Research objects, executable figures, environments
and provenance remain inspectable outside the conversation. The Science journal and
RO-Crate are the data sources, not UI state.

- Notebook and figure code use Docker and an immutable image ID, with no host fallback.
  Containers have no network, a read-only root filesystem, dropped capabilities, no
  new privileges and bounded CPU, memory and processes. Only the execution directory and
  declared immutable artifact inputs are mounted. Never expose host credentials or
  the Docker socket to research code.
- Settings persist locally and are validated at the Host boundary. Host tool grants
  inherit by intersection. Native mode selections and approvals retain harness semantics;
  ordinary tasks have no unified cross-harness filesystem ceiling. Background reviews have a
  separate restricted path. Native Agents are not advertised as Docker-isolated.
  The bundled Typst compiler and explicit Git/DVC operations retain their trusted host boundary.
- Environment setup is explicit, reports failures, and records the resolved image
  and installed package versions. Execution records include the actual environment
  and isolation policy. Settings and environment records survive restart.
- Each Host uses one canonical execution directory. Sessions and research records retain directory
  ownership; path traversal and symlink escapes are rejected. Settings are shared across launches.
- Research UI supports research collection creation, entity search, graph/list inspection,
  artifact previews, executable figure creation/editing and RO-Crate inspection and
  export. Edits preserve revisions, hashes and source relationships. Empty and failed
  states are actionable; illustrative examples are explicitly identified.
- Runs distinguish success, failure and cancellation. Failed code cannot register a
  stale output as a newly successful figure.

## Implemented workbench foundation

The table maps existing components to their verification paths. A check listed here is not a
claim that a live environment or remote release job ran for the current candidate.

| Area | Implementation | Acceptance evidence |
| --- | --- | --- |
| Python isolation | Stateless Python in an immutable Docker image; explicit mounts, no credentials/network, bounded resources | Real Docker confinement, read-only, declared inputs and child cancellation tests |
| Settings and work directory | Fixed execution directory, shared private settings, validated IPC and external protocol calls | Directory rejection, persistent settings and native permission API tests |
| Environment setup | Official Jupyter Data Science base pinned by digest, native platform, setup/cancel/inspect and installed package export | Real setup plus image/package inspection; failed setup shown in UI |
| Research assets | Conversation with asset/editor side view; Observe groups react-o11y, scientific runs and RO-Crate | Renderer interaction tests and desktop workflow checks |
| Languages | English/Chinese labels and formatting; Host-persisted preference | Translation coverage, authenticated persistence and draft-preserving language-switch tests |
| Research graph | React Flow projection of RO-Crate, semantic edges, search and neighborhoods; JSON-LD inspector/export | Identity/relation/filter tests and visual graph inspection |
| Failure handling | Failed runs retain error evidence and cannot create a figure artifact | Regression test with an existing stale output; real Docker figure test |
| Build and release | Linux/macOS quality matrix, Docker integration job, native macOS DMGs, source archive and automatic public release with checksums | Local packaging and quality commands; remote GitHub jobs require a tag push; see [macOS releases](macos-release.md) |
| Software citation | CITATION.cff based on software authors and repository license | Author verification and archived version still required |

## Design evidence

- [Claude Science](https://www.anthropic.com/news/claude-science-ai-workbench):
  artifacts beside conversations, exact source/environment/history and iterative
  figure editing. The local example at localhost:8000 was also inspected.
- [Codex App Server](https://developers.openai.com/codex/app-server): native session,
  turn, approval and sandbox settings remain authoritative for Codex.
- [Docker run](https://docs.docker.com/reference/cli/docker/container/run/): maintained
  runtime for isolation, mounts and resource limits.
- [Jupyter Docker Stacks](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html):
  use the maintained Quay Data Science image, pinned by multi-platform digest; inspect
  the daemon's actual architecture and installed dependencies.
- [Obsidian graph](https://obsidian.md/help/plugins/graph) and
  [Neo4j Bloom](https://neo4j.com/docs/bloom-user-guide/current/bloom-visual-tour/bloom-overview/):
  search/filter, node neighborhoods and a details inspector. Preserve edge semantics.
- [RO-Crate](https://www.researchobject.org/ro-crate/specification/1.3/introduction.html):
  exchange entities and provenance using the existing versioned JSON-LD document,
  preserving identifiers instead of introducing a second ontology.

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

The isolated scientific execution check requires Docker:

```sh
docker build --tag swarmx-research:validation apps/desktop/resources/python
SWARMX_TEST_DOCKER_IMAGE=swarmx-research:validation pnpm vitest run apps/desktop/tests/research-environment.test.ts
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

The product exports RO-Crate metadata and individual verified artifact bytes. It does not claim
that metadata alone is a portable archive containing all data. Figures can be edited through
Python/Pillow and plotting code; a generative image service is not configured. The current
Jupyter recipe uses the Docker daemon's native amd64 or arm64 image. macOS packaging is implemented
as described in [macOS releases](macos-release.md); a successful build on one architecture does not
verify another. Windows/Linux installer support, broader live Harness coverage and independent
security assessment require their own implementation or execution evidence. Docker/Host compromise
is outside the research container isolation contract.

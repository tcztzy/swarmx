# Research workbench and SoftwareX readiness

## Contract

The conversation uses assistant-ui. Research objects, executable figures, environments
and provenance remain inspectable outside the conversation. The Science journal and
RO-Crate are the data sources, not UI state.

- Notebook and figure code use Docker and an immutable image ID, with no host fallback.
  Containers have no network, a read-only root filesystem, dropped capabilities, no
  new privileges and bounded CPU, memory and processes. Only the workspace and
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
- Named projects have separate canonical directories, sessions, journals, settings and research
  objects. Project-scoped requests stay bound across navigation; traversal and symlink escapes
  are rejected. The sidebar manages projects; global settings contain only user preferences.
- Research UI supports research collection creation, entity search, graph/list inspection,
  artifact previews, executable figure creation/editing and RO-Crate inspection and
  export. Edits preserve revisions, hashes and source relationships. Empty and failed
  states are actionable; illustrative examples are explicitly identified.
- Runs distinguish success, failure and cancellation. Failed code cannot register a
  stale output as a newly successful figure.

## Delivered components

| Initial gap | Implementation | Acceptance evidence |
| --- | --- | --- |
| Python execution on the host; JupyMCP bypassing the injected process runtime | Desktop uses stateless Python in an immutable Docker image; explicit mounts, no credentials/network, bounded resources | Real Docker confinement, read-only, declared inputs and child cancellation tests |
| Implicit configuration and work directory | Named project catalog, per-project directories/settings, scoped gateways, native Codex/Claude permission mapping | Catalog restart/deduplication, concurrent project routing, cross-project rejection and native API mapping tests |
| No environment setup or export | Official Jupyter Data Science base pinned by digest, native platform, setup/cancel/inspect and actual installed Python package export | Real setup plus image/package inspection; failed setup shown in UI |
| Research assets inaccessible from conversation | assistant-ui conversation with asset/editor side view; Observe groups react-o11y, scientific runs and RO-Crate | Renderer interaction tests and Chrome workflow checks |
| Single-language interface | English/Chinese UI, accessible labels and formatting; Host-persisted language preference | Translation coverage, authenticated persistence and draft-preserving language-switch tests |
| Graph data inaccessible to researchers | React Flow projection of RO-Crate, semantic edges, search and neighborhoods; JSON-LD inspector/export | Identity/relation/filter tests and visual graph inspection |
| Failed code could capture an old output | Failed runs retain error evidence and cannot create a figure artifact | Regression test with an existing stale output; real Docker figure test |
| Obsolete CI/release paths | Linux/macOS quality matrix, Docker integration job, source archive and checksum draft release | Local equivalents of quality commands; remote GitHub jobs require a push |
| Missing machine-readable software citation | CITATION.cff based on manuscript authors and repository license | Author verification and archived version still required |

Initial audit: native Agents, an execution journal, scientific entities, immutable
artifacts, figure proposals and RO-Crate export exist. Research subprocesses run on
the host; the renderer exposes only conversations; configuration is implicit;
CI/release scripts reference deleted packages and commands. Typecheck passes before
this work. Existing uncommitted work is preserved.
The obsolete `paper:model` entry point also referenced a removed prototype module. It is
retired; archived bounded-model outputs are historical artifacts, not checks of the current
implementation. The current release gates use executable tests and `paper:demo`.

## Local acceptance, 2026-09-06

Validated on macOS arm64 with Node 26 and Docker, using the pinned official Jupyter
Data Science base on native arm64. The resolved image contains Python 3.13.15 and
240 installed Python packages. R and Julia are included by the base; the application
currently executes Python only.

- `pnpm lint`, `pnpm docs:check`, `pnpm typecheck` and `pnpm build` pass.
- `pnpm test`: 272 pass, 5 skip (the opt-in Docker and native Agent tests), across
  40 files. The environment test file also passes all 5 tests when enabled: 2 policy
  tests and 3 real Docker scenarios covering setup/package capture, versioned figures,
  confinement and cancellation.
- The real Codex smoke test passes with two native turns, persisted history and a traced
  `swarm status` MCP call. Its observer approves only that exact read-only elicitation.
  The Hermes smoke test remains skipped; no authenticated live validation is claimed
  for Claude, Hermes or OpenClaw.
- `pnpm paper:demo` passes: 4 artifacts, 4 tasks, committed admission and 10 checksums.
- Chrome checks cover project/question creation, environment setup and rebuild, PNG
  generation, SVG style revision, preserved original hash/source, recorded executions,
  graph selection/neighborhoods and RO-Crate export. The downloaded JSON-LD passes the
  repository schema and contains both figure versions. At 720 px, the document has no
  horizontal overflow; the checked page reports no browser console errors or warnings.
- The conversation side view is checked in Chinese and English: selected-figure references
  append to the existing draft; full-page Settings preserves it; Observe exposes live
  react-o11y completion, scientific runs and RO-Crate from one entry. Language persists
  through authenticated Host settings; the figure editor retains unsaved source when
  switching to Observe. Jupyter setup from Settings resolves the same native image as
  the integration test.
- Rebuilding resolves the same image ID and leaves no SwarmX containers behind.
  API tests verify import, byte-for-byte downloads and cross-workspace rejection.
  Chrome's extension denied automated file upload, and its organization policy blocked
  the direct image download. These two browser flows are not claimed as passed;
  browser permissions and organization policies were not changed.

The renderer build reports a large main-bundle warning. Remote CI jobs, signed desktop
installers and the manuscript's empirical provenance study were not run in this acceptance.

## Design evidence

- [SoftwareX reviewer form](https://legacyfileshare.elsevier.com/promis_misc/softwarex-reviewer-form.pdf):
  installation, reproducible experiments, documented dependencies, licensing, tests
  and scientific usefulness are review criteria. UI completeness alone is insufficient.
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

## Publication gate

Commit all pending repository changes as the freeze baseline before further validation or
development. After that commit, change production code only to fix a reproduced bug blocking
installation, the manuscript example, required release checks, or the integrity of its records.
Keep blocker fixes in separate commits. Complete the reproducible scientific-use example,
source release and manuscript against this baseline plus any blocker fixes; narrow unsupported
manuscript claims to the evidence.

1. Record one complete workflow on the release candidate using the existing tools: a native
   Agent delegates analysis, Science records the input, execution and figure, the result is
   checked, Memory stores the sourced finding, and a new session retrieves it. Preserve the
   commands, environment, outputs, provenance and screenshots needed to inspect and reproduce
   the example. Reuse existing data and scripts where suitable; synthetic fixtures establish
   software behavior, not biological findings or researcher productivity.
2. Verify installation from a clean checkout and run the existing release checks. Tag the
   validated revision, publish its archive/checksums, and cite that exact revision. The
   release workflow creates a draft; publication remains a separate step.
3. Align the manuscript's architecture, approval semantics, examples, validation and source
   link with the release. Compile the paper with resolved references and verify authors,
   affiliations, software citation and demonstration-data licenses.

The SoftwareX reviewer form asks for evidence proportionate to the claimed or expected impact;
it does not prescribe a participant study for every software paper. The existing
`provenance-study-protocol.md` is a possible follow-up for measuring user benefits.
Comparative multi-agent gains, researcher productivity and generalizable learning
remain unevaluated unless measured. They do not require new experiments for this submission
while those claims remain outside its scope.

Signed installers, additional Harness validation, UI polish and architecture refactoring are
deferred unless a promised release behavior demonstrably depends on them. Once the example,
release checks and manuscript agree, stop expanding scope and proceed to publication.

The product exports RO-Crate metadata and individual verified artifact bytes. It does not claim
that metadata alone is a portable archive containing all data. Figures can be edited through
Python/Pillow and plotting code; a generative image service is not configured. The current
Jupyter recipe uses the Docker daemon's native amd64 or arm64 image. Docker/Host compromise,
independent security certification, authenticated live behavior of every native Harness and
cross-platform installer packaging are outside the validated claims.

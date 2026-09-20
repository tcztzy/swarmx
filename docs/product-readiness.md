# Research workbench readiness

## Contract

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

## Delivered components

| Initial gap | Implementation | Acceptance evidence |
| --- | --- | --- |
| Python execution on the host; JupyMCP bypassing the injected process runtime | Desktop uses stateless Python in an immutable Docker image; explicit mounts, no credentials/network, bounded resources | Real Docker confinement, read-only, declared inputs and child cancellation tests |
| Implicit configuration and work directory | Fixed execution directory, shared private settings, validated IPC and external protocol calls | Directory rejection, persistent settings and native permission API tests |
| No environment setup or export | Official Jupyter Data Science base pinned by digest, native platform, setup/cancel/inspect and actual installed Python package export | Real setup plus image/package inspection; failed setup shown in UI |
| Research assets inaccessible from conversation | assistant-ui conversation with asset/editor side view; Observe groups react-o11y, scientific runs and RO-Crate | Renderer interaction tests and desktop workflow checks |
| Single-language interface | English/Chinese UI, accessible labels and formatting; Host-persisted language preference | Translation coverage, authenticated persistence and draft-preserving language-switch tests |
| Graph data inaccessible to researchers | React Flow projection of RO-Crate, semantic edges, search and neighborhoods; JSON-LD inspector/export | Identity/relation/filter tests and visual graph inspection |
| Failed code could capture an old output | Failed runs retain error evidence and cannot create a figure artifact | Regression test with an existing stale output; real Docker figure test |
| Obsolete CI/release paths | Linux/macOS quality matrix, Docker integration job, source archive and checksum draft release | Local equivalents of quality commands; remote GitHub jobs require a push |
| Missing machine-readable software citation | CITATION.cff based on software authors and repository license | Author verification and archived version still required |

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

## Validated scope

Publication planning and evaluation protocols live in the separate `swarmx-paper` project.
Comparative multi-agent gains, researcher productivity and generalizable learning remain
unevaluated unless measured; passing software checks does not establish them.

The product exports RO-Crate metadata and individual verified artifact bytes. It does not claim
that metadata alone is a portable archive containing all data. Figures can be edited through
Python/Pillow and plotting code; a generative image service is not configured. The current
Jupyter recipe uses the Docker daemon's native amd64 or arm64 image. Docker/Host compromise,
independent security certification, authenticated live behavior of every native Harness and
cross-platform installer packaging are outside the validated claims.

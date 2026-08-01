# SwarmX Product Specification

Status: current

SwarmX is a local-first desktop workspace and TypeScript platform for running
direct model agents and ACP-compatible coding agents. It composes a runtime
Harness with an independent Model, gives that Agent bounded access to a Project,
and preserves the resulting task as a resumable Session.

This document defines the durable product contract. It is intentionally not an
implementation map, test plan, backlog, changelog, or incident log.

## Product model

| Concept | Contract |
| --- | --- |
| Project | A local folder bookmark and the containment root for task tools. It is not a remote workspace or authorization domain. |
| Provider | A connection and credential source that may supply Models. It does not own Model identity. |
| Model | A primary entity with a stable id, API compatibility, and capabilities. |
| Harness | A reproducible runtime recipe: Software, selected Skills and MCP servers, Project context, delivery capabilities, and permission policy. |
| Agent | Exactly one Harness paired with one Model. Its identity is `harnessId:modelId`; Provider routing and effort do not change it. |
| Session | The canonical, resumable conversation and task record, persisted as an append-only event log. |
| Workflow | A `SwarmConfig` graph of agents, tools, nested swarms, and explicit edges. |

## Product invariants

1. **One workflow format.** `SwarmConfig` is the only persisted execution graph.
   Imports, editors, previews, CLI calls, and Desktop execution converge on it.
2. **Explicit composition.** Model, Provider, Harness, Agent, Extension, and
   Project remain separate concepts. The ordinary composer asks for Harness,
   Model, and Effort; trusted runtime code resolves supply details.
3. **Host-owned tools.** Direct SwarmX tasks may receive bounded Project tools.
   External ACP Harnesses retain their native tools and permission systems and
   never receive a duplicate SwarmX tool surface.
4. **Local-first authority.** Sessions, Projects, settings, credentials, and
   managed media are local by default. Optional server and telemetry features
   are explicit and disabled unless configured.
5. **Isolation by construction.** The Electron renderer has no direct
   filesystem, process, network-credential, or secret access. Privileged work
   crosses a typed Preload/Main boundary and is authorized for the main frame.
6. **Schema and secret boundaries.** External and persisted data is validated at
   its boundary. Metadata stores secret references or status, never plaintext
   credentials, raw environment snapshots, or secret-bearing logs. The dedicated
   local Provider auth file is an explicit user-editable plaintext credential
   store and remains outside metadata, Renderer data, and logs.
7. **Explicit side effects.** Discovery and planning are read-only. Installation,
   repair, trust changes, permission grants, destructive actions, and external
   writes require an explicit action and the appropriate confirmation.
8. **Honest capabilities.** SwarmX advertises only behavior it implements.
   Unsupported host tools, media transports, provider features, and ACP
   capabilities fail clearly instead of silently degrading to different
   semantics.
9. **Portable core.** Generic schemas and decision helpers stay host-neutral and
   side-effect free. Filesystem, process, keychain, installer, and UI effects
   belong to host adapters.

## Required capabilities

### Runtime and workflows

- Run a direct single-Agent task without requiring a workflow.
- Parse, validate, preview, and execute `SwarmConfig` workflows.
- Structurally import n8n workflow JSON into `SwarmConfig`, preserving topology
  and inert metadata without importing secrets or executing n8n node runtimes.
- Run native Provider APIs and external ACP Harnesses with streaming,
  cancellation, MCP integration, and resumable sessions where the host supports
  them.
- Keep ACP server capabilities, prompt history, working directory, resources,
  MCP state, and cancellation consistent with the persisted Session.

### Composition and extensions

- Discover Models from explicit Provider connections, Extension metadata, and
  manual declarations, then resolve a compatible `Harness x Model` route.
- Create reusable Custom Agents from Harness recipes and Models, including
  deterministic Agent/Model-specific Skill variants.
- Load Extension, marketplace, Agent profile, Skill, MCP, connector, LSP, hook,
  command, asset, permission, and UI-contribution metadata as passive inventory.
- Resolve and display composition readiness before execution. Inventory loading
  alone never executes bundled code, starts services, changes trust, or mutates
  host configuration.
- Read native Claude Code Markdown and Codex TOML Agent definitions without
  activating or rewriting their source files.

### Project tools and permissions

- Let direct SwarmX tasks inspect, edit, search, and validate the active Project
  through bounded host tools with containment, stale-read protection,
  cancellation, output limits, and a fail-closed platform sandbox where
  supported.
- Project direct tools as `auto`, Claude Code, Codex, or Kimi Code contracts
  while dispatching through the same safety boundary. The resolved style is
  fixed for a persisted Session.
- Support bounded foreground, background, PTY, LSP, planning, child-Agent, and
  scheduled work only when the selected contract has real backing behavior.
- Resolve direct-tool authority from managed, Project, personal, Agent, and
  conversation layers. Denials and lower-authority ceilings win; approval never
  escapes the operating-system sandbox.

### Desktop experience

- Group local task history by Project and support resume, title editing, pinning,
  archiving, timing, cancellation, and safe migration from supported legacy
  Session formats.
- Stream reasoning, commentary, tool progress, and results while a task runs;
  collapse transient work after completion and retain the final answer and
  canonical trace.
- Render conversational Markdown, math, code, sanitized trace cards, and typed
  attachments without executing content or fetching arbitrary remote media.
- Attach, persist, transport, preview, and capability-gate images, PDFs, audio,
  video, text, and general files without storing inline binary data in Sessions.
- Support transient read-only side chats anchored to a parent snapshot, with
  explicit promotion as the only route into normal persisted task history.
- Provide dedicated Settings surfaces for Providers, permissions, Extensions,
  Custom Agents, runtime health, and updates; Doctor inspection is read-only and
  repair is separately confirmed.

### Reusable platform

- Export validated TypeScript contracts for orchestration, Sessions, Providers,
  Models, Harnesses, Agent profiles, Extensions, Skills, actions, context,
  normalized rendering, telemetry, managed dependencies, and deterministic
  autonomy primitives.
- Keep browser-safe public subpaths free of Node-only imports.
- Provide a CLI, ACP server adapter, runtime Doctor, and Desktop-first npm
  launcher without launching GUI code during package installation.
- Package releases from one version-aligned, quality-gated commit; release
  artifacts must not contain generated residue, credentials, or known avoidable
  production vulnerabilities.

## Explicit limits

- SwarmX core does not own downstream domain workflows such as biosecurity,
  HPC/LSF execution, scientific interpretation, benchmark policy, paper
  generation, or memory-claim admission.
- It does not provide cloud team identity, email activation, hosted knowledge
  storage, or product-specific analytics and collaboration policy.
- n8n import does not make SwarmX an n8n node runtime.
- Extension manifests are declarative metadata, not executable UI or script
  delivery. Executable UI components must be registered by the embedding host.
- Raw conversations, prompts, responses, source files, terminal output, and
  credentials are not telemetry payloads.
- Claude-compatible `PowerShell`, `SendMessage`, and `Workflow` remain absent
  until SwarmX has the corresponding Windows sandbox, concurrent team runtime,
  and persisted workflow VM. Similar-looking existing features are not aliases
  for those contracts.

## Acceptance and change policy

The zod schemas, public TypeScript types, and focused tests are the executable
authority for field-level and protocol-level behavior. `DESIGNS.md` explains
architecture; `docs/` explains feature details and operator behavior.

Changes must preserve the invariants above and pass validation proportional to
their scope. The canonical Node quality gate is:

```shell
pnpm run ci:node
```

Tagged releases additionally run the production dependency audit and the
repository's release workflow gates.

Keep this file short:

- Add a requirement only for a durable user-visible capability or cross-cutting
  invariant.
- Put schemas and implementation details in code, architecture rationale in
  `DESIGNS.md`, feature guidance in `docs/`, and acceptance details in tests.
- Track unfinished work in `ROADMAP.md` or the issue tracker.
- Track completed work and failures in Git history and release notes.
- Do not add task ids, test ids, file-by-file implementation indexes, or bug
  ledgers here.

The former `G/C/I/V/T/B` ledger remains available in Git history:

```shell
git show 780fb8e:SPEC.md
```

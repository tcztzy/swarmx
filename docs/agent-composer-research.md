# Agent Composer research archive (superseded)

> **Superseded on 2026-07-12.** This file preserves the 2026-07-11
> pre-migration investigation and external-source links. Its Provider-owned
> Model proposals, `providerProduct`, default-Model, Harness override, and
> provider-suffixed selector shapes are historical and must not be implemented.
> The authoritative current contracts are `SPEC.md`, `DESIGNS.md`, and
> `docs/index.md`.

## Current implementation after the migration

- `Model` is an independent primary entity with a stable id, runtime name, API
  protocols, and capability references.
- `ModelSupply` is the explicit many-to-many link between a Model and a Provider
  connection. Only route aliases and bridge metadata live on that link.
- Provider is a supply label/connection. It owns no Model catalog, default
  Model, Harness override, compatibility decision, or Agent identity.
- The capability matrix and Agent id are always `harnessId:modelId`. Effort is
  the only user-facing execution option; ModelSupply and Provider routing are
  internal runtime details.
- Harness model control is `direct`, `session`, or `unsupported`. Direct
  Harnesses receive request-scoped runtime variables. ACP Harnesses negotiate
  stable session `configOptions`, applying Model before refreshed effort, with
  legacy model negotiation only when no stable Model option exists.
- Session Harnesses receive explicit Model selections only for evidence-backed
  routes: a fixed adapter runtime id or a ModelSupply that names the adapter and
  runtime model. Anthropic and official Codex discovery supplies name their ACP
  Harness only for runtime ids proven in the pinned adapter session surface;
  Codex-only catalog drift remains direct-SwarmX-only. OpenCode and Hermes
  remain absent until their provider-prefixed runtime ids are imported.
  OpenClaw remains an explicit unsupported Model-switching gap.
- The desktop lists each executable Harness x Model pair once and exposes
  exactly Harness, Model, and Effort. Manual composition sends `modelId`, an
  optional internal supply id, and effort; trusted core code validates the
  route before launch.
- yallm remains an internal route adapter. Protected container launches
  allowlist Model/bootstrap variables and translate loopback bridge URLs to the
  container host without changing Model or Agent identity.

Everything below this point is an archived snapshot. Statements using “current”
refer to the pre-migration 2026-07-11 working tree, not the repository today.

Research cut-off: **2026-07-11**. Repository snapshot: SwarmX **3.0.1**,
`@agentclientprotocol/sdk` **0.22.1** in the lockfile, OpenAI JavaScript SDK
**6.39.0**. The findings below describe the unreleased working tree at that
cut-off. The first intended GEEPilot compatibility floor is a future published
`@swarmx/core >= 3.1.0` **and** `@swarmx/desktop >= 3.1.0`; the local 3.0.1
working tree is not that floor and must not be treated as a published API
promise.

This document uses three evidence levels deliberately:

- **Product** means a behavior documented by the product's official project or
  documentation.
- **Adapter** means a behavior implemented by the exact ACP adapter revision
  inspected here. It does not imply that the underlying product has native ACP.
- **SwarmX wiring** means the behavior is connected end-to-end in this
  repository. A product or adapter capability is not a SwarmX capability until
  that mapping is present and tested.

Only first-party project documentation, first-party repositories, the ACP
specification, and this repository were used. Model names and controls are not
inferred from their spelling.

## Archived conclusions from the 2026-07-11 snapshot

1. `agent = harness + model` is not sufficient as a universal execution rule.
   For a **provider-managed** harness, the minimum useful selection is
   `harnessId + providerProfileId + model`, with `effort` as a separate
   capability-gated field. For a **harness-managed** ACP backend, the safe
   minimum is the harness plus its negotiated/default session configuration;
   SwarmX must not fabricate provider, model, or effort fields. Credentials are
   references or host environment, never composition data. Permission policy,
   working directory, and environment remain distinct concerns.
2. A configured provider inventory is the only trustworthy static source for a
   provider-managed model menu. Provider records carry an explicit
   `providerProduct` in addition to transport `kind`, because OpenAI-compatible
   transports do not establish product identity. The desktop also projects
   runtime readiness; disabled or currently unusable profiles are excluded.
   For the built-in ACP harnesses, model and effort controls stay hidden until
   live ACP `configOptions` discovery and setting are implemented.
3. OpenClaw **does** provide an official `openclaw acp` command. It is a stdio
   bridge to a running OpenClaw Gateway, not a self-contained local coding
   harness. CLI presence alone is therefore not operational readiness.
4. The current source-backed native OpenAI Chat set covers GPT-5, GPT-5.1,
   GPT-5.2, GPT-5.4, GPT-5.4 mini/nano, and GPT-5.5 with their exact model-page
   defaults. GPT-5.6 is Responses-only in this implementation; its effort
   values are `none | low | medium | high | xhigh | max`, default `medium`.
   “Ultra” is not an API effort value. In Codex, ultra-style operation is
   orchestration, not a synonym for `max`.
5. DeepSeek V4 documents `reasoning_effort: high | max`, default `high`, and a
   separate `thinking` switch. This applies to the documented V4 contract; it
   must not be projected onto legacy aliases.
6. Anthropic's current API uses `output_config.effort` on supported models.
   Older manual extended-thinking token budgets and Claude Code's
   `ultracode`/`ultrathink` behaviors are different controls.
7. Request-scoped cancellation is now wired across renderer, preload, Electron
   main, core, native OpenAI calls, MCP SDK calls, and ACP. The current request
   `AbortSignal` is passed directly to OpenAI and MCP calls. ACP cancellation is
   attempted first; bounded process termination remains the fallback.
8. The ACP client can consume the legacy `newSession.models` surface and call
   `unstable_setSessionModel`, but it does not yet consume current ACP session
   `configOptions` or set model-specific effort. All built-in custom ACP
   harnesses are therefore marked `modelSelection: "harness"`: the Composer
   exposes neither static model nor effort rows for them and execution uses the
   harness-negotiated default. Product/adapter support is not proof that a
   SwarmX Composer selection reaches that adapter.
9. Open-source asset licensing and trademark permission are separate. Bundled
   assets need a revisioned provenance record and deterministic fallback.
10. ACP session discovery is opt-in and gated twice: Electron accepts only
    explicitly requested harness IDs that are both ready and native, and the
    ACP client checks advertised list/load capabilities before invoking those
    optional methods. Launcher presence still does not prove authenticated
    operational readiness.

## Archived pre-migration SwarmX architecture

### Ownership and data flow

| Layer | Current responsibility | Boundary that must remain |
|---|---|---|
| Renderer | `packages/desktop/src/renderer/src/App.tsx` owns Composer state, the Agent menu, `idle | running | stopping`, a renderer-generated UUID, local session presentation, and the invoke call. | It may select a declared composition; it must not replace or mutate the preload bridge. |
| Preload | `packages/desktop/src/preload/api.ts` builds the frozen invoke-only API; `index.ts` binds it to `ipcRenderer` and `contextBridge`. | No Node/Electron authority is exposed directly to the renderer; downstream hosts import the factory instead of copying or mutating the bridge. |
| Electron main | `packages/desktop/src/main/ipc.ts` validates the IPC route, resolves protected harness backends, builds or resolves a composition, and owns active requests through `DesktopRequestRegistry`; `main/library.ts` re-exports host-safe integration points without starting Electron. | Request ownership is scoped to the sending `webContents`; one window cannot cancel another. Hosts must not register duplicate handlers/request tables. |
| Core composition | `packages/core/src/extensions.ts` resolves agent profile, harness, provider profile, model, effort, plugins, skills, MCP, context, permission summary, visual metadata, and host. | Invalid or ambiguous requirements block execution. Secrets are resolved at runtime, not serialized in composition. |
| Native agent | `packages/core/src/agent.ts` executes the built-in SwarmX backend through OpenAI-compatible Chat Completions and MCP tools. | Only a verified native API mapping may be emitted. The built-in harness currently advertises only `openai_chat`. |
| ACP client | `packages/core/src/acp.ts` spawns one stdio ACP child for an operation, initializes it, creates or loads a session, sends prompts, converts session updates, and tears the child down. | Capabilities must be negotiated; unsupported methods or config options must not be assumed. |
| Persistence | Native sessions are JSON records under `~/.swarmx/sessions`; ACP session discovery delegates to ready native harnesses only. | A local SwarmX session ID is not interchangeable with an ACP harness session ID. Optional ACP list/load calls require advertised capabilities. |

Send path:

`Composer -> frozen preload API -> agent:send -> DesktopRequestRegistry ->
resolve AgentComposition -> Swarm/Agent -> native OpenAI-compatible API or ACP
child -> completed invoke response -> local session save`.

The core can consume streaming model/ACP updates, but desktop `agent:send` still
returns a completed IPC invoke response. There is no renderer-facing streaming
IPC channel in this snapshot.

Stop path:

`Stop -> agent:cancel(requestId) -> owner-checked main registry -> core request
AbortController -> native OpenAI/MCP AbortSignal and request checks, or ACP
session/cancel -> SIGTERM -> SIGKILL`.

### Composition shape and responsibility split

`AgentCompositionSchema` currently has first-class fields for identity and
display metadata, `agentProfileId`, `harnessId`, `providerProfileId`, `model`,
`effort`, skills, MCP servers, plugin selections, definition, context, memory,
permissions, visual metadata, and execution host. Profiles may also provide a
permission mode and effort default.

The split should remain:

- **Core composition:** harness selection mode, and—only for a
  provider-managed harness—provider profile, exact model, and optional verified
  effort; plus agent/profile capabilities and a permission policy reference.
- **Harness-managed ACP composition:** harness identity plus the omission of
  provider/model/effort until live session config is discovered. Omission means
  “use the harness-negotiated default,” not a hidden SwarmX fallback model.
- **Trusted runtime/host:** current directory, environment allow-list, process
  isolation, secret resolution, account login, and container/Gateway access.
- **Host extension:** provider/model inventory, project agents, credentials by
  reference, plugins/skills/MCP, and optional registered UI contributions.
- **SwarmX desktop:** the Composer DOM, selection/correction logic, send/stop
  state machine, and stable menu geometry.

Raw environment values and `cwd` should not be added to user-facing composition
records merely to make a harness work. They already belong to `ProcessOptions`
and the runtime boundary.

### Inventory and readiness

Built-in harness definitions live in `packages/core/src/harness.ts` and are
projected into the built-in extension bundle. `swarmx` is provider-managed. Its
Composer models come from extension provider profiles, are filtered by harness
transport compatibility and runtime readiness, and derive effort from the
central exact-model capability registry. Claude Code, Codex, OpenCode, Hermes,
and OpenClaw are harness-managed. Their model resolver intentionally returns an
empty list, so the desktop does not display a synthetic provider/model/effort
catalog while ACP `configOptions` are not consumed. The built-in bundle itself
declares no provider or model inventory.

Provider identity and readiness are separate inputs:

- `kind` identifies the API transport contract such as `openai_chat`;
  `providerProduct` identifies the actual product such as `openai` or
  `deepseek`. Exact model capability lookup requires both, preventing the same
  model-shaped string on two compatible endpoints from inheriting the wrong
  effort enum.
- The Electron inventory response annotates each provider with
  `runtimeReady`/`runtimeNote`. Disabled profiles, unsupported secret sources,
  and missing environment secrets are not offered as executable model choices;
  agent plans using them are projected as blocked, and main rechecks readiness
  before composition execution. A profile without a secret reference may be
  ready at this boundary, but that still does not prove account entitlement or
  endpoint health.
- A missing `providerProduct` fails closed for reasoning: the model may remain
  a configured provider choice, but SwarmX emits no effort parameter because
  product-specific capability identity is unproven.
- A provider-managed send without a resolved model is disabled in the renderer
  and rejected by composition resolution. There is no hidden `gpt-4o` or other
  fallback. Harness-managed omission is the separate, explicit negotiated-
  default case described above.

Desktop environment detection currently proves only a launcher or container
runtime condition:

- Native mode runs `<command> --version` for Claude Code, Codex, OpenCode,
  Hermes, and OpenClaw, storing only the semver token.
- Claude and Codex adapter launch uses pinned `npx` packages. CLI detection does
  not prove package resolution, Node compatibility, authentication, or ACP handshake.
- Protected mode prefers Apple Container where the harness is self-contained;
  Claude ACP **0.58.1** and Codex ACP **1.1.2** are pinned there. OpenCode,
  Hermes, and OpenClaw remain native so their installed CLI, account state, and
  user configuration are available. A protected path for them needs explicit,
  tested state/config mounts rather than a fresh unpinned install on every run.
- `ready` therefore does not prove credentials, account model access, OpenClaw
  Gateway reachability, ACP initialization, or a successful prompt.

Operational readiness needs a separate, bounded smoke probe: executable,
version, ACP initialize, required capabilities, new session, model/config
surface, authentication, and—where applicable—Gateway reachability.

The 2026-07-16 no-prompt smoke audit exercised the resulting local matrix rather
than trusting discovery metadata: all 11 Claude Code cells and all 6 Codex cells
were selected through stable ACP session configuration in protected Apple
Container launches. The live OpenCode session advertised 123 provider-prefixed
ids and Hermes advertised 4 provider-prefixed ids, but neither matched the
catalog's bare runtime ids, so both executable columns correctly remained
empty. OpenClaw also remained empty because it exposes no ACP model switch.

The native state-preservation policy is intentional. OpenCode runs the installed
`opencode acp` against the user's provider/login configuration; Hermes runs the
installed `hermes acp` from the existing `~/.hermes/hermes-agent` installation;
OpenClaw runs `openclaw acp` where the user's configured Gateway route and
authentication are reachable. Their official installer definitions are setup
fallbacks for a missing executable, not per-run ephemeral environments. In
particular, Hermes is already present on this machine, so this task performs no
Hermes install and no remote Git operation. A future protected mode is valid
only if it explicitly mounts/routes the same required state, pins code, and—for
OpenClaw—provides a secure Gateway connection.

Session discovery is deliberately narrower than execution readiness. Desktop
`session:listGrouped` does not auto-probe ACP harnesses on mount: a host must
explicitly request harness IDs, which are then restricted to launchers reported
`ready` in **native** mode. `session:loadDiscovered` applies the same ready/native
gate. This avoids installing protected adapters or starting an unconfigured
OpenClaw Gateway bridge merely to populate the sidebar. The ACP client then
checks `sessionCapabilities.list` for listing and `loadSession` for loading
before it calls either method. Missing capabilities produce a bounded
diagnostic rather than an optimistic request or startup noise.

Loaded ACP history is currently a **read-only view** in the desktop. The public
preload send shape does not expose `sessionId`: SwarmX does not yet perform the
required load/resume-plus-prompt sequence on one ACP connection, so the
Composer disables continuation instead of treating an ACP ID as a local
session ID or starting an unrelated new turn. Harness-level resume support in
the matrix below is therefore product/adapter evidence, not a wired SwarmX
desktop capability.

## Harness capability matrix

### Installation and detection

| Harness | SwarmX launch/install route | Current detection | Additional operational proof required |
|---|---|---|---|
| SwarmX | Bundled with the desktop | No external requirement | Provider profile, secret reference, endpoint, exact model, and one bounded API probe |
| Claude Code ACP | Pinned `npx --yes @agentclientprotocol/claude-agent-acp@0.58.1` | `claude --version` | Adapter package starts under its Node >=22 constraint, authentication, ACP initialize/new session, model/config options, prompt/cancel |
| Codex ACP | Pinned `npx --yes @agentclientprotocol/codex-acp@1.1.2`; package bundles compatible Codex | `codex --version` | Adapter/App Server startup, Codex authentication, ACP initialize/new session, model/config options, prompt/cancel |
| OpenCode | [Official installer](https://opencode.ai/docs/cli/) or platform package; SwarmX runs the installed CLI natively | `opencode --version` | Provider authentication, `opencode acp` initialize, live models/config, prompt/cancel |
| Hermes Agent | Reuse the existing `~/.hermes/hermes-agent` checkout and installed CLI; SwarmX runs it natively. This task must not clone, fetch, or pull Hermes. | `hermes --version` | Provider authentication, `hermes acp` initialize, model state, prompt/cancel |
| OpenClaw | [Official install/onboarding](https://github.com/openclaw/openclaw/blob/v2026.6.11/README.md); SwarmX uses `--no-onboard` | `openclaw --version` | Configured/authenticated Gateway, secure Gateway route, `openclaw acp` initialize/new session, prompt/cancel |

Installer success and launcher detection are intentionally distinct from the
last column. The desktop should never label a harness operational merely because
its executable or protected container exists.

| Harness | Product / license; audited version | ACP entry used by SwarmX | Model and effort source | Cancel and session surface | SwarmX applicability at cut-off |
|---|---|---|---|---|---|
| SwarmX | This repository, MIT; 3.0.1 snapshot | Built in, no ACP child | Extension provider inventory plus exact central API capability records | Native `AbortSignal`; local JSON new/list/load | Wired for OpenAI-compatible Chat only. No live account discovery. |
| Claude Code | [Claude Code](https://github.com/anthropics/claude-code), distributed under Anthropic terms; [ACP adapter 0.58.1](https://github.com/agentclientprotocol/claude-agent-acp/releases/tag/v0.58.1), Apache-2.0 | `npx --yes @agentclientprotocol/claude-agent-acp@0.58.1` | Adapter/Claude SDK session model and per-model effort options | Adapter implements cancel plus list/load/resume/fork/close/delete | Adapter-backed, not native Claude Code ACP. SwarmX exposes only proven runtime ids and applies Model, then refreshed Effort, through stable `configOptions`. |
| Codex | [OpenAI Codex 0.144.1](https://github.com/openai/codex/releases/tag/rust-v0.144.1), Apache-2.0; [ACP adapter 1.1.2](https://github.com/agentclientprotocol/codex-acp/releases/tag/v1.1.2), Apache-2.0 | `npx --yes @agentclientprotocol/codex-acp@1.1.2` | Codex App Server `model/list` intersected with pinned ACP session options; session reasoning config | Adapter interrupts the active turn; list/load/resume/close/delete are advertised | Protected auth reuses an exact read-only official auth file; SwarmX exposes only the proven catalog/adapter intersection and applies Model, then Effort, through stable `configOptions`. |
| OpenCode | [anomalyco/opencode v1.17.18](https://github.com/anomalyco/opencode/releases/tag/v1.17.18), MIT | Native `opencode acp` | Configured providers; `opencode models`; ACP model and model-variant effort options | Backing session abort; list/load/resume/fork/close | Native ACP confirmed. Marked harness-managed; SwarmX imports neither the live catalog nor config options, so Model/Effort stay hidden and the installed configuration supplies the default. |
| Hermes Agent | Local official checkout `~/.hermes/hermes-agent` at [7acaff5e](https://github.com/NousResearch/hermes-agent/tree/7acaff5ef2bcbaa22bd23b72efe60906123a4f55), `git describe` `v2026.7.7.2-270-g7acaff5ef`, package/CLI 0.18.2 (2026.7.7.2), MIT | Native `hermes acp` | Hermes provider/model configuration; `hermes model`; ACP `/model` route | Cancel/interrupt plus list/load/resume/fork | Native ACP confirmed from the clean local checkout and `hermes acp --check`. Marked harness-managed; no safe static model/effort list is exposed. The existing checkout is authoritative for this task and must not be cloned, fetched, or pulled. |
| OpenClaw | [openclaw/openclaw v2026.6.11](https://github.com/openclaw/openclaw/releases/tag/v2026.6.11), MIT | Native `openclaw acp`, a bridge to the Gateway | Gateway/session/provider configuration | Prompt cancel; list/resume/close; load is partial; shared-key cancel can be best effort | Command and native installer/detection are wired. The Composer exposes no provider-model catalog because model selection is Gateway-managed; CLI readiness is not Gateway readiness. |

License evidence is revisioned independently of feature documentation:
[SwarmX MIT](../LICENSE),
[Claude ACP Apache-2.0](https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.58.1/LICENSE),
[Codex Apache-2.0](https://github.com/openai/codex/blob/rust-v0.144.1/LICENSE),
[Codex ACP Apache-2.0](https://github.com/agentclientprotocol/codex-acp/blob/v1.1.2/LICENSE),
[OpenCode MIT](https://github.com/anomalyco/opencode/blob/v1.17.18/LICENSE),
[Hermes MIT](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/LICENSE),
and [OpenClaw MIT](https://github.com/openclaw/openclaw/blob/v2026.6.11/LICENSE).
Claude Code itself is distributed under Anthropic's
[legal and compliance terms](https://code.claude.com/docs/en/legal-and-compliance),
not the adapter's Apache license.

### SwarmX native backend

The built-in backend is the only one whose wire mapping is wholly controlled by
this repository. It creates an OpenAI client from resolved runtime
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and model, invokes Chat Completions, and can
pass a verified `reasoning_effort`. It propagates the request signal to the SDK
and passes that same signal to MCP SDK tool calls while retaining cancellation
checks around client fallback. A composition-scoped runtime environment is a
closed boundary: it never falls back to an ambient OpenAI key/base URL, and
custom ACP children receive only the harness allow-list plus resolved provider
values. For DeepSeek tool continuation, returned `reasoning_content` is retained
on the assistant tool-call message so the next request satisfies the documented
multi-turn contract.

The compatibility list was narrowed to `openai_chat`; Anthropic, Responses, and
Ollama must not be advertised as native until a corresponding client and
parameter mapping exists. OpenAI-compatible services such as DeepSeek still
need provider identity and a proven endpoint contract; “OpenAI-compatible” is
not itself a model capability.

### Claude Code via `claude-agent-acp`

Anthropic's official Claude Code materials do not document a `claude acp`
server command. SwarmX uses the ACP organization's Zed-maintained adapter, not
the Claude Code executable as an ACP server. The audited adapter package is
0.58.1, requires Node >=22, depends on ACP SDK 1.2.1 and Claude Agent SDK
0.3.205, and is Apache-2.0:

- [package metadata](https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.58.1/package.json)
- [agent/capability implementation](https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.58.1/src/acp-agent.ts)
- [model configuration and deployment overrides](https://github.com/agentclientprotocol/claude-agent-acp/blob/v0.58.1/docs/model-configuration.md)

The adapter advertises and implements load, additional directories, close,
delete, fork, list, and resume. Its cancel path interrupts the SDK and has a
bounded forced-cancel fallback. It derives model and effort options from Claude
SDK metadata; deployments can override availability with
`CLAUDE_MODEL_CONFIG`.

SwarmX detects the Claude Code CLI separately while the adapter runs through
Node.js and `npx`. Runtime probing is still required to prove package
resolution, Node compatibility, authentication, initialization, and model
config. The Composer exposes only evidence-backed Model routes. SwarmX applies
the stable general `configOptions` Model first and then the refreshed Effort
option; the older `models` response remains a fallback only when the stable
Model option is absent.

### Codex via `codex-acp`

SwarmX also uses an ACP adapter rather than a native `codex acp` command. The
audited adapter 1.1.2 starts Codex App Server, depends on ACP SDK 1.2.1, and
bundles a compatible `@openai/codex` ^0.144.0; `CODEX_PATH` can override it:

- [adapter README](https://github.com/agentclientprotocol/codex-acp/blob/v1.1.2/README.md)
- [package metadata](https://github.com/agentclientprotocol/codex-acp/blob/v1.1.2/package.json)
- [session/cancel implementation](https://github.com/agentclientprotocol/codex-acp/blob/v1.1.2/src/CodexAcpServer.ts)
- [model and reasoning configuration](https://github.com/agentclientprotocol/codex-acp/blob/v1.1.2/src/ModelConfigOption.ts)

The adapter gets models and their real supported reasoning choices from Codex
App Server. It exposes model, reasoning, fast mode, approval, and sandbox as ACP
configuration rather than requiring SwarmX to reproduce Codex's catalog.
SwarmX does not yet read and set that full config surface. Codex is therefore
harness-managed: no static Model/Effort menu is shown and the adapter's
negotiated default is used. The correct future design is handshake-driven
configuration, with any extension policy acting only as a configured upper
bound—not a fabricated Codex model list.

### OpenCode

OpenCode's official CLI includes `opencode acp`; this is native product ACP:

- [ACP documentation](https://opencode.ai/docs/acp/)
- [CLI and `opencode models`](https://opencode.ai/docs/cli/)
- [provider configuration](https://opencode.ai/docs/providers/)
- [agent/provider-specific options](https://opencode.ai/docs/agents/)
- [v1.17.18 ACP service](https://github.com/anomalyco/opencode/blob/v1.17.18/packages/opencode/src/acp/service.ts)
- [v1.17.18 ACP config options](https://github.com/anomalyco/opencode/blob/v1.17.18/packages/opencode/src/acp/config-option.ts)

Its model catalog is the intersection of configured providers, models.dev data,
and custom configuration. ACP exposes a model selector and a separate `effort`
thought-level option derived from the selected model's variants. Provider
options such as `reasoningEffort` are passthrough and therefore not universal.
The adapter aborts the backing session and supports the documented session
methods.

The built-in OpenCode declaration still records an
`anthropic | openai_chat | ollama` compatibility subset, but because the
harness is harness-managed that metadata does not populate the current Composer
model menu and is not a statement about OpenCode's full provider ecosystem.
Model/Effort stay hidden until SwarmX consumes live ACP options. The native
launcher preserves the user's installed version, authentication, and
configuration; the version probe must still be recorded with any operational
smoke result.

### Hermes Agent

The audited source is the user's existing official checkout at
`~/.hermes/hermes-agent`, clean commit
`7acaff5ef2bcbaa22bd23b72efe60906123a4f55`, described locally as
`v2026.7.7.2-270-g7acaff5ef`. `pyproject.toml` and the installed CLI report
package 0.18.2; the CLI identifies build 2026.7.7.2 and install directory
`~/.hermes/hermes-agent`. No additional Hermes clone was made, and this task
must not run `git clone`, `git fetch`, or `git pull` for Hermes:

- [package and ACP extra](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/pyproject.toml)
- [installation and `hermes model`](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/README.md)
- [programmatic/ACP integration](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/website/docs/developer-guide/programmatic-integration.md)
- [ACP server implementation](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/acp_adapter/server.py)

`hermes acp` supports session creation, prompt/update streaming, tools,
permissions, fork, cancellation, and authentication; its server implements
list/load/resume/fork and interrupt. Provider/model selection belongs to Hermes
configuration, and the documented ACP hot-swap route is `/model`. There is no
stable universal model list or universal effort enum to copy into SwarmX.

Hermes is harness-managed in SwarmX: Model/Effort stay hidden and `hermes acp`
uses the user's configured/negotiated default until live ACP configuration is
supported. SwarmX uses the installed native Hermes CLI, including the user's
existing checkout, credentials, and configuration. The CLI probe does not prove
provider credentials or ACP readiness; a future container path must preserve
state and pin its code.

### OpenClaw

The official [`openclaw acp` documentation](https://docs.openclaw.ai/cli/acp)
describes a stdio ACP server that forwards to an OpenClaw Gateway over
WebSocket. It implements initialize/new/prompt/cancel/list/resume/close; load is
partial. Cancellation routing may be best effort when clients share a Gateway
session key. Authentication should use token/password files rather than command
line secrets.

This must not be confused with the opposite direction: OpenClaw launching an
external ACP harness through its official `acpx` plugin, documented under
[ACP agents](https://docs.openclaw.ai/tools/acp-agents). `openclaw mcp serve` is
also a separate MCP surface.

The bridge requires a running, configured, authenticated Gateway. Models and
effort are Gateway/provider/session state, not a direct Anthropic catalog.
SwarmX therefore marks OpenClaw model selection as harness-managed, declares no
provider compatibility list for the Composer, and omits Model/Effort controls.
It runs the bridge natively so existing Gateway configuration can be resolved.
A future protected path needs an explicit secure Gateway route plus auth/config;
container presence alone would not establish operational readiness.

## ACP protocol implications

The [ACP protocol](https://github.com/agentclientprotocol/agent-client-protocol)
has a stable v1 specification. A client must inspect initialization and session
capabilities before calling optional operations. The stabilized
[session list contract](https://agentclientprotocol.com/announcements/session-list-stabilized)
does not make every agent implement list/load/resume.

For [prompt cancellation](https://agentclientprotocol.com/protocol/v1/prompt-turn),
the client sends `session/cancel` for the active session; the agent should stop
model/tool work, and the original prompt should complete with a cancelled stop
reason. Process termination is a compatibility fallback, not the primary ACP
operation.

SwarmX's lockfile SDK is 0.22.1 while the pinned Claude/Codex adapters use ACP
SDK 1.2.1. Stable wire compatibility is intended, but version skew plus use of
the legacy `newSession.models`/unstable set-model API requires a real
initialize/new/config/model/prompt/cancel probe for each adapter. `listSessions`
and `loadSession` are now guarded by advertised `sessionCapabilities.list` and
`agentCapabilities.loadSession` respectively, rather than being optimistically
invoked. These method guards complement—but do not replace—the desktop's
ready/native discovery gate.

The ACP client currently rejects every interactive permission request with a
cancelled outcome. That is safe, but it means a harness needing approval can
fail until a permission broker is designed; a composition permission summary
does not by itself implement interactive approval.

## Model and reasoning capability matrix

### Verified API contracts

| Provider/API and exact applicability | Control and type | Legal values; official default | Dynamic or harness difference | Current SwarmX mapping |
|---|---|---|---|---|
| OpenAI GPT-5 (`gpt-5`, `gpt-5-2025-08-07`) Chat Completions | `reasoning_effort`, enum | `minimal | low | medium | high`; `medium` | Responses uses `reasoning.effort` instead of the Chat field | Native SwarmX Chat mapping is enabled. |
| OpenAI GPT-5.1 (`gpt-5.1`, `gpt-5.1-2025-11-13`) Chat Completions | `reasoning_effort`, enum | `none | low | medium | high`; `none` | Responses uses `reasoning.effort` | Native SwarmX Chat mapping is enabled. |
| OpenAI GPT-5.2 (`gpt-5.2`, `gpt-5.2-2025-12-11`) Chat Completions | `reasoning_effort`, enum | `none | low | medium | high | xhigh`; `none` | Responses uses `reasoning.effort` | Native SwarmX Chat mapping is enabled. |
| OpenAI GPT-5.4 family (`gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano` and the exact snapshots on their model pages) Chat Completions | `reasoning_effort`, enum | `none | low | medium | high | xhigh`; `none` | Responses uses `reasoning.effort`; the three products retain separate exact capability records and source URLs | Native SwarmX Chat mapping is enabled. |
| OpenAI GPT-5.5 (`gpt-5.5`, `gpt-5.5-2026-04-23`) Chat Completions | `reasoning_effort`, enum | `none | low | medium | high | xhigh`; `medium` | Responses uses `reasoning.effort` | Native SwarmX Chat mapping is enabled. |
| OpenAI GPT-5.6 (`gpt-5.6`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`) Responses | `reasoning.effort`, enum | `none | low | medium | high | xhigh | max`; `medium` | `reasoning.mode: "pro"` is independent. Codex orchestration is not an API effort value. | Recorded as source data, but `harnesses: []`; no current SwarmX execution mapping, so no selectable effort. |
| Anthropic Messages, Claude Opus 4.5 | `output_config.effort`, enum | `low | medium | high`; `high` | Older manual thinking budgets are a different API form | Source record only; no harness mapping. |
| Anthropic Messages, Claude Opus/Sonnet 4.6 | `output_config.effort`, enum | `low | medium | high | max`; `high` | Claude Code may expose a model-specific subset/config default | Source record only; no harness mapping. |
| Anthropic Messages, Claude Opus 4.7/4.8 and Sonnet 5 | `output_config.effort`, enum | `low | medium | high | xhigh | max`; `high` at API level | Claude Code Opus 4.7 defaults to `xhigh`; product config is distinct from raw API default | Source record only; no harness mapping. |
| DeepSeek API `deepseek-v4-pro` | `reasoning_effort`, enum; separate `thinking` enabled/disabled | `high | max`; `high`; thinking defaults enabled | Compatibility accepts lossy aliases (`low`/`medium` -> `high`, `xhigh` -> `max`), but UI should expose canonical values only | Native SwarmX Chat mapping is enabled only for exact `deepseek-v4-pro`. |
| DeepSeek legacy `deepseek-chat`, `deepseek-reasoner` | Contract not safely covered by the V4 effort record | Unknown; omit | Official docs schedule legacy retirement for 2026-07-24 15:59 UTC | Explicitly unknown; no effort is emitted. |
| Ollama native `/api/chat` | `think`, boolean or model-dependent enum | For supported models, commonly `low | medium | high | max`; no universal default | GPT-OSS supports only `low | medium | high`; thinking may not fully disable. Local models come from `/api/tags`. | No native SwarmX mapping/capability record. |
| Ollama OpenAI compatibility | `reasoning_effort` or `reasoning.effort`, enum | `none | low | medium | high | max` for thinking models | Runtime/model-dependent | No current SwarmX mapping. |

Primary sources:

- [OpenAI GPT-5](https://developers.openai.com/api/docs/models/gpt-5),
  [GPT-5.1](https://developers.openai.com/api/docs/models/gpt-5.1),
  [GPT-5.2](https://developers.openai.com/api/docs/models/gpt-5.2),
  [GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4),
  [GPT-5.4 mini](https://developers.openai.com/api/docs/models/gpt-5.4-mini),
  [GPT-5.4 nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano),
  [GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5),
  [GPT-5.6 guide](https://developers.openai.com/api/docs/guides/latest-model), and
  [Chat Completions request schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Anthropic effort](https://platform.claude.com/docs/en/build-with-claude/effort),
  [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking),
  and [Claude Code model configuration](https://code.claude.com/docs/en/model-config)
- [DeepSeek API](https://api-docs.deepseek.com/),
  [Chat request schema](https://api-docs.deepseek.com/api/create-chat-completion/),
  and [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- [Ollama chat API](https://docs.ollama.com/api/chat),
  [thinking](https://docs.ollama.com/capabilities/thinking),
  [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), and
  [local model tags](https://docs.ollama.com/api/tags)

### Corrections to ambiguous product terms

- UI label **Extra High** maps only to the exact wire value `xhigh`.
- **Max** maps to `max` only where the selected model/API/harness advertises it.
- **Ultra** must never be normalized to `max`. OpenAI describes Codex
  multi-agent/ultra-style behavior as orchestration. Anthropic's `ultracode` is
  also a Claude Code orchestration mode, not another API effort enum.
- Claude Code's `ultrathink` prompt keyword does not change the API effort
  setting. Current Claude Code docs expose model-specific low through max
  choices; that product surface must not be copied to raw Anthropic API records.
- Manual Anthropic `thinking: { type: "enabled", budget_tokens: N }` applies to
  older model families. It is neither an effort enum nor a safe fallback for
  newer adaptive-thinking models.

The workflow example now omits static model IDs for ACP agents. This removes the
previous retired `claude-sonnet-4-20250514` dependency and correctly delegates
to each harness's negotiated default until live configuration support exists.

### Effective capability rule

The selectable set must be computed, not guessed:

For a provider-managed harness:

`configured, runtime-ready provider models`

`intersect harness-compatible provider/API transport`

`intersect exact providerProduct + model SwarmX parameter mapping`

`intersect live ACP session model/config options and account policy when available`.

Static records are a reviewed upper bound; runtime discovery may narrow them
but must not widen them to undocumented values. Provider kind alone is
insufficient: `openai_chat` can represent OpenAI, DeepSeek, or another
compatible endpoint. The implemented provider record therefore includes
`providerProduct`/endpoint identity in addition to transport kind and exact
model ID. Runtime-unready profiles are excluded before this intersection.

For a harness-managed ACP backend, the current set is deliberately empty in the
Composer. A future live `configOptions` implementation may construct a
session-scoped list from what that exact adapter/account/session advertises; it
must not fall back to the static provider-managed registry.

If capability is absent, ambiguous, unsupported, or unknown:

- show no Effort submenu, or show a disabled `Default` row for layout clarity;
- store no selected effort;
- emit no effort/thinking parameter;
- never translate from another provider's enum.

When a model change invalidates an effort, reset to the official or live
harness default. If there is no documented default, omit it. The UI label may
be friendly, but the stored and transmitted value must remain canonical.

### Archived capability-data proposal (do not implement)

The implemented schema is close to the required boundary:

```ts
interface ModelCapability {
  id: string;
  providerKind: ProviderKind;
  providerProduct: string;           // product identity, not transport identity
  models: string[];                  // exact identifiers, no broad guesses
  reasoningControl:
    | "none"
    | "effort_enum"
    | "token_budget"
    | "adaptive"
    | "unknown";
  supportedEfforts: string[];
  defaultEffort?: string;
  parameterMapping?: { api: string; path: string };
  effortAliases: Record<string, string>; // lossy input aliases; UI stays canonical
  harnesses: string[];               // proven end-to-end mappings only
  source: {
    url: string;
    checkedAt: string;
    version: string;
    applicability: string;
  };
}
```

The provider inventory separately records `providerProduct`, model/default
model, harness overrides, enabled state, and projected `runtimeReady` state.
Capability lookup requires provider kind, provider product, exact model, and a
proven harness mapping; absent identity produces no effort parameters.

Two additions are still needed for robust ACP use:

1. A harness mapping record that says whether model/effort is sent through a
   native API field, legacy ACP model API, ACP session config option, harness
   command/config, or is unavailable.
2. Runtime capability evidence containing adapter version, protocol version,
   config option ID/category, available values, current/default value, and
   discovery time. It is session/account state and should not silently become
   a global static registry entry.

## Cancellation design and current status

The implementation now follows these invariants:

- Renderer creates one stable UUID per send, holds it in a ref, and transitions
  `idle -> running -> stopping -> idle`. Repeat Stop in `stopping` is inert.
- The frozen preload API exposes `cancelMessage(requestId)`; renderer code does
  not overwrite `window.swarmxAPI`.
- Main rejects duplicate live IDs and records the owning `webContents`. Cancel
  is owner-checked. Destruction cancels that owner's active work.
- Core uses `AsyncLocalStorage` plus one `AbortController` per request. Token/
  identity-checked cleanup prevents a stale `finally` from deleting newer work.
- Native OpenAI calls receive the signal. MCP `callTool` receives the same
  request signal through the SDK request-options argument; MCP connection/tool
  discovery is abort-aware, and managers close clients on every Agent/Tool
  terminal path. Cancellation is rechecked before any fallback to another MCP
  client. The remote server still determines how quickly already-started work
  stops, but SwarmX no longer waits only at pre/post cooperative boundaries.
- ACP sends protocol cancellation when a session and active prompt exist. The
  cancel await is bounded at 500 ms. If it does not settle, the child is
  terminated; on POSIX the dedicated process group is signalled so npx/CLI
  descendants are included, and termination escalates from SIGTERM to SIGKILL
  after another 500 ms. Windows retains direct-child termination semantics.
- Every terminal path removes the request entry, listeners, connection, and
  child. The child is per operation, so the fallback is request-scoped.

Cancellation success means the matching live request accepted cancellation,
not that every remote provider immediately stopped billing or work. Tests need
to cover completion, failure, protocol cancel, spawn failure, repeated Stop,
window destruction, stale IDs, duplicate IDs, and a rapid old/new request
sequence.

## Brand resources and licensing

Revisioned source/provenance copies are kept under
`packages/desktop/src/renderer/public/harness-icons`; the reusable renderer
embeds the distributable artwork through
`packages/desktop/src/renderer/src/harness-icon-data.ts`, so a consuming host
does not depend on its current URL, public directory, or a runtime CDN.
Provenance at the research cut-off is:

| Asset | Revisioned source | File license / status | Trademark caveat |
|---|---|---|---|
| SwarmX | Lucide `Workflow` UI glyph from the declared `lucide-react` dependency | Lucide ISC license; a project-selected UI glyph, not a SwarmX brand asset | No external product mark is claimed. |
| Claude Code ACP | [`agentclientprotocol/registry` `claude-acp/icon.svg`](https://github.com/agentclientprotocol/registry/blob/c47300d575354b69c348bd0ed77265bb9a698336/claude-acp/icon.svg) | Registry is Apache-2.0; adapter asset, not proof of an Anthropic brand grant | Claude Code/product use remains governed by Anthropic terms and mark rights. |
| Codex ACP | [`agentclientprotocol/registry` `codex-acp/icon.svg`](https://github.com/agentclientprotocol/registry/blob/c47300d575354b69c348bd0ed77265bb9a698336/codex-acp/icon.svg) | Registry is Apache-2.0 | [OpenAI brand rules](https://openai.com/brand/) still prohibit implied endorsement and unauthorized alteration; copyright license is not trademark permission. |
| OpenCode | [`agentclientprotocol/registry` `opencode/icon.svg`](https://github.com/agentclientprotocol/registry/blob/c47300d575354b69c348bd0ed77265bb9a698336/opencode/icon.svg) | Registry asset, Apache-2.0; upstream project MIT | Mark remains its owner's. |
| Hermes | [`hermes-agent/acp_registry/icon.svg`](https://github.com/NousResearch/hermes-agent/blob/7acaff5ef2bcbaa22bd23b72efe60906123a4f55/acp_registry/icon.svg) | MIT repository | Mark remains its owner's. |
| OpenClaw | No revisioned, redistributable ACP/product asset was verified for this bundle | Use the deterministic text/glyph fallback | ACP command verification does not establish logo rights. |

Extension-defined harnesses also use a deterministic Lucide fallback today;
the current renderer does not dereference an arbitrary host-supplied `icon`
string. A future extension asset resolver must validate provenance, trust, and
packaging before custom artwork is displayed.

Each copied asset should retain source URL, immutable revision, retrieval date,
file license, whether it was modified, and a trademark note. The UI must handle
load failure without a broken image and preserve recognition in light and dark
themes. A generic tool icon must not be presented as an official logo.
Redistribution notices and applicable license texts are bundled in
`packages/desktop/THIRD_PARTY_NOTICES.md` and
`packages/desktop/third_party_licenses/`; those notices do not grant trademark
rights.

## Archived implementation/data-flow proposal (do not implement)

1. Keep `App.tsx` as the native owner of the compact Composer. The host may
   contribute inventory and registered surfaces, but must not portal over the
   Composer or override preload methods.
2. Resolve `harness -> provider profile -> exact model -> effective reasoning`
   in one core path. The renderer displays the result; it does not duplicate
   provider rules.
3. Keep every current custom ACP harness in harness-managed mode. In the future,
   fetch live ACP session config after `newSession`, intersect it with explicit
   host policy, and set model/effort using server-advertised option IDs. Cache
   only for the relevant adapter/account/session scope. Preserve the current
   no-setting/negotiated-default path for servers that do not advertise the
   options.
4. Keep the top-level Agent menu anchored as its own fixed-width, content-height
   surface. Position each submenu absolutely outside primary layout flow, with
   its own max-height and vertical scrolling, so Harness, Model, or Effort option
   count cannot move, resize, or re-render the primary menu. Apply one
   deterministic viewport flip/clamp policy.
5. On harness change, choose its valid default model; on model change, choose
   its verified default effort; on no capability, clear effort. The composition
   sent to main must contain the same selected provider/model/effort shown by
   the UI.
6. Keep cancellation request-scoped as implemented. If desktop streaming is
   later added, stream events must also carry request ID and be ignored after
   terminal cleanup.
7. Pin or report every executable adapter/harness version. A launcher probe,
   provider runtime readiness, session-method capability gate, and ACP
   operational probe are separate facts in UI and diagnostics.

Required verification includes unit tests for filtering, correction, parameter
omission, request ownership and cleanup; main/preload tests; production build;
and real Electron checks for send/stop, console errors, light/dark themes,
normal/narrow windows, and unchanged parent-menu position for all three
submenus. Browser/JSDOM-only rendering cannot validate the Electron or window
geometry requirements.

## Archived GEEPilot migration boundary

Do not change the GEEPilot repository as part of this SwarmX-only task. The
first intended migration floor is a **published** pair of
`@swarmx/core >= 3.1.0` and `@swarmx/desktop >= 3.1.0`. Both packages must be
pinned together because composition contracts, desktop IPC, preload, renderer,
and bundled assets cross the package boundary. The current 3.0.1 working tree
is unreleased implementation evidence, not the minimum compatible version.

### Published host entry points

The future 3.1.0 package surface makes the three Electron layers composable:

- `@swarmx/desktop/main` is side-effect-free with respect to app startup and
  exports `registerIpcHandlers`, `DesktopRequestRegistry`,
  `HarnessEnvironmentService`, `configureDesktopHarnessEnvironment`, `LspHost`,
  and their public types. A normal host should call `registerIpcHandlers()` and
  should **not** create a second request registry for those same channels.
- `@swarmx/desktop/preload` exports
  `createSwarmxDesktopApi(invoke)`. It returns a frozen bridge and maps the
  stock channel names (`agent:send`, `agent:cancel`, `session:*`,
  `extension:list`, `harnessEnvironment:*`, `lsp:*`, and assets) without
  importing Electron into the factory.
- `@swarmx/desktop/renderer` exports `App`,
  `createSwarmxDesktopApp(appProps)`, and public product/UI-contribution types;
  `@swarmx/desktop/styles.css` supplies the matching styles. Harness artwork is
  part of the renderer package rather than a GEEPilot public-directory lookup.

Main-process integration, after Electron is ready:

```ts
import { registerIpcHandlers } from "@swarmx/desktop/main";

process.env.SWARMX_EXTENSION_PATHS = geepilotExtensionManifestPath;
registerIpcHandlers();
```

`geepilotExtensionManifestPath` must resolve to GEEPilot's generated
`swarmx.extension.json`/supported manifest directory and must be set before
inventory requests. The manifest carries provider secret **references**, not
raw credentials; GEEPilot retains responsibility for making those references
available at invocation time.

The equivalent isolated preload should be migrated to TypeScript/ESM or bundled
so it imports the published ESM entry rather than copying its implementation:

```ts
import { contextBridge, ipcRenderer } from "electron";
import { createSwarmxDesktopApi } from "@swarmx/desktop/preload";

const swarmxAPI = createSwarmxDesktopApi((channel, ...args) =>
  ipcRenderer.invoke(channel, ...args),
);
contextBridge.exposeInMainWorld("swarmxAPI", swarmxAPI);
```

The stock main host currently resolves provider secrets from environment
references only. Before deleting a GEEPilot keychain/prompt bridge, migrate the
needed secret into a narrowly scoped environment reference at process launch,
or retain that GEEPilot-only credential adapter until a published
`registerIpcHandlers` secret-resolver option exists. Unsupported
`local_keychain`, `server_keychain`, and `prompt` references are projected as
not ready rather than silently falling back.

Renderer integration keeps GEEPilot product metadata and registered
non-Composer contributions while using the native Composer:

```tsx
import { createSwarmxDesktopApp } from "@swarmx/desktop/renderer";
import "@swarmx/desktop/styles.css";

export const SwarmxApp = createSwarmxDesktopApp({
  product: { name: "GEEPilot" },
  uiComponentRegistry: geepilotUiContributions,
});
```

`registerIpcHandlers()` and the preload factory must move as one change. The
existing GEEPilot bridge uses `swarmx:*` channels, while the published bridge
uses the stock SwarmX channel names; leaving one old side in place produces
unhandled IPC calls. Likewise, registering old and new handlers for the same
final channel is an error rather than a safe dual-run strategy.

### Exact GEEPilot deletion/replacement list

The file names and symbols below were inspected as migration targets; line
numbers are intentionally omitted because GEEPilot will continue to evolve.

- `ui/src/GeepilotChromeAdapter.tsx`: remove only the Composer overlays:
  `useComposerPortalTarget`, `useComposerSendPortalTarget`,
  `CancelableSwarmxApi`, `ComposerSendButton`, `AgentHarnessOption`,
  `AgentModelOption`, `AgentCompositionSelection`, `AgentSelectionApi`,
  `harnessApiTypes`, `harnessEfforts`, `harnessIconSources`, `HarnessIcon`,
  `effortOptionsForModel`, `hostHarnessOptions`, `selectHostHarness`,
  `providerSupportsHarness`, `AgentPicker`, and the two `createPortal` calls
  that mount `AgentPicker`/`ComposerSendButton`. Keep sidebar, writing, figure,
  and other GEEPilot-specific contribution portals.
- `ui/src/styles.css`: remove the Composer replacement block beginning with the
  “Codex-style composer” comment: the `.geepilot-chat-workspace .composer*`
  overrides, the rule that hides the native footer button,
  `.geepilot-composer-send*`, `.geepilot-agent-picker*`,
  `.geepilot-harness-icon`, and their narrow-window overrides. Keep unrelated
  GEEPilot workspace, settings, writing, and artifact styles.
- `electron/preload.cjs`: replace the hand-written `swarmxAPI` object with
  `createSwarmxDesktopApi`. Delete `selectedAgentComposition`,
  `currentAgentRequestId`, both sequence counters, the generation listener map,
  `notifyGenerationState`, the send wrapper, `cancelGeneration`,
  `onGenerationStateChange`, `removeGenerationStateListener`, and
  `selectAgentComposition`. Keep the separate `window.geepilot` bridge and its
  GEEPilot-only APIs. Because the published preload entry is ESM, compile or
  bundle the new preload instead of using CommonJS `require()` against it.
- `electron/main.cjs`: after switching the preload, remove
  `activeSwarmxAgentRequests`, the compatibility-only ACP client ownership in
  `executeSwarmxAgentCompositionCompatibility`, and the
  `swarmx:agent-send`/`swarmx:agent-cancel` handlers. The stock
  `agent:send`/`agent:cancel` path then owns stable UUIDs, window ownership,
  protocol cancellation, process fallback, and cleanup through
  `DesktopRequestRegistry`. Remove other copied generic `swarmx:*` handlers
  only after their GEEPilot-specific session/conversation semantics are either
  represented through extension inventory or deliberately retained behind a
  GEEPilot-only API; local conversation persistence is not automatically
  equivalent to `~/.swarmx/sessions`.
- `ui/public/harness-icons/`: delete `claudecode.svg`, `codex.svg`,
  `opencode.svg`, `nousresearch.svg`, and `openclaw.svg` only after the migrated
  renderer test proves package-bundled icons and fallback behavior in the
  packaged Electron app. Do not retain a second icon lookup table.
- `ui/src/App.test.tsx`: remove tests that mutate `window.swarmxAPI`, manually
  drive `onGenerationStateChange`, or assert the portal picker. Replace them
  with host integration tests that render the published SwarmX App and supply
  GEEPilot extension inventory.
- `tests/electron_main_local_mode.test.cjs`: replace source-text assertions for
  `harnessEfforts`, `selectAgentComposition`, `cancelGeneration`,
  `swarmx:agent-cancel`, and `activeSwarmxAgentRequests` with packaged API tests
  for frozen preload wiring, exact stock channels, owner-scoped cancellation,
  and extension inventory. Retain business-specific local-mode tests.
- `design-qa.md`: supersede the Composer screenshots/checklist after real
  Electron migration. In particular, remove the unsupported claims that GPT-5.6
  has an `Ultra` effort value or that effort can be selected from a generic
  harness-only table.

GEEPilot should keep its project agents, provider/credential storage,
skills/MCP/plugins, registered non-Composer UI contributions, local scientific
workflow/context, and business-specific persistence. Its extension provider
profiles must add explicit `providerProduct` values; a profile without one may
remain visible as a configured model, but it receives no product-specific
Effort control. Profiles must also resolve their secret into the supported
runtime boundary or they will be marked not ready and omitted from the Model
menu. GEEPilot's old explicit ACP model/effort environment overrides are not a
compatibility path: all current custom ACP harnesses are harness-managed until
live `configOptions` support lands.

### Compatibility and rollback matrix

| State | Package pair | Composer / IPC owner | Compatibility behavior | Rollback |
|---|---|---|---|---|
| Existing bridge | Current GEEPilot pin, including SwarmX 3.0.1-era integration | GEEPilot portals and `swarmx:*` IPC | Keep all current compatibility code; do not import the unreleased local working tree as a version floor. | No change. |
| Migration branch | Published `@swarmx/core >= 3.1.0` and `@swarmx/desktop >= 3.1.0`, pinned together | SwarmX renderer, main registration, and frozen preload factory | Switch main and preload atomically; declare `providerProduct`; expect ACP Model/Effort to be hidden and harness defaults to run. Run full GEEPilot Electron tests before deleting the old bridge commit. | Revert the migration commit and restore the previous package pair; do not mix old preload with new main. |
| Native steady state | Same tested 3.1.x pair | SwarmX owns Composer/send/cancel; GEEPilot owns only project extensions and business UI | No Composer portals, duplicate request table, duplicate icon directory, or harness/model/effort inference. | Pin the last known-good 3.1.x pair; if reverting below 3.1.0, restore the entire legacy bridge as one unit. |
| Future ACP config options | A later release that explicitly documents live config support | SwarmX session-scoped ACP resolver | Enable a custom harness Model/Effort UI only for advertised option IDs/values and the tested adapter/account/session. | Disable the live option feature and return to harness-negotiated defaults; never restore guessed static ACP catalogs. |

Migration is complete only when every GEEPilot-only agent/provider is present in
extension inventory, no duplicate handler or portal is mounted, packaged icons
resolve, and GEEPilot's real Electron tests pass against the published package
pair in light/dark and narrow/normal windows.

## Risks, unknowns, and release gates

- **ACP config interoperability:** current adapters expose modern config options,
  while SwarmX uses the older model surface and does not map ACP effort. Live
  adapter probes and config-option support are release gates for claiming ACP
  model/effort selection.
- **Protocol version skew:** SwarmX SDK 0.22.1 versus adapter SDK 1.2.1 requires
  matrix testing, not an assumption of source-level compatibility.
- **Operational readiness:** executable presence does not prove download,
  authentication, account access, model access, permissions, or Gateway health.
  Provider `runtimeReady` currently covers disabled profiles and supported
  secret availability, while ACP session discovery additionally requires a
  ready native harness; neither is an end-to-end health probe.
- **Protected runtimes:** Claude/Codex have pinned container launchers.
  OpenCode/Hermes/OpenClaw run natively because an ephemeral install would lose
  required user state. Their future protected paths require pinned code and
  explicit state/config routing; OpenClaw also needs a secure Gateway route.
- **Permission brokerage:** ACP permission requests are currently cancelled by
  the client; interactive tool approvals are not implemented.
- **Provider identity:** explicit `providerProduct` is now persisted and used by
  exact capability lookup. Existing host inventories must populate it; absence
  intentionally disables product-specific effort rather than guessing from
  transport kind or model spelling.
- **Dynamic catalogs:** product releases, account policy, local Ollama tags, and
  adapter defaults change. Every static record needs source, applicability,
  version, and review date; live data may only narrow it.
- **Harness defaults:** the example ACP workflow now omits static model IDs and
  uses each adapter's negotiated default until config-option discovery is wired.
  This applies to Claude Code, Codex, OpenCode, Hermes, and OpenClaw; their
  Composer Model/Effort rows remain hidden meanwhile.
- **OpenClaw semantics:** launcher readiness is not operational readiness. Do
  not label it operational until Gateway connectivity and provider/session
  mapping pass; model and effort remain hidden meanwhile.
- **Trademark review:** Apache/MIT redistribution does not grant product-mark
  rights. Distribution review remains required even when the SVG is licensed.
- **Streaming:** Stop is real, but the desktop invoke remains completion-based;
  renderer-visible token streaming is a separate feature.
- **ACP continuation:** loaded ACP histories are read-only. The desktop has not
  wired a same-connection load/resume-and-prompt sequence, so it deliberately
  does not send follow-ups or persist ACP IDs as local continuation sessions.
- **Host credential injection:** the reusable main entry currently supports
  environment-backed secret references, not an injected keychain or prompt
  resolver. Downstream hosts must retain a narrow credential adapter until a
  typed resolver option is published.

Unsupported or unconfirmed items must fail closed: hide or disable the control,
omit the wire field, expose a diagnostic, and retain a deterministic fallback.

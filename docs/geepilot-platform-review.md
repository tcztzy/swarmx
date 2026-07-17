# GEEPilot Platform Boundary Review

This review maps `~/GitHub/geepilot/specs` into SwarmX platform work versus
GEEPilot-owned domain work. The goal is for GEEPilot to depend on SwarmX for the
agent runtime, extension inventory, harness/provider/agent composition, desktop
inventory UI, and protocol surfaces, while keeping GEEPilot-specific science and
analysis workflows in GEEPilot.

## Platform-Owned In SwarmX

| GEEPilot spec | Platform capability SwarmX should own |
| --- | --- |
| `0200-local-desktop-app` | Local-first Electron shell, isolated preload IPC, local session and runtime surfaces. |
| `0201-app-message-rendering` | Typed message rendering, normalized tool/trace events, safe Markdown pipeline. |
| `0202-context-compression-and-packets` | Generic context packet metadata and invocation trace references. |
| `0300-optional-server` | Optional server boundary, explicit connection model, loopback/token defaults. |
| `0302-product-telemetry-and-analysis` | Opt-in telemetry envelope, nonblocking outbox, and generic ingest validation/decision contract, without product-specific measurements, storage, or analysis. |
| `0401-harness-management` | Harness identity, discovery, selector failure behavior, invocation metadata. |
| `0402-provider-profiles-and-secrets` | Provider profile metadata and secret-reference separation; no inline keys in persisted metadata. |
| `0403-agent-profiles-and-definitions` | Agent profile schema, selector resolution, read-only plugin-provided profiles. |
| `0404-acp-agent-protocol-and-composition` | Agent composition resolver: exactly one Harness x Model identity; SwarmX resolves any ModelSupply route internally before run. |
| `0500-plugin-marketplace-and-harness-bundles` | Plugin bundle inventory model covering software, skills, MCP servers, agents, harnesses, and app connectors. |
| `0600-environment-bootstrap-and-managed-dependencies` | Generic dependency class metadata and explicit-user-action install boundary. |
| `0700-autonomous-iteration-and-traceability` | Durable traceability contracts and verification evidence primitives. |
| `0702-autonomous-scheduler-and-reporting` | Generic work-item, lease, report, evidence, dashboard path, and scheduler-ledger primitives. |
| `0703-deterministic-autonomous-runtime` | Deterministic event/state-machine/DAG/validator/replay primitives for autonomous runs. |

## GEEPilot-Owned

| GEEPilot spec | Reason to keep in GEEPilot |
| --- | --- |
| `0301-ibm-lsf-hpc-workflows` | IBM LSF, cluster SSH, array-job completeness, and remote-compute policy are deployment/domain adapters. |
| `0400-host-compatibility-and-skills` | Shared host and skill packaging metadata can be represented as SwarmX plugin metadata, but GEEPilot owns its skill tree and biosecurity routing. |
| `0701-runtime-memory-and-knowledge` | SwarmX can provide extension and trace surfaces, but GEEPilot owns its memory skill, knowledge schema, and claim semantics. |
| `0800-data-analysis-workflow-autonomy` | SwarmX owns reusable evidence packets and DAG/validator contracts; GEEPilot owns analysis questions, interpretation, benchmark/paper boundaries, and biosecurity gates. |
| `0001-spec-process`, `0100-engineering-lifecycle-governance` | GEEPilot process remains project-local; SwarmX may adopt similar mechanics separately. |

## Implemented SwarmX Foundation

SwarmX now exposes a generic extension inventory in `@swarmx/core`:

- `ExtensionBundleSchema` models plugin bundles with software, standalone
  Models, ModelSupplies, Provider connections, skills, MCP servers, Harnesses,
  Agent profiles, app connectors,
  GUI contributions, commands, LSP servers, hooks, monitors, output styles,
  settings, assets, permission declarations, auth policies, marketplace
  sources, and plugin catalog entries.
- `UiContributionSchema` models passive GUI extension points such as navigation
  items, views, panels, settings panels, dashboard widgets, composer actions,
  message actions, inspector sections, toolbar actions, menu items, and status
  items through refs rather than inline components or scripts.
- `MarketplaceSourceSchema` and `PluginCatalogEntrySchema` model the read-only
  management metadata from spec 0500: host compatibility, source path or URL,
  trust state, install/update state, component counts, and whether a plugin
  supplies a runnable harness.
- `SkillCapabilitySchema`, `SkillHostExposureSchema`, and
  `validateSkillHostCompatibility()` model the generic part of spec 0400:
  canonical skill paths, governance refs, gate-skill ids, plugin versus
  rules-only host exposure, marketplace source references, and passive issue
  reporting for caller-configured canonical roots and host local-path rules.
  SwarmX records and validates metadata; GEEPilot still owns the checked-in
  `skills/` tree, biosecurity ordering, and host-specific package files.
- Extended component schemas model the rest of the host plugin bundle inventory:
  command entries, LSP declarations, hooks, monitors, output styles, plugin
  settings, assets, permission declarations, and authentication policies with
  secret references only.
- `loadExtensionInventory()` reads built-in SwarmX harnesses plus path manifests
  from `SWARMX_EXTENSION_PATHS` or `SWARMX_EXTENSION_ROOTS`.
- `parseExtensionBundle()` rejects inline secret-looking fields such as API keys,
  tokens, passwords, private keys, and credentials. Secret references remain
  metadata only.
- `resolveAgentComposition()` turns an Agent composition into an `AgentConfig`
  only after it resolves an explicit Harness x Model identity. Optional
  ModelSupply metadata comes from a trusted profile or fixed core route, not an
  ordinary composer choice.
  The resulting agent records extension provenance but does not copy secret refs.
- `resolveAgentCompositionPlan()` produces a side-effect-free ACP execution
  preflight plan with status, canonical selector, Harness x Model, optional supply,
  definition source, enabled plugin ids, selected skill and MCP provenance,
  context, permissions, visual metadata, auth requirement metadata, and missing
  or blocked requirements. The execution resolver uses the same readiness model
  so unknown skills, missing MCP servers, disabled profiles, unsupported context
  strategies, missing harnesses, and missing models fail before invocation
  instead of falling back to defaults.
- `resolveAgentCompositionRuntimeEnv()` follows only internally bound
  ModelSupply metadata to its Provider connection and fails on missing env
  secrets or unsupported secret sources before execution. Harness compatibility
  is already decided from Harness and Model API metadata, never from Provider.

Minimal manifest shape:

```json
{
  "schemaVersion": 1,
  "id": "geepilot",
  "name": "GEEPilot",
  "version": "0.1.0",
  "capabilities": {
    "models": [
      {
        "id": "gpt-5",
        "label": "GPT-5",
        "runtimeModel": "gpt-5",
        "apiProtocols": ["openai_responses"]
      }
    ],
    "skills": [{ "id": "geepilot.memory", "path": "skills/memory/SKILL.md" }],
    "mcpServers": [
      {
        "id": "project-fs",
        "server": { "type": "stdio", "command": "npx", "args": ["-y", "server"] }
      }
    ],
    "harnesses": [
      {
        "id": "geepilot-codex",
        "label": "GEEPilot Codex",
        "modelControl": "session",
        "modelCompatibility": "any",
        "supportedModelApis": [],
        "backend": {
          "type": "custom",
          "program": "npx",
          "args": ["--yes", "@agentclientprotocol/codex-acp"]
        }
      }
    ],
    "agents": [
      {
        "id": "analysis-lead",
        "name": "analysis lead",
        "harnessId": "geepilot-codex",
        "modelId": "gpt-5",
        "skills": ["geepilot.memory"],
        "mcpServers": ["project-fs"]
      }
    ],
    "uiContributions": [
      {
        "id": "geepilot.nav",
        "kind": "navigation_item",
        "name": "GEEPilot",
        "placement": "sidebar",
        "route": "/extensions/geepilot",
        "componentRef": "geepilot.ui.shell",
        "readOnly": true
      }
    ]
  }
}
```

The desktop app now exposes a read-only Extensions inventory page through
preload IPC. It separates marketplace sources, plugin catalog entries, plugin
bundles, executable harnesses, agent profiles, provider profiles, skills, MCP
servers, app connectors, GUI contributions, and generic plugin components, so
future GEEPilot management UI can build on SwarmX without hard-coding GEEPilot
concepts. Skill rows show canonical path, governance, gate-skill, host exposure,
manifest/rules/source, and read-only metadata as passive chips. GUI
contribution rows are passive refs for contributed navigation, views, panels,
settings, dashboards, composer actions, message actions, inspector sections,
toolbar/menu items, and status items; the renderer does not navigate, load
component code, mount iframes or webviews, evaluate scripts,
execute commands, open assets, write settings, grant permissions, authenticate,
or change plugin trust from those declarations. Agent profiles listed in that
page can be selected for the composer; sends carry an `agentComposition` payload
and are resolved in the main process from the same extension inventory before
execution. Resolved `custom` harness backends execute through the ACP client
rather than falling back to the native OpenAI call path. Provider secrets are
injected into that child process only at invocation time.
The same preload response includes composition plans for extension agent
profiles, so the renderer can show ready, draft, blocked, and disabled states,
canonical selectors, skill/MCP provenance, context/permission summaries, auth
requirements, and blocked requirements without importing Node-only core modules.
Blocked or disabled plans remain visible for inspection but are not invokable
from the read-only Extensions inventory view.

The marketplace/catalog layer remains passive metadata. SwarmX does not install
or update host plugins, enable hooks, start MCP servers, or change trust state.
The extended component inventory is passive as well: SwarmX does not execute
commands, start LSP servers, activate hooks or monitors, apply output styles,
write settings, open assets, grant permissions, or authenticate external
services from a manifest. The GUI contribution layer is passive too: SwarmX
does not navigate to contributed routes, load components, mount webviews,
evaluate scripts, or treat a UI declaration as an executable harness.

SwarmX now exposes explicit user-action primitives through the browser-safe
`@swarmx/core/actions` subpath:

- `ActionIntentSchema` models action kind, host, source and target references,
  risks, confirmation requirement, confirmation text, and sanitized payload.
- `createActionIntent()` assigns deterministic `act_` ids and infers default
  risk and confirmation policy for installs, updates, enable/disable actions,
  MCP startup, command reruns, raw-payload reveal, and trust changes.
- `ActionConfirmationSchema` plus `assertActionConfirmed()` gives execution
  layers a reusable guard before mutating state or running code.
- `actionIntentFromDependencyPlan()` maps dependency plans from spec 0600 into
  confirmation-gated managed install or external installer actions, while
  existing and unavailable dependencies remain read-only.

This covers the reusable explicit-user-action boundary referenced by GEEPilot
specs 0201, 0401, 0500, and 0600. GEEPilot still owns the actual installer
commands, plugin enablement, hook activation, MCP process startup, trust-store
writes, rerun execution, and product-specific confirmation UI.

SwarmX also exposes local desktop settings primitives through the browser-safe
`@swarmx/core/desktop-settings` subpath:

- `DesktopSettingsDocumentSchema` models a generic settings document with
  desktop root metadata, server metadata, UI state, standalone Models,
  ModelSupplies, Provider connection metadata, Agent profile metadata, and
  extension state.
- `resolveDesktopRoot()` implements the local-first root precedence from spec
  0200: explicit desktop-root environment variables win, settings desktop roots
  come next, legacy app-root values are compatibility fallbacks, and server data
  roots are tracked separately rather than becoming desktop roots.
- `LocaleRegistrySchema` and `resolveLocaleSelection()` provide a centralized
  locale registry and selected-locale resolution so desktop UIs do not scatter
  translation tables through components.
- The settings schema reuses SwarmX Model, ModelSupply, Provider, and Agent
  profile metadata so persisted identities, routes, connections, Harnesses,
  plugins, and extensions remain separate records.

This covers the reusable settings, root, locale, and profile-metadata substrate
from GEEPilot specs 0200, 0402, and 0403. GEEPilot still owns the actual
`~/.geepilot/settings.json` and `~/.geepilot/auth.json` file IO, `0600` file
mode checks, keychain or auth-file writes, provider secret migration, concrete
settings panels, and product-specific defaults.

SwarmX also exposes secret-store primitives through the browser-safe
`@swarmx/core/secrets` subpath:

- `SecretRefSchema` models env, local auth file, local keychain, server
  keychain, prompt, and request-only secret references with purpose metadata.
- `SecretVaultDocumentSchema` and `SecretWriteRequestSchema` are the only
  generic records allowed to contain secret values, and only under explicit
  `value` fields.
- `SecretStatusSchema` reports configured/missing state without returning
  secret values.
- `evaluateSecretFileMode()` checks whether a reported mode matches restrictive
  expectations such as `0600` without touching the filesystem.
- `secretStatusFromVault()`, `secretValueFromVault()`, and redaction helpers
  provide pure lookup and safe-display behavior.
- `assertSecretRefPolicy()` rejects persisted storage for request-only secrets
  such as remote-compute passwords.

This covers the reusable secret-reference, status, redaction, and mode-policy
contract from GEEPilot spec 0402. GEEPilot still owns concrete local
`~/.geepilot/auth.json` writes, file creation with mode `0600`, server Keychain
routes, migration of misplaced keys, and any provider prompt execution.

SwarmX also exposes provider profile primitives through the browser-safe
`@swarmx/core/providers` subpath:

- `ProviderProfileMetadataSchema` models provider ids, preset ids, display
  names, API supply label, base URL, read-only state, metadata, and secret
  references. Provider-owned Model catalogs, defaults, and Harness overrides
  are rejected.
- `ProviderSecretRefSchema` and `ProviderSecretStatusSchema` keep secret
  location/status separate from secret values. Status records can report that a
  key exists, but they reject returned key values.
- `resolveProviderProfile()` resolves an explicit connection and fails on
  missing, unknown, or ambiguous profile ids; it never chooses a default Model
  or Provider.
- `ProviderPromptRequestSchema` models direct provider prompt metadata without
  pretending direct prompts are ACP harness calls.
- `buildProviderRuntimeEnv()` constructs request-scoped env vars only from the
  selected Model/ModelSupply route plus an explicit caller-supplied runtime
  secret value and never mutates persisted profile metadata.

This covers the reusable contract from GEEPilot spec 0402. GEEPilot still owns
the concrete local `~/.geepilot/auth.json` file mode, server Keychain routes,
provider settings persistence, direct prompt execution, and any migration of
misplaced keys out of settings.

SwarmX also exposes harness management primitives through the browser-safe
`@swarmx/core/harness-management` subpath:

- `HarnessDiscoveryRecordSchema` models runtime adapter discovery state:
  adapter id, display name, host scope, availability, command path, version,
  installability, dependency id, and status note.
- `resolveHarnessSelector()` parses `@harness`, `@harness:model`, and named
  Agent aliases, strips the selector from the delegated prompt, and fails on
  unknown or ambiguous selectors instead of falling back silently.
- `HarnessAgentAliasSchema` maps a named Agent selector to one explicit Harness,
  Model, optional ModelSupply, and optional Agent profile.
- `HarnessInvocationMetadataSchema` models invocation trace metadata with
  context packet id, trigger message id, Harness x Model identity, optional
  ModelSupply and Agent profile, canonical selector, external session reference,
  status, timestamps, and errors.

This covers the reusable contract from GEEPilot spec 0401. GEEPilot still owns
the actual local/server harness discovery implementations, adapter install
actions, host CLI probing, vendor installers, and product-specific policy about
which adapters are exposed.

SwarmX also exposes agent profile definition primitives through the browser-safe
`@swarmx/core/agent-profiles` subpath:

- `AgentDefinitionFrontmatterSchema` accepts Claude-compatible agent fields such
  as tools, disallowed tools, model, permission mode, MCP servers, hooks, turn
  budget, skills, initial prompt, memory, effort, background execution,
  isolation, and color.
- `AgentDefinitionGeepilotMetadataSchema` keeps GEEPilot-only binding metadata
  under `geepilot`, including Harness id, optional supply id, selector alias,
  enablement, and source label.
- `parseAgentDefinitionMarkdown()` parses YAML frontmatter and preserves the
  Markdown body as the behavior prompt while keeping unknown frontmatter fields
  as inert metadata.
- `createAgentProfileFromDefinition()` turns a definition into profile metadata
  that links to Harness, Model, optional supply, plugin/source provenance, tools,
  skills, memory, and permission-like fields without copying secrets or
  collapsing profiles into harness records.
- `projectAgentDefinitionForClaudeCode()` produces a host-compatible Claude Code
  projection that omits GEEPilot-only metadata.

This covers the reusable parser and metadata contract from GEEPilot spec 0403.
SwarmX Desktop now owns passive, read-only discovery for project/user
`.claude/agents/*.md` and `.codex/agents/*.toml`, including deterministic
precedence and warning isolation. GEEPilot still owns server profile APIs,
enable/disable flows, imported-agent forking, test invocation, product-specific
UI navigation, and host-specific export writes.

Extension-provided agent profiles now also carry passive policy metadata in the
same inventory layer: tools, disallowed tools, permission mode, turn budget,
memory, effort, background execution preference, isolation, color, and
definition source. `resolveAgentComposition()` records that metadata under
`parameters.extension.profile`, and the desktop Extensions view renders it as
inspection chips. This improves the GUI inspection surface required by specs
0403 and 0404 without making SwarmX responsible for enforcing host-specific
tool policy or activating hooks and MCP servers.

SwarmX also exposes typed context packet primitives in `@swarmx/core`:

- `ContextObjectSchema` models the reusable object kinds from spec 0202:
  instructions, summary checkpoints, message tails, agent invocation summaries,
  and delegated requests.
- `ContextPacketSchema` records requested/resolved strategy, packet mode,
  included/dropped/truncated object ids, included message ids, prompt bytes, and
  prompt SHA-256.
- `SummaryCheckpointSchema` models append-only checkpoint metadata with optional
  Model/ModelSupply runtime metadata and compression prompt hash.
- `AgentInvocationContextMetadataSchema` lets child Harness x Model invocations
  cite the context packet and optional ModelSupply they used while keeping
  external Codex or Claude sessions as trace references only.
- `buildContextPacket()` fits typed objects to a prompt budget while preserving
  the delegated request and making dropped or truncated context explicit.

SwarmX also exposes append-only conversation ledger primitives through the
browser-safe `@swarmx/core/conversation` subpath:

- `ConversationIndexRecordSchema` models local session index metadata with
  agent, harness, model, timestamps, message count, archived state, context
  strategy, and portable storage references.
- `ConversationEventSchema` models `cev_` rollout events for session creation,
  message append, checkpoint append, invocation lifecycle, artifact linking,
  title updates, and archive state.
- `conversationJsonlLine()` and `parseConversationJsonl()` provide stable JSONL
  serialization with line-numbered parse errors for append-only desktop storage.
- `replayConversationEvents()` reconstructs current session/message state
  deterministically and records invalid append order as rejected events.

These primitives give GEEPilot a reusable local conversation substrate without
making SwarmX responsible for provider-backed compression, concrete desktop root
selection, runtime memory admission, local Wiki drafts, or scientific knowledge
semantics.

SwarmX also exposes normalized rendering primitives through the browser-safe
`@swarmx/core/rendering` subpath and wires the desktop run timeline through
them:

- `NormalizedRenderEventSchema` gives messages, tool calls, tool results,
  artifacts, traces, and agent metadata one host-neutral rendering contract.
- `RenderArtifactReferenceSchema` and `RenderProvenanceSchema` carry artifact,
  plugin, MCP, Harness x Model, optional ModelSupply, Agent, and external-session
  metadata without making the renderer branch by host.
- `normalizeMessageChunk()` maps current SwarmX message chunks into normalized
  render events with stable `rne_` ids, status, title, summary, sanitized
  input/output, artifact references, and raw payload references.
- The desktop `RunEvent` renderer displays tool-call and tool-result payloads
  from the normalized event, so secret-looking tool arguments or outputs are
  redacted before they reach the visible GUI. It also renders expandable
  read-only trace cards with sanitized detail blocks, provenance chips,
  artifact-reference chips, and raw-payload refs without opening artifacts,
  revealing raw host payloads, rerunning tools, or mutating session state.
  Terminal, file, diff/patch, test/check, MCP, browser/app automation, and
  generated-media blocks are display adapters over normalized fields only; they
  do not read file paths, fetch media, call MCP servers, execute commands, or
  infer GEEPilot-specific HPC or data-analysis meaning.
- The desktop `MessageContent` renderer uses `react-markdown` plus GFM and
  KaTeX for conversational messages, escapes raw HTML, blocks remote Markdown
  images by default, resolves trusted local images through preload IPC, renders
  fenced code in labeled blocks with exact-code copy controls and offline Shiki
  highlighting fallback, and renders inline/display math offline while keeping
  invalid formulas local to the affected span or block.

SwarmX also exposes telemetry primitives through the browser-safe
`@swarmx/core/telemetry` subpath:

- `TelemetryEnvelopeSchema` models v1 opt-in product telemetry envelopes with
  event id, timestamp, event type, source, installation id, release metadata,
  and sanitized payload.
- `resolveTelemetryConfig()` keeps telemetry disabled unless explicitly enabled
  and an endpoint is configured, while supporting both `SWARMX_TELEMETRY_*` and
  downstream `GEEPILOT_TELEMETRY_*` env naming.
- `sanitizeTelemetryPayload()` redacts secret-looking fields and omits default
  raw content fields such as prompts, responses, terminal output, stack traces,
  source code, Wiki bodies, and run logs.
- `createTelemetryClient()` uses injected send/outbox adapters so failed sends
  can be appended to downstream local outboxes without blocking conversations,
  agent runs, or GUI navigation.
- `TelemetryIngestConfigSchema`, `TelemetryIngestAcceptedRecordSchema`, and
  `TelemetryIngestDecisionSchema` define the reusable ingest-side contract:
  disabled-by-default opt-in, optional bearer-token gate, schema-version
  allowlist, accepted records, and accepted/rejected decisions.
- `evaluateTelemetryIngest()` and `createTelemetryIngestHandler()` validate
  caller-supplied envelopes, reject unsupported schema versions or unsafe raw
  payloads before append, preserve event ids and timestamps, and call only a
  caller-injected append adapter when a record is accepted.
- The v1 event-type validator rejects `task_created` and `task_updated` because
  a Task product surface is not part of the reusable platform contract.

This gives GEEPilot a stable telemetry intake dependency without moving product
semantics into SwarmX. GEEPilot still owns the actual FastAPI route, JSONL or
database persistence, deduplication policy, aggregate analysis, dashboards, and
any HPC/job workflow telemetry.

SwarmX also exposes managed dependency primitives through the browser-safe
`@swarmx/core/dependencies` subpath:

- `ManagedDependencyManifestSchema` models dependency classes, owners, version
  sources, install roots, platform entries, hashes, signature metadata, and
  trust models without tying the contract to GEEPilot's concrete data roots.
- `parseManagedDependencyManifest()` accepts GEEPilot-style snake_case manifest
  fields, normalizes them to camelCase, rejects duplicate ids, and applies the
  managed-download policy before runtime code can plan an install.
- `validateManagedDependencyPolicy()` rejects inline secrets, unpinned
  managed-download versions, missing SHA-256 values, non-HTTPS or credentialed
  URLs, unsafe archive members, unsafe target names, and install roots that are
  not expressed through documented env vars or settings keys.
- `DependencyDetectionResultSchema` and `DependencyInstallReceiptSchema` give
  downstream code portable read-only detection and audit-record shapes.
- `planDependencyAction()` is side-effect free: it can select
  `use_existing`, `install_managed`, `requires_user_action`, or `unavailable`,
  but it never probes PATH, downloads artifacts, opens installers, runs package
  managers, or mutates receipts.

This covers the shared base from GEEPilot spec 0600. GEEPilot still owns its
actual `uv` and npm bootstrap, `ripgrep` download implementation, archive
extraction, atomic replacement, Tuqiao installer flow, harness vendor installer
commands, and data-root-specific receipt writes.

SwarmX also hardens the optional HTTP server boundary:

- `createServer()` binds to loopback by default and refuses non-loopback hosts
  such as `0.0.0.0` unless an `apiToken` is supplied.
- Browser origins are exact-match allowlisted through `allowedOrigins`; wildcard
  CORS is rejected.
- `Origin: null` is rejected unless `allowNullOrigin` is explicitly configured
  for a trusted desktop bridge.
- Configuring `apiToken` requires `Authorization: Bearer <token>` on requests.
- The `swarmx serve` CLI exposes `--api-token`, `--allowed-origin`, and
  `--allow-null-origin` for explicit deployments.

SwarmX also exposes deterministic autonomy primitives in `@swarmx/core`:

- `AutonomyWorkItemSchema` models `project_iteration` and `analysis_execution`
  ledger items with stable `awi_` ids, lifecycle status, autonomy level,
  source refs, evidence requirements, budgets, blockers, retry state, leases,
  workflow stage, and downstream metadata.
- `AutonomyRuntimeEventSchema` models append-only `evt_` events with source,
  idempotency key, run/work-item refs, previous/next state, command/validator
  refs, and redacted payloads.
- `AutonomyTriggerRecordSchema` and `runtimeEventFromTrigger()` model typed
  schedule, manual, issue, validation-failure, feedback, analysis-finding,
  file-change, and dependency-update triggers before any work item executes.
- `EngineeringLifecycleStateSchema`, `EngineeringIntakeRecordSchema`,
  `EngineeringProposalRecordSchema`, and `EngineeringApprovalRecordSchema`
  model the generic intake, triage, proposal, discussion, specification,
  implementation, validation, report, close, and side-state records from spec
  0100 without creating tickets, accepting specs, or editing specs.
- `EngineeringLifecycleTransitionDecisionSchema`,
  `evaluateEngineeringLifecycleTransition()`, and
  `engineeringLifecycleWorkflowState()` gate engineering lifecycle moves on
  typed state edges, current workflow stage, evidence ids, approval ids, and
  optional validator-gate decisions while leaving the actual mutation to the
  caller.
- `AutonomyAgentRunRecordSchema`,
  `AutonomyWorkflowDecisionRecordSchema`,
  `createAutonomyAgentRunRuntimeEvent()`,
  `createAutonomyWorkflowDecisionRuntimeEvent()`, and
  `linkAgentStageToWorkflowState()` give GEEPilot reusable `agt_` agent-stage
  records, `dec_` workflow-decision records, append-only runtime event
  projection, and workflow-state linkage. Agent-run identity is Harness x Model
  with optional ModelSupply, without making SwarmX the worker,
  daemon, issue-intake, or analysis-decision owner.
- `AutonomyTransitionDecisionSchema`, `evaluateAutonomyTransition()`, and
  `createAutonomyTransitionRuntimeEvent()` keep lifecycle mutation behind
  deterministic guard decisions that check current state, allowed status edge,
  idempotency, and caller-supplied approval/validator/budget/autonomy
  preconditions.
- `CommandDagSchema` validates command DAG node shape, declared dependencies,
  retry policy, validators, artifact policy, duplicate node ids, missing
  dependencies, and cycles.
- `ValidatorManifestSchema` and `EvidencePacketSchema` provide reusable
  validation and `evp_` evidence packet shapes that GEEPilot analysis runs can
  reference without making SwarmX responsible for scientific conclusions.
- `ValidatorGateDecisionSchema` and `evaluateValidatorGate()` summarize
  required, missing, passed, failed, waived, and skipped validator ids from
  supplied outcomes without running validators or copying raw validator output.
- `AutonomyScheduleStateSchema`, `AutonomyScheduleTriggerSchema`,
  `AutonomyReportMetadataSchema`, `AutonomyDashboardMetadataSchema`,
  `AutonomyFeedbackRecordSchema`, `AutonomyWakeupStateSchema`,
  `AutonomyDaemonRunMetadataSchema`, and
  `AutonomyCircuitBreakerDecisionSchema` cover the generic scheduler and
  reporting state from spec 0702 without implementing GEEPilot's daemon,
  GitHub webhook intake, OS wakeup installation, or worker execution loop.
- `evaluateAutonomySchedule()`, `createAutonomyScheduleTrigger()`,
  `defaultReportSchedule()`, `evaluateCircuitBreaker()`, and
  `wakeupStatePath()` provide deterministic helpers for report cadence,
  schedule triggers, retry breakers, and app/server wakeup state paths from
  explicit inputs.
- `replayAutonomyEvents()` reconstructs work-item state deterministically,
  applies idempotency keys once, and records rejected transition events instead
  of trusting model self-assessment.
- `AutonomyReplayRecordSchema` and `createAutonomyReplayRecord()` summarize
  replay output with event counts, applied/rejected ids, work-item status
  counts, deterministic `sha256:` state hash, and explicit missing external
  dependencies.
- `AUTONOMY_PATHS` and `autonomyDatedPath()` define portable local ledger paths
  such as `autonomy/runtime/events.jsonl`,
  `autonomy/runs/YYYY/MM/DD/<runId>.jsonl`, and
  `autonomy/evidence/YYYY/MM/DD/<evidenceId>.json`.

These primitives intentionally stop before GEEPilot-specific provider-backed
compression, checkpoint append policy, runtime memory admission, telemetry
ingest storage, aggregate telemetry analysis, server user accounts, email
activation, Wiki storage, deployment database models, daemon ownership, GitHub
webhook verification, wakeup installation, launchd/systemd file generation,
tick locking, report file writes, dashboard serving, issue intake policy, HPC
adapters or cluster queue contracts, benchmark execution, paper updates,
biosecurity routing, skill operation policy, or scientific interpretation.
Trigger records, transition decisions, validator gate decisions, replay
records, lifecycle intake records, proposal records, agent-stage records,
workflow-decision records, and skill host-compatibility records also reject
secret-looking fields plus raw issue bodies, raw prompts, raw responses,
terminal transcripts, validator logs, and data-analysis outputs where those raw
payloads belong in caller-owned artifacts with explicit references.

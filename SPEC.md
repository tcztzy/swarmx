# SPEC

## §G
G1: n8n-like workflow. JSON defines multi-agent process. Desktop shows same workflow and can run it.
G2: n8n import. User can load exported n8n workflow JSON and convert it into the same SwarmX workflow JSON.
G3: Downstream agent products can depend on SwarmX for generic platform surfaces: extension bundle inventory, harness/provider metadata, agent profiles, and ACP agent composition.
G4: Desktop GUI exposes extension inventory as a reusable, read-only platform surface without hard-coding downstream product logic.
G5: Desktop GUI can run extension-provided agent profiles through SwarmX composition resolution without registering downstream harness ids in the built-in harness list.
G6: Downstream agent products can depend on SwarmX for generic deterministic autonomy primitives: work items, runtime events, command DAGs, validator manifests, evidence packets, path contracts, and replay.
G7: Downstream agent products can depend on SwarmX for generic context packet and trace metadata primitives for parented delegation and resumption.
G8: Downstream agent products can depend on SwarmX for host-neutral normalized rendering events that make GUI message, tool, artifact, and trace presentation extensible.
G9: Downstream agent products can depend on SwarmX for opt-in telemetry primitives: v1 envelopes, redaction, event-name policy, status, headers, and injected nonblocking sender/outbox behavior.
G10: Downstream agent products can depend on SwarmX for an optional server boundary with loopback defaults, explicit non-loopback bearer-token requirements, and exact Origin policy.
G11: Downstream agent products can depend on SwarmX for generic managed dependency contracts: dependency classes, manifest validation, detection results, install receipts, and side-effect-free install planning.
G12: Downstream agent products can depend on SwarmX extension inventory for marketplace source and plugin catalog metadata that supports extensible GUI management without product-specific registry code.
G13: Downstream agent products can depend on SwarmX for generic append-only conversation ledger contracts: local session index records, rollout events, JSONL serialization, and deterministic replay.
G14: Downstream agent products can depend on SwarmX for generic provider-supply contracts: connection metadata, API compatibility metadata, secret references, secret statuses, direct prompt requests, explicit supply resolution, and request-scoped runtime env construction without treating providers as model owners.
G15: Downstream agent products can depend on SwarmX for generic harness management contracts: discovery records, selector resolution, named agent aliases, and invocation metadata.
G16: Downstream agent products can depend on SwarmX desktop rendering for safe conversational Markdown, blocked-by-default remote media, preload-mediated local images, and stable labeled code blocks.
G17: Downstream agent products can depend on SwarmX for generic explicit-user-action contracts: action intents, risk classification, confirmations, dependency-plan action mapping, and secret-safe records.
G18: Downstream agent products can depend on SwarmX for generic agent profile definition contracts: Claude Code Markdown and Codex TOML parsing, GEEPilot-scoped metadata preservation, profile metadata conversion, and native host projection.
G19: Downstream agent products can depend on SwarmX extension inventory and desktop GUI to passively surface agent profile definition and policy metadata for inspection before execution.
G20: Downstream agent products can depend on SwarmX for generic local desktop settings primitives: settings document shape, desktop root resolution, locale registry and selection, UI state metadata, and secret-safe provider/agent profile persistence.
G21: Downstream agent products can depend on SwarmX extension inventory for passive plugin component metadata beyond skills and MCPs: commands, LSP servers, hooks, monitors, output styles, settings, assets, permissions, and auth policy.
G22: Downstream agent products can depend on SwarmX for generic secret-store contracts: secret references, vault documents, redaction, status records, restrictive file-mode evaluation, runtime lookup, and request-only secret policy.
G23: Downstream agent products can depend on SwarmX for generic ACP agent composition preflight: resolved execution plans, readiness requirements, capability provenance, and GUI-visible health state before invocation.
G24: Downstream agent products can depend on SwarmX for generic autonomous scheduler and reporting contracts: schedule state, feedback records, report metadata, artifact dashboard metadata, wakeup state, budget/circuit-breaker policy, and side-effect-free review helpers.
G25: Downstream agent products can depend on SwarmX extension inventory for passive GUI contribution contracts: navigation entries, views, panels, settings panels, dashboard widgets, composer actions, message actions, inspector sections, toolbar actions, menu items, and status items.
G26: Downstream agent products can depend on SwarmX for generic deterministic autonomy runtime contracts: typed triggers, trigger-to-event conversion, transition decisions, validator gate decisions, and replay records.
G27: Downstream agent products can depend on SwarmX for generic engineering lifecycle governance contracts: intake records, proposal records, lifecycle states, and evidence/approval-gated lifecycle transition decisions.
G28: Downstream agent products can depend on SwarmX extension inventory for generic skill host-compatibility metadata: canonical skill paths, host exposure state, rules-only host surfaces, gate-skill references, and passive compatibility issue reporting.
G29: Downstream agent products can depend on SwarmX desktop message rendering for offline mathematical notation in technical conversations without requiring product-specific Markdown renderers.
G30: Downstream agent products can depend on SwarmX desktop message rendering for structured code blocks with copy controls, offline enhancement, and stable layout without product-specific renderers.
G31: Downstream agent products can depend on SwarmX desktop run timelines for expandable normalized tool and trace cards that expose sanitized details, provenance, artifact references, and raw-payload references without host-specific transcript renderers.
G32: Downstream agent products can depend on SwarmX desktop run timelines for read-only specialized trace presentations for terminal, file, diff, test/check, MCP, automation, and generated-media events without product-specific transcript parsing.
G33: Downstream agent products can depend on SwarmX for generic autonomous agent-stage trace contracts: agent-run records, workflow-decision records, runtime event projection, and workflow-state linking without product-specific worker code.
G34: Downstream agent products can depend on SwarmX for generic telemetry ingest contracts: ingest config, bearer-token gate, schema-version allowlist, accepted-record shape, and injected append handling without product-specific analytics.
G35: Downstream desktop products can depend on SwarmX desktop as a GUI host by explicitly registering safe React components for declared `uiContributions`, requiring only product config plus a component registry for customization.
G36: Downstream desktop products can call a stable SwarmX core helper to execute an extension-provided agent composition without copying SwarmX desktop IPC internals.
G37: Downstream desktop products can depend on SwarmX desktop for explicit LSP completion hosting from extension-declared stdio language servers without product-specific process bridges.
G38: Downstream desktop products can depend on SwarmX for generic local workspace file completions as a product-neutral reference source.
G39: Downstream desktop products can depend on SwarmX for generic skill completions from extension inventory as a product-neutral `$` reference source.
G40: Desktop users can run external ACP harnesses through a protected container runtime, preferring Apple Container on supported macOS, with host setup hidden behind explicit one-click confirmation.
G41: CLI and desktop users can diagnose runtime health and explicitly repair fixable harness issues without a permanent Setup destination.
G42: Desktop and downstream products can select and execute agents across the supported `Harness x Model` capability matrix, with Model as an independent primary entity and Provider as an optional many-to-many supply label.
G43: Desktop users can populate the Model catalog from configured Provider APIs, refresh it on demand, and persist manually declared Models without making Provider part of Model or Agent identity.
G44: Desktop users can configure Provider connections with a Base URL and API key or auth token through a dedicated Settings surface, with settings and encrypted auth state updated together.
G45: Desktop users can reach Provider settings and release updates from a persistent Codex-style lower-left account area, with usage integrated into the Provider workspace instead of duplicated in account navigation.
G46: Desktop users can inspect supported Provider balances and rate-limit windows beside each Provider connection, plus supported local tool-account quota such as Codex 5-hour and weekly windows, without exposing credentials to the renderer.
G47: Desktop users can manage every configured and supported local Provider in one quota matrix, including multi-protocol Providers and New API account/token summaries, with independent refresh and no duplicated shared balance.
G48: Desktop users manage Extensions, reusable Custom Agents, Harness software health, and shared Runtime dependencies from dedicated Settings surfaces without permanent diagnostic chrome in the conversation sidebar.
G49: Desktop users can compose a reproducible Harness from Software, Extension-provided Skills, MCP servers, project context, and delivery/permission policy, then pair that Harness with a Model as a Custom Agent with deterministic agent/model-specific Skill variants and explicit context cost.
G50: Desktop users can refresh marketplace sources, install and update plugins, preserve local Skill evolution overlays, evaluate candidates for a target Agent/Model, and roll back active revisions through explicit audited actions.
G51: Desktop users can inspect project-scoped and user-scoped Codex and Claude Code Agent definitions through the same read-only Agent profile inventory without copying, activating, or rewriting native files.
G52: A desktop task bound to a Project can identify that Project and inspect its contained text files through bounded read-only tools before answering repository questions.
G53: Desktop task history exposes comprehensible work timing plus automatic, editable, pinnable, and deletable local task titles.
G54: A direct SwarmX task bound to a Project can inspect, edit, and validate that repository through host-provided coding tools with safety boundaries comparable to Claude Code and Codex.
G55: Expanded Worked reasoning reads as part of the conversation body instead of as a nested card.
G56: A running desktop task exposes its live reasoning, commentary, tool calls, and tool results, then collapses that work and leaves only the final answer visible when execution ends.
G57: Direct SwarmX Models receive coding-tool signatures aligned with their trained Claude Code or Codex tool surface instead of one SwarmX-specific schema.
G58: Direct SwarmX Models receive Claude Code/Codex-compatible tool results and can manage bounded long-running background commands within a task.
G59: Direct desktop Claude sessions can persist project-scoped scheduled prompts across app restarts with Claude Code-compatible task files, ownership, recovery, and deletion semantics.
G60: Publish the completed trained-in tool alignment and durable scheduler work as SwarmX 3.1.1 while retaining explicit TODOs for the three unimplemented Claude tools.
G61: Desktop users can rely on Custom Agent permission policy and ACP permission requests as executable authorization boundaries instead of passive labels or automatic cancellation.
G62: Desktop users can understand and manage effective permission authority across managed, Project, personal, and Agent layers, then review sanitized one-call decisions without confusing approval with sandbox escape.
G63: Desktop users can choose their default direct-tool permission mode from General Settings and override that default for each persisted conversation from the Composer, with the active choice visible before sending.
G64: Desktop permission controls match Codex's information architecture before SwarmX-specific expansion: General exposes Default permissions, Auto-review, and Full access as independent capabilities, each conversation has a compact local selector, and conservative/custom policy details remain in Advanced permissions.
G65: New users can install `swarmx` from npm and launch Desktop by default, while macOS users can download matching install packages directly from GitHub Releases.

## §C
C1: Reuse existing `SwarmConfig`; no second workflow DSL.
C2: JSON examples must match zod schema: nodes use `kind`, node payload under `agent`/`tool`/`swarm`.
C3: Desktop keeps Electron isolation: renderer uses preload API only.
C4: UI must stay useful at desktop and narrow widths; no overlapping text/controls.
C5: Default single-agent path must still work when no workflow JSON is active.
C6: `FORMAT.md` missing in repo; this SPEC uses required §G/§C/§I/§V/§T/§B shape without extra local encoding rules.
C7: In the desktop workflow surface, "agent" means ACP agent identity: model plus harness/backend. Tool use is configured inside an agent, not represented as the default agent node identity.
C8: Harness identity is not a short label. It is a reproducible composition descriptor: Software name/version, selected Skill bindings and resolved variants, selected MCP servers, project context files, delivery capabilities, and permission policy.
C9: n8n import is structural. It preserves workflow topology, node metadata, parameters, and credential references, but does not import credential secrets or execute native n8n node implementations.
C10: n8n import must output `SwarmConfig`; no persisted n8n DSL or alternate runtime graph.
C11: GEEPilot-specific LSF/HPC, biosecurity, memory-claim semantics, benchmark, paper, and data-analysis workflows remain downstream plugin/domain code, not SwarmX core logic.
C12: Extension discovery is passive metadata loading. Installing or updating software, changing trust, enabling hooks, starting MCP servers, mutating Agent bindings, or promoting evolved Skills requires a separate explicit Settings action with confirmation and an audit receipt.
C13: Extension manifests must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, or deployment secrets; only secret references are metadata.
C14: Desktop extension inventory uses preload IPC only and must remain compatible with Electron isolation.
C15: Provider profile secret injection is execution-scoped. Persisted extension metadata and resolved agent metadata keep only profile ids, API compatibility metadata, and secret references, while child-process env receives resolved secret values only at invocation time.
C16: Generic autonomy primitives are side-effect free core contracts. They do not install wakeups, start daemons, execute commands, submit remote jobs, mutate specs, or promote downstream claims.
C17: GEEPilot-specific daemon ownership, issue intake policy, LSF/HPC adapters, biosecurity routing, benchmark execution, paper updates, memory-claim admission, and data-analysis interpretation remain downstream code.
C18: Runtime records, DAGs, validator manifests, and evidence packets must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, or deployment secrets.
C19: Generic context primitives are side-effect free core contracts. They do not own local conversation files, call provider compression, append summary checkpoints, or decide downstream UI rendering policy.
C20: Context records, summary checkpoints, and invocation context metadata must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, or deployment secrets.
C21: Normalized render events are derived presentation state. They do not replace canonical message text, run logs, context packets, or append-only trace records.
C22: GUI rendering must use sanitized normalized payloads for tool calls/results and keep raw host payloads behind references rather than visible by default.
C23: Renderer-facing core APIs must use browser-safe subpath exports and must not pull Node-only ACP, session, filesystem, or server modules into the renderer bundle.
C24: Telemetry primitives are metadata-only by default. They do not upload raw conversations, prompts, responses, source files, terminal output, Wiki bodies, run logs, stack traces, or downstream analysis outputs.
C25: Telemetry is disabled unless explicitly enabled and an endpoint is configured. Failed telemetry sends must be representable as outbox entries without throwing through user workflows.
C26: Generic server boundary primitives do not implement downstream team identity, email activation, Wiki storage, deployment databases, telemetry ingest storage, or product-specific routes.
C27: Browser CORS must not use wildcard origins. Desktop `Origin: null` is trusted only when explicitly configured.
C28: Generic managed dependency primitives are side-effect free core contracts. They do not inspect PATH, download artifacts, extract archives, run package managers, open installers, mutate managed roots, or write receipts.
C29: GEEPilot-specific `uv` and npm bootstrap, managed `ripgrep` installation, archive extraction, atomic replacement, Tuqiao installer flow, harness vendor installer commands, data-root resolution, and receipt writes remain downstream code.
C30: Dependency manifests, detection results, install plans, and receipts must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, remote-compute passwords, or URLs with embedded credentials.
C31: Marketplace source and plugin catalog inventory remains passive metadata. A host-owned Extension manager may install, update, uninstall, enable, disable, or roll back plugins only after an explicit validated action; inventory loading itself never activates hooks, starts MCP servers, executes bundled software, or changes trust.
C32: Desktop extension UI must keep marketplace sources, plugin catalog entries, plugin bundles, executable harnesses, agent profiles, providers, skills, MCP servers, and app connectors visually separate.
C33: Generic conversation ledger primitives are side-effect free core contracts. They do not read or write files, choose concrete desktop roots, compress context, call providers, mutate runtime memory, write Wiki drafts, or decide downstream knowledge admission.
C34: Conversation index records, rollout events, and replay payloads must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, or remote-compute passwords.
C35: Generic provider profile primitives are side-effect free core contracts. They do not read or write local auth files, call provider APIs, persist settings, implement keychain storage, migrate misplaced keys, or execute direct prompts.
C36: Provider profile metadata, provider secret statuses, provider selections, and direct prompt metadata must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, or deployment secrets.
C37: Generic harness management primitives are side-effect free core contracts. They do not probe PATH, install adapters, run vendor installers, start ACP subprocesses, choose product adapter policy, or persist host login state.
C38: Harness discovery records, selector aliases, selector resolutions, and invocation metadata must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, or host login state.
C39: Desktop message rendering must not fetch arbitrary remote Markdown images as a side effect of rendering model output.
C40: Tool-call and tool-result payloads must remain literal text unless they are first normalized into sanitized render events.
C41: Generic action intent primitives are side-effect free core contracts. They do not install, update, uninstall, enable, disable, start, rerun, open, reveal, execute, mutate trust, write settings, or write receipts.
C42: Action intents and confirmations must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, or raw secret-bearing payloads.
C43: Generic agent profile primitives are side-effect free core contracts. They do not import files from host directories, mutate plugin caches, enable imported agents, start hooks, start MCP servers, run harnesses, persist settings, or write host-specific exports.
C44: Agent definition documents, profile metadata, imported frontmatter, and projection records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, or copied host login state.
C45: Extension agent policy metadata is passive inspection data. SwarmX may carry and render tool, permission, memory, and definition metadata, but it does not enforce host-specific tool policy, start hooks, start MCP servers, or enable imported profiles as a side effect of inventory loading.
C46: Generic desktop settings primitives are side-effect free core contracts. They do not read or write settings files, create directories, persist localStorage, manage keychains, migrate secrets, call providers, launch harnesses, or decide downstream product defaults.
C47: Desktop settings documents, UI state records, locale records, provider profile arrays, and agent profile arrays must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, or copied host login state.
C48: Extended plugin component inventory is passive metadata. SwarmX does not execute commands, start LSP servers, activate hooks, start monitors, apply output styles, write settings, open assets, grant permissions, or authenticate external services from manifest loading.
C49: Plugin component inventory records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, or copied host login state.
C50: Generic secret-store primitives are side-effect free core contracts. They do not read or write auth files, create files with mode `0600`, call keychains, migrate misplaced secrets, prompt users, send secrets to servers, call providers, or inject child-process environments.
C51: Secret values are allowed only in explicit secret vault entries and secret write requests. Secret refs, statuses, file-mode statuses, settings, profile metadata, logs, traces, telemetry, and UI state must not contain inline secret values.
C52: Generic agent composition preflight is side-effect free core contract. It does not spawn ACP harnesses, start MCP servers, call providers, read secrets, prompt users, mutate settings, install plugins, activate hooks, or write runtime state.
C53: Composition plans, readiness requirements, selected capability refs, context summaries, permission summaries, visual metadata, and desktop plan rendering must not persist or display inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, or copied host login state.
C54: Generic scheduler/reporting primitives are side-effect free core contracts. They do not install wakeups, start daemons, run ticks, acquire filesystem locks, call GitHub, write reports, serve dashboards, run validators, invoke agents, mutate memory, amend specs, refresh benchmarks, or update papers.
C55: Scheduler state, report metadata, dashboard metadata, wakeup state, feedback records, budget records, and circuit-breaker decisions must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, private deployment URLs, or copied host login state.
C56: GUI contribution inventory is passive metadata. SwarmX does not navigate to contributed routes, load components, evaluate scripts, mount iframes or webviews, execute commands, open assets, write settings, grant permissions, authenticate services, or change plugin trust from manifest loading.
C57: GUI contribution records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, private deployment URLs, copied host login state, inline HTML, inline scripts, inline component bodies, or inline render functions.
C58: Generic deterministic autonomy runtime contracts are side-effect free core contracts. They do not append event logs, acquire locks, run ticks, execute command DAG nodes, run validators, invoke agents, call issue trackers, read clocks, read files, write files, mutate memory, amend specs, refresh benchmarks, or update papers.
C59: Runtime trigger records, transition decisions, validator gate decisions, and replay records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, private deployment URLs, copied host login state, raw issue bodies, raw terminal output, raw validator output, or raw data-analysis outputs.
C60: Generic engineering lifecycle governance contracts are side-effect free core contracts. They do not create issue tickets, write proposal ledgers, accept specs, edit specs, merge patches, run validators, invoke agents, request approvals, write reports, close work items, or decide biosecurity policy.
C61: Lifecycle intake, proposal, discussion, approval, validation, and transition records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, private deployment URLs, copied host login state, or request-only secrets.
C62: Generic Skill host-compatibility and variant metadata is passive extension inventory. SwarmX may resolve versioned Agent/Model-specific variants and record optimizer-neutral evolution lineage, but it does not implement downstream Skill behavior, silently rewrite upstream or active revisions, enforce domain routing, or claim executable adapter support without a declared delivery path.
C63: Skill capability records, host exposure records, and compatibility issue records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, telemetry tokens, SMTP passwords, cluster passwords, remote-compute passwords, private deployment URLs, copied host login state, or request-only secrets.
C64: Desktop math rendering is presentation-only derived state. It does not change canonical message text, execute scripts, fetch remote assets, evaluate formulas semantically, or decide downstream scientific interpretation.
C65: Math rendering must stay inside the message layout on desktop and narrow widths; invalid formulas and parser failures must degrade locally without hiding surrounding message content.
C66: Desktop code-block rendering is presentation-only derived state. It must not execute code, auto-run commands, fetch remote grammars, rewrite copied code text, or parse tool outputs as Markdown.
C67: Code-block enhancements must degrade to escaped plain text when highlighting or clipboard APIs are unavailable, while keeping the message layout stable.
C68: Desktop trace cards are presentation-only derived state. They must not execute tools, open files, fetch artifact URLs, reveal unsafe raw payloads, rerun commands, start MCP servers, or mutate sessions.
C69: Trace-card details must render only sanitized normalized event fields and passive references; raw host payloads and artifacts remain behind refs until a separate explicit action surface authorizes access.
C70: Specialized trace presentations are display adapters over normalized sanitized fields only. They must not read file paths, run terminal commands, fetch screenshots/media, call MCP servers, inspect raw host payloads, or infer downstream scientific meaning.
C71: Generic agent-stage trace contracts are side-effect free core contracts. They do not invoke agents, dispatch workers, submit jobs, verify issues, run validators, advance lifecycle state, write ledgers, or interpret downstream analysis results.
C72: Agent-run records and workflow-decision records must not persist inline provider keys, bearer tokens, passwords, private keys, credentials, raw prompts, raw responses, terminal transcripts, validator logs, data-analysis outputs, cluster passwords, or remote-compute credentials.
C73: Telemetry ingest contracts are side-effect free except through caller-injected append adapters. They do not open sockets, write JSONL files, create databases, run aggregation pipelines, deduplicate stores, expose HTTP routes, or decide product measurement semantics.
C74: Telemetry ingest records and decisions must preserve event ids and timestamps while rejecting unsupported schema versions, missing/invalid ingest bearer tokens when configured, inline secret fields, and default raw-content payloads.
C75: GUI component customization is host-owned code. Extension manifests may name `componentRef` values, but SwarmX must render executable UI only when the embedding desktop explicitly registers a matching React component.
C76: Registered GUI contribution components receive sanitized contribution and inventory metadata plus explicit host callbacks; they must not be loaded from manifest strings, inline HTML, inline scripts, remote URLs, iframes, or webviews.
C77: Agent composition execution helpers may run SwarmX agents, but they must keep extension inventory loading, composition resolution, runtime env injection, and single-agent Swarm construction inside SwarmX rather than downstream product bridges.
C78: Agent composition execution helpers must not persist provider secrets, host login state, raw env snapshots, or downstream product session state.
C79: LSP hosting is an explicit desktop action. Extension inventory loading and rendering remain passive and must not start LSP processes.
C80: LSP completion requests provide workspace root, document text, document URI, language id, and position; SwarmX does not infer domain references or read arbitrary document files for completion.
C81: LSP server declarations may use either `command: string[]` or `command: string` with `args`, but runtime host startup must not persist inline secrets or accept manifest inline env secrets.
C82: LSP host behavior is generic process and JSON-RPC lifecycle code; product-specific completion sources such as data references, identifier registries, bibliography keys, or domain task routing remain downstream language-server logic.
C83: SwarmX local file completions are workspace-bound directory metadata. They do not inspect file contents, dereference remote URLs, resolve biological identifiers, index bibliography keys, or enumerate downstream skill catalogs.
C84: SwarmX skill completions are extension-inventory metadata. They do not execute skills, read skill files, enforce downstream governance gates, or infer product-specific task routing.
C85: Container runtime detection, installer execution, system-service startup, image pulls, and harness process wrapping are desktop host behavior; core contracts remain side-effect free.
C86: On macOS, Apple Container is preferred over Docker and requires Apple silicon plus a supported macOS version; Docker and native fallback must not be assumed or silently preferred.
C87: Host container installation or service mutation requires an explicit user-confirmed setup action, may surface OS administrator prompts, and must not require copying commands into a terminal.
C88: External built-in ACP harness execution should prefer protected container mode when a supported container backend is ready, and should block with setup guidance when protection is required but unavailable.
C89: Containerized harness launches pass only the request-scoped env needed by the harness, avoid raw env snapshots in logs/UI, mount the workspace deliberately, and keep ephemeral containers removable.
C90: Doctor inspection is read-only by default; no installer, service mutation, PATH write, or privileged action runs without an explicit fix request.
C91: Doctor repairs expose a plan first and require explicit confirmation before downloads, installers, service mutation, or administrator prompts.
C92: Hermes diagnosis prefers the existing `~/.hermes/hermes-agent` checkout and must not clone, fetch, pull, or update that repository.
C93: Desktop slash commands use preload IPC, keep the chat context visible, and open diagnostics on demand instead of adding a permanent Setup navigation destination.
C94: Harness registry entries are optional capabilities by default; registration alone never makes a harness required for global environment health or repair.
C95: Model is an independent primary entity. Provider is only a supply label connected to Model through an explicit many-to-many relation.
C96: Provider metadata must not own model catalogs, select default models, carry harness-specific model overrides, participate in harness compatibility, or determine model identity or ownership.
C97: Agent identity is the resolved `harnessId:modelId` pair. Harness/Model compatibility uses that pair; Skill/MCP selection and variant resolution are part of the resolved Harness recipe, while optional ModelSupply remains request-routing metadata and never becomes Agent identity.
C98: External harness model selection is request scoped. SwarmX must not mutate a harness vendor's global configuration file to switch a model.
C99: Provider discovery uses connection/API metadata and secret references only; every response is normalized into independent Model records plus internal ModelSupply links.
C100: Provider network discovery is desktop host behavior, bounded by timeout, and failure-isolated; it must not make extension loading side-effectful or delete the last successful cache.
C101: Manual Model and Provider connection settings never persist inline secret values; user-managed Provider credentials use encrypted local references, while an env reference is resolved only when an explicit extension Provider declares its exact key.
C102: Model catalog management lives inside the Model secondary surface and must not add Provider or Supply to the three primary Agent-picker choices.
C103: Manual Models require explicit stable id, runtime model id, and API protocol; their source/provenance is metadata, not identity or compatibility authority.
C104: User-managed Provider settings contain connection/auth-mode metadata only; they cannot own Models, set default Models, override Harness Models, or participate in compatibility and identity.
C105: User-entered Provider secrets must use Electron safe storage encryption with no plaintext fallback; desktop settings persist only a `local_keychain` secret reference.
C106: Provider auth mode is explicit: API key and auth token select the documented discovery header and request-scoped runtime variable without copying secrets into inventory, plans, logs, errors, or renderer responses.
C107: Provider management lives in the dedicated Settings surface and must not add Provider or Supply to the three primary Agent-picker choices.
C108: Removing a user Provider removes its settings record, encrypted auth entry, and discovery cache without deleting an independent Model supplied elsewhere.
C109: The expanded desktop sidebar ends with a persistent anonymous-user trigger; its popover contains exactly one `Settings` action until real identity/auth features exist.
C110: Provider create/update/remove and credential entry must not appear inside the Agent Picker; the Model secondary surface keeps only Model catalog refresh/manual Model controls and Model selection.
C111: Update UI is absent until a newer stable `@swarmx/desktop` version is verified from the npm registry; when available it is integrated into the anonymous-user row instead of occupying a separate sidebar line.
C112: Usage inside the Provider workspace is an honest view of configured Provider connections and explicitly linked local tool accounts; it must not invent quota, billing, account, or Provider ownership data when a supported source is unavailable.
C113: Automatic npm update is supported only when the running Electron default app can relaunch a versioned npm app directory; signed/packaged hosts and reusable embedded hosts stay hidden instead of mutating their application bundle or dependency tree.
C114: Update metadata and tarballs come only from the canonical npm registry over HTTPS; tarball integrity is verified before installation, package lifecycle scripts are disabled, and the running app directory is never overwritten in place.
C115: Version checking is read-only and silent on network/no-update failures; download/install starts only after an explicit click, publishes bounded progress, remains retryable on failure, and relaunches only after the installed package version is verified.
C116: Provider usage requests execute only in the Electron main process against adapter-owned exact official HTTPS endpoints or a same-origin New API endpoint explicitly selected on a user-managed Provider; credentials are limited to id-bound user-managed keychain entries, resolved in memory, and never selected from ambient env or extension metadata. Redirects are refused, time and response size are bounded, and raw payloads, headers, tokens, and vendor error bodies never cross IPC.
C117: Codex subscription usage is read only through the installed official `codex app-server` `account/rateLimits/read` method and maps declared 300-minute and 10,080-minute windows; SwarmX must not parse Codex auth files or call private web-session endpoints.
C118: Claude Code, Gemini, OpenCode Go/Zen, and any other source without a credential-compatible official usage interface remain explicitly unsupported; SwarmX must not scrape consoles, collect browser cookies, or reuse private OAuth/session endpoints to fabricate support.
C119: Balance and quota data is presentation metadata only. It never changes Provider/Model ownership, Agent identity, ModelSupply routing, compatibility, or execution selection.
C120: Desktop never scans ambient environment variables to create Provider connections or initiate Provider discovery. Env secrets are eligible only when an explicit extension Provider declares the exact reference; user-managed connections require encrypted Settings storage.
C121: Core native Agent codecs are side-effect free. Desktop native Agent discovery is read-only and bounded to the active workspace's `.codex/agents` and `.claude/agents` directories plus the corresponding user directories.
C122: Native `inherit` or omitted Model settings remain unresolved profile metadata until an explicit SwarmX Harness x Model composition supplies a Model; discovery must not bind them to an ambient or guessed default.
C123: Native Agent precedence is deterministic within each host: project definitions override same-name user definitions, while Codex and Claude Code definitions with the same name remain distinct profiles.
C124: Unknown native Agent fields, hooks, inline MCP definitions, and host policy remain inert round-trip metadata. Discovery does not start hooks, MCP servers, skills, subprocesses, or Agent sessions.
C125: Custom Agent Model selection reuses the Composer's Harness-compatible routed Model inventory, Provider grouping, and canonical display order; selecting a Provider route persists its internal ModelSupply id without changing Agent identity.
C126: Persisted Project metadata remains host-owned: Core and Preload may read the local registry before Renderer mount, while the isolated Renderer receives only a validated snapshot and never direct filesystem access.
C127: Project-aware model tools are host-injected, bounded, and rooted at the session `cwd`; traversal, absolute paths, escaping symlinks, binary file mutation, and unbounded repository dumps remain rejected.
C128: A Project-bound prompt identifies the active Project and instructs the Agent to inspect relevant files, but file contents cross the model boundary only through explicit bounded tool calls.
C129: Automatic task naming runs only for a placeholder title, uses the configured cheap title Model route, returns one sanitized short line, and never blocks or replaces a successful primary response when naming fails.
C130: Manual rename and pin state belong to local Session persistence. ACP-discovered sessions stay read-only unless their Harness exposes an explicit mutation capability.
C131: Session deletion remains an explicit destructive action with confirmation; a context-menu dismissal never deletes data.
C132: Work timing is persisted presentation metadata derived from request start/end timestamps; rendering does not fabricate provider execution telemetry.
C133: Existing Project files require a complete prior read and an unchanged content digest before host-provided write or edit operations may replace them; writes are bounded UTF-8 and atomically renamed so stale model context never silently overwrites concurrent user changes.
C134: Host-provided Shell execution starts in the active Project, has a bounded command, duration, and captured output, propagates request cancellation to the whole process group, and never inherits Provider credentials or arbitrary ambient environment variables.
C135: On supported macOS hosts, direct SwarmX Shell execution uses Seatbelt to deny network access and filesystem writes outside the canonical Project and a private request temporary directory. If that sandbox cannot be established, execution fails closed instead of running an unrestricted shell.
C136: Host coding tools are injected only into direct SwarmX compositions. Claude Code and Codex ACP Harnesses retain their native tool and permission systems and are never given duplicate host tools.
C137: Tool descriptions and Project context make read-before-edit, relative paths, exact text replacement, bounded Shell use, and post-change validation explicit; the Renderer never receives direct filesystem or process capability.
C138: A compact Thought/Reasoning event inside Worked uses the same unboxed Markdown content treatment, font size, line height, and foreground hierarchy as assistant body text; tool calls and tool results may retain structured trace containers.
C139: Live agent chunks cross Electron isolation only through a request-id-scoped Preload subscription; the Renderer cannot mix concurrent requests, and persisted history is replaced by the runtime's consolidated terminal result instead of storing raw token deltas.
C140: Direct host tools select one model-family profile per request: Claude-family Models receive Claude Code names/arguments; other Models receive Codex names/arguments. Both profiles retain SwarmX root, stale-content, sandbox, cancellation, and output bounds; harmless unknown arguments are ignored.
C141: Provider-hosted search is exposed only after the selected Provider/API explicitly declares that native capability. SwarmX never weakens the Project Shell network denial or presents an unconfigured local fetch/search implementation as Claude Code or Codex web search.
C142: Tool execution keeps model-facing text separate from client-facing structured content. Structured content crosses existing sanitization and persistence boundaries; it never changes model authority.
C143: Background commands remain Project-sandboxed, output-bounded, runtime-bounded, cancellation-aware, and request-scoped. Manager close stops live process groups and removes private temporary state.
C144: Codex `exec_command` defaults to pipe-backed stdin, while `tty: true` allocates a real PTY whose merged terminal stream remains available through the same `write_stdin` polling lifecycle.
C145: PTY execution retains Project Seatbelt policy, sanitized environment, workdir containment, output/runtime bounds, request scope, process-group termination, and private temporary-state cleanup; PTY selection never escalates authority.
C146: Claude Code parity is tracked against the complete public 42-tool inventory from `code.claude.com/docs/en/tools-reference`; SwarmX exposes a name only when it can provide the documented schema and a real local, provider-hosted, or explicitly configured side effect.
C147: Claude task, Todo, and review state is request-scoped. It is shared by tools in one direct-model tool manager, discarded on manager close, and never presented as durable or cross-task state.
C148: `NotebookEdit` uses the same Project containment, complete-read digest, UTF-8 size, atomic-write, and stale-content checks as `Write`; it never bypasses guarded mutation because the payload is JSON.
C149: Claude MCP resource tools are registered only after at least one configured MCP client connects. Resource calls reuse that client and active request cancellation; binary resource content is never represented by a fabricated local path.
C150: Claude `Skill` is registered only for composition-selected Skill capabilities with an explicit file path. Invocation selects by configured id/name and may read only that fixed file or its `SKILL.md`; model input never supplies a filesystem path.
C151: Interactive Claude tools exist only with a desktop request bridge. Pending prompts bind request, renderer owner, and interaction id; cancellation, window destruction, or tool-manager close rejects and cleans them without default approval.
C152: Claude plan mode stores draft in one private request-scoped plan file. While active, Project mutation and Shell execution fail; only exact plan-file `Read`/`Write` bypass Project containment. Approval restores execution; rejection keeps plan mode active.
C153: Claude MCP discovery starts configured servers without blocking the first model request, tracks request-scoped pending/connected/failed state, and exposes deferred server tools only through a real `ToolSearch` match or after `WaitForMcpServers`; every connected tool uses the collision-safe `mcp__<server>__<tool>` public name and remains callable through its owning client.
C154: Claude `LSP` is projected only when a command-backed language server is configured. The desktop host chooses that server from the contained Project file's language, synchronizes bounded UTF-8 content, and executes only the nine documented read-only LSP operations with request cancellation and timeout.
C155: Claude worktree state is request-scoped and operates only on the canonical `.claude/worktrees/<name>` path created or resumed by that request. Entering atomically rebinds guarded file, Shell, and LSP roots; exiting restores the original root. Removal refuses unverifiable state, uncommitted files, or commits after the entry baseline unless `discard_changes: true`, stops live processes rooted in the removed worktree, and never silently deletes an active worktree during manager cleanup.
C156: Claude `Agent` is exposed only with a real child-composition runner. Each invocation uses an independent tool manager rooted at the parent's current dynamic Project root, inherits the selected composition and cancellation boundary, returns only after a real child model response, and stores bounded request-scoped conversation history for explicit resume. Unsupported background, team, cwd, model-switch, and isolation semantics fail instead of silently degrading.
C157: `SendMessage` and `Workflow` remain unexposed until SwarmX has concurrent team mailboxes or the deterministic workflow VM required by their upstream contracts. `PowerShell` remains unexposed until a Windows-native sandboxed process host is available. Existing child resume, graph execution, and macOS background Shell polling are not presented as those missing semantics.
C158: Direct desktop Claude sessions may own a session-scoped runtime keyed by persisted `sessionId`. It serializes foreground turns and automatic activations, retains one sandboxed Shell across request tool-manager cleanup, and is closed on session deletion, Project-root replacement, or app shutdown.
C159: Claude `Monitor` is exposed only through that session runtime. Monitor commands retain the Project sandbox, stream bounded stdout lines as untrusted task notifications, apply timeout/persistent lifecycle and output-rate controls, and remain stoppable through the same `TaskStop` task id.
C160: Claude cron tools use the session runtime's local-time scheduler. Jobs are bounded, validated as standard five-field cron expressions with a next occurrence inside one year, serialized with foreground/monitor activations, and auto-delete after a one-shot fire. Session-only recurring jobs expire after three days. Durable jobs use the canonical Project's `.claude/scheduled_tasks.json`, survive session/app close, use Claude Code's seven-day recurring lifetime unless marked `permanent`, coordinate eligible execution through `.claude/scheduled_tasks.lock`, persist recurring `lastFiredAt`, and are visible/deletable from every session on that Project. A durable one-shot occurrence missed while SwarmX was closed is removed and converted to a confirmation-required activation instead of executing its original prompt automatically.
C161: A 3.1.1 release keeps every workspace package version aligned, publishes dependency-first from one verified commit, creates one matching `v3.1.1` tag/release, excludes generated residue and secrets, and does not present the three remaining TODO tools as implemented.
C162: Permission decisions and OS sandboxing remain separate layers. Approval may authorize one tool call inside its existing Project/runtime boundary; it never grants network, path, environment, or sandbox escalation, and ACP Harnesses keep their native permission semantics.
C163: Direct SwarmX tool policy resolves `deniedTools` before mode/allow rules, treats plan mode as a hard read-only boundary, and fails closed for unsupported modes, unavailable interaction bridges, or unclassified side-effecting tools.
C164: Desktop approval prompts bind request, renderer owner, interaction id, and offered option ids; task cancellation, renderer destruction, or tool-manager close rejects pending authority without a default approval. Prompt summaries remain bounded and omit file contents, patch bodies, and raw ACP input/output.
C165: Effective direct policy = Agent base + optional personal/managed ceilings + restriction-only Project layer from `<Project>/.swarmx/permissions.json`; deny union wins, least-authority declared mode wins, and Project content cannot add pre-approval.
C166: Managed policy uses explicit secret-free `SWARMX_MANAGED_PERMISSION_POLICY` JSON, remains renderer read-only, and malformed managed/Project policy blocks direct execution instead of disappearing or widening authority.
C167: Approval receipts persist ≤200 newest sanitized records containing time, source, tool label/kind, decision, and policy provenance only; no command, prompt, file content, patch, ACP raw payload, secret, credential, or durable allow rule.
C168: Permissions UX reuses existing Settings layout/tokens, separates effective authority from sandbox scope, uses structured allow/deny chips instead of newline policy text, exposes conflicts/source precedence, and keeps keyboard/narrow-width behavior complete.
C169: A conversation permission override is one of `inherit`, `default`, `auto`, `plan`, or `trusted`. `inherit` follows the personal and Agent defaults; an explicit conversation mode replaces those default mode declarations for that conversation, while managed/Project mode ceilings and every managed/Project/personal/Agent explicit deny remain authoritative.
C170: Conversation permission state persists in `SessionData`, defaults to `inherit` for existing sessions, is loaded by the Main process from the authoritative session id before direct-tool execution, and does not claim to override an external ACP Harness's native permission system.
C171: General Settings owns the availability of the three Codex-aligned conversation profiles (`default`, `auto`, and `trusted`). Advanced Permissions owns the personal fallback mode, conservative `plan`/`restricted` modes, exact-tool allow/deny rules, source hierarchy, sandbox explanation, and decision history; changing either surface preserves the other settings.
C172: Auto-review is an executable direct-tool mode, not decorative copy: read and Project-contained write calls are automatically allowed after deterministic policy review, execute/control calls still ask once, explicit denies and managed/Project ceilings still win, and the host OS sandbox never changes.
C173: Disabling a General permission profile removes it from new conversation choices and causes persisted/inherited uses of that profile to degrade to `plan` at the authoritative Main-process resolution boundary; `plan` and `inherit` remain available so General can never disable the safe path.
C174: Package installation never launches GUI code as a lifecycle side effect. No-argument `swarmx` launches Desktop; explicit CLI arguments retain existing commands and automation compatibility.
C175: macOS release packaging covers Apple silicon and Intel, uses tag/manifests as one version source, and supports optional Apple signing/notarization without making unsigned artifact generation impossible.
C176: Root README stays a short user entrypoint; detailed architecture, Provider, Extension, permission, and runtime material remains in `docs/`.

## §I
I1: `packages/core/src/types.ts` `SwarmConfigSchema`.
I2: `packages/desktop/src/preload/index.ts` `window.swarmxAPI.sendMessage`.
I3: `packages/desktop/src/main/ipc.ts` `agent:send`.
I4: `packages/desktop/src/renderer/src/App.tsx` desktop runtime UI.
I5: `packages/desktop/src/renderer/src/assets/styles.css` desktop layout.
I6: `packages/desktop/src/renderer/src/App.test.tsx` renderer behavior tests.
I7: `docs/index.md` workflow JSON docs.
I8: `packages/core/src/n8n.ts` n8n import converter.
I9: `packages/core/tests/n8n.test.ts` n8n converter tests.
I10: `packages/desktop/src/preload/index.ts` `window.swarmxAPI.importN8nWorkflow`.
I11: `packages/desktop/src/main/ipc.ts` `workflow:importN8n`.
I12: `packages/core/src/extensions.ts` extension schemas, inventory loading, and composition resolver.
I13: `packages/core/tests/extensions.test.ts` extension inventory and resolver tests.
I14: `packages/desktop/src/preload/index.ts` `window.swarmxAPI.listExtensions`.
I15: `packages/desktop/src/main/ipc.ts` `extension:list`.
I16: `packages/desktop/src/renderer/src/App.tsx` Extensions inventory view.
I17: `packages/desktop/src/renderer/src/assets/styles.css` Extensions inventory layout.
I18: `docs/geepilot-platform-review.md` downstream boundary review.
I19: `packages/core/src/autonomy.ts` deterministic autonomy schemas and replay helpers.
I20: `packages/core/tests/autonomy.test.ts` autonomy primitive tests.
I21: `docs/index.md` autonomy runtime primitive docs.
I22: `packages/core/src/context.ts` context packet schemas and helpers.
I23: `packages/core/tests/context.test.ts` context packet primitive tests.
I24: `packages/core/src/rendering.ts` normalized render event schemas and message mappers.
I25: `packages/core/tests/rendering.test.ts` normalized render event tests.
I26: `packages/desktop/src/renderer/src/App.tsx` desktop run timeline rendering.
I27: `packages/core/src/telemetry.ts` telemetry schemas, redaction, config, status, and injected client.
I28: `packages/core/tests/telemetry.test.ts` telemetry primitive tests.
I29: `packages/core/src/server.ts` optional HTTP server boundary.
I30: `packages/core/tests/server.test.ts` server boundary tests.
I31: `packages/cli/src/cli.ts` `swarmx serve` boundary flags.
I32: `packages/core/src/dependencies.ts` managed dependency schemas, policy validators, and planning helpers.
I33: `packages/core/tests/dependencies.test.ts` managed dependency primitive tests.
I34: `packages/core/package.json` browser-safe dependency subpath export.
I35: `packages/core/src/extensions.ts` marketplace source and plugin catalog inventory schemas.
I36: `packages/desktop/src/renderer/src/App.tsx` Extensions marketplace/catalog view.
I37: `packages/core/src/conversation.ts` append-only conversation ledger schemas and replay helpers.
I38: `packages/core/tests/conversation.test.ts` conversation ledger primitive tests.
I39: `packages/core/package.json` browser-safe conversation subpath export.
I40: `packages/core/src/providers.ts` provider profile metadata, secret status, selection, direct prompt, and runtime env helpers.
I41: `packages/core/tests/providers.test.ts` provider profile primitive tests.
I42: `packages/core/package.json` browser-safe provider subpath export.
I43: `packages/core/src/harness-management.ts` harness discovery, selector, alias, and invocation metadata helpers.
I44: `packages/core/tests/harness-management.test.ts` harness management primitive tests.
I45: `packages/core/package.json` browser-safe harness management subpath export.
I46: `packages/desktop/src/renderer/src/message-content.tsx` safe Markdown, media, and code-block renderer.
I47: `packages/desktop/src/renderer/src/message-content.test.tsx` message content rendering tests.
I48: `packages/core/src/actions.ts` action intent, risk, confirmation, and dependency-plan action helpers.
I49: `packages/core/tests/actions.test.ts` action primitive tests.
I50: `packages/core/package.json` browser-safe actions subpath export.
I51: `packages/core/src/agent-profiles.ts` agent definition frontmatter, profile metadata, parser, converter, and projection helpers.
I52: `packages/core/tests/agent-profiles.test.ts` agent profile definition tests.
I53: `packages/core/package.json` browser-safe agent profile subpath export.
I54: `packages/core/src/extensions.ts` extension agent profile definition and policy metadata.
I55: `packages/core/tests/extensions.test.ts` extension profile policy metadata tests.
I56: `packages/desktop/src/renderer/src/App.tsx` extension agent profile policy rendering.
I57: `packages/desktop/src/renderer/src/App.test.tsx` extension agent profile policy rendering tests.
I58: `packages/core/src/desktop-settings.ts` desktop settings, root resolution, locale registry, and UI state helpers.
I59: `packages/core/tests/desktop-settings.test.ts` desktop settings primitive tests.
I60: `packages/core/package.json` browser-safe desktop settings subpath export.
I61: `packages/core/src/extensions.ts` extended plugin component inventory schemas and aggregation.
I62: `packages/core/tests/extensions.test.ts` extended plugin component inventory tests.
I63: `packages/desktop/src/renderer/src/App.tsx` extended component inventory rendering.
I64: `packages/desktop/src/renderer/src/App.test.tsx` extended component inventory rendering tests.
I65: `packages/core/src/secrets.ts` secret refs, vault documents, redaction, status, file-mode, runtime lookup, and policy helpers.
I66: `packages/core/tests/secrets.test.ts` secret primitive tests.
I67: `packages/core/package.json` browser-safe secrets subpath export.
I68: `packages/core/src/extensions.ts` agent composition plan schemas, requirement records, and side-effect-free preflight helper.
I69: `packages/core/tests/extensions.test.ts` composition plan readiness, provenance, blocked-state, and secret-safety tests.
I70: `packages/desktop/src/main/ipc.ts` `extension:list` plan augmentation.
I71: `packages/desktop/src/renderer/src/App.tsx` extension agent readiness and resolved execution preview rendering.
I72: `packages/desktop/src/renderer/src/App.test.tsx` extension agent readiness rendering tests.
I73: `packages/core/src/autonomy.ts` scheduler/reporting schemas, schedule cadence helpers, feedback, dashboard, wakeup, and circuit-breaker contracts.
I74: `packages/core/tests/autonomy.test.ts` scheduler/reporting primitive tests.
I75: `packages/core/src/extensions.ts` passive GUI contribution schemas and aggregation.
I76: `packages/core/tests/extensions.test.ts` passive GUI contribution inventory tests.
I77: `packages/desktop/src/renderer/src/App.tsx` passive GUI contribution rendering.
I78: `packages/desktop/src/renderer/src/App.test.tsx` passive GUI contribution rendering tests.
I79: `packages/core/src/autonomy.ts` deterministic trigger, transition decision, validator gate, and replay record contracts.
I80: `packages/core/tests/autonomy.test.ts` deterministic runtime contract tests.
I81: `packages/core/src/autonomy.ts` engineering lifecycle intake, proposal, state, and transition decision contracts.
I82: `packages/core/tests/autonomy.test.ts` engineering lifecycle contract tests.
I83: `packages/core/src/extensions.ts` skill host exposure schemas and compatibility issue helper.
I84: `packages/core/tests/extensions.test.ts` skill host compatibility tests.
I85: `packages/desktop/src/renderer/src/App.tsx` skill host compatibility rendering.
I86: `packages/desktop/src/renderer/src/App.test.tsx` skill host compatibility rendering tests.
I87: `packages/desktop/src/renderer/src/message-content.tsx` safe Markdown and math renderer.
I88: `packages/desktop/src/renderer/src/message-content.test.tsx` message math rendering tests.
I89: `packages/desktop/src/renderer/src/assets/styles.css` message math layout styles.
I90: `packages/desktop/package.json` renderer math dependencies.
I91: `packages/desktop/src/renderer/src/message-content.tsx` structured code-block renderer.
I92: `packages/desktop/src/renderer/src/message-content.test.tsx` code-block rendering and copy tests.
I93: `packages/desktop/src/renderer/src/assets/styles.css` code-block control, highlighting, and fallback styles.
I94: `packages/desktop/src/renderer/src/code-highlighter.ts` browser-only offline Shiki highlighter loader.
I95: `packages/core/src/rendering.ts` normalized render event artifact option mapping.
I96: `packages/core/tests/rendering.test.ts` normalized render event artifact option tests.
I97: `packages/desktop/src/renderer/src/App.tsx` expandable tool and trace card rendering.
I98: `packages/desktop/src/renderer/src/App.test.tsx` trace-card rendering tests.
I99: `packages/desktop/src/renderer/src/App.tsx` specialized trace presentation detection and rendering.
I100: `packages/desktop/src/renderer/src/App.test.tsx` specialized trace presentation tests.
I101: `packages/core/src/autonomy.ts` agent-run and workflow-decision trace contracts.
I102: `packages/core/tests/autonomy.test.ts` agent-stage trace contract tests.
I103: `packages/core/src/telemetry.ts` telemetry ingest schemas, decisions, and injected append helper.
I104: `packages/core/tests/telemetry.test.ts` telemetry ingest contract tests.
I105: `packages/desktop/src/renderer/src/App.tsx` product config, GUI contribution registry, and registered contribution workspace rendering.
I106: `packages/desktop/src/renderer/src/App.test.tsx` registered GUI contribution host tests.
I107: `packages/desktop/src/renderer/src/assets/styles.css` registered GUI contribution navigation and workspace styles.
I108: `docs/index.md` downstream GUI host customization documentation.
I109: `packages/desktop/package.json` renderer host subpath export.
I110: `packages/core/src/extensions.ts` `executeAgentComposition`.
I111: `packages/core/tests/extensions.test.ts` agent composition execution helper tests.
I112: `packages/core/src/extensions.ts` LSP capability command and language metadata schema.
I113: `packages/core/tests/extensions.test.ts` LSP capability schema compatibility tests.
I114: `packages/desktop/src/main/lsp-host.ts` explicit stdio LSP process host.
I115: `packages/desktop/src/main/ipc.ts` `lsp:complete` and `lsp:stop`.
I116: `packages/desktop/src/preload/index.ts` `window.swarmxAPI.lspComplete` and `window.swarmxAPI.lspStop`.
I117: `packages/desktop/src/main/lsp-host.test.ts` LSP host lifecycle tests.
I118: `packages/core/src/extensions.ts` built-in `swarmx.local-files` LSP capability metadata.
I119: `packages/desktop/src/main/lsp-host.ts` built-in workspace local file completion provider.
I120: `packages/core/src/extensions.ts` built-in `swarmx.skills` LSP capability metadata.
I121: `packages/desktop/src/main/lsp-host.ts` built-in skill completion provider.
I122: `packages/desktop/src/main/harness-environment.ts` container runtime detection, Apple Container setup, and protected backend wrapping.
I123: `packages/desktop/src/main/ipc.ts` desktop harness environment IPC and protected harness send path.
I124: `packages/desktop/src/preload/index.ts` renderer-safe harness environment APIs.
I125: `packages/desktop/src/renderer/src/App.tsx` runtime setup and protected-mode UI.
I126: `packages/desktop/src/main/harness-environment.test.ts` container runtime setup and backend wrapping tests.
I127: `packages/desktop/src/renderer/src/App.test.tsx` protected-mode setup UI tests.
I128: `docs/index.md` and `DESIGNS.md` desktop containerized harness documentation.
I129: `packages/runtime/src/harness-environment.ts` shared host runtime detection, setup, and protected backend service.
I130: `packages/runtime/src/doctor.ts` shared doctor report, repair plan, and confirmed fix service.
I131: `packages/runtime/src/doctor.test.ts` doctor inspection, planning, confirmation, repair, and Hermes-local tests.
I132: `packages/cli/src/doctor.ts` and `packages/cli/src/cli.ts` `swarmx doctor` command surface.
I133: `packages/desktop/src/main/ipc.ts` and `packages/desktop/src/preload/api.ts` doctor IPC surface.
I134: `packages/desktop/src/renderer/src/App.tsx` and `App.test.tsx` slash command and on-demand doctor panel.
I135: `packages/core/src/model-capabilities.ts` standalone Model, many-to-many ModelSupply, and `Harness x Model` capability resolution contracts.
I136: `packages/core/src/harness.ts` harness API compatibility and request-scoped model-control metadata.
I137: `packages/core/src/providers.ts` provider supply routing and request-scoped runtime environment construction without provider-owned model state.
I138: `packages/core/src/acp.ts` stable ACP session configuration option negotiation for model and reasoning effort.
I139: `packages/core/src/extensions.ts` model, model-supply, and `Harness x Model` composition inventory.
I140: `packages/runtime/src/harness-environment.ts` protected-runtime model bootstrap environment and host bridge translation.
I141: `packages/desktop/src/renderer/src/App.tsx` and `App.test.tsx` three-choice `Harness x Model` Agent picker with internal supply routing.
I142: `packages/core/tests/model-capabilities.test.ts`, `providers.test.ts`, `acp.test.ts`, `harness.test.ts`, and `extensions.test.ts` matrix and launch behavior tests.
I143: `docs/index.md` and `DESIGNS.md` standalone Model and harness model-control documentation.
I144: `packages/desktop/src/main/model-catalog.ts` provider discovery, manual settings, cache, merge, and persistence service.
I145: `packages/desktop/src/main/model-catalog.test.ts` discovery, merge, persistence, timeout, cache, and secret-safety tests.
I146: `packages/desktop/src/main/ipc.ts` and `packages/desktop/src/preload/api.ts` renderer-safe Model catalog IPC.
I147: `packages/desktop/src/renderer/src/App.tsx` and `App.test.tsx` Model refresh and manual-entry surface inside the three-choice Agent picker.
I148: `packages/core/src/extensions.ts` internal deterministic ModelSupply resolution for compositions that omit user-facing supply selection.
I149: `docs/index.md`, `README.md`, and `DESIGNS.md` dynamic Model catalog documentation.
I150: `packages/core/src/providers.ts`, `extensions.ts`, and provider tests explicit auth-mode and in-memory runtime secret override contracts.
I151: `packages/desktop/src/main/provider-auth.ts` and `.test.ts` encrypted Provider secret persistence abstraction.
I152: `packages/desktop/src/main/model-catalog.ts` and `.test.ts` user Provider settings CRUD, auth readiness, discovery, cache, and runtime-secret resolution.
I153: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/preload/api.ts`, and preload tests Provider-management IPC and safe-storage adapter.
I154: `packages/desktop/src/renderer/src/App.tsx`, `App.test.tsx`, and styles single-action lower-left Settings menu plus integrated Provider-management and Usage workspace.
I155: `docs/index.md`, `README.md`, and `DESIGNS.md` Provider connection and encrypted-auth documentation.
I156: `packages/desktop/src/main/updater.ts`, `index.ts`, and updater tests npm metadata check, integrity-verified download, versioned install, and Electron relaunch service.
I157: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/preload/api.ts`, preload tests, and renderer update state/progress IPC.
I158: `packages/desktop/src/renderer/src/App.tsx`, `App.test.tsx`, styles, docs, and rendered QA for the Codex-style account-row update control.
I159: `packages/desktop/src/main/provider-usage.ts` and `.test.ts` fixed-origin Provider balance/quota adapters, bounded response parsing, secret-safe failure isolation, and Codex app-server rate-limit bridge.
I160: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/preload/api.ts`, and preload tests renderer-safe Provider Usage refresh IPC.
I161: `packages/desktop/src/renderer/src/App.tsx`, `App.test.tsx`, styles, `README.md`, `DESIGNS.md`, and `docs/index.md` Provider-adjacent balance/quota presentation and support matrix.
I162: `packages/desktop/src/main/model-catalog.ts`, catalog tests, renderer Provider cards/tests, and Provider docs canonical vendor naming plus credential-source presentation.
I163: `packages/desktop/src/main/model-catalog.ts`, `provider-usage.ts`, their tests, renderer fixtures, and desktop docs removal of ambient-environment Provider synthesis.
I164: `packages/core/src/providers.ts`, `extensions.ts`, `packages/desktop/src/main/model-catalog.ts`, and their tests Provider-native multi-entrypoint routing with one secret reference.
I165: `packages/desktop/src/main/provider-usage.ts`, IPC/preload surfaces, and tests targeted Provider refresh plus secret-safe New API account/token summaries.
I166: `packages/desktop/src/renderer/src/App.tsx`, styles, local brand assets, renderer tests, and `design-qa.md` unified Provider usage matrix.
I167: `packages/core/src/agent.ts`, `types.ts`, `harness.ts`, `extensions.ts`, and their tests SwarmX native Anthropic, OpenAI Responses, OpenAI Chat, and bridge-fallback execution.
I168: `packages/desktop/src/main/model-catalog.ts`, `provider-usage.ts`, and catalog tests legacy Provider normalization, Codex app-server model discovery, per-Provider cache metadata, New API group preservation, and stable display labels.
I169: `packages/core/src/model-capabilities.ts`, `extensions.ts`, renderer Agent picker files, and their tests Provider/group model presentation, internal ModelSupply route selection, and provider-advertised reasoning metadata.
I170: `packages/core/src/skill-variants.ts`, browser-safe exports, and tests logical Skill variants, Agent bindings, delivery metadata, deterministic resolution, context cost, lineage, evaluation, promotion, and rollback contracts.
I171: `packages/core/src/extension-management.ts`, browser-safe exports, and tests marketplace source state, plugin revision/action plans, install/update receipts, trust changes, immutable upstream provenance, and rollback contracts.
I172: `packages/desktop/src/main/settings-store.ts`, `custom-agents.ts`, `extension-manager.ts`, and tests shared atomic Settings persistence plus host-owned Custom Agent and Extension actions.
I173: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/preload/api.ts`, and preload tests renderer-safe Custom Agent, Extension management, and Runtime Settings IPC.
I174: `packages/desktop/src/renderer/src/App.tsx`, `App.test.tsx`, and styles Settings navigation plus Extensions, Custom Agents, and Runtime workspaces and removal of the conversation-sidebar Doctor status control.
I175: `packages/runtime/src/harness-environment.ts`, desktop Harness environment adapters, and tests separation of Harness software health from shared Runtime dependency health.
I176: `README.md`, `DESIGNS.md`, `docs/index.md`, and `design-qa.md` Custom Agent, Extension lifecycle, Runtime, Skill variant/evolution, trust, and migration documentation.
I177: `packages/runtime/src/harness-environment.ts`, `packages/core/src/harness.ts`, desktop Runtime IPC/UI, and tests Node.js baseline detection, npm-distributed Harness setup, semantic-version normalization, and embedded Doctor interaction.
I178: `packages/core/src/agent-profiles.ts`, `packages/core/package.json`, and exports dual Claude Code Markdown/Codex TOML definition codecs and normalized native metadata.
I179: `packages/core/tests/agent-profiles.test.ts` dual-format parsing, projection, round-trip, inheritance, and secret-safety tests.
I180: `packages/desktop/src/main/custom-agents.ts` read-only native Agent directory discovery and deterministic precedence.
I181: `packages/desktop/src/main/custom-agents.test.ts` native discovery, warning isolation, precedence, and no-persistence tests.
I182: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/renderer/src/App.tsx`, and renderer tests native Agent inventory merge and source presentation.
I183: `README.md`, `DESIGNS.md`, `docs/index.md`, and `docs/extensions-custom-agents.md` native Agent compatibility and execution-boundary documentation.
I184: `packages/desktop/src/renderer/src/App.tsx` and `App.test.tsx` shared Composer/Custom Agent routed Model option resolution, Provider grouping, ordering, and selection tests.
I185: `packages/core/src/project.ts`, session persistence/discovery, desktop project IPC/preload, workspace tools, renderer sidebar, and focused tests project registry plus project-bound task execution.
I186: `@swarmx/core/project`, `packages/desktop/src/preload/api.ts`, and `packages/desktop/src/renderer/src/App.tsx` synchronous persisted-Project bootstrap snapshot.
I187: `packages/core/src/agent.ts`, `mcp.ts`, `extensions.ts`, and tests host-injected local tool execution for single-Agent compositions.
I188: `packages/desktop/src/main/workspace-tools.ts`, `ipc.ts`, and tests Project identity plus bounded workspace list/read tools for native SwarmX sessions.
I189: `packages/core/src/types.ts`, `session.ts`, and `session-discovery.ts` persisted message timing and local Session pin/title state.
I190: `packages/desktop/src/main/session-title.ts`, `ipc.ts`, preload API, and tests cheap-model automatic title generation plus local Session mutations.
I191: `packages/desktop/src/renderer/src/App.tsx`, tests, and styles compact Worked disclosure, elapsed time, rename dialog, and task context menu.
I192: `packages/desktop/src/main/workspace-tools.ts` and focused tests Project-rooted read, atomic write, exact edit, content-digest concurrency protection, and direct SwarmX tool registration.
I193: `packages/desktop/src/main/workspace-shell.ts`, request cancellation, IPC composition wiring, and focused tests sandboxed Project Shell execution.
I194: `packages/desktop/src/renderer/src/App.tsx`, `App.test.tsx`, and `assets/styles.css` unboxed Worked reasoning flow.
I195: `packages/core/src/swarm.ts`, `extensions.ts`, and `native-model.ts` request-scoped streaming propagation and Responses output reconstruction.
I196: `packages/desktop/src/main/ipc.ts`, Preload API, Renderer, and tests `agent:chunk` live-run lifecycle.
I197: `docs/native-tool-compatibility.md` Claude Code/Codex source audit and compatibility policy.
I198: `packages/core/src/mcp.ts`, `agent.ts`, and `native-model.ts` function/freeform local-tool transport.
I199: `packages/desktop/src/main/workspace-tools.ts`, `workspace-shell.ts`, and tests model-family coding-tool profiles.
I200: `packages/desktop/THIRD_PARTY_NOTICES.md` Codex `apply_patch` grammar attribution.
I201: `packages/core/src/mcp.ts`, `types.ts`, `rendering.ts`, `agent.ts`, and `native-model.ts` dual tool-result transport.
I202: `packages/desktop/src/main/workspace-shell.ts` managed background process sessions.
I203: `packages/desktop/src/main/workspace-tools.ts` Claude Code/Codex result adapters and session tools.
I204: `packages/core/tests/agent.test.ts`, `rendering.test.ts`, and desktop workspace tool/Shell tests output and lifecycle coverage.
I205: `packages/desktop/src/main/workspace-shell.ts`, `terminal-host.ts`, and `node-pty` real PTY transport.
I206: `packages/desktop/src/main/workspace-tools.ts` request-scoped Claude task/Todo/review state and guarded notebook editing.
I207: `docs/claude-code-tool-parity.md` complete official-tool inventory, implementation state, and non-faked capability gaps.
I208: `packages/core/src/mcp.ts` and `agent.ts` conditional Claude MCP resource tool projection.
I209: `packages/desktop/src/main/ipc.ts` and `workspace-tools.ts` selected Skill projection, bounded loading, and argument substitution.
I210: `packages/desktop/src/main/agent-interactions.ts`, `ipc.ts`, and preload API request-scoped interactive tool transport.
I211: `packages/desktop/src/renderer/src/agent-interaction-dialog.tsx`, `App.tsx`, and `assets/styles.css` question and plan-approval dialogs.
I212: `packages/core/src/mcp.ts`, `agent.ts`, and `native-model.ts` background MCP connection state, Claude deferred-tool discovery, and per-step tool projection.
I213: `packages/desktop/src/main/lsp-host.ts`, `workspace-tools.ts`, and `ipc.ts` Claude LSP operation projection over configured desktop language servers.
I214: `packages/desktop/src/main/workspace-tools.ts` and `workspace-shell.ts` request-scoped Claude Git worktree lifecycle and dynamic workspace-root binding.
I215: `packages/desktop/src/main/ipc.ts`, `child-agent-host.ts`, and `workspace-tools.ts` synchronous Claude child-Agent composition bridge and request-scoped resume state.
I216: `packages/desktop/src/main/claude-session-runtime.ts`, `workspace-shell.ts`, `workspace-tools.ts`, IPC/preload session-message transport, and Renderer authoritative-session refresh for Monitor/Cron activation.
I217: `packages/desktop/src/main/claude-scheduled-tasks.ts`, `claude-session-runtime.ts`, and focused tests Claude Code-compatible scheduled-task persistence, lock ownership, startup recovery, and multi-session refresh.
I218: root/workspace `package.json` manifests, npm package tarballs, Git `main`/`v3.1.1`, and the GitHub release are the 3.1.1 release surfaces.
I219: `packages/core/src/skill-variants.ts`, `extensions.ts`, `agent.ts`, and `acp.ts` typed Harness permission policy, deterministic tool decision, composition projection, and optional ACP permission handler.
I220: `packages/desktop/src/main/workspace-tools.ts`, `agent-interactions.ts`, `ipc.ts`, Preload API, Renderer dialog/Settings, and focused tests direct-tool plus ACP approval enforcement.
I221: `docs/native-tool-compatibility.md` Claude Code/Codex permission-model audit, SwarmX policy semantics, and staged follow-up plan.
I222: `packages/core/src/skill-variants.ts`, `desktop-settings.ts`, browser-safe exports, and tests layered permission policy, provenance, personal settings, and sanitized approval receipt contracts.
I223: `packages/desktop/src/main/permission-service.ts`, `ipc.ts`, Preload API, and tests managed/Project/personal/Agent resolution, fail-closed loading, receipt persistence, and renderer-safe IPC.
I224: `packages/desktop/src/renderer/src/App.tsx`, `agent-interaction-dialog.tsx`, `assets/styles.css`, and tests dedicated Permissions workspace, structured Agent rule editor, effective-policy explanation, approval history, and one-call dialog UX.
I225: `packages/core/src/types.ts`, `session.ts`, `skill-variants.ts`, exports, and focused tests persisted conversation permission mode plus session-layer policy provenance.
I226: `packages/desktop/src/main/permission-service.ts`, `ipc.ts`, Preload API, and tests authoritative conversation override loading and safe layered resolution.
I227: `packages/desktop/src/renderer/src/App.tsx`, `assets/styles.css`, renderer tests, and `design-qa.md` General default-permission placement, Advanced Permissions split, Composer permission menu, persistence, responsive interaction, and visual QA.
I228: `packages/core/src/desktop-settings.ts`, `types.ts`, `skill-variants.ts`, Main permission service/IPC, Preload API, Renderer, focused tests, permission documentation, and visual QA Codex-aligned profile availability plus executable Auto-review semantics.
I229: `packages/swarmx/bin/swarmx.js`, package manifest, and focused tests npm-installed default Desktop launcher plus explicit CLI routing.
I230: `packages/desktop/package.json` and `electron-builder.yml` packaged Electron runtime plus macOS DMG/ZIP configuration.
I231: `.github/workflows/release.yml` tag-version validation, dual-architecture macOS builds, artifact upload, and GitHub Release creation.
I232: `README.md`, workspace manifests, lockfile, runtime version, and version tests concise cold-start guidance plus 3.1.2 release alignment.
I233: `packages/desktop/src/main/provider-key-pool.ts` and focused tests encrypted-key routing metadata, local usage ledger, quota classification, cooldown, and failover policy.
I234: `packages/desktop/src/main/provider-auth.ts`, `model-catalog.ts`, `provider-usage.ts`, and focused tests Custom Provider discovery plus OpenCode Go routing, encrypted key-pool persistence, and local usage projection.
I235: `packages/desktop/src/main/ipc.ts`, `packages/desktop/src/preload/api.ts`, Renderer Provider Settings, styles, and focused tests request-scoped key selection, safe retry, key management, and local status display.
I236: `docs/index.md` and `DESIGNS.md` Custom Provider and OpenCode Go key-pool documentation.

## §V
V1: Workflow JSON source of truth is `SwarmConfig`; UI preview, run badges, and send payload derive from parsed JSON.
V2: When `Use workflow` is enabled, invalid workflow JSON blocks the entire send request instead of falling back to a manual Agent; UI shows an actionable parse/shape error.
V3: Valid workflow JSON renders node cards and edge list from `nodes` and `edges`.
V4: Sending a message with valid workflow JSON includes the exact parsed `swarmConfig` in `sendMessage`.
V5: Sending without active workflow JSON keeps old `{ harnessId, userText }` payload shape.
V6: Docs examples validate against actual `SwarmConfigSchema`; no stale `type` node shape.
V7: Default workflow nodes render and serialize ACP agent identity: every default `kind: "agent"` node shows model and harness/backend, and no default node is presented as a standalone tool.
V8: Default workflow agent nodes render and serialize fine-grained Harness descriptors, including Software version, selected Skills and resolved variants, selected MCP servers, delivery/permission policy, and project files; different recipes or Software versions must not collapse to the same Harness revision.
V9: n8n workflow JSON with `nodes` and `connections` converts into valid `SwarmConfig` with unique sanitized node ids, a derived root, and matching edges.
V10: n8n import preserves each node's n8n name, type, type version, position, parameters, disabled state, notes, and credential references under `parameters.n8n`, without credential secret values.
V11: Invalid n8n JSON returns an actionable import error and must not replace the current workflow JSON.
V12: Desktop import uses preload IPC, replaces the workflow editor JSON with the converted `SwarmConfig`, enables the workflow, and shows import warnings.
V13: n8n connections to missing nodes and cycle-forming connections are reported as warnings; cycle-forming edges are made non-executable so SwarmX keeps a valid DAG.
V14: Extension bundles model software, standalone models, many-to-many model supplies, skills, MCP servers, provider profiles, harnesses, agent profiles, and app connectors without requiring downstream code to fork SwarmX schemas.
V15: Extension inventory loads built-in SwarmX harnesses plus path manifests from `SWARMX_EXTENSION_PATHS` or `SWARMX_EXTENSION_ROOTS`, reporting manifest errors as warnings instead of crashing the GUI.
V16: Agent composition resolution fails on missing or ambiguous agent profiles, harnesses, MCP servers, models, or explicitly requested model supplies; it must not silently fall back to defaults.
V17: Extension manifests and resolved agent configs do not copy inline secret values. Secret references may identify where a caller should request or inject a secret later.
V18: Settings > Extensions separates marketplace sources, plugin packages, executable Harness software, Agent profiles, Provider profiles, Skills, MCP servers, and app connectors; Skills and MCP servers never share one undifferentiated lifecycle state.
V19: Selecting an extension agent profile in the desktop GUI sends an `agentComposition` payload that the main process resolves against the extension inventory; unknown extension harness ids must not be forced through the built-in harness registry.
V20: `AgentBackendSchema` custom backends execute through the ACP client with the resolved command, args, process cwd/env, agent instructions, and latest user request; they must not silently fall back to the native OpenAI call path.
V21: Explicit model supplies inject scoped provider runtime env only for the selected request. Missing secret env vars, unsupported secret sources, or a supply that does not link the selected model and provider fail before execution; provider metadata never decides harness compatibility.
V22: Autonomy work items, runtime events, command DAGs, validator manifests, and evidence packets have exported zod schemas and TypeScript types with stable prefixes for portable ledger state.
V23: Autonomy runtime replay is deterministic: duplicate idempotency keys apply once, valid state transitions advance work items, and invalid transitions record rejection events without moving state.
V24: Command DAG validation rejects missing dependencies, duplicate node ids, cycle-forming dependencies, and nodes that declare neither a command nor an internal operation.
V25: Evidence packets expose reusable provenance fields for workspace, inputs, commands, parameters, environment, artifacts, validation, limitations, observations, conclusions, confidence, and follow-up without embedding downstream scientific interpretation policy.
V26: Autonomy runtime records reject inline secret-looking fields while allowing secret references as metadata.
V27: Context object, packet, summary checkpoint, and invocation context metadata have exported zod schemas and TypeScript types for downstream delegation and resumption; invocation identity is Harness x Model with optional ModelSupply routing and no Provider identity.
V28: Context strategy resolution is deterministic: `auto` resolves to `checkpoint_tail` when a usable checkpoint exists and to `microcompact` otherwise.
V29: Context packet construction preserves the delegated request, records included/dropped/truncated object ids and included message ids, and computes prompt byte count plus SHA-256 from the rendered prompt.
V30: Context packet validation rejects stale prompt byte counts or prompt hashes.
V31: Context records reject inline secret-looking fields while allowing secret references as metadata.
V32: Normalized render events expose stable ids, kind, status, source, parent references, title, summary, sanitized input/output, artifact references, raw payload references, and provenance without branching by host adapter; Agent provenance uses Harness x Model and optional ModelSupply rather than Provider identity.
V33: Message chunks map deterministically into normalized render events, including running tool calls, succeeded or failed tool results, completed assistant/thinking messages, and stable `rne_` ids.
V34: Render payload sanitization redacts secret-looking fields in tool inputs, tool outputs, artifacts, and provenance before GUI display.
V35: Desktop run events render tool-call and tool-result content from normalized sanitized payloads rather than raw host output.
V36: The desktop renderer imports normalized rendering through `@swarmx/core/rendering`, a browser-safe subpath that does not bundle Node-only core modules.
V37: Telemetry envelopes have exported zod schemas and TypeScript types for v1 schema version, event id, timestamp, lower-snake event type, source, pseudonymous installation id, optional session/release metadata, and sanitized payload.
V38: Telemetry config resolution keeps telemetry disabled by default and sends only when enabled, endpoint, and installation id are present.
V39: Telemetry payload sanitization redacts provider keys, bearer tokens, telemetry tokens, SMTP passwords, private credentials, and default raw content fields before envelope validation.
V40: Telemetry v1 rejects Task-product event names such as `task_created` and `task_updated`.
V41: Telemetry client sends through injected adapters, adds bearer headers without copying tokens into envelopes, and returns `outboxed` instead of throwing when a sender fails and an outbox is provided.
V42: The optional server binds to loopback by default and refuses non-loopback hosts unless a bearer token is configured before listen.
V43: Server requests with browser `Origin` headers are rejected unless the origin is explicitly allowlisted; wildcard origins are invalid configuration.
V44: `Origin: null` is rejected unless trusted desktop bridge mode is explicitly enabled.
V45: Configured server bearer tokens are required on requests and are exposed through explicit CLI flags rather than implicit ambient state.
V46: Managed dependency primitives expose exported zod schemas and TypeScript types for dependency kind, owner, version source, platform entries, manifests, detection results, install receipts, and install plans.
V47: Managed dependency manifest parsing accepts GEEPilot-style snake_case fields, normalizes them to camelCase, rejects duplicate dependency ids, and rejects unsupported dependency kinds, owners, version sources, detection statuses, and install actions.
V48: Managed download policy rejects inline secrets, `latest` or range-style versions, non-fixed version sources, missing install roots, unknown install-root styles, missing platform entries, missing SHA-256 hashes, non-HTTPS or credentialed URLs, unsafe archive members, and unsafe target names.
V49: Dependency detection results and install receipts reject inline secret-looking fields, while receipts also reject source URLs with embedded credentials.
V50: Dependency install planning is side-effect free: existing detections produce `use_existing`, managed binaries can produce `install_managed`, external harness CLIs and managed installers require explicit user action, failed detections produce `unavailable`, and benchmark assets never become product-startup installs.
V51: The dependency API is exposed through `@swarmx/core/dependencies`, a browser-safe subpath that does not bundle Node-only core modules.
V52: Extension inventory schemas expose marketplace source metadata with host, kind, path/url/package, enabled state, trust, read-only state, and plugin catalog metadata with source id, bundle id, host compatibility, install/update state, component counts, and runnable-harness flag.
V53: Extension inventory loading aggregates marketplace sources and plugin catalog entries from extension bundles while preserving existing bundle, harness, provider, agent, skill, MCP, and connector aggregation.
V54: Desktop Extensions view renders marketplace sources and plugin catalog entries in sections separate from plugin bundles, executable harnesses, agent profiles, providers, skills, MCP servers, and app connectors.
V55: Marketplace/catalog rendering remains read-only by itself. Install, update, enable, disable, rollback, hook activation, MCP startup, software execution, and trust changes are separate explicit Extension-manager actions with validated intent, confirmation when risk changes, and an audit receipt.
V56: Conversation ledger primitives expose exported zod schemas and TypeScript types for context strategy, storage refs, session index records, artifact refs, rollout event types, rollout events, and replay state.
V57: Conversation event creation is deterministic when an event id is not supplied, using stable `cev_` ids derived from event content.
V58: Conversation JSONL helpers serialize stable event lines, parse append-only rollout logs, ignore blank lines, and include the failing line number in parse errors.
V59: Conversation replay reconstructs session index state and message arrays from ordered events, applies title/archive/lifecycle updates, and records missing-session or duplicate-session events as rejected rather than mutating state.
V60: Conversation records reject inline secret-looking structured fields while allowing secret references as metadata.
V61: The conversation API is exposed through `@swarmx/core/conversation`, a browser-safe subpath that does not bundle Node-only core modules.
V62: Provider supply primitives expose exported zod schemas and TypeScript types for API kind, secret source, secret reference, connection profile metadata, secret status, explicit provider selection, direct provider prompt request, and request-scoped runtime env.
V63: Provider profile parsing accepts downstream snake_case connection fields and `label` aliases while rejecting provider-owned `model`, `models`, `defaultModel`, `default_model`, and harness model override fields.
V64: Provider profile metadata, secret status records, provider selections, and direct prompt requests reject inline secret-looking fields, while secret status records also reject returned secret values.
V65: Provider profile and model-supply resolution fail on unknown, ambiguous, unlinked, or missing explicit selections instead of silently falling back to a default provider or supply.
V66: Provider runtime env construction uses the explicitly selected Model and ModelSupply, supports declared API bridging, requires an explicit runtime secret value when a profile has a secret reference, and keeps secrets and model ownership out of persisted provider metadata.
V67: The provider API is exposed through `@swarmx/core/providers`, a browser-safe subpath that does not bundle Node-only core modules.
V68: Harness management primitives expose exported zod schemas and TypeScript types for adapter availability, host scope, discovery records, agent aliases, selector resolution, invocation status, and invocation metadata.
V69: Harness selector resolution strips leading `@harness`, `@harness:model`, and named-agent selectors from delegated prompts while returning canonical selectors plus selected Harness, Model, optional ModelSupply, and profile metadata; Provider is never selector or Agent identity.
V70: Unknown harness selectors, unknown Harness-Model selectors, and ambiguous named-agent aliases fail instead of falling back to a default harness or model.
V71: Harness discovery records and invocation metadata reject inline secret-looking fields while allowing secret references as metadata.
V72: The harness management API is exposed through `@swarmx/core/harness-management`, a browser-safe subpath that does not bundle Node-only core modules.
V73: Conversational message rendering escapes raw HTML while preserving Markdown inline code and GFM rendering.
V74: Remote Markdown images render as placeholders by default, while local image paths resolve only through preload IPC before an `<img>` receives a data URL.
V75: Fenced code blocks render in stable framed blocks with a language label, preserve literal code text, and keep tool outputs as literal text instead of Markdown.
V76: Action intent primitives expose exported zod schemas and TypeScript types for action kinds, risks, hosts, source refs, intents, confirmations, and confirmed actions.
V77: Action intent creation is deterministic when an action id is not supplied, using stable `act_` ids derived from sanitized intent content.
V78: Mutating, code-executing, destructive, trust-changing, download, write, network, or secret-exposure actions require explicit confirmation text and a matching positive confirmation before execution.
V79: Read-only actions can be represented without confirmation, and the `read_only` risk cannot be combined with mutating or sensitive risks.
V80: Action records reject inline secret-looking fields, while constructor helpers sanitize action payload secrets and allow secret references as metadata.
V81: Dependency install plans map to side-effect-free action intents: managed installs and external installers require explicit confirmation, while existing or unavailable dependencies remain read-only.
V82: The action intent API is exposed through `@swarmx/core/actions`, a browser-safe subpath that does not bundle Node-only core modules.
V83: Agent profile definition primitives expose exported zod schemas and TypeScript types for Claude-compatible frontmatter, GEEPilot-scoped frontmatter metadata, definition documents, definition sources, and agent profile metadata.
V84: Agent definition Markdown parsing preserves the body prompt, parses YAML frontmatter, accepts Claude-compatible fields, keeps unknown frontmatter fields as inert metadata, and rejects malformed or non-object frontmatter.
V85: Agent profile conversion maps definition frontmatter into separate profile metadata for harness id, model id, optional model-supply id, selector aliases, skills, tools, permission-like fields, source provenance, read-only state, and instructions without collapsing profiles into harnesses, models, or providers.
V86: Agent definition and profile records reject inline secret-looking fields while allowing secret references and source references as metadata.
V87: Claude Code projection omits GEEPilot-only frontmatter while preserving host-compatible fields and the Markdown body.
V88: The agent profile definition API is exposed through `@swarmx/core/agent-profiles`, a browser-safe subpath that does not bundle Node-only core modules.
V89: Extension agent profiles expose typed definition and policy metadata for tools, disallowed tools, permission mode, turn budget, memory, effort, background, isolation, color, and definition source while preserving `Harness x Model` identity and optional model-supply routing metadata.
V90: Agent composition resolution records extension profile policy and definition metadata in traceable parameters without enforcing host-specific policy or starting tools, hooks, MCP servers, or profile enablement.
V91: Desktop extension inventory renders agent profile policy metadata separately from harness, provider, skill, MCP, marketplace, and plugin catalog sections as passive inspection chips.
V92: Desktop settings primitives expose exported zod schemas and TypeScript types for settings documents, desktop root config, UI state, locale resources, locale registries, provider profile metadata arrays, and agent profile metadata arrays.
V93: Desktop root resolution is deterministic: explicit env desktop roots win over settings desktop roots, legacy app-root env/settings values are compatibility fallbacks, and server data roots are tracked separately rather than silently becoming desktop roots.
V94: Locale registry and selection helpers centralize locale metadata/resources, reject duplicate locale ids, resolve persisted locale choices when supported, and fall back to the default locale when selections are missing or unsupported.
V95: Desktop settings parsing preserves provider profile metadata and agent profile metadata without treating providers, harnesses, plugins, or agents as the same object.
V96: Desktop settings, UI state, locale records, provider profile metadata, and agent profile metadata reject inline secret-looking fields while allowing secret references as metadata.
V97: The desktop settings API is exposed through `@swarmx/core/desktop-settings`, a browser-safe subpath that does not bundle Node-only core modules.
V98: Extension bundle schemas expose passive component inventory for commands, LSP servers, hooks, monitors, output styles, settings, assets, permission declarations, and authentication policies while preserving existing software, skill, MCP, provider, harness, agent, connector, marketplace, and catalog fields.
V99: Extension inventory loading aggregates the extended component arrays from all bundles and keeps manifest errors as warnings rather than crashing the GUI.
V100: Extended plugin component inventory rejects inline secret-looking fields while allowing secret references and auth requirements as metadata.
V101: Desktop Extensions view renders extended component inventory separately from plugin bundles, executable harnesses, agent profiles, providers, skills, MCP servers, app connectors, marketplace sources, and plugin catalog entries.
V102: Extended component inventory rendering remains read-only and must not expose implicit command execution, LSP startup, hook activation, monitor startup, output style activation, settings writes, permission grants, asset opening, or authentication actions.
V103: Secret primitives expose exported zod schemas and TypeScript types for secret purposes, secret sources, secret references, vault entries, vault documents, write requests, secret statuses, and file-mode statuses.
V104: Secret vault documents and secret write requests can carry secret values only in explicit `value` fields, while redaction helpers produce safe documents that never contain the original values.
V105: Secret status and file-mode status records report configuration and safety metadata without returning secret values.
V106: Secret file-mode evaluation deterministically marks `0600` as secure, rejects broader modes such as `0644`, and keeps actual chmod/file creation outside core.
V107: Secret reference policy rejects persistent storage for request-only secrets such as remote-compute passwords while allowing provider API keys to reference env, local auth file, local keychain, server keychain, or prompt sources.
V108: Secret metadata rejects inline secret-looking fields outside the explicit vault/write `value` fields while allowing secret references as metadata.
V109: The secrets API is exposed through `@swarmx/core/secrets`, a browser-safe subpath that does not bundle Node-only core modules.
V110: Agent composition preflight exposes exported zod schemas and TypeScript types for stable status values, readiness requirements, selected capability refs, context policy summary, permission policy summary, visual metadata, and resolved execution plans.
V111: Composition plan resolution is side-effect free and reports agent id, display name, canonical selector, host, harness id, model id, optional model-supply id, behavior definition source, enabled plugin ids, selected skills, selected MCP servers, context, permissions, visual metadata, health status, and requirements without spawning ACP, starting MCP, reading secrets, calling providers, or mutating settings.
V112: Composition plan resolution marks missing, disabled, ambiguous, unavailable, or unsupported required components as `blocked` or `draft`, and execution resolution must continue to fail before invocation instead of silently replacing missing harnesses, models, explicitly selected supplies, plugins, skills, MCP servers, or context strategies.
V113: Selected skill and MCP refs in composition plans preserve source plugin ids when known; unknown selected skills, unknown MCP servers, disabled plugin sources, or missing plugin ids are visible as requirements rather than hidden behind an opaque agent name.
V114: Agent composition inputs, plans, and plan requirement records reject inline secret-looking fields while allowing secret references as metadata.
V115: Desktop `extension:list` returns read-only composition plans for extension agent profiles, and the Extensions view renders readiness, canonical selector, `Harness x Model` identity, optional supply label, selected skill/MCP provenance, plugin count, permission/context summaries, and blocked requirements before selection.
V116: Desktop Extensions rendering must not make blocked or disabled composition plans invokable from the read-only inventory view.
V117: Scheduler/reporting primitives expose exported zod schemas and TypeScript types for schedule state, schedule triggers, feedback actions, feedback records, report metadata, artifact dashboard metadata, wakeup controller state, daemon-run metadata, and circuit-breaker decisions.
V118: Schedule helpers deterministically decide when reports or wakeups are due from explicit timestamps and cadence, including the default 24-hour report cadence and default 60-second live-dashboard refresh interval, without reading clocks or writing files.
V119: Report metadata validates attempted/completed/failed/blocked/deferred work, evidence ids, artifact ids, verification outcomes, budget usage, decisions, risks, next work, and at most three human decision prompts.
V120: Artifact dashboard metadata distinguishes portable self-contained local dashboards from optional server-published dashboards, rejects external resource usage for offline local dashboards, and keeps dashboards as review surfaces rather than authoritative scheduler state.
V121: Feedback records preserve source report ids and accepted actions while mapping only to typed downstream routes. SwarmX records the route metadata but does not mutate memory, specs, papers, benchmarks, or work items from feedback.
V122: Wakeup controller state models app/server roles, adapter kind, desired cadence, heartbeat, next due time, last error, and tick-lock path without installing launchd/systemd/cron entries or starting timers.
V123: Scheduler/reporting records reject inline secret-looking structured fields while allowing secret references as metadata.
V124: Extension bundle schemas expose passive GUI contribution inventory for navigation items, views, panels, settings panels, dashboard widgets, composer actions, message actions, inspector sections, toolbar actions, menu items, and status items.
V125: GUI contribution records carry placement, route, component reference, asset reference, command id, setting ids, permission ids, auth policy ids, source plugin id, provenance, and read-only state as metadata without inline component bodies, inline HTML, inline scripts, or render functions.
V126: Extension inventory loading aggregates GUI contribution arrays from all bundles and keeps them separate from app connectors, commands, assets, settings, and executable harnesses.
V127: Plugin catalog component counts can report GUI contribution counts without making a plugin catalog entry executable or invokable.
V128: GUI contribution records reject inline secret-looking fields while allowing auth policy ids and secret references as metadata.
V129: Desktop Extensions view renders GUI contributions in a section separate from plugin bundles, marketplace sources, plugin catalog entries, generic plugin components, executable harnesses, agent profiles, providers, skills, MCP servers, and app connectors.
V130: GUI contribution rendering remains read-only and must not expose implicit navigation, component loading, script evaluation, iframe/webview mounting, command execution, asset opening, settings writes, permission grants, authentication actions, or trust-changing actions.
V131: Autonomy trigger records expose exported zod schemas and TypeScript types for trigger id, trigger type, timestamp, source, idempotency key, optional schedule/work/run/feedback/source refs, and sanitized payload.
V132: Trigger-to-runtime-event helpers are deterministic: the same trigger produces the same runtime event id, preserves idempotency, and records the trigger without creating work items, starting runs, or bypassing eligibility, lease, budget, or validation policy.
V133: Transition decision records expose exported zod schemas and TypeScript types for requested from/to state, decision status, preconditions, missing requirements, idempotency status, reason, and optional patch.
V134: Transition decision helpers are deterministic and side-effect free: they allow a transition only when current state matches, the status edge is valid, required preconditions pass, and the proposed idempotency key has not already been applied.
V135: Validator gate decision records expose exported zod schemas and TypeScript types for manifest id, gate id, required validator ids, missing validator ids, outcome status, required failures, waived validators, skipped validators, and reason without running validators.
V136: Replay record helpers summarize replay output deterministically with event counts, applied/rejected ids, work item ids, work item status counts, state hash, and explicit missing external dependencies without inventing external state.
V137: Trigger records, transition decisions, validator gate decisions, and replay records reject inline secret-looking structured fields while allowing secret references as metadata.
V138: Engineering lifecycle primitives expose exported zod schemas and TypeScript types for primary states, side states, intake source types, intake records, proposal records, approval records, and lifecycle transition decisions.
V139: Intake records preserve source id/type, received timestamp, title, summary, source refs, optional report/validation ids, initial autonomy level, and initial lifecycle state without creating work items or accepting execution.
V140: Proposal records preserve problem, affected surfaces, alternatives, compatibility, security/privacy notes, deterministic validation plan, migration/rollback plan, approvals, and status without becoming accepted specs by themselves.
V141: Lifecycle transition decision helpers are deterministic and side-effect free: they allow only valid lifecycle edges, match the current lifecycle state, require configured evidence ids, require configured approvals, and require a passing validator gate when supplied.
V142: Lifecycle transition decisions report missing evidence, missing approvals, invalid edge/current-state errors, validator-gate failures, status, reason, and optional next workflow state without mutating a work item.
V143: Lifecycle helper state maps work items through `workflow.kind = "engineering"` and typed lifecycle stages while preserving existing autonomy work-item status.
V144: Lifecycle intake, proposal, approval, and transition records reject inline secret-looking structured fields while allowing secret references as metadata.
V145: Skill capability schemas expose canonical path, governance ref, gate-skill ids, host exposure records, read-only state, and source plugin id while preserving existing skill id/name/path metadata.
V146: Skill host exposure records distinguish plugin, rules-only, unsupported, and unknown host surfaces and carry manifest path, marketplace source id, rules path, package, and read-only metadata without claiming executable harness support.
V147: Skill host compatibility validation is side-effect free and returns passive issues for skill paths outside configured canonical roots, missing or self-referential gate skills, unknown marketplace sources, rules-only manifest claims, and configured host local paths that do not use `./` repository-root form.
V148: Skill host compatibility helpers do not install plugins, mutate manifests, inspect host files, start adapters, enforce downstream biosecurity policy, or depend on `.mcp.json` or runtime-local memory roots.
V149: Skill host compatibility records reject inline secret-looking fields while allowing secret references as metadata.
V150: Desktop Extensions view renders skill canonical path, governance, gate-skill, host exposure, manifest/rules/source, and read-only metadata as passive chips in the Skills and MCP section without making them invokable.
V151: Conversational message rendering supports inline and display math from `$...$`, `\(...\)`, `$$...$$`, and `\[...\]` using offline KaTeX with unsafe trust disabled.
V152: Dollar-delimited inline math avoids obvious currency and shell-prompt false positives while preserving canonical message text and keeping tool outputs literal.
V153: Invalid or unsupported math degrades inside the affected formula block with the original source visible and does not hide surrounding message content.
V154: Display formulas, inline formulas, tables, and code blocks stay within the message width through wrapping or horizontal scrolling without overlapping neighboring content.
V155: Math rendering does not fetch remote scripts or assets and must continue to escape raw HTML and block remote Markdown images by default.
V156: Fenced code blocks render as structured figures with language labels, keyboard-reachable copy controls, and stable header/body layout.
V157: Copying a code block uses the exact original code text from the Markdown block and does not copy visual labels, line numbers, generated markup, or rewritten content.
V158: Known-language code highlighting is an offline renderer enhancement with escaped plain-text fallback for unknown languages, highlight failures, server rendering, or unavailable highlighter APIs.
V159: Code-block rendering must not execute code, fetch remote grammars or themes, expose raw HTML, or parse tool-call/tool-result content as Markdown.
V160: Long code lines and highlighted output stay inside the message width through internal horizontal scrolling without shifting neighboring content.
V161: Normalized message chunk mapping can carry caller-supplied artifact references into render events while preserving existing sanitized input/output and raw-payload-reference behavior.
V162: Desktop tool and trace events render an expandable details card with status, title, summary, sanitized input/output, provenance, artifact reference chips, and raw-payload reference when present.
V163: Trace-card controls are disclosure-only and must not open artifacts, reveal raw payload contents, execute software, rerun commands, or mutate trust/session state.
V164: Failed tool results remain visible with failed status and sanitized error details instead of collapsing into generic assistant text or hiding behind success summaries.
V165: Trace-card artifact and provenance fields reject or omit inline secret-looking values and never render provider keys, bearer tokens, passwords, private keys, credentials, cluster passwords, or remote-compute credentials.
V166: Trace-card layout stays within the message width on desktop and narrow viewports through wrapped metadata chips and internal scrolling for structured detail blocks.
V167: Desktop trace cards derive read-only specialized presentations for terminal, file, diff/patch, test/check, MCP, automation, and generated-media events from normalized sanitized payload fields, artifact refs, and provenance.
V168: Terminal presentations show command, working directory, status, duration, exit code, stdout, stderr, and truncation state when present without executing or rerunning the command.
V169: File, diff, test/check, MCP, automation, and generated-media presentations show only passive metadata and sanitized excerpts; file paths, artifact refs, screenshots, and media refs are not opened or fetched.
V170: Specialized trace presentation fallback remains the generic sanitized input/output card when a payload is unrecognized or missing required shape.
V171: Specialized trace blocks stay within the message width through wrapped field rows and internal scrolling for excerpt blocks.
V172: Agent-run records expose stable `agt_` ids, work/run refs, workflow kind/stage, role, Harness x Model identity, optional ModelSupply/adapter/profile refs, status, timestamps, artifact/evidence refs, compact summaries, and result/error refs without Provider identity or raw transcript content.
V173: Workflow-decision records expose stable `dec_` ids, work/run refs, workflow kind, current/next stage, decision status, decision kind, linked agent-run/evidence ids, reason, and optional next workflow state without mutating work items by themselves.
V174: Agent-run and workflow-decision helpers can project sanitized runtime events with deterministic ids and event types that downstream ledgers can append while keeping canonical records caller-owned.
V175: Agent-run and workflow-decision records reject inline secret-looking and raw-output fields while allowing explicit refs to caller-owned artifacts.
V176: Workflow-state helper updates agent-run ids and final decision id deterministically from existing workflow state without executing agents or deciding downstream policy.
V177: Telemetry ingest config is disabled by default and can require a caller-configured bearer token independent of provider telemetry client tokens.
V178: Telemetry ingest decisions reject unsupported schema versions and unsafe envelope payloads before append while preserving accepted event ids and timestamps.
V179: Telemetry ingest append handling uses a caller-injected store and returns accepted/rejected decisions without throwing through user workflows or writing files itself.
V180: Telemetry ingest accepted records contain stable `ing_` ids, received timestamp, envelope, source metadata, and optional storage refs without copying bearer tokens into stored records.
V181: Desktop exports a product-configurable App factory so downstream products can set brand title/subtitle and register `componentRef` handlers without forking the SwarmX renderer.
V182: Registered GUI contribution navigation is driven by extension inventory metadata and appears only for contributions with a matching host-registered component.
V183: Selecting a registered GUI contribution renders the registered component with contribution, inventory, and explicit host callbacks while preserving existing Workflow, Extensions, chat, and agent-selection flows.
V184: Unregistered GUI contributions remain passive inventory rows and are not navigable, mounted, fetched, evaluated, or executed from manifest metadata.
V185: `executeAgentComposition` resolves the selected composition against caller-supplied or loaded extension inventory, injects request-scoped runtime env into the resolved agent process only, builds a single-agent Swarm, and returns the resulting message chunks.
V186: `executeAgentComposition` rejects blocked compositions through the existing plan readiness checks and does not copy provider secret values into returned messages, plans, agent parameters, or inventory records.
V187: Extension LSP capability parsing accepts both array commands and string-plus-args commands, preserves `languages`, and accepts `languageIds` for host-specific language id declarations.
V188: A desktop LSP completion request starts or reuses only the requested declared stdio LSP server, initializes it with the supplied workspace root, and does not run from extension inventory loading or read-only rendering.
V189: LSP completion sends caller-supplied document text through `textDocument/didOpen`, forwards `textDocument/completion` with the requested position and trigger context, and returns the server completion result unchanged.
V190: `lsp:stop` explicitly shuts down and removes managed LSP server sessions by server id and optional workspace root without mutating extension manifests.
V191: LSP host requests reject unknown servers, missing commands, invalid workspaces, failed processes, and timed-out JSON-RPC requests with actionable errors while keeping stderr limited.
V192: The built-in extension inventory exposes a read-only `swarmx.local-files` LSP capability for local file references without starting a process during inventory loading.
V193: `swarmx.local-files` completion returns LSP completion items for bare `@` workspace-relative paths from caller-supplied text and position, including directory entries, without recursively scanning the workspace.
V194: `swarmx.local-files` rejects or returns no completions for scheme-qualified references, absolute paths, parent-directory traversal, missing directories, and non-`@` reference tokens so completions remain bounded to the workspace root.
V195: The built-in extension inventory exposes a read-only `swarmx.skills` LSP capability for `$` skill references without starting a process during inventory loading.
V196: `swarmx.skills` completion returns LSP completion items from `inventory.skills` for caller-supplied `$` tokens, preserving skill ids, names, paths, canonical paths, governance refs, and read-only metadata as passive item data.
V197: `swarmx.skills` returns no completions for non-`$` tokens and must not read skill files, execute skills, apply gate policy, or mutate inventory.
V198: Desktop harness environment status reports container backend support, installed CLI state, service readiness, setup availability, and the selected protection mode separately from harness CLI requirements.
V199: Apple Container setup verifies Apple silicon and supported macOS before installation, runs only from an explicit setup request, starts `container system start` after install or detection, and redetects status afterward.
V200: Protected ACP harness wrapping preserves stdin/stdout ACP transport, cwd, minimal environment, deliberate workspace mount, resource limits, and removable ephemeral container semantics without logging secret values.
V201: When protected mode is required and no container backend is ready, desktop blocks external ACP harness sends before spawning and returns setup-oriented state instead of silently falling back to native execution.
V202: Runtime diagnosis UI shows protected-mode status, unsupported reasons, and confirmed repair controls without terminal copy/paste instructions and remains stable at desktop and narrow widths.
V203: Tests cover Apple Container detection/setup, unsupported platform handling, protected backend wrapping, blocked protected sends, and renderer setup controls.
V204: Agent-picker primary panel height and anchor depend only on its three primary rows: Harness, Model, and Effort. Their secondary panels stay out of primary layout flow, own max-height, and scroll independently.
V205: Desktop Vitest resolves the core model-capability subpath from current workspace source, so core capability changes cannot be masked by stale package build output.
V206: Agent-picker Escape closes from trigger, primary, or secondary focus and restores focus to the trigger; ArrowDown opens from the trigger into the primary menu.
V207: A shared host runtime doctor returns stable health, issue, repair-action, and underlying environment-status records for all harnesses or one explicit harness filter.
V208: Doctor inspection calls environment detection only and never runs setup, installers, service starts, PATH writes, repository operations, or privileged actions.
V209: Doctor repair plans are deterministic, deduplicate actions, label safe/install/admin risk, and keep unsupported issues inside the explicit diagnostic scope visible without inventing a fix.
V210: Doctor fix executes no action until confirmation is explicit, then applies only planned requests, captures bounded logs, and redetects health afterward.
V211: Hermes detection includes `~/.hermes/hermes-agent` and its local virtual environment before offering installation; doctor never clones, fetches, pulls, or updates Hermes.
V212: `swarmx doctor` supports human and JSON reports, harness filtering, stable exit status, and an explicit `--fix` flow that previews and confirms mutations.
V213: Desktop `/doctor` opens a read-only environment panel, `/doctor --fix` opens a repair confirmation state, and `/setup` opens the same panel in first-run setup mode without sending a chat message.
V214: Desktop removes permanent Setup navigation, keeps a clickable runtime health indicator, preserves chat/composer context while diagnostics are open, and remains stable at narrow widths.
V215: Tests cover read-only inspection, confirmation gating, post-fix redetection, local Hermes preference, CLI formatting/options, slash dispatch, and transient panel behavior.
V216: Full-App renderer integration tests declare a bounded timeout above Vitest's unit-test default, so workspace-wide test concurrency does not turn module-load cost into a false product failure.
V217: Unfiltered environment status and Doctor treat missing, failed, or unsupported optional harnesses as availability metadata only: global health remains ready, issue/repair counts stay zero, and CLI exit status stays successful.
V218: An explicit harness filter, the selected desktop harness, or an active workflow dependency scopes diagnosis to that harness; unmet runtime or protection requirements then remain actionable and can block execution until repaired.
V219: No-argument setup or Doctor fix never installs optional harnesses or container support; installation requires an explicit harness, requirement, or container selection plus normal confirmation.
V220: Desktop runtime health does not count every unavailable registry entry, and the global Doctor panel renders optional harness availability neutrally while keeping chat and explicit scoped repair behavior intact.
V221: On macOS, sidebar navigation keeps a traffic-light-safe inset in both expanded and collapsed states; moving Open sidebar, Back, and Forward into the main title bar must not overlap native window controls.
V222: Model records have stable ids, labels, runtime model ids, supported API protocols, and model capabilities independently of any provider profile; model ids remain stable when supply providers change.
V223: ModelSupply records link one model id and one provider profile id, allow multiple providers per model and multiple models per provider, and may carry only route-specific runtime model and bridge metadata.
V224: Harness compatibility is resolved exclusively from harness model-control/API metadata and Model capability metadata; Provider never filters, grants, or denies a `Harness x Model` pair.
V225: Resolved agent identity is the stable pair `harnessId:modelId`; reasoning effort is the only user-facing execution option, while supply and bridge routing remain internal and never create a different model or agent identity.
V226: Harness model-control metadata distinguishes direct API selection, ACP session configuration, and unsupported external selection; omission preserves the harness default, while an explicit unsupported model request fails visibly.
V227: ACP launch applies a requested model through stable session `configOptions` first, refreshes returned options, then applies reasoning effort; legacy model negotiation is used only when the harness exposes no stable model option.
V228: Claude Code, Codex, OpenCode, and Hermes receive model choice through their supported request-scoped launch/session mechanisms; SwarmX never edits global vendor configuration, and a harness without a supported control surface remains an explicit matrix gap.
V229: Protected container launches forward only allowlisted request-scoped model/bootstrap variables and translate loopback yallm bridge URLs to the host bridge without changing Model identity or supply membership.
V230: Desktop exposes exactly three primary choices: Harness, Model, and Effort. The Model menu may show one `Harness x Model` identity under each usable Provider route, but choosing a routed item carries its internal ModelSupply id without adding a Provider/Supply primary row or changing Agent identity.
V231: Tests cover standalone model parsing, internal many-to-many supplies, provider-independent harness compatibility, stable and legacy ACP model negotiation, fixed Harness-Model launch routes, protected host bridging, three-row picker identity, and explicit unsupported gaps.
V232: Desktop Harness selection keeps disabled or `modelControl: unsupported` Harnesses visible but natively disabled with an actionable reason; pointer and keyboard interaction cannot make them active. OpenClaw remains disabled until its model-switching control is configured.
V233: Composer never renders a Supply primary row or supply-selection menu. Supply inventory may remain visible in advanced extension diagnostics, but ordinary users never choose a Provider or ModelSupply.
V234: The built-in `claude_code:deepseek-v4-pro` route resolves runtime model `deepseek-v4-pro[1m]`, requires `DEEPSEEK_API_KEY`, and injects the fixed DeepSeek Anthropic endpoint, Opus/Sonnet/Haiku aliases, flash sub-Agent model, and selected Claude Code effort without persisting or exposing the secret.
V235: Desktop catalog uses extension declarations, manual settings, and last-successful Provider discovery as list sources. The built-in registry only enriches matching ids with API/capability metadata and cannot display an undiscovered Model by itself; all sources merge by stable Model id without Provider-derived identity.
V236: Provider discovery supports OpenAI-compatible `data[].id`, Anthropic Models API pagination, and Ollama `models[].name`; each discovered item becomes an independent Model and an internal ModelSupply to the discovering connection.
V237: Desktop catalog discovery uses only user-managed Settings connections or explicit extension Provider profiles; ambient OpenAI, Anthropic, DeepSeek, and Ollama env variables never synthesize built-in Provider profiles.
V238: Manual Model add/remove persists through a secret-safe desktop settings document, validates id/runtime/API fields, and does not require or create a Provider.
V239: Model secondary UI starts with cached/manual inventory, refreshes Provider APIs only after explicit `Refresh Models`, reports bounded per-Provider failures without clearing usable Models, and keeps exactly Harness, Model, Effort as primary choices.
V240: Desktop execution resolves against the same augmented catalog shown to the user. If composition omits ModelSupply, core may choose one enabled matching supply deterministically for runtime routing, without changing Harness compatibility or Agent identity.
V241: Tests cover provider response parsing, env-ref authorization, pagination, timeout/failure cache preservation, manual persistence, stable-id merge, internal supply resolution, renderer refresh/manual entry, three-row identity, and execution inventory reuse.
V242: Settings exposes user Provider create/update/remove with display name, API protocol, Base URL, auth mode, and secret; save validates and persists the connection without querying its Models API or requiring manual ModelSupply input.
V243: Provider secret persistence encrypts with Electron `safeStorage`, writes only ciphertext to a mode-`0600` auth document, refuses writes when encryption is unavailable, and never returns secret values through IPC.
V244: Anthropic API-key mode sends `x-api-key` for discovery and injects `ANTHROPIC_API_KEY` at execution; auth-token mode sends `Authorization: Bearer` and injects `ANTHROPIC_AUTH_TOKEN`. OpenAI-compatible credentials remain bearer discovery/runtime credentials.
V245: Desktop settings persist Provider id/display name/API/Base URL/auth mode and a `local_keychain` secret reference only; no API key or auth token value appears in settings, cache, inventory, plans, errors, or UI responses.
V246: Catalog readiness, Provider refresh, and Agent execution resolve encrypted secrets in the main process and pass them to core only as request-scoped in-memory overrides; missing/unreadable secrets block with an actionable non-secret error.
V247: Removing a user Provider deletes its settings row, encrypted auth entry, and Provider-specific discovery cache while stable Models from manual, extension, or another Provider source remain.
V248: Provider management remains inside Settings; the Agent picker still has exactly Harness, Model, and Effort as primary rows, contains no Provider configuration control, and manual sends still omit Provider/ModelSupply.
V249: Tests cover encrypted-at-rest auth CRUD, unavailable-encryption refusal, settings redaction, API-key/auth-token headers and runtime env, Provider CRUD/refresh/removal, execution secret override, renderer form behavior, and three-row identity.
V250: Provider runtime readiness exposed through ExtensionInventory is explicitly schema-typed as optional boolean/string metadata so desktop readiness checks never depend on passthrough `unknown` values.
V251: Expanded sidebar renders a persistent `Anonymous user` trigger anchored below scrollable sessions; click/keyboard opens a Codex-style popover with exactly one `Settings` action, supports Escape/outside-click dismissal, and preserves the trigger at narrow supported desktop heights.
V252: `Settings` opens the dedicated Provider workspace; its add/edit/remove/model-refresh/usage-refresh interactions reuse safe IPC, place supported balance/quota data beside the corresponding connection, and never return secrets to renderer responses.
V253: The sidebar has no persistent `Check for updates` row. When and only when npm reports a newer stable `@swarmx/desktop`, the anonymous-user row shows a Codex-style circular update icon that expands to `Update` on hover/focus.
V254: Renderer and preload/main tests cover the one-action account menu, dismissal, Settings navigation, Provider form relocation, absence from Agent Picker, integrated Provider usage, and hidden/available/downloading/installing/restarting update states.
V255: Desktop checks the canonical npm latest endpoint at startup and on a bounded interval, compares stable semantic versions, ignores same/older/prerelease-invalid metadata, and keeps the update control hidden when unsupported, offline, or current.
V256: Clicking an available update publishes integer download progress in the same account-row control, then shows installing and restarting states without adding another sidebar row or changing the one-action account popover.
V257: Update download accepts only the canonical HTTPS npm tarball, verifies its declared SHA-512 integrity, installs into a versioned SwarmX update root through `npm install --ignore-scripts`, and verifies the installed `@swarmx/desktop` version before launch.
V258: A verified update relaunches the current Electron default-app executable with the new versioned app path and exits the old process; packaged or embedded hosts never receive an unsafe relaunch target.
V259: Tests cover semantic version gating, silent check failure, integrity rejection, progress publication, safe npm arguments, installed-version verification, IPC subscription, Codex hover/progress UI, retry behavior, and relaunch handoff.
V260: Provider Usage supports documented DeepSeek and regional Moonshot/Kimi balance responses, preserves all returned currencies as decimal strings, and renders total plus granted/voucher and topped-up/cash amounts beside the matching Provider.
V261: Provider Usage supports the official-client Kimi Code, Z.AI/GLM Coding Plan, and MiniMax Token Plan endpoints with body-level error checks and normalized 5-hour/weekly windows; MiniMax preserves model-specific rows, count-derived percentages, weekly boosts up to the official display ceiling, and unlimited status, while undocumented response drift fails closed per Provider.
V262: Codex Usage initializes the installed official app-server over stdio, reads `account/rateLimits/read`, maps 300-minute and 10,080-minute windows to 5-hour and weekly remaining percentages, and may show returned plan, credits, and reset credits.
V263: OpenAI API-key, Anthropic/Claude Code, Google Gemini, and OpenCode Go/Zen cards state why automatic quota lookup is unavailable instead of using API keys as subscription credentials or accessing private browser/OAuth state.
V264: Every automatically selected Provider Usage adapter chooses an exact allowlisted host and fixed endpoint; an explicitly selected New API adapter may use only the configured credential-free HTTPS origin plus `/api/usage/token/`. No adapter rewrites credentials across origins; all refuse embedded URL credentials and redirects, send the documented authorization form, stream no more than the bounded response size, and sanitize all failure output.
V265: Provider Usage accepts only a `local_keychain` reference whose key and Provider id both occur in the desktop-managed Provider set, excludes ambient Provider credentials from requests and the Codex subprocess environment, and returns only normalized status, meter, plan, reset, and display metadata through IPC.
V266: Tests cover DeepSeek multi-currency balance, Moonshot regional balance, Kimi Code relative resets, Z.AI regional routing/body status, current MiniMax model rows/boost/unlimited semantics, Codex window mapping and real subprocess framing, lookalike-host and credential-alias refusal, HTTP/body/stream failure isolation, secret redaction, mutation refresh, preload IPC, and Provider-adjacent renderer output.
V267: The account popover has no separate Usage action after Usage is integrated into Providers; Settings is the sole navigation action and opens the Provider workspace with usage refreshed there.
V268: OpenAI, Anthropic, DeepSeek, Ollama, and other connections appear as peer Providers only when explicitly configured through Settings or an extension; Provider identity never derives from ambient environment state.
V269: Tests verify the single Settings menu action, absence of Usage navigation, explicitly configured OpenAI/DeepSeek peer cards, and Settings/Extension source presentation without environment-generated cards.
V270: Refreshing or listing the desktop catalog with an empty Provider inventory and known ambient Provider variables performs zero Provider network requests and returns zero Provider connections, supplies, or discovered Models.
V271: The Provider form exposes a separate `Usage API` selection with `Automatic` and `New API`; the explicit choice persists as secret-free user Provider metadata, round-trips through preload/main inventory, and never changes the Provider's Model API protocol.
V272: `New API` usage sends the Provider's id-bound local credential as Bearer auth only to the configured HTTPS origin at `/api/usage/token/`, normalizes finite and unlimited API-key quota, and otherwise fails closed without exposing request or response secrets.
V273: Settings renders one responsive Provider matrix with Codex as an OpenAI Provider peer; every row reserves aligned 5-hour, 7-day, combined credit/balance, resets, updated, and actions positions, and unavailable values remain explicit instead of collapsing the grid.
V274: Credit and balance share one column. The primary total remains visible, while granted, topped-up, voucher, cash, used, raw quota, and additional-currency detail is available through a keyboard-focusable hover popup instead of duplicated inline text.
V275: Provider identity uses bundled real vendor assets selected by canonical origin/adapter: OpenAI for Codex/OpenAI, DeepSeek for `api.deepseek.com`, Packy for `packyapi.com`, and the New API mark as the fallback for other explicit New API Providers; runtime rendering performs no remote image fetch.
V276: Global usage refresh remains available, and every Provider row independently refreshes only its own usage source, retains other rows and the last successful row data while loading/failing, and exposes its own timestamp.
V277: One DeepSeek Provider may declare both `https://api.deepseek.com/anthropic` and `https://api.deepseek.com` entrypoints behind one secret reference. Anthropic is the default/preferred native route; an explicit `openai_chat` target uses the Chat Completions entrypoint directly without duplicating the Provider or invoking a protocol bridge.
V278: An explicit New API Provider may additionally store one encrypted account access token plus non-secret user id separately from its primary API token. Account refresh calls only the configured HTTPS origin's `/api/status`, `/api/user/self`, and bounded paginated `/api/token/`, returns masked token metadata and one account balance, never returns credentials, and never sums per-token quota into the account wallet.
V279: Tests cover matrix alignment/copy, Codex peer rendering, local vendor icon selection, accessible balance detail, targeted refresh merge, New API account secret isolation/pagination, and DeepSeek shared-secret dual-entrypoint preference/override.
V280: If Provider settings persistence fails after primary and account credentials change, rollback serially restores both encrypted entries so whole-document auth stores cannot lose one credential through concurrent rewrites.
V281: Provider credential controls keep stable explicit accessible names when conditional helper or security copy is rendered, so assistive input labels do not change with explanatory text.
V282: Desktop boot, Renderer remount, and Provider create/update read the last-successful Model catalog from the main-process disk cache without calling Provider Models APIs; Provider discovery runs only from explicit `Refresh Models`, and a failed refresh retains the usable cached catalog.
V283: The built-in SwarmX direct Harness accepts `anthropic`, `openai_responses`, and `openai_chat` Models, carries the resolved API protocol into request-scoped Agent configuration, and executes each protocol through its native Messages, Responses, or Chat Completions request shape with MCP tool continuation, reasoning output, streaming, and cancellation support.
V284: Provider execution uses a matching native Provider kind or declared `apiEntrypoints[targetApi]` directly. yallm bridging is used only when the selected ModelSupply explicitly requires bridging or auto routing has no native target; native-capable requests are never converted merely to normalize APIs.
V285: Tests cover SwarmX compatibility and explicit protocol propagation, native Anthropic and OpenAI Responses request/tool/stream behavior, native multi-entrypoint routing without yallm variables, and bridge fallback without changing Model or Agent identity.
V286: Catalog listing and refresh canonicalize persisted exact-origin DeepSeek Providers even when they predate multi-entrypoint support: Anthropic remains preferred, Chat remains available, and model discovery targets the official origin `/models` endpoint without requiring the user to re-save credentials.
V287: The disk catalog remains partitioned by Provider and stores each successful Provider's models plus route/group metadata. Startup, menu entry, and Provider create/update read it without model-network calls; only explicit model refresh replaces successful entries, and one Provider failure retains its own previous entry without blocking others.
V288: The Model menu renders `Provider -> optional group -> Model`, using New API/OpenAI-compatible `owned_by` as optional group metadata. Search spans Provider, group, label, id, and API; duplicate model ids from different Providers remain independently selectable while resolving to the same stable `harnessId:modelId` Agent identity.
V289: A routed Model selection sends the chosen internal ModelSupply id through trusted composition metadata. Core validates that supply against the selected model and uses its native Provider protocol preference; ordinary UI still has no fourth Provider or Supply primary control.
V290: Provider-discovered models receive deterministic human-readable labels for known vendor/name patterns, including `claude-fable-5` -> `Fable 5`, `gpt-5.6-sol` -> `GPT 5.6 Sol`, and `deepseek-v4-pro` -> `DeepSeek V4 Pro`; the canonical known label wins over punctuation-only Provider variants, while an explicit Provider label is preserved for unmatched ids.
V291: Codex catalog refresh uses the installed official `codex app-server` `model/list` method and preserves its display name plus supported/default reasoning efforts. It never reads `auth.json`, calls a refresh-token endpoint, or treats a Codex refresh token as an OpenAI API key; Codex-managed routes appear only for Harnesses that can execute them.
V292: Effort choices come from a verified built-in capability or Provider/app-server-advertised metadata and map through the selected native API. Unknown models without either source remain explicit rather than receiving fabricated effort support.
V293: Tests cover legacy DeepSeek migration, independent cache retention, Codex app-server framing/failure isolation, Provider/group hierarchy, duplicate routed selection, human labels/fallback, and advertised effort propagation.
V294: Expanded main sidebar contains no top-level Extensions entry and no Doctor/runtime status control above conversations. Settings navigation contains Providers, Extensions, Custom Agents, and Runtime, preserves Back to app/search behavior, keyboard focus, and supported narrow widths.
V295: Custom Agent CRUD persists a stable id, display name, reproducible Harness recipe/revision, Model id, optional routed ModelSupply id, effort, instructions, and enabled/read-only state; the Harness recipe contains Software, logical Skill bindings, MCP ids, project context, delivery, and permission policy, while read-only extension Agent profiles remain inspectable but cannot be overwritten.
V296: Custom Agent composition shows one Harness recipe builder containing Software, Skills, MCP servers, project context, delivery, and permissions, plus one separate Model choice. Saving a changed Harness recipe produces a new immutable recipe revision while the Custom Agent keeps its stable profile id and resolves identity from the active `harnessId:modelId` pair.
V297: A logical Skill has a stable namespaced id and one or more uniquely identified immutable variants. Legacy single-path Skills normalize to a default variant without breaking existing manifests or `skills: string[]` Agent profiles.
V298: Harness Skill bindings support `off`, `auto`, and `required` plus an optional pinned variant. Resolution order is pinned variant, exact Custom Agent profile, exact Model, Model family/capability, Harness Software, then default; a same-priority tie blocks instead of using last-writer-wins.
V299: Resolved Harness and Agent composition plans record logical Skill id, resolved variant id/version/digest, selection reason, delivery mode, context-token estimate, source plugin, and MCP delivery. `required` without a compatible variant or delivery blocks execution; `off` contributes nothing to the Harness revision.
V300: Harness software version, path, protection mode, readiness, and repair actions appear within Custom Agent configuration for the selected Harness. Settings > Runtime presents Node.js as the standalone baseline runtime, a separate inventory of Harness tools, and container-runtime requirements; Bun is neither a baseline dependency nor a requirement of Claude Code or Codex.
V301: `/doctor`, `/setup`, and on-demand repair APIs remain supported, but the complete Doctor status, diagnostics, and confirmed repair flow are built into Settings > Runtime instead of requiring a separate panel or permanent conversation navigation. Settings surfaces refresh after successful setup/fix and preserve explicit confirmation for install/admin actions.
V302: One main-process atomic Settings store owns Provider, Model, Custom Agent, Extension, source, active-revision, and binding persistence so concurrent feature services cannot overwrite unrelated sections; settings and IPC responses remain secret-safe.
V303: Extension action state distinguishes available, installed, enabled, running, update-available, blocked, diverged, conflict, and pinned. The UI never presents declaration, installation, enablement, or execution as equivalent.
V304: Upstream revisions record canonical source, version/ref, content digest, optional signature, and provenance. Updates stage and validate before atomic activation, preserve the previous active revision on failure, never overwrite local overlays, and support rollback.
V305: Skill evolution produces immutable candidate and evaluation records bound to target Agent, Model fingerprint, baseline revision, dataset digest, optimizer id/version/config digest, quality/safety/failure/latency/context-cost metrics, and a promotion verdict. Candidates may be generated and evaluated automatically but cannot silently replace upstream or active revisions.
V306: Promotion requires an explicit policy or human gate after target-model held-out evaluation; rejection, quarantine, canary, and rollback preserve lineage. Active runs use a resolved immutable snapshot and are unaffected by later updates or promotions.
V307: Tests cover legacy Skill migration, deterministic variant selection/conflict, secret rejection, Custom Agent CRUD, atomic cross-section persistence, Extension action validation/rollback, Settings navigation, Harness/Runtime separation, Doctor-control removal, responsive layout, and unchanged Provider/Agent-picker behavior.
V308: Completion tests derive valid cursor positions from their document fixtures and assert suggestion-surface state within the suggestion surface, so document text cannot be mistaken for a completion item and bounds validation is exercised intentionally.
V309: Built-in Claude Code and Codex ACP launchers use the Node.js ecosystem (`npx`) on the host and in protected containers; Runtime detection and setup never install or require Bun for either Harness.
V310: Runtime lists SwarmX, Claude Code, Codex, OpenCode, Hermes, and OpenClaw as Doctor-style Harness rows with tool identity/path on the left and a semver button or explicit Install action on the right. Activating a version button reruns only that Harness version check.
V311: Node.js, Harness CLI, and container version detection stores and renders only the semantic-version token from version output; banners, product names, build metadata outside the token, and unrelated output never appear as the displayed version.
V312: Runtime's embedded Doctor shares the existing read-only inspect and explicitly confirmed fix APIs, shows health and actionable diagnostics inline, and refreshes both environment and Doctor state after setup or repair.
V313: Tests cover Node.js-only baseline runtime presentation, Node/npm-based ACP launchers, Claude Code and Codex tool rows, Hermes/banner semver normalization, per-Harness version refresh, embedded Doctor diagnostics, and the absence of an Open Doctor detour.
V314: Agent definition contracts identify `claude_code` Markdown and `codex` TOML formats while preserving source host, scope, path, read-only state, and native unknown fields without changing existing Claude-compatible APIs.
V315: Codex TOML parsing requires `name`, `description`, and `developer_instructions`; maps `model`, `model_reasoning_effort`, `sandbox_mode`, `nickname_candidates`, `mcp_servers`, and `skills.config` into normalized profile metadata; and preserves other supported config keys as inert native metadata.
V316: Claude Code and Codex projections preserve their native prompt/config shape and unknown safe fields, omit SwarmX/GEEPilot-only metadata, and round-trip known fields without converting host-specific policy semantics into another host's fields.
V317: Desktop native Agent discovery reads only bounded `.claude/agents/*.md` and `.codex/agents/*.toml` locations, isolates malformed-file failures as warnings, namespaces cross-host ids, and applies project-over-user precedence per host and Agent name.
V318: Discovered native profiles are read-only, use their native Harness id, preserve omitted/`inherit` Model state as unresolved, and cannot become execution-ready until the normal Harness x Model composition preflight succeeds.
V319: Desktop inventory and Custom Agents Settings present native definitions separately from persisted Custom Agents and Extension profiles; listing never writes settings or native files and never activates hooks, MCP servers, skills, or Agent sessions.
V320: Dual-format definitions, normalized profiles, discovery warnings, and projections reject inline secret-looking fields while allowing explicit secret references as inert metadata.
V321: Settings > Custom Agents renders Model choices as `Provider -> optional group -> Model`, uses the same descending GPT version and `sol > terra > luna` order plus `mythos > fable > opus > sonnet > haiku` then descending Claude version order as the Composer, filters by the selected Harness Software, and saves the selected route's ModelSupply id while keeping Provider/Supply out of Agent identity.
V322: Project registration persists and compares canonical filesystem realpaths, so macOS `/var` and `/private/var` aliases or symlinked roots cannot create duplicate project identities; project tests assert the canonical path rather than the caller's unresolved spelling.
V323: Tests that exercise the user-level Project registry must remove every registered fixture in `finally` cleanup, including when an assertion fails, so validation cannot leave synthetic Projects in the user's Desktop state.
V324: Project registry tests scope ordering and membership assertions to the fixture ids they created and never assume the user's persistent registry is otherwise empty.
V325: Renderer assets copied from Vite's public directory use document-relative URLs, so development HTTP origins and packaged `file:` renderer origins resolve the same icon files without root-filesystem requests.
V326: The Projects heading overflow is a collection-level organization menu only, with `By project`/`In one list` and `Priority`/`Last updated`/`Manual order` radio choices; it never dispatches actions against the active Project.
V327: Each rendered Project row owns its own overflow target and exposes `Pin project`, `Reveal in Finder`, `Rename project`, `Archive tasks`, and `Remove` for that exact Project, independently of the Projects heading menu.
V328: Renderer tests that need a Project row wait for asynchronous Project discovery before interacting with its row-level controls instead of assuming the SWR response committed synchronously.
V329: Interactive Project hover details render as a labeled native nonmodal `dialog`, preserving keyboard-reachable pin control semantics without remapping a generic container to the dialog role.
V330: Renderer tests submit text-field workflows with deterministic value-change and form-submit events when per-character timing is not the behavior under test, so suite load cannot drop characters or create false product failures.
V331: Desktop Preload reads the validated persisted Project registry before Renderer mount, exposes it as an immutable bootstrap snapshot, and the Project sidebar renders that snapshot on its first frame without a startup `project:list` request or `Loading projects` placeholder; later Project mutations keep using isolated IPC and cache updates.
V332: A session-controlled Harness appears in the executable Harness x Model matrix only when the Model declares a fixed runtime id for that Harness adapter or a selected ModelSupply explicitly names that adapter and provides its runtime model id; API compatibility or `modelCompatibility: any` alone never fabricates an executable cell.
V333: A Custom Agent Harness preserves its stable custom Harness identity while carrying the built-in Software adapter id separately; matrix routing, request-scoped model bootstrap, runtime aliases, and protected-container selection use the adapter id rather than silently treating the custom id as a new adapter.
V334: Claude Code catalog supplies name that Harness only for runtime model ids proven to exist in the pinned ACP session configuration; the fixed DeepSeek V4 Pro route advertises `deepseek-v4-pro[1m]` through the same stable session model option that ACP selects, while an Anthropic-compatible catalog entry alone never creates a Claude Code cell.
V335: `swarmx send --harness <id> --model <runtime-id>` serializes the selected backend and model into the executed Swarm node, adds request-scoped harness bootstrap environment, and rejects unknown or unsupported Harnesses instead of constructing and discarding an Agent.
V336: Protected Codex ACP reuses only a private official Codex `auth.json` through an exact read-only container file volume, honors `CODEX_HOME`, and continues to support `CODEX_API_KEY`/`OPENAI_API_KEY`; it never forwards `CODEX_ACCESS_TOKEN`, which the pinned adapter does not consume.
V337: Codex App Server catalog Models remain available to direct SwarmX execution, but a catalog supply names the pinned Codex ACP Harness only when its runtime model id is also present in that adapter's proven session configuration; catalog/adapter drift never fabricates a Codex cell.
V338: Direct SwarmX execution in `codex_responses` mode always uses and fully consumes a streaming Responses request, even when its caller did not supply a chunk callback, because the ChatGPT Codex subscription endpoint rejects unary Responses requests.
V339: A direct SwarmX composition with a validated Project `cwd` receives the active Project identity plus `workspace_list_directory` and `workspace_read_file` tools; repository questions inspect relevant contained files instead of claiming that no Project context exists.
V340: Host-injected workspace tools reuse the same root containment, traversal, symlink, UTF-8, file-size, and directory-entry limits as the desktop Workspace surface, and no tool is injected when the task has no validated `cwd`.
V341: Every completed desktop request decorates its returned work/final chunks with one request start, end, and elapsed duration, persists that metadata across reload, and renders `Worked for <duration>` whenever work details exist.
V342: Expanded Worked content uses compact semantic reasoning/tool rows, avoids raw normalized Agent identifiers as the primary label, and never animates a completed reasoning indicator.
V343: The first successful user request in a placeholder-titled local Session asks `gpt-5.4-mini` for a concise title, sanitizes it to one nonempty line of at most 60 characters, persists it, and leaves the primary response intact on title failure.
V344: Double-clicking a writable local task or choosing Rename opens a centered rename dialog with selected current text; Save persists the trimmed title, while Cancel/Escape leaves it unchanged and a manual title is never replaced by automatic naming.
V345: Right-clicking a local sidebar task opens a keyboard-accessible menu with Pin/Unpin, Rename, and Delete; pinned tasks persist and sort before unpinned siblings, Delete requires confirmation, and ACP tasks expose no unsupported mutation actions.
V346: A direct SwarmX composition with a validated Project `cwd` receives `workspace_list_directory`, `workspace_read_file`, `workspace_write_file`, `workspace_edit_file`, and `workspace_shell`; a task without a validated `cwd`, or an ACP Claude Code/Codex task, receives none of these host tools.
V347: A complete `workspace_read_file` result includes a content digest. Creating a new contained UTF-8 file succeeds atomically, while replacing an existing file without reading it first or after an external modification fails with an actionable stale-content error.
V348: `workspace_edit_file` replaces one unique exact match by default, supports an explicit replace-all mode, rejects missing or ambiguous matches, and shares write size, root containment, symlink, and optimistic-concurrency protections.
V349: `workspace_shell` executes from the canonical Project root, returns exit code, stdout, stderr, duration, timeout, and truncation metadata, enforces command/time/output limits, and terminates the process group on timeout or request cancellation.
V350: On macOS, the Shell sandbox permits Project writes needed by builds and tests but denies network access, writes through an escaping symlink, writes elsewhere on the host, and access to Provider secret environment variables; a missing or unsupported sandbox fails closed.
V351: Focused tests cover tool registration, Project prompts, file creation/replacement/edit races, traversal and symlink escapes, Shell cwd/output/timeout/cancellation/environment isolation, and direct-SwarmX-only runtime injection; desktop tests and builds remain green.
V352: Expanding Worked renders each Thought/Reasoning message directly as Markdown body content with no `run-event__card`, header/meta row, border, background, or inset padding, while structured tool events remain distinguishable.
V353: A running desktop turn starts with Worked expanded, merges request-scoped streaming Thought and assistant-text deltas in order, renders commentary plus tool calls/results inside the open work region, and automatically collapses all work when the terminal result or error arrives while keeping the final assistant/system summary visible.
V354: Streaming OpenAI Responses reconstructs missing completed output items from `response.output_item.done`, executes every recovered function call, emits its tool call/result, submits the matching `function_call_output`, and continues until a final assistant message instead of accepting a reasoning-only terminal result.
V355: Compact Thought rendering removes one full-message `**...**` wrapper commonly emitted by reasoning summaries without suppressing ordinary inline Markdown, and a successful direct desktop Agent composition without a final assistant message is surfaced as an explicit failure rather than silently ending.
V356: A Project-bound direct composition selects `claude_code` when its resolved Model id names Claude, Sonnet, Opus, Haiku, or Fable and `codex` otherwise; it exposes only that profile and injects no host tools into ACP Harnesses.
V357: `claude_code` exposes tolerant `Bash`, `Read`, `Edit`, `Write`, `Glob`, and `Grep` definitions with Claude Code argument names; `codex` exposes tolerant `exec_command` and `apply_patch` definitions with Codex argument names. Both dispatch to the existing bounded Project operations.
V358: OpenAI Responses transports Codex `apply_patch` as a freeform custom tool and continues `custom_tool_call` with matching `custom_tool_call_output`; APIs limited to JSON function tools receive a `patch` string fallback without changing the handler semantics.
V359: Claude/Codex-compatible paths accept and ignore extra optional fields that do not change authority, while missing required values, unsupported operations, traversal, escaping symlinks, stale files, and unsafe Shell requests still fail with actionable tool errors.
V360: Focused tests pin profile selection, exact public tool names and principal argument keys, freeform Responses continuation, extra-argument tolerance, bounded file/search/Shell behavior, and direct-SwarmX-only injection; docs cite primary vendor sources and distinguish hosted web tools from local client tools.
V361: Native protocol test doubles implement the same `toolsForNative()` union surface as `McpManager`; regression tests cover both JSON function and freeform text tools so Responses/Anthropic tests cannot silently retain the older function-only `toolsForOpenai()` contract.
V362: ACP integration tests that start multiple subprocess sessions declare an explicit bounded timeout above Vitest's 5-second unit default, so full-suite process contention does not create a false protocol failure.
V363: Every local or MCP call resolves to model-facing `content`, optional client-facing `structuredContent`, and error state; Anthropic, Responses, and Chat continuations send only `content`, while tool-result chunks retain sanitized structured content for clients.
V364: Claude `Read`, `Edit`, `Write`, `Glob`, `Grep`, and `Bash` results expose the corresponding Agent SDK field shapes; model-facing text stays compact and native-like without serializing the client envelope back into the prompt.
V365: Claude `Bash` accepts `run_in_background`, returns `backgroundTaskId`, and pairs with tolerant `TaskOutput` and `TaskStop` tools for bounded wait, output inspection, and process-group termination.
V366: Codex `exec_command` yields a live numeric `session_id` when unfinished, and `write_stdin` writes or polls that session. Both expose `chunk_id`, `wall_time_seconds`, optional `exit_code`/`session_id`/`original_token_count`, and `output`, while the model sees Codex-style formatted text.
V367: Foreground cancellation rejects the call; background cancellation or tool-manager close stops the process group. Timeout marks the result, completed sessions remain readable only for bounded retention, and all session cleanup removes temporary directories.
V368: Focused tests cover dual result transport, Claude structured file/search/Shell outputs, background start/wait/stop, Codex yield/poll/stdin/final output, timeout/cancellation/close cleanup, and real PTY interaction.
V369: Native protocol execution normalizes both the dual tool-result contract and legacy plain-object tool results from compatible MCP adapters/test doubles; legacy values remain JSON model text plus structured client output.
V370: Background process integration tests use explicit process-runtime and `TaskOutput` wait bounds above full desktop-suite scheduling contention, so a valid `running` poll is not mistaken for command failure.
V371: `exec_command({ tty: true })` exposes a true terminal (`test -t 0` succeeds) with `TERM=xterm-256color`; omitted/false `tty` preserves `TERM=dumb` pipe behavior.
V372: PTY `write_stdin` forwards text and control bytes such as Ctrl-C, reports one merged incremental `output` stream with empty structured `stderr`, and preserves timeout, cancellation, stop, close, and retention semantics.
V373: PTY Seatbelt grants `file-ioctl` only for terminal device paths needed by canonical input and `stty`; it adds no generic device, filesystem, or network permission.
V374: A checked compatibility inventory names all 42 tools from the public Claude Code reference and distinguishes implemented, partial, provider/configuration-dependent, and not-yet-implemented behavior without counting placeholder tools as support.
V375: `TaskCreate`, `TaskGet`, `TaskList`, and `TaskUpdate` accept Agent SDK 0.3.211 field names, share request-scoped state, preserve dependency links, support metadata merge/delete and task deletion, and return the documented structured output shapes.
V376: `TodoWrite` atomically replaces a validated request-scoped Todo list and returns exact `oldTodos`/`newTodos`; `ReportFindings` validates at most 32 repo-relative findings and returns `count`, optional `level`, and the findings unchanged.
V377: `NotebookEdit` supports replace, insert, and delete by notebook cell id, preserves valid notebook structure, reports Agent SDK-compatible original/updated file fields, and rejects missing cells, invalid modes/types, escaping paths, stale files, and malformed notebooks.
V378: Focused tests verify the first parity batch's exposed names, tolerant schemas, task/Todo state transitions, finding validation, notebook mutations, structured output, and Project guards.
V379: Every non-throwing Claude tool failure has definite non-empty model-facing text, `isError: true`, and a structured error string; optional structured fields never leak `undefined` into the model-result boundary.
V380: `ListMcpResourcesTool` is absent without connected MCP servers; when enabled it accepts optional `server`, follows paginated `resources/list`, and returns Agent SDK-compatible URI/name/MIME/description/server records.
V381: `ReadMcpResourceTool` requires `server` and `uri`, reads only from that connected server with the active cancellation signal, returns Agent SDK-compatible text contents, and reports binary content as an explicit unsupported error instead of inventing `blobSavedTo`.
V382: Claude `Glob` includes hidden and gitignored files except `.git`, sorts newest modification time first with deterministic path ties, returns at most 100 paths, and distinguishes result truncation from whether `totalMatches` is exact.
V383: Claude foreground `Bash` waits for its requested/default foreground timeout, then leaves a still-running non-`sleep` command in the same managed background session and returns its task ID; leading `sleep` commands retain terminal timeout/termination, and cancellation still rejects the call.
V384: When at least one selected Skill has a real path, Claude receives `Skill({ skill, args? })`; invocation reads at most 512 KiB from that configured Skill only, strips frontmatter, expands full/indexed/named arguments plus Skill directory/effort/session variables, appends unused arguments, and returns expanded instructions as model-facing text.
V385: `TaskUpdate` validates every requested field, status, metadata object, and dependency target before mutating any task or reciprocal link; a rejected update leaves request-scoped task state unchanged.
V386: `AskUserQuestion` is conditional on an interaction bridge, validates Agent SDK 0.3.211 question shape, waits without default timeout, and returns question-keyed string answers; UI supports single-select, multi-select, and automatic free-text Other.
V387: Interactive responses resolve only the matching request, renderer owner, interaction id, and kind. Task cancellation, renderer destruction, and broker close reject pending calls once and remove listeners/state.
V388: `EnterPlanMode` creates a private bounded plan file and enables request-scoped read-only enforcement. `Bash`, `Edit`, `NotebookEdit`, and ordinary `Write` calls fail while active; exact plan-file `Read`/`Write` remains available.
V389: `ExitPlanMode` reads the real plan file, blocks on explicit UI approval with no idle timeout, returns Agent SDK-compatible plan/file output after approval, and stays in plan mode with user feedback after rejection.
V390: Every standalone renderer component test that uses Testing Library declares the `jsdom` Vitest environment, so DOM behavior runs under the same browser-like contract as the main renderer suite.
V391: Renderer tests use Vitest built-in assertions for DOM state unless repository setup explicitly installs extra DOM matchers; standalone tests do not assume `@testing-library/jest-dom` globals.
V392: Real PTY integration tests treat session creation and readiness output as separate asynchronous events, then poll the live session with a bounded wait before sending dependent input.
V393: Pipe-backed stdin integration tests treat input delivery and child termination as separate asynchronous events, then poll the same live session with a bounded wait before asserting terminal output.
V394: Multi-step renderer integration tests whose normal full-suite runtime can cross Vitest's 5-second unit default declare an explicit bounded timeout, so scheduler contention does not create false UI failures.
V395: Desktop IPC annotates composed workspace-tool options with the exported boundary type, preserving interaction callback input types under strict declaration builds.
V396: Configured MCP servers begin connecting concurrently without delaying the first model request; manager close waits for and closes both pending and connected transports without leaking a rejected background promise.
V397: `ToolSearch` matches Claude Code 2.1.187 input/output fields, supports exact comma-separated `select:` and ranked keyword lookup with a validated result cap, reports pending MCP names when no match exists, activates only matched deferred schemas, and makes them visible on the immediately following Chat, Responses, or Anthropic model step.
V398: Connected MCP tools are projected and dispatched as `mcp__<server>__<tool>` so equal upstream names cannot collide and a call never falls through to an unrelated server.
V399: `WaitForMcpServers` accepts optional server names, waits at most five seconds for matching pending connections, and returns exact `ready`, `connected`, `failed`, `stillPending`, `needsAuth`, `disabled`, and `unknown` arrays; successful connections become directly visible on the immediately following model step.
V400: Focused MCP and native-model tests cover deferred visibility, exact selection, keyword result limits, pending/failed/unknown waits, namespaced dispatch, concurrent startup/close, and per-step tool-list refresh.
V401: New MCP discovery implementation and regression tests conform to the repository's Biome formatting contract before full validation.
V402: Claude `LSP` uses the 2.1.187 operation enum and exact `filePath`, one-based `line`, one-based `character`, and optional `query` input names; its structured output contains `operation`, formatted `result`, `filePath`, and applicable `resultCount`/`fileCount`.
V403: LSP operation routing canonicalizes a Project-contained regular UTF-8 file, rejects traversal/symlink escape and files above the bounded read limit, infers its language, and selects exactly one matching command-backed inventory server without accepting a model-supplied command or server id.
V404: Definition, reference, hover, document/workspace symbol, implementation, and call-hierarchy operations reuse the persistent initialized JSON-RPC session, synchronize the real file, use bounded requests, and propagate active request cancellation without leaving a pending request entry.
V405: Direct Claude desktop compositions expose `LSP` only when the inventory has a usable command-backed server; Codex, ACP Harnesses, and inventories containing only SwarmX mention-completion pseudo-servers receive no duplicate or nonfunctional tool.
V406: Focused tests exercise exact LSP schema/output, automatic file-language routing, location formatting/counts, two-step incoming/outgoing hierarchy, missing/ambiguous server errors, Project escape rejection, cancellation cleanup, and conditional Claude-only registration.
V407: New LSP host/tool code and tests conform to Biome's canonical wrapping and import order before full validation.
V408: Claude `EnterWorktree` uses the 2.1.187 optional `name` input and returns exact `worktreePath`, optional `worktreeBranch`, and `message` fields; `ExitWorktree` requires `action: "keep" | "remove"`, accepts optional `discard_changes`, and returns the documented action, original/worktree paths, optional branch, discard counts, and message fields.
V409: Worktree entry requires a real Git repository, validates a traversal-free portable name, confines the destination to canonical `.claude/worktrees/<name>`, creates branch `worktree-<name>` from `HEAD`, and resumes only an existing Git worktree registered to the same repository.
V410: Entering and exiting clear file read digests and synchronously rebind the same request's `Read`, `Edit`, `Write`, `Glob`, `Grep`, `NotebookEdit`, `Bash`, and conditional `LSP` operations between the worktree and original Project roots; linked-worktree Git metadata is the only additional Shell write path.
V411: `ExitWorktree action: remove` counts uncommitted status entries and commits after the entry baseline, refuses destructive removal or unverifiable state without `discard_changes: true`, stops live Shell sessions rooted in the worktree, removes the registered worktree and its generated branch, then restores the original root. `keep` restores the root without removing files or branch.
V412: Calling `ExitWorktree` without a current request-owned worktree is a no-op error, a second entry while active is rejected, and tool-manager cleanup closes processes while preserving an unexited worktree on disk.
V413: Focused tests cover exact schemas and structured outputs, generated and named entry, dynamic file/Shell root behavior, duplicate/no-op errors, keep, guarded remove, explicit discard, branch cleanup, containment, and manager cleanup preservation.
V414: New worktree implementation and regression tests conform to the repository's Biome formatting contract before full validation.
V415: Worktree tests compare model-facing canonical worktree/original paths separately from the restored caller-supplied workspace binding, so macOS `/var` and `/private/var` aliases do not create false lifecycle failures.
V416: Conditional Claude `Agent` uses the 2.1.187 `description`, `prompt`, optional `subagent_type`, optional `model`, and optional `resume` field names from the upstream background-disabled schema and returns `status: completed`, `prompt`, `agentId`, text content blocks, tool-use/duration/token totals, and the upstream usage field names.
V417: A direct desktop Claude child call executes the same resolved Agent composition with a fresh workspace tool manager at the parent's current dynamic root, carries selected Skills and conditional LSP, inherits request cancellation, and does not receive another `Agent` bridge or interactive user bridge.
V418: Request-scoped child history has a generated stable agent id, accepts `resume` only for an id created by the current parent request, appends the new prompt and prior assistant result, refreshes the system Project context to the dynamic root used by the resumed execution, and is discarded when the parent request ends.
V419: The synchronous implementation rejects unavailable model-family overrides, specialized agent types other than `general-purpose`, background execution, team naming/mode, explicit child cwd, and isolation rather than claiming those upstream semantics occurred.
V420: Focused tests cover conditional registration, exact synchronous schema/output, extra-argument tolerance, explicit unsupported-mode errors, child result/usage mapping, dynamic-root projection, request-scoped resume, unknown resume rejection, and child non-recursion.
V421: New child-Agent bridge code and tests conform to the repository's Biome formatting contract before full validation.
V422: Claude tool output tests compare the model-visible `content` and `structuredContent` fields independently from SwarmX's internal symbol marker, which is transport metadata rather than part of the upstream output contract.
V423: The parity matrix records the exact remaining runtime dependencies and updated direct-desktop/max exposure counts without classifying a schema-only or polling substitute as tool support.
V424: One direct desktop Claude session reuses a sandboxed Shell and automatic-activation queue across request tool managers; foreground turns wait for prior activations, queued events wait for foreground completion, activations execute in order, root replacement closes old processes/jobs, and registry disposal releases every timer/process.
V425: Conditional `Monitor` uses the 2.1.187 strict `command`, `description`, optional `timeout_ms` (default 300000, range 1000..3600000 unless persistent), and optional `persistent` fields and returns exact `taskId`, `timeoutMs`, and `persistent` fields using the same task id accepted by `TaskStop`.
V426: Monitor stdout is line-buffered, coalesced for at most 200ms, bounded per line/event, token-bucket rate limited, marked as untrusted process output, and enqueued as an automatic session activation; timeout stops the process, persistent mode lasts until TaskStop/session close, and trailing output flushes on exit.
V427: Conditional `CronCreate` uses strict `cron`, `prompt`, optional `recurring` (default true), and optional `durable` (default false), returns `id`, `humanSchedule`, `recurring`, and `durable`; `CronDelete` uses strict `{id}` and returns `{id}`; `CronList` uses strict `{}` and returns exact job fields.
V428: Cron parsing supports wildcard, list, range, and step syntax in minute/hour/day-of-month/month/day-of-week local-time fields, including Sunday 0/7 and Vixie day-of-month/day-of-week OR behavior. Creation rejects malformed/no-next-year expressions and >50 combined jobs; one-shot jobs delete after firing, recurring session jobs re-arm and expire after three days, deletion cancels timers, and durable jobs follow the persistence and ownership contract in V432-V437.
V429: Main-process successful foreground and automatic turns append messages atomically to the persisted session before publishing. Renderer success/background paths reload the authoritative session through removable preload transport instead of overwriting concurrent messages with stale snapshots.
V430: Focused tests cover exact Monitor/Cron schemas and outputs, persistent borrowed-Shell cleanup, monitor line/event/timeout/TaskStop behavior, cron parsing/next-fire/delete/list/expiry/limits, foreground-event serialization, session deletion/shutdown, preload forwarding, and Renderer authoritative refresh.
V431: New session-runtime, scheduler, monitor, transport, and regression code conforms to repository Biome formatting before full validation.
V432: A durable Cron create writes the Claude Code-compatible `{ tasks: [...] }` document to `<Project>/.claude/scheduled_tasks.json` using an atomic private-file replacement. Each task has a raw eight-character UUID id plus `cron`, `prompt`, `createdAt`, optional true-only `recurring`, optional `lastFiredAt`/`permanent`, and creator session/process identity; the input-only `durable` flag is never persisted.
V433: Scheduled-task reads treat a missing or invalid document as empty, skip malformed or invalid-cron entries, preserve the supported optional fields, serialize concurrent in-process mutations, and never expose or execute an entry before its required persistence mutation succeeds.
V434: Project schedulers coordinate through the Claude Code-compatible `.claude/scheduled_tasks.lock` identity document. A live matching process owns the lock, stale or PID-reused ownership is recovered, acquisition retries while sessions are active, only the owning identity may release it, and Project runtimes observe externally written task-file changes.
V435: A session may schedule its own durable jobs without the global lock; the lock owner additionally schedules legacy and orphaned jobs but never duplicates work owned by another live session. Closing or replacing a runtime cancels local timers and releases owned locks without deleting durable tasks.
V436: Future durable one-shots re-arm on startup and delete durably before activation. Missed durable one-shots are deleted and produce a confirmation-required activation that does not contain an instruction to execute automatically. Durable recurring jobs recover from `lastFiredAt ?? createdAt`, fire at most once for elapsed time, persist `lastFiredAt` before activation, re-arm their next local-time occurrence, and use Claude Code's seven-day lifetime unless the stored internal `permanent` flag is true.
V437: `CronList` combines every valid Project-durable task with the current session's ephemeral jobs, `CronDelete` removes either kind, and the 50-job limit and collision check cover both sets. Session-only behavior remains unchanged, while durable operations fail actionably when storage or locking cannot preserve their contract.
V438: Focused persistence/runtime tests cover exact JSON shape and permissions, malformed-file filtering, atomic concurrent mutation, live/stale locks, durable create/list/delete, close/reopen recovery, missed one-shot confirmation, recurring `lastFiredAt`, external refresh, and multi-session duplicate suppression.
V439: Durable Cron implementation, tests, compatibility matrix, and user documentation conform to the repository's Biome formatting contract and pass targeted plus full validation.
V440: The parity inventory labels `PowerShell`, `SendMessage`, and `Workflow` as TODO, keeps their exact blockers visible, and gives each one a distinct pending §T task without exposing a placeholder tool.
V441: The root manifest and all six publishable packages declare 3.1.1; packed workspace dependencies resolve to the same released version, and every tarball contains its declared runtime/type/assets without generated or secret-bearing residue.
V442: The 3.1.1 commit is released only after full lint, tests, builds, package dry-runs, and diff checks pass; the verified commit is pushed to `main`, npm packages publish dependency-first, then the matching tag/GitHub release and registry metadata are verified against that same commit/version.
V443: Harness permission mode accepts only `default`, `plan`, `restricted`, or `trusted`; policy preserves exact `allowedTools`/`deniedTools`, composition plans carry those lists, and unknown direct-runtime policy blocks execution instead of degrading to trusted behavior.
V444: Direct tool decisions are deterministic and deny-first: explicit deny always blocks; plan allows read-only tools only; explicit allow may pre-approve non-plan calls; read-only calls auto-allow; default asks for remaining calls; auto-review allows Project-contained writes but asks for execute/control calls; restricted denies remaining calls; trusted allows remaining calls only inside unchanged Project sandbox/path/output/cancellation limits.
V445: Every direct `ask` decision waits for one request-scoped desktop choice between allow-once and reject-once. Missing bridge, invalid/mismatched option, cancellation, close, or rejection blocks the call; approval runs exactly that original call and creates no durable rule.
V446: `AcpClient` cancels permission requests when no handler exists; a desktop handler projects exact ACP option ids/kinds and bounded display names into the same request-scoped approval dialog and returns only an explicitly selected offered option id to the requesting ACP session.
V447: Permission prompt payloads expose bounded tool title/kind and safe operation summary only; they omit write content, patch text, ACP raw input/output, secrets, and credentials. Permission policy never makes `dangerouslyDisableSandbox` or `sandbox_permissions=require_escalated` effective.
V448: Focused Core/Desktop/Renderer tests cover schema rejection, deny/plan/allow/default/restricted/trusted precedence, no-bridge failure, approve/reject execution, ACP cancel/selection, broker ownership/cancellation, policy persistence/UI, and unchanged sandbox-escalation rejection; docs cite official Claude Code and Codex sources.
V449: Layer schemas identify `managed`, `project`, `personal`, `agent`, and `session` sources plus optional mode ceiling and exact allow/deny tools. Merge is deterministic: mode rank `plan < restricted < default < auto < trusted`, all denies union, Agent/personal/managed allows union minus denies, and Project allows reject validation.
V450: Desktop loads personal policy from atomic Settings, managed policy from `SWARMX_MANAGED_PERMISSION_POLICY`, and Project restriction from canonical `<Project>/.swarmx/permissions.json` with bounded size. Missing layers inherit; malformed configured layers return visible errors and block direct SwarmX execution.
V451: Permission status IPC returns personal policy, managed/Project layer summaries, effective preview for selected Project/Agent inputs, immutable-source flags, and no raw environment/file content; save mutates personal policy only.
V452: Every resolved direct or ACP approval choice appends one sanitized receipt with stable id/time/source/tool kind/decision/policy source ids; history truncates to newest 200 and never becomes an allow rule or contains request payload content.
V453: Settings navigation contains dedicated `Permissions` workspace with hierarchy cards, sandbox reassurance, personal mode selector, structured exact-tool allow/deny editor, current Project policy state, and recent decisions; missing managed/Project policy reads neutral rather than broken.
V454: Custom Agent editor replaces newline allow/deny textareas with keyboard-usable structured exact-tool rows/chips, prevents duplicates/conflicts before save, describes each mode in plain language, and states higher layers may only reduce effective authority.
V455: Tool approval dialog shows action/risk/source hierarchy, bounded safe facts, `One call only` and `Project sandbox stays on` reassurance, safe-default reject focus, offered ACP options, and responsive controls without exposing prohibited payload fields.
V456: Core/Main/Preload/Renderer tests cover layer precedence, Project restriction-only validation, malformed fail-closed behavior, personal persistence, receipt redaction/truncation, navigation/editor/dialog accessibility, and unchanged Project sandbox enforcement; desktop screenshots verify full and scrolled Settings states.
V457: `SessionDataSchema` persists `permissionMode` with backward-compatible `inherit`; session create/save/load round trips preserve `inherit`, `default`, `auto`, `plan`, and `trusted` and reject unsupported values.
V458: Explicit conversation `default`, `auto`, `plan`, or `trusted` replaces personal and Agent mode defaults for that conversation, but cannot exceed managed/Project mode ceilings or bypass any explicit deny; `inherit` preserves the existing layered result.
V459: Direct SwarmX execution resolves the permission mode from the authoritative persisted session identified by the request. Missing/invalid session state fails safely, while ACP Harness execution keeps its native permission semantics.
V460: Settings opens on `General`, whose first card presents independent Default permissions, Auto-review, and Full access switches with Codex-aligned row geometry and concise risk copy. The separate `Advanced permissions` workspace retains fallback/conservative modes, hierarchy, exact allow/deny, sandbox, Project state, and receipts.
V461: Every direct SwarmX conversation Composer shows a compact keyboard-usable permission trigger with `Use default`, `Ask for approval`, `Approve for me`, `Full access`, and the secondary `Plan only` profile; unavailable General profiles are not selectable, the current choice is marked, existing-session changes persist immediately, and new-session creation receives the selection.
V462: Focused Core/Main/Preload/Renderer tests cover migration, profile persistence, Auto-review decisions, disabled-profile safe degradation, override precedence, General rule preservation, Composer selection, new-session payloads, existing-session saves, Escape/outside dismissal, and ACP scope copy; rendered desktop QA compares General and conversation states with the supplied Codex reference at desktop and narrow widths.
V463: Desktop permission settings migrate with all three General profiles enabled, persist each switch independently, expose only the booleans through renderer-safe status/save IPC, and preserve personal policy plus sanitized receipts on profile updates.
V464: Main resolution rewrites a disabled explicit or inherited `default`, `auto`, or `trusted` declaration to a session-scoped `plan` ceiling before layer merge; malformed availability data fails schema validation and no Renderer-only state can widen authority.
V465: The General permission card uses three divider-separated rows, right-aligned native-looking purple switches, Codex-like typography/spacing/border treatment, no invented selection checkmarks, and responsive stacking without clipping; the Composer menu removes the large title/footer chrome and stays legible at narrow width.
V466: Visual QA records reference-versus-implementation comparison, desktop and narrow screenshots, interactive switch/menu checks, and any source-capture limitation without treating a screenshot alone as proof.
V467: Running npm-installed `swarmx` with no arguments resolves the installed `@swarmx/desktop` app and Electron binary, launches Desktop, and does not import CLI; `swarmx desktop` is equivalent, while any other argument delegates unchanged to `@swarmx/cli`.
V468: Published `swarmx` depends on matching `@swarmx/desktop` and `@swarmx/cli`, carries the supported Electron runtime, and resolves both from its own install context; published `@swarmx/desktop` carries complete built main/preload/renderer assets, so cold start has no workspace or pnpm dependency.
V469: A `v<version>` tag matching every manifest builds DMG and ZIP artifacts for macOS arm64 and x64 from that exact commit, names artifacts with version and architecture, uploads them to one GitHub Release, and never publishes a mismatched tag.
V470: Root README remains at most 150 lines and leads with macOS Release download plus `npm install`/`npx swarmx` Desktop startup; it preserves concise CLI, source-development, library, and documentation links without duplicating long configuration internals.
V471: Root manifest, runtime constant, and all six publishable packages declare 3.1.2; packed workspace dependencies resolve to 3.1.2 and npm publication remains dependency-first before the matching Git tag and GitHub Release.
V472: Focused tests cover default Desktop resolution, explicit Desktop alias, unchanged CLI routing, published dependency ownership, Release tag/version and architecture gates, artifact targets, and README size/startup commands; isolated tarball installation and local macOS packaging validate the real cold path.
V473: Electron remains a `devDependency` of the electron-builder application manifest because app packaging rejects it as a production dependency; the published `swarmx` launcher owns Electron as a runtime dependency and resolves it from launcher context before spawning the separately installed Desktop app.
V474: Local Electron package output stays under `packages/desktop/release/` and that entire generated tree is excluded by both Git and Biome, so creating DMG/ZIP artifacts cannot pollute source diffs or fail lint on vendored application files.
V475: Only the top-level `swarmx` package publishes a binary named `swarmx`; `@swarmx/cli` publishes `swarmx-cli`, while the top-level launcher delegates all CLI arguments to it, so npm cannot replace the Desktop-first bin link with a transitive dependency link.
V476: macOS packaging uses electron-builder only for the architecture-specific `.app`, then uses host `ditto` for ZIP and `hdiutil` for DMG, preserving any app signature/notarization while avoiding a runtime download of electron-builder's optional DMG helper bundle.
V477: If npm installs `electron` without running its lifecycle script, first Desktop launch detects the missing binary, uses that installed version's official downloader and bundled checksums to complete the runtime, then resolves and spawns Electron; bootstrap failure exits with a concise error and the GitHub Releases fallback instead of showing an internal module stack.
V478: After Electron bootstrap, a missing `path.txt` is repaired only when the installed package version matches `dist/version` and the expected platform executable exists; the launcher then clears the failed module cache and re-resolves Electron, while missing or mismatched runtime state remains a hard failure.
V479: First-launch Electron bootstrap awaits both the checked download and platform extraction completion, clears any partial `dist` tree before extraction, and never trusts an early successful exit from Electron 33's lifecycle wrapper as proof of a complete runtime.
V480: Electron bootstrap holds one referenced event-loop timer while awaiting downloader/extractor promises and clears it in `finally`, preventing Node 26 unsettled-top-level-await exit without leaving a timer after success or failure.
V481: On macOS, first-launch Electron bootstrap extracts the checked archive with host `ditto -x -k`; other platforms retain the awaited `extract-zip` fallback. Version, executable, and marker validation remains identical after either extraction path.
V482: A normal Custom Provider owns one Base URL and one primary API key or auth token; OpenAI-compatible Model discovery appends `/models` to that exact API Base URL, and neither URL shape nor response shape silently enables the New API usage/account adapter.
V483: The exact official OpenCode Go origin normalizes one Provider into native OpenAI Chat and Anthropic Messages entrypoints plus its documented `/v1/models` discovery route. It may reference multiple separately encrypted API keys through secret-free stable key-slot metadata, while ordinary Custom Providers retain one key and one secret reference.
V484: OpenCode Go usage is local observed state only: SwarmX records per-key request count, returned input/output/reasoning/cache token usage, last use, quota exhaustion, and cooldown without calling undocumented usage endpoints, scraping the console, persisting raw Provider errors, or exposing key values through IPC/UI.
V485: A direct OpenCode Go request selects one enabled non-cooling key for the request/session and switches to the next key only after a classified quota-exhaustion response. Automatic replay is allowed only before any model output or tool event is emitted; response reset metadata controls cooldown when available and otherwise uses a bounded five-hour cooldown.
V486: Removing an OpenCode Go key deletes its encrypted entry and retains unrelated Provider/settings state. If every key is unavailable, execution returns one actionable secret-free error; tests cover generic Custom Provider isolation, official route normalization, encrypted CRUD, local usage, quota-only failover, no replay after output, all-key exhaustion, preload isolation, and responsive key-management UI.
V487: pnpm 11 dependency build policy explicitly allows required runtime/compiler scripts and denies the unused macOS-release graph's Windows installer script, so frozen CI installs fail only for newly unreviewed build scripts.
V488: The repository's declared pnpm 9/10 range retains its legacy required-build allowlist alongside pnpm 11 `allowBuilds`; installing with either supported policy runs Electron, esbuild, Biome, and node-pty scripts while skipping the unused Windows installer helper.
V489: macOS Release jobs expose electron-builder signing and notarization variables only when their corresponding GitHub secrets are nonempty; a manual dispatch can rebuild an existing matching `v<version>` tag without moving it, while every job still checks out and validates that tagged commit.
V490: The macOS artifact script resolves electron-builder's architecture directory convention explicitly as `mac-arm64` for arm64 and `mac` for x64 before creating architecture-named DMG/ZIP files; missing app output remains a hard failure before upload.

## §T
|id|status|task|cites|
|-|-|-|-|
|T1|x|write workflow JSON/UI spec|V1,V2,V3,V4,V5,V6,I1,I2,I3,I4,I5,I6,I7|
|T2|x|add renderer tests for workflow parse, graph preview, invalid JSON, and send payload|V1,V2,V3,V4,V5,I2,I4,I6|
|T3|x|implement workflow JSON editor, graph preview, validation state, and send integration|V1,V2,V3,V4,V5,I2,I4,I5|
|T4|x|fix workflow JSON docs to match `SwarmConfigSchema`|V6,I1,I7|
|T5|x|run targeted tests, build, and rendered UI validation|V1,V2,V3,V4,V5,V6,I4,I5,I6,I7|
|T6|x|correct workflow UI and JSON to model ACP agents as model plus harness|V1,V3,V4,V7,I1,I4,I5,I6,I7|
|T7|x|refine harness identity to software version, MCPs, skills, and project files|V1,V3,V4,V8,I1,I4,I5,I6,I7|
|T8|x|add n8n import converter and tests|V9,V10,V13,I1,I8,I9|
|T9|x|wire desktop n8n import IPC/UI/tests|V11,V12,I2,I3,I4,I6,I10,I11|
|T10|x|document n8n import scope|V9,V10,V11,V12,I7|
|T11|x|run targeted tests and build|V9,V10,V11,V12,V13,I4,I6,I8,I9,I10,I11|
|T12|x|review GEEPilot specs and document SwarmX platform boundary|V14,V18,I18|
|T13|x|add core extension schemas, passive inventory loader, secret guard, and composition resolver|V14,V15,V16,V17,I12,I13|
|T14|x|wire desktop read-only extension inventory IPC and renderer view|V15,V18,I14,I15,I16,I17|
|T15|x|document extension inventory API for downstream products|V14,V15,V17,I7,I18|
|T16|x|run targeted extension, renderer, and typecheck validation|V14,V15,V16,V17,V18,I13,I16,I17|
|T17|x|let desktop select and execute extension agent profiles through composition resolution|V16,V17,V19,I2,I3,I12,I14,I15,I16|
|T18|x|execute custom agent backends through ACP instead of native OpenAI fallback|V19,V20,I3,I12,I13|
|T19|x|inject extension provider profile env at invocation time without persisting secrets|V16,V17,V21,I3,I12,I13|
|T20|x|add core deterministic autonomy schemas, path contracts, and replay helpers|V22,V23,V24,V25,V26,I19,I20|
|T21|x|document autonomy primitive boundary for downstream GEEPilot dependence|V22,V25,I18,I21|
|T22|x|run targeted autonomy tests and core typecheck|V22,V23,V24,V25,V26,I19,I20|
|T23|x|add core context object, packet, checkpoint, invocation metadata schemas and packet helper|V27,V28,V29,V30,V31,I22,I23|
|T24|x|document context packet primitive boundary for downstream GEEPilot dependence|V27,V29,I18,I21|
|T25|x|run targeted context tests and core typecheck|V27,V28,V29,V30,V31,I22,I23|
|T26|x|add core normalized render event schemas, artifact/provenance schemas, and message chunk mapper|V32,V33,V34,I24,I25|
|T27|x|wire desktop run timeline through normalized sanitized render events|V34,V35,I24,I26|
|T28|x|document normalized rendering event boundary for downstream GEEPilot dependence|V32,V34,I18,I21|
|T29|x|expose normalized rendering as a browser-safe core subpath and keep desktop renderer off the Node-only root entrypoint|V36,I24,I26|
|T30|x|add core telemetry envelope schemas, redaction, opt-in config, status, headers, and injected client|V37,V38,V39,V40,V41,I27,I28|
|T31|x|document telemetry primitive boundary for downstream GEEPilot dependence|V37,V39,V41,I18,I21|
|T32|x|harden optional server boundary with loopback default, non-loopback token requirement, exact Origin allowlist, trusted null-origin switch, and bearer auth|V42,V43,V44,V45,I29,I30,I31|
|T33|x|document optional server boundary for downstream GEEPilot dependence|V42,V43,V44,I18,I21|
|T34|x|add core managed dependency schemas, policy validation, receipt/detection records, and side-effect-free planning helpers|V46,V47,V48,V49,V50,I32,I33|
|T35|x|document managed dependency boundary and expose browser-safe dependency API subpath|V46,V48,V50,V51,I18,I21,I34|
|T36|x|extend extension inventory with marketplace source and plugin catalog metadata|V52,V53,V55,I12,I13,I35|
|T37|x|render marketplace sources and plugin catalog entries separately in desktop Extensions view and docs|V52,V54,V55,I16,I18,I21,I36|
|T38|x|add core append-only conversation ledger schemas, JSONL helpers, deterministic event ids, and replay helpers|V56,V57,V58,V59,V60,I37,I38|
|T39|x|document conversation ledger boundary and expose browser-safe conversation API subpath|V56,V59,V60,V61,I18,I21,I39|
|T40|x|add core provider profile metadata, secret reference/status, selection, prompt request, and runtime env contracts|V62,V63,V64,V65,V66,I40,I41|
|T41|x|document provider profile boundary and expose browser-safe provider API subpath|V62,V64,V66,V67,I18,I21,I42|
|T42|x|add core harness discovery, selector resolution, alias, and invocation metadata contracts|V68,V69,V70,V71,I43,I44|
|T43|x|document harness management boundary and expose browser-safe harness management API subpath|V68,V70,V71,V72,I18,I21,I45|
|T44|x|harden desktop message rendering for safe Markdown, blocked remote media, preload-mediated local images, and labeled code blocks|V73,V74,V75,I46,I47|
|T45|x|document desktop message rendering policy for downstream GEEPilot dependence|V73,V74,V75,I18,I21|
|T46|x|add core action intent, risk, confirmation, secret guard, and dependency-plan action helpers|V76,V77,V78,V79,V80,V81,I48,I49|
|T47|x|document action intent boundary and expose browser-safe actions API subpath|V76,V78,V80,V82,I18,I21,I50|
|T48|x|add core agent definition frontmatter schemas, Markdown parser, profile conversion, projection helpers, and secret guards|V83,V84,V85,V86,V87,I51,I52|
|T49|x|document agent profile definition boundary and expose browser-safe agent profile API subpath|V83,V85,V87,V88,I18,I21,I53|
|T50|x|extend extension agent profiles and composition traces with definition and policy metadata|V89,V90,I12,I13,I54,I55|
|T51|x|render extension agent profile policy metadata in desktop inventory|V91,I16,I56,I57|
|T52|x|add core desktop settings document, root resolution, locale registry, UI state, and secret guard helpers|V92,V93,V94,V95,V96,I58,I59|
|T53|x|document desktop settings boundary and expose browser-safe desktop settings API subpath|V92,V93,V94,V97,I18,I21,I60|
|T54|x|extend core extension inventory with passive plugin component schemas and aggregation|V98,V99,V100,I12,I13,I61,I62|
|T55|x|render extended plugin component inventory in desktop Extensions view and docs|V101,V102,I16,I18,I21,I63,I64|
|T56|x|add core secret refs, vault document, redaction, status, file-mode, runtime lookup, and request-only policy helpers|V103,V104,V105,V106,V107,V108,I65,I66|
|T57|x|document secret-store boundary and expose browser-safe secrets API subpath|V103,V104,V105,V109,I18,I21,I67|
|T58|x|add core composition preflight schemas, plan helper, provenance requirements, and tests|V110,V111,V112,V113,V114,I12,I13,I68,I69|
|T59|x|return composition plans from desktop extension inventory and render readiness/preflight state|V115,V116,I15,I16,I70,I71,I72|
|T60|x|document composition preflight boundary and run targeted validation|V110,V111,V112,V113,V114,V115,V116,I18,I21|
|T61|x|add core scheduler/reporting schemas, cadence helpers, feedback/dashboard/wakeup contracts, and tests|V117,V118,V119,V120,V121,V122,V123,I19,I20,I73,I74|
|T62|x|document scheduler/reporting boundary for downstream GEEPilot dependence|V117,V118,V119,V120,V121,V122,V123,I18,I21|
|T63|x|run targeted autonomy validation, core typecheck, full tests, build, and diff checks|V117,V118,V119,V120,V121,V122,V123,I19,I20,I73,I74|
|T64|x|add core passive GUI contribution schemas, catalog counts, aggregation, and tests|V124,V125,V126,V127,V128,I75,I76|
|T65|x|render GUI contributions separately in desktop Extensions view and docs|V129,V130,I18,I21,I77,I78|
|T66|x|run targeted extension/renderer validation, core typecheck, full tests, build, and diff checks|V124,V125,V126,V127,V128,V129,V130,I75,I76,I77,I78|
|T67|x|add core deterministic trigger, transition decision, validator gate decision, replay record schemas, helpers, and tests|V131,V132,V133,V134,V135,V136,V137,I79,I80|
|T68|x|document deterministic runtime API boundary for downstream GEEPilot dependence|V131,V132,V133,V134,V135,V136,V137,I18,I21|
|T69|x|run targeted autonomy validation, core typecheck, full tests, build, and diff checks|V131,V132,V133,V134,V135,V136,V137,I79,I80|
|T70|x|add core engineering lifecycle state, intake, proposal, approval, transition decision schemas, helpers, and tests|V138,V139,V140,V141,V142,V143,V144,I81,I82|
|T71|x|document engineering lifecycle governance boundary for downstream GEEPilot dependence|V138,V139,V140,V141,V142,V143,V144,I18,I21|
|T72|x|run targeted autonomy validation, core typecheck, full tests, build, and diff checks|V138,V139,V140,V141,V142,V143,V144,I81,I82|
|T73|x|add core skill host exposure schemas, passive compatibility validator, and tests|V145,V146,V147,V148,V149,I12,I13,I83,I84|
|T74|x|render skill host compatibility metadata in desktop Extensions view and docs|V150,I16,I18,I21,I85,I86|
|T75|x|run targeted extension/renderer validation, core typecheck, full tests, build, and diff checks|V145,V146,V147,V148,V149,V150,I83,I84,I85,I86|
|T76|x|add offline KaTeX math rendering pipeline and renderer tests|V151,V152,V153,V155,I87,I88,I90|
|T77|x|style math blocks for stable message layout and document rendering boundary|V154,V155,I18,I21,I89|
|T78|x|run targeted renderer validation, desktop build, full tests, build, lint, and diff checks|V151,V152,V153,V154,V155,I87,I88,I89,I90|
|T79|x|add structured code-block renderer, copy controls, offline highlight fallback, and tests|V156,V157,V158,V159,I91,I92,I94|
|T80|x|style code-block controls, highlighted output, and stable overflow layout; document boundary|V156,V158,V160,I18,I21,I93|
|T81|x|run targeted renderer validation, desktop build, full tests, build, lint, and diff checks|V156,V157,V158,V159,V160,I91,I92,I93,I94|
|T82|x|add artifact option mapping and expandable sanitized trace cards with tests|V161,V162,V163,V164,V165,I95,I96,I97,I98|
|T83|x|style trace-card metadata, artifact refs, raw refs, and stable overflow layout; document boundary|V162,V163,V166,I18,I21,I97|
|T84|x|run targeted rendering/renderer validation, desktop build, full tests, build, lint, and diff checks|V161,V162,V163,V164,V165,V166,I95,I96,I97,I98|
|T85|x|add read-only specialized trace presentation detection and renderer tests|V167,V168,V169,V170,I99,I100|
|T86|x|style specialized trace fields/excerpts and document presentation-only boundary|V167,V168,V169,V171,I18,I21,I99|
|T87|x|run targeted renderer validation, desktop build, full tests, build, lint, and diff checks|V167,V168,V169,V170,V171,I99,I100|
|T88|x|add generic agent-run and workflow-decision autonomy schemas, helpers, and tests|V172,V173,V174,V175,V176,I101,I102|
|T89|x|document agent-stage trace contract and GEEPilot boundary|V172,V173,V174,V175,V176,I18,I21,I101|
|T90|x|run targeted autonomy validation, core typecheck, full tests, build, lint, and diff checks|V172,V173,V174,V175,V176,I101,I102|
|T91|x|add generic telemetry ingest config, decision, accepted-record, and injected append helper tests|V177,V178,V179,V180,I103,I104|
|T92|x|document telemetry ingest boundary for downstream GEEPilot dependence|V177,V178,V179,V180,I18,I21,I103|
|T93|x|run targeted telemetry validation, core typecheck, full tests, build, lint, and diff checks|V177,V178,V179,V180,I103,I104|
|T94|x|add desktop product config, registered GUI contribution component host, and renderer tests|V181,V182,V183,V184,I105,I106|
|T95|x|style and document downstream GUI host customization path|V181,V182,V183,V184,I107,I108,I109|
|T96|x|run targeted renderer validation, desktop build, full tests, build, lint, and diff checks|V181,V182,V183,V184,I105,I106,I107,I108,I109|
|T97|x|add stable core `executeAgentComposition` helper for downstream desktop hosts|V185,V186,I110,I111|
|T98|x|add explicit desktop LSP host lifecycle and schema compatibility|V187,V188,V189,V190,V191,I112,I113,I114,I115,I116,I117|
|T99|x|add built-in local workspace file completions through LSP API|V192,V193,V194,I118,I119,I117|
|T100|x|add built-in extension skill completions through LSP API|V195,V196,V197,I120,I121,I117|
|T101|x|add desktop Apple Container detection, setup, and protected backend wrapper|V198,V199,V200,V203,I122,I126|
|T102|x|wire protected harness send path and renderer host-repair UI|V198,V201,V202,V203,I123,I124,I125,I127|
|T103|x|document containerized harness policy and run targeted validation|V198,V199,V200,V201,V202,V203,I128|
|T104|x|decouple Agent-picker secondary panel height and scrolling from primary panel layout|V204|
|T105|x|keep desktop model-capability tests source-fresh across workspace packages|V205|
|T106|x|make Agent-picker keyboard open/close focus behavior complete|V206|
|T107|x|extract shared host runtime doctor service and tests|V207,V208,V209,V210,V211,V215,I129,I130,I131|
|T108|x|add `swarmx doctor` human/JSON/check/fix command and tests|V208,V209,V210,V212,V215,I130,I132|
|T109|x|replace permanent Setup navigation with `/setup`, `/doctor`, and on-demand doctor panel|V208,V210,V213,V214,V215,I133,I134|
|T110|x|document doctor policy and run full validation|V207,V208,V209,V210,V211,V212,V213,V214,V215,V216,I128|
|T111|~|make registered harnesses optional until explicitly selected or required|V217,V218,V219,V220,I129,I130,I131,I132,I134|
|T112|x|replace provider-owned model semantics with standalone Model and many-to-many ModelSupply contracts|V69,V222,V223,V224,V225,I135,I136,I137,I139,I142|
|T113|x|implement stable ACP model/effort negotiation and request-scoped external harness model controls|V226,V227,V228,I136,I138,I142|
|T114|x|forward model bootstrap variables and yallm host bridging through protected runtime launches|V228,V229,I137,I140,I142|
|T115|x|render and send the provider-independent `Harness x Model` matrix in desktop composition flows|V224,V225,V230,I139,I141,I142|
|T116|x|document standalone Model semantics and run targeted plus full validation|V222,V223,V224,V225,V226,V227,V228,V229,V230,V231,I143|
|T117|x|disable unconfigured Harness options without hiding capability gaps|V226,V232,I141|
|T118|x|add fixed Harness-Model runtime routes and Claude Code DeepSeek env injection|V225,V228,V229,V234,I136,I139,I140,I142|
|T119|x|reduce desktop Agent picker to Harness, Model, and Effort|V204,V225,V230,V233,I141,I142|
|T120|x|document fixed internal routing and run full validation|V230,V231,V233,V234,I143|
|T121|x|add dynamic Model catalog service, provider discovery adapters, cache, and manual persistence|V235,V236,V237,V238,V241,I144,I145|
|T122|x|add Model catalog IPC and three-row renderer management surface|V238,V239,V241,I146,I147|
|T123|x|reuse augmented catalog for execution and resolve internal ModelSupply deterministically|V225,V224,V240,V241,I148,I146|
|T124|x|document dynamic Model sources and run full validation|V235,V236,V237,V238,V239,V240,V241,I149|
|T125|x|add Provider auth-mode schemas and request-scoped runtime secret overrides|V244,V245,V246,V249,I150|
|T126|x|add encrypted Provider auth store and catalog Provider CRUD|V242,V243,V245,V246,V247,V249,I151,I152|
|T127|x|wire Provider management IPC, Settings UI, refresh, and execution secrets|V242,V246,V247,V248,V249,I153,I154|
|T128|x|document Provider configuration and run full validation|V242,V243,V244,V245,V246,V247,V248,V249,I155|
|T129|x|restore lower-left account/update area and move Provider CRUD into Settings|V242,V248,V251,V252,V253,V254,I153,I154,I156|
|T130|x|update Settings docs, run rendered design QA, and complete full validation|V242,V248,V251,V252,V253,V254,I155|
|T131|x|implement npm update check, verified download, versioned install, and relaunch service|V255,V257,V258,V259,I156|
|T132|x|replace the update row with Codex account-row available/hover/progress states and IPC|V253,V254,V256,V259,I157,I158|
|T133|x|update updater docs, run rendered reference QA, and complete full validation|V253,V255,V256,V257,V258,V259,I158|
|T134|x|integrate secure Provider balance/quota adapters and Codex account limits into the Provider workspace|V252,V260,V261,V262,V263,V264,V265,V266,I159,I160,I161|
|T135|x|document the Provider Usage support boundary and complete validation|V260,V261,V262,V263,V264,V265,V266,I161|
|T136|x|remove duplicate Usage navigation and separate Provider identity from environment provenance|V251,V252,V254,V256,V267,V268,V269,I154,I162|
|T137|~|remove ambient-environment Provider synthesis from Desktop|V237,V265,V268,V269,V270,I163|
|T138|x|add explicit New API Provider usage selection, adapter, tests, and live Packy verification|V264,V265,V271,V272,I153,I154,I159,I160|
|T139|x|build unified Provider quota matrix, account summaries, and native DeepSeek dual entrypoints|V243,V245,V252,V264,V265,V267,V268,V270,V271,V272,V273,V274,V275,V276,V277,V278,V279,V280,V281,I151,I152,I153,I154,I159,I160,I161,I162,I163,I164,I165,I166|
|T140|~|persist Model catalog across restarts without automatic Provider discovery|V235,V239,V241,V242,V282,I144,I145,I146,I147|
|T141|~|add native Anthropic and OpenAI Responses execution to the SwarmX Harness with yallm fallback only|V224,V225,V229,V230,V240,V244,V277,V283,V284,V285,I135,I136,I137,I139,I140,I142,I164,I167|
|T142|~|group and persist routed Provider model catalogs with safe Codex discovery and effort metadata|V222,V223,V225,V230,V233,V235,V239,V240,V277,V282,V286,V287,V288,V289,V290,V291,V292,V293,I144,I145,I146,I147,I148,I164,I168,I169|
|T143|x|add logical Skill variant, binding, delivery, evolution, and deterministic resolver contracts with legacy migration tests|V297,V298,V299,V305,V306,V307,I170|
|T144|x|add Extension marketplace/update/action/receipt/rollback contracts and host manager tests|V303,V304,V305,V306,V307,I171,I172|
|T145|x|extract shared atomic Settings store and implement secret-safe Custom Agent persistence plus IPC|V295,V296,V302,V307,I172,I173|
|T146|x|move Extension management into Settings and add explicit source/plugin lifecycle actions|V18,V55,V294,V303,V304,V307,I171,I173,I174|
|T147|x|build Custom Agents Settings Harness recipe UI with Software health, Skills, MCPs, project context, permissions, Model, and context cost|V295,V296,V298,V299,V300,V307,I170,I173,I174,I175|
|T148|x|build shared Runtime Settings, remove sidebar Doctor status, and keep slash-command/on-demand repair flows|V294,V300,V301,V307,I173,I174,I175|
|T149|x|document manifest migration, Custom Agents, Extension update/evolution, Runtime placement, trust, and rollback|V297,V303,V304,V305,V306,I176|
|T150|x|run targeted tests, full tests, builds, lint, and rendered Settings QA|V294,V295,V296,V297,V298,V299,V300,V301,V302,V303,V304,V305,V306,V307,V308,I170,I171,I172,I173,I174,I175,I176|
|T151|x|replace Bun requirements and ACP launchers with the Node.js/npm ecosystem across host and protected execution|V300,V309,V313,I177|
|T152|x|rebuild Runtime around Node.js, Harness tool rows, semver refresh, and embedded Doctor diagnostics|V301,V310,V311,V312,V313,I174,I175,I177|
|T153|x|update tests and documentation, run full validation, and perform rendered Runtime QA|V300,V301,V309,V310,V311,V312,V313,I176,I177|
|T154|x|add dual Claude Code Markdown and Codex TOML Agent codecs, normalized metadata, projections, and tests|V314,V315,V316,V318,V320,I178,I179|
|T155|x|discover native Agent files read-only, merge them into desktop inventory, and render source groups with tests|V317,V318,V319,V320,I180,I181,I182|
|T156|x|document native Agent compatibility and run targeted plus full validation|V314,V315,V316,V317,V318,V319,V320,I183|
|T157|x|reuse Composer Provider grouping, routed selection, and canonical Model order in Custom Agents|V288,V289,V290,V295,V296,V321,I169,I174,I184|
|T158|x|separate collection-level Project organization from per-Project actions and verify both menu flows|V326,V327,V328|
|T159|x|bootstrap persisted Projects before Renderer mount and remove the Project loading transition|V331,I185,I186|
|T160|x|backprop and repair the executable Harness x Model matrix, adapter identity, protected auth bridge, and CLI model switching|V332,V333,V334,V335,V336,V337,I123,I135,I136,I138,I139,I140,I141,I142|
|T161|x|backprop and repair direct Codex subscription conversation execution|V338,I135,I136,I139|
|T162|x|inject bounded Project identity and workspace tools into direct SwarmX compositions|V339,V340,I187,I188|
|T163|x|persist request timing and rebuild Worked disclosure presentation|V341,V342,I189,I191|
|T164|x|add cheap-model automatic title generation and local Session mutation IPC|V343,V344,V345,I189,I190|
|T165|x|add centered rename dialog, double-click rename, and task context menu with pin/delete|V344,V345,I191|
|T166|x|run targeted/full validation and rendered desktop QA for Project-aware task lifecycle|V339,V340,V341,V342,V343,V344,V345,I187,I188,I189,I190,I191|
|T167|x|backprop the missing direct SwarmX coding-tool contract and implement atomic Project write/edit with stale-content protection|V346,V347,V348,I192|
|T168|x|implement fail-closed sandboxed Project Shell execution with limits, environment isolation, timeout, and cancellation|V349,V350,I193|
|T169|x|wire the complete coding tool set only into direct SwarmX compositions and update Project instructions|V346,V351,I192,I193|
|T170|x|run targeted/full validation and restart the desktop development Main process|V347,V348,V349,V350,V351,I192,I193|
|T171|x|render Worked Thought as unboxed conversation body content and verify the disclosure flow|V352,I194|
|T172|x|backprop and repair live desktop work streaming, Responses tool-call continuation, terminal collapse, and Thought emphasis|V353,V354,V355,I195,I196|
|T173|x|audit current Claude Code and Codex tool definitions and backprop profile invariants|V356,V357,V358,V359,V360,I197|
|T174|x|add function/freeform local-tool transport and model-family Project tool profiles|V356,V357,V358,V359,I198,I199|
|T175|x|add profile, compatibility, patch, search, Shell, and Responses continuation tests|V356,V357,V358,V359,V360,I198,I199|
|T176|x|document primary sources, attribution, hosted-search boundary, and run validation|V360,I197,I200|
|T177|x|add dual model/client tool-result transport across native and Chat protocols|V363,V369,I201,I204|
|T178|x|implement managed Project Shell sessions, background lifecycle, stdin, polling, stop, timeout, and cleanup|V365,V366,V367,I202,I204|
|T179|x|map Claude Code/Codex file, search, Shell, and session result structures|V364,V365,V366,I203,I204|
|T180|x|document compatibility boundary and run targeted/full validation|V363,V364,V365,V366,V367,V368,V370,I197,I204|
|T181|x|implement sandboxed Codex PTY sessions, interaction, cleanup, docs, and validation|V349,V350,V359,V366,V367,V368,V371,V372,V373,I199,I202,I203,I204,I205|
|T182|x|implement first Claude Code parity batch: tasks, Todo, findings, NotebookEdit, inventory, and validation|V374,V375,V376,V377,V378,V379,I197,I203,I204,I206,I207|
|T183|x|project conditional Claude MCP resource listing/reading and validate cancellation/output contracts|V380,V381,I208|
|T184|x|align Claude Glob ignore, ordering, cap, and count-completeness semantics|V364,V374,V382,I203,I204|
|T185|x|align Claude Bash timeout-to-background lifecycle without weakening sandbox or cancellation|V365,V367,V383,I202,I203,I204|
|T186|x|project selected Claude Skills with bounded loading and native argument expansion|V374,V384,I206,I209|
|T187|x|implement Claude interactive questions and plan-mode approval bridge|V374,V386,V387,V388,V389,V390,V391,V392,V393,V394,V395,I203,I204,I210,I211|
|T188|x|implement Claude deferred MCP ToolSearch and WaitForMcpServers lifecycle|V374,V380,V381,V396,V397,V398,V399,V400,V401,I208,I212|
|T189|x|project configured desktop language servers through the Claude LSP tool|V374,V402,V403,V404,V405,V406,V407,I203,I204,I213|
|T190|x|implement Claude request-scoped Git worktree entry, root rebinding, guarded exit, tests, and documentation|V374,V408,V409,V410,V411,V412,V413,V414,V415,I203,I204,I214|
|T191|x|implement conditional synchronous Claude Agent composition and request-scoped resume|V374,V416,V417,V418,V419,V420,V421,V422,I203,I204,I215|
|T192|x|audit the remaining trained-in tools and document non-fake runtime blockers|C146,C157,V374,V423,I207|
|T193|x|implement session event pump plus Claude Monitor and Cron tools|C158,C159,C160,V424,V425,V426,V427,V428,V429,V430,V431,I203,I204,I207,I216|
|T194|x|implement durable Claude Cron persistence, ownership, recovery, and documentation|G59,C160,V432,V433,V434,V435,V436,V437,V438,V439,I207,I216,I217|
|T195|.|implement Windows-native sandboxed Claude PowerShell parity|C157,V374,V423,V440,I203,I207|
|T196|.|implement concurrent Claude teammate SendMessage mailboxes and lifecycle|C157,V374,V423,V440,I203,I207|
|T197|.|implement persisted resumable Claude Workflow VM parity|C157,V374,V423,V440,I203,I207|
|T198|x|clean, version, verify, publish, and verify SwarmX 3.1.1|G60,C161,V440,V441,V442,I207,I218|
|T199|x|enforce direct and ACP permission decisions with desktop approval UI, policy Settings, tests, and source audit|G61,C162,C163,C164,V443,V444,V445,V446,V447,V448,I219,I220,I221|
|T200|x|build layered permission governance, dedicated UX, sanitized receipts, tests, and visual QA|G62,C162,C163,C164,C165,C166,C167,C168,V443,V444,V445,V446,V447,V448,V449,V450,V451,V452,V453,V454,V455,V456,I219,I220,I221,I222,I223,I224|
|T201|x|move defaults to General and add safe persisted conversation permission overrides|G63,C162,C163,C165,C166,C167,C168,C169,C170,C171,V443,V444,V449,V450,V451,V452,V457,V458,V459,V460,V461,V462,I222,I223,I224,I225,I226,I227|
|T202|x|backprop Codex permission UI parity and implement profile availability plus Auto-review|G64,C162,C163,C165,C166,C168,C169,C170,C171,C172,C173,V444,V449,V457,V458,V459,V460,V461,V462,V463,V464,V465,V466,I222,I223,I224,I225,I226,I227,I228|
|T203|x|fix npm Desktop cold start, add dual-architecture macOS Release packages, simplify README, verify, and publish 3.1.2|G65,C174,C175,C176,V467,V468,V469,V470,V471,V472,V473,V474,V475,V476,V477,V478,V479,V480,V481,V487,V488,V489,V490,I229,I230,I231,I232|
|T204|x|implement Custom Provider exact discovery and OpenCode Go encrypted multi-key local-usage failover with Provider Settings UI|G43,G44,G46,G47,C3,C15,C47,C51,V236,V243,V245,V246,V282,V283,V284,V302,V482,V483,V484,V485,V486,I144,I151,I152,I153,I154,I159,I160,I161,I233,I234,I235,I236|

## §B
|id|date|cause|fix|
|-|-|-|-|
|B1|2026-06-17|renderer tests used per-char input and hit 5s suite timeout|use deterministic input change in tests|
|B2|2026-06-19|workflow UI modeled generic agent/tool nodes instead of ACP agent identity|V7|
|B3|2026-06-19|workflow UI collapsed harness to backend label and lost software version/MCP/skills/project context|V8|
|B4|2026-07-11|secondary options shared parent grid height and stretched primary Agent menu|V204|
|B5|2026-07-11|desktop Vitest loaded stale core model-capability dist after a core source change|V205|
|B6|2026-07-11|Escape listener existed only inside the menu while pointer-open focus stayed on trigger|V206|
|B7|2026-07-11|root Vitest concurrency made full-App dynamic import exceed the 5s unit-test default|V216|
|B8|2026-07-11|filtered Doctor JSON inherited global environment readiness and contradicted its healthy report|V207|
|B9|2026-07-11|environment and Doctor treated the full harness registry as a required install set|V217,V218,V219,V220|
|B10|2026-07-12|collapsed sidebar moved navigation into a main title bar without the macOS traffic-light inset|V221|
|B11|2026-07-12|partial standalone-Model migration left consumers importing removed provider-owned harness fields|V224,V226|
|B12|2026-07-12|legacy tests and fixtures still embedded models in Provider records or re-registered an existing standalone Model id|V222,V223,V231|
|B13|2026-07-12|new agent-identity test expected a hyphenated runtime name although AgentConfig names normalize separators to underscores|V225|
|B14|2026-07-12|desktop picker still projected provider-owned model fields after the core matrix became Harness x Model|V230|
|B15|2026-07-12|renderer mock inventories still used legacy Harness and Provider-owned Model fields, leaving the new matrix empty|V230,V231|
|B16|2026-07-12|enabled invalid workflow JSON fell through to the newly available manual Harness x Model composition|V2|
|B17|2026-07-12|new Harness x Model render-provenance fixture omitted the normalized event's required summary|V32|
|B18|2026-07-12|matrix migration edits were build-correct but had not yet been normalized by the repository Biome formatter|V231|
|B19|2026-07-12|OpenClaw remained selectable in the desktop Harness menu despite its unsupported model-control state|V232|
|B20|2026-07-12|Composer exposed ModelSupply as a fourth primary choice even though Harness-Model routing is SwarmX-owned infrastructure|V233,V234|
|B21|2026-07-12|ad hoc desktop `tsconfig.node.json` verification included preload under a main-only rootDir and failed before checking task code|use the package build or dedicated main/preload library tsconfigs|
|B22|2026-07-12|Model catalog status used `div role=status` instead of the available semantic element|render status with native `output`|
|B23|2026-07-12|Provider readiness was accepted only as passthrough data, so desktop library type-checking saw `runtimeNote` as `unknown`|V250|
|B24|2026-07-12|Provider update lookup reread an optional input property inside a callback after a guard, which TypeScript could not narrow|normalize the optional id once before lookup|
|B25|2026-07-12|new Model catalog imports were format-correct but not in Biome's semantic import order|apply Biome import organization before final lint|
|B26|2026-07-12|React sidebar migration carried sessions/navigation but dropped the old fixed Settings footer and left complex Provider configuration inside Agent Picker|V251,V252,V248|
|B27|2026-07-12|Usage integration retained duplicate account navigation and encoded credential provenance in canonical Provider labels|V267,V268,V269|
|B28|2026-07-13|desktop catalog startup scanned ambient Provider variables and synthesized network-active connections without explicit user configuration|V270|
|B29|2026-07-13|resetting a Provider Usage API emitted an own `usageAdapter: undefined` property because runtime mapping always materialized the optional field|V271|
|B30|2026-07-13|credential rollback mutated a whole-document auth store concurrently and could overwrite one restored entry|V280|
|B31|2026-07-13|conditional helper copy was nested inside a Provider field label and silently changed the control's accessible name|V281|
|B32|2026-07-13|persisted DeepSeek records created before native dual-entrypoint canonicalization bypassed save-time normalization and attempted model discovery below `/anthropic`|V286|
|B33|2026-07-13|the Agent picker collapsed catalog routes into one protocol-only model row, hiding Provider provenance and preventing an intentional Provider route from reaching composition|V288,V289|
|B34|2026-07-14|completion tests queried document text as if it were popup state and supplied a cursor beyond the fixture line while claiming to test a URI scheme|V308|
|B35|2026-07-14|merging cached marketplace candidates with manifest catalog entries inferred a narrow union that hid optional catalog display fields|project both sources to `ExtensionPluginCatalogEntrySummary` before deduplication|
|B36|2026-07-14|Claude Code and Codex ACP launchers were modeled as Bun-dependent even though the supported adapters run in the Node.js/npm ecosystem|V309|
|B37|2026-07-14|Runtime mixed baseline dependencies with selected Harness CLI requirements, so Claude Code and Codex were absent from the tool inventory|V300,V310|
|B38|2026-07-14|generic requirement detection displayed the first raw version-output line, leaking the Hermes banner and container build text into the version column|V311|
|B39|2026-07-14|Runtime linked to Doctor as a separate transient panel instead of owning the environment diagnosis and repair workflow|V301,V312|
|B40|2026-07-15|new Project tests compared unresolved macOS temp paths even though the registry correctly canonicalized them through realpath|V322|
|B41|2026-07-15|the desktop Project picker test cleaned its user-level registry fixture only after assertions, so an earlier failure left a synthetic Project in the real sidebar|V323|
|B42|2026-07-15|the Project pin-order test compared the entire user-level registry against two fixture ids and failed when a real Project was already registered|V324|
|B43|2026-07-15|Model and Provider icons used root-relative public paths, which became `file:///harness-icons/...` and failed in the packaged Electron renderer|V325|
|B44|2026-07-15|the Projects heading overflow reused the active Project action menu, conflating collection organization with per-Project commands|V326,V327|
|B45|2026-07-15|the new row-level Project menu test queried controls before asynchronous Project discovery rendered the row|V328|
|B46|2026-07-16|the interactive Project hover card remapped an `aside` to `role=dialog` instead of using the available semantic element|V329|
|B47|2026-07-16|the full renderer suite's load caused per-character rename input simulation to drop its first character, repeating the timing failure class recorded in B1|V330|
|B48|2026-07-16|two full-App integration tests still used per-character text simulation and Vitest's 5s unit-test timeout, so concurrent desktop-suite load produced false timeouts|V216,V330|
|B49|2026-07-16|the Renderer asynchronously re-fetched an already persisted Project registry and blocked the sidebar behind a redundant loading placeholder|V331|
|B50|2026-07-16|session Harness compatibility treated `any` as every registered Model, while custom Harness ids bypassed built-in bootstrap/protection and CLI send discarded the selected Agent|V332,V333,V335|
|B51|2026-07-16|provider protocol compatibility was mistaken for a pinned Claude ACP session route, and protected Codex attempted to forward a token variable the adapter never reads instead of reusing its official authentication state|V334,V336|
|B52|2026-07-16|Codex App Server and pinned Codex ACP advertised different model sets, but one catalog supply named both SwarmX and Codex without proving the adapter route|V337|
|B53|2026-07-16|direct `codex_responses` used the generic unary SDK path when desktop execution supplied no chunk callback, while the ChatGPT Codex endpoint accepts only streaming Responses requests|V338|
|B54|2026-07-16|the long-running Electron development Main process predated the V338 core rebuild and retained the externalized `@swarmx/core` module in Node's process cache, so Renderer reloads still executed the old unary request path|restart the desktop Main process after core runtime changes|
|B55|2026-07-16|Project selection supplied only a subprocess `cwd`, while the direct native SwarmX Model had neither Project identity nor filesystem tools|V339,V340|
|B56|2026-07-16|desktop response chunks were persisted without request timing metadata, so Worked could not derive elapsed time after completion or reload|V341,V342|
|B57|2026-07-16|new local Sessions kept the placeholder title and exposed no title-generation or rename mutation path|V343,V344|
|B58|2026-07-16|sidebar Session rows were plain navigation buttons without local pin state or a task action context menu|V345|
|B59|2026-07-16|Project capability was implemented as read-only context inspection, so the direct SwarmX Harness lacked the write, edit, and Shell primitives expected from a coding agent|V346,V347,V348,V349,V350|
|B60|2026-07-16|compact Worked reasoning reused the generic run-event card and header treatment, creating a redundant nested frame unlike normal assistant text|V352|
|B61|2026-07-16|desktop composition execution exposed no chunk IPC and mounted Worked closed, while streaming Responses ignored `response.output_item.done` and trusted an incomplete reasoning-only `response.completed.output`, so the Project tool call disappeared and the turn ended without a final answer|V353,V354,V355|
|B62|2026-07-16|custom `workspace_*` schemas discarded trained-in Claude Code/Codex signatures and rejected harmless extra arguments more strictly than their native hosts|V356,V357,V358,V359|
|B63|2026-07-16|native-protocol MCP test doubles retained only `toolsForOpenai()`, so the new function/freeform `toolsForNative()` contract broke three existing native tests before their provider calls|V361|
|B64|2026-07-16|Shell workdir containment compared a canonical Project root with a lexical macOS `/var` path before `realpath`, falsely rejecting a contained `/private/var` directory alias|V359|
|B65|2026-07-16|the two-session ACP model-advertisement integration test used Vitest's 5-second unit default and timed out only under the full core-suite process load while passing alone in 647ms|V362|
|B66|2026-07-16|the guarded write/edit errors still instructed models to call removed `workspace_read_file`, leaking the superseded host vocabulary after profile selection|V357,V359|
|B67|2026-07-17|native protocol test doubles returned legacy plain objects during dual-result migration, so continuation output became `undefined` until the boundary normalized both contracts|V369|
|B68|2026-07-17|full desktop-suite scheduling delayed a real sandbox process beyond the background test's 2-second internal wait, so a valid `running` poll failed a completion assertion|V370|
|B69|2026-07-17|the deny-default Seatbelt profile omitted terminal-device `file-ioctl`, so a real node-pty session passed `isatty` but rejected `stty` and canonical interactive input|V373|
|B70|2026-07-17|`TaskUpdate` returned a definite runtime error string but its inferred success/failure union left model-facing failure content typed as optional|V379|
|B71|2026-07-17|the Bash timeout-to-background test assumed a newly spawned process would flush prefix output within 10ms under full-suite scheduling load|V370,V383|
|B72|2026-07-17|an `as const` default inferred a literal timeout parameter and a merged Shell-result union could not prove that running status implied `sessionId`|V383|
|B73|2026-07-17|the selected-Skill test compared lexical macOS temp paths while bounded Skill loading correctly returned canonical `/private/var` paths|V322,V384|
|B74|2026-07-17|`TaskUpdate` validated optional status after applying earlier fields and dependency links, allowing an invalid multi-field call to partially mutate request state|V385|
|B75|2026-07-17|the standalone interaction-dialog test inherited Vitest's Node environment, so Testing Library failed before rendering with `document is not defined`|V390|
|B76|2026-07-17|the standalone interaction-dialog test assumed `@testing-library/jest-dom` matchers that repository Vitest setup does not install|V391|
|B77|2026-07-17|the real PTY test assumed readiness output would flush in the first yield, so full-suite scheduling returned a valid running session before its marker appeared|V392|
|B78|2026-07-17|new interaction wiring left one unsorted import pair and split one interpolated message across concatenated template literals|`pnpm lint`|
|B79|2026-07-17|the pipe-stdin test assumed child termination and output flush would finish in the same 2-second write call under full-suite scheduling load|V393|
|B80|2026-07-17|a multi-step Agent Picker renderer test used Vitest's 5-second unit default and timed out at 5.078 seconds only under full-monorepo contention|V394|
|B81|2026-07-17|the separately declared workspace-tool options object gave its interaction callback no contextual parameter type during strict desktop declaration build|V395|
|B82|2026-07-17|the first MCP discovery implementation passed behavior and type checks but left four changed TypeScript files outside Biome's canonical line wrapping|V401|
|B83|2026-07-17|the first Claude LSP implementation passed protocol tests but retained noncanonical wrapping and one unsorted imported symbol group|V407|
|B84|2026-07-17|new worktree tests compared the restored caller-supplied macOS `/var` root binding with the canonical `/private/var` path returned in model-facing worktree output|V415|
|B85|2026-07-17|the first exact Agent-output assertion compared SwarmX's private local-result symbol marker in addition to the Claude-visible output fields|V422|
|B86|2026-07-17|the initial recurring-Cron fixture used a yearly schedule, so the correct three-day auto-expiry removed it after its first run before its next occurrence|V428|
|B87|2026-07-17|the Claude-visible Cron job output correctly made recurring optional, but the internal scheduler record inherited that optional field even though re-arming requires a definite boolean|V427,V428|
|B88|2026-07-17|a Cron semantics assertion was added after the targeted formatter pass and required Biome's canonical single-line wrapping|V431|
|B89|2026-07-17|the final zsh matrix-count check used the shell's read-only `status` variable name and stopped after the diff check|use a non-reserved loop variable and rerun the count|
|B90|2026-07-17|the first Claude binary source-extraction command double-escaped a control-character regex and Node 26 rejected the invalid range|replace only NUL bytes without a regex and rerun extraction|
|B91|2026-07-17|the initial Cron runtime modeled every job as session memory, rejected the already-public durable mode, and generated a prefixed id unlike Claude Code, leaving no project store, ownership lock, or restart recovery path|V432,V433,V434,V435,V436,V437,V438|
|B92|2026-07-17|the durable-Cron red test run correctly failed module loading before the new scheduled-task store implementation existed|V432,V433,V434,V438|
|B93|2026-07-17|three new durable-store fixtures used mnemonic eight-character ids containing non-hex letters even though Claude Code creates UUID-prefix hex ids and the new-write boundary correctly enforced that shape|V432,V438|
|B94|2026-07-17|the session flush barrier tracked queued activations but not the new asynchronous persistence-before-activation Cron operation, allowing a test or shutdown boundary to observe no activation while the durable write was still in flight|V436,V438|
|B95|2026-07-17|a combined retry/ownership patch targeted two identical-looking Cron catch blocks with insufficient surrounding context and failed without changing the runtime|reapply the changes as smaller context-specific patches|
|B96|2026-07-17|an ad hoc `tsc -p tsconfig.node.json --noEmit` validation included test/preload files under an incompatible main-only `rootDir` and surfaced numerous unrelated dirty-worktree fixture errors, so it was not the package's canonical declaration-build boundary|use the repository's desktop/recursive build scripts for authoritative type validation and keep focused behavior tests separate|
|B97|2026-07-17|the first generated-residue cleanup used a prohibited force-delete command even though both artifact paths were known exactly|delete each known artifact with exact `unlink` calls and remove only the empty artifact directory|
|B98|2026-07-17|`pnpm ci` invoked pnpm 10's unimplemented built-in command instead of the repository script named `ci` after lint had passed|invoke the release gate explicitly as `pnpm run ci`|
|B99|2026-07-17|the combined `pnpm run ci` exceeded the command yield after tests/build passed, and the wrapper printed an undefined exit status without retaining the session id or final Inspect output|verify the remaining `uv sync` and Inspect gate in a separate bounded command|
|B100|2026-07-17|the Inspect release gate selected uv's Python 3.14 environment whose downloaded `pydantic_core` extension was rejected by macOS code-signing policy before any project test ran|recreate the Inspect environment with a supported signed Python runtime and rerun the unchanged gate|
|B101|2026-07-18|Custom Agent permission policy was passive metadata and ACP permission requests always returned cancelled, so displayed authority did not match executable desktop behavior|V443,V444,V445,V446,V447|
|B102|2026-07-19|the first General permission redesign treated Codex's three independent capabilities as four exclusive modes, used oversized conversation chrome, and left no executable Auto-review profile|V460,V461,V463,V464,V465,V466|
|B103|2026-07-20|published `swarmx` imported only `@swarmx/cli`, `@swarmx/desktop` did not own an installable Electron runtime, and no macOS package/Release workflow existed|V467,V468,V469,V470|
|B104|2026-07-20|the first cold-start design put Electron in `@swarmx/desktop` production dependencies, but electron-builder rejects Electron outside app `devDependencies`|V473|
|B105|2026-07-20|npm offline cache entries do not provide packuments for exact unpublished internal 3.1.2 dependencies, so a root-tarball-only prepublish install could not resolve `@swarmx/cli@3.1.2`|validate the true topology through a temporary local registry or after publication|
|B106|2026-07-20|the first local macOS package build wrote vendored Electron JSON under an unignored `packages/desktop/release/` tree, so repository lint scanned generated application files and failed|V474|
|B107|2026-07-20|Verdaccio still required npm API authentication for publish even when package policy allowed `$all`, so the first temporary-registry cold-install setup published no packages|use an isolated temporary npm user config and local test account|
|B108|2026-07-20|Verdaccio marked its npmjs uplink offline behind the current network proxy, so a fully proxied local-registry install could not resolve public Electron/React packages|route only `@swarmx/*` to the temporary registry and keep public dependencies on canonical npm|
|B109|2026-07-20|npm linked the transitive `@swarmx/cli` binary over the direct `swarmx` package's same-named launcher, so a real root-only install still opened CLI instead of Desktop|V475|
|B110|2026-07-20|the new Biome release-ignore array used manual multiline JSON formatting that differed from the repository formatter|apply Biome's canonical single-line array formatting|
|B111|2026-07-20|electron-builder completed the arm64 app and ZIP but timed out after 600 seconds downloading its optional DMG helper bundle, preventing local DMG completion despite a healthy packaged app|V476|
|B112|2026-07-20|the new macOS artifact script passed behavior checks but retained one noncanonical import order and one overlong builder-argument array|apply Biome's canonical import and wrapping format|
|B113|2026-07-20|npm 11 installed Electron but suppressed its lifecycle download in a fresh cache, leaving no Electron binary for the now-correct top-level Desktop launcher|V477|
|B114|2026-07-20|Electron's official bootstrap returned success with a matching downloaded binary and version file but no `path.txt`, so `electron/index.js` still rejected the completed runtime|V478|
|B115|2026-07-20|Electron 33 `install.js` exited zero under Node 26 after extracting only about 312 KB of a 253 MB archive, leaving no complete Frameworks or version marker for Desktop startup|V479|
|B116|2026-07-20|Node 26 exited an otherwise awaited Electron downloader promise as unsettled top-level await because the dependency left no referenced event-loop handle while work remained pending|V480|
|B117|2026-07-20|after keeping Node alive, Electron 33's `extract-zip` dependency on macOS/Node 26 stalled on the first archive entry instead of completing the 253 MB runtime extraction|V481|
|B118|2026-07-20|the first full-suite release run hit the existing scheduled-task file-watcher test's two-second notification timeout under concurrent load; the focused retry and unchanged full-suite rerun passed|V434|
|B119|2026-07-20|electron-builder introduced `electron-winstaller@5.4.0`, whose unused Windows-only install script was absent from pnpm 11 `allowBuilds`, so both fresh Linux CI jobs failed during frozen install before tests|V487|
|B120|2026-07-20|removing pnpm 11's deprecated legacy build allowlist made the repository's still-supported pnpm 10 skip every required dependency script during a forced install|V488|
|B121|2026-07-20|the new cross-version build-policy test used a one-line file read that differed from Biome's canonical wrapping|apply Biome formatting before the release rerun|
|B122|2026-07-20|pnpm 10 forwarded its `--no-git-checks` publish flag to npm 11, which rejected the unknown option before uploading any package|publish the dependency-ordered pnpm-generated tarballs with npm|
|B123|2026-07-20|GitHub injected an absent macOS certificate secret as empty `CSC_LINK`; electron-builder resolved it to the Desktop directory and failed both architectures with `not a file` before artifact upload|V489|
|B124|2026-07-20|the original Release regression test required GitHub's push-only `GITHUB_REF_NAME` after the workflow intentionally unified tag pushes and manual rebuilds under validated `RELEASE_TAG`|V469,V489|
|B125|2026-07-20|electron-builder emitted the Intel application under its conventional `release/mac` directory while the artifact script looked only for `release/mac-x64`, so x64 failed after successful packaging but before archive upload|V490|

# SwarmX Documentation

SwarmX runs direct model agents, ACP-compatible coding agents, and durable
language-independent tasks in one local-first desktop workspace. The repository
also publishes a TypeScript orchestration core, CLI, ACP server adapter, and
runtime diagnostics.

## Start here

- [Product specification](https://github.com/tcztzy/swarmx/blob/main/SPEC.md) -
  durable product contract and limits.
- [Codebase map](codebase/index.md) - token-efficient source ownership, flows,
  and per-file contracts for document-driven development.
- [Product vision](vision.md) - user experience direction.
- [Architecture](https://github.com/tcztzy/swarmx/blob/main/DESIGNS.md) - package
  and runtime boundaries.
- [Roadmap](https://github.com/tcztzy/swarmx/blob/main/ROADMAP.md) - unfinished
  work only.
- [Publication-first research strategy](publication-research-strategy.md) -
  ontology, information bounds, decisive experiments, and stop conditions.
- [Agentic execution ontology charter](agentic-execution-ontology.md) -
  identity rules, disjoint categories, action lifecycle, and operational
  invariants for the research program.
- [Repository README](https://github.com/tcztzy/swarmx/blob/main/README.md) -
  installation and source development.

Feature guides:

- [First run and Doctor](first-run-and-doctor.md)
- [Session causal timeline](session-timeline.md)
- [Personal Memory](personal-memory.md)
- [Memory](memory.md)
- [Reference Library](reference-library.md)
- [Model, Harness, and Agent guidance](agent-guidance.md)
- [Durable task runtime](durable-task-runtime.md)
- [Coding-agent Context Engine](context-engine.md)
- [Context-policy evaluation](context-evaluation.md)
- [Auditability](auditability.md)
- [Extensions and Custom Agents](extensions-custom-agents.md)
- [Skill self-improvement (evolution)](skill-evolution.md)
- [Multimedia attachments](multimedia.md)
- [Native tool compatibility](native-tool-compatibility.md)
- [Claude Code tool parity](claude-code-tool-parity.md)
- [OpenAI-compatible server](server.md)
- [Hooks](hooks.md)
- [Direct Harness release acceptance](direct-harness-release-e2e.md)

## Core concepts

| Concept | Meaning |
| --- | --- |
| Project | Local folder bookmark and containment root for direct task tools |
| Provider | Explicit connection and credential source that may supply Models |
| Model | Independent model identity with API and capability metadata |
| Harness | Runtime recipe containing Software, Skills, MCP servers, context, and policy |
| Agent | One Harness paired with one Model |
| Session | Canonical persisted conversation history; an observer of durable work |
| Personal Memory surface | Bounded Settings snapshot within Memory, injected with a visible per-run receipt |
| Memory | User-owned subjective knowledge; currently organized as revision-safe Markdown/Git pages with CRUD, search, versions, restore, and derived links |
| Reference Library | Read-only, on-demand search and bounded plaintext reads from explicit ZIM and local Zotero sources |
| WorkItem | Session-independent durable work with event-replayed runs and checkpoints |
| Workflow | A `SwarmConfig` graph |
| Memory graph | A bounded projection of caller-owned entity Markdown into non-executable knowledge edges |

Provider routing and reasoning effort do not change Agent identity. The ordinary
Desktop composer selects Harness, Model, and Effort; runtime code resolves the
Provider supply route.

## Install

### Desktop

Download the matching macOS package from
[GitHub Releases](https://github.com/tcztzy/swarmx/releases/latest), or install
the npm launcher:

```shell
npm install --global swarmx
swarmx
```

Package installation does not launch the application. Running `swarmx` without
arguments opens Desktop.

On macOS, the terminal launchers omit the known InputMethodKit
`IMKCFRunLoopWakeUpReliable` system diagnostic. This message is not an
application failure; all other Electron standard-error output remains visible.

In Desktop:

1. Open **Runtime** and inspect the local baseline; optional Harnesses do not
   block the direct default.
2. Open **Local workspace -> Settings -> Providers** and add one explicit
   connection when the selected Model needs it.
3. Refresh its Model catalog, then choose a Harness and Model in the composer.
4. Add a writable Project when the task needs local files or coding tools.
5. Start with a read-only question; Doctor provides an actionable blocker if
   the selected route is not ready.

Desktop stores Provider credentials as plaintext in the editable
`~/.swarmx/provider-auth.json` file. The file is written with restrictive
permissions; credentials are read only in the Main process and never enter the
Renderer.

The file uses a simple `schemaVersion: 2` document with string values under
`entries`. Older encrypted Provider auth files and legacy `local_keychain`
Provider references are intentionally not migrated; configure affected
Providers again using the current format after upgrading.

### CLI

Passing an argument uses the CLI:

```shell
swarmx doctor
swarmx harnesses
swarmx send "Explain this repository" --model <runtime-model-id>
swarmx sessions
swarmx sessions timeline <session-id>
swarmx serve --port 8000
```

The direct CLI may read request credentials from its environment. Start from
`.env.example` for OpenAI-compatible, Anthropic, DeepSeek, or local endpoints.
Desktop Provider discovery remains explicit and does not synthesize connections
from ambient variables.

Use a workflow file with:

```shell
swarmx send --config path/to/workflow.json "Run the workflow"
```

Preview runtime repairs before applying them:

```shell
swarmx doctor --fix
swarmx doctor --fix --yes
```

## TypeScript core

Install `@swarmx/core` and construct a single-Agent swarm:

```ts
import { Swarm } from "@swarmx/core";

const swarm = new Swarm({
  name: "example",
  root: "writer",
  nodes: {
    writer: {
      kind: "agent",
      agent: {
        name: "writer",
        instructions: "Write a concise answer.",
      },
    },
  },
  edges: [],
});

const result = await swarm.execute({
  messages: [{ role: "user", content: "Summarize the project." }],
});
```

API inputs are validated with zod. Prefer the package subpath matching the
feature when one exists, especially for browser bundles:

```ts
import { normalizeMessageChunk } from "@swarmx/core/rendering";
import { resolveTelemetryConfig } from "@swarmx/core/telemetry";
import { parseManagedDependencyManifest } from "@swarmx/core/dependencies";
```

The package manifest is the authoritative list of public subpaths and exported
types.

## Workflow format

`SwarmConfig` is the only persisted workflow format:

```json
{
  "name": "review_then_write",
  "root": "reviewer",
  "nodes": {
    "reviewer": {
      "kind": "agent",
      "agent": {
        "name": "reviewer",
        "instructions": "Identify the important facts."
      }
    },
    "writer": {
      "kind": "agent",
      "agent": {
        "name": "writer",
        "instructions": "Write the final response."
      }
    }
  },
  "edges": [
    {
      "source": "reviewer",
      "target": "writer"
    }
  ]
}
```

Nodes use `kind: agent | tool | swarm` and place their payload under the matching
field. `root` names the first node. Edges describe explicit transitions and may
use CEL conditions.

The n8n importer converts topology and inert metadata into `SwarmConfig`. It
does not import credentials, retain a second runtime DSL, or execute n8n node
implementations.

## Desktop behavior

Desktop uses an isolated Electron Main/Preload/Renderer architecture:

- Main owns filesystem, subprocess, network, credential, update, and persistence
  access.
- Preload exposes a narrow typed bridge.
- Renderer is a React application with no direct Node.js authority.

Direct SwarmX tasks can receive bounded Project tools. Existing files require a
complete read and unchanged digest before mutation. Shell commands run from the
Project with bounded output, cancellation, a sanitized environment, and
platform sandboxing where supported.

External ACP Harnesses own their native tools and permission systems. They do
not receive duplicate SwarmX coding tools.

Sessions are append-only JSONL event logs grouped by working directory under
`~/.swarmx/projects/`, with projectless history in `__recents__`. Each Project
directory has a rebuildable index, so task lists do not load message bodies.
Older `.json` Session files and migration backups are unsupported and are not
discovered, loaded, indexed, or migrated.

Durable WorkItems use a separate append-only authority under
`~/.swarmx/task-runtime/`. Multiple Sessions may observe the same WorkItem;
archiving a conversation does not terminate it. See the
[durable task runtime guide](durable-task-runtime.md) for recovery, worker
protocol, Python environment, and the authenticated detached supervisor.

## Extensions and composition

Extension discovery loads passive metadata for Software, Harnesses, Models,
Providers, Agent profiles, Skills, MCP servers, connectors, language servers,
hooks, commands, assets, policies, and UI contribution references.

Discovery does not execute components or change trust. Installation, update,
rollback, repair, and other side effects use separate explicit actions.

A Custom Agent resolves to exactly one Harness and one Model. Composition
preflight reports missing runtime, Provider, Model, Skill, MCP, context, or
permission requirements before execution. It also resolves the selected
Extension capability graph, load reasons, conflicts, deterministic order,
source/trust/integrity, protected-kernel claims, and explicit permission grants.

See [Extensions and Custom Agents](extensions-custom-agents.md) for manifest,
variant, trust, and persistence details.

## Security boundaries

- Metadata contains secret references and status, not inline credentials.
- Renderer-visible tool and trace payloads are normalized and sanitized.
- Remote Markdown media is blocked by default.
- Managed attachments are copied into a content-addressed local store; Session
  logs keep metadata rather than Base64.
- Telemetry is opt-in and excludes raw conversations, prompts, responses,
  source files, terminal output, and credentials.
- Privileged decisions and side effects use a local tamper-evident audit chain;
  audit metadata is deliberately compact and excludes those same raw contents.
- The optional HTTP server binds to loopback by default; non-loopback binding
  requires a bearer token and explicit Origin policy.

## Packages

| Package | Responsibility |
| --- | --- |
| `swarmx` | Desktop-first npm launcher with CLI compatibility |
| `@swarmx/core` | Agents, workflows, durable task control, ACP/MCP, Sessions, and platform contracts |
| `@swarmx/desktop` | Electron host and reusable renderer shell |
| `@swarmx/cli` | Terminal commands and OpenAI-compatible server |
| `@swarmx/acp-server` | ACP server adapter |
| `@swarmx/runtime` | Harness, Python worker, and managed module runtime detection, Doctor, and repair planning |

## Develop and validate

Requires Node.js 22.13 or newer:

```shell
corepack enable
pnpm install
pnpm --filter @swarmx/desktop dev
```

Run the normal quality checks with:

```shell
pnpm lint
pnpm test
pnpm -r build
```

The canonical Node CI gate is:

```shell
pnpm run ci:node
```

Repository conventions and validation expectations are documented in
[`AGENTS.md`](https://github.com/tcztzy/swarmx/blob/main/AGENTS.md).

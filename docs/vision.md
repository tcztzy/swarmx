# SwarmX Product Vision

Status: product direction, not an implementation contract

The current product contract is
[`SPEC.md`](https://github.com/tcztzy/swarmx/blob/main/SPEC.md). Architecture
lives in
[`DESIGNS.md`](https://github.com/tcztzy/swarmx/blob/main/DESIGNS.md). This
document describes the experience SwarmX is trying to create and is allowed to
be more aspirational than either.

## One sentence

SwarmX is a workspace where people and AI agents can divide, execute, review,
and deliver real work without forcing the user to understand the runtime
machinery underneath.

The product should feel task-first:

> Tell SwarmX what must be delivered, choose or accept the proposed AI
> coworkers, review important decisions, and keep the evidence.

It should not feel like a catalog of models, tools, protocols, and configuration
forms that the user must assemble before doing useful work.

## Primary users

- **Individual contributor:** wants a useful result from local files with
  minimal setup.
- **Reviewer or owner:** needs to understand scope, evidence, risk, and who is
  responsible for a decision.
- **Agent builder:** creates reusable AI coworkers and workflows without
  exposing technical choices to ordinary users.
- **Administrator:** manages Providers, runtime health, Extensions, permissions,
  and trust.

SwarmX remains useful to developers, but ordinary users should not need to know
what ACP, MCP, CEL, Harness, ModelSupply, or a Provider route is.

## North-star experience

A user can ask:

> Summarize this week's customer feedback, identify the three most common
> problems, and prepare a product brief. Ask before quoting customer text.

The interface should make five things obvious:

1. What will be delivered.
2. Which AI coworkers are involved and what each one owns.
3. What is happening now.
4. Which decisions require a person.
5. What was delivered, based on which evidence, under whose responsibility.

## Product principles

### Start with the goal

The primary entry point is the work to accomplish. Harness, Model, Provider,
Skill, MCP, and permission details are secondary explanations or advanced
controls.

### Treat agents as accountable participants

An AI coworker has a name, role, capabilities, limits, and reproducible runtime
identity. The product never implies that an anonymous hidden model made a
decision.

### Keep human authority explicit

People set direction, approve consequential actions, resolve ambiguity, and own
external impact. AI may prepare a decision but must not impersonate the human
who is responsible for it.

### Reveal complexity progressively

Show a concise status and result first. Keep tool calls, runtime provenance,
artifacts, policy sources, and raw evidence available for inspection without
placing them in the main reading path.

### Make control reversible

Users can pause, reject, edit, retry, replace an Agent, or take over. A failure
keeps completed work and explains the safest next action.

### Separate plans from facts

A proposed workflow is not an execution trace. The UI distinguishes what is
planned, what is running, what actually happened, and what remains uncertain.

### Earn trust instead of assuming it

New Providers, Extensions, permissions, data scopes, and external actions begin
with the least authority that can work. Authority never expands silently.

### Be honest about capability

Unavailable parallelism, collaboration, media transport, scheduling, or native
tool behavior is shown as unavailable. SwarmX does not relabel a weaker feature
to make the product appear complete.

## User language and system language

| User-facing concept | System concept |
| --- | --- |
| Project | Local Project bookmark and tool containment root |
| Task | Persisted Session and its execution history |
| AI coworker | Agent profile resolved to one Harness and one Model |
| How it works | Harness recipe, Skills, MCP servers, context, and policy |
| AI service | Provider connection and secret reference |
| Thinking engine | Model |
| Work plan | `SwarmConfig` or a single-Agent execution plan |
| Work record | Session events, normalized trace, artifacts, and receipts |

The system vocabulary remains visible in advanced settings and diagnostics, but
the ordinary workflow uses the left column.

## Core experience loops

### Start useful work

1. Choose or create a Project.
2. Describe the desired result and attach relevant material.
3. Select an AI coworker or accept a compatible default.
4. Review scope and any important permission boundary.
5. Run the task and follow meaningful progress.

The default path should work with one Agent. Multi-Agent workflows are an
extension of that path, not a prerequisite.

### Supervise a task

The task has a small, stable state vocabulary:

- draft
- running
- waiting for a person
- completed
- failed
- canceled

The main timeline reports milestones, decisions, handoffs, and results. Detailed
reasoning and tool traces remain expandable. Live output should not become
permanent transcript noise.

### Review a consequential action

A decision surface states:

- the intended outcome;
- the exact scope and target;
- what will change externally or irreversibly;
- the evidence and Agent that proposed it;
- whether it can be undone;
- the choices available to the person.

Project-local, reversible work may follow the configured permission policy.
External writes, destructive operations, trust changes, and authority expansion
require explicit review appropriate to their risk.

### Configure an AI coworker

An Agent builder chooses a role, one compatible Harness and Model, required
Skills and MCP servers, Project context, and permission policy. SwarmX validates
the composition before it becomes available for normal tasks.

Ordinary users see the resulting coworker and its limits, not every ingredient.

### Recover from failure

A failure explains:

- where work stopped;
- what output is still valid;
- whether any side effect may already have happened;
- whether retry is safe;
- how to provide missing information, change the Agent, or take over.

Retry must not duplicate an uncertain external side effect.

## Product layers

### Current foundation

- Local Projects and resumable task history.
- Direct SwarmX execution and external ACP Harnesses.
- Independent Harness, Model, Provider, and Agent identities.
- Custom Agents, Extension inventory, Skills, MCP servers, and composition
  preflight.
- Bounded Project tools, layered permissions, streaming work, and cancellation.
- Safe Markdown, traces, side chats, and multimedia attachments.
- Provider, runtime, Extension, Custom Agent, permission, and update settings.

### Near-term direction

- A more goal-first new-task flow with technical choices progressively
  disclosed.
- Clearer plan, milestone, decision, handoff, and deliverable presentations.
- Better reusable workflow authoring and import without creating a second
  runtime format.
- Stronger recovery, provenance, and review surfaces around external actions.

### Later, only with validated demand

- Shared workspaces, team identity, invitations, responsibility transfer, and
  organization policy.
- Remote collaboration and hosted knowledge.
- Organization-level autonomous scheduling.
- Cross-task memory with explicit scope, provenance, expiry, review, and
  deletion.

These later capabilities are not promises in the current product specification.

## Success criteria

SwarmX is moving in the right direction when:

- a new user can produce a useful local result without learning the runtime
  taxonomy;
- a reviewer can understand a consequential action without reading a raw tool
  trace;
- a failed task preserves useful work and offers a safe recovery path;
- the same Agent identity and policy can be reproduced and inspected;
- advanced extensibility does not make the default single-Agent path harder;
- the UI never claims that a plan, capability declaration, or cached state is a
  completed fact.

## Non-goals

- Hiding which model, runtime, or authority produced work.
- Treating every integration as a tool or every model as a Provider-owned item.
- Maximizing autonomy at the expense of reviewability.
- Building domain-specific scientific, HPC, legal, or organizational policy
  into the generic SwarmX core.
- Shipping speculative team or cloud concepts before their identity, policy,
  persistence, and failure semantics are real.

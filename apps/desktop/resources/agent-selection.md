# Agent selection knowledge

Use this project's integration facts together with the user's current request and private
experience. These notes describe supported routes, not measured model quality rankings.
Sources are this release's [native integration contract](../../../docs/native-agents.md),
[delegation contract](../../../docs/swarm.md), and executable integration tests in
`apps/desktop/tests/{pi,codex,claude,dsh,hermes,openclaw}-native.test.ts`.
Interpret these facts against the same Git revision as this guide, not an unspecified latest release.

- Pi is the default lead and exposes its available provider/model routes and thinking levels
  through its native catalog. Use the exact advertised IDs.
- Codex and Claude use their official native runtimes. Discover their models and reasoning
  settings through `swarm.models`; do not infer capability or writing quality from a family name.
- DSH accepts an explicit `provider/model` route and an adapter-owned reasoning effort when
  starting an independent execution. It has no SDK model-discovery API. An empty catalog does
  not mean explicit routes are unavailable; the SDK validates the selected route at startup.
- DSH `sdk` is the full SDK composition. `sdk-minimal` is the standalone persistent-shell
  composition. SwarmX adds its product MCP tools to both. A profile changes the tool composition,
  not Host permissions. DSH tasks cannot resume or steer through this integration.
- Hermes and OpenClaw retain their native model/provider configuration and discovered settings.
  Check the candidate's catalog and task capabilities before choosing it.

Choose for the actual task: model version, writing language, tool needs, instruction adherence,
provider deployment, latency and the user's cost preference can all matter. Prefer the user's
explicit choice over general recommendations. Apply private observations only to the route and
conditions they describe; a failing deployment is not evidence against every provider of a model.
Treat stale, anecdotal and conflicting evaluations as qualified evidence, not universal facts.
Explain the choice in `send_message.reason`, including relevant knowledge or memory references.

Save durable harness/model/provider experience as existing Memory concepts tagged `agent-selection`,
with aliases/tags for the route and task, the observed conditions, and sources or execution evidence.
Supply structured `evaluation` fields: kind, task, criteria, original execution-event references,
counterevidence and limitations. Read the Host's evidence status, exact sample scope and computed
statistics before relying on a note. Unknown cost or identity is not zero or an inferred route;
wall time includes tools and waiting. Successful termination alone does not establish correctness.
Distinguish an explicit user preference from an observed result or an unverified opinion. Keep private
provider incidents in user Memory. Project-maintained evaluations belong in this versioned guide;
do not overwrite user concepts when the application updates.
Add a public quality recommendation only with a task/route/version, dated public source or
versioned reproducible experiment, sample scope, criteria and limitations. Do not ship private
execution URNs as public evidence or promote an anecdote into a model-wide ranking.

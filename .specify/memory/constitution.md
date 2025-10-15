<!--
Sync Impact Report
Version change: 1.0.1 → 1.0.2
Modified principles: Human audit
Added sections: None
Removed sections: None
Templates requiring updates: ✅ .specify/templates/plan-template.md; ✅ .specify/templates/spec-template.md; ✅ .specify/templates/tasks-template.md; ✅ .specify/templates/agent-file-template.md
Follow-up TODOs: None
-->
# SwarmX Constitution

## Core Principles

### I. Non-Negotiable Code Quality

- Code MUST remain concise, readable, and explicitly intention-revealing; avoid clever constructs that obscure control flow.
- Every change MUST eliminate unnecessary complexity, remove dead paths, and align with Ruff and mypy findings before review.
Rationale: Predictable, legible code keeps multi-agent orchestration safe for the maintainers who inherit it.

### II. Test-Driven Development (NON-NEGOTIABLE)

- Contributors MUST author failing automated tests with pytest before implementing behavior, covering success, error, and regression paths.
- Code merges only after the new tests pass and the full suite runs green via `uv run pytest`.
- Ask for human contributors' help while cannot resolved by yourself after several trials.
- DO NOT just filter out the failed test cases and make test passed.
Rationale: TDD guards against regressions in distributed agent workflows and sustains long-term contributor confidence.

### III. Document-Driven Delivery (NON-NEGOTIABLE)

- Significant changes MUST begin with updated plan, research, and spec documents in `docs/` hierarchies before writing production code.
- Implementation tasks MUST reference document anchors and keep them in sync with the delivered behavior.
Rationale: Forward documentation synchronizes expectations across maintainers, reviewers, and users.

### IV. Competitive Research Before Implementation

- New features MUST include a recorded survey of comparable agent frameworks or tools, captured in `specs/<feature>/research.md` or docs with clear citations.
- Plans MUST summarize the evaluation conclusions and justify the chosen direction before any coding effort starts.
Rationale: Understanding peer solutions avoids redundant work and accelerates best-practice adoption.

### V. Minor Version Continuity

- Releases within the same minor version MUST preserve public APIs, CLI ergonomics, and user experience; breaking changes require a major bump with a migration plan.
- Deprecations MUST ship with documented fallbacks and regression tests that exercise legacy flows until removal.
Rationale: Predictable upgrades sustain user trust and protect integrations built on SwarmX.

### VI. Token Economy

- Performance MUST be defined as achieving the required behavior with minimal additional wall-clock time and computational overhead on top of raw LLM inference, while sustaining a high agent task success rate across representative scenarios.
- Teams MUST instrument and report both speed metrics (latency, throughput) and outcome metrics (success rate, retry counts) so that performance regressions are detected before release.
- The performance-cost ratio MUST never trade away previously achieved performance; cost reductions (primarily token consumption and related usage fees) are acceptable only when success rates and latency remain at or above baseline.
- When the token or time cost of a workflow renders an otherwise simple task impractical—e.g., multi-minute runs consuming millions of tokens—the team MUST redesign the approach instead of accepting the expense.
Rationale: SwarmX operates on a token economy—deliver the strongest achievable results at an acceptable cost ceiling without eroding speed or success guarantees.

### VII. Language-Neutral Agent Interoperability

- Agents MUST remain fully describable as JSON payloads so that behavior, tools, and routing can be exported, reviewed, and rehydrated by any runtime.
- Execution requiring code MUST be delegated to MCP servers or equivalent tool bridges, enabling alternate language hosts to achieve feature parity by consuming JSON and speaking MCP.
- Changes to the core MUST avoid baking in language-specific assumptions that would prevent interoperable agent definitions or MCP-driven execution flows.
Rationale: A language-agnostic architecture keeps SwarmX portable and inexpensive to adopt across ecosystems.

## Delivery Research Mandate

- Feature kickoffs MUST populate `/specs/<feature>/research.md` with competitive findings, explicit evaluation criteria, and performance benchmarks.
- Each plan derived from `.specify/templates/plan-template.md` MUST cite the research document, articulate the chosen approach, and note open risks before approval.
- Documentation in `/docs/` MUST be updated alongside implementation to reflect the decisions, trade-offs, and usage guidance that emerged from research.

## Operational Workflow

- Development follows the sequence: documentation update → research review → failing tests committed → implementation → validation → release notes.
- Pull requests MUST demonstrate passing tests, updated docs, and evidence that principles I–VII remain satisfied (e.g., links to benchmarks or compatibility checks).
- Release candidates MUST include a compatibility verification checklist covering CLI flows, API responses, and user experience touchpoints.
- Contributors MUST break work into the smallest user-observable or process-observable increments and commit as soon as each increment passes local validation.
- Every commit MUST focus on a single intent—tests, documentation sync, configuration, or code fix—and reference the corresponding task or evidence in the compliance audit.
- Mixing unrelated changes in one commit is prohibited unless a maintainer grants explicit written approval and follow-up remediation is scheduled.

## Governance

- This constitution supersedes conflicting process documents; deviations require a written exception approved by the maintainers and recorded with rationale.
- Amendments require documented proposals, review by maintainers, impact assessment on existing users, and, when adopted, version increments per semantic rules (major for incompatible shifts, minor for new mandates, patch for clarifications).
- Compliance reviews occur each release cycle to verify documentation freshness, research logs, test coverage, and compatibility commitments; unresolved findings block release.

**Version**: 1.0.2 | **Ratified**: 2025-10-13 | **Last Amended**: 2025-10-15

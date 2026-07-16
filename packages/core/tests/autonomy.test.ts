import { describe, expect, it } from "vitest";
import {
  AUTONOMY_PATHS,
  autonomyDatedPath,
  canTransitionWorkItemStatus,
  createAutonomyAgentRunRuntimeEvent,
  createAutonomyReplayRecord,
  createAutonomyRuntimeEvent,
  createAutonomyScheduleTrigger,
  createAutonomyTransitionRuntimeEvent,
  createAutonomyWorkflowDecisionRuntimeEvent,
  defaultReportSchedule,
  engineeringLifecycleWorkflowState,
  evaluateAutonomySchedule,
  evaluateAutonomyTransition,
  evaluateCircuitBreaker,
  evaluateEngineeringLifecycleTransition,
  evaluateValidatorGate,
  linkAgentStageToWorkflowState,
  parseAutonomyAgentRunRecord,
  parseAutonomyDaemonRunMetadata,
  parseAutonomyDashboardMetadata,
  parseAutonomyFeedbackRecord,
  parseAutonomyReplayRecord,
  parseAutonomyReportMetadata,
  parseAutonomyTransitionDecision,
  parseAutonomyTriggerRecord,
  parseAutonomyWakeupState,
  parseAutonomyWorkItem,
  parseAutonomyWorkflowDecisionRecord,
  parseCommandDag,
  parseEngineeringIntakeRecord,
  parseEngineeringLifecycleTransitionDecision,
  parseEngineeringProposalRecord,
  parseEvidencePacket,
  parseValidatorGateDecision,
  parseValidatorManifest,
  replayAutonomyEvents,
  runtimeEventFromTrigger,
  wakeupStatePath,
} from "../src/autonomy.js";

const timestamp = "2026-07-03T00:00:00.000Z";

function workItem(overrides: Record<string, unknown> = {}) {
  return {
    id: "awi_platform_foundation",
    class: "project_iteration",
    type: "spec_review",
    status: "queued",
    priority: 10,
    autonomyLevel: "A2",
    sourceRefs: [{ kind: "spec", id: "0703" }],
    nextAction: "draft a proposal",
    requiredEvidence: ["validator_manifest"],
    ...overrides,
  };
}

describe("autonomy runtime primitives", () => {
  it("parses work items and replays idempotent creation events once", () => {
    const parsed = parseAutonomyWorkItem(workItem());
    expect(parsed.retry.count).toBe(0);
    expect(parsed.sourceRefs).toEqual([{ kind: "spec", id: "0703" }]);

    const createEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:awi_platform_foundation",
      payload: { workItem: parsed },
    });
    const duplicateCreateEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:awi_platform_foundation",
      payload: { workItem: parsed },
    });

    expect(duplicateCreateEvent.eventId).toBe(createEvent.eventId);

    const state = replayAutonomyEvents([createEvent, duplicateCreateEvent]);
    expect(Object.keys(state.workItems)).toEqual(["awi_platform_foundation"]);
    expect(state.events.map((event) => event.eventId)).toEqual([createEvent.eventId]);
    expect(state.workItems.awi_platform_foundation?.status).toBe("queued");
  });

  it("converts typed triggers into deterministic runtime events without creating work", () => {
    const trigger = parseAutonomyTriggerRecord({
      triggerId: "trg_manual_platform",
      triggerType: "manual_request",
      timestamp,
      source: "desktop",
      idempotencyKey: "manual:platform-foundation",
      workItemId: "awi_platform_foundation",
      sourceRefs: [{ kind: "spec", id: "0703" }],
      payload: { summary: "Review next platform runtime contract." },
    });

    const event = runtimeEventFromTrigger(trigger);
    const duplicate = runtimeEventFromTrigger(trigger);
    expect(duplicate.eventId).toBe(event.eventId);
    expect(event).toMatchObject({
      eventType: "manual_request",
      source: "desktop",
      idempotencyKey: "manual:platform-foundation",
      workItemId: "awi_platform_foundation",
      payload: {
        triggerId: "trg_manual_platform",
        triggerType: "manual_request",
        sourceRefs: [{ kind: "spec", id: "0703" }],
      },
    });

    const state = replayAutonomyEvents([event, duplicate]);
    expect(state.events).toHaveLength(1);
    expect(state.workItems).toEqual({});

    expect(() =>
      parseAutonomyTriggerRecord({
        ...trigger,
        triggerId: "trg_bad_raw",
        payload: { rawIssueBody: "full copied issue body" },
      }),
    ).toThrow(/raw content field/);
  });

  it("validates lifecycle intake and proposal records without accepting specs", () => {
    const intake = parseEngineeringIntakeRecord({
      intakeId: "int_bug_1",
      sourceId: "issue-42",
      sourceType: "bug_report",
      receivedAt: timestamp,
      title: "Fix failed validator",
      summary: "Validation failed on the autonomy runtime contract.",
      sourceRefs: [{ kind: "spec", id: "0100" }],
      validationId: "unit-tests",
      autonomyLevel: "A2",
      risk: "medium",
    });
    expect(intake.initialState).toBe("intake");

    const proposal = parseEngineeringProposalRecord({
      proposalId: "prp_runtime_gate",
      workItemId: "awi_platform_foundation",
      status: "discussion",
      problem: "Lifecycle transitions need deterministic gates.",
      affectedSurfaces: ["packages/core/src/autonomy.ts", "docs/index.md"],
      alternatives: ["Keep free-form workflow.stage", "Add typed lifecycle contracts"],
      compatibility: "Additive schema only.",
      securityPrivacy: "No secrets in lifecycle records.",
      validationPlan: ["pnpm --filter @swarmx/core test -- autonomy"],
      migrationRollback: "Remove additive schema exports.",
      approvals: [
        {
          approvalId: "apr_spec_owner",
          status: "approved",
          actor: "maintainer",
          decidedAt: timestamp,
        },
      ],
    });
    expect(proposal.status).toBe("discussion");
    expect(proposal.validationPlan).toEqual(["pnpm --filter @swarmx/core test -- autonomy"]);
    expect(JSON.stringify(proposal)).not.toContain("accepted spec");

    expect(() =>
      parseEngineeringProposalRecord({
        ...proposal,
        proposalId: "prp_bad_secret",
        apiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("applies valid deterministic transitions and rejects invalid ones without moving state", () => {
    const createEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:create",
      payload: { workItem: workItem() },
    });
    const invalidTransition = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "transition:invalid",
      workItemId: "awi_platform_foundation",
      previousState: "queued",
      nextState: "succeeded",
      payload: {},
    });

    const invalidState = replayAutonomyEvents([createEvent, invalidTransition]);
    expect(invalidState.workItems.awi_platform_foundation?.status).toBe("queued");
    expect(invalidState.rejectedEvents).toHaveLength(1);
    expect(invalidState.rejectedEvents[0]?.payload.reason).toMatch(/Invalid work item transition/);

    const lease = {
      leaseId: "lease_1",
      runId: "run_20260703",
      actor: "runner",
      startedAt: timestamp,
      expiresAt: "2026-07-03T01:00:00.000Z",
    };
    const leased = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "transition:leased",
      workItemId: "awi_platform_foundation",
      previousState: "queued",
      nextState: "leased",
      payload: { patch: { lease } },
    });
    const running = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "transition:running",
      workItemId: "awi_platform_foundation",
      previousState: "leased",
      nextState: "running",
      payload: {},
    });
    const succeeded = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "transition:succeeded",
      workItemId: "awi_platform_foundation",
      previousState: "running",
      nextState: "succeeded",
      payload: {},
    });

    const validState = replayAutonomyEvents([createEvent, leased, running, succeeded]);
    expect(validState.rejectedEvents).toEqual([]);
    expect(validState.workItems.awi_platform_foundation?.status).toBe("succeeded");
    expect(validState.workItems.awi_platform_foundation?.lease).toBeUndefined();
    expect(canTransitionWorkItemStatus("queued", "leased")).toBe(true);
    expect(canTransitionWorkItemStatus("queued", "succeeded")).toBe(false);
  });

  it("evaluates engineering lifecycle transitions with evidence, approvals, and validator gates", () => {
    const work = workItem({
      workflow: engineeringLifecycleWorkflowState("proposal", ["agt_1"]),
    });
    const gate = evaluateValidatorGate(
      {
        manifestId: "val_lifecycle_gate",
        name: "Lifecycle gate",
        validators: [{ id: "spec-check", kind: "policy", required: true }],
        gates: { acceptance: ["spec-check"] },
      },
      [{ validatorId: "spec-check", status: "passed" }],
      "acceptance",
    );
    const allowed = evaluateEngineeringLifecycleTransition({
      workItem: work,
      to: "discussion",
      requiredEvidenceIds: ["evp_proposal"],
      presentEvidenceIds: ["evp_proposal"],
      requiredApprovalIds: ["apr_spec_owner"],
      approvals: [{ approvalId: "apr_spec_owner", status: "approved", actor: "maintainer" }],
      validatorGate: gate,
    });
    expect(allowed).toMatchObject({
      status: "allowed",
      from: "proposal",
      to: "discussion",
      currentStateMatched: true,
      validEdge: true,
      missingEvidenceIds: [],
      missingApprovalIds: [],
      nextWorkflow: { kind: "engineering", stage: "discussion", agentRunIds: ["agt_1"] },
    });
    expect(parseEngineeringLifecycleTransitionDecision(allowed).decisionId).toMatch(/^dec_/);

    const missing = evaluateEngineeringLifecycleTransition({
      workItem: work,
      to: "specification",
      requiredEvidenceIds: ["evp_proposal"],
      presentEvidenceIds: [],
      requiredApprovalIds: ["apr_spec_owner"],
      approvals: [],
      validatorGate: { ...gate, status: "failed", failedValidatorIds: ["spec-check"] },
    });
    expect(missing).toMatchObject({
      status: "rejected",
      from: "proposal",
      to: "specification",
      missingEvidenceIds: ["evp_proposal"],
      missingApprovalIds: ["apr_spec_owner"],
    });
    expect(missing.reason).toMatch(/Missing evidence/);
    expect(missing.reason).toMatch(/Missing approval/);
    expect(missing.reason).toMatch(/Validator gate acceptance is failed/);
    expect(missing.nextWorkflow).toBeUndefined();

    const invalidEdge = evaluateEngineeringLifecycleTransition({
      currentState: "intake",
      to: "close",
      workItemId: "awi_platform_foundation",
    });
    expect(invalidEdge).toMatchObject({
      status: "rejected",
      validEdge: false,
      currentStateMatched: true,
    });
  });

  it("records agent-stage runs and workflow decisions without owning execution", () => {
    const agentRun = parseAutonomyAgentRunRecord({
      agentRunId: "agt_review_1",
      workItemId: "awi_platform_foundation",
      runId: "run_20260703",
      workflowKind: "engineering",
      stage: "triage",
      role: "agent.bug-reviewer",
      status: "succeeded",
      harnessId: "codex",
      modelId: "gpt-5",
      modelSupplyId: "openai-gpt-5",
      adapter: "codex",
      agentProfileId: "analysis-lead",
      startedAt: timestamp,
      endedAt: "2026-07-03T00:01:00.000Z",
      durationMs: 60_000,
      outputRefs: [{ kind: "artifact", id: "run-log", title: "agent summary" }],
      artifactIds: ["art_agent_summary"],
      evidenceIds: ["evp_review"],
      summary: "Confirmed the report and selected impact assessment.",
      resultRef: "autonomy/agent-runs/2026/07/03/agt_review_1.json",
    });
    expect(agentRun.agentRunId).toBe("agt_review_1");
    expect(agentRun).toMatchObject({ harnessId: "codex", modelId: "gpt-5" });
    expect(agentRun.outputRefs).toEqual([
      { kind: "artifact", id: "run-log", title: "agent summary" },
    ]);

    const decision = parseAutonomyWorkflowDecisionRecord({
      decisionId: "dec_triage_continue",
      workItemId: "awi_platform_foundation",
      runId: "run_20260703",
      workflowKind: "engineering",
      currentStage: "triage",
      nextStage: "proposal",
      status: "accepted",
      decisionKind: "continue",
      agentRunIds: ["agt_review_1"],
      evidenceIds: ["evp_review"],
      reason: "Triage evidence is sufficient for proposal drafting.",
      nextWorkflow: {
        kind: "engineering",
        stage: "proposal",
        agentRunIds: ["agt_review_1"],
        decisionId: "dec_triage_continue",
      },
    });
    expect(decision.nextWorkflow?.stage).toBe("proposal");

    const agentRunEvent = createAutonomyAgentRunRuntimeEvent({ agentRun, timestamp });
    const duplicateAgentRunEvent = createAutonomyAgentRunRuntimeEvent({ agentRun, timestamp });
    expect(duplicateAgentRunEvent.eventId).toBe(agentRunEvent.eventId);
    expect(agentRunEvent).toMatchObject({
      eventType: "agent_run_written",
      workItemId: "awi_platform_foundation",
      runId: "run_20260703",
      payload: {
        agentRunId: "agt_review_1",
        role: "agent.bug-reviewer",
        artifactIds: ["art_agent_summary"],
        evidenceIds: ["evp_review"],
        resultRef: "autonomy/agent-runs/2026/07/03/agt_review_1.json",
      },
    });

    const decisionEvent = createAutonomyWorkflowDecisionRuntimeEvent({ decision, timestamp });
    expect(decisionEvent).toMatchObject({
      eventType: "workflow_decision_recorded",
      payload: {
        decisionId: "dec_triage_continue",
        decisionKind: "continue",
        nextStage: "proposal",
        agentRunIds: ["agt_review_1"],
      },
    });

    const linked = linkAgentStageToWorkflowState(
      engineeringLifecycleWorkflowState("triage"),
      agentRun,
      decision,
    );
    expect(linked).toEqual({
      kind: "engineering",
      stage: "proposal",
      agentRunIds: ["agt_review_1"],
      decisionId: "dec_triage_continue",
    });

    const createEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:stage-create",
      payload: { workItem: workItem({ workflow: engineeringLifecycleWorkflowState("triage") }) },
    });
    const linkEvent = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "workflow:stage-link",
      workItemId: "awi_platform_foundation",
      previousState: "queued",
      nextState: "leased",
      payload: { patch: { workflow: linked } },
    });
    const replayed = replayAutonomyEvents([createEvent, agentRunEvent, decisionEvent, linkEvent]);
    expect(replayed.workItems.awi_platform_foundation?.workflow).toEqual(linked);
    expect(replayed.events.map((event) => event.eventType)).toEqual([
      "work_item_created",
      "agent_run_written",
      "workflow_decision_recorded",
      "deterministic_transition",
    ]);

    expect(() =>
      parseAutonomyAgentRunRecord({
        ...agentRun,
        agentRunId: "agt_bad_raw",
        rawResponse: "full model transcript",
      }),
    ).toThrow(/raw content field.*rawResponse/);
    expect(() =>
      parseAutonomyWorkflowDecisionRecord({
        ...decision,
        decisionId: "dec_bad_secret",
        apiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*apiKey/);
    expect(() =>
      linkAgentStageToWorkflowState({ kind: "analysis", stage: "triage" }, agentRun),
    ).toThrow(/does not match agent run/);
  });

  it("evaluates transition decisions before emitting transition events", () => {
    const createEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:create-decision",
      payload: { workItem: workItem() },
    });
    const baseState = replayAutonomyEvents([createEvent]);
    const lease = {
      leaseId: "lease_decision",
      runId: "run_20260703",
      actor: "runner",
      startedAt: timestamp,
      expiresAt: "2026-07-03T01:00:00.000Z",
    };
    const decision = evaluateAutonomyTransition({
      workItem: baseState.workItems.awi_platform_foundation,
      to: "leased",
      runId: "run_20260703",
      idempotencyKey: "transition:decision:leased",
      appliedIdempotencyKeys: baseState.appliedIdempotencyKeys,
      preconditions: [
        { kind: "budget", status: "passed", message: "Budget available." },
        { kind: "validator", id: "val_preflight", status: "passed" },
      ],
      patch: { lease },
    });

    expect(decision).toMatchObject({
      status: "allowed",
      from: "queued",
      to: "leased",
      idempotencyStatus: "new",
      missingRequirements: [],
    });
    expect(parseAutonomyTransitionDecision(decision).decisionId).toMatch(/^dec_/);

    const transitionEvent = createAutonomyTransitionRuntimeEvent({ decision, timestamp });
    expect(transitionEvent).toMatchObject({
      eventType: "deterministic_transition",
      previousState: "queued",
      nextState: "leased",
      payload: { decisionId: decision.decisionId },
    });
    const leasedState = replayAutonomyEvents([createEvent, transitionEvent]);
    expect(leasedState.workItems.awi_platform_foundation?.status).toBe("leased");

    const duplicateDecision = evaluateAutonomyTransition({
      workItem: baseState.workItems.awi_platform_foundation,
      to: "leased",
      idempotencyKey: "transition:decision:leased",
      appliedIdempotencyKeys: ["transition:decision:leased"],
    });
    expect(duplicateDecision).toMatchObject({
      status: "rejected",
      idempotencyStatus: "duplicate",
    });
    expect(duplicateDecision.missingRequirements.join("; ")).toMatch(/already applied/);

    const rejectedEvent = createAutonomyTransitionRuntimeEvent({
      decision: duplicateDecision,
      timestamp,
    });
    expect(rejectedEvent.eventType).toBe("state_transition_rejected");
    expect(
      replayAutonomyEvents([createEvent, rejectedEvent]).workItems.awi_platform_foundation?.status,
    ).toBe("queued");
  });

  it("validates command DAG dependencies, node shape, and cycles", () => {
    const dag = parseCommandDag({
      dagId: "dag_analysis_refresh",
      nodes: [
        {
          nodeId: "prepare",
          operationId: "analysis.prepare",
          outputs: [{ kind: "derived_intermediate", path: "artifacts/prepared.json" }],
        },
        {
          nodeId: "summarize",
          command: { program: "python", args: ["scripts/summarize.py"] },
          dependencies: ["prepare"],
          validators: ["schema"],
          artifactPolicy: { overwrite: "if_authorized" },
        },
      ],
    });

    expect(dag.nodes[1]?.dependencies).toEqual(["prepare"]);
    expect(() =>
      parseCommandDag({
        dagId: "dag_bad_missing_dependency",
        nodes: [{ nodeId: "summarize", operationId: "summarize", dependencies: ["missing"] }],
      }),
    ).toThrow(/Unknown command DAG dependency/);
    expect(() =>
      parseCommandDag({
        dagId: "dag_bad_cycle",
        nodes: [
          { nodeId: "a", operationId: "a", dependencies: ["b"] },
          { nodeId: "b", operationId: "b", dependencies: ["a"] },
        ],
      }),
    ).toThrow(/acyclic/);
    expect(() =>
      parseCommandDag({
        dagId: "dag_bad_node",
        nodes: [{ nodeId: "empty" }],
      }),
    ).toThrow(/either command or operationId/);
  });

  it("parses validator manifests and evidence packets while rejecting inline secrets", () => {
    const manifest = parseValidatorManifest({
      manifestId: "val_analysis_gate",
      name: "analysis gate",
      validators: [
        {
          id: "unit-tests",
          kind: "command",
          command: { program: "pnpm", args: ["--filter", "@swarmx/core", "test"] },
        },
        { id: "approval", kind: "manual_approval", waiverAllowed: true },
      ],
      gates: { completion: ["unit-tests", "approval"] },
    });
    expect(manifest.gates.completion).toEqual(["unit-tests", "approval"]);

    const packet = parseEvidencePacket({
      evidenceId: "evp_analysis_refresh",
      workItemId: "awi_platform_foundation",
      runId: "run_20260703",
      workspace: { gitCommit: "abc123" },
      inputs: [{ kind: "raw_input", path: "data/input.json", checksum: "sha256:input" }],
      commands: [
        {
          nodeId: "summarize",
          command: { program: "python", args: ["scripts/summarize.py"] },
          status: "succeeded",
        },
      ],
      parameters: { seed: 7 },
      environment: { node: "22.x" },
      validation: [{ validatorId: "unit-tests", status: "passed" }],
      observations: ["refresh completed"],
      conclusions: ["ready for human review"],
    });
    expect(packet.confidence).toBe("review_required");

    expect(() =>
      parseEvidencePacket({
        evidenceId: "evp_bad_secret",
        workItemId: "awi_platform_foundation",
        runId: "run_20260703",
        workspace: {},
        environment: { apiKey: "sk-test" },
      }),
    ).toThrow(/inline secret field.*apiKey/);
  });

  it("evaluates validator gates from manifests and outcomes without copying raw output", () => {
    const manifest = parseValidatorManifest({
      manifestId: "val_analysis_gate",
      name: "analysis gate",
      validators: [
        { id: "unit-tests", kind: "command", required: true },
        { id: "schema", kind: "schema", required: true },
        { id: "approval", kind: "manual_approval", required: true, waiverAllowed: true },
      ],
      gates: { completion: ["unit-tests", "schema", "approval"] },
    });

    const passed = evaluateValidatorGate(
      manifest,
      [
        { validatorId: "unit-tests", status: "passed", output: "hidden from decision" },
        { validatorId: "schema", status: "passed" },
        { validatorId: "approval", status: "waived", waiverReason: "Maintainer approved." },
      ],
      "completion",
    );
    expect(passed).toMatchObject({
      manifestId: "val_analysis_gate",
      gateId: "completion",
      status: "passed",
      requiredValidatorIds: ["unit-tests", "schema", "approval"],
      passedValidatorIds: ["unit-tests", "schema"],
      waivedValidatorIds: ["approval"],
    });
    expect(JSON.stringify(passed)).not.toContain("hidden from decision");

    const blocked = evaluateValidatorGate(
      manifest,
      [{ validatorId: "unit-tests", status: "passed" }],
      "completion",
    );
    expect(blocked).toMatchObject({
      status: "blocked",
      missingValidatorIds: ["schema", "approval"],
    });

    const skipped = evaluateValidatorGate(
      manifest,
      [
        { validatorId: "unit-tests", status: "passed" },
        { validatorId: "schema", status: "skipped" },
        { validatorId: "approval", status: "waived" },
      ],
      "completion",
    );
    expect(skipped).toMatchObject({
      status: "failed",
      skippedValidatorIds: ["schema"],
    });
    expect(parseValidatorGateDecision(passed).decisionId).toMatch(/^dec_/);
    expect(() =>
      parseValidatorGateDecision({
        ...passed,
        rawValidatorOutput: "full validator log",
      }),
    ).toThrow(/raw content field/);
  });

  it("derives portable autonomy ledger paths from ids and timestamps", () => {
    expect(AUTONOMY_PATHS.runtimeEvents).toBe("autonomy/runtime/events.jsonl");
    expect(autonomyDatedPath("run", "run_20260703", timestamp)).toBe(
      "autonomy/runs/2026/07/03/run_20260703.jsonl",
    );
    expect(autonomyDatedPath("evidence", "evp_analysis_refresh", timestamp)).toBe(
      "autonomy/evidence/2026/07/03/evp_analysis_refresh.json",
    );
    expect(() => autonomyDatedPath("evidence", "bad", timestamp)).toThrow(/evp_/);
  });

  it("summarizes replay records with deterministic state hashes and missing external refs", () => {
    const createEvent = createAutonomyRuntimeEvent({
      eventType: "work_item_created",
      timestamp,
      source: "manual_request",
      idempotencyKey: "manual:create-replay",
      payload: { workItem: workItem() },
    });
    const invalidTransition = createAutonomyRuntimeEvent({
      eventType: "deterministic_transition",
      timestamp,
      source: "swarmx.runtime",
      idempotencyKey: "transition:replay:invalid",
      workItemId: "awi_platform_foundation",
      previousState: "queued",
      nextState: "succeeded",
    });
    const state = replayAutonomyEvents([createEvent, invalidTransition]);
    const replay = createAutonomyReplayRecord(state, {
      recordedAt: timestamp,
      missingExternalRefs: [{ kind: "artifact", id: "remote-dashboard" }],
    });
    const duplicate = createAutonomyReplayRecord(state, {
      recordedAt: timestamp,
      missingExternalRefs: [{ kind: "artifact", id: "remote-dashboard" }],
    });

    expect(duplicate.replayId).toBe(replay.replayId);
    expect(replay).toMatchObject({
      eventLogPath: AUTONOMY_PATHS.runtimeEvents,
      eventCount: 2,
      appliedEventIds: [createEvent.eventId, invalidTransition.eventId].sort(),
      workItemIds: ["awi_platform_foundation"],
      workItemStatusCounts: { queued: 1 },
      missingExternalRefs: [{ kind: "artifact", id: "remote-dashboard" }],
    });
    expect(replay.rejectedEventIds[0]).toMatch(/^evt_/);
    expect(replay.stateHash).toMatch(/^sha256:/);
    expect(parseAutonomyReplayRecord(replay).replayId).toBe(replay.replayId);
    expect(() =>
      parseAutonomyReplayRecord({
        ...replay,
        terminalOutput: "full terminal transcript",
      }),
    ).toThrow(/raw content field/);
  });

  it("evaluates report schedules and creates deterministic schedule triggers", () => {
    const schedule = defaultReportSchedule();
    expect(schedule.kind).toBe("report");
    expect(schedule.cadence.everySeconds).toBe(24 * 60 * 60);

    const lastTriggeredAt = "2026-07-02T00:00:00.000Z";
    const notDue = evaluateAutonomySchedule(
      { ...schedule, lastTriggeredAt },
      "2026-07-02T23:59:59.000Z",
    );
    expect(notDue).toMatchObject({
      scheduleId: "daily-report",
      due: false,
      nextDueAt: "2026-07-03T00:00:00.000Z",
    });

    const due = evaluateAutonomySchedule(
      { ...schedule, lastTriggeredAt },
      "2026-07-03T00:00:00.000Z",
    );
    expect(due).toMatchObject({
      scheduleId: "daily-report",
      due: true,
      dueAt: "2026-07-03T00:00:00.000Z",
      nextDueAt: "2026-07-04T00:00:00.000Z",
      idempotencyKey: "schedule:daily-report:2026-07-03T00:00:00.000Z",
    });

    const trigger = createAutonomyScheduleTrigger(
      { ...schedule, lastTriggeredAt },
      "2026-07-03T00:00:00.000Z",
    );
    expect(trigger).toMatchObject({
      triggerType: "schedule_tick",
      scheduleId: "daily-report",
      dueAt: "2026-07-03T00:00:00.000Z",
      emittedAt: "2026-07-03T00:00:00.000Z",
    });
    expect(trigger.triggerId).toMatch(/^trg_/);
    expect(() =>
      createAutonomyScheduleTrigger({ ...schedule, lastTriggeredAt }, "2026-07-02T23:59:59.000Z"),
    ).toThrow(/not due/);
  });

  it("validates report metadata, dashboard metadata, and human prompt limits", () => {
    const report = parseAutonomyReportMetadata({
      reportId: "rpt_daily",
      period: {
        startedAt: "2026-07-02T00:00:00.000Z",
        endedAt: "2026-07-03T00:00:00.000Z",
      },
      runIds: ["run_20260703"],
      attemptedWorkItemIds: ["awi_platform_foundation"],
      completedWorkItemIds: ["awi_platform_foundation"],
      evidenceIds: ["evp_analysis_refresh"],
      artifactIds: ["art_daily"],
      verification: [{ id: "unit-tests", status: "passed", evidenceIds: ["evp_analysis_refresh"] }],
      budget: { used: { retries: 1 }, remaining: { retries: 2 } },
      decisions: ["continue"],
      risks: ["stale assumption"],
      nextWork: ["inspect next spec"],
      humanPrompts: [
        { promptId: "approve-next", question: "Approve next run?", actions: ["approve"] },
        { promptId: "redirect", question: "Redirect work?", actions: ["redirect"] },
        { promptId: "stop", question: "Stop work?", actions: ["stop"] },
      ],
    });
    expect(report.humanPrompts).toHaveLength(3);

    expect(() =>
      parseAutonomyReportMetadata({
        ...report,
        humanPrompts: [
          { promptId: "p1", question: "One?", actions: ["accept"] },
          { promptId: "p2", question: "Two?", actions: ["reject"] },
          { promptId: "p3", question: "Three?", actions: ["redirect"] },
          { promptId: "p4", question: "Four?", actions: ["approve"] },
        ],
      }),
    ).toThrow(/at most 3/);

    const dashboard = parseAutonomyDashboardMetadata({
      artifactId: "art_daily",
      kind: "live_local",
      path: "autonomy/artifacts/latest.html",
    });
    expect(dashboard.refreshSeconds).toBe(60);
    expect(dashboard.authoritative).toBe(false);

    expect(() =>
      parseAutonomyDashboardMetadata({
        artifactId: "art_bad",
        kind: "report_artifact",
        path: "autonomy/artifacts/2026/07/03/art_bad.html",
        externalResources: ["https://cdn.example/app.js"],
      }),
    ).toThrow(/external resources/);
  });

  it("validates feedback, wakeup, daemon metadata, and circuit breaker decisions", () => {
    const feedback = parseAutonomyFeedbackRecord({
      feedbackId: "fbk_redirect",
      reportId: "rpt_daily",
      action: "redirect",
      timestamp,
      targetRefs: [{ kind: "work_item", id: "awi_platform_foundation" }],
      route: { kind: "work_item", targetId: "awi_followup" },
    });
    expect(feedback.route.kind).toBe("work_item");

    const wakeup = parseAutonomyWakeupState({
      role: "server",
      adapter: "systemd",
      desiredCadenceSeconds: 3600,
      installed: true,
      running: true,
      nextDueAt: "2026-07-03T01:00:00.000Z",
    });
    expect(wakeup.tickLockPath).toBe(AUTONOMY_PATHS.tickLock);
    expect(wakeupStatePath("server")).toBe(AUTONOMY_PATHS.serverWakeup);
    expect(wakeupStatePath("app")).toBe(AUTONOMY_PATHS.wakeup);

    const run = parseAutonomyDaemonRunMetadata({
      runId: "run_1",
      source: "github",
      sourceKey: "issues:1",
      title: "Fix scheduler bug",
      kind: "bug",
      status: "queued",
      labels: ["geepilot"],
      selectedAdapter: "codex",
    });
    expect(run.attemptCount).toBe(0);

    const allowed = evaluateCircuitBreaker({
      workItemId: "awi_platform_foundation",
      consecutiveFailures: 2,
      failureSignature: "same-error",
    });
    expect(allowed.status).toBe("allow");
    const tripped = evaluateCircuitBreaker({
      workItemId: "awi_platform_foundation",
      consecutiveFailures: 3,
      failureSignature: "same-error",
    });
    expect(tripped).toMatchObject({
      status: "needs_human",
      nextStatus: "needs_human",
      maxFailures: 3,
    });
  });

  it("rejects inline secrets in scheduler and reporting records", () => {
    expect(() =>
      parseAutonomyFeedbackRecord({
        feedbackId: "fbk_bad",
        reportId: "rpt_daily",
        action: "accept",
        timestamp,
        apiKey: "sk-test",
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(() =>
      parseAutonomyWakeupState({
        role: "app",
        adapter: "desktop_timer",
        desiredCadenceSeconds: 3600,
        metadata: { password: "cluster-password" },
      }),
    ).toThrow(/inline secret field.*password/);
  });
});

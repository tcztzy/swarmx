import { describe, expect, it } from "vitest";
import {
  projectSessionTimeline,
  type SessionTimelineAuditRecord,
  SessionTimelineSchema,
  type SessionTimelineSource,
  type SessionTimelineSourceRecord,
} from "../src/session-timeline.js";
import type { MessageChunk } from "../src/types.js";

const SESSION_ID = "session-1";
const PROJECT_ID = "project-1";

describe("Session causal timeline", () => {
  it("projects a normal single-step Turn without retaining message content", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-1", [message("user", "private user prompt")]),
        appended(3, "request-1", [message("assistant", "private model response")]),
      ]),
    );

    expect(timeline.turns).toHaveLength(1);
    expect(timeline.turns[0]).toMatchObject({ status: "completed", inferred: false });
    expect(timeline.steps).toMatchObject([{ kind: "model", status: "succeeded" }]);
    expect(timeline.events.map((event) => event.type)).toEqual(["turn_started", "model_response"]);
    expect(JSON.stringify(timeline)).not.toContain("private user prompt");
    expect(JSON.stringify(timeline)).not.toContain("private model response");
    expect(SessionTimelineSchema.parse(timeline)).toEqual(timeline);
  });

  it("opens and settles one Turn from durable request receipts", () => {
    const timeline = projectSessionTimeline(
      source([
        {
          ...appended(2, "request-durable", [message("user", "run")]),
          requestDigest: "sha256:durable",
          requestState: "started",
        },
        {
          ...appended(3, "request-durable", [message("assistant", "done")]),
          requestDigest: "sha256:durable",
          requestState: "settled",
          requestOutcome: "completed",
        },
      ]),
    );

    expect(timeline.turns).toHaveLength(1);
    expect(timeline.turns[0]).toMatchObject({ status: "completed", inferred: false });
    expect(timeline.events.filter((event) => event.type === "turn_started")).toHaveLength(1);
    expect(timeline.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "turn_state",
          outcome: "completed",
          summary: "The request reached its durable terminal outcome.",
        }),
      ]),
    );
    expect(timeline.unsettled).toEqual([]);
  });

  it("projects a background activation as its own system-origin Turn", () => {
    const timeline = projectSessionTimeline(
      source([appended(2, "foreground", [message("user", "foreground")])]),
      [
        audit(1, {
          category: "session",
          action: "session.activation.started",
          outcome: "attempted",
          activationId: "activation_1",
        }),
        audit(2, {
          category: "session",
          action: "session.activation.bootstrap",
          outcome: "completed",
          activationId: "activation_1",
        }),
        audit(3, {
          category: "session",
          action: "session.activation.result",
          outcome: "completed",
          activationId: "activation_1",
        }),
      ],
    );

    expect(timeline.turns).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ origin: "user", correlationId: "foreground" }),
        expect.objectContaining({
          origin: "system",
          correlationId: "activation_1",
          status: "completed",
        }),
      ]),
    );
    expect(timeline.events.filter((event) => event.activationId === "activation_1")).toEqual(
      expect.arrayContaining([expect.objectContaining({ actor: { kind: "system" } })]),
    );
    expect(timeline.turns.filter((turn) => turn.origin === "user")).toHaveLength(1);
  });

  it("keeps activation-observed Session messages on the system Turn", () => {
    const timeline = projectSessionTimeline(
      source([
        {
          ...appended(2, "foreground", [message("user", "foreground")]),
          requestState: "started",
        },
        {
          ...appended(3, "foreground", [message("assistant", "foreground done")]),
          requestState: "settled",
          requestOutcome: "completed",
        },
        {
          ...appended(4, undefined, [message("system", "background prompt")]),
          activationId: "activation_1",
        },
        {
          ...appended(5, undefined, [message("assistant", "background result")]),
          activationId: "activation_1",
        },
      ]),
      [
        audit(1, {
          category: "session",
          action: "session.activation.started",
          outcome: "attempted",
          activationId: "activation_1",
        }),
        audit(2, {
          category: "session",
          action: "session.activation.result",
          outcome: "completed",
          activationId: "activation_1",
        }),
      ],
    );

    const activationTurn = timeline.turns.find((turn) => turn.activationId === "activation_1");
    expect(timeline.turns).toHaveLength(2);
    expect(activationTurn).toMatchObject({ origin: "system", status: "completed" });
    expect(
      timeline.events.find(
        (event) => event.type === "model_response" && event.sourceSequence === 5,
      ),
    ).toMatchObject({ turnId: activationTurn?.turnId });
  });

  it("aggregates late chunk observations without reopening or adding unsettled work", () => {
    const timeline = projectSessionTimeline(
      source([
        {
          ...appended(2, "request-late", [message("user", "run")]),
          requestState: "started",
        },
        {
          ...appended(3, "request-late", [message("assistant", "done")]),
          requestState: "settled",
          requestOutcome: "completed",
        },
      ]),
      [
        audit(1, {
          category: "session",
          action: "session.late_chunk_observed",
          outcome: "completed",
          requestId: "request-late",
          metadata: {
            adapter: "swarmx",
            chunkKind: "tool_result",
            boundary: "closed",
            observationCount: 1,
          },
        }),
        audit(2, {
          category: "session",
          action: "session.late_chunk_observed",
          outcome: "completed",
          requestId: "request-late",
          metadata: {
            adapter: "swarmx",
            chunkKind: "tool_result",
            boundary: "closed",
            observationCount: 2,
          },
        }),
      ],
    );

    expect(timeline.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "late_chunk_observed",
          late: true,
          observationCount: 3,
        }),
      ]),
    );
    expect(timeline.events.filter((event) => event.type === "late_chunk_observed")).toHaveLength(1);
    expect(timeline.turns).toMatchObject([{ status: "completed" }]);
    expect(timeline.unsettled).toEqual([]);
  });

  it("links model, approval, tool progress, and result steps with deterministic causation", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-tools", [message("user", "run")]),
        appended(3, "request-tools", [toolCall("call-1", "Read")]),
        appended(4, "request-tools", [toolProgress("call-1", "secret terminal chunk")]),
        appended(5, "request-tools", [toolProgress("call-1", "another terminal chunk")]),
        appended(6, "request-tools", [toolResult("call-1", "succeeded", "secret result")]),
        appended(7, "request-tools", [message("assistant", "done")]),
      ]),
      [
        audit(1, {
          category: "permission",
          action: "tool.decision",
          outcome: "attempted",
          requestId: "request-tools",
          target: { kind: "tool", id: "Read" },
          metadata: { decision: "allowed" },
        }),
        audit(2, {
          category: "permission",
          action: "tool.decision",
          outcome: "completed",
          requestId: "request-tools",
          target: { kind: "tool", id: "Read" },
          metadata: { decision: "allowed" },
        }),
      ],
    );

    const model = timeline.events.find(
      (event) => event.type === "model_response" && event.toolName === "Read",
    );
    const started = timeline.events.find((event) => event.type === "tool_started");
    const progress = timeline.events.find((event) => event.type === "tool_progress");
    const finished = timeline.events.find((event) => event.type === "tool_finished");
    const approval = timeline.events.filter((event) => event.type === "approval_decided");
    expect(started?.causationId).toBe(model?.eventId);
    expect(progress).toMatchObject({ causationId: started?.eventId, observationCount: 2 });
    expect(finished).toMatchObject({ stepId: started?.stepId, outcome: "succeeded" });
    expect(approval).toMatchObject([{ outcome: "allowed", stepId: started?.stepId }]);
    const eventOrdinals = new Map(timeline.events.map((event) => [event.eventId, event.ordinal]));
    for (const event of timeline.events) {
      if (!event.causationId) continue;
      expect(eventOrdinals.get(event.causationId)).toBeLessThan(event.ordinal);
    }
    expect(timeline.unsettled).toEqual([]);
    expect(timeline.turns[0]?.status).toBe("completed");
    expect(JSON.stringify(timeline)).not.toContain("terminal chunk");
    expect(JSON.stringify(timeline)).not.toContain("secret result");
  });

  it("fails closed on denied approval and reports unsettled structured work", () => {
    const denied = projectSessionTimeline(
      source([
        appended(2, "request-denied", [message("user", "delete")]),
        appended(3, "request-denied", [toolCall("call-denied", "Delete")]),
      ]),
      [
        audit(1, {
          category: "permission",
          action: "tool.decision",
          outcome: "denied",
          requestId: "request-denied",
          target: { kind: "tool", id: "Delete" },
          metadata: { decision: "denied" },
        }),
      ],
    );
    expect(denied.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "approval_decided", outcome: "denied" }),
      ]),
    );
    expect(denied.unsettled).toEqual([]);

    const pending = projectSessionTimeline(
      source([
        appended(2, "request-pending", [message("user", "run")]),
        appended(3, "request-pending", [toolCall("call-pending", "Shell")]),
      ]),
    );
    expect(pending.unsettled).toMatchObject([
      { kind: "tool_result", summary: "Tool Shell has no settled result in canonical history." },
    ]);
    expect(pending.turns[0]).toMatchObject({ status: "active" });
  });

  it("keeps late results on their original Turn and collapses duplicate transport records", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-1", [message("user", "first")]),
        appended(3, "request-1", [toolCall("shared-call", "Fetch")]),
        appended(4, "request-2", [message("user", "second")]),
        appended(5, "request-2", [message("assistant", "second done")]),
        appended(6, "request-1", [toolResult("shared-call", "succeeded", "late")]),
        appended(7, "request-1", [toolResult("shared-call", "succeeded", "duplicate")]),
      ]),
    );

    const firstTurn = timeline.turns.find((turn) => turn.correlationId === "request-1");
    const late = timeline.events.find((event) => event.type === "tool_finished");
    expect(timeline.turns.map((turn) => turn.correlationId)).toEqual(["request-1", "request-2"]);
    expect(late).toMatchObject({ turnId: firstTurn?.turnId, late: true });
    expect(timeline.events.filter((event) => event.type === "tool_finished")).toHaveLength(1);
    expect(timeline.diagnostics).toEqual(
      expect.arrayContaining([expect.objectContaining({ code: "duplicate_transport" })]),
    );
    expect(timeline.unsettled).toEqual([]);
  });

  it("does not guess a missing-invocation lifecycle when multiple Turns are candidates", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, undefined, [message("user", "first")]),
        appended(3, undefined, [toolCall("first-call", "Read")]),
        appended(4, undefined, [message("user", "second")]),
        appended(5, undefined, [toolCall("second-call", "Write")]),
        {
          ...appended(6, undefined, [
            {
              role: "tool",
              content: "late result",
              kind: "tool_result",
              toolName: "Read",
              render: { status: "succeeded" },
            },
          ]),
        },
      ]),
    );

    const finished = timeline.events.find((event) => event.type === "tool_finished");
    expect(finished?.turnId).not.toBe(timeline.turns[0]?.turnId);
    expect(finished?.turnId).not.toBe(timeline.turns[1]?.turnId);
    expect(timeline.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "missing_invocation_id" }),
        expect.objectContaining({ code: "orphan_tool_result" }),
      ]),
    );
    expect(timeline.unsettled).toHaveLength(2);
  });

  it("does not fall back to the only active Turn when another unresolved lifecycle is settled", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-settled", [message("user", "first")]),
        appended(3, "request-settled", [toolCall("settled-turn-call", "Read")]),
        {
          ...appended(4, "request-settled", []),
          requestState: "settled",
          requestOutcome: "completed",
        },
        appended(5, "request-active", [message("user", "second")]),
        appended(6, "request-active", [toolCall("active-turn-call", "Read")]),
        appended(7, undefined, [
          {
            role: "tool",
            content: "late result",
            kind: "tool_result",
            toolName: "Read",
            render: { status: "succeeded" },
          },
        ]),
      ]),
    );

    const finished = timeline.events.find(
      (event) => event.type === "tool_finished" && event.sourceSequence === 7,
    );
    expect(finished?.turnId).not.toBe(timeline.turns[1]?.turnId);
    expect(timeline.diagnostics).toEqual(
      expect.arrayContaining([expect.objectContaining({ code: "orphan_tool_result" })]),
    );
  });

  it("projects cancellation, resumption, retries, concurrent child tasks, and external ownership", () => {
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-control", [message("user", "coordinate")]),
        appended(3, "request-control", [message("user", "coordinate retry")]),
      ]),
      [
        audit(1, {
          category: "task",
          action: "task.started",
          outcome: "attempted",
          requestId: "request-control",
          taskId: "child-a",
          actor: { kind: "agent", id: "parent-agent" },
        }),
        audit(2, {
          category: "task",
          action: "task.started",
          outcome: "attempted",
          requestId: "request-control",
          taskId: "child-b",
          actor: { kind: "agent", id: "parent-agent" },
        }),
        audit(3, {
          category: "task",
          action: "task.finished",
          outcome: "completed",
          requestId: "request-control",
          taskId: "child-a",
          actor: { kind: "process", id: "worker-a" },
        }),
        audit(4, {
          category: "session",
          action: "acp.prompt",
          outcome: "completed",
          requestId: "request-control",
          actor: { kind: "process", id: "external-harness" },
        }),
        audit(5, {
          category: "session",
          action: "acp.prompt.cancel",
          outcome: "cancelled",
          requestId: "request-control",
        }),
        audit(6, {
          category: "session",
          action: "acp.session.resume",
          outcome: "completed",
          requestId: "request-control",
        }),
      ],
    );

    expect(timeline.turns).toMatchObject([{ retryCount: 1, status: "active" }]);
    expect(timeline.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "task_observed",
          actor: { kind: "agent", id: "parent-agent" },
        }),
        expect.objectContaining({
          type: "external_operation",
          actor: { kind: "process", id: "external-harness" },
        }),
        expect.objectContaining({ type: "turn_state", outcome: "cancelled" }),
        expect.objectContaining({ type: "turn_state", outcome: "resumed" }),
      ]),
    );
    expect(timeline.unsettled).toMatchObject([
      { kind: "task_outcome", summary: "Child task child-b has no terminal audit outcome." },
    ]);
  });

  it("explains pauses and terminal failures from correlated audit evidence", () => {
    const timeline = projectSessionTimeline(
      source([appended(2, "request-failed", [message("user", "run")])]),
      [
        audit(1, {
          category: "system",
          action: "agent.human_needed",
          outcome: "attempted",
          requestId: "request-failed",
        }),
        audit(2, {
          category: "system",
          action: "agent.execution",
          outcome: "failed",
          requestId: "request-failed",
        }),
      ],
    );

    expect(timeline.events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "turn_state", outcome: "pending" }),
        expect.objectContaining({ type: "turn_state", outcome: "failed" }),
      ]),
    );
    expect(timeline.turns).toMatchObject([
      { status: "failed", statusReason: "A correlated audit event records a terminal failure." },
    ]);
  });

  it("replays old and restarted logs deterministically with conservative ids", () => {
    const legacy = source(
      [
        appended(2, undefined, [message("user", "legacy secret")]),
        appended(3, undefined, [message("assistant", "legacy answer")]),
      ],
      true,
    );
    const first = projectSessionTimeline(legacy);
    const restarted = projectSessionTimeline(structuredClone(legacy));

    expect(restarted).toEqual(first);
    expect(first.turns).toMatchObject([{ inferred: true, status: "completed" }]);
    expect(first.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ code: "legacy_missing_request_id" }),
        expect.objectContaining({ code: "torn_tail" }),
      ]),
    );
    expect(first.fingerprint).toMatch(/^timeline_[a-f0-9]{16}$/);
  });

  it("never emits prompts, responses, source, terminal output, credentials, or environment data", () => {
    const sensitive = [
      "raw prompt sk-secret",
      "model response password=hunter2",
      "const sourceCode = true",
      "terminal stdout private",
      "HOME=/Users/private",
    ].join(" | ");
    const timeline = projectSessionTimeline(
      source([
        appended(2, "request-safe", [message("user", sensitive)]),
        appended(3, "request-safe", [toolCall("call-safe", "Read", sensitive)]),
        appended(4, "request-safe", [toolResult("call-safe", "succeeded", sensitive)]),
        appended(5, "request-safe", [message("assistant", sensitive)]),
      ]),
    );
    const serialized = JSON.stringify(timeline);
    for (const forbidden of ["sk-secret", "hunter2", "sourceCode", "stdout", "/Users/private"]) {
      expect(serialized).not.toContain(forbidden);
    }
  });
});

function source(records: SessionTimelineSourceRecord[], tornTail = false): SessionTimelineSource {
  return {
    sessionId: SESSION_ID,
    projectId: PROJECT_ID,
    tornTail,
    records: [
      {
        sequence: 1,
        type: "session_created",
        timestamp: at(1),
        messages: [],
      },
      ...records,
    ],
  };
}

function appended(
  sequence: number,
  requestId: string | undefined,
  messages: MessageChunk[],
): SessionTimelineSourceRecord {
  return {
    sequence,
    type: "messages_appended",
    timestamp: at(sequence),
    ...(requestId ? { requestId } : {}),
    messages,
  };
}

function message(role: string, content: string): MessageChunk {
  return { role, content, kind: "message" };
}

function toolCall(
  invocationId: string,
  toolName: string,
  content = "private arguments",
): MessageChunk {
  return {
    role: "assistant",
    content,
    kind: "tool_call",
    toolName,
    render: { invocationId, status: "running" },
  };
}

function toolProgress(invocationId: string, content: string): MessageChunk {
  return {
    role: "tool",
    content,
    kind: "tool_progress",
    toolName: "Read",
    render: { invocationId, status: "running" },
  };
}

function toolResult(
  invocationId: string,
  status: "succeeded" | "failed" | "canceled",
  content: string,
): MessageChunk {
  return {
    role: "tool",
    content,
    kind: "tool_result",
    toolName: "Read",
    render: { invocationId, status },
  };
}

function audit(
  sequence: number,
  input: Omit<
    SessionTimelineAuditRecord,
    "eventId" | "metadata" | "sequence" | "sessionId" | "timestamp"
  > & {
    metadata?: SessionTimelineAuditRecord["metadata"];
  },
): SessionTimelineAuditRecord {
  return {
    sequence,
    eventId: `aud_${sequence}`,
    timestamp: at(sequence + 20),
    sessionId: SESSION_ID,
    metadata: input.metadata ?? {},
    ...input,
  };
}

function at(sequence: number): string {
  return new Date(Date.UTC(2026, 7, 14, 0, 0, sequence)).toISOString();
}

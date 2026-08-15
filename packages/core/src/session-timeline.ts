import { z } from "zod";
import { stableHash, stableJson } from "./canonical-json.js";
import { MessageChunkSchema } from "./types.js";

const SOURCE_RECORD_TYPES = [
  "session_created",
  "messages_appended",
  "messages_replaced",
  "session_updated",
] as const;

const TIMELINE_EVENT_TYPES = [
  "turn_started",
  "model_response",
  "tool_started",
  "tool_progress",
  "tool_finished",
  "approval_decided",
  "task_observed",
  "external_operation",
  "late_chunk_observed",
  "turn_state",
  "history_replaced",
  "session_updated",
] as const;

const TIMELINE_OUTCOMES = [
  "observed",
  "pending",
  "allowed",
  "denied",
  "succeeded",
  "failed",
  "cancel_requested",
  "cancelled",
  "completed",
  "resumed",
  "retried",
  "unknown",
] as const;

const TimelineAuditCategorySchema = z.enum([
  "session",
  "task",
  "tool",
  "permission",
  "provider",
  "secret",
  "extension",
  "workspace",
  "telemetry",
  "system",
]);

const TimelineAuditOutcomeSchema = z.enum([
  "attempted",
  "completed",
  "failed",
  "denied",
  "cancel_requested",
  "cancelled",
]);

const TimelineIdentifierSchema = z
  .string()
  .min(1)
  .max(256)
  .regex(/^[A-Za-z0-9][A-Za-z0-9_.:@/-]*$/);

const TimelineAuditMetadataValueSchema: z.ZodType<unknown> = z.lazy(() =>
  z.union([
    z.null(),
    z.boolean(),
    z.number().finite(),
    z.string().max(160),
    z.array(TimelineAuditMetadataValueSchema).max(16),
    z.record(z.string().min(1).max(64), TimelineAuditMetadataValueSchema),
  ]),
);

const TimelineAuditMetadataSchema = z
  .record(z.string().min(1).max(64), TimelineAuditMetadataValueSchema)
  .superRefine((metadata, context) => {
    if (Object.keys(metadata).length > 64) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Timeline audit metadata must contain at most 64 top-level entries.",
      });
    }
  });

export const SessionTimelineSourceRecordSchema = z
  .object({
    sequence: z.number().int().positive(),
    type: z.enum(SOURCE_RECORD_TYPES),
    timestamp: z.string().min(1).max(128),
    activationId: z.string().min(1).max(256).optional(),
    requestId: z.string().min(1).max(256).optional(),
    requestDigest: z.string().min(1).max(256).optional(),
    requestState: z.enum(["started", "settled"]).optional(),
    requestOutcome: z.enum(["completed", "canceled", "failed"]).optional(),
    messages: z.array(MessageChunkSchema).default([]),
    reason: z.literal("edit_last_user_message").optional(),
    replacedFromIndex: z.number().int().nonnegative().optional(),
    replacedMessageCount: z.number().int().positive().optional(),
  })
  .superRefine((record, context) => {
    if (record.requestState === "settled" && !record.requestOutcome) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Settled Session records require a request outcome.",
        path: ["requestOutcome"],
      });
    }
    if (record.requestOutcome && record.requestState !== "settled") {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Session request outcomes require a settled record.",
        path: ["requestState"],
      });
    }
  })
  .strict();

export const SessionTimelineSourceSchema = z
  .object({
    sessionId: z.string().min(1).max(256),
    projectId: z.string().min(1).max(256).optional(),
    records: z.array(SessionTimelineSourceRecordSchema),
    tornTail: z.boolean().default(false),
  })
  .strict();

export const SessionTimelineAuditRecordSchema = z
  .object({
    sequence: z.number().int().positive(),
    eventId: TimelineIdentifierSchema,
    timestamp: z.string().min(1).max(128),
    category: TimelineAuditCategorySchema,
    action: z
      .string()
      .min(1)
      .max(96)
      .regex(/^[a-z][a-z0-9_.-]*$/),
    outcome: TimelineAuditOutcomeSchema,
    actor: z
      .object({
        kind: z.enum(["user", "agent", "system", "process", "service"]),
        id: TimelineIdentifierSchema.optional(),
      })
      .strict()
      .optional(),
    target: z
      .object({
        kind: TimelineIdentifierSchema,
        id: TimelineIdentifierSchema.optional(),
      })
      .strict()
      .optional(),
    sessionId: TimelineIdentifierSchema.optional(),
    taskId: TimelineIdentifierSchema.optional(),
    requestId: TimelineIdentifierSchema.optional(),
    activationId: TimelineIdentifierSchema.optional(),
    metadata: TimelineAuditMetadataSchema,
  })
  .strict();

const TimelineActorSchema = z
  .object({
    kind: z.enum(["user", "agent", "system", "tool", "process", "service"]),
    id: TimelineIdentifierSchema.optional(),
  })
  .strict();

export const SessionTimelineEventSchema = z
  .object({
    eventId: z.string().regex(/^evt_[a-f0-9]{16}$/),
    projectId: TimelineIdentifierSchema.optional(),
    sessionId: TimelineIdentifierSchema,
    turnId: z
      .string()
      .regex(/^turn_[a-f0-9]{16}$/)
      .optional(),
    stepId: z
      .string()
      .regex(/^step_[a-f0-9]{16}$/)
      .optional(),
    correlationId: TimelineIdentifierSchema,
    activationId: TimelineIdentifierSchema.optional(),
    causationId: z
      .string()
      .regex(/^evt_[a-f0-9]{16}$/)
      .optional(),
    ordinal: z.number().int().positive(),
    source: z.enum(["session", "audit"]),
    sourceSequence: z.number().int().positive(),
    sourceEventId: TimelineIdentifierSchema.optional(),
    timestamp: z.string().datetime({ offset: true }),
    type: z.enum(TIMELINE_EVENT_TYPES),
    outcome: z.enum(TIMELINE_OUTCOMES),
    summary: z.string().min(1).max(160),
    actor: TimelineActorSchema.optional(),
    toolName: z.string().min(1).max(64).optional(),
    late: z.boolean().default(false),
    inferred: z.boolean().default(false),
    observationCount: z.number().int().positive().default(1),
  })
  .strict();

export const SessionTimelineStepSchema = z
  .object({
    stepId: z.string().regex(/^step_[a-f0-9]{16}$/),
    turnId: z.string().regex(/^turn_[a-f0-9]{16}$/),
    kind: z.enum(["model", "tool", "approval", "task", "external"]),
    status: z.enum(["observed", "pending", "succeeded", "failed", "cancelled", "unknown"]),
    eventIds: z.array(z.string().regex(/^evt_[a-f0-9]{16}$/)),
    toolName: z.string().min(1).max(64).optional(),
    invocationId: TimelineIdentifierSchema.optional(),
    inferred: z.boolean().default(false),
  })
  .strict();

export const SessionTimelineTurnSchema = z
  .object({
    turnId: z.string().regex(/^turn_[a-f0-9]{16}$/),
    correlationId: TimelineIdentifierSchema,
    activationId: TimelineIdentifierSchema.optional(),
    origin: z.enum(["user", "system"]),
    openedByEventId: z.string().regex(/^evt_[a-f0-9]{16}$/),
    status: z.enum(["active", "completed", "failed", "cancelled", "unknown"]),
    statusReason: z.string().min(1).max(160),
    stepIds: z.array(z.string().regex(/^step_[a-f0-9]{16}$/)),
    eventIds: z.array(z.string().regex(/^evt_[a-f0-9]{16}$/)),
    inferred: z.boolean().default(false),
    retryCount: z.number().int().nonnegative().default(0),
  })
  .strict();

export const SessionTimelineUnsettledSchema = z
  .object({
    pendingId: z.string().regex(/^pending_[a-f0-9]{16}$/),
    turnId: z.string().regex(/^turn_[a-f0-9]{16}$/),
    stepId: z.string().regex(/^step_[a-f0-9]{16}$/),
    kind: z.enum(["tool_result", "task_outcome"]),
    summary: z.string().min(1).max(160),
  })
  .strict();

export const SessionTimelineDiagnosticSchema = z
  .object({
    code: z.enum([
      "torn_tail",
      "legacy_missing_request_id",
      "invalid_timestamp",
      "missing_invocation_id",
      "orphan_tool_result",
      "duplicate_transport",
      "history_replaced",
      "unlinked_audit_event",
    ]),
    summary: z.string().min(1).max(160),
    source: z.enum(["session", "audit"]),
    sourceSequence: z.number().int().positive().optional(),
  })
  .strict();

export const SessionTimelineSchema = z
  .object({
    schemaVersion: z.literal(1),
    sessionId: TimelineIdentifierSchema,
    projectId: TimelineIdentifierSchema.optional(),
    fingerprint: z.string().regex(/^timeline_[a-f0-9]{16}$/),
    authority: z.literal("derived_diagnostic_projection"),
    events: z.array(SessionTimelineEventSchema),
    turns: z.array(SessionTimelineTurnSchema),
    steps: z.array(SessionTimelineStepSchema),
    unsettled: z.array(SessionTimelineUnsettledSchema),
    diagnostics: z.array(SessionTimelineDiagnosticSchema),
  })
  .strict();

export type SessionTimelineSourceRecord = z.infer<typeof SessionTimelineSourceRecordSchema>;
export type SessionTimelineSource = z.infer<typeof SessionTimelineSourceSchema>;
export type SessionTimelineAuditRecord = z.infer<typeof SessionTimelineAuditRecordSchema>;
export type SessionTimelineEvent = z.infer<typeof SessionTimelineEventSchema>;
export type SessionTimelineStep = z.infer<typeof SessionTimelineStepSchema>;
export type SessionTimelineTurn = z.infer<typeof SessionTimelineTurnSchema>;
export type SessionTimelineUnsettled = z.infer<typeof SessionTimelineUnsettledSchema>;
export type SessionTimelineDiagnostic = z.infer<typeof SessionTimelineDiagnosticSchema>;
export type SessionTimeline = z.infer<typeof SessionTimelineSchema>;

export interface VerifiedSessionTimelineAuditEvent extends SessionTimelineAuditRecord {
  previousHash?: string;
  eventHash?: string;
  schemaVersion?: number;
}

interface MutableTurn {
  turnId: string;
  correlationId: string;
  activationId?: string;
  openedByEventId: string;
  status: SessionTimelineTurn["status"];
  statusReason: string;
  stepIds: string[];
  eventIds: string[];
  inferred: boolean;
  retryCount: number;
  finalResponseObserved: boolean;
  settled: boolean;
  origin: SessionTimelineTurn["origin"];
}

interface MutableStep extends SessionTimelineStep {
  eventIds: string[];
}

interface ToolLifecycle {
  invocationId: string;
  turnId: string;
  stepId: string;
  toolName: string;
  startedEventId: string;
  settled: boolean;
}

interface PendingTask {
  taskId: string;
  turnId: string;
  stepId: string;
  terminal: boolean;
}

interface PendingEvent {
  event: Omit<SessionTimelineEvent, "ordinal">;
  sortTimestamp: string;
  localOrder: number;
}

/** Converts verified audit records into the strict, content-free projection input. */
export function sessionTimelineAuditRecords(
  events: readonly VerifiedSessionTimelineAuditEvent[],
): SessionTimelineAuditRecord[] {
  return events.map((event) =>
    SessionTimelineAuditRecordSchema.parse({
      sequence: event.sequence,
      eventId: event.eventId,
      timestamp: event.timestamp,
      category: event.category,
      action: event.action,
      outcome: event.outcome,
      actor: event.actor,
      target: event.target,
      sessionId: event.sessionId,
      taskId: event.taskId,
      requestId: event.requestId,
      activationId: event.activationId,
      metadata: event.metadata,
    }),
  );
}

/**
 * Rebuilds a safe causal diagnostic view. It never mutates Session, audit, task,
 * approval, or completion-barrier state and therefore has no execution authority.
 */
export function projectSessionTimeline(
  sourceInput: unknown,
  auditInput: readonly unknown[] = [],
): SessionTimeline {
  const source = SessionTimelineSourceSchema.parse(sourceInput);
  const audit = auditInput.map((event) => SessionTimelineAuditRecordSchema.parse(event));
  const sessionId = safeIdentifier(source.sessionId, "session");
  const projectId = source.projectId ? safeIdentifier(source.projectId, "project") : undefined;
  const pendingEvents: PendingEvent[] = [];
  const diagnostics: SessionTimelineDiagnostic[] = [];
  const turns = new Map<string, MutableTurn>();
  const turnsByRequest = new Map<string, MutableTurn>();
  const turnsByActivation = new Map<string, MutableTurn>();
  const steps = new Map<string, MutableStep>();
  const tools = new Map<string, ToolLifecycle>();
  const pendingTasks = new Map<string, PendingTask>();
  const seenLifecycle = new Set<string>();
  const seenAudit = new Set<string>();
  const progressEvents = new Map<string, Omit<SessionTimelineEvent, "ordinal">>();
  const lateChunkEvents = new Map<string, Omit<SessionTimelineEvent, "ordinal">>();
  let currentTurn: MutableTurn | undefined;
  let localOrder = 0;

  const addDiagnostic = (diagnostic: SessionTimelineDiagnostic): void => {
    const key = stableJson(diagnostic);
    if (!diagnostics.some((item) => stableJson(item) === key)) diagnostics.push(diagnostic);
  };

  const addEvent = (
    input: Omit<SessionTimelineEvent, "eventId" | "ordinal"> & { eventKey: string },
  ): Omit<SessionTimelineEvent, "ordinal"> => {
    const { eventKey, ...body } = input;
    const event = SessionTimelineEventSchema.omit({ ordinal: true }).parse({
      ...body,
      eventId: deterministicId("evt", sessionId, eventKey),
    });
    pendingEvents.push({ event, sortTimestamp: event.timestamp, localOrder: localOrder++ });
    if (event.turnId) turns.get(event.turnId)?.eventIds.push(event.eventId);
    if (event.stepId) steps.get(event.stepId)?.eventIds.push(event.eventId);
    return event;
  };

  const openTurn = (
    requestId: string | undefined,
    record: SessionTimelineSourceRecord,
    inferred: boolean,
  ): MutableTurn => {
    const requestKey = requestId ? safeIdentifier(requestId, "request") : undefined;
    const existing = requestKey ? turnsByRequest.get(requestKey) : undefined;
    if (existing) {
      existing.retryCount += 1;
      currentTurn = existing;
      const event = addEvent({
        eventKey: `retry:${record.sequence}:${existing.retryCount}`,
        projectId,
        sessionId,
        turnId: existing.turnId,
        correlationId: existing.correlationId,
        causationId: existing.eventIds.at(-1),
        source: "session",
        sourceSequence: record.sequence,
        timestamp: safeTimestamp(record.timestamp, record.sequence, addDiagnostic),
        type: "turn_state",
        outcome: "retried",
        summary: "A retry resumed the existing Turn.",
        actor: { kind: "user" },
        late: false,
        inferred,
        observationCount: 1,
      });
      existing.status = "active";
      existing.statusReason =
        "A retry was observed; settlement remains derived from later evidence.";
      existing.eventIds.push(event.eventId);
      return existing;
    }

    const correlationId = requestKey ?? deterministicId("corr", sessionId, record.sequence);
    const turnId = deterministicId("turn", sessionId, correlationId, record.sequence);
    const event = addEvent({
      eventKey: `turn:${turnId}`,
      projectId,
      sessionId,
      turnId,
      correlationId,
      source: "session",
      sourceSequence: record.sequence,
      timestamp: safeTimestamp(record.timestamp, record.sequence, addDiagnostic),
      type: "turn_started",
      outcome: "observed",
      summary: inferred
        ? "A Turn was conservatively inferred from Session activity."
        : "A user request opened a Turn.",
      actor: inferred ? { kind: "system" } : { kind: "user" },
      late: false,
      inferred,
      observationCount: 1,
    });
    const turn: MutableTurn = {
      turnId,
      correlationId,
      openedByEventId: event.eventId,
      status: "active",
      statusReason: "The Turn has not yet reached a conservative terminal projection.",
      stepIds: [],
      eventIds: [event.eventId],
      inferred,
      retryCount: 0,
      finalResponseObserved: false,
      settled: false,
      origin: "user",
    };
    turns.set(turnId, turn);
    if (requestKey) turnsByRequest.set(requestKey, turn);
    currentTurn = turn;
    return turn;
  };

  const ensureTurn = (record: SessionTimelineSourceRecord): MutableTurn => {
    const requestKey = record.requestId ? safeIdentifier(record.requestId, "request") : undefined;
    const correlated = requestKey ? turnsByRequest.get(requestKey) : undefined;
    if (correlated) return correlated;
    if (!requestKey) {
      const unresolvedTurns = [
        ...new Set(
          [...tools.values()]
            .filter((lifecycle) => !lifecycle.settled)
            .map((lifecycle) => lifecycle.turnId),
        ),
      ]
        .map((turnId) => turns.get(turnId))
        .filter((turn): turn is MutableTurn => Boolean(turn));
      const activeCandidates = [...turns.values()].filter(
        (turn) => turn.status === "active" && !turn.finalResponseObserved,
      );
      const candidates = unresolvedTurns.length > 0 ? unresolvedTurns : activeCandidates;
      if (candidates.length === 1) return candidates[0];
      if (candidates.length === 0 && turns.size === 1) {
        const onlyTurn = turns.values().next().value;
        if (onlyTurn) return onlyTurn;
      }
    }
    if (!record.requestId) {
      addDiagnostic({
        code: "legacy_missing_request_id",
        summary: "An older Session record lacked a request id; its Turn id was inferred.",
        source: "session",
        sourceSequence: record.sequence,
      });
    }
    return openTurn(record.requestId, record, true);
  };

  const openActivationTurn = (
    activationId: string,
    evidence: {
      source: "session" | "audit";
      sourceSequence: number;
      sourceEventId?: string;
      timestamp: string;
    },
  ): MutableTurn => {
    const activationKey = safeIdentifier(activationId, "activation");
    const existing = turnsByActivation.get(activationKey);
    if (existing) return existing;
    const correlationId = activationKey;
    const turnId = deterministicId("turn", sessionId, "activation", activationKey);
    const event = addEvent({
      eventKey: `activation-start:${activationKey}`,
      projectId,
      sessionId,
      turnId,
      activationId: activationKey,
      correlationId,
      source: evidence.source,
      sourceSequence: evidence.sourceSequence,
      ...(evidence.sourceEventId ? { sourceEventId: evidence.sourceEventId } : {}),
      timestamp: safeTimestamp(
        evidence.timestamp,
        evidence.sourceSequence,
        addDiagnostic,
        evidence.source,
      ),
      type: "turn_started",
      outcome: "observed",
      summary: "A background activation opened a system Turn.",
      actor: { kind: "system" },
      late: false,
      inferred: false,
      observationCount: 1,
    });
    const turn: MutableTurn = {
      turnId,
      correlationId,
      activationId: activationKey,
      openedByEventId: event.eventId,
      status: "active",
      statusReason: "The background activation has not reached a terminal outcome.",
      stepIds: [],
      eventIds: [event.eventId],
      inferred: false,
      retryCount: 0,
      finalResponseObserved: false,
      settled: false,
      origin: "system",
    };
    turns.set(turnId, turn);
    turnsByActivation.set(activationKey, turn);
    return turn;
  };

  const projectActivationAudit = (auditEvent: SessionTimelineAuditRecord): boolean => {
    if (!auditEvent.activationId) return false;
    const activationKey = safeIdentifier(auditEvent.activationId, "activation");
    const phase = auditEvent.action.split(".").at(-1) ?? "observed";
    const turn =
      phase === "started"
        ? openActivationTurn(activationKey, {
            source: "audit",
            sourceSequence: auditEvent.sequence,
            sourceEventId: auditEvent.eventId,
            timestamp: auditEvent.timestamp,
          })
        : (turnsByActivation.get(activationKey) ??
          openActivationTurn(activationKey, {
            source: "audit",
            sourceSequence: auditEvent.sequence,
            sourceEventId: auditEvent.eventId,
            timestamp: auditEvent.timestamp,
          }));
    if (phase === "started") {
      if (auditEvent.outcome === "failed") {
        turn.status = "failed";
        turn.statusReason = "The background activation failed while starting.";
        turn.settled = true;
      }
      return true;
    }
    const outcome = auditOutcome(auditEvent.outcome);
    const isTerminal = phase === "result" || phase === "failure";
    const terminalOutcome =
      phase === "failure" || outcome === "failed"
        ? ("failed" as const)
        : phase === "result" && outcome === "completed"
          ? ("completed" as const)
          : outcome;
    const step = addStep(turn, "external", `activation:${phase}`, false);
    addEvent({
      eventKey: `activation:${activationKey}:${auditEvent.eventId}`,
      projectId,
      sessionId,
      turnId: turn.turnId,
      activationId: activationKey,
      stepId: step.stepId,
      correlationId: turn.correlationId,
      causationId: turn.eventIds.at(-1),
      source: "audit",
      sourceSequence: auditEvent.sequence,
      sourceEventId: auditEvent.eventId,
      timestamp: safeTimestamp(auditEvent.timestamp, auditEvent.sequence, addDiagnostic, "audit"),
      type: phase === "bootstrap" ? "external_operation" : "turn_state",
      outcome: terminalOutcome,
      summary:
        phase === "bootstrap"
          ? "Background activation bootstrap was observed."
          : phase === "result"
            ? "Background activation produced a terminal result."
            : phase === "failure"
              ? "Background activation failed."
              : "Background activation state was observed.",
      actor: { kind: "system" },
      late: false,
      inferred: false,
      observationCount: 1,
    });
    step.status = terminalStepStatus(terminalOutcome);
    if (isTerminal) {
      turn.status =
        terminalOutcome === "completed"
          ? "completed"
          : terminalOutcome === "cancelled"
            ? "cancelled"
            : terminalOutcome === "failed"
              ? "failed"
              : "unknown";
      turn.statusReason =
        terminalOutcome === "completed"
          ? "A correlated activation result records completion."
          : "A correlated activation result records failure or cancellation.";
      turn.settled = true;
    }
    return true;
  };

  const settleRequest = (record: SessionTimelineSourceRecord, turn: MutableTurn): void => {
    if (record.requestState !== "settled" || !record.requestOutcome) return;
    const outcome =
      record.requestOutcome === "completed"
        ? ("completed" as const)
        : record.requestOutcome === "canceled"
          ? ("cancelled" as const)
          : ("failed" as const);
    addEvent({
      eventKey: `request-settled:${record.sequence}`,
      projectId,
      sessionId,
      turnId: turn.turnId,
      correlationId: turn.correlationId,
      causationId: turn.eventIds.at(-1),
      source: "session",
      sourceSequence: record.sequence,
      timestamp: safeTimestamp(record.timestamp, record.sequence, addDiagnostic),
      type: "turn_state",
      outcome,
      summary:
        outcome === "completed"
          ? "The request reached its durable terminal outcome."
          : outcome === "cancelled"
            ? "The request was durably cancelled."
            : "The request durably failed.",
      actor: { kind: "system" },
      late: false,
      inferred: false,
      observationCount: 1,
    });
    turn.status = outcome;
    turn.statusReason =
      outcome === "completed"
        ? "A durable settled receipt records successful completion."
        : outcome === "cancelled"
          ? "A durable settled receipt records cancellation."
          : "A durable settled receipt records failure.";
    turn.settled = true;
  };

  const addStep = (
    turn: MutableTurn,
    kind: SessionTimelineStep["kind"],
    key: string,
    inferred: boolean,
    details: Pick<SessionTimelineStep, "toolName" | "invocationId"> = {},
  ): MutableStep => {
    const stepId = deterministicId("step", sessionId, turn.turnId, key);
    const existing = steps.get(stepId);
    if (existing) return existing;
    const step: MutableStep = {
      stepId,
      turnId: turn.turnId,
      kind,
      status: kind === "tool" || kind === "task" ? "pending" : "observed",
      eventIds: [],
      ...details,
      inferred,
    };
    steps.set(stepId, step);
    turn.stepIds.push(stepId);
    return step;
  };

  if (source.tornTail) {
    addDiagnostic({
      code: "torn_tail",
      summary: "The incomplete final JSONL record was ignored; the projection is conservative.",
      source: "session",
    });
  }

  for (const record of source.records) {
    const timestamp = safeTimestamp(record.timestamp, record.sequence, addDiagnostic);
    if (record.type === "session_updated") {
      const correlationId =
        currentTurn?.correlationId ?? deterministicId("corr", sessionId, record.sequence);
      addEvent({
        eventKey: `session-updated:${record.sequence}`,
        projectId,
        sessionId,
        turnId: currentTurn?.turnId,
        correlationId,
        causationId: currentTurn?.eventIds.at(-1),
        source: "session",
        sourceSequence: record.sequence,
        timestamp,
        type: "session_updated",
        outcome: "observed",
        summary: "Session metadata changed.",
        actor: { kind: "system" },
        late: false,
        inferred: false,
        observationCount: 1,
      });
      continue;
    }
    const activationTurn = record.activationId
      ? openActivationTurn(record.activationId, {
          source: "session",
          sourceSequence: record.sequence,
          timestamp: record.timestamp,
        })
      : undefined;
    let receiptTurn: MutableTurn | undefined;
    if (record.requestState === "started") {
      receiptTurn = openTurn(record.requestId, record, !record.requestId);
    } else if (record.requestState === "settled") {
      receiptTurn = ensureTurn(record);
      settleRequest(record, receiptTurn);
    }
    if (record.type === "messages_replaced") {
      const turn = activationTurn ?? receiptTurn ?? ensureTurn(record);
      addEvent({
        eventKey: `history-replaced:${record.sequence}`,
        projectId,
        sessionId,
        turnId: turn.turnId,
        correlationId: turn.correlationId,
        causationId: turn.eventIds.at(-1),
        source: "session",
        sourceSequence: record.sequence,
        timestamp,
        type: "history_replaced",
        outcome: "observed",
        summary: "Canonical message history was replaced after a user edit.",
        actor: { kind: "user" },
        late: false,
        inferred: !record.requestId,
        observationCount: 1,
      });
      addDiagnostic({
        code: "history_replaced",
        summary:
          "The projection preserves the edit boundary without replaying removed message content.",
        source: "session",
        sourceSequence: record.sequence,
      });
      if (!record.requestId) {
        currentTurn = undefined;
        if (
          record.messages.some((message) => message.role === "user" && message.kind === "message")
        ) {
          openTurn(undefined, record, true);
        }
      } else {
        currentTurn = turn;
      }
      continue;
    }

    for (const [messageIndex, message] of record.messages.entries()) {
      if (message.role === "user" && message.kind === "message") {
        if (!record.requestId) {
          addDiagnostic({
            code: "legacy_missing_request_id",
            summary: "An older user request lacked a request id; its Turn id was inferred.",
            source: "session",
            sourceSequence: record.sequence,
          });
        }
        if (!receiptTurn && record.requestState !== "settled") {
          receiptTurn = openTurn(record.requestId, record, !record.requestId);
        }
        continue;
      }
      if (message.kind === "thinking" || message.role === "system") continue;
      const turn =
        activationTurn ??
        (message.kind === "tool_call" &&
        !record.requestId &&
        currentTurn &&
        currentTurn.status === "active" &&
        !currentTurn.settled
          ? currentTurn
          : ensureTurn(record));
      const lastEventId = turn.eventIds.at(-1);

      if (message.kind === "tool_call") {
        const invocationId = safeOptionalIdentifier(message.render?.invocationId, "invocation");
        const lifecycleId =
          invocationId ?? deterministicId("invocation", sessionId, record.sequence, messageIndex);
        if (!invocationId) {
          addDiagnostic({
            code: "missing_invocation_id",
            summary:
              "A tool call lacked an invocation id; its lifecycle was conservatively inferred.",
            source: "session",
            sourceSequence: record.sequence,
          });
        }
        const lifecycleKey = `tool_call:${turn.turnId}:${lifecycleId}`;
        if (seenLifecycle.has(lifecycleKey)) {
          addDiagnostic({
            code: "duplicate_transport",
            summary:
              "A repeated tool lifecycle observation was collapsed without replaying an effect.",
            source: "session",
            sourceSequence: record.sequence,
          });
          continue;
        }
        seenLifecycle.add(lifecycleKey);
        const toolName = safeToolName(message.toolName);
        const modelStep = addStep(
          turn,
          "model",
          `model-tool:${record.sequence}:${messageIndex}`,
          false,
        );
        const modelEvent = addEvent({
          eventKey: `model-tool:${record.sequence}:${messageIndex}`,
          projectId,
          sessionId,
          turnId: turn.turnId,
          stepId: modelStep.stepId,
          correlationId: turn.correlationId,
          causationId: lastEventId,
          source: "session",
          sourceSequence: record.sequence,
          timestamp,
          type: "model_response",
          outcome: "observed",
          summary: `Model output requested tool ${toolName}.`,
          actor: { kind: "agent", id: safeOptionalIdentifier(message.agent, "agent") },
          toolName,
          late: false,
          inferred: false,
          observationCount: 1,
        });
        modelStep.status = "observed";
        const toolStep = addStep(turn, "tool", `tool:${lifecycleId}`, !invocationId, {
          toolName,
          invocationId: lifecycleId,
        });
        const toolEvent = addEvent({
          eventKey: `tool-start:${turn.turnId}:${lifecycleId}`,
          projectId,
          sessionId,
          turnId: turn.turnId,
          stepId: toolStep.stepId,
          correlationId: turn.correlationId,
          causationId: modelEvent.eventId,
          source: "session",
          sourceSequence: record.sequence,
          timestamp,
          type: "tool_started",
          outcome: "pending",
          summary: `Tool ${toolName} started.`,
          actor: { kind: "tool", id: safeOptionalIdentifier(toolName, "tool") },
          toolName,
          late: false,
          inferred: !invocationId,
          observationCount: 1,
        });
        tools.set(lifecycleId, {
          invocationId: lifecycleId,
          turnId: turn.turnId,
          stepId: toolStep.stepId,
          toolName,
          startedEventId: toolEvent.eventId,
          settled: false,
        });
        turn.finalResponseObserved = false;
        continue;
      }

      if (message.kind === "tool_progress" || message.kind === "tool_result") {
        const invocationId = safeOptionalIdentifier(message.render?.invocationId, "invocation");
        const messageToolName = message.toolName?.trim()
          ? safeToolName(message.toolName)
          : undefined;
        let lifecycle = invocationId ? tools.get(invocationId) : undefined;
        if (!lifecycle && !invocationId) {
          const candidates = [...tools.values()].filter(
            (item) =>
              !item.settled &&
              item.turnId === turn.turnId &&
              messageToolName !== undefined &&
              item.toolName === messageToolName,
          );
          lifecycle = candidates.length === 1 ? candidates[0] : undefined;
          addDiagnostic({
            code: "missing_invocation_id",
            summary: lifecycle
              ? "A tool update lacked an invocation id; one same-Turn, same-tool lifecycle was unambiguous."
              : "A tool update lacked an invocation id; no same-Turn, same-tool lifecycle was unambiguous.",
            source: "session",
            sourceSequence: record.sequence,
          });
        }
        if (!lifecycle) {
          addDiagnostic({
            code: "orphan_tool_result",
            summary: "A tool update could not be linked to a prior invocation.",
            source: "session",
            sourceSequence: record.sequence,
          });
          const inferredId =
            invocationId ?? deterministicId("invocation", sessionId, record.sequence, messageIndex);
          const step = addStep(turn, "tool", `orphan:${inferredId}`, true, {
            toolName: safeToolName(message.toolName),
            invocationId: inferredId,
          });
          lifecycle = {
            invocationId: inferredId,
            turnId: turn.turnId,
            stepId: step.stepId,
            toolName: safeToolName(message.toolName),
            startedEventId: turn.openedByEventId,
            settled: false,
          };
          tools.set(inferredId, lifecycle);
        }
        const lifecycleKey = `${message.kind}:${lifecycle.invocationId}:${message.render?.status ?? "unknown"}`;
        if (seenLifecycle.has(lifecycleKey) && message.kind === "tool_result") {
          addDiagnostic({
            code: "duplicate_transport",
            summary:
              "A repeated tool lifecycle observation was collapsed without replaying an effect.",
            source: "session",
            sourceSequence: record.sequence,
          });
          continue;
        }
        const originalTurn = turns.get(lifecycle.turnId) ?? turn;
        const late = currentTurn !== undefined && originalTurn.turnId !== currentTurn.turnId;
        const step = steps.get(lifecycle.stepId);
        if (message.kind === "tool_progress") {
          const existing = progressEvents.get(lifecycle.invocationId);
          if (existing) {
            existing.observationCount += 1;
            continue;
          }
          const event = addEvent({
            eventKey: `tool-progress:${lifecycle.invocationId}`,
            projectId,
            sessionId,
            turnId: originalTurn.turnId,
            stepId: lifecycle.stepId,
            correlationId: originalTurn.correlationId,
            causationId: lifecycle.startedEventId,
            source: "session",
            sourceSequence: record.sequence,
            timestamp,
            type: "tool_progress",
            outcome: "observed",
            summary: `Tool ${lifecycle.toolName} reported progress.`,
            actor: { kind: "tool", id: safeOptionalIdentifier(lifecycle.toolName, "tool") },
            toolName: lifecycle.toolName,
            late,
            inferred: false,
            observationCount: 1,
          });
          progressEvents.set(lifecycle.invocationId, event);
          continue;
        }
        seenLifecycle.add(lifecycleKey);
        const status = message.render?.status;
        const outcome =
          status === "failed"
            ? "failed"
            : status === "canceled"
              ? "cancelled"
              : status === "skipped"
                ? "unknown"
                : "succeeded";
        const event = addEvent({
          eventKey: `tool-result:${lifecycle.invocationId}:${outcome}`,
          projectId,
          sessionId,
          turnId: originalTurn.turnId,
          stepId: lifecycle.stepId,
          correlationId: originalTurn.correlationId,
          causationId: lifecycle.startedEventId,
          source: "session",
          sourceSequence: record.sequence,
          timestamp,
          type: "tool_finished",
          outcome,
          summary:
            outcome === "failed"
              ? `Tool ${lifecycle.toolName} failed.`
              : outcome === "cancelled"
                ? `Tool ${lifecycle.toolName} was cancelled.`
                : `Tool ${lifecycle.toolName} finished.`,
          actor: { kind: "tool", id: safeOptionalIdentifier(lifecycle.toolName, "tool") },
          toolName: lifecycle.toolName,
          late,
          inferred: false,
          observationCount: 1,
        });
        lifecycle.settled = true;
        if (step) step.status = outcome === "succeeded" ? "succeeded" : outcome;
        originalTurn.eventIds.push(event.eventId);
        continue;
      }

      if (message.kind === "message" && message.role === "assistant") {
        const step = addStep(turn, "model", `model:${record.sequence}:${messageIndex}`, false);
        addEvent({
          eventKey: `model:${record.sequence}:${messageIndex}`,
          projectId,
          sessionId,
          turnId: turn.turnId,
          stepId: step.stepId,
          correlationId: turn.correlationId,
          causationId: lastEventId,
          source: "session",
          sourceSequence: record.sequence,
          timestamp,
          type: "model_response",
          outcome: "completed",
          summary: "A model response completed a Step.",
          actor: { kind: "agent", id: safeOptionalIdentifier(message.agent, "agent") },
          late: false,
          inferred: false,
          observationCount: 1,
        });
        step.status = "succeeded";
        turn.finalResponseObserved = true;
      }
    }
  }

  for (const auditEvent of [...audit].sort((left, right) => left.sequence - right.sequence)) {
    if (seenAudit.has(auditEvent.eventId)) {
      addDiagnostic({
        code: "duplicate_transport",
        summary: "A repeated audit transport record was collapsed without replaying an effect.",
        source: "audit",
        sourceSequence: auditEvent.sequence,
      });
      continue;
    }
    seenAudit.add(auditEvent.eventId);
    if (projectActivationAudit(auditEvent)) continue;
    const requestKey = auditEvent.requestId
      ? safeIdentifier(auditEvent.requestId, "request")
      : undefined;
    const turn = requestKey
      ? turnsByRequest.get(requestKey)
      : turns.size === 1
        ? [...turns.values()][0]
        : undefined;
    if (!turn) {
      addDiagnostic({
        code: "unlinked_audit_event",
        summary: "A relevant audit event had no provable Session Turn correlation.",
        source: "audit",
        sourceSequence: auditEvent.sequence,
      });
      continue;
    }
    const timestamp = safeTimestamp(
      auditEvent.timestamp,
      auditEvent.sequence,
      addDiagnostic,
      "audit",
    );
    const base = {
      projectId,
      sessionId,
      turnId: turn.turnId,
      correlationId: turn.correlationId,
      source: "audit" as const,
      sourceSequence: auditEvent.sequence,
      sourceEventId: auditEvent.eventId,
      timestamp,
      actor: auditActor(auditEvent),
      late: false,
      inferred: false,
      observationCount: 1,
    };

    if (auditEvent.action === "session.late_chunk_observed") {
      const boundary = auditString(auditEvent.metadata.boundary) ?? "closed";
      const key = `late-chunk:${turn.turnId}:${boundary}`;
      const observationCount = auditNumber(auditEvent.metadata.observationCount);
      const existing = lateChunkEvents.get(key);
      if (existing) {
        existing.observationCount += observationCount;
        continue;
      }
      const event = addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "late_chunk_observed",
        outcome: "observed",
        summary: "A chunk arrived after the foreground output barrier closed.",
        late: true,
        observationCount,
      });
      lateChunkEvents.set(key, event);
      continue;
    }

    if (auditEvent.category === "permission" && auditEvent.action === "tool.decision") {
      const laterTerminalDecision = audit.some(
        (candidate) =>
          candidate.sequence > auditEvent.sequence &&
          candidate.category === auditEvent.category &&
          candidate.action === auditEvent.action &&
          candidate.requestId === auditEvent.requestId &&
          candidate.target?.id === auditEvent.target?.id &&
          candidate.outcome !== "attempted",
      );
      if (auditEvent.outcome === "attempted" && laterTerminalDecision) continue;
      const toolName = safeToolName(auditEvent.target?.id);
      const lifecycle = [...tools.values()]
        .reverse()
        .find(
          (item) =>
            item.turnId === turn.turnId && (toolName === "tool" || item.toolName === toolName),
        );
      const step = lifecycle
        ? steps.get(lifecycle.stepId)
        : addStep(turn, "approval", `approval:${auditEvent.eventId}`, true, { toolName });
      if (!step) continue;
      const decision = auditString(auditEvent.metadata.decision);
      const denied =
        auditEvent.outcome === "denied" || decision === "denied" || decision === "reject";
      const pending = auditEvent.outcome === "attempted";
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        stepId: step.stepId,
        causationId: lifecycle?.startedEventId ?? turn.openedByEventId,
        type: "approval_decided",
        outcome: pending ? "pending" : denied ? "denied" : "allowed",
        summary: pending
          ? `Permission for tool ${toolName} awaits a recorded outcome.`
          : denied
            ? `Permission for tool ${toolName} was denied.`
            : `Permission for tool ${toolName} was allowed.`,
        toolName,
      });
      if (denied && lifecycle && !lifecycle.settled) {
        lifecycle.settled = true;
        step.status = "cancelled";
      }
      continue;
    }

    if (auditEvent.category === "task") {
      const taskId = safeIdentifier(
        auditEvent.taskId ?? auditEvent.target?.id ?? auditEvent.eventId,
        "task",
      );
      const existing = pendingTasks.get(taskId);
      const step = existing
        ? steps.get(existing.stepId)
        : addStep(turn, "task", `task:${taskId}`, false);
      if (!step) continue;
      const terminal = ["completed", "failed", "denied", "cancelled"].includes(auditEvent.outcome);
      const outcome = auditOutcome(auditEvent.outcome);
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        stepId: step.stepId,
        causationId: turn.eventIds.at(-1),
        type: "task_observed",
        outcome,
        summary: terminal
          ? `Child task ${taskId} reached ${outcome}.`
          : `Child task ${taskId} was observed.`,
      });
      pendingTasks.set(taskId, { taskId, turnId: turn.turnId, stepId: step.stepId, terminal });
      step.status = terminal ? terminalStepStatus(outcome) : "pending";
      continue;
    }

    if (auditEvent.action.includes("cancel")) {
      const outcome = auditEvent.outcome === "cancel_requested" ? "cancel_requested" : "cancelled";
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "turn_state",
        outcome,
        summary:
          outcome === "cancel_requested"
            ? "Turn cancellation was requested."
            : "The Turn was cancelled.",
      });
      turn.status = outcome === "cancelled" ? "cancelled" : "active";
      turn.statusReason =
        outcome === "cancelled"
          ? "A correlated audit event records cancellation."
          : "Cancellation is requested but not settled.";
      continue;
    }
    if (auditEvent.action.includes("resume")) {
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "turn_state",
        outcome: "resumed",
        summary: "The Turn resumed from existing Session state.",
      });
      turn.status = "active";
      turn.statusReason = "A correlated audit event records resumption.";
      continue;
    }
    if (auditEvent.action.includes("retry")) {
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "turn_state",
        outcome: "retried",
        summary: "The Turn was retried without assuming a second side effect.",
      });
      turn.retryCount += 1;
      turn.status = "active";
      turn.statusReason = "A correlated audit event records a retry.";
      continue;
    }
    if (auditEvent.action.includes("pause") || auditEvent.action.includes("human_needed")) {
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "turn_state",
        outcome: "pending",
        summary: "The Turn paused pending an external decision or result.",
      });
      turn.status = "active";
      turn.statusReason = "A correlated audit event records a pause that is not settled.";
      continue;
    }
    if (auditEvent.category === "session" && auditEvent.action.startsWith("acp.")) {
      const step = addStep(turn, "external", `external:${auditEvent.eventId}`, false);
      const outcome = auditOutcome(auditEvent.outcome);
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        stepId: step.stepId,
        causationId: turn.eventIds.at(-1),
        type: "external_operation",
        outcome,
        summary: `External Harness operation ${auditEvent.action} was ${outcome}.`,
      });
      step.status = terminalStepStatus(outcome);
      if (outcome === "failed") {
        turn.status = "failed";
        turn.statusReason = "A correlated external Harness operation failed.";
      }
      continue;
    }
    if (
      auditEvent.outcome === "failed" &&
      (auditEvent.category === "session" || auditEvent.category === "system")
    ) {
      addEvent({
        ...base,
        eventKey: `audit:${auditEvent.eventId}`,
        causationId: turn.eventIds.at(-1),
        type: "turn_state",
        outcome: "failed",
        summary: "The Turn failed according to correlated audit evidence.",
      });
      turn.status = "failed";
      turn.statusReason = "A correlated audit event records a terminal failure.";
    }
  }

  const unsettled: SessionTimelineUnsettled[] = [];
  for (const lifecycle of tools.values()) {
    if (lifecycle.settled || turns.get(lifecycle.turnId)?.settled) continue;
    unsettled.push({
      pendingId: deterministicId("pending", sessionId, lifecycle.turnId, lifecycle.stepId),
      turnId: lifecycle.turnId,
      stepId: lifecycle.stepId,
      kind: "tool_result",
      summary: `Tool ${lifecycle.toolName} has no settled result in canonical history.`,
    });
  }
  for (const task of pendingTasks.values()) {
    if (task.terminal || turns.get(task.turnId)?.settled) continue;
    unsettled.push({
      pendingId: deterministicId("pending", sessionId, task.turnId, task.stepId),
      turnId: task.turnId,
      stepId: task.stepId,
      kind: "task_outcome",
      summary: `Child task ${task.taskId} has no terminal audit outcome.`,
    });
  }

  for (const turn of turns.values()) {
    if (turn.settled || turn.status === "cancelled" || turn.status === "failed") continue;
    const pending = unsettled.some((item) => item.turnId === turn.turnId);
    if (pending) {
      turn.status = "active";
      turn.statusReason = "Structured tool or child-task work remains unsettled.";
    } else if (turn.finalResponseObserved) {
      turn.status = "completed";
      turn.statusReason = "A final model response follows all observed structured work.";
    } else {
      turn.status = "unknown";
      turn.statusReason = "Canonical records do not prove a terminal Turn outcome.";
    }
  }

  const orderedEvents = orderPendingEvents(pendingEvents).map((item, index) =>
    SessionTimelineEventSchema.parse({ ...item.event, ordinal: index + 1 }),
  );
  const eventOrder = new Map(orderedEvents.map((event) => [event.eventId, event.ordinal]));
  const outputWithoutFingerprint = {
    schemaVersion: 1 as const,
    sessionId,
    ...(projectId ? { projectId } : {}),
    authority: "derived_diagnostic_projection" as const,
    events: orderedEvents,
    turns: [...turns.values()]
      .map(({ finalResponseObserved: _finalResponseObserved, settled: _settled, ...turn }) => ({
        ...turn,
        eventIds: uniqueSortedIds(turn.eventIds, eventOrder),
        stepIds: [...new Set(turn.stepIds)],
      }))
      .sort(
        (left, right) =>
          (eventOrder.get(left.openedByEventId) ?? Number.MAX_SAFE_INTEGER) -
          (eventOrder.get(right.openedByEventId) ?? Number.MAX_SAFE_INTEGER),
      ),
    steps: [...steps.values()]
      .map((step) => ({ ...step, eventIds: uniqueSortedIds(step.eventIds, eventOrder) }))
      .sort(
        (left, right) =>
          (eventOrder.get(left.eventIds[0] ?? "") ?? Number.MAX_SAFE_INTEGER) -
            (eventOrder.get(right.eventIds[0] ?? "") ?? Number.MAX_SAFE_INTEGER) ||
          left.stepId.localeCompare(right.stepId),
      ),
    unsettled: unsettled.sort((left, right) => left.pendingId.localeCompare(right.pendingId)),
    diagnostics: diagnostics.sort(
      (left, right) =>
        left.source.localeCompare(right.source) ||
        (left.sourceSequence ?? 0) - (right.sourceSequence ?? 0) ||
        left.code.localeCompare(right.code),
    ),
  };
  return SessionTimelineSchema.parse({
    ...outputWithoutFingerprint,
    fingerprint: `timeline_${stableHash(stableJson(outputWithoutFingerprint))}`,
  });
}

function deterministicId(prefix: string, ...parts: Array<string | number>): string {
  return `${prefix}_${stableHash(parts.join("\u001f"))}`;
}

function safeIdentifier(value: string, namespace: string): string {
  const normalized = value.trim();
  if (
    /^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$/.test(normalized) &&
    !/(?:secret|password|passwd|credential|bearer|private[_-]?key)/i.test(normalized)
  ) {
    return normalized;
  }
  return deterministicId(namespace, normalized);
}

function safeOptionalIdentifier(value: string | undefined, namespace: string): string | undefined {
  return value ? safeIdentifier(value, namespace) : undefined;
}

function safeToolName(value: string | undefined): string {
  const normalized = value?.trim();
  return normalized && /^[A-Za-z][A-Za-z0-9_.:/-]{0,63}$/.test(normalized) ? normalized : "tool";
}

function safeTimestamp(
  value: string,
  sequence: number,
  addDiagnostic: (diagnostic: SessionTimelineDiagnostic) => void,
  source: "session" | "audit" = "session",
): string {
  const parsed = new Date(value);
  if (Number.isFinite(parsed.getTime())) return parsed.toISOString();
  addDiagnostic({
    code: "invalid_timestamp",
    summary: "An invalid source timestamp was replaced with a deterministic epoch value.",
    source,
    sourceSequence: sequence,
  });
  return new Date(sequence).toISOString();
}

function auditActor(event: SessionTimelineAuditRecord): SessionTimelineEvent["actor"] {
  if (!event.actor) return { kind: "system" };
  return {
    kind: event.actor.kind,
    ...(event.actor.id ? { id: safeIdentifier(event.actor.id, "actor") } : {}),
  };
}

function auditString(value: unknown): string | undefined {
  return typeof value === "string" ? value.toLowerCase() : undefined;
}

function auditNumber(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : 1;
}

function auditOutcome(
  outcome: SessionTimelineAuditRecord["outcome"],
): SessionTimelineEvent["outcome"] {
  if (outcome === "attempted") return "pending";
  if (outcome === "completed") return "completed";
  return outcome;
}

function terminalStepStatus(
  outcome: SessionTimelineEvent["outcome"],
): SessionTimelineStep["status"] {
  if (outcome === "completed" || outcome === "succeeded" || outcome === "allowed")
    return "succeeded";
  if (outcome === "failed" || outcome === "denied") return "failed";
  if (outcome === "cancelled") return "cancelled";
  if (outcome === "pending" || outcome === "cancel_requested") return "pending";
  return "unknown";
}

function uniqueSortedIds(ids: string[], order: ReadonlyMap<string, number>): string[] {
  return [...new Set(ids)].sort(
    (left, right) =>
      (order.get(left) ?? Number.MAX_SAFE_INTEGER) - (order.get(right) ?? Number.MAX_SAFE_INTEGER),
  );
}

function orderPendingEvents(events: PendingEvent[]): PendingEvent[] {
  const byId = new Map(events.map((item) => [item.event.eventId, item]));
  const children = new Map(events.map((item) => [item.event.eventId, new Set<string>()]));
  const indegree = new Map(events.map((item) => [item.event.eventId, 0]));
  for (const item of events) {
    const cause = item.event.causationId;
    if (!cause || !byId.has(cause) || cause === item.event.eventId) continue;
    children.get(cause)?.add(item.event.eventId);
    indegree.set(item.event.eventId, (indegree.get(item.event.eventId) ?? 0) + 1);
  }
  const compare = (left: PendingEvent, right: PendingEvent): number =>
    left.sortTimestamp.localeCompare(right.sortTimestamp) ||
    left.event.source.localeCompare(right.event.source) ||
    left.event.sourceSequence - right.event.sourceSequence ||
    left.localOrder - right.localOrder;
  const ready = events.filter((item) => indegree.get(item.event.eventId) === 0).sort(compare);
  const ordered: PendingEvent[] = [];
  while (ready.length > 0) {
    const item = ready.shift();
    if (!item) break;
    ordered.push(item);
    for (const childId of children.get(item.event.eventId) ?? []) {
      const next = (indegree.get(childId) ?? 1) - 1;
      indegree.set(childId, next);
      if (next !== 0) continue;
      const child = byId.get(childId);
      if (child) {
        ready.push(child);
        ready.sort(compare);
      }
    }
  }
  if (ordered.length === events.length) return ordered;
  const emitted = new Set(ordered.map((item) => item.event.eventId));
  return [...ordered, ...events.filter((item) => !emitted.has(item.event.eventId)).sort(compare)];
}

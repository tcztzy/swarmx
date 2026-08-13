import { z } from "zod";
import { stableHash, stableJson } from "./canonical-json.js";
import { findInlineSecretFields } from "./secret-scanner.js";
import type { MessageChunk } from "./types.js";
import { MessageChunkSchema } from "./types.js";

const NON_MODEL_REPLAY_MESSAGE_SOURCES = new Set([
  "personal_memory_receipt",
  "project_bootstrap_receipt",
]);

const idWithPrefix = (prefix: string) =>
  z.string().regex(new RegExp(`^${prefix}[A-Za-z0-9][A-Za-z0-9_-]*$`), `Must use ${prefix} prefix`);

export const ConversationEventTypeSchema = z.enum([
  "session_created",
  "message_appended",
  "messages_appended",
  "summary_checkpoint_appended",
  "agent_invocation_started",
  "agent_invocation_completed",
  "artifact_linked",
  "session_title_updated",
  "session_archived",
]);

export const ConversationActorSchema = z.enum(["user", "assistant", "system", "tool", "host"]);

export const ConversationContextStrategySchema = z.enum([
  "auto",
  "checkpoint_tail",
  "microcompact",
  "full_tail",
  "isolated",
]);

export const ConversationStorageRefSchema = z
  .object({
    root: z.enum(["desktop", "server", "benchmark", "custom"]).default("desktop"),
    rootRef: z.string().min(1).optional(),
    indexPath: z.string().min(1).optional(),
    rolloutPath: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ConversationIndexRecordSchema = z
  .object({
    sessionId: z.string().min(1),
    title: z.string().min(1),
    agentName: z.string().min(1),
    harnessId: z.string().min(1),
    model: z.string().min(1).optional(),
    createdAt: z.string().min(1),
    updatedAt: z.string().min(1),
    messageCount: z.number().int().nonnegative().default(0),
    archived: z.boolean().default(false),
    contextStrategy: ConversationContextStrategySchema.default("auto"),
    storage: ConversationStorageRefSchema.prefault({}),
    metadata: z.record(z.string(), z.unknown()).prefault({}),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ConversationArtifactReferenceSchema = z
  .object({
    artifactId: z.string().min(1).optional(),
    kind: z.string().min(1).default("other"),
    path: z.string().min(1).optional(),
    url: z.string().min(1).optional(),
    title: z.string().min(1).optional(),
    checksum: z.string().min(1).optional(),
  })
  .passthrough()
  .superRefine(addSecretIssues);

export const ConversationEventSchema = z
  .object({
    eventId: idWithPrefix("cev_"),
    sessionId: z.string().min(1),
    timestamp: z.string().min(1),
    eventType: ConversationEventTypeSchema,
    actor: ConversationActorSchema.default("host"),
    session: ConversationIndexRecordSchema.optional(),
    message: MessageChunkSchema.optional(),
    messages: z.array(MessageChunkSchema).optional(),
    title: z.string().min(1).optional(),
    contextPacketId: z.string().min(1).optional(),
    checkpointId: z.string().min(1).optional(),
    invocationId: z.string().min(1).optional(),
    renderEventIds: z.array(z.string().min(1)).default([]),
    artifacts: z.array(ConversationArtifactReferenceSchema).default([]),
    payload: z.record(z.string(), z.unknown()).prefault({}),
  })
  .passthrough()
  .superRefine(addConversationEventIssues);

export const ConversationReplayStateSchema = z
  .object({
    sessions: z.record(z.string(), ConversationIndexRecordSchema).prefault({}),
    messagesBySession: z.record(z.string(), z.array(MessageChunkSchema)).prefault({}),
    rejectedEvents: z.array(ConversationEventSchema).default([]),
  })
  .passthrough();

export type ConversationEventType = z.infer<typeof ConversationEventTypeSchema>;
export type ConversationActor = z.infer<typeof ConversationActorSchema>;
export type ConversationContextStrategy = z.infer<typeof ConversationContextStrategySchema>;
export type ConversationStorageRef = z.infer<typeof ConversationStorageRefSchema>;
export type ConversationIndexRecord = z.infer<typeof ConversationIndexRecordSchema>;
export type ConversationArtifactReference = z.infer<typeof ConversationArtifactReferenceSchema>;
export type ConversationEvent = z.infer<typeof ConversationEventSchema>;
export type ConversationReplayState = z.infer<typeof ConversationReplayStateSchema>;

export type CreateConversationEventInput = Omit<
  ConversationEvent,
  "actor" | "artifacts" | "eventId" | "payload" | "renderEventIds"
> & {
  eventId?: string;
  actor?: ConversationActor;
  renderEventIds?: string[];
  artifacts?: ConversationArtifactReference[];
  payload?: Record<string, unknown>;
};

export function parseConversationIndexRecord(input: unknown): ConversationIndexRecord {
  return ConversationIndexRecordSchema.parse(input);
}

export function parseConversationEvent(input: unknown): ConversationEvent {
  return ConversationEventSchema.parse(input);
}

export function createConversationEvent(input: CreateConversationEventInput): ConversationEvent {
  const withoutId = {
    sessionId: input.sessionId,
    timestamp: input.timestamp,
    eventType: input.eventType,
    actor: input.actor ?? "host",
    session: input.session,
    message: input.message,
    messages: input.messages,
    title: input.title,
    contextPacketId: input.contextPacketId,
    checkpointId: input.checkpointId,
    invocationId: input.invocationId,
    renderEventIds: input.renderEventIds ?? [],
    artifacts: input.artifacts ?? [],
    payload: input.payload ?? {},
  };

  return ConversationEventSchema.parse({
    ...withoutId,
    eventId: input.eventId ?? `cev_${stableHash(stableJson(withoutId))}`,
  });
}

export function conversationJsonlLine(input: unknown): string {
  return `${stableJson(ConversationEventSchema.parse(input))}\n`;
}

export function parseConversationJsonl(input: string): ConversationEvent[] {
  return input.split(/\r?\n/).flatMap((line, index) => {
    const trimmed = line.trim();
    if (!trimmed) return [];
    try {
      return [ConversationEventSchema.parse(JSON.parse(trimmed))];
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Invalid conversation JSONL line ${index + 1}: ${message}`);
    }
  });
}

export function emptyConversationReplayState(): ConversationReplayState {
  return ConversationReplayStateSchema.parse({});
}

/** Keeps host receipts in canonical history while excluding them from later model context. */
export function modelReplayableMessages(messages: readonly MessageChunk[]): MessageChunk[] {
  return messages.filter(
    (message) =>
      !message.render?.source || !NON_MODEL_REPLAY_MESSAGE_SOURCES.has(message.render.source),
  );
}

export function replayConversationEvents(eventsInput: unknown[]): ConversationReplayState {
  return eventsInput.reduce<ConversationReplayState>(
    (state, input) => applyConversationEvent(state, input),
    emptyConversationReplayState(),
  );
}

export function applyConversationEvent(
  stateInput: unknown,
  eventInput: unknown,
): ConversationReplayState {
  const state = ConversationReplayStateSchema.parse(stateInput);
  const event = ConversationEventSchema.parse(eventInput);
  const existing = state.sessions[event.sessionId];

  if (event.eventType === "session_created") {
    if (!event.session || existing) return rejectEvent(state, event);
    return ConversationReplayStateSchema.parse({
      ...state,
      sessions: {
        ...state.sessions,
        [event.sessionId]: {
          ...event.session,
          sessionId: event.sessionId,
          updatedAt: event.session.updatedAt || event.timestamp,
        },
      },
      messagesBySession: {
        ...state.messagesBySession,
        [event.sessionId]: [],
      },
    });
  }

  if (!existing) return rejectEvent(state, event);

  if (event.eventType === "message_appended") {
    if (!event.message) return rejectEvent(state, event);
    return appendMessagesToState(state, event, [event.message]);
  }

  if (event.eventType === "messages_appended") {
    if (!event.messages || event.messages.length === 0) return rejectEvent(state, event);
    return appendMessagesToState(state, event, event.messages);
  }

  if (event.eventType === "session_title_updated") {
    if (!event.title) return rejectEvent(state, event);
    return updateSession(state, event.sessionId, {
      title: event.title,
      updatedAt: event.timestamp,
    });
  }

  if (event.eventType === "session_archived") {
    return updateSession(state, event.sessionId, {
      archived: true,
      updatedAt: event.timestamp,
    });
  }

  return updateSession(state, event.sessionId, {
    updatedAt: event.timestamp,
  });
}

function appendMessagesToState(
  state: ConversationReplayState,
  event: ConversationEvent,
  messages: MessageChunk[],
): ConversationReplayState {
  const currentMessages = state.messagesBySession[event.sessionId] ?? [];
  const nextMessages = [...currentMessages, ...messages];
  return ConversationReplayStateSchema.parse({
    ...state,
    messagesBySession: {
      ...state.messagesBySession,
      [event.sessionId]: nextMessages,
    },
    sessions: {
      ...state.sessions,
      [event.sessionId]: {
        ...state.sessions[event.sessionId],
        updatedAt: event.timestamp,
        messageCount: nextMessages.length,
      },
    },
  });
}

function updateSession(
  state: ConversationReplayState,
  sessionId: string,
  patch: Partial<ConversationIndexRecord>,
): ConversationReplayState {
  const session = state.sessions[sessionId];
  if (!session) return state;
  return ConversationReplayStateSchema.parse({
    ...state,
    sessions: {
      ...state.sessions,
      [sessionId]: {
        ...session,
        ...patch,
      },
    },
  });
}

function rejectEvent(
  state: ConversationReplayState,
  event: ConversationEvent,
): ConversationReplayState {
  return ConversationReplayStateSchema.parse({
    ...state,
    rejectedEvents: [...state.rejectedEvents, event],
  });
}

function addConversationEventIssues(event: unknown, ctx: z.RefinementCtx): void {
  addSecretIssues(event, ctx);
  if (!isObjectRecord(event)) return;

  const eventType = event.eventType;
  if (eventType === "session_created" && !event.session) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["session"],
      message: "session_created events must include a session index record.",
    });
  }
  if (eventType === "message_appended" && !event.message) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["message"],
      message: "message_appended events must include a message.",
    });
  }
  if (
    eventType === "messages_appended" &&
    (!Array.isArray(event.messages) || event.messages.length === 0)
  ) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["messages"],
      message: "messages_appended events must include one or more messages.",
    });
  }
  if (eventType === "session_title_updated" && !event.title) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ["title"],
      message: "session_title_updated events must include a title.",
    });
  }
}

function addSecretIssues(value: unknown, ctx: z.RefinementCtx): void {
  for (const issue of findInlineSecretFields(value, { allowRedacted: true })) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: issue.path,
      message: `Conversation records must not contain inline secret field "${issue.key}".`,
    });
  }
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

import { createHash } from "node:crypto";
import { z } from "zod";
import { stableJson } from "./canonical-json.js";
import { findInlineSecretFields } from "./secret-scanner.js";
import { type MessageChunk, MessageChunkSchema } from "./types.js";

const EVENT_ID_PATTERN = /^evt_[A-Za-z0-9][A-Za-z0-9_-]*$/u;
const SHA256_PATTERN = /^sha256:[a-f0-9]{64}$/u;

export const ContextEventKindSchema = z.enum([
  "user_message",
  "assistant_message",
  "tool_call",
  "tool_result",
  "patch",
  "test_result",
  "decision",
  "checkpoint",
  "compaction",
]);

export const ContextArtifactReferenceSchema = z
  .object({
    uri: z.string().regex(/^artifact:\/\/sha256\/[a-f0-9]{64}$/u),
    contentHash: z.string().regex(SHA256_PATTERN),
    sizeBytes: z.number().int().nonnegative(),
    mediaType: z.string().min(1).max(256).optional(),
  })
  .strict();

export const ContextEventMetadataSchema = z
  .object({
    paths: z.array(z.string().min(1)).default([]),
    symbols: z.array(z.string().min(1)).default([]),
    branch: z.string().min(1).optional(),
    commitSha: z.string().min(1).optional(),
    exitCode: z.number().int().optional(),
    errorSignature: z.string().min(1).optional(),
  })
  .strict();

const ContextEngineEventContentSchema = z
  .object({
    id: z.string().regex(EVENT_ID_PATTERN),
    seq: z.number().int().nonnegative(),
    sessionId: z.string().min(1),
    taskId: z.string().min(1).optional(),
    turnId: z.string().min(1),
    timestamp: z.string().datetime({ offset: true }),
    kind: ContextEventKindSchema,
    toolCallId: z.string().min(1).optional(),
    payload: z.unknown().optional(),
    artifactRef: ContextArtifactReferenceSchema.optional(),
    causalParents: z.array(z.string().regex(EVENT_ID_PATTERN)).default([]),
    supersedes: z.array(z.string().regex(EVENT_ID_PATTERN)).default([]),
    labels: z.array(z.string().min(1)).default([]),
    metadata: ContextEventMetadataSchema.prefault({}),
  })
  .strict()
  .superRefine((value, ctx) => {
    for (const issue of findInlineSecretFields(value)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: issue.path,
        message: `Context events must not contain inline secret field "${issue.key}".`,
      });
    }
    if ((value.kind === "tool_call" || value.kind === "tool_result") && !value.toolCallId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["toolCallId"],
        message: `${value.kind} requires toolCallId.`,
      });
    }
    if (value.kind !== "tool_call" && value.kind !== "tool_result" && value.toolCallId) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["toolCallId"],
        message: `${value.kind} cannot carry toolCallId.`,
      });
    }
  });

export const ContextEngineEventSchema = ContextEngineEventContentSchema.extend({
  contentHash: z.string().regex(SHA256_PATTERN),
})
  .strict()
  .superRefine((value, ctx) => {
    const { contentHash: _contentHash, ...content } = value;
    const expected = contextContentHash(content);
    if (value.contentHash !== expected) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["contentHash"],
        message: "Context event contentHash must bind the complete event content.",
      });
    }
  });

export type ContextEventKind = z.infer<typeof ContextEventKindSchema>;
export type ContextArtifactReference = z.infer<typeof ContextArtifactReferenceSchema>;
export type ContextEventMetadata = z.infer<typeof ContextEventMetadataSchema>;
export type ContextEngineEvent = z.infer<typeof ContextEngineEventSchema>;
export type ContextEngineEventInput = z.input<typeof ContextEngineEventContentSchema> & {
  contentHash?: string;
};

export interface ContextHistorySnapshot {
  snapshotId: string;
  sessionId: string;
  firstSeq: number | null;
  lastSeq: number | null;
  eventCount: number;
  events: ContextEngineEvent[];
}

export interface CreateContextHistorySnapshotOptions {
  validateAtomicTools?: boolean;
}

export interface ContextEventScanSpec {
  sessionId?: string;
  taskId?: string;
  afterSeq?: number;
  beforeSeq?: number;
  kinds?: ContextEventKind[];
  limit?: number;
}

export interface ContextEventStore {
  append(events: unknown | readonly unknown[]): void;
  get(eventId: string): ContextEngineEvent | undefined;
  scan(spec?: ContextEventScanSpec): ContextEngineEvent[];
  snapshot(scope: { sessionId: string }): ContextHistorySnapshot;
}

export interface ContextArtifactStore {
  put(content: Uint8Array, options?: { mediaType?: string }): ContextArtifactReference;
  readRange(
    reference: ContextArtifactReference,
    range: { startByte: number; endByte?: number },
  ): Uint8Array;
  preview(
    reference: ContextArtifactReference,
    options?: { maxBytes?: number },
  ): { text: string; truncated: boolean };
}

export function createContextEvent(input: ContextEngineEventInput): ContextEngineEvent {
  const { contentHash: suppliedHash, ...candidate } = input;
  const content = ContextEngineEventContentSchema.parse(candidate);
  const contentHash = contextContentHash(content);
  if (suppliedHash !== undefined && suppliedHash !== contentHash) {
    throw new Error("Supplied context event contentHash does not match its content.");
  }
  return ContextEngineEventSchema.parse({ ...content, contentHash });
}

export function createContextHistorySnapshot(
  input: readonly unknown[],
  options: CreateContextHistorySnapshotOptions = {},
): ContextHistorySnapshot {
  const events = input.map((item) => ContextEngineEventSchema.parse(item));
  validateEventOrder(events);
  if (options.validateAtomicTools !== false) validateToolReferences(events);
  const sessionIds = new Set(events.map((event) => event.sessionId));
  if (sessionIds.size > 1) throw new Error("A context snapshot must contain one session.");
  const sessionId = events[0]?.sessionId ?? "empty";
  const snapshotContent = events.map((event) => [event.id, event.seq, event.contentHash]);
  return {
    snapshotId: `snapshot_${sha256Hex(stableJson(snapshotContent))}`,
    sessionId,
    firstSeq: events[0]?.seq ?? null,
    lastSeq: events.at(-1)?.seq ?? null,
    eventCount: events.length,
    events,
  };
}

export type ContextAtomicUnitKind = "event" | "tool_exchange";
export type ContextAtomicUnitStatus = "pending" | "succeeded" | "failed";

export interface ContextAtomicUnit {
  unitId: string;
  kind: ContextAtomicUnitKind;
  status: ContextAtomicUnitStatus;
  startSeq: number;
  endSeq: number;
  eventIds: string[];
  events: ContextEngineEvent[];
}

export interface ContextNormalizer {
  normalize(events: readonly unknown[]): ContextAtomicUnit[];
}

export function normalizeContextEvents(input: readonly unknown[]): ContextAtomicUnit[] {
  const snapshot = createContextHistorySnapshot(input);
  const resultsByCallId = new Map<string, ContextEngineEvent>();
  const callsByCallId = new Map<string, ContextEngineEvent>();

  for (const event of snapshot.events) {
    if (event.kind === "tool_call") {
      const callId = event.toolCallId as string;
      if (callsByCallId.has(callId)) throw new Error(`Duplicate tool call id: ${callId}`);
      callsByCallId.set(callId, event);
    } else if (event.kind === "tool_result") {
      const callId = event.toolCallId as string;
      if (resultsByCallId.has(callId)) throw new Error(`Duplicate tool result id: ${callId}`);
      resultsByCallId.set(callId, event);
    }
  }

  const consumed = new Set<string>();
  const units: ContextAtomicUnit[] = [];
  for (const event of snapshot.events) {
    if (consumed.has(event.id) || event.kind === "tool_result") continue;
    if (event.kind === "tool_call") {
      const result = resultsByCallId.get(event.toolCallId as string);
      const events = result ? [event, result] : [event];
      for (const child of events) consumed.add(child.id);
      units.push({
        unitId: `unit_${event.id}`,
        kind: "tool_exchange",
        status: result ? statusFromEvent(result) : "pending",
        startSeq: event.seq,
        endSeq: result?.seq ?? event.seq,
        eventIds: events.map((child) => child.id),
        events,
      });
      continue;
    }
    consumed.add(event.id);
    units.push({
      unitId: `unit_${event.id}`,
      kind: "event",
      status: statusFromEvent(event),
      startSeq: event.seq,
      endSeq: event.seq,
      eventIds: [event.id],
      events: [event],
    });
  }
  return units.sort((left, right) => left.startSeq - right.startSeq);
}

export const ContextVisibilitySchema = z.enum(["full", "capsule", "ref", "omit"]);
export type ContextVisibility = z.infer<typeof ContextVisibilitySchema>;

export interface ContextMaskPolicy {
  currentTurnId?: string;
  preserveRecentAtomicUnits?: number;
  evidenceEventIds?: string[];
  latestFailureEventId?: string;
  capsuleMaxChars?: number;
}

export interface MaskedContextUnit {
  unit: ContextAtomicUnit;
  visibility: ContextVisibility;
  rendered: string;
}

export interface ContextObservationMasker {
  render(units: readonly ContextAtomicUnit[], policy: ContextMaskPolicy): MaskedContextUnit[];
}

const MANDATORY_LABELS = new Set([
  "pinned",
  "constraint",
  "task_contract",
  "blocker",
  "uncommitted",
]);

export function maskContextUnits(
  units: readonly ContextAtomicUnit[],
  policy: ContextMaskPolicy,
): MaskedContextUnit[] {
  const recentCount = Math.max(0, policy.preserveRecentAtomicUnits ?? 6);
  const recentIds = new Set(units.slice(-recentCount).map((unit) => unit.unitId));
  const evidenceIds = new Set(policy.evidenceEventIds ?? []);
  const supersededIds = new Set(
    units.flatMap((unit) => unit.events.flatMap((event) => event.supersedes)),
  );

  return units.map((unit) => {
    const current = unit.events.some((event) => event.turnId === policy.currentTurnId);
    const mandatory = unit.events.some(
      (event) =>
        event.kind === "decision" ||
        event.kind === "checkpoint" ||
        event.labels.some((label) => MANDATORY_LABELS.has(label)),
    );
    const latestFailure = unit.eventIds.includes(policy.latestFailureEventId ?? "");
    const retrieved = unit.eventIds.some((id) => evidenceIds.has(id));
    const superseded = unit.eventIds.every((id) => supersededIds.has(id));
    const artifact = unit.events.find((event) => event.artifactRef)?.artifactRef;

    let visibility: ContextVisibility;
    if (current || mandatory || latestFailure) visibility = "full";
    else if (retrieved || recentIds.has(unit.unitId)) visibility = "capsule";
    else if (superseded && unit.status === "succeeded") visibility = "omit";
    else if (artifact) visibility = "ref";
    else if (unit.status === "succeeded") visibility = "omit";
    else visibility = "capsule";

    return {
      unit,
      visibility,
      rendered: renderMaskedUnit(unit, visibility, policy.capsuleMaxChars ?? 800),
    };
  });
}

export interface SourcedField<T> {
  value: T;
  basis: "observed" | "inferred";
  sourceEventIds: string[];
  validAtSeq: number;
  supersededBy?: string;
}

export interface ContextDecisionProjection {
  statement: string;
  rationale?: string;
  status: "current" | "superseded";
  sourceEventIds: string[];
  validAtSeq: number;
  supersededBy?: string;
}

export interface ContextTaskProjection {
  goal?: SourcedField<string>;
  acceptanceCriteria: Array<SourcedField<string>>;
  constraints: Array<SourcedField<string>>;
  plan: Array<SourcedField<string>>;
  decisions: ContextDecisionProjection[];
  completed: Array<SourcedField<string>>;
  openWork: Array<SourcedField<string>>;
  blockers: Array<SourcedField<string>>;
  repoState?: {
    branch?: SourcedField<string>;
    headSha?: SourcedField<string>;
    dirtyPaths: Array<SourcedField<string>>;
  };
  tests: Array<{
    command: SourcedField<string>;
    status: SourcedField<string>;
  }>;
  errors: Array<SourcedField<string>>;
  unknowns: Array<SourcedField<string>>;
}

export interface ContextStateProjector {
  fold(events: readonly unknown[]): ContextTaskProjection;
}

export function projectContextTaskState(input: readonly unknown[]): ContextTaskProjection {
  const snapshot = createContextHistorySnapshot(input);
  const state: ContextTaskProjection = {
    acceptanceCriteria: [],
    constraints: [],
    plan: [],
    decisions: [],
    completed: [],
    openWork: [],
    blockers: [],
    tests: [],
    errors: [],
    unknowns: [],
  };
  const decisionIndexByEventId = new Map<string, number>();

  for (const event of snapshot.events) {
    const payload = objectRecord(event.payload);
    if (event.labels.includes("task_contract")) {
      const goal = stringValue(payload.goal);
      if (goal) state.goal = sourced(goal, event);
      state.acceptanceCriteria.push(
        ...stringList(payload.acceptanceCriteria).map((item) => sourced(item, event)),
      );
      state.constraints.push(
        ...stringList(payload.constraints).map((item) => sourced(item, event)),
      );
      state.plan.push(...stringList(payload.plan).map((item) => sourced(item, event)));
    }

    for (const supersededId of event.supersedes) {
      const index = decisionIndexByEventId.get(supersededId);
      const decision = index === undefined ? undefined : state.decisions[index];
      if (decision) {
        decision.status = "superseded";
        decision.supersededBy = event.id;
      }
    }

    if (event.kind === "decision") {
      const statement = stringValue(payload.statement) ?? stringValue(event.payload);
      if (statement) {
        const decision: ContextDecisionProjection = {
          statement,
          rationale: stringValue(payload.rationale),
          status: payload.status === "superseded" ? "superseded" : "current",
          sourceEventIds: [event.id],
          validAtSeq: event.seq,
        };
        decisionIndexByEventId.set(event.id, state.decisions.length);
        state.decisions.push(decision);
      }
    }

    if (event.kind === "checkpoint") {
      state.completed.push(...stringList(payload.completed).map((item) => sourced(item, event)));
      state.openWork.push(...stringList(payload.openWork).map((item) => sourced(item, event)));
      state.blockers.push(...stringList(payload.blockers).map((item) => sourced(item, event)));
      state.unknowns.push(...stringList(payload.unknowns).map((item) => sourced(item, event)));
    }

    if (event.kind === "test_result") {
      const command = stringValue(payload.command);
      const status = stringValue(payload.status);
      if (command && status) {
        state.tests.push({ command: sourced(command, event), status: sourced(status, event) });
      }
      const signature = event.metadata.errorSignature ?? stringValue(payload.errorSignature);
      if (signature) state.errors.push(sourced(signature, event));
    }
  }
  return state;
}

export const EvidenceStrategySchema = z.enum(["retrieval", "map_reduce", "rlm_d0", "rlm_d1"]);
export const EvidenceSourceSchema = z
  .object({
    sourceId: z.string().min(1),
    eventId: z.string().regex(EVENT_ID_PATTERN),
    contentHash: z.string().regex(SHA256_PATTERN),
    charRange: z.tuple([z.number().int().nonnegative(), z.number().int().nonnegative()]),
    excerpt: z.string(),
    status: z.enum(["current", "superseded", "conflicting"]),
  })
  .strict();
export const EvidenceClaimSchema = z
  .object({
    text: z.string().min(1),
    relation: z.enum(["direct", "inference", "aggregation"]),
    supportSourceIds: z.array(z.string().min(1)).min(1),
  })
  .strict();
export const EvidencePackSchema = z
  .object({
    requestId: z.string().min(1),
    snapshotId: z.string().min(1),
    strategy: EvidenceStrategySchema,
    sources: z.array(EvidenceSourceSchema),
    claims: z.array(EvidenceClaimSchema),
    conflicts: z.array(z.string()),
    unresolved: z.array(z.string()),
    coverage: z
      .object({
        mode: z.enum(["top_k", "sampled", "exhaustive"]),
        eventsExamined: z.number().int().nonnegative(),
        partitionsExamined: z.number().int().nonnegative(),
        omittedReasons: z.array(z.string()),
      })
      .strict(),
    usage: z
      .object({
        inputTokens: z.number().int().nonnegative(),
        outputTokens: z.number().int().nonnegative(),
        subcalls: z.number().int().nonnegative(),
        latencyMs: z.number().nonnegative(),
      })
      .strict(),
  })
  .strict();

export type EvidenceStrategy = z.infer<typeof EvidenceStrategySchema>;
export type EvidenceSource = z.infer<typeof EvidenceSourceSchema>;
export type EvidenceClaim = z.infer<typeof EvidenceClaimSchema>;
export type EvidencePack = z.infer<typeof EvidencePackSchema>;

export interface EvidenceRequest {
  requestId: string;
  snapshotId: string;
  query: string;
  maxSources?: number;
}

export interface EvidenceProvider {
  resolve(request: EvidenceRequest, signal?: AbortSignal): Promise<EvidencePack>;
}

export class Bm25EvidenceProvider implements EvidenceProvider {
  readonly snapshot: ContextHistorySnapshot;
  private readonly documents: Bm25Document[];
  private readonly documentFrequency = new Map<string, number>();
  private readonly averageLength: number;

  constructor(snapshot: ContextHistorySnapshot) {
    this.snapshot = createContextHistorySnapshot(snapshot.events, { validateAtomicTools: false });
    if (this.snapshot.snapshotId !== snapshot.snapshotId) {
      throw new Error("BM25 snapshot id does not match its event content.");
    }
    this.documents = this.snapshot.events.map((event) => {
      const text = contextEventText(event);
      const tokens = tokenize(text);
      const frequencies = frequencyMap(tokens);
      for (const token of frequencies.keys()) {
        this.documentFrequency.set(token, (this.documentFrequency.get(token) ?? 0) + 1);
      }
      return { event, text, tokens, frequencies };
    });
    this.averageLength =
      this.documents.length === 0
        ? 0
        : this.documents.reduce((sum, document) => sum + document.tokens.length, 0) /
          this.documents.length;
  }

  async resolve(request: EvidenceRequest, signal?: AbortSignal): Promise<EvidencePack> {
    if (request.snapshotId !== this.snapshot.snapshotId) {
      throw new Error("Evidence request snapshot does not match the provider snapshot.");
    }
    if (signal?.aborted) throw signal.reason ?? new Error("Evidence request aborted.");
    const queryTokens = [...new Set(tokenize(request.query))];
    const maxSources = Math.max(0, request.maxSources ?? 8);
    const superseded = new Set(this.snapshot.events.flatMap((event) => event.supersedes));
    const sources = this.documents
      .map((document) => ({ document, score: this.score(document, queryTokens) }))
      .filter((item) => item.score > 0)
      .sort(
        (left, right) =>
          right.score - left.score || left.document.event.seq - right.document.event.seq,
      )
      .slice(0, maxSources)
      .map(({ document }, index): EvidenceSource => {
        const excerpt = document.text.slice(0, 2_000);
        return {
          sourceId: `source_${index + 1}_${document.event.id}`,
          eventId: document.event.id,
          contentHash: document.event.contentHash,
          charRange: [0, excerpt.length],
          excerpt,
          status: document.event.labels.includes("conflicting")
            ? "conflicting"
            : superseded.has(document.event.id)
              ? "superseded"
              : "current",
        };
      });
    return EvidencePackSchema.parse({
      requestId: request.requestId,
      snapshotId: request.snapshotId,
      strategy: "retrieval",
      sources,
      claims: [],
      conflicts: [],
      unresolved: sources.length === 0 ? ["No matching historical evidence."] : [],
      coverage: {
        mode: "top_k",
        eventsExamined: this.documents.length,
        partitionsExamined: this.documents.length > 0 ? 1 : 0,
        omittedReasons: this.documents.length > sources.length ? ["below_top_k_or_zero_score"] : [],
      },
      usage: { inputTokens: 0, outputTokens: 0, subcalls: 0, latencyMs: 0 },
    });
  }

  private score(document: Bm25Document, queryTokens: string[]): number {
    if (this.documents.length === 0 || this.averageLength === 0) return 0;
    const k1 = 1.2;
    const b = 0.75;
    return queryTokens.reduce((score, token) => {
      const frequency = document.frequencies.get(token) ?? 0;
      if (frequency === 0) return score;
      const containing = this.documentFrequency.get(token) ?? 0;
      const idf = Math.log(1 + (this.documents.length - containing + 0.5) / (containing + 0.5));
      const denominator =
        frequency + k1 * (1 - b + b * (document.tokens.length / this.averageLength));
      return score + idf * ((frequency * (k1 + 1)) / denominator);
    }, 0);
  }
}

export interface VerifiedEvidencePack {
  pack: EvidencePack;
  rejectedSourceIds: string[];
  rejectedClaimIndexes: number[];
  issues: string[];
}

export interface ContextVerifier {
  verify(pack: unknown, snapshot: ContextHistorySnapshot): VerifiedEvidencePack;
}

export function verifyEvidencePack(
  input: unknown,
  snapshot: ContextHistorySnapshot,
): VerifiedEvidencePack {
  const pack = EvidencePackSchema.parse(input);
  if (pack.snapshotId !== snapshot.snapshotId) {
    throw new Error("Evidence pack snapshot does not match the verified history snapshot.");
  }
  const events = new Map(snapshot.events.map((event) => [event.id, event]));
  const issues: string[] = [];
  const rejectedSourceIds: string[] = [];
  const sources = pack.sources.filter((source) => {
    const event = events.get(source.eventId);
    let issue: string | undefined;
    if (!event) issue = `${source.sourceId} references a missing event.`;
    else if (event.contentHash !== source.contentHash) issue = `${source.sourceId} hash mismatch.`;
    else {
      const text = contextEventText(event);
      const [start, end] = source.charRange;
      if (end < start || end > text.length) issue = `${source.sourceId} has an invalid range.`;
      else if (text.slice(start, end) !== source.excerpt) {
        issue = `${source.sourceId} excerpt does not match its source range.`;
      }
    }
    if (!issue) return true;
    issues.push(issue);
    rejectedSourceIds.push(source.sourceId);
    return false;
  });
  const validSourceIds = new Set(sources.map((source) => source.sourceId));
  const rejectedClaimIndexes: number[] = [];
  const claims = pack.claims.filter((claim, index) => {
    const supported =
      claim.supportSourceIds.length > 0 &&
      claim.supportSourceIds.every((sourceId) => validSourceIds.has(sourceId));
    if (supported) return true;
    rejectedClaimIndexes.push(index);
    issues.push(`${claim.text} has invalid or missing support sources.`);
    return false;
  });
  return {
    pack: EvidencePackSchema.parse({ ...pack, sources, claims }),
    rejectedSourceIds,
    rejectedClaimIndexes,
    issues,
  };
}

export const ContextSlotSchema = z.enum([
  "system",
  "taskContract",
  "state",
  "recent",
  "evidence",
  "summary",
  "capsules",
]);
export type ContextSlot = z.infer<typeof ContextSlotSchema>;

export const ContextItemSchema = z
  .object({
    itemId: z.string().min(1),
    slot: ContextSlotSchema,
    content: z.string(),
    tokenCount: z.number().int().nonnegative(),
    priority: z.number().finite(),
    mandatory: z.boolean(),
    trust: z.enum(["trusted", "untrusted"]),
    sourceEventIds: z.array(z.string().regex(EVENT_ID_PATTERN)),
  })
  .strict();
export type ContextItem = z.infer<typeof ContextItemSchema>;

const ContextComponentConfigSchema = z
  .object({
    eventStore: z.enum(["sqlite_wal", "jsonl_replay"]),
    artifactStore: z.literal("local_cas"),
    normalizer: z.literal("deterministic_atomic"),
    masker: z.literal("deterministic_capsule"),
    stateProjector: z.literal("sourced_state_v1"),
    evidenceProvider: z.literal("bm25"),
    assembler: z.literal("priority_quota"),
    verifier: z.literal("deterministic"),
  })
  .strict();

const ContextSlotBudgetsSchema = z
  .object({
    system: z.number().int().nonnegative(),
    taskContract: z.number().int().nonnegative(),
    state: z.number().int().nonnegative(),
    recent: z.number().int().nonnegative(),
    evidence: z.number().int().nonnegative(),
    summary: z.number().int().nonnegative(),
    capsules: z.number().int().nonnegative(),
  })
  .strict();

export const ContextEngineConfigSchema = z
  .object({
    components: ContextComponentConfigSchema,
    assembler: z
      .object({
        inputTokenBudget: z.number().int().positive(),
        reservedOutputTokens: z.number().int().nonnegative(),
        slotTokenBudgets: ContextSlotBudgetsSchema,
      })
      .strict(),
  })
  .strict();
export type ContextEngineConfig = z.infer<typeof ContextEngineConfigSchema>;

export function parseContextEngineConfig(input: unknown): ContextEngineConfig {
  return ContextEngineConfigSchema.parse(input);
}

export function contextEngineConfigHash(input: unknown): string {
  return `sha256:${sha256Hex(stableJson(ContextEngineConfigSchema.parse(input)))}`;
}

export class ContextOverflow extends Error {
  readonly slot?: ContextSlot;
  readonly requiredTokens: number;
  readonly availableTokens: number;

  constructor(
    message: string,
    requiredTokens: number,
    availableTokens: number,
    slot?: ContextSlot,
  ) {
    super(message);
    this.name = "ContextOverflow";
    this.slot = slot;
    this.requiredTokens = requiredTokens;
    this.availableTokens = availableTokens;
  }
}

export interface CompileContextInput {
  requestId: string;
  snapshot: ContextHistorySnapshot;
  config: ContextEngineConfig;
  modelVersion: string;
  requestedMode: EvidenceStrategy;
  effectiveMode: EvidenceStrategy;
  fallbackChain?: string[];
  items: readonly unknown[];
}

export interface ContextManifest {
  requestId: string;
  snapshotId: string;
  configHash: string;
  modelVersion: string;
  requestedMode: EvidenceStrategy;
  effectiveMode: EvidenceStrategy;
  fallbackChain: string[];
  includedItemIds: string[];
  includedEventIds: string[];
  omittedItems: Array<{ itemId: string; reason: "slot_budget" | "input_budget" }>;
  inputTokens: number;
  reservedOutputTokens: number;
  slotTokens: Record<ContextSlot, number>;
  contextHash: string;
  latencyMs: number;
}

export interface CompiledContext {
  context: string;
  items: ContextItem[];
  manifest: ContextManifest;
}

export interface AgentContextEngineCompileInput {
  requestId: string;
  agentName: string;
  modelVersion: string;
  instructions: string;
  arguments: Readonly<Record<string, unknown>>;
  runtimeContext: Readonly<Record<string, unknown>>;
}

/** Request-scoped adapter from canonical host history into the bounded compiler. */
export interface AgentContextEngine {
  compile(input: AgentContextEngineCompileInput): CompiledContext | Promise<CompiledContext>;
  onCompiled?(manifest: ContextManifest): void | Promise<void>;
}

export interface SessionContextEngineOptions {
  sessionId: string;
  history?: readonly MessageChunk[];
  config?: ContextEngineConfig;
  preserveRecentAtomicUnits?: number;
  onCompiled?(manifest: ContextManifest): void | Promise<void>;
}

/**
 * Adapts one immutable canonical Session view into the request-scoped compiler.
 * The current user turn remains a native Provider input; only prior history and
 * trusted system observations are rendered into the compiled instruction block.
 */
export function createSessionContextEngine(
  options: SessionContextEngineOptions,
): AgentContextEngine {
  const sessionId = z.string().min(1).parse(options.sessionId);
  const history = (options.history ?? []).map((message) => MessageChunkSchema.parse(message));
  const config = options.config
    ? ContextEngineConfigSchema.parse(options.config)
    : defaultSessionContextEngineConfig();

  return {
    compile: (input) => {
      const argumentMessages = chatMessagesFromArguments(input.arguments);
      const sourceMessages = history.length > 0 ? history : argumentMessages;
      const snapshot = createContextHistorySnapshot(
        sessionContextEvents(sessionId, sourceMessages),
      );
      const currentUserContent = latestRoleContent(argumentMessages, "user");
      const currentUserEventId = [...snapshot.events]
        .reverse()
        .find(
          (event) =>
            event.kind === "user_message" && contextEventText(event) === currentUserContent,
        )?.id;
      const currentTurnId = currentUserEventId
        ? snapshot.events.find((event) => event.id === currentUserEventId)?.turnId
        : undefined;
      const masked = maskContextUnits(normalizeContextEvents(snapshot.events), {
        currentTurnId,
        preserveRecentAtomicUnits: options.preserveRecentAtomicUnits ?? 8,
      });
      const systemContents = uniqueStrings([
        ...history
          .filter((message) => message.kind === "message" && message.role === "system")
          .map((message) => message.content),
        ...argumentMessages
          .filter((message) => message.kind === "message" && message.role === "system")
          .map((message) => message.content),
      ]).filter((content) => content.trim().length > 0);
      const systemItems: ContextItem[] = systemContents.map((content, index) => ({
        itemId: `session_system_${index + 1}`,
        slot: "system",
        content,
        tokenCount: estimateContextTokens(content),
        priority: 1_000 - index,
        mandatory: true,
        trust: "trusted",
        sourceEventIds: [],
      }));
      const historyItems: ContextItem[] = masked.flatMap((item, index) => {
        if (
          item.visibility === "omit" ||
          !item.rendered ||
          item.unit.eventIds.includes(currentUserEventId ?? "")
        ) {
          return [];
        }
        return [
          {
            itemId: `session_history_${index + 1}`,
            slot: item.visibility === "full" ? "recent" : "capsules",
            content: item.rendered,
            tokenCount: estimateContextTokens(item.rendered),
            priority: item.unit.endSeq,
            mandatory: false,
            trust: "untrusted",
            sourceEventIds: item.unit.eventIds,
          } satisfies ContextItem,
        ];
      });
      return compileContext({
        requestId: input.requestId,
        snapshot,
        config,
        modelVersion: input.modelVersion,
        requestedMode: "retrieval",
        effectiveMode: "retrieval",
        items: [...systemItems, ...historyItems],
      });
    },
    ...(options.onCompiled ? { onCompiled: options.onCompiled } : {}),
  };
}

export interface ContextAssembler {
  compile(input: CompileContextInput): CompiledContext;
}

const SLOT_ORDER: ContextSlot[] = [
  "system",
  "taskContract",
  "state",
  "recent",
  "evidence",
  "summary",
  "capsules",
];

export function compileContext(input: CompileContextInput): CompiledContext {
  const config = ContextEngineConfigSchema.parse(input.config);
  const snapshot = createContextHistorySnapshot(input.snapshot.events, {
    validateAtomicTools: false,
  });
  if (snapshot.snapshotId !== input.snapshot.snapshotId) {
    throw new Error("Context compilation snapshot id does not match its events.");
  }
  const eventIds = new Set(snapshot.events.map((event) => event.id));
  const items = input.items.map((item) => ContextItemSchema.parse(item));
  for (const item of items) {
    const missing = item.sourceEventIds.find((eventId) => !eventIds.has(eventId));
    if (missing) throw new Error(`Context item ${item.itemId} cites missing event ${missing}.`);
  }

  const slotTokens = emptySlotTokens();
  const included: ContextItem[] = [];
  const omittedItems: ContextManifest["omittedItems"] = [];
  const mandatory = items.filter((item) => item.mandatory);
  for (const slot of SLOT_ORDER) {
    const required = mandatory
      .filter((item) => item.slot === slot)
      .reduce((total, item) => total + item.tokenCount, 0);
    const available = config.assembler.slotTokenBudgets[slot];
    if (required > available) {
      throw new ContextOverflow(
        `Mandatory ${slot} context requires ${required} tokens; slot budget is ${available}.`,
        required,
        available,
        slot,
      );
    }
  }
  const mandatoryTokens = mandatory.reduce((total, item) => total + item.tokenCount, 0);
  if (mandatoryTokens > config.assembler.inputTokenBudget) {
    throw new ContextOverflow(
      `Mandatory context requires ${mandatoryTokens} tokens; input budget is ${config.assembler.inputTokenBudget}.`,
      mandatoryTokens,
      config.assembler.inputTokenBudget,
    );
  }

  const order = new Map(items.map((item, index) => [item.itemId, index]));
  let inputTokens = 0;
  for (const slot of SLOT_ORDER) {
    const candidates = items
      .filter((item) => item.slot === slot)
      .sort(
        (left, right) =>
          Number(right.mandatory) - Number(left.mandatory) ||
          right.priority - left.priority ||
          (order.get(left.itemId) ?? 0) - (order.get(right.itemId) ?? 0),
      );
    for (const item of candidates) {
      const slotAvailable = config.assembler.slotTokenBudgets[slot] - slotTokens[slot];
      const inputAvailable = config.assembler.inputTokenBudget - inputTokens;
      if (item.tokenCount <= slotAvailable && item.tokenCount <= inputAvailable) {
        included.push(item);
        slotTokens[slot] += item.tokenCount;
        inputTokens += item.tokenCount;
      } else if (!item.mandatory) {
        omittedItems.push({
          itemId: item.itemId,
          reason: item.tokenCount > slotAvailable ? "slot_budget" : "input_budget",
        });
      }
    }
  }

  const context = included.map(renderContextItem).join("\n\n---\n\n");
  const includedEventIds = uniqueStrings(included.flatMap((item) => item.sourceEventIds));
  return {
    context,
    items: included,
    manifest: {
      requestId: input.requestId,
      snapshotId: snapshot.snapshotId,
      configHash: contextEngineConfigHash(config),
      modelVersion: input.modelVersion,
      requestedMode: EvidenceStrategySchema.parse(input.requestedMode),
      effectiveMode: EvidenceStrategySchema.parse(input.effectiveMode),
      fallbackChain: [...(input.fallbackChain ?? [])],
      includedItemIds: included.map((item) => item.itemId),
      includedEventIds,
      omittedItems,
      inputTokens,
      reservedOutputTokens: config.assembler.reservedOutputTokens,
      slotTokens,
      contextHash: `sha256:${sha256Hex(context)}`,
      latencyMs: 0,
    },
  };
}

export type ContextComponentKind =
  | "normalizer"
  | "masker"
  | "stateProjector"
  | "evidenceProvider"
  | "assembler"
  | "verifier";

/** Small explicit registry; callers depend on stable interfaces, not implementation fields. */
export class ContextEngineRegistry {
  private readonly components = new Map<ContextComponentKind, Map<string, unknown>>();

  register<T>(kind: ContextComponentKind, id: string, component: T): this {
    if (!id) throw new Error("Context component id must not be empty.");
    const byId = this.components.get(kind) ?? new Map<string, unknown>();
    if (byId.has(id)) throw new Error(`Context component already registered: ${kind}:${id}`);
    byId.set(id, component);
    this.components.set(kind, byId);
    return this;
  }

  resolve<T>(kind: ContextComponentKind, id: string): T {
    const component = this.components.get(kind)?.get(id);
    if (component === undefined) throw new Error(`Unknown context component: ${kind}:${id}`);
    return component as T;
  }
}

export function contextEventText(event: ContextEngineEvent): string {
  if (typeof event.payload === "string") return event.payload;
  if (event.payload === undefined) return event.kind;
  return stableJson(event.payload);
}

function contextContentHash(content: unknown): string {
  return `sha256:${sha256Hex(stableJson(content))}`;
}

function sha256Hex(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function validateEventOrder(events: ContextEngineEvent[]): void {
  const ids = new Set<string>();
  let previousSeq = -1;
  for (const event of events) {
    if (ids.has(event.id)) throw new Error(`Duplicate context event id: ${event.id}`);
    if (event.seq <= previousSeq) {
      throw new Error(`Context event sequence must increase: ${event.seq} after ${previousSeq}.`);
    }
    for (const parent of event.causalParents) {
      if (!ids.has(parent))
        throw new Error(`Context event ${event.id} has missing causal parent ${parent}.`);
    }
    ids.add(event.id);
    previousSeq = event.seq;
  }
}

function validateToolReferences(events: ContextEngineEvent[]): void {
  const calls = new Map<string, ContextEngineEvent>();
  const results = new Set<string>();
  for (const event of events) {
    if (event.kind === "tool_call") {
      const callId = event.toolCallId as string;
      if (calls.has(callId)) throw new Error(`Duplicate tool call id: ${callId}`);
      calls.set(callId, event);
    } else if (event.kind === "tool_result") {
      const callId = event.toolCallId as string;
      if (!calls.has(callId)) throw new Error(`Orphan tool result: ${event.id} (${callId}).`);
      if (results.has(callId)) throw new Error(`Duplicate tool result id: ${callId}`);
      results.add(callId);
    }
  }
}

function statusFromEvent(event: ContextEngineEvent): ContextAtomicUnitStatus {
  if (event.metadata.exitCode !== undefined) {
    return event.metadata.exitCode === 0 ? "succeeded" : "failed";
  }
  const payload = objectRecord(event.payload);
  const status = stringValue(payload.status);
  if (status === "failed" || status === "error") return "failed";
  if (status === "succeeded" || status === "passed" || status === "complete") return "succeeded";
  return event.kind === "tool_call" ? "pending" : "succeeded";
}

function renderMaskedUnit(
  unit: ContextAtomicUnit,
  visibility: ContextVisibility,
  capsuleMaxChars: number,
): string {
  if (visibility === "omit") return "";
  if (visibility === "ref") {
    const refs = unit.events.flatMap((event) => (event.artifactRef ? [event.artifactRef.uri] : []));
    return refs.join("\n");
  }
  const full = unit.events
    .map((event) => `[${event.id} ${event.kind}]\n${contextEventText(event)}`)
    .join("\n\n");
  if (visibility === "full" || full.length <= capsuleMaxChars) return full;
  return `${full.slice(0, capsuleMaxChars)}…`;
}

function sourced<T>(value: T, event: ContextEngineEvent): SourcedField<T> {
  return { value, basis: "observed", sourceEventIds: [event.id], validAtSeq: event.seq };
}

function objectRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.flatMap((item) => stringValue(item) ?? []) : [];
}

interface Bm25Document {
  event: ContextEngineEvent;
  text: string;
  tokens: string[];
  frequencies: Map<string, number>;
}

function tokenize(value: string): string[] {
  return value.toLocaleLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
}

function frequencyMap(tokens: string[]): Map<string, number> {
  const frequencies = new Map<string, number>();
  for (const token of tokens) frequencies.set(token, (frequencies.get(token) ?? 0) + 1);
  return frequencies;
}

function renderContextItem(item: ContextItem): string {
  const title = item.slot.replace(/([a-z])([A-Z])/gu, "$1 $2");
  const content =
    item.trust === "untrusted"
      ? `<untrusted_history>\n${item.content}\n</untrusted_history>`
      : item.content;
  return `## ${title}\n\n${content}`;
}

function emptySlotTokens(): Record<ContextSlot, number> {
  return {
    system: 0,
    taskContract: 0,
    state: 0,
    recent: 0,
    evidence: 0,
    summary: 0,
    capsules: 0,
  };
}

function uniqueStrings(values: string[]): string[] {
  return [...new Set(values)];
}

function defaultSessionContextEngineConfig(): ContextEngineConfig {
  return ContextEngineConfigSchema.parse({
    components: {
      eventStore: "sqlite_wal",
      artifactStore: "local_cas",
      normalizer: "deterministic_atomic",
      masker: "deterministic_capsule",
      stateProjector: "sourced_state_v1",
      evidenceProvider: "bm25",
      assembler: "priority_quota",
      verifier: "deterministic",
    },
    assembler: {
      inputTokenBudget: 65_536,
      reservedOutputTokens: 8_192,
      slotTokenBudgets: {
        system: 24_576,
        taskContract: 4_096,
        state: 8_192,
        recent: 16_384,
        evidence: 8_192,
        summary: 2_048,
        capsules: 8_192,
      },
    },
  });
}

function chatMessagesFromArguments(arguments_: Readonly<Record<string, unknown>>): MessageChunk[] {
  if (!Array.isArray(arguments_.messages)) return [];
  return arguments_.messages.flatMap((candidate) => {
    const record = objectRecord(candidate);
    const role = typeof record.role === "string" ? record.role : undefined;
    const content = typeof record.content === "string" ? record.content : undefined;
    if (!role || content === undefined) return [];
    return [
      MessageChunkSchema.parse({
        role,
        content,
        kind: "message",
        ...(Array.isArray(record.attachments) ? { attachments: record.attachments } : {}),
      }),
    ];
  });
}

function latestRoleContent(messages: readonly MessageChunk[], role: string): string | undefined {
  return [...messages].reverse().find((message) => message.role === role)?.content;
}

function sessionContextEvents(
  sessionId: string,
  messages: readonly MessageChunk[],
): ContextEngineEvent[] {
  const events: ContextEngineEvent[] = [];
  const pendingByTool = new Map<string, string[]>();
  let turn = 0;

  for (const [index, message] of messages.entries()) {
    const seq = index + 1;
    if (message.kind === "thinking" || message.kind === "tool_progress") continue;
    if (message.kind === "message" && message.role === "system") continue;
    if (message.kind === "message" && message.role === "user") turn += 1;
    const id = `evt_message_${seq}`;
    const common = {
      id,
      seq,
      sessionId,
      turnId: `turn_${Math.max(turn, 1)}`,
      timestamp: stableMessageTimestamp(message.createdAt, seq),
      causalParents: events.length > 0 ? [events.at(-1)?.id as string] : [],
      labels: [],
      metadata: {},
    };

    if (message.kind === "message") {
      const kind =
        message.role === "user"
          ? "user_message"
          : message.role === "assistant"
            ? "assistant_message"
            : undefined;
      if (!kind) continue;
      events.push(createContextEvent({ ...common, kind, payload: message.content }));
      continue;
    }

    const toolName = message.toolName?.trim() || "unknown_tool";
    if (message.kind === "tool_call") {
      const toolCallId = message.render?.invocationId ?? `session_call_${seq}`;
      const pending = pendingByTool.get(toolName) ?? [];
      pending.push(toolCallId);
      pendingByTool.set(toolName, pending);
      events.push(
        createContextEvent({
          ...common,
          kind: "tool_call",
          toolCallId,
          payload: { toolName, text: message.content },
        }),
      );
      continue;
    }

    if (message.kind === "tool_result") {
      const pending = pendingByTool.get(toolName) ?? [];
      const toolCallId = message.render?.invocationId ?? pending[0] ?? `orphan_${seq}`;
      const pendingIndex = pending.indexOf(toolCallId);
      if (pendingIndex >= 0) pending.splice(pendingIndex, 1);
      const status = message.render?.status;
      const exitCode =
        status === "succeeded" || status === "completed"
          ? 0
          : status === "failed" || status === "canceled"
            ? 1
            : undefined;
      events.push(
        createContextEvent({
          ...common,
          kind: "tool_result",
          toolCallId,
          payload: { toolName, text: message.content, ...(status ? { status } : {}) },
          metadata: exitCode === undefined ? {} : { exitCode },
        }),
      );
    }
  }

  return events;
}

function stableMessageTimestamp(createdAt: string | undefined, seq: number): string {
  const parsed = createdAt ? Date.parse(createdAt) : Number.NaN;
  if (Number.isFinite(parsed)) return new Date(parsed).toISOString();
  return new Date(Date.UTC(2000, 0, 1) + seq * 1_000).toISOString();
}

function estimateContextTokens(content: string): number {
  const compact = content.trim();
  if (!compact) return 0;
  const cjkCount = (
    compact.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu) ?? []
  ).length;
  return cjkCount + Math.ceil(([...compact].length - cjkCount) / 4);
}

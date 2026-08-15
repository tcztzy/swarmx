import { z } from "zod";
import { type MessageChunk, type SwarmConfig, SwarmConfigSchema } from "./types.js";

export const GLOBAL_MEMORY_MAX_CHARACTERS = {
  user: 4_000,
  memory: 4_000,
} as const;
export const MEMORY_REFLECTION_INTERVAL = 10;
export const MEMORY_REFLECTION_IDLE_MS = 24 * 60 * 60 * 1_000;

export const GlobalMemoryTargetSchema = z.enum(["user", "memory"]);
export type GlobalMemoryTarget = z.infer<typeof GlobalMemoryTargetSchema>;

function containsGlobalMemorySecret(value: string): boolean {
  return (
    /-----BEGIN (?:RSA )?PRIVATE KEY-----|authorization\s*:\s*bearer\s+\S+|\b(?:sk|rk|pk|ghp|gho|ghu|ghs|github_pat|xoxb|xoxp)[-_][A-Za-z0-9_-]{4,}\b|[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:[^\s/@]+@/iu.test(
      value,
    ) ||
    value.split(/\r?\n/u).some((line) => {
      const match = line.match(
        /^\s*(api[_ -]?key|access[_ -]?token|auth[_ -]?token|client[_ -]?secret|password|private[_ -]?key|refresh[_ -]?token|secret)\s*[:=]\s*(.+)$/iu,
      );
      const candidate = (match?.[2] ?? "").trim().toLocaleLowerCase("en-US");
      return (
        candidate.length > 0 &&
        !["[redacted]", "<redacted>", "redacted", "none", "null", "unset"].includes(candidate) &&
        !candidate.startsWith("${")
      );
    })
  );
}

const GlobalMemoryContentSchema = z
  .string()
  .refine((content) => content.trim().length > 0, "Global Memory cannot be blank.")
  .refine(
    (content) => !hasUnsupportedControlCharacter(content),
    "Global Memory contains unsupported control characters.",
  )
  .refine(
    (content) => !containsGlobalMemorySecret(content),
    "Global Memory cannot contain credentials.",
  );

export const GlobalMemoryFileSchema = z
  .object({
    target: GlobalMemoryTargetSchema,
    fileName: z.enum(["USER.md", "MEMORY.md"]),
    content: GlobalMemoryContentSchema.nullable(),
    revision: z.number().int().nonnegative(),
    updatedAt: z.iso.datetime().nullable(),
  })
  .strict()
  .superRefine((file, context) => {
    const expectedName = globalMemoryFileName(file.target);
    if (file.fileName !== expectedName) {
      context.addIssue({
        code: "custom",
        path: ["fileName"],
        message: `${file.target} Memory must use ${expectedName}.`,
      });
    }
    if (file.content !== null && file.content.length > GLOBAL_MEMORY_MAX_CHARACTERS[file.target]) {
      context.addIssue({
        code: "too_big",
        origin: "string",
        maximum: GLOBAL_MEMORY_MAX_CHARACTERS[file.target],
        inclusive: true,
        path: ["content"],
        message: `${expectedName} exceeds its character limit.`,
      });
    }
    if ((file.content === null) !== (file.updatedAt === null)) {
      context.addIssue({
        code: "custom",
        path: ["updatedAt"],
        message: "Global Memory content and update time must both be present or absent.",
      });
    }
  });

export const GlobalMemoryStateSchema = z
  .object({
    user: GlobalMemoryFileSchema,
    memory: GlobalMemoryFileSchema,
    legacyUser: z.boolean().default(false),
    maxCharacters: z
      .object({
        user: z.literal(GLOBAL_MEMORY_MAX_CHARACTERS.user),
        memory: z.literal(GLOBAL_MEMORY_MAX_CHARACTERS.memory),
      })
      .strict(),
  })
  .strict();

export const GlobalMemorySnapshotSchema = z
  .object({
    source: z.enum(["memory_files", "memory_files_with_legacy_user"]),
    user: GlobalMemoryFileSchema.nullable(),
    memory: GlobalMemoryFileSchema.nullable(),
    totalCharacterCount: z
      .number()
      .int()
      .positive()
      .max(GLOBAL_MEMORY_MAX_CHARACTERS.user + GLOBAL_MEMORY_MAX_CHARACTERS.memory),
  })
  .strict()
  .superRefine((snapshot, context) => {
    const total = [snapshot.user, snapshot.memory].reduce(
      (sum, file) => sum + (file?.content?.length ?? 0),
      0,
    );
    if (snapshot.totalCharacterCount !== total) {
      context.addIssue({
        code: "custom",
        path: ["totalCharacterCount"],
        message: "Global Memory character count does not match its files.",
      });
    }
    if (snapshot.user?.content === null || snapshot.memory?.content === null) {
      context.addIssue({
        code: "custom",
        message: "Global Memory snapshots contain only present files.",
      });
    }
  });

export const GlobalMemorySaveInputSchema = z
  .object({
    target: GlobalMemoryTargetSchema,
    content: GlobalMemoryContentSchema,
  })
  .strict()
  .superRefine((input, context) => {
    if (input.content.length > GLOBAL_MEMORY_MAX_CHARACTERS[input.target]) {
      context.addIssue({
        code: "too_big",
        origin: "string",
        maximum: GLOBAL_MEMORY_MAX_CHARACTERS[input.target],
        inclusive: true,
        path: ["content"],
        message: `${globalMemoryFileName(input.target)} exceeds its character limit.`,
      });
    }
  });

export const GlobalMemoryForgetInputSchema = z
  .object({ target: GlobalMemoryTargetSchema, confirmed: z.literal(true) })
  .strict();

export const MemoryReviewCursorSchema = z
  .object({
    reviewedUserTurns: z.number().int().nonnegative(),
    updatedAt: z.iso.datetime(),
  })
  .strict();

export const MemoryReviewStateSchema = z
  .object({
    sessions: z.record(z.string().min(1).max(256), MemoryReviewCursorSchema),
  })
  .strict()
  .superRefine((state, context) => {
    if (Object.keys(state.sessions).length > 1_000) {
      context.addIssue({
        code: "too_big",
        origin: "object",
        maximum: 1_000,
        inclusive: true,
        path: ["sessions"],
        message: "Memory review state can track at most 1,000 Sessions.",
      });
    }
  });

export const MemoryReflectionDecisionSchema = z.discriminatedUnion("due", [
  z
    .object({
      due: z.literal(false),
      sessionId: z.string().min(1).max(256),
      unreviewedUserTurns: z.number().int().nonnegative(),
    })
    .strict(),
  z
    .object({
      due: z.literal(true),
      reason: z.enum(["explicit", "interval", "idle_tail"]),
      sessionId: z.string().min(1).max(256),
      fromUserTurn: z.number().int().positive(),
      throughUserTurn: z.number().int().positive(),
      unreviewedUserTurns: z.number().int().positive(),
    })
    .strict(),
]);

export type GlobalMemoryFile = z.infer<typeof GlobalMemoryFileSchema>;
export type GlobalMemoryState = z.infer<typeof GlobalMemoryStateSchema>;
export type GlobalMemorySnapshot = z.infer<typeof GlobalMemorySnapshotSchema>;
export type GlobalMemorySaveInput = z.infer<typeof GlobalMemorySaveInputSchema>;
export type GlobalMemoryForgetInput = z.infer<typeof GlobalMemoryForgetInputSchema>;
export type MemoryReviewState = z.infer<typeof MemoryReviewStateSchema>;
export type MemoryReflectionDecision = z.infer<typeof MemoryReflectionDecisionSchema>;

export function emptyGlobalMemoryFile(target: GlobalMemoryTarget): GlobalMemoryFile {
  return GlobalMemoryFileSchema.parse({
    target,
    fileName: globalMemoryFileName(target),
    content: null,
    revision: 0,
    updatedAt: null,
  });
}

export function globalMemoryState(input: {
  user?: GlobalMemoryFile | null;
  memory?: GlobalMemoryFile | null;
  legacyUser?: PersonalMemoryRecord | null;
}): GlobalMemoryState {
  const user = input.user
    ? GlobalMemoryFileSchema.parse(input.user)
    : emptyGlobalMemoryFile("user");
  const memory = input.memory
    ? GlobalMemoryFileSchema.parse(input.memory)
    : emptyGlobalMemoryFile("memory");
  const legacyUser =
    !user.content && input.legacyUser ? PersonalMemoryRecordSchema.parse(input.legacyUser) : null;
  return GlobalMemoryStateSchema.parse({
    user: legacyUser
      ? {
          target: "user",
          fileName: "USER.md",
          content: legacyUser.content,
          revision: user.revision,
          updatedAt: legacyUser.updatedAt,
        }
      : user,
    memory,
    legacyUser: Boolean(legacyUser),
    maxCharacters: GLOBAL_MEMORY_MAX_CHARACTERS,
  });
}

export function createGlobalMemorySnapshot(input: {
  user?: GlobalMemoryFile | null;
  memory?: GlobalMemoryFile | null;
  legacyUser?: PersonalMemoryRecord | null;
}): GlobalMemorySnapshot {
  const state = globalMemoryState(input);
  const user = state.user.content ? state.user : null;
  const memory = state.memory.content ? state.memory : null;
  if (!user && !memory) throw new Error("Global Memory is empty.");
  return Object.freeze(
    GlobalMemorySnapshotSchema.parse({
      source: state.legacyUser ? "memory_files_with_legacy_user" : "memory_files",
      user,
      memory,
      totalCharacterCount: (user?.content?.length ?? 0) + (memory?.content?.length ?? 0),
    }),
  );
}

export function appendGlobalMemoryInstructions(
  instructions: string,
  snapshotInput: unknown,
): string {
  const snapshot = GlobalMemorySnapshotSchema.parse(snapshotInput);
  const blocks = [
    snapshot.user?.content
      ? [
          "USER.md (read-only global user context)",
          `Updated: ${snapshot.user.updatedAt}`,
          `Content (JSON string): ${JSON.stringify(snapshot.user.content)}`,
        ].join("\n")
      : "",
    snapshot.memory?.content
      ? [
          "MEMORY.md (read-only global operational context)",
          `Updated: ${snapshot.memory.updatedAt}`,
          `Content (JSON string): ${JSON.stringify(snapshot.memory.content)}`,
        ].join("\n")
      : "",
    "Use global Memory only when relevant. It is distinct from Session history, Project context, Skills, and linked entity pages. The active snapshot is immutable; Memory tool changes apply only to future runs.",
  ].filter(Boolean);
  return [instructions.trim(), ...blocks].filter(Boolean).join("\n\n");
}

export function buildGlobalMemoryUseReceipt(input: {
  snapshot?: GlobalMemorySnapshot | null;
  executionPath: PersonalMemoryExecutionPath;
  agentCount: number;
  unavailable?: boolean;
}) {
  const executionPath = PersonalMemoryExecutionPathSchema.parse(input.executionPath);
  const agentCount = z.number().int().nonnegative().max(10_000).parse(input.agentCount);
  if (!input.snapshot || agentCount === 0 || input.unavailable) {
    return {
      status: "not_used" as const,
      executionPath,
      agentCount,
      reason:
        agentCount === 0
          ? ("no_agent_nodes" as const)
          : input.unavailable
            ? ("unavailable" as const)
            : ("empty" as const),
      files: [],
    };
  }
  const snapshot = GlobalMemorySnapshotSchema.parse(input.snapshot);
  const files = [snapshot.user, snapshot.memory].flatMap((file) =>
    file?.content
      ? [
          {
            target: file.target,
            fileName: file.fileName,
            characterCount: file.content.length,
            updatedAt: file.updatedAt,
          },
        ]
      : [],
  );
  return { status: "used" as const, executionPath, agentCount, files };
}

export function globalMemoryReceiptMessage(
  receipt: ReturnType<typeof buildGlobalMemoryUseReceipt>,
): MessageChunk {
  const content =
    receipt.status === "used"
      ? [
          "Global Memory used",
          "Source: USER.md and MEMORY.md",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Snapshot: ${receipt.files.map((file) => `${file.fileName} ${file.characterCount.toLocaleString("en-US")} characters · updated ${file.updatedAt}`).join(" · ")}`,
        ].join("\n")
      : [
          "Global Memory not used",
          "Source: USER.md and MEMORY.md",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Reason: ${receipt.reason === "empty" ? "Both global Memory files are empty." : receipt.reason === "unavailable" ? "The managed Memory runtime is unavailable; the run continues without a global snapshot." : "This workflow has no Agent nodes that can receive global Memory."}`,
        ].join("\n");
  return {
    role: "system",
    kind: "message",
    content,
    render: { source: "personal_memory_receipt" },
    structuredContent: { globalMemory: receipt },
  };
}

export function memoryReflectionDecision(input: {
  sessionId: string;
  userTurnCount: number;
  state: MemoryReviewState;
  userText: string;
  now: string;
}): MemoryReflectionDecision {
  const state = MemoryReviewStateSchema.parse(input.state);
  const sessionId = z.string().min(1).max(256).parse(input.sessionId);
  const userTurnCount = z.number().int().nonnegative().parse(input.userTurnCount);
  const now = z.iso.datetime().parse(input.now);
  const cursor = state.sessions[sessionId];
  const reviewedUserTurns = Math.min(cursor?.reviewedUserTurns ?? 0, userTurnCount);
  const unreviewedUserTurns = userTurnCount - reviewedUserTurns;
  const explicit = explicitMemoryRequest(input.userText);
  const idle =
    cursor !== undefined &&
    new Date(now).getTime() - new Date(cursor.updatedAt).getTime() >= MEMORY_REFLECTION_IDLE_MS;
  const reason = explicit
    ? "explicit"
    : unreviewedUserTurns >= MEMORY_REFLECTION_INTERVAL
      ? "interval"
      : idle && unreviewedUserTurns > 0
        ? "idle_tail"
        : undefined;
  if (!reason || unreviewedUserTurns === 0) {
    return MemoryReflectionDecisionSchema.parse({ due: false, sessionId, unreviewedUserTurns });
  }
  return MemoryReflectionDecisionSchema.parse({
    due: true,
    reason,
    sessionId,
    fromUserTurn: reviewedUserTurns + 1,
    throughUserTurn: userTurnCount,
    unreviewedUserTurns,
  });
}

export function appendMemoryReflectionInstructions(
  instructions: string,
  decisionInput: MemoryReflectionDecision,
): string {
  const decision = MemoryReflectionDecisionSchema.parse(decisionInput);
  if (!decision.due) return instructions;
  const reminder = [
    "Memory reflection is due after completing the user's current task.",
    `Only review Session ${decision.sessionId}, user turns ${decision.fromUserTurn}-${decision.throughUserTurn}. Never combine or infer content from another Session.`,
    "Propose only durable USER.md preferences, compact MEMORY.md cross-Project experience, or structured entity research that is costly to reconstruct from ordinary public documentation.",
    "Use the Memory tool. Preserve source references, distinguish observed/derived/decision/hypothesis claims, and skip transient, obvious, secret-bearing, or unsupported content.",
    "A proposal requires user confirmation and affects future runs only.",
  ].join("\n");
  return [instructions.trim(), reminder].filter(Boolean).join("\n\n");
}

function globalMemoryFileName(target: GlobalMemoryTarget): "USER.md" | "MEMORY.md" {
  return target === "user" ? "USER.md" : "MEMORY.md";
}

function explicitMemoryRequest(input: string): boolean {
  return /(?:\bremember\b|\bsave (?:this|that) (?:to|in) memory\b|记住|记下来|保存到记忆|写入记忆)/iu.test(
    input,
  );
}

export const PERSONAL_MEMORY_MAX_CHARACTERS = 4_000;
const PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS = 160;

export const PersonalMemoryContentSchema = z
  .string()
  .max(PERSONAL_MEMORY_MAX_CHARACTERS)
  .refine((content) => content.trim().length > 0, "Personal Memory cannot be blank.")
  .refine(
    (content) => !hasUnsupportedControlCharacter(content),
    "Personal Memory contains unsupported control characters.",
  );

export const PersonalMemoryRecordSchema = z
  .object({
    content: PersonalMemoryContentSchema,
    updatedAt: z.iso.datetime(),
  })
  .strict();

export const PersonalMemorySaveInputSchema = z
  .object({ content: PersonalMemoryContentSchema })
  .strict();

export const PersonalMemoryForgetInputSchema = z.object({ confirmed: z.literal(true) }).strict();

export const PersonalMemoryAgentMutationSchema = z.discriminatedUnion("operation", [
  z
    .object({
      operation: z.literal("save"),
      content: PersonalMemoryContentSchema,
    })
    .strict(),
  z.object({ operation: z.literal("forget") }).strict(),
]);

export const PersonalMemorySnapshotSchema = z
  .object({
    source: z.literal("desktop_settings"),
    content: PersonalMemoryContentSchema,
    updatedAt: z.iso.datetime(),
    characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
  })
  .strict()
  .superRefine((snapshot, context) => {
    if (snapshot.characterCount !== snapshot.content.length) {
      context.addIssue({
        code: "custom",
        path: ["characterCount"],
        message: "Personal Memory character count does not match its snapshot.",
      });
    }
  });

export const PersonalMemoryStateSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("empty"),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
  z
    .object({
      status: z.literal("saved"),
      content: PersonalMemoryContentSchema,
      updatedAt: z.iso.datetime(),
      characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
]);

export const PersonalMemoryExecutionPathSchema = z.enum([
  "direct_agent",
  "external_acp",
  "workflow",
]);

export const PersonalMemoryNotUsedReasonSchema = z.enum(["empty", "no_agent_nodes"]);

export const PersonalMemoryUseReceiptSchema = z.discriminatedUnion("status", [
  z
    .object({
      status: z.literal("used"),
      source: z.literal("desktop_settings"),
      executionPath: PersonalMemoryExecutionPathSchema,
      agentCount: z.number().int().positive().max(10_000),
      summary: z.string().min(1).max(PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS),
      updatedAt: z.iso.datetime(),
      characterCount: z.number().int().positive().max(PERSONAL_MEMORY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
  z
    .object({
      status: z.literal("not_used"),
      source: z.literal("desktop_settings"),
      executionPath: PersonalMemoryExecutionPathSchema,
      agentCount: z.number().int().nonnegative().max(10_000),
      reason: PersonalMemoryNotUsedReasonSchema,
      summary: z.string().min(1).max(PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS),
      maxCharacters: z.literal(PERSONAL_MEMORY_MAX_CHARACTERS),
    })
    .strict(),
]);

export type PersonalMemoryRecord = z.infer<typeof PersonalMemoryRecordSchema>;
export type PersonalMemorySaveInput = z.infer<typeof PersonalMemorySaveInputSchema>;
export type PersonalMemoryForgetInput = z.infer<typeof PersonalMemoryForgetInputSchema>;
export type PersonalMemoryAgentMutation = z.infer<typeof PersonalMemoryAgentMutationSchema>;
export type PersonalMemorySnapshot = z.infer<typeof PersonalMemorySnapshotSchema>;
export type PersonalMemoryState = z.infer<typeof PersonalMemoryStateSchema>;
export type PersonalMemoryExecutionPath = z.infer<typeof PersonalMemoryExecutionPathSchema>;
export type PersonalMemoryNotUsedReason = z.infer<typeof PersonalMemoryNotUsedReasonSchema>;
export type PersonalMemoryUseReceipt = z.infer<typeof PersonalMemoryUseReceiptSchema>;

export function createPersonalMemorySnapshot(record: unknown): PersonalMemorySnapshot {
  const parsed = PersonalMemoryRecordSchema.parse(record);
  return Object.freeze(
    PersonalMemorySnapshotSchema.parse({
      source: "desktop_settings",
      content: parsed.content,
      updatedAt: parsed.updatedAt,
      characterCount: parsed.content.length,
    }),
  );
}

export function personalMemoryState(record: PersonalMemoryRecord | null): PersonalMemoryState {
  if (!record) {
    return PersonalMemoryStateSchema.parse({
      status: "empty",
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
  }
  const parsed = PersonalMemoryRecordSchema.parse(record);
  return PersonalMemoryStateSchema.parse({
    status: "saved",
    content: parsed.content,
    updatedAt: parsed.updatedAt,
    characterCount: parsed.content.length,
    maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
  });
}

export function appendPersonalMemoryInstructions(
  instructions: string,
  snapshotInput: unknown,
): string {
  const snapshot = PersonalMemorySnapshotSchema.parse(snapshotInput);
  const memoryInstructions = [
    "Personal context: read-only Personal Memory snapshot from Desktop Settings.",
    "Use these durable user-authored facts and preferences when relevant.",
    "Keep this separate from Activity Profile, Session history, Project context, and Skills.",
    "Do not claim to update or forget this active snapshot; if the host provides a PersonalMemory tool, use it only to propose changes for future runs.",
    `Snapshot updated: ${snapshot.updatedAt}`,
    `Memory content (JSON string): ${JSON.stringify(snapshot.content)}`,
  ].join("\n");
  return [instructions.trim(), memoryInstructions].filter(Boolean).join("\n\n");
}

export function buildPersonalMemoryUseReceipt(input: {
  snapshot?: PersonalMemorySnapshot | null;
  executionPath: PersonalMemoryExecutionPath;
  agentCount: number;
}): PersonalMemoryUseReceipt {
  const executionPath = PersonalMemoryExecutionPathSchema.parse(input.executionPath);
  const agentCount = z.number().int().nonnegative().max(10_000).parse(input.agentCount);
  const reason: PersonalMemoryNotUsedReason | undefined =
    agentCount === 0 ? "no_agent_nodes" : input.snapshot ? undefined : "empty";
  if (reason) {
    return PersonalMemoryUseReceiptSchema.parse({
      status: "not_used",
      source: "desktop_settings",
      executionPath,
      agentCount,
      reason,
      summary: notUsedSummary(reason),
      maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
    });
  }
  if (!input.snapshot) throw new Error("Personal Memory snapshot is required when Memory is used.");
  const snapshot = PersonalMemorySnapshotSchema.parse(input.snapshot);
  return PersonalMemoryUseReceiptSchema.parse({
    status: "used",
    source: "desktop_settings",
    executionPath,
    agentCount,
    summary: summarizePersonalMemory(snapshot.content),
    updatedAt: snapshot.updatedAt,
    characterCount: snapshot.characterCount,
    maxCharacters: PERSONAL_MEMORY_MAX_CHARACTERS,
  });
}

export function personalMemoryReceiptMessage(receiptInput: unknown): MessageChunk {
  const receipt = PersonalMemoryUseReceiptSchema.parse(receiptInput);
  const content =
    receipt.status === "used"
      ? [
          "Personal Memory used",
          "Source: Settings → Personal Memory",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Snapshot: ${receipt.characterCount.toLocaleString("en-US")} / ${receipt.maxCharacters.toLocaleString("en-US")} characters · updated ${receipt.updatedAt}`,
          `Summary: ${receipt.summary}`,
        ].join("\n")
      : [
          "Personal Memory not used",
          "Source: Settings → Personal Memory",
          `Execution: ${executionPathLabel(receipt.executionPath)} · ${receipt.agentCount} Agent${receipt.agentCount === 1 ? "" : "s"}`,
          `Reason: ${receipt.summary}`,
        ].join("\n");
  return {
    role: "system",
    kind: "message",
    content,
    render: { source: "personal_memory_receipt" },
    structuredContent: { personalMemory: receipt },
  };
}

function summarizePersonalMemory(content: string): string {
  const normalized = content.replace(/\s+/gu, " ").trim();
  if (normalized.length <= PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS) return normalized;
  return `${normalized.slice(0, PERSONAL_MEMORY_SUMMARY_MAX_CHARACTERS - 1).trimEnd()}…`;
}

function notUsedSummary(reason: PersonalMemoryNotUsedReason): string {
  if (reason === "empty") return "No Personal Memory is saved.";
  return "This workflow has no Agent nodes that can receive Personal Memory.";
}

function executionPathLabel(path: PersonalMemoryExecutionPath): string {
  if (path === "direct_agent") return "Direct Agent";
  if (path === "external_acp") return "External ACP";
  return "Workflow";
}

export function countPersonalMemoryAgentTargets(configInput: unknown): number {
  return countAgentNodes(SwarmConfigSchema.parse(configInput));
}

function countAgentNodes(config: SwarmConfig): number {
  return Object.values(config.nodes).reduce((count, node) => {
    if (node.kind === "agent") return count + 1;
    if (node.kind === "swarm") {
      return count + countAgentNodes(SwarmConfigSchema.parse(node.swarm));
    }
    return count;
  }, 0);
}

function hasUnsupportedControlCharacter(content: string): boolean {
  return [...content].some((character) => {
    const codePoint = character.codePointAt(0);
    return (
      codePoint !== undefined &&
      ((codePoint < 32 && codePoint !== 9 && codePoint !== 10 && codePoint !== 13) ||
        codePoint === 127)
    );
  });
}

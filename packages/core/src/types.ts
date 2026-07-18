import { z } from "zod";

// ── McpServer ────────────────────────────────────────────────────────────────

export const StdioServerConfigSchema = z.object({
  type: z.literal("stdio"),
  command: z.string().min(1),
  args: z.array(z.string()).optional(),
  env: z.record(z.string(), z.string()).optional(),
});

export const SseServerConfigSchema = z.object({
  type: z.literal("sse"),
  url: z.string().min(1),
});

export const McpServerConfigSchema = z.discriminatedUnion("type", [
  StdioServerConfigSchema,
  SseServerConfigSchema,
]);

// ── Hook ─────────────────────────────────────────────────────────────────────

export const HookConfigSchema = z.object({
  onStart: z.string().optional(),
  onEnd: z.string().optional(),
  onHandoff: z.string().optional(),
  onChunk: z.string().optional(),
});

// ── AgentBackend ─────────────────────────────────────────────────────────────

export const AgentBackendSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("swarmx") }),
  z.object({ type: z.literal("echo") }),
  z.object({ type: z.literal("claude_code") }),
  z.object({
    type: z.literal("custom"),
    program: z.string().min(1),
    args: z.array(z.string()).optional(),
  }),
]);

// ── ProcessOptions ───────────────────────────────────────────────────────────

export const ProcessOptionsSchema = z.object({
  currentDir: z.string().optional(),
  env: z.record(z.string(), z.string()).optional(),
  clearEnv: z.boolean().optional(),
});

// ── Agent ────────────────────────────────────────────────────────────────────

export const AgentConfigSchema = z.object({
  name: z
    .string()
    .min(1)
    .regex(
      /^[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*$/,
      "Agent name must match /[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*/",
    ),
  description: z.string().optional(),
  parameters: z.record(z.string(), z.unknown()).optional(),
  returns: z.record(z.string(), z.unknown()).optional(),
  model: z.string().optional(),
  instructions: z.string().optional(),
  client: z.record(z.string(), z.unknown()).optional(),
  mcpServers: z.record(z.string(), McpServerConfigSchema).optional(),
  hooks: z.array(HookConfigSchema).optional(),
  backend: AgentBackendSchema.optional(),
  process: ProcessOptionsSchema.optional(),
});

// ── Tool ─────────────────────────────────────────────────────────────────────

export const ToolConfigSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  parameters: z.record(z.string(), z.unknown()).optional(),
  returns: z.record(z.string(), z.unknown()).optional(),
  instructions: z.string().optional(),
  mcpServers: z.record(z.string(), McpServerConfigSchema).optional(),
});

// ── Edge ─────────────────────────────────────────────────────────────────────

export const EdgeConfigSchema = z.object({
  source: z.string().min(1),
  target: z.string().min(1),
  condition: z.string().optional(),
});

// ── Swarm (mutually recursive with SwarmNode via z.lazy) ────────────────────

export const SwarmNodeConfigSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("agent"),
    agent: AgentConfigSchema,
  }),
  z.object({
    kind: z.literal("tool"),
    tool: ToolConfigSchema,
  }),
  z.object({
    kind: z.literal("swarm"),
    swarm: z.lazy((): z.ZodTypeAny => SwarmConfigSchema),
  }),
]);

export const SwarmConfigSchema = z.object({
  name: z.string().min(1),
  description: z.string().optional(),
  parameters: z.record(z.string(), z.unknown()).optional(),
  returns: z.record(z.string(), z.unknown()).optional(),
  mcpServers: z.record(z.string(), McpServerConfigSchema).optional(),
  queen: AgentConfigSchema.optional(),
  nodes: z.record(z.string(), SwarmNodeConfigSchema),
  edges: z.array(EdgeConfigSchema),
  root: z.string().min(1),
  hooks: z.array(HookConfigSchema).optional(),
});

// ── Messages ──────────────────────────────────────────────────���──────────────

export const ChatMessageSchema = z.object({
  role: z.enum(["user", "assistant", "system", "tool"]),
  content: z.string(),
});

export const MessageRenderMetadataSchema = z
  .object({
    durationMs: z.number().int().nonnegative().optional(),
    endedAt: z.string().min(1).optional(),
    invocationId: z.string().min(1).optional(),
    parentMessageId: z.string().min(1).optional(),
    rawPayloadRef: z.string().min(1).optional(),
    source: z.string().min(1).optional(),
    startedAt: z.string().min(1).optional(),
    status: z
      .enum(["queued", "running", "succeeded", "failed", "canceled", "skipped", "completed"])
      .optional(),
  })
  .passthrough();

export const MessageChunkSchema = z.object({
  role: z.string(),
  content: z.string(),
  kind: z.enum(["message", "thinking", "tool_call", "tool_result"]),
  agent: z.string().optional(),
  swarmEvent: z.string().optional(),
  toolName: z.string().optional(),
  structuredContent: z.unknown().optional(),
  render: MessageRenderMetadataSchema.optional(),
});

export const ModelTokenUsageSchema = z.object({
  inputTokens: z.number().int().nonnegative().default(0),
  outputTokens: z.number().int().nonnegative().default(0),
  reasoningTokens: z.number().int().nonnegative().default(0),
  cachedInputTokens: z.number().int().nonnegative().default(0),
  totalTokens: z.number().int().nonnegative(),
  estimated: z.boolean().default(false),
  model: z.string().min(1).optional(),
  provider: z.string().min(1).optional(),
});

// ── Eval output ─────────────────────────────────────────────────────────────

export const EvalTraceEventSchema = z.object({
  runId: z.string(),
  swarm: z.string(),
  node: z.string(),
  kind: z.enum(["agent", "tool", "swarm"]),
  step: z.number().int().positive(),
  startedAt: z.string(),
  endedAt: z.string(),
  status: z.enum(["completed", "failed"]),
  messageCount: z.number().int().nonnegative(),
  error: z.string().optional(),
});

export const EvalRunResultSchema = z.object({
  output: z.string(),
  messages: z.array(MessageChunkSchema),
  trace: z.array(EvalTraceEventSchema),
  error: z.string().nullable(),
  metrics: z.object({
    steps: z.number().int().nonnegative(),
    messages: z.number().int().nonnegative(),
    toolCalls: z.number().int().nonnegative(),
    toolResults: z.number().int().nonnegative(),
  }),
});

// ── Session ──────────────────────────────────────────────────────────────────

export const SessionPermissionModeSchema = z.enum(["inherit", "default", "plan", "trusted"]);

export const SessionDataSchema = z.object({
  id: z.string(),
  title: z.string(),
  acpSessionId: z.string().optional(),
  projectId: z.string().optional(),
  cwd: z.string().optional(),
  agentName: z.string(),
  harness: z.string(),
  model: z.string().optional(),
  permissionMode: SessionPermissionModeSchema.default("inherit"),
  pinned: z.boolean().default(false),
  messages: z.array(MessageChunkSchema),
  archivedAt: z.string().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

// ── Inferred Types ───────────────────────────────────────────────────────────

export type McpServerConfig = z.infer<typeof McpServerConfigSchema>;
export type HookConfig = z.infer<typeof HookConfigSchema>;
export type AgentBackend = z.infer<typeof AgentBackendSchema>;
export type ProcessOptions = z.infer<typeof ProcessOptionsSchema>;
export type AgentConfig = z.infer<typeof AgentConfigSchema>;
export type ToolConfig = z.infer<typeof ToolConfigSchema>;
export type SwarmNodeConfig = z.infer<typeof SwarmNodeConfigSchema>;
export type SwarmConfig = z.infer<typeof SwarmConfigSchema>;
export type EdgeConfig = z.infer<typeof EdgeConfigSchema>;
export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type MessageRenderMetadata = z.infer<typeof MessageRenderMetadataSchema>;
export type MessageChunk = z.infer<typeof MessageChunkSchema>;
export type ModelTokenUsage = z.infer<typeof ModelTokenUsageSchema>;
export type EvalTraceEvent = z.infer<typeof EvalTraceEventSchema>;
export type EvalRunResult = z.infer<typeof EvalRunResultSchema>;
export type SessionPermissionMode = z.infer<typeof SessionPermissionModeSchema>;
export type SessionData = z.infer<typeof SessionDataSchema>;

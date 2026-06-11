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

export const MessageChunkSchema = z.object({
  role: z.string(),
  content: z.string(),
  kind: z.enum(["message", "thinking", "tool_call", "tool_result"]),
  agent: z.string().optional(),
  swarmEvent: z.string().optional(),
  toolName: z.string().optional(),
});

// ── Session ──────────────────────────────────────────────────────────────────

export const SessionDataSchema = z.object({
  id: z.string(),
  title: z.string(),
  acpSessionId: z.string().optional(),
  agentName: z.string(),
  harness: z.string(),
  model: z.string().optional(),
  messages: z.array(MessageChunkSchema),
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
export type MessageChunk = z.infer<typeof MessageChunkSchema>;
export type SessionData = z.infer<typeof SessionDataSchema>;

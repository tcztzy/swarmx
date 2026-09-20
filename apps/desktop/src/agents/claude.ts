import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { PassThrough } from "node:stream";
import {
  filterEscalatingDefaultMode,
  getSessionInfo,
  getSessionMessages,
  listSessions,
  type McpServerConfig,
  type Options,
  type PermissionResult,
  query,
  resolveSettings,
  type SDKMessage,
  type SDKResultMessage,
  type SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import { ElicitResultSchema } from "@modelcontextprotocol/sdk/types.js";
import type { ModelCatalog, RunOptions, RunResult } from "@swarmx/swarm";
import { z } from "zod";
import { requestApproval } from "./approval.js";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

const modeSchema = z.enum([
  "default",
  "acceptEdits",
  "plan",
  "dontAsk",
  "auto",
  "bypassPermissions",
]);
const effortSchema = z.enum(["low", "medium", "high", "xhigh", "max"]);
const messageSchema = z.object({
  id: z.string().optional(),
  content: z.union([z.string(), z.array(z.looseObject({ type: z.string() }))]),
});
const questionSchema = z.object({
  questions: z.array(
    z.object({
      question: z.string(),
      header: z.string().optional(),
      multiSelect: z.boolean().optional(),
      options: z.array(z.object({ label: z.string(), description: z.string().optional() })),
    }),
  ),
});

function resultOutcome(result: SDKResultMessage): RunResult {
  if (result.terminal_reason === "aborted_streaming" || result.terminal_reason === "aborted_tools")
    return { stopReason: "cancelled" };
  if (result.subtype === "error_max_turns") return { stopReason: "max_turn_requests" };
  if (result.subtype !== "success") throw new Error(result.errors.join("\n"));
  if (result.is_error) throw new Error(result.result);
  return {
    stopReason:
      result.stop_reason === "max_tokens"
        ? "max_tokens"
        : result.stop_reason === "refusal"
          ? "refusal"
          : "end_turn",
  };
}

/** Native messages are retained as raw records; only displayable blocks are projected. */
function projection(observer: Observer) {
  const streamed = new Set<string>();
  const messages = new Map<string | null, string>();
  const tools = new Map<string, { name: string; input: unknown }>();
  const blocks = (uuid: string, raw: unknown, role: "assistant" | "user", output?: unknown) => {
    const message = messageSchema.parse(raw);
    const id = message.id ?? uuid;
    if (typeof message.content === "string") {
      observer.text(id, message.content, role);
      return;
    }
    for (const [index, value] of message.content.entries()) {
      const blockId = `${id}:${index}`;
      switch (value.type) {
        case "text": {
          const block = z.object({ text: z.string() }).parse(value);
          if (!streamed.has(blockId)) observer.text(blockId, block.text, role);
          break;
        }
        case "thinking": {
          const block = z.object({ thinking: z.string() }).parse(value);
          if (!streamed.has(blockId)) observer.text(blockId, block.thinking, "reasoning");
          break;
        }
        case "tool_use": {
          const block = z
            .object({ id: z.string(), name: z.string(), input: z.unknown() })
            .parse(value);
          tools.set(block.id, block);
          observer.activity?.({ type: "tool", toolCallId: block.id, status: "in_progress" });
          observer.tool(block.id, block.name, block.input);
          break;
        }
        case "tool_result": {
          const block = z
            .object({
              tool_use_id: z.string(),
              content: z.unknown(),
              is_error: z.boolean().optional(),
            })
            .parse(value);
          const tool = tools.get(block.tool_use_id);
          observer.activity?.({
            type: "tool",
            toolCallId: block.tool_use_id,
            status: block.is_error ? "failed" : "completed",
          });
          observer.tool(
            block.tool_use_id,
            tool?.name ?? "tool",
            tool?.input ?? {},
            output ?? block.content,
          );
          break;
        }
      }
    }
  };
  return {
    blocks,
    event(message: SDKMessage) {
      observer.raw(
        message,
        message.type === "system" && message.subtype === "init"
          ? {
              "swarmx.agent.model": message.model,
              "swarmx.agent.mode": message.permissionMode,
              "swarmx.harness.version": message.claude_code_version,
            }
          : message.type === "assistant"
            ? { "gen_ai.response.model": message.message.model ?? null }
            : undefined,
      );
      if (message.type === "assistant" || message.type === "user")
        blocks(
          message.uuid ?? randomUUID(),
          message.message,
          message.type,
          message.type === "user" ? message.tool_use_result : undefined,
        );
      if (message.type !== "stream_event") return;
      const event = message.event;
      if (event.type === "message_start")
        messages.set(message.parent_tool_use_id, event.message.id);
      if (event.type !== "content_block_delta") return;
      const messageId = messages.get(message.parent_tool_use_id);
      if (!messageId) throw new Error("Claude streamed a block without a message start.");
      const id = `${messageId}:${event.index}`;
      if (event.delta.type === "text_delta" || event.delta.type === "thinking_delta") {
        streamed.add(id);
        observer.text(
          id,
          event.delta.type === "text_delta" ? event.delta.text : event.delta.thinking,
          event.delta.type === "text_delta" ? "assistant" : "reasoning",
        );
      }
    },
  };
}

export async function createClaude(options: AgentOptions): Promise<NativeAgent> {
  const fresh = new Map<string, string | undefined>();
  const runtimes = new Map<string, ReturnType<typeof open>>();
  type Active = {
    observer: Observer;
    project: ReturnType<typeof projection>;
    completion: ReturnType<typeof Promise.withResolvers<RunResult | Error>>;
    runtime?: ReturnType<typeof open>;
    result?: SDKResultMessage;
    stopped: boolean;
    submitted: boolean;
  };
  const running = new Map<string, Active>();
  let disposed = false;
  const available = () => {
    if (disposed) throw new Error("Claude Agent is disposed.");
  };
  const checkPrompt = (text: string) => {
    if (options.reviewOnly && /^\s*\/[a-z][a-z0-9_-]*(?:\s|$)/iu.test(text))
      throw new Error("Memory reviews cannot dispatch native commands.");
  };
  const owned = async (sessionId: string) => {
    if (fresh.has(sessionId) || runtimes.has(sessionId)) return;
    const info = await getSessionInfo(sessionId, { dir: options.cwd });
    if (!info || (info.cwd !== undefined && info.cwd !== options.cwd))
      throw new Error("Claude session does not belong to this directory.");
  };

  function open(sessionId?: string, instructions?: string) {
    const input = new PassThrough({ objectMode: true });
    let childClosed: Promise<void> = Promise.resolve();
    let closing = false;
    let closePromise: Promise<void> | undefined;
    let failure: Error | undefined;
    const current: RunOptions = {};
    const active = () => (sessionId === undefined ? undefined : running.get(sessionId));
    const readonly = options.reviewOnly || sessionId === undefined;
    const native = query({
      prompt: input as AsyncIterable<SDKUserMessage>,
      options: {
        cwd: options.cwd,
        settingSources: ["user", "project", "local"],
        includePartialMessages: true,
        ...(sessionId === undefined
          ? {}
          : fresh.has(sessionId)
            ? { sessionId }
            : { resume: sessionId }),
        ...(instructions
          ? {
              systemPrompt: {
                type: "preset",
                preset: "claude_code",
                append: memoryContextSuffix(instructions),
              },
            }
          : {}),
        // Allows an explicitly selected native bypass mode; does not select that mode.
        ...(!readonly
          ? { allowDangerouslySkipPermissions: true }
          : {
              tools: [],
              disallowedTools: ["Agent", "Task", "ExitPlanMode", "EnterWorktree"],
              mcpServers: {},
              strictMcpConfig: true,
              persistSession: false,
              permissionMode: "dontAsk",
              settings: { disableAllHooks: true },
              sandbox: {
                enabled: true,
                failIfUnavailable: true,
                autoAllowBashIfSandboxed: false,
                allowUnsandboxedCommands: false,
                network: { allowedDomains: [], strictAllowlist: true },
                filesystem: { denyWrite: ["/"] },
              },
            }),
        spawnClaudeCodeProcess({ command, args, cwd, env, signal }) {
          const child = spawn(command, args, {
            cwd,
            env,
            signal,
            stdio: ["pipe", "pipe", "inherit"],
          });
          childClosed = new Promise((resolve) => child.once("close", () => resolve()));
          return child;
        },
        async canUseTool(name, input, context) {
          const run = active();
          const deny: PermissionResult = {
            behavior: "deny",
            message: "Tool request declined.",
            toolUseID: context.toolUseID,
          };
          if (!run || run.stopped || readonly) return deny;
          if (name === "AskUserQuestion") {
            const { questions } = questionSchema.parse(input);
            const fields = questions.flatMap((question, index) => {
              const values = question.options.map((option) => option.label);
              const choices = {
                type: "string",
                oneOf: question.options.map((option) => ({
                  const: option.label,
                  title: option.description
                    ? `${option.label}: ${option.description}`
                    : option.label,
                })),
              };
              return [
                {
                  key: `answer_${index}`,
                  schema: {
                    ...(question.multiSelect ? { type: "array", items: choices } : choices),
                    title: question.question,
                  },
                  validate: (question.multiSelect
                    ? z.array(z.enum(values))
                    : z.enum(values)
                  ).optional(),
                },
                {
                  key: `other_${index}`,
                  schema: {
                    type: "string",
                    title: `${question.header ?? question.question} — Other answer`,
                  },
                  validate: z.string().optional(),
                },
              ];
            });
            const answer = await run.observer.interact(
              {
                id: context.requestId,
                title: context.title ?? "Claude needs input",
                schema: {
                  type: "object",
                  properties: Object.fromEntries(fields.map((field) => [field.key, field.schema])),
                  additionalProperties: false,
                },
              },
              context.signal,
            );
            if (answer === undefined || context.signal.aborted) return deny;
            const parsed = z
              .strictObject(Object.fromEntries(fields.map((field) => [field.key, field.validate])))
              .parse(answer);
            const answers = Object.fromEntries(
              questions.map((question, index) => {
                const selected = parsed[`answer_${index}`];
                const other = parsed[`other_${index}`];
                const text = [
                  ...(Array.isArray(selected) ? selected : selected ? [selected] : []),
                  ...(other ? [other] : []),
                ].join(", ");
                if (!text)
                  throw new Error(`Claude question requires an answer: ${question.question}`);
                return [question.question, text];
              }),
            );
            return { behavior: "allow", updatedInput: { ...input, answers } };
          }
          return (
            (await requestApproval<PermissionResult>(
              run.observer,
              context.requestId,
              context.title ?? context.decisionReason ?? `Allow ${name}?`,
              context.toolUseID,
              [
                {
                  id: "allow",
                  label: "Allow once",
                  kind: "allow_once",
                  answer: { behavior: "allow", updatedInput: input, toolUseID: context.toolUseID },
                },
                ...(context.suggestions?.length
                  ? [
                      {
                        id: "remember",
                        label: "Always allow",
                        kind: "allow_always" as const,
                        answer: {
                          behavior: "allow" as const,
                          updatedInput: input,
                          updatedPermissions: context.suggestions,
                          toolUseID: context.toolUseID,
                        },
                      },
                    ]
                  : []),
                { id: "deny", label: "Decline", kind: "reject_once", answer: deny },
              ],
              context.signal,
            )) ?? deny
          );
        },
        async onElicitation(request, context) {
          const run = active();
          if (!run || run.stopped || readonly || request.mode === "url")
            return { action: "cancel" };
          if (!request.requestedSchema) throw new Error("Claude elicitation has no form schema.");
          const answer = await run.observer.interact(
            { id: context.requestId, title: request.message, schema: request.requestedSchema },
            context.signal,
          );
          if (answer === undefined || context.signal.aborted) return { action: "cancel" };
          return ElicitResultSchema.parse({ action: "accept", content: answer });
        },
      } satisfies Options,
    });
    const consume = (async () => {
      try {
        for await (const message of native) {
          if (message.type === "system" && message.subtype === "init") {
            Object.assign(current, { model: message.model, mode: message.permissionMode });
          }
          if (message.type === "system" && message.subtype === "status" && message.permissionMode)
            Object.assign(current, { mode: message.permissionMode });
          const run = active();
          if (!run?.submitted || run.stopped) continue;
          run.project.event(message);
          if (message.type === "result") {
            run.result = message;
            const outcome = resultOutcome(message);
            if (outcome.stopReason !== "end_turn") {
              run.submitted = false;
              run.completion.resolve(outcome);
            }
          }
          if (
            message.type === "system" &&
            message.subtype === "session_state_changed" &&
            message.state === "idle"
          ) {
            if (!run.result) throw new Error("Claude became idle without a terminal result.");
            run.submitted = false;
            run.completion.resolve(resultOutcome(run.result));
          }
        }
        if (!closing) throw new Error("Claude output ended without closing the session.");
      } catch (error) {
        if (closing) return;
        failure = error instanceof Error ? error : new Error(String(error));
        active()?.completion.resolve(failure);
        native.close();
      }
    })();
    return {
      native,
      current,
      get failure() {
        return failure;
      },
      send(text: string) {
        input.write({
          type: "user",
          uuid: randomUUID(),
          ...(sessionId === undefined ? {} : { session_id: sessionId }),
          parent_tool_use_id: null,
          message: { role: "user", content: text },
        } satisfies SDKUserMessage);
      },
      close() {
        closePromise ??= (async () => {
          closing = true;
          input.end();
          native.close();
          await childClosed;
          await consume;
        })();
        return closePromise;
      },
    };
  }

  return {
    name: "claude",
    capabilities: HARNESS_CAPABILITIES.claude,
    restoreEmptySessions(ids) {
      for (const id of ids) fresh.set(id, undefined);
    },
    async list() {
      available();
      const sessions = (await listSessions({ dir: options.cwd, includeWorktrees: false })).filter(
        (session) => session.cwd === undefined || session.cwd === options.cwd,
      );
      for (const session of sessions) fresh.delete(session.sessionId);
      return [
        ...sessions.map((session) => ({
          sessionId: session.sessionId,
          title: session.customTitle ?? session.summary,
          updatedAt: new Date(session.lastModified).toISOString(),
        })),
        ...[...fresh.keys()].map((sessionId) => ({ sessionId })),
      ];
    },
    async create(selection) {
      available();
      const id = randomUUID();
      fresh.set(id, selection?.instructions);
      return id;
    },
    async models(sessionId) {
      available();
      if (sessionId !== undefined) await owned(sessionId);
      const existing = sessionId === undefined ? undefined : runtimes.get(sessionId);
      if (existing?.failure) throw existing.failure;
      const runtime = existing ?? open();
      try {
        const models = await runtime.native.supportedModels();
        const settings = filterEscalatingDefaultMode(await resolveSettings({ cwd: options.cwd }));
        return {
          models: models.map((model) => ({
            id: model.value,
            name: model.displayName,
            description: model.description,
            efforts: (model.supportedEffortLevels ?? []).map((id) => ({ id, name: id })),
          })),
          modes: modeSchema.options.map((id) => ({ id, name: id })),
          current: existing
            ? { ...existing.current }
            : sessionId !== undefined && !fresh.has(sessionId)
              ? {}
              : {
                  model: settings.model,
                  effort: settings.effortLevel,
                  mode: settings.permissions?.defaultMode,
                },
        } satisfies ModelCatalog;
      } finally {
        if (!existing) await runtime.close();
      }
    },
    async read(sessionId, observer) {
      available();
      await owned(sessionId);
      if (fresh.has(sessionId)) return;
      const project = projection(observer);
      for (const message of await getSessionMessages(sessionId, { dir: options.cwd })) {
        observer.raw(message);
        if (message.type === "assistant" || message.type === "user")
          project.blocks(message.uuid, message.message, message.type);
      }
    },
    async start(sessionId, text, observer, selection) {
      available();
      checkPrompt(text);
      if (options.reviewOnly && selection?.mode !== undefined)
        throw new Error("Memory reviews cannot select an ordinary task mode.");
      const mode = selection?.mode === undefined ? undefined : modeSchema.parse(selection.mode);
      const effort =
        selection?.effort === undefined ? undefined : effortSchema.parse(selection.effort);
      if (running.has(sessionId)) throw new Error("Session is busy.");
      const run: Active = {
        observer,
        project: projection(observer),
        completion: Promise.withResolvers(),
        stopped: false,
        submitted: false,
      };
      running.set(sessionId, run);
      let endpoint: ReturnType<NonNullable<AgentOptions["registerMcp"]>> | undefined;
      try {
        await owned(sessionId);
        if (run.stopped) return { stopReason: "cancelled" };
        const runtime =
          runtimes.get(sessionId) ??
          open(sessionId, fresh.get(sessionId) ?? selection?.instructions);
        run.runtime = runtime;
        runtimes.set(sessionId, runtime);
        if (runtime.failure) throw runtime.failure;
        await runtime.native.initializationResult();
        if (run.stopped) return { stopReason: "cancelled" };
        if (selection?.model !== undefined) {
          await runtime.native.setModel(selection.model);
          Object.assign(runtime.current, { model: selection.model });
        }
        if (effort !== undefined) {
          await runtime.native.applyFlagSettings({ effortLevel: effort });
          Object.assign(runtime.current, { effort });
        }
        if (mode !== undefined) {
          await runtime.native.setPermissionMode(mode);
          Object.assign(runtime.current, { mode });
        }
        const servers: Record<string, McpServerConfig> = {};
        if (!options.reviewOnly) {
          const token = randomUUID();
          if (options.registerMcp) {
            if (!observer.executionId)
              throw new Error("Claude tool access requires a bound Host execution.");
            endpoint = options.registerMcp(token);
            endpoint.bind(`claude:${sessionId}`, observer.executionId);
          }
          servers.swarmx = {
            type: "stdio",
            command: options.mcp.command,
            args: [...options.mcp.args],
            env: endpoint
              ? { ...options.mcp.env, SWARMX_MCP_TOKEN: token }
              : { ...options.mcp.env },
          };
          const response = await runtime.native.setMcpServers(servers);
          if (Object.keys(response.errors).length)
            throw new Error(`Claude MCP configuration failed: ${JSON.stringify(response.errors)}`);
        }
        if (run.stopped) return { stopReason: "cancelled" };
        run.submitted = true;
        runtime.send(text);
        fresh.delete(sessionId);
        const result = await run.completion.promise;
        if (result instanceof Error) throw result;
        if (result.stopReason !== "end_turn") {
          await runtime.close();
          runtimes.delete(sessionId);
        }
        return result;
      } catch (error) {
        if (run.runtime) {
          await run.runtime.close();
          runtimes.delete(sessionId);
        }
        if (run.stopped) return { stopReason: "cancelled" };
        throw error;
      } finally {
        endpoint?.dispose();
        if (options.reviewOnly || run.stopped) {
          await run.runtime?.close();
          runtimes.delete(sessionId);
        }
        running.delete(sessionId);
      }
    },
    async steer(sessionId, text) {
      checkPrompt(text);
      const run = running.get(sessionId);
      if (!run?.submitted || run.stopped || !run.runtime)
        throw new Error("No running Claude session.");
      run.runtime.send(text);
    },
    async interrupt(sessionId) {
      const run = running.get(sessionId);
      if (!run || run.stopped) return;
      run.stopped = true;
      // Closing the owned SDK process also discards queued steering inputs.
      if (run.runtime) await run.runtime.close();
      run.completion.resolve({ stopReason: "cancelled" });
    },
    async dispose() {
      disposed = true;
      for (const run of running.values()) run.stopped = true;
      await Promise.all([...runtimes.values()].map((runtime) => runtime.close()));
      for (const run of running.values()) run.completion.resolve({ stopReason: "cancelled" });
      runtimes.clear();
    },
  };
}

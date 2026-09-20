import { randomUUID } from "node:crypto";
import {
  getSupportedThinkingLevels,
  type Model,
  type ToolResultMessage,
} from "@earendil-works/pi-ai";
import {
  type AgentSession,
  createAgentSession,
  DefaultResourceLoader,
  defineTool,
  getAgentDir,
  ModelRuntime,
  SessionManager,
  VERSION,
} from "@earendil-works/pi-coding-agent";
import type { RunResult } from "@swarmx/swarm";
import {
  type AgentOptions,
  type EventAttributes,
  HARNESS_CAPABILITIES,
  type NativeAgent,
  type Observer,
} from "./types.js";

type Message = AgentSession["messages"][number];
const modelId = (model: Model<string>) => `${model.provider}/${model.id}`;

function projection(observer: Observer, productTools: AgentOptions["productTools"]) {
  const tools = new Map<string, { name: string; input: unknown }>();
  return {
    raw(event: unknown, attributes?: EventAttributes) {
      observer.raw(JSON.parse(JSON.stringify(event)), attributes);
    },
    tool(
      id: string,
      name: string,
      input: unknown = tools.get(id)?.input,
      output?: Pick<ToolResultMessage<unknown>, "content" | "details">,
      isError = false,
    ) {
      tools.set(id, { name, input });
      observer.tool(
        id,
        name,
        input,
        output && !isError && productTools?.definitions.some((tool) => tool.name === name)
          ? output.details
          : output,
      );
    },
    message(message: Message, id: string, streamed = new Set<number>(), history = false) {
      if (message.role === "user") {
        const text =
          typeof message.content === "string"
            ? message.content
            : message.content
                .filter((block) => block.type === "text")
                .map((block) => block.text)
                .join("");
        if (text) observer.text(id, text, "user");
      } else if (message.role === "assistant") {
        for (const [index, block] of message.content.entries()) {
          if (block.type === "text" && !streamed.has(index))
            observer.text(`${id}:${index}`, block.text);
          else if (block.type === "thinking" && !streamed.has(index))
            observer.text(`${id}:${index}`, block.thinking, "reasoning");
          else if (block.type === "toolCall" && history)
            this.tool(block.id, block.name, block.arguments);
        }
      } else if (message.role === "toolResult" && history) {
        const tool = tools.get(message.toolCallId);
        observer.activity?.({
          type: "tool",
          toolCallId: message.toolCallId,
          status: message.isError ? "failed" : "completed",
        });
        this.tool(
          message.toolCallId,
          message.toolName,
          tool?.input ?? {},
          message,
          message.isError,
        );
      }
    },
  };
}

export async function createPi(options: AgentOptions): Promise<NativeAgent> {
  const modelRuntime = await ModelRuntime.create();
  const empty = new Map<string, SessionManager>();
  const running = new Map<
    string,
    {
      controller: AbortController;
      session?: AgentSession;
      settled: Promise<void>;
    }
  >();
  let disposed = false;
  const available = () => {
    if (disposed) throw new Error("Pi Agent is disposed.");
  };
  const manager = async (id: string) => {
    available();
    const stored = (await SessionManager.list(options.cwd)).find((session) => session.id === id);
    const session = stored ? SessionManager.open(stored.path) : empty.get(id);
    if (!session) throw new Error("Pi session does not belong to this directory.");
    return session;
  };
  const open = async (
    sessionManager: SessionManager,
    instructions?: string,
    signal?: AbortSignal,
  ) => {
    const resourceLoader = new DefaultResourceLoader({
      cwd: options.cwd,
      agentDir: getAgentDir(),
      ...(instructions ? { appendSystemPrompt: [instructions] } : {}),
    });
    await resourceLoader.reload();
    const productTools = options.productTools;
    return createAgentSession({
      cwd: options.cwd,
      modelRuntime,
      sessionManager,
      resourceLoader,
      ...(options.reviewOnly ? { noTools: "all" as const } : {}),
      customTools: productTools
        ? productTools.definitions.map((tool) =>
            defineTool({
              name: tool.name,
              label: tool.name,
              description: tool.description,
              parameters: tool.inputSchema,
              execute: async (callId, args, toolSignal) => {
                if (!signal) throw new Error("Product tools require an active Pi run.");
                const result = await productTools.call(
                  tool.name,
                  args,
                  callId,
                  toolSignal ? AbortSignal.any([signal, toolSignal]) : signal,
                );
                return {
                  content: [{ type: "text", text: JSON.stringify(result) }],
                  details: result,
                };
              },
            }),
          )
        : [],
    });
  };
  return {
    name: "pi",
    capabilities: HARNESS_CAPABILITIES.pi,
    restoreEmptySessions(ids) {
      for (const id of ids) empty.set(id, SessionManager.create(options.cwd, undefined, { id }));
    },
    async list() {
      available();
      const stored = await SessionManager.list(options.cwd);
      for (const session of stored) empty.delete(session.id);
      return [
        ...stored.map((session) => ({
          sessionId: session.id,
          title: session.name ?? session.firstMessage,
          updatedAt: session.modified.toISOString(),
        })),
        ...[...empty.keys()].map((sessionId) => ({ sessionId })),
      ];
    },
    async create() {
      available();
      const session = SessionManager.create(options.cwd);
      empty.set(session.getSessionId(), session);
      return session.getSessionId();
    },
    async models(id) {
      available();
      const snapshot = SessionManager.inMemory(
        options.cwd,
        undefined,
        id ? (await manager(id)).getBranch() : undefined,
      );
      const { session } = await open(snapshot);
      try {
        return {
          models: (await modelRuntime.getAvailable()).map((model) => ({
            id: modelId(model),
            name: model.name,
            description: model.provider,
            efforts: getSupportedThinkingLevels(model).map((id) => ({ id, name: id })),
          })),
          current: {
            ...(session.model ? { model: modelId(session.model) } : {}),
            effort: session.thinkingLevel,
          },
        };
      } finally {
        session.dispose();
      }
    },
    async read(id, observer) {
      const view = projection(observer, options.productTools);
      for (const entry of (await manager(id)).getBranch()) {
        view.raw(entry);
        if (entry.type === "message") view.message(entry.message, entry.id, undefined, true);
      }
    },
    async start(id, text, observer, selection = {}) {
      available();
      if (running.has(id)) throw new Error("Pi session is busy.");
      if (selection.mode !== undefined)
        throw new Error("Pi does not expose native permission modes.");
      const settled = Promise.withResolvers<void>();
      const active: {
        controller: AbortController;
        session?: AgentSession;
        settled: Promise<void>;
      } = {
        controller: new AbortController(),
        settled: settled.promise,
      };
      running.set(id, active);
      const view = projection(observer, options.productTools);
      let succeeded = false;
      try {
        const result = await open(
          await manager(id),
          selection.instructions,
          active.controller.signal,
        );
        const session = result.session;
        active.session = session;
        if (result.modelFallbackMessage)
          view.raw({ type: "model_fallback", message: result.modelFallbackMessage });
        if (active.controller.signal.aborted) return { stopReason: "cancelled" };
        if (selection.model !== undefined) {
          const model = (await modelRuntime.getAvailable()).find(
            (model) => modelId(model) === selection.model,
          );
          if (!model) throw new Error(`Unknown or unavailable Pi model "${selection.model}".`);
          await session.setModel(model);
        }
        if (selection.effort !== undefined) {
          const level = session
            .getAvailableThinkingLevels()
            .find((level) => level === selection.effort);
          if (!level) throw new Error(`Unsupported Pi thinking level "${selection.effort}".`);
          session.setThinkingLevel(level);
        }
        let outcome: RunResult | undefined;
        let failure: Error | undefined;
        let messageId = "";
        let streamed = new Set<number>();
        const measured = new Set<Message>();
        let usageKnown = true;
        let inputTokens = 0;
        let outputTokens = 0;
        let costUsd = 0;
        session.subscribe((event) => {
          const assistant =
            event.type === "message_end" && event.message.role === "assistant"
              ? event.message
              : undefined;
          if (assistant && !measured.has(assistant)) {
            measured.add(assistant);
            const usage = assistant.usage;
            // SDK zero placeholders after missing usage/errors cannot prove zero consumption.
            usageKnown &&=
              assistant.stopReason !== "error" &&
              assistant.stopReason !== "aborted" &&
              usage.totalTokens > 0 &&
              [
                usage.input,
                usage.cacheRead,
                usage.cacheWrite,
                usage.output,
                usage.cost.total,
              ].every((value) => Number.isFinite(value) && value >= 0);
            inputTokens += usage.input + usage.cacheRead + usage.cacheWrite;
            outputTokens += usage.output;
            costUsd += usage.cost.total;
          }
          view.raw(
            event,
            assistant
              ? {
                  "gen_ai.response.model":
                    assistant.responseModel ??
                    (assistant.model !== session.model?.id ? assistant.model : null),
                  "gen_ai.provider.name": assistant.provider,
                }
              : undefined,
          );
          if (event.type === "message_start") {
            messageId = randomUUID();
            streamed = new Set();
          } else if (event.type === "message_update") {
            const delta = event.assistantMessageEvent;
            if (delta.type === "text_delta" || delta.type === "thinking_delta") {
              streamed.add(delta.contentIndex);
              observer.text(
                `${messageId}:${delta.contentIndex}`,
                delta.delta,
                delta.type === "thinking_delta" ? "reasoning" : "assistant",
              );
            }
          } else if (event.type === "message_end") {
            view.message(event.message, messageId, streamed);
          } else if (event.type === "tool_execution_start") {
            observer.activity?.({
              type: "tool",
              toolCallId: event.toolCallId,
              status: "in_progress",
            });
            view.tool(event.toolCallId, event.toolName, event.args);
          } else if (event.type === "tool_execution_end") {
            observer.activity?.({
              type: "tool",
              toolCallId: event.toolCallId,
              status: event.isError ? "failed" : "completed",
            });
            view.tool(event.toolCallId, event.toolName, undefined, event.result, event.isError);
          } else if (event.type === "agent_end" && !event.willRetry) {
            const last = event.messages.findLast((message) => message.role === "assistant");
            if (
              last?.stopReason === "stop" ||
              last?.stopReason === "length" ||
              last?.stopReason === "aborted"
            )
              outcome = {
                stopReason:
                  last.stopReason === "stop"
                    ? "end_turn"
                    : last.stopReason === "length"
                      ? "max_tokens"
                      : "cancelled",
              };
            else
              failure = new Error(
                last?.errorMessage ?? "Pi ended without a terminal assistant result.",
              );
          }
        });
        await session.bindExtensions({ mode: "json", onError: (error) => view.raw(error) });
        if (active.controller.signal.aborted) return { stopReason: "cancelled" };
        view.raw(
          {
            type: "run_config",
            model: session.model && modelId(session.model),
            thinkingLevel: session.thinkingLevel,
          },
          {
            "swarmx.agent.model": session.model && modelId(session.model),
            "swarmx.agent.effort": session.thinkingLevel,
            "swarmx.harness.version": VERSION,
          },
        );
        try {
          await session.prompt(text, {
            preflightResult(accepted) {
              // SDK preparation can still be awaiting extensions while abort() sees idle.
              if (accepted) active.controller.signal.throwIfAborted();
            },
          });
        } catch (error) {
          if (active.controller.signal.aborted && error === active.controller.signal.reason)
            return { stopReason: "cancelled" };
          throw error;
        }
        if (failure) throw failure;
        if (!outcome) throw new Error("Pi ended without a terminal assistant result.");
        view.raw(
          { type: "run_usage" },
          {
            "gen_ai.usage.input_tokens": usageKnown && measured.size ? inputTokens : null,
            "gen_ai.usage.output_tokens": usageKnown && measured.size ? outputTokens : null,
            "swarmx.usage.cost_usd": usageKnown && measured.size ? costUsd : null,
            "swarmx.usage.basis":
              "Pi assistant messages; input includes cache; SDK estimated USD; excludes auxiliary calls",
          },
        );
        succeeded = true;
        return outcome;
      } finally {
        let cleanup: unknown;
        try {
          active.session?.dispose();
        } catch (error) {
          cleanup = error;
          view.raw({
            type: "cleanup_failure",
            message: error instanceof Error ? error.message : String(error),
          });
        } finally {
          running.delete(id);
          settled.resolve();
        }
        // biome-ignore lint/correctness/noUnsafeFinally: cleanup already ran; only a successful run rejects here.
        if (succeeded && cleanup !== undefined) throw cleanup;
      }
    },
    async steer(id, text) {
      const active = running.get(id);
      if (!active?.session || active.controller.signal.aborted)
        throw new Error("Pi session is not running.");
      await active.session.steer(text);
    },
    async interrupt(id) {
      const active = running.get(id);
      if (!active) return;
      active.controller.abort();
      active.session?.clearQueue();
      await active.session?.abort();
    },
    async dispose() {
      disposed = true;
      await Promise.all(
        [...running.values()].map(async (active) => {
          active.controller.abort();
          active.session?.clearQueue();
          await active.session?.abort();
          await active.settled;
        }),
      );
    },
  };
}

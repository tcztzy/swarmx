import { randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import * as acp from "@agentclientprotocol/sdk";
import type { ModelCatalog, Session } from "@swarmx/swarm";
import { z } from "zod";
import type { AgentId } from "../agent.js";
import { connectAcpProcess } from "./acp-process.js";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

const require = createRequire(import.meta.url);
type Configuration = Pick<acp.NewSessionResponse, "configOptions" | "modes">;
const choices = (option?: acp.SessionConfigOption) =>
  option?.type === "select"
    ? option.options.flatMap((entry) => ("group" in entry ? entry.options : [entry]))
    : [];

export function modelCatalog(configuration: Configuration): ModelCatalog {
  const model = configuration.configOptions?.find((option) => option.category === "model");
  const effort = configuration.configOptions?.find((option) => option.category === "thought_level");
  const mode = configuration.configOptions?.find((option) => option.category === "mode");
  const modes =
    mode?.type === "select"
      ? choices(mode).map(({ value, name, description }) => ({
          id: value,
          name,
          ...(description == null ? {} : { description }),
        }))
      : configuration.modes?.availableModes.map(({ id, name, description }) => ({
          id,
          name,
          ...(description == null ? {} : { description }),
        }));
  return {
    ...(modes ? { modes } : {}),
    models: choices(model).map((entry) => ({
      id: entry.value,
      name: entry.name,
      ...(entry.description == null ? {} : { description: entry.description }),
      efforts: choices(effort).map((entry) => ({ id: entry.value, name: entry.name })),
    })),
    current: {
      ...(model?.type === "select" ? { model: model.currentValue } : {}),
      ...(effort?.type === "select" ? { effort: effort.currentValue } : {}),
      ...(mode?.type === "select"
        ? { mode: mode.currentValue }
        : configuration.modes
          ? { mode: configuration.modes.currentModeId }
          : {}),
    },
  };
}

export async function createAcpHarness(id: AgentId, options: AgentOptions): Promise<NativeAgent> {
  if (options.reviewOnly && id !== "claude" && id !== "codex")
    throw new Error(`Harness "${id}" cannot run a restricted memory review.`);
  const codexConfig =
    id === "codex"
      ? z.record(z.string(), z.unknown()).parse(JSON.parse(process.env.CODEX_CONFIG ?? "{}"))
      : {};
  const launch = () => {
    if (id === "codex" || id === "claude") {
      const pkg = id === "codex" ? "codex-acp" : "claude-agent-acp";
      const config =
        id === "codex" && options.reviewOnly
          ? {
              ...codexConfig,
              sandbox_mode: "read-only",
              approval_policy: "never",
              sandbox_workspace_write: {
                writable_roots: [options.cwd],
                network_access: false,
                exclude_tmpdir_env_var: true,
                exclude_slash_tmp: true,
              },
              web_search: "disabled",
              features: Object.fromEntries(
                [
                  "multi_agent",
                  "multi_agent_v2",
                  "collab",
                  "plugins",
                  "plugin_hooks",
                  "apps",
                  "connectors",
                  "browser_use",
                  "computer_use",
                  "in_app_browser",
                  "hooks",
                  "codex_hooks",
                  "remote_control",
                  "request_permissions_tool",
                  "request_permissions",
                  "request_rule",
                  "goals",
                  ...(options.reviewOnly
                    ? [
                        "shell_tool",
                        "unified_exec",
                        "apply_patch_freeform",
                        "js_repl",
                        "code_mode",
                        "code_mode_only",
                        "image_generation",
                        "memory_tool",
                        "memories",
                        "tool_suggest",
                        "view_image",
                        "sleep_tool",
                        "tool_search",
                        "search_tool",
                        "imagegenext",
                        "codex_git_commit",
                        "skill_search",
                        "in_app_local_automation",
                        "send_async_message",
                        "in_app_chat",
                        "recommended_plugins",
                      ]
                    : []),
                ].map((feature) => [feature, false]),
              ),
              ...(options.reviewOnly
                ? {
                    tools: {
                      update_plan: { enabled: false },
                      experimental_request_user_input: { enabled: false },
                    },
                  }
                : {}),
            }
          : undefined;
      return {
        command: process.execPath,
        args: [require.resolve(`@agentclientprotocol/${pkg}/dist/index.js`)],
        env: {
          ...process.env,
          ELECTRON_RUN_AS_NODE: "1",
          ...(options.reviewOnly
            ? {
                ...(id === "codex" ? { ACP_DISABLE_TITLE_GENERATION: "1" } : {}),
                SWARMX_MEMORY_REVIEW: "1",
              }
            : {}),
          ...(config ? { CODEX_CONFIG: JSON.stringify(config) } : {}),
        },
      };
    }
    if (id === "hermes" && process.env.SWARMX_HERMES_PYTHON)
      return {
        command: process.env.SWARMX_HERMES_PYTHON,
        args: ["-m", "acp_adapter"],
        env: process.env,
      };
    const args = ["acp"];
    if (id === "openclaw" && process.env.OPENCLAW_GATEWAY_URL)
      args.push("--url", process.env.OPENCLAW_GATEWAY_URL);
    return { command: id, args, env: process.env };
  };
  const opened = new Set<{ close(): Promise<void> }>();
  let disposed = false;
  const fresh = new Map<string, Awaited<ReturnType<typeof open>>>();
  const reservations = new Set<string>();
  const running = new Map<
    string,
    { peer?: Awaited<ReturnType<typeof open>>; cancelled: boolean }
  >();
  const checkPrompt = (text: string) => {
    if (options.reviewOnly && /^\s*\/[a-z][a-z0-9_-]*(?:\s|$)/iu.test(text))
      throw new Error("Memory reviews cannot execute upstream slash commands.");
  };

  async function open(catalogProbe = false) {
    if (disposed) throw new Error("ACP harness is disposed.");
    let observer: Observer | undefined;
    let configuration: Configuration = {};
    const token = randomUUID();
    const tools = new Map<string, { name: string; input: unknown }>();
    const client = acp
      .client({ name: "swarmx" })
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        if (params.update.sessionUpdate === "config_option_update")
          configuration = { ...configuration, configOptions: params.update.configOptions };
        if (params.update.sessionUpdate === "current_mode_update" && configuration.modes)
          configuration.modes.currentModeId = params.update.currentModeId;
        if (!observer) return;
        const update = params.update;
        const mode =
          update.sessionUpdate === "current_mode_update"
            ? update.currentModeId
            : update.sessionUpdate === "config_option_update"
              ? modelCatalog(configuration).current.mode
              : undefined;
        if (mode === undefined) observer.raw(params);
        else observer.raw(params, { "swarmx.native.mode": mode });
        switch (update.sessionUpdate) {
          case "agent_message_chunk":
          case "agent_thought_chunk":
          case "user_message_chunk":
            if (update.content.type === "text")
              observer.text(
                update.messageId ?? params.sessionId,
                update.content.text,
                update.sessionUpdate === "user_message_chunk"
                  ? "user"
                  : update.sessionUpdate === "agent_thought_chunk"
                    ? "reasoning"
                    : "assistant",
              );
            break;
          case "tool_call":
            tools.set(update.toolCallId, { name: update.title, input: update.rawInput });
            observer.tool(update.toolCallId, update.title, update.rawInput, update.rawOutput);
            break;
          case "tool_call_update": {
            const previous = tools.get(update.toolCallId);
            observer.tool(
              update.toolCallId,
              update.title ?? previous?.name ?? update.toolCallId,
              update.rawInput ?? previous?.input,
              update.rawOutput ??
                (update.status === "completed" || update.status === "failed"
                  ? update.content
                  : undefined),
            );
            break;
          }
        }
      })
      .onRequest(acp.methods.client.session.requestPermission, async ({ params, signal }) => {
        if (!observer || options.reviewOnly) return { outcome: { outcome: "cancelled" } };
        const permitted = params.options;
        const answers = Object.fromEntries(
          permitted.map((option) => [option.optionId, { optionId: option.optionId }]),
        );
        const answer = await observer.interact(
          {
            id: params.toolCall.toolCallId,
            title: params.toolCall.title ?? "Permission",
            schema: {
              type: "object",
              properties: {
                optionId: { type: "string", enum: permitted.map((option) => option.optionId) },
              },
              required: ["optionId"],
              additionalProperties: false,
            },
            permission: { toolCall: params.toolCall, options: permitted, answers },
          },
          signal,
        );
        if (signal.aborted || answer === undefined) return { outcome: { outcome: "cancelled" } };
        const { optionId } = z.object({ optionId: z.string() }).parse(answer);
        if (!Object.hasOwn(answers, optionId))
          throw acp.RequestError.invalidParams(undefined, "Unknown permission option.");
        return { outcome: { outcome: "selected", optionId } };
      })
      .onRequest(acp.methods.client.elicitation.create, async ({ params, signal }) => {
        if (!observer || params.mode !== "form" || options.reviewOnly) return { action: "cancel" };
        const answer = await observer.interact(
          {
            id: randomUUID(),
            title: params.message,
            schema: z.record(z.string(), z.unknown()).parse(params.requestedSchema),
          },
          signal,
        );
        if (signal.aborted || answer === undefined) return { action: "cancel" };
        return { action: "accept", content: z.record(z.string(), z.unknown()).parse(answer) };
      });
    const { command, args, env } = launch();
    const process = await connectAcpProcess(client, command, args, options.cwd, env);
    if (disposed) {
      await process.close();
      throw new Error("ACP harness is disposed.");
    }
    const endpoint = options.reviewOnly || catalogProbe ? undefined : options.registerMcp?.(token);
    const owned = {
      async close() {
        endpoint?.dispose();
        opened.delete(owned);
        await process.close();
      },
    };
    opened.add(owned);
    const agent = process.connection.agent;
    try {
      const initialized = await agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: { elicitation: { form: {} } },
      });
      if (initialized.protocolVersion !== acp.PROTOCOL_VERSION)
        throw new Error("Unsupported ACP version.");
      const policy = options.executionPolicy?.();
      const mcpUrl = new URL(options.mcp.url);
      if (endpoint) mcpUrl.searchParams.set("acp", token);
      const params: acp.NewSessionRequest = {
        cwd: options.cwd,
        ...(id === "codex" && (options.reviewOnly || catalogProbe)
          ? { _meta: { ephemeral: true } }
          : {}),
        mcpServers:
          options.reviewOnly || catalogProbe || id === "openclaw"
            ? []
            : [
                {
                  type: "http",
                  name: "swarmx",
                  url: mcpUrl.href,
                  headers: Object.entries(
                    endpoint
                      ? { ...options.mcp.headers, authorization: `Bearer ${token}` }
                      : options.mcp.headers,
                  ).map(([name, value]) => ({
                    name,
                    value,
                  })),
                },
              ],
        ...(id === "claude" && options.reviewOnly
          ? {
              _meta: {
                claudeCode: {
                  options: {
                    ...(catalogProbe ? { persistSession: false } : {}),
                    strictMcpConfig: true,
                    settings: {
                      disableAllHooks: true,
                      env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
                      ...(policy?.harnesses?.claude
                        ? { availableModels: policy.harnesses.claude }
                        : {}),
                    },
                    disallowedTools: ["Agent", "Task", "ExitPlanMode", "EnterWorktree"],
                    tools: [],
                    mcpServers: {},
                    persistSession: false,
                    permissionMode: "dontAsk",
                    sandbox: {
                      enabled: true,
                      failIfUnavailable: true,
                      autoAllowBashIfSandboxed: false,
                      allowUnsandboxedCommands: false,
                      network: { allowedDomains: [], strictAllowlist: true },
                      filesystem: { denyWrite: ["/"] },
                    },
                  },
                },
              },
            }
          : {}),
      };
      const peer = {
        agent,
        endpoint,
        initialized,
        token,
        params,
        get configuration() {
          return configuration;
        },
        observe(value?: Observer) {
          observer = value;
        },
        async session(sessionId?: string, replay = false, reserved = false) {
          const createParams = reserved
            ? {
                ...params,
                _meta: {
                  ...params._meta,
                  claudeCode: {
                    ...z.record(z.string(), z.unknown()).parse(params._meta?.claudeCode ?? {}),
                    options: {
                      ...z
                        .record(z.string(), z.unknown())
                        .parse(
                          (params._meta?.claudeCode as { options?: unknown } | undefined)
                            ?.options ?? {},
                        ),
                      sessionId,
                    },
                  },
                },
              }
            : params;
          const result =
            sessionId === undefined || reserved
              ? await agent.request(acp.methods.agent.session.new, createParams)
              : await agent.request(
                  replay ? acp.methods.agent.session.load : acp.methods.agent.session.resume,
                  { ...params, sessionId },
                );
          configuration = result;
          if (reserved && (result as acp.NewSessionResponse).sessionId !== sessionId)
            throw new Error("ACP adapter did not retain the reserved session identity.");
          return sessionId ?? (result as acp.NewSessionResponse).sessionId;
        },
        close: owned.close,
      };
      return peer;
    } catch (error) {
      await owned.close();
      throw error;
    }
  }

  return {
    name: id,
    capabilities: HARNESS_CAPABILITIES[id],
    ...(id === "claude"
      ? {
          restoreEmptySessions(ids: readonly string[]) {
            for (const session of ids) reservations.add(session);
          },
        }
      : {}),
    async list() {
      const peer = await open();
      try {
        if (!peer.initialized.agentCapabilities?.sessionCapabilities?.list)
          throw new Error(`${id} ACP server does not support session/list.`);
        const sessions: Session[] = [];
        let cursor: string | undefined;
        do {
          const result = await peer.agent.request(acp.methods.agent.session.list, {
            cwd: options.cwd,
            ...(cursor ? { cursor } : {}),
          });
          sessions.push(
            ...result.sessions
              .filter((session) => session.cwd === options.cwd)
              .map(({ sessionId, title, updatedAt }) => ({
                sessionId,
                ...(title == null ? {} : { title }),
                ...(updatedAt == null ? {} : { updatedAt }),
              })),
          );
          cursor = result.nextCursor ?? undefined;
        } while (cursor);
        for (const session of sessions) reservations.delete(session.sessionId);
        return [
          ...sessions,
          ...[...fresh.keys(), ...reservations]
            .filter((sessionId) => !sessions.some((session) => session.sessionId === sessionId))
            .map((sessionId) => ({ sessionId })),
        ];
      } finally {
        await peer.close();
      }
    },
    async models(sessionId) {
      const existing =
        sessionId === undefined
          ? undefined
          : (fresh.get(sessionId) ?? running.get(sessionId)?.peer);
      if (existing) return modelCatalog(existing.configuration);
      const peer = await open(sessionId === undefined || reservations.has(sessionId));
      try {
        await peer.session(reservations.has(sessionId ?? "") ? undefined : sessionId);
        return modelCatalog(peer.configuration);
      } finally {
        await peer.close();
      }
    },
    async create() {
      if (id === "claude") {
        const session = randomUUID();
        reservations.add(session);
        return session;
      }
      const peer = await open();
      try {
        const sessionId = await peer.session();
        fresh.set(sessionId, peer);
        return sessionId;
      } catch (error) {
        await peer.close();
        throw error;
      }
    },
    async read(sessionId, observer) {
      if (fresh.has(sessionId) || reservations.has(sessionId)) return;
      const peer = await open();
      try {
        peer.observe(observer);
        await peer.session(sessionId, true);
      } finally {
        await peer.close();
      }
    },
    async start(sessionId, text, observer, selection) {
      checkPrompt(text);
      if (running.has(sessionId)) throw new Error("Session is busy.");
      const active: { peer?: Awaited<ReturnType<typeof open>>; cancelled: boolean } = {
        cancelled: false,
      };
      running.set(sessionId, active);
      try {
        const peer = fresh.get(sessionId) ?? (await open());
        active.peer = peer;
        const wasFresh = fresh.delete(sessionId);
        if (peer.endpoint && observer.executionId)
          peer.endpoint.bind(`${id}:${sessionId}`, observer.executionId);
        else if (options.registerMcp && !options.reviewOnly)
          throw new Error("ACP tool access requires a bound Host execution.");
        if (!wasFresh) await peer.session(sessionId, false, reservations.has(sessionId));
        reservations.delete(sessionId);
        peer.observe(observer);
        observer.raw(peer.initialized, {
          "rpc.method": "initialize",
          "swarmx.acp.agent.version": peer.initialized.agentInfo?.version,
        });
        if (id === "claude" && options.reviewOnly)
          await peer.agent.request(acp.methods.agent.session.setMode, {
            sessionId,
            modeId: "dontAsk",
          });
        if (options.reviewOnly && selection?.mode !== undefined)
          throw new Error("Memory reviews cannot select an ordinary task mode.");
        if (selection?.mode !== undefined) {
          const mode = peer.configuration.configOptions?.find(
            (option) => option.category === "mode",
          );
          if (mode) {
            if (!choices(mode).some((choice) => choice.value === selection.mode))
              throw new Error(`Unsupported ACP mode: ${selection.mode}`);
            const response = await peer.agent.request(acp.methods.agent.session.setConfigOption, {
              sessionId,
              configId: mode.id,
              value: selection.mode,
            });
            peer.configuration.configOptions = response.configOptions;
          } else {
            if (
              !peer.configuration.modes?.availableModes.some((mode) => mode.id === selection.mode)
            )
              throw new Error(`Unsupported ACP mode: ${selection.mode}`);
            await peer.agent.request(acp.methods.agent.session.setMode, {
              sessionId,
              modeId: selection.mode,
            });
            peer.configuration.modes.currentModeId = selection.mode;
          }
        }
        for (const [category, value] of [
          ["model", selection?.model],
          ["thought_level", selection?.effort],
        ] as const) {
          if (value === undefined) continue;
          const config = peer.configuration.configOptions?.find(
            (option) => option.category === category,
          );
          if (!config || !choices(config).some((option) => option.value === value))
            throw new Error(`Unsupported ACP ${category}: ${value}`);
          const result = await peer.agent.request(acp.methods.agent.session.setConfigOption, {
            sessionId,
            configId: config.id,
            value,
          });
          peer.configuration.configOptions = result.configOptions;
        }
        if (active.cancelled) return { stopReason: "cancelled" };
        return await peer.agent.request(acp.methods.agent.session.prompt, {
          sessionId,
          prompt: [
            {
              type: "text",
              text:
                text + (selection?.instructions ? memoryContextSuffix(selection.instructions) : ""),
            },
          ],
        });
      } finally {
        if (active.peer) await active.peer.close();
        running.delete(sessionId);
      }
    },
    async steer(sessionId, text) {
      checkPrompt(text);
      const peer = running.get(sessionId)?.peer;
      if (!peer) throw new Error("No running ACP session.");
      const supported = z
        .object({ steering: z.object({ supported: z.boolean() }).optional() })
        .parse(peer.initialized._meta ?? {});
      if (!supported.steering?.supported)
        throw new Error(`${id} ACP server does not support steering.`);
      await peer.agent.request("_session/steering", {
        sessionId,
        prompt: [{ type: "text", text }],
      });
    },
    async interrupt(sessionId) {
      const active = running.get(sessionId);
      if (!active) return;
      active.cancelled = true;
      await active.peer?.agent.notify(acp.methods.agent.session.cancel, { sessionId });
    },
    async dispose() {
      disposed = true;
      await Promise.all([...opened].map((peer) => peer.close()));
      fresh.clear();
      reservations.clear();
    },
  };
}

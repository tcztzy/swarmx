import { randomUUID } from "node:crypto";
import type { ModelCatalog, RunResult, Session } from "@swarmx/swarm";
import { z } from "zod";
import manifest from "../../package.json" with { type: "json" };
import { requestApproval } from "./approval.js";
import type { ClientRequest } from "./generated/ClientRequest.js";
import type { ServerNotification } from "./generated/ServerNotification.js";
import type { ServerRequest } from "./generated/ServerRequest.js";
import type { JsonValue } from "./generated/serde_json/JsonValue.js";
import type { ConfigReadResponse } from "./generated/v2/ConfigReadResponse.js";
import type { ModelListResponse } from "./generated/v2/ModelListResponse.js";
import type { PermissionProfileListResponse } from "./generated/v2/PermissionProfileListResponse.js";
import type { PermissionsRequestApprovalResponse } from "./generated/v2/PermissionsRequestApprovalResponse.js";
import type { ThreadItem } from "./generated/v2/ThreadItem.js";
import type { ThreadItemsListResponse } from "./generated/v2/ThreadItemsListResponse.js";
import type { ThreadListResponse } from "./generated/v2/ThreadListResponse.js";
import type { ThreadReadResponse } from "./generated/v2/ThreadReadResponse.js";
import type { ThreadResumeResponse } from "./generated/v2/ThreadResumeResponse.js";
import type { ThreadStartResponse } from "./generated/v2/ThreadStartResponse.js";
import type { ThreadTurnsListResponse } from "./generated/v2/ThreadTurnsListResponse.js";
import type { TokenUsageBreakdown } from "./generated/v2/TokenUsageBreakdown.js";
import type { Turn } from "./generated/v2/Turn.js";
import type { TurnStartResponse } from "./generated/v2/TurnStartResponse.js";
import { rpcProcess } from "./rpc-process.js";
import {
  type AgentOptions,
  type EventAttributes,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

type Responses = {
  initialize: { userAgent: string };
  "config/read": ConfigReadResponse;
  "model/list": ModelListResponse;
  "permissionProfile/list": PermissionProfileListResponse;
  "thread/list": ThreadListResponse;
  "thread/read": ThreadReadResponse;
  "thread/start": ThreadStartResponse;
  "thread/resume": ThreadResumeResponse;
  "thread/turns/list": ThreadTurnsListResponse;
  "thread/items/list": ThreadItemsListResponse;
  "turn/start": TurnStartResponse;
  "turn/steer": unknown;
  "turn/interrupt": unknown;
};
const tokenUsageSchema = z
  .object({
    inputTokens: z.number().int().nonnegative(),
    cachedInputTokens: z.number().int().nonnegative(),
    cacheWriteInputTokens: z.number().int().nonnegative(),
    outputTokens: z.number().int().nonnegative(),
    reasoningOutputTokens: z.number().int().nonnegative(),
    totalTokens: z.number().int().nonnegative(),
  })
  .refine(
    (usage) =>
      usage.cachedInputTokens + usage.cacheWriteInputTokens <= usage.inputTokens &&
      usage.reasoningOutputTokens <= usage.outputTokens &&
      usage.totalTokens === usage.inputTokens + usage.outputTokens,
  );
const responseUsageSchema = z.object({
  responseId: z.string().min(1),
  usage: tokenUsageSchema.nullable(),
});

// These public mode IDs already occur in persisted conversations. Keep their original policies.
const approvalPresets = {
  "read-only": {
    name: "Ask for approval",
    approvalPolicy: "on-request",
    approvalsReviewer: "user",
    sandbox: "workspace-write",
  },
  agent: {
    name: "Approve for me",
    approvalPolicy: "on-request",
    approvalsReviewer: "auto_review",
    sandbox: "workspace-write",
  },
  "agent-full-access": {
    name: "Full access",
    approvalPolicy: "never",
    approvalsReviewer: "user",
    sandbox: "danger-full-access",
  },
} as const;
function permissionOptions(mode?: string) {
  if (mode === undefined) return {};
  if (Object.hasOwn(approvalPresets, mode)) {
    const {
      name: _name,
      sandbox,
      ...approval
    } = approvalPresets[mode as keyof typeof approvalPresets];
    return {
      ...approval,
      permissions: sandbox === "workspace-write" ? ":workspace" : ":danger-full-access",
    };
  }
  return { permissions: mode };
}

export async function createCodex(options: AgentOptions): Promise<NativeAgent> {
  const config = z.record(z.string(), z.json()).parse(JSON.parse(process.env.CODEX_CONFIG ?? "{}"));
  type Peer = Awaited<ReturnType<typeof open>>;
  type Active = {
    cancelled: boolean;
    model: string | undefined;
    peer?: Peer;
    finished: ReturnType<typeof Promise.withResolvers<void>>;
  };
  const opened = new Set<Peer>();
  const fresh = new Map<string, Peer>();
  const running = new Map<string, Active>();
  let disposed = false;

  async function open(execution = false) {
    if (disposed) throw new Error("Codex integration is disposed.");
    let observer: Observer | undefined;
    let threadId: string | undefined;
    let nativeTurn: Turn | undefined;
    const responseUsage = new Map<string, TokenUsageBreakdown | null>();
    let invalidUsage = false;
    const streamed = new Set<string>();
    const interactionSignal = new AbortController();
    const done = Promise.withResolvers<RunResult>();
    // A process may fail during setup, before a caller starts awaiting its turn.
    const settled = done.promise.then(
      (result) => ({ result }),
      (error) => ({ error }),
    );
    const ready = Promise.withResolvers<string | undefined>();
    const rpc = rpcProcess(
      process.env.CODEX_PATH ?? "codex",
      ["app-server"],
      options.cwd,
      async (raw) => {
        const message = raw as ServerNotification | ServerRequest;
        if (message.method === "currentTime/read")
          return { currentTimeAt: Math.floor(Date.now() / 1000) };
        if (!observer) {
          if (raw.id !== undefined) throw new Error(`No active Codex turn for ${message.method}.`);
          return;
        }
        let attributes: EventAttributes | undefined;
        if (
          message.method === "rawResponse/completed" &&
          message.params.threadId === threadId &&
          message.params.turnId === nativeTurn?.id
        ) {
          const parsed = responseUsageSchema.safeParse(message.params);
          if (!parsed.success) invalidUsage = true;
          else {
            const previous = responseUsage.get(parsed.data.responseId);
            if (
              previous &&
              parsed.data.usage &&
              JSON.stringify(previous) !== JSON.stringify(parsed.data.usage)
            )
              invalidUsage = true;
            responseUsage.set(parsed.data.responseId, parsed.data.usage);
          }
          const reports = [...responseUsage.values()];
          const known =
            !invalidUsage && reports.length > 0 && reports.every((usage) => usage !== null);
          const sum = (key: keyof TokenUsageBreakdown) =>
            known ? reports.reduce((total, usage) => total + (usage?.[key] ?? 0), 0) : null;
          attributes = {
            "swarmx.native.run_id": message.params.turnId,
            "gen_ai.usage.input_tokens": sum("inputTokens"),
            "gen_ai.usage.output_tokens": sum("outputTokens"),
            "swarmx.usage.cached_input_tokens": known
              ? (sum("cachedInputTokens") ?? 0) + (sum("cacheWriteInputTokens") ?? 0)
              : null,
            "swarmx.usage.reasoning_output_tokens": sum("reasoningOutputTokens"),
            "swarmx.usage.cost_usd": null,
            "swarmx.usage.cost_source": "unknown",
            "swarmx.usage.coverage": known ? "partial" : "unknown",
            "swarmx.usage.scope": "native-responses",
            "swarmx.usage.basis":
              "Codex exact response usage by response ID within native turn; input includes cache; output includes reasoning; excludes unreported calls; no USD report",
          };
        }
        observer.raw(raw, attributes);
        if (
          message.params &&
          "threadId" in message.params &&
          message.params.threadId !== threadId
        ) {
          if (raw.id !== undefined) throw new Error("Codex request belongs to another thread.");
          return;
        }
        if (
          message.method === "model/rerouted" &&
          message.params.turnId === nativeTurn?.id &&
          threadId &&
          running.get(threadId)?.model &&
          message.params.fromModel !== message.params.toModel
        )
          throw new Error(
            `Codex model changed from "${message.params.fromModel}" to "${message.params.toModel}".`,
          );
        if ("id" in message) {
          if (options.reviewOnly)
            throw new Error("Memory review cannot request tools or permissions.");
          return interaction(message, observer, interactionSignal.signal);
        }
        switch (message.method) {
          case "turn/started":
            nativeTurn = message.params.turn;
            ready.resolve(nativeTurn.id);
            timing(nativeTurn, observer);
            break;
          case "item/agentMessage/delta":
          case "item/reasoning/summaryTextDelta":
          case "item/reasoning/textDelta":
            streamed.add(message.params.itemId);
            observer.text(
              message.params.itemId,
              message.params.delta,
              message.method === "item/agentMessage/delta" ? "assistant" : "reasoning",
            );
            break;
          case "item/started":
          case "item/completed": {
            const { item, turnId } = message.params;
            projectItem(
              item,
              observer,
              { turnId, startedAt: nativeTurn?.startedAt },
              message.method === "item/completed",
              streamed.has(item.id),
            );
            break;
          }
          case "turn/completed": {
            nativeTurn = message.params.turn;
            timing(nativeTurn, observer);
            interactionSignal.abort();
            if (nativeTurn.status === "completed") done.resolve({ stopReason: "end_turn" });
            else if (nativeTurn.status === "interrupted") done.resolve({ stopReason: "cancelled" });
            else
              done.reject(
                new Error(
                  nativeTurn.error?.message ??
                    `Invalid terminal Codex status: ${nativeTurn.status}`,
                ),
              );
            break;
          }
          case "error":
            if (!message.params.willRetry) done.reject(new Error(message.params.error.message));
            break;
        }
      },
      (error) => {
        interactionSignal.abort(error);
        done.reject(error);
        void peer.close();
      },
    );
    function request<M extends keyof Responses>(
      method: M,
      params: Extract<ClientRequest, { method: M }>["params"],
    ): Promise<Responses[M]> {
      return rpc.request(method, params ?? {}) as Promise<Responses[M]>;
    }
    const token = randomUUID();
    let endpoint: ReturnType<NonNullable<AgentOptions["registerMcp"]>> | undefined;
    const peer = {
      request,
      ready,
      settled,
      observe(id: string, value: Observer) {
        threadId = id;
        nativeTurn = undefined;
        responseUsage.clear();
        invalidUsage = false;
        observer = value;
      },
      async settings(instructions?: string) {
        const { config: nativeConfig } = await request("config/read", { cwd: options.cwd });
        const overrides: Record<string, JsonValue> = { ...config };
        if (execution && !options.reviewOnly) {
          overrides["mcp_servers.swarmx"] = {
            command: options.mcp.command,
            args: [...options.mcp.args],
            env: endpoint ? { ...options.mcp.env, SWARMX_MCP_TOKEN: token } : options.mcp.env,
          };
        }
        if (options.reviewOnly) {
          const servers = z.record(z.string(), z.unknown()).parse(nativeConfig.mcp_servers ?? {});
          for (const name of Object.keys(servers)) overrides[`mcp_servers.${name}.enabled`] = false;
          overrides.web_search = "disabled";
          for (const feature of [
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
          ])
            overrides[`features.${feature}`] = false;
          overrides["tools.update_plan.enabled"] = false;
          overrides["tools.experimental_request_user_input.enabled"] = false;
        }
        return {
          cwd: options.cwd,
          config: overrides,
          ...(instructions === undefined
            ? {}
            : {
                developerInstructions: `${
                  z
                    .string()
                    .nullish()
                    .parse(config.developer_instructions ?? nativeConfig.developer_instructions) ??
                  ""
                }${memoryContextSuffix(instructions)}`,
              }),
          ...(options.reviewOnly
            ? { sandbox: "read-only" as const, approvalPolicy: "never" as const }
            : {}),
        };
      },
      bind(id: string, value: Observer) {
        if (endpoint) {
          if (!value.executionId)
            throw new Error("Codex tool access requires a bound Host execution.");
          endpoint.bind(`codex:${id}`, value.executionId);
        }
      },
      async close() {
        if (!opened.delete(peer)) return;
        for (const [id, runtime] of fresh) if (runtime === peer) fresh.delete(id);
        endpoint?.dispose();
        interactionSignal.abort();
        ready.resolve(undefined);
        done.reject(new Error("Codex runtime closed."));
        await rpc.dispose();
      },
    };
    opened.add(peer);
    try {
      if (execution && !options.reviewOnly) endpoint = options.registerMcp?.(token);
      await request("initialize", {
        clientInfo: { name: "swarmx", title: "SwarmX", version: manifest.version },
        capabilities: { experimentalApi: true, requestAttestation: false },
      });
      rpc.notify("initialized", {});
      if (disposed) throw new Error("Codex integration is disposed.");
      return peer;
    } catch (error) {
      await peer.close();
      throw error;
    }
  }

  return {
    name: "Codex",
    capabilities: HARNESS_CAPABILITIES.codex,
    async models(id) {
      const peer = await open();
      try {
        const models: ModelCatalog["models"] = [];
        let cursor: string | null = null;
        do {
          const page: ModelListResponse = await peer.request("model/list", { cursor });
          models.push(
            ...page.data.map((model) => ({
              id: model.model,
              name: model.displayName,
              description: model.description,
              efforts: model.supportedReasoningEfforts.map((effort) => ({
                id: effort.reasoningEffort,
                name: effort.description,
              })),
              defaultEffort: model.defaultReasoningEffort,
            })),
          );
          cursor = page.nextCursor;
        } while (cursor);
        const modes: NonNullable<ModelCatalog["modes"]> = [];
        do {
          const page: PermissionProfileListResponse = await peer.request("permissionProfile/list", {
            cursor,
          });
          modes.push(
            ...page.data
              .filter((profile) => profile.allowed)
              .map((profile) => ({
                id: profile.id,
                name: profile.id,
                ...(profile.description === null ? {} : { description: profile.description }),
              })),
          );
          cursor = page.nextCursor;
        } while (cursor);
        for (const [id, preset] of Object.entries(approvalPresets)) {
          const profileId =
            preset.sandbox === "workspace-write" ? ":workspace" : ":danger-full-access";
          if (modes.some((mode) => mode.id === profileId)) modes.push({ id, name: preset.name });
        }
        if (id) {
          const { thread } = await (fresh.get(id) ?? peer).request("thread/read", {
            threadId: id,
            includeTurns: false,
          });
          if (thread.cwd !== options.cwd)
            throw new Error("Codex session belongs to another directory.");
          return {
            models,
            modes,
            current: {
              ...(thread.model === null ? {} : { model: thread.model }),
              ...(thread.reasoningEffort === null ? {} : { effort: thread.reasoningEffort }),
            },
          };
        }
        const { config: nativeConfig } = await peer.request("config/read", { cwd: options.cwd });
        const current = { ...nativeConfig, ...config };
        return {
          models,
          modes,
          current: {
            ...(typeof current.model === "string" ? { model: current.model } : {}),
            ...(typeof current.model_reasoning_effort === "string"
              ? { effort: current.model_reasoning_effort }
              : {}),
            ...(typeof current.default_permissions === "string"
              ? { mode: current.default_permissions }
              : {}),
          },
        };
      } finally {
        await peer.close();
      }
    },
    async list() {
      const peer = await open();
      try {
        const sessions: Session[] = [];
        let cursor: string | null = null;
        do {
          const page: ThreadListResponse = await peer.request("thread/list", {
            cwd: options.cwd,
            cursor,
          });
          sessions.push(
            ...page.data.map((thread) => ({
              sessionId: thread.id,
              title: thread.name ?? thread.preview,
              updatedAt: new Date(thread.updatedAt * 1000).toISOString(),
            })),
          );
          cursor = page.nextCursor;
        } while (cursor);
        return sessions;
      } finally {
        await peer.close();
      }
    },
    async create(context) {
      const peer = await open(true);
      try {
        const { thread } = await peer.request("thread/start", {
          ...(await peer.settings(context?.instructions)),
          historyMode: "legacy",
          ...(options.reviewOnly ? { ephemeral: true } : {}),
        });
        fresh.set(thread.id, peer);
        return thread.id;
      } catch (error) {
        await peer.close();
        throw error;
      }
    },
    async read(id, observer) {
      if (fresh.has(id)) return;
      const peer = await open();
      try {
        const { thread } = await peer.request("thread/read", { threadId: id, includeTurns: false });
        if (thread.cwd !== options.cwd)
          throw new Error("Codex session belongs to another directory.");
        observer.raw(thread, { "swarmx.harness.version": thread.cliVersion });
        if (thread.historyMode === "paginated") {
          const turns = new Map<string, Turn>();
          let cursor: string | null = null;
          do {
            const page: ThreadTurnsListResponse = await peer.request("thread/turns/list", {
              threadId: id,
              cursor,
              sortDirection: "asc",
              itemsView: "notLoaded",
            });
            for (const turn of page.data) turns.set(turn.id, turn);
            cursor = page.nextCursor;
          } while (cursor);
          do {
            const page: ThreadItemsListResponse = await peer.request("thread/items/list", {
              threadId: id,
              cursor,
              sortDirection: "asc",
            });
            for (const { item, turnId } of page.data) {
              const turn = turns.get(turnId);
              projectItem(item, observer, {
                turnId,
                startedAt: turn?.startedAt,
                durationMs: turn?.durationMs,
              });
            }
            cursor = page.nextCursor;
          } while (cursor);
        } else {
          const { thread: history } = await peer.request("thread/read", {
            threadId: id,
            includeTurns: true,
          });
          for (const turn of history.turns)
            for (const item of turn.items)
              projectItem(item, observer, {
                turnId: turn.id,
                startedAt: turn.startedAt,
                durationMs: turn.durationMs,
              });
        }
      } finally {
        await peer.close();
      }
    },
    async start(id, text, observer, selection) {
      if (options.reviewOnly && /^\s*\/[a-z][a-z0-9_-]*(?:\s|$)/iu.test(text))
        throw new Error("Memory reviews cannot execute native slash commands.");
      if (running.has(id)) throw new Error("Codex session is busy.");
      const active: Active = {
        cancelled: false,
        model: selection?.model,
        finished: Promise.withResolvers<void>(),
      };
      running.set(id, active);
      const wasFresh = fresh.has(id);
      let rollout = !wasFresh;
      let peer: Peer | undefined;
      try {
        peer = fresh.get(id) ?? (await open(true));
        active.peer = peer;
        peer.bind(id, observer);
        if (!wasFresh) {
          const settings = await peer.settings(selection?.instructions);
          if (active.cancelled) return { stopReason: "cancelled" };
          const { thread, activePermissionProfile } = await peer.request("thread/resume", {
            ...settings,
            threadId: id,
            excludeTurns: true,
            ...(!options.reviewOnly ? permissionOptions(selection?.mode) : {}),
          });
          if (thread.cwd !== options.cwd)
            throw new Error("Codex session belongs to another directory.");
          observer.raw(thread, {
            "swarmx.harness.version": thread.cliVersion,
            "swarmx.native.mode": selection?.mode ?? activePermissionProfile?.id,
          });
        }
        if (active.cancelled) return { stopReason: "cancelled" };
        peer.observe(id, observer);
        const started = await peer.request("turn/start", {
          threadId: id,
          input: [{ type: "text", text, text_elements: [] }],
          ...(selection?.model === undefined ? {} : { model: selection.model }),
          ...(selection?.effort === undefined ? {} : { effort: selection.effort }),
          ...(!options.reviewOnly ? permissionOptions(selection?.mode) : {}),
        });
        rollout = true;
        fresh.delete(id);
        peer.ready.resolve(started.turn.id);
        const outcome = await peer.settled;
        if ("error" in outcome) throw outcome.error;
        return outcome.result;
      } finally {
        try {
          // An empty thread has no resumable rollout; keep its runtime for a retry.
          if (wasFresh && !rollout && !disposed && peer && opened.has(peer)) fresh.set(id, peer);
          else await peer?.close();
        } finally {
          running.delete(id);
          active.finished.resolve();
        }
      }
    },
    async steer(id, text) {
      const active = running.get(id);
      if (!active || active.cancelled) throw new Error("No running Codex turn.");
      const turnId = await Promise.race([active.peer?.ready.promise, active.finished.promise]);
      if (!turnId || active.cancelled || running.get(id) !== active)
        throw new Error("No running Codex turn.");
      await active.peer?.request("turn/steer", {
        threadId: id,
        expectedTurnId: turnId,
        input: [{ type: "text", text, text_elements: [] }],
      });
    },
    async interrupt(id) {
      const active = running.get(id);
      if (!active) return;
      active.cancelled = true;
      const turnId = await Promise.race([active.peer?.ready.promise, active.finished.promise]);
      if (turnId && running.get(id) === active)
        await active.peer?.request("turn/interrupt", { threadId: id, turnId });
    },
    async dispose() {
      disposed = true;
      fresh.clear();
      await Promise.all([...opened].map((peer) => peer.close()));
    },
  };
}

function timing(turn: Turn, observer: Observer) {
  observer.activity?.({
    type: "message",
    turnId: turn.id,
    startedAt: turn.startedAt,
    durationMs: turn.durationMs,
  });
}

function projectItem(
  item: ThreadItem,
  observer: Observer,
  turn: {
    turnId: string;
    startedAt?: number | null | undefined;
    durationMs?: number | null | undefined;
  },
  complete = true,
  streamed = false,
) {
  observer.raw(item);
  if (item.type === "agentMessage") {
    observer.activity?.({
      type: "message",
      messageId: item.id,
      ...turn,
      ...(item.phase ? { phase: item.phase } : {}),
    });
    if (complete && !streamed) observer.text(item.id, item.text, "assistant");
  } else if (item.type === "userMessage") {
    if (complete)
      observer.text(
        item.id,
        item.content
          .filter((part) => part.type === "text")
          .map((part) => part.text)
          .join(""),
        "user",
      );
  } else if (item.type === "reasoning") {
    if (complete && !streamed)
      observer.text(item.id, [...item.summary, ...item.content].join("\n"), "reasoning");
  } else if (item.type === "plan") {
    if (complete) observer.text(item.id, item.text, "reasoning");
  } else {
    const kind =
      item.type === "commandExecution"
        ? "execute"
        : item.type === "fileChange"
          ? "edit"
          : item.type === "webSearch"
            ? "search"
            : "other";
    const failed = "status" in item && (item.status === "failed" || item.status === "declined");
    observer.activity?.({
      type: "tool",
      toolCallId: item.id,
      kind,
      status: complete ? (failed ? "failed" : "completed") : "in_progress",
    });
    observer.tool(
      item.id,
      "tool" in item ? item.tool : item.type,
      "arguments" in item ? item.arguments : item,
      complete ? item : undefined,
    );
  }
}

async function interaction(
  message: ServerRequest,
  observer: Observer,
  signal: AbortSignal,
): Promise<unknown> {
  const id = String(message.id);
  switch (message.method) {
    case "item/commandExecution/requestApproval":
    case "item/fileChange/requestApproval": {
      const decisions =
        message.method === "item/commandExecution/requestApproval"
          ? (message.params.availableDecisions ??
            (["accept", "acceptForSession", "decline", "cancel"] as const))
          : (["accept", "acceptForSession", "decline", "cancel"] as const);
      const decision = await requestApproval(
        observer,
        id,
        message.params.reason ?? "Approve Codex action?",
        message.params.itemId,
        decisions.map((answer, index) => ({
          id: String(index),
          label: typeof answer === "string" ? answer : JSON.stringify(answer),
          kind:
            answer === "accept"
              ? "allow_once"
              : answer === "decline" || answer === "cancel"
                ? "reject_once"
                : "allow_always",
          answer,
        })),
        signal,
      );
      return { decision: decision ?? "cancel" };
    }
    case "item/permissions/requestApproval": {
      const { network, fileSystem } = message.params.permissions;
      const permissions = {
        ...(network ? { network } : {}),
        ...(fileSystem ? { fileSystem } : {}),
      };
      const denied: PermissionsRequestApprovalResponse = { permissions: {}, scope: "turn" };
      return (
        (await requestApproval<PermissionsRequestApprovalResponse>(
          observer,
          id,
          message.params.reason ?? "Approve additional permissions?",
          message.params.itemId,
          [
            {
              id: "turn",
              label: `Allow this turn: ${JSON.stringify(permissions)}`,
              kind: "allow_once",
              answer: { permissions, scope: "turn" },
            },
            {
              id: "session",
              label: `Allow this session: ${JSON.stringify(permissions)}`,
              kind: "allow_always",
              answer: { permissions, scope: "session" },
            },
            { id: "deny", label: "Deny", kind: "reject_once", answer: denied },
          ],
          signal,
        )) ?? denied
      );
    }
    case "item/tool/requestUserInput": {
      const answer = await observer.interact(
        {
          id,
          title: "Codex needs input",
          sensitive: message.params.questions.some((question) => question.isSecret),
          schema: {
            type: "object",
            properties: Object.fromEntries(
              message.params.questions.map((question) => [
                question.id,
                {
                  type: "string",
                  title: question.question,
                  ...(question.options
                    ? { examples: question.options.map((option) => option.label) }
                    : {}),
                  ...(question.isSecret ? { format: "password" } : {}),
                },
              ]),
            ),
            required: message.params.questions.map((question) => question.id),
            additionalProperties: false,
          },
        },
        signal,
      );
      const values =
        answer === undefined || signal.aborted
          ? {}
          : z.record(z.string(), z.string()).parse(answer);
      if (
        Object.keys(values).some(
          (key) => !message.params.questions.some((question) => question.id === key),
        )
      )
        throw new Error("Unknown Codex question.");
      return {
        answers: Object.fromEntries(
          Object.entries(values).map(([key, value]) => [key, { answers: [value] }]),
        ),
      };
    }
    case "mcpServer/elicitation/request": {
      if (message.params.mode !== "form")
        throw new Error("SwarmX supports form elicitations only.");
      const answer = await observer.interact(
        {
          id,
          title: message.params.message,
          schema: message.params.requestedSchema as Record<string, unknown>,
        },
        signal,
      );
      return {
        action: answer === undefined || signal.aborted ? "cancel" : "accept",
        content: signal.aborted ? null : (answer ?? null),
        _meta: null,
      };
    }
    case "currentTime/read":
      return { currentTimeAt: Math.floor(Date.now() / 1000) };
    default:
      throw new Error(`Unsupported native Codex request: ${message.method}`);
  }
}

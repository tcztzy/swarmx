import { realpath } from "node:fs/promises";
import * as acp from "@agentclientprotocol/sdk";
import type { RunOptions } from "@swarmx/swarm";
import { z } from "zod";
import type { NativeAgent, Observer } from "../agents/types.js";
import { SwarmxCapability, SwarmxRequest } from "./acp-extension.js";
import { publicCapabilities } from "./capabilities.js";

/** Leaf adapter from ACP to the Host-protected native SDK boundary. */
export function acpAgent(native: NativeAgent, cwd: string): acp.AgentApp {
  const supportsPermissions = native.permissions !== undefined;
  async function bind(directory: string): Promise<NativeAgent> {
    if ((await realpath(directory)) !== (await realpath(cwd)))
      throw acp.RequestError.invalidParams(
        undefined,
        "The requested directory does not match the Host working directory.",
      );
    return native;
  }
  let state: { forms: boolean; permissions: NativeAgent["permissions"] } | undefined;
  let connected = false;
  const selections = new Map<string, RunOptions>();
  const bound = new Map<string, NativeAgent>();
  const active = new Map<string, NativeAgent>();
  function agentFor(sessionId: string): NativeAgent {
    const agent = bound.get(sessionId);
    if (!agent)
      throw acp.RequestError.invalidParams(
        undefined,
        "Create, load or resume the ACP session before continuing it.",
      );
    return agent;
  }
  async function permissionsFor(
    sessionId: string,
  ): Promise<Awaited<ReturnType<NonNullable<NativeAgent["permissions"]>>>> {
    const check = agentFor(sessionId).permissions;
    if (!check)
      throw acp.RequestError.invalidParams(
        undefined,
        "This Agent cannot enforce the SwarmX permission extension.",
      );
    return check(sessionId);
  }
  async function config(sessionId: string): Promise<acp.SessionConfigOption[]> {
    const agent = agentFor(sessionId);
    const catalog = await call(agent.models(sessionId));
    const selection = selections.get(sessionId) ?? catalog.current;
    const options: acp.SessionConfigOption[] = [];
    if (catalog.modes?.length)
      options.push({
        id: "mode",
        name: "Native mode",
        category: "mode",
        type: "select",
        currentValue: selection.mode ?? "",
        options: catalog.modes.map((mode) => ({
          value: mode.id,
          name: mode.name,
          ...(mode.description === undefined ? {} : { description: mode.description }),
        })),
      });
    if (catalog.models.length)
      options.push({
        id: "model",
        name: "Model",
        category: "model",
        type: "select",
        currentValue: selection.model ?? "",
        options: [
          { value: "", name: "Native default / select an allowed model" },
          ...catalog.models.map((model) => ({ value: model.id, name: model.name })),
        ],
      });
    const model = catalog.models.find((model) => model.id === selection.model);
    if (model?.efforts.length)
      options.push({
        id: "effort",
        name: "Reasoning effort",
        category: "thought_level",
        type: "select",
        currentValue: selection.effort ?? "",
        options: [
          { value: "", name: "Native default" },
          ...model.efforts.map((effort) => ({ value: effort.id, name: effort.name })),
        ],
      });
    return options;
  }
  async function select(sessionId: string, configId: string, value: string) {
    peer();
    const option = (await config(sessionId)).find((option) => option.id === configId);
    if (
      option?.type !== "select" ||
      !option.options.some((item) => "value" in item && item.value === value)
    )
      throw acp.RequestError.invalidParams(
        undefined,
        "Select an advertised native mode; an explicit permitted model or reasoning effort is required.",
      );
    const agent = agentFor(sessionId);
    const current = selections.get(sessionId) ?? (await call(agent.models(sessionId))).current;
    selections.set(
      sessionId,
      configId === "model"
        ? { ...current, model: value || undefined, effort: undefined }
        : { ...current, [configId]: value || undefined },
    );
    return { configOptions: await config(sessionId) };
  }
  function peer() {
    if (!state)
      throw acp.RequestError.invalidRequest(undefined, "Initialize the ACP connection first.");
    return state;
  }
  function extension(meta: Record<string, unknown> | null | undefined) {
    if (meta?.swarmx === undefined) return undefined;
    if (!peer().permissions)
      throw acp.RequestError.invalidParams(
        undefined,
        "SwarmX permission extension was not negotiated.",
      );
    return SwarmxRequest.parse(meta.swarmx).permissions;
  }
  async function metadata(sessionId: string) {
    const permissions = peer().permissions;
    return permissions
      ? { _meta: { swarmx: { version: 2, permissions: await permissions(sessionId) } } }
      : {};
  }
  function scope(request: {
    mcpServers?: acp.McpServer[] | null;
    additionalDirectories?: string[] | null;
  }) {
    if (request.mcpServers?.length)
      throw new Error("Configure MCP in the native Agent; SwarmX owns its product carrier.");
    if (request.additionalDirectories?.length)
      throw acp.RequestError.invalidParams(
        undefined,
        "Additional directories are not available in this SwarmX execution.",
      );
  }
  function attach(sessionId: string, agent: NativeAgent) {
    bound.set(sessionId, agent);
  }
  return acp
    .agent({ name: "swarmx" })
    .onConnect((connection) => {
      if (connected) throw new Error("Create one ACP Agent app per connection.");
      connected = true;
      connection.signal.addEventListener(
        "abort",
        () => {
          // The peer is gone, so cancellation failures can only be reported to the Host log.
          void Promise.all([...active].map(([id, agent]) => agent.interrupt(id))).catch((error) =>
            console.error("ACP disconnect cancellation failed:", error),
          );
          connected = false;
          state = undefined;
          selections.clear();
          bound.clear();
          active.clear();
        },
        { once: true },
      );
    })
    .onRequest(acp.methods.agent.initialize, async ({ params }) => {
      const requested = params.clientCapabilities?._meta?.swarmx;
      const swarmx = requested !== undefined && SwarmxCapability.parse(requested).permissions;
      if (swarmx && !supportsPermissions)
        throw acp.RequestError.invalidParams(
          undefined,
          "This Agent cannot enforce the SwarmX permission extension.",
        );
      state = {
        forms: params.clientCapabilities?.elicitation?.form != null,
        permissions: swarmx ? permissionsFor : undefined,
      };
      return {
        protocolVersion: acp.PROTOCOL_VERSION,
        agentInfo: { name: "swarmx", version: "3.3.0" },
        agentCapabilities: publicCapabilities(native.capabilities, supportsPermissions),
      };
    })
    .onRequest(acp.methods.agent.session.list, async () => {
      peer();
      const sessions: Array<{
        sessionId: string;
        title?: string | null;
        updatedAt?: string | null;
        cwd: string;
      }> = [];
      for (const session of await call(native.list())) {
        attach(session.sessionId, native);
        sessions.push({ ...session, cwd });
      }
      return { sessions };
    })
    .onRequest(acp.methods.agent.session.new, async ({ params }) => {
      peer();
      scope(params);
      const agent = await call(bind(params.cwd));
      const permissions = extension(params._meta);
      const sessionId = await call(agent.create(permissions ? { permissions } : undefined));
      attach(sessionId, agent);
      return { sessionId, configOptions: await config(sessionId), ...(await metadata(sessionId)) };
    })
    .onRequest(acp.methods.agent.session.load, async ({ params, client }) => {
      scope(params);
      if (extension(params._meta))
        throw acp.RequestError.invalidParams(
          undefined,
          "Narrow permissions on session/new or session/prompt.",
        );
      const agent = await call(bind(params.cwd));
      const projection = observe(params.sessionId, client, peer().forms, !!peer().permissions);
      await call(agent.read(params.sessionId, projection.observer));
      attach(params.sessionId, agent);
      await projection.flush();
      return {
        configOptions: await config(params.sessionId),
        ...(await metadata(params.sessionId)),
      };
    })
    .onRequest(acp.methods.agent.session.resume, async ({ params }) => {
      scope(params);
      if (extension(params._meta))
        throw acp.RequestError.invalidParams(
          undefined,
          "Narrow permissions on session/new or session/prompt.",
        );
      peer();
      const agent = await call(bind(params.cwd));
      if (agent.capabilities.history)
        await call(
          agent.read(params.sessionId, {
            text() {},
            tool() {},
            raw() {},
            interact: async () => undefined,
          }),
        );
      else await call(agent.models(params.sessionId));
      attach(params.sessionId, agent);
      return {
        configOptions: await config(params.sessionId),
        ...(await metadata(params.sessionId)),
      };
    })
    .onRequest(acp.methods.agent.session.setConfigOption, async ({ params }) => {
      return select(params.sessionId, params.configId, z.string().parse(params.value));
    })
    .onRequest(acp.methods.agent.session.setMode, async ({ params }) => {
      await select(params.sessionId, "mode", params.modeId);
      return {};
    })
    .onRequest(acp.methods.agent.session.prompt, async ({ params, client }) => {
      if (!params.prompt.every((part) => part.type === "text" || part.type === "resource_link"))
        throw new Error("SwarmX accepts ACP text and resource links.");
      const agent = agentFor(params.sessionId);
      const projection = observe(params.sessionId, client, peer().forms, !!peer().permissions);
      const permissions = extension(params._meta);
      if (active.has(params.sessionId))
        throw acp.RequestError.invalidRequest(undefined, "Session is busy.");
      active.set(params.sessionId, agent);
      try {
        const result = await call(
          agent.start(
            params.sessionId,
            params.prompt
              .map((part) =>
                part.type === "text"
                  ? part.text
                  : part.type === "resource_link"
                    ? `\n${part.name}: ${part.uri}\n`
                    : "",
              )
              .join(""),
            projection.observer,
            { ...selections.get(params.sessionId), ...(permissions ? { permissions } : {}) },
          ),
        );
        return { ...result, ...(await metadata(params.sessionId)) };
      } finally {
        active.delete(params.sessionId);
        await projection.flush();
      }
    })
    .onNotification(acp.methods.agent.session.cancel, async ({ params }) => {
      const control =
        params._meta?.swarmx === undefined
          ? undefined
          : z
              .strictObject({ version: z.literal(2), expectedRunId: z.string() })
              .parse(params._meta.swarmx);
      if (control && !peer().permissions)
        throw acp.RequestError.invalidParams(
          undefined,
          "SwarmX permission extension was not negotiated.",
        );
      await agentFor(params.sessionId).interrupt(params.sessionId, control?.expectedRunId);
    })
    .onRequest("_swarmx/models", z.strictObject({ sessionId: z.string() }), async ({ params }) => {
      if (!peer().permissions) throw acp.RequestError.methodNotFound("_swarmx/models");
      return call(agentFor(params.sessionId).models(params.sessionId));
    })
    .onRequest(
      "_swarmx/session/permissions",
      z.strictObject({ sessionId: z.string() }),
      async ({ params }) => {
        if (!peer().permissions)
          throw acp.RequestError.methodNotFound("_swarmx/session/permissions");
        return metadata(params.sessionId);
      },
    )
    .onRequest(
      "_swarmx/session/steer",
      z.strictObject({
        sessionId: z.string().min(1),
        text: z.string().trim().min(1),
        expectedRunId: z.string().optional(),
      }),
      async ({ params }) => {
        if (!peer().permissions) throw acp.RequestError.methodNotFound("_swarmx/session/steer");
        await call(
          agentFor(params.sessionId).steer(params.sessionId, params.text, params.expectedRunId),
        );
        return {};
      },
    );
}

async function call<T>(operation: Promise<T>): Promise<T> {
  try {
    return await operation;
  } catch (error) {
    if (error instanceof acp.RequestError) throw error;
    throw new acp.RequestError(-32000, error instanceof Error ? error.message : String(error));
  }
}

function observe(sessionId: string, client: acp.AgentContext, forms: boolean, swarmx: boolean) {
  const pending: Promise<void>[] = [];
  const tools = new Set<string>();
  const toolActivity = new Map<string, { status?: acp.ToolCallStatus; kind?: acp.ToolKind }>();
  const messageActivity = new Map<string, Record<string, unknown>>();
  const update = (update: acp.SessionUpdate, metadata?: Record<string, unknown>) => {
    pending.push(
      client.notify(acp.methods.client.session.update, {
        sessionId,
        update,
        ...(metadata ? { _meta: metadata } : {}),
      }),
    );
  };
  const observer: Observer = {
    execution(context) {
      if (swarmx)
        update(
          { sessionUpdate: "session_info_update" },
          { swarmx: { version: 2, execution: context } },
        );
    },
    text(id, text, role = "assistant") {
      update({
        sessionUpdate:
          role === "user"
            ? "user_message_chunk"
            : role === "reasoning"
              ? "agent_thought_chunk"
              : "agent_message_chunk",
        content: { type: "text", text },
        messageId: id,
        ...(messageActivity.has(id) ? { _meta: messageActivity.get(id) ?? null } : {}),
      });
    },
    tool(id, name, input, output) {
      if (!tools.has(id)) {
        tools.add(id);
        update({
          sessionUpdate: "tool_call",
          toolCallId: id,
          title: name,
          status: "in_progress",
          rawInput: input,
          ...toolActivity.get(id),
        });
      }
      if (output !== undefined)
        update({
          sessionUpdate: "tool_call_update",
          toolCallId: id,
          status: "completed",
          rawOutput: output,
          ...toolActivity.get(id),
        });
    },
    activity(event) {
      if (event.type === "tool") {
        const kind = z
          .enum([
            "read",
            "edit",
            "delete",
            "move",
            "search",
            "execute",
            "think",
            "fetch",
            "switch_mode",
            "other",
          ])
          .optional()
          .parse(event.kind);
        const metadata = {
          ...toolActivity.get(event.toolCallId),
          ...(kind ? { kind } : {}),
          ...(event.status ? { status: event.status } : {}),
        };
        toolActivity.set(event.toolCallId, metadata);
        if (tools.has(event.toolCallId))
          update({ sessionUpdate: "tool_call_update", toolCallId: event.toolCallId, ...metadata });
      } else {
        const { type: _type, phase, messageId, ...timing } = event;
        if (messageId)
          messageActivity.set(messageId, {
            ...(phase ? { codex: { phase } } : {}),
            swarmx: timing,
          });
        update({
          sessionUpdate: "session_info_update",
          _meta: {
            ...(phase ? { codex: { phase } } : {}),
            swarmx: timing,
            ...(messageId ? { messageId } : {}),
          },
        });
      }
    },
    raw(event, attributes) {
      update(
        { sessionUpdate: "session_info_update" },
        { "swarmx/native": event, ...(attributes ? { "swarmx/attributes": attributes } : {}) },
      );
    },
    async interact(request, signal) {
      if (signal?.aborted) return undefined;
      if (request.approval) {
        const { toolId, choices, input } = request.approval;
        const params: acp.RequestPermissionRequest = {
          sessionId,
          toolCall: {
            toolCallId: toolId,
            title: request.title,
            ...(input === undefined ? {} : { rawInput: input }),
          },
          options: choices.map(({ id, label, kind }) => ({ optionId: id, name: label, kind })),
        };
        const response = await client.request(
          acp.methods.client.session.requestPermission,
          params,
          signal ? { cancellationSignal: signal } : undefined,
        );
        if (signal?.aborted || response.outcome.outcome === "cancelled") return undefined;
        const id = response.outcome.optionId;
        const choice = choices.find((choice) => choice.id === id);
        if (!choice) throw acp.RequestError.invalidParams(undefined, "Unknown permission option.");
        return choice.answer;
      }
      if (!forms) throw new Error("Native interactions require ACP form elicitation support.");
      const answer = await client.request(
        acp.methods.client.elicitation.create,
        {
          mode: "form",
          sessionId,
          message: request.title,
          requestedSchema: request.schema as Extract<
            acp.CreateElicitationRequest,
            { requestedSchema: unknown }
          >["requestedSchema"],
          ...(swarmx ? { _meta: { swarmx: { version: 2, interactionId: request.id } } } : {}),
        },
        signal ? { cancellationSignal: signal } : undefined,
      );
      return answer.action === "accept" ? answer.content : undefined;
    },
  };
  return { observer, flush: () => Promise.all(pending) };
}

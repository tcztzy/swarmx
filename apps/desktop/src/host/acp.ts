import * as acp from "@agentclientprotocol/sdk";
import type { RunOptions } from "@swarmx/swarm";
import { z } from "zod";
import type { NativeAgent, Observer } from "../agents/types.js";
import { SwarmxCapability, SwarmxRequest } from "./acp-extension.js";

/** Leaf adapter from ACP to the Host-protected native SDK boundary. */
export function acpAgent(agent: NativeAgent, cwd: string): acp.AgentApp {
  let state: { forms: boolean; permissions: NativeAgent["permissions"] } | undefined;
  const getPermissions = agent.permissions?.bind(agent);
  let connected = false;
  const selections = new Map<string, RunOptions>();
  const active = new Set<string>();
  async function config(sessionId: string): Promise<acp.SessionConfigOption[]> {
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
  function workspace(request: {
    cwd: string;
    mcpServers?: acp.McpServer[] | null;
    additionalDirectories?: string[] | null;
  }) {
    if (request.cwd !== cwd) throw new Error("This SwarmX Host owns a different workspace.");
    if (request.mcpServers?.length)
      throw new Error("Configure MCP in the native Agent; SwarmX owns its product carrier.");
    if (request.additionalDirectories?.length)
      throw acp.RequestError.invalidParams(
        undefined,
        "Additional directories exceed this Host's workspace.",
      );
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
          void Promise.all([...active].map((id) => agent.interrupt(id))).catch((error) =>
            console.error("ACP disconnect cancellation failed:", error),
          );
          connected = false;
          state = undefined;
          selections.clear();
        },
        { once: true },
      );
    })
    .onRequest(acp.methods.agent.initialize, ({ params }) => {
      const requested = params.clientCapabilities?._meta?.swarmx;
      const swarmx = requested !== undefined && SwarmxCapability.parse(requested).permissions;
      if (swarmx && !getPermissions)
        throw acp.RequestError.invalidParams(
          undefined,
          "This Agent cannot enforce the SwarmX permission extension.",
        );
      state = {
        forms: params.clientCapabilities?.elicitation?.form != null,
        permissions: swarmx ? getPermissions : undefined,
      };
      return {
        protocolVersion: acp.PROTOCOL_VERSION,
        agentInfo: { name: agent.name, version: "0.1.0" },
        agentCapabilities: {
          ...agent.capabilities,
          _meta: {
            ...agent.capabilities._meta,
            swarmx: {
              version: 2,
              permissions: !!agent.permissions,
              steer: true,
              activeRunResume: false,
              interactionResume: false,
            },
          },
        },
      };
    })
    .onRequest(acp.methods.agent.session.list, async () => {
      peer();
      return { sessions: (await call(agent.list())).map((session) => ({ ...session, cwd })) };
    })
    .onRequest(acp.methods.agent.session.new, async ({ params }) => {
      peer();
      workspace(params);
      const permissions = extension(params._meta);
      const sessionId = await call(agent.create(permissions ? { permissions } : undefined));
      return { sessionId, configOptions: await config(sessionId), ...(await metadata(sessionId)) };
    })
    .onRequest(acp.methods.agent.session.load, async ({ params, client }) => {
      workspace(params);
      if (extension(params._meta))
        throw acp.RequestError.invalidParams(
          undefined,
          "Narrow permissions on session/new or session/prompt.",
        );
      const projection = observe(params.sessionId, client, peer().forms, !!peer().permissions);
      await call(agent.read(params.sessionId, projection.observer));
      await projection.flush();
      return {
        configOptions: await config(params.sessionId),
        ...(await metadata(params.sessionId)),
      };
    })
    .onRequest(acp.methods.agent.session.resume, async ({ params }) => {
      workspace(params);
      if (extension(params._meta))
        throw acp.RequestError.invalidParams(
          undefined,
          "Narrow permissions on session/new or session/prompt.",
        );
      peer();
      await call(
        agent.read(params.sessionId, {
          text() {},
          tool() {},
          raw() {},
          interact: async () => undefined,
        }),
      );
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
      const projection = observe(params.sessionId, client, peer().forms, !!peer().permissions);
      const permissions = extension(params._meta);
      if (active.has(params.sessionId))
        throw acp.RequestError.invalidRequest(undefined, "Session is busy.");
      active.add(params.sessionId);
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
      await agent.interrupt(params.sessionId, control?.expectedRunId);
    })
    .onRequest(
      "_swarmx/models",
      z.strictObject({ sessionId: z.string().optional() }),
      async ({ params }) => {
        if (!peer().permissions) throw acp.RequestError.methodNotFound("_swarmx/models");
        return call(agent.models(params.sessionId));
      },
    )
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
        await call(agent.steer(params.sessionId, params.text, params.expectedRunId));
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
        });
      }
      if (output !== undefined)
        update({
          sessionUpdate: "tool_call_update",
          toolCallId: id,
          status: "completed",
          rawOutput: output,
        });
    },
    raw(event, attributes) {
      update(
        { sessionUpdate: "session_info_update" },
        { "swarmx/native": event, ...(attributes ? { "swarmx/attributes": attributes } : {}) },
      );
    },
    async interact(request, signal) {
      if (signal?.aborted) return undefined;
      if (request.permission) {
        const { toolCall, options, answers } = request.permission;
        const response = await client.request(
          acp.methods.client.session.requestPermission,
          { sessionId, toolCall, options },
          signal ? { cancellationSignal: signal } : undefined,
        );
        if (signal?.aborted || response.outcome.outcome === "cancelled") return undefined;
        const id = response.outcome.optionId;
        if (!options.some((option) => option.optionId === id) || !Object.hasOwn(answers, id))
          throw acp.RequestError.invalidParams(undefined, "Unknown permission option.");
        return answers[id];
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

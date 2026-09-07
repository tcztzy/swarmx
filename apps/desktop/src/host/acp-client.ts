import { randomUUID } from "node:crypto";
import * as acp from "@agentclientprotocol/sdk";
import type { ModelCatalog } from "@swarmx/swarm";
import { z } from "zod";
import type { NativeAgent, Observer } from "../agents/types.js";
import {
  type AgentPermissions,
  AgentPermissionsSchema,
  narrowPermissions,
} from "../permissions.js";
import { acknowledgedPermissions, SwarmxCapability } from "./acp-extension.js";

const catalogSchema = z.object({
  models: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      description: z.string().optional(),
      efforts: z.array(z.object({ id: z.string(), name: z.string() })),
      defaultEffort: z.string().optional(),
    }),
  ),
  modes: z
    .array(z.object({ id: z.string(), name: z.string(), description: z.string().optional() }))
    .optional(),
  current: z.object({
    model: z.string().optional(),
    effort: z.string().optional(),
    mode: z.string().optional(),
  }),
});
const executionSchema = z.object({
  runId: z.string(),
  parentRunId: z.string().nullable(),
  causedBy: z.string().nullable(),
  permissions: AgentPermissionsSchema.optional(),
});

/** Projection for existing Host gateways. Every operation crosses an official ACP connection. */
export function acpClient(
  name: string,
  capabilities: acp.AgentCapabilities,
  cwd: string,
  connect: (client: acp.ClientApp) => acp.ClientConnection,
  ceiling: () => AgentPermissions,
  preparing = new Map<string, Set<AbortController>>(),
): NativeAgent {
  const connections = new Set<acp.ClientConnection>();
  async function run<T>(
    operation: (agent: acp.ClientContext, permissions: AgentPermissions) => Promise<T>,
    observer?: Observer,
  ): Promise<T> {
    const permissions = ceiling();
    const tools = new Map<string, { name: string; input: unknown }>();
    const client = acp
      .client({ name: "swarmx-host" })
      .onNotification(acp.methods.client.session.update, ({ params }) => {
        if (!observer) return;
        const { update } = params;
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
            const tool = tools.get(update.toolCallId);
            if (tool)
              observer.tool(
                update.toolCallId,
                update.title ?? tool.name,
                update.rawInput ?? tool.input,
                update.rawOutput,
              );
            break;
          }
        }
        if (params._meta && Object.hasOwn(params._meta, "swarmx/native"))
          observer.raw(
            params._meta["swarmx/native"],
            params._meta["swarmx/attributes"] === undefined
              ? undefined
              : z
                  .record(
                    z.string(),
                    z.union([z.string(), z.number(), z.boolean(), z.null()]).optional(),
                  )
                  .parse(params._meta["swarmx/attributes"]),
          );
        const execution = z
          .object({ execution: executionSchema.optional() })
          .parse(params._meta?.swarmx ?? {}).execution;
        if (execution)
          observer.execution?.({
            runId: execution.runId,
            parentRunId: execution.parentRunId,
            causedBy: execution.causedBy,
            ...(execution.permissions ? { permissions: execution.permissions } : {}),
          });
      })
      .onRequest(acp.methods.client.session.requestPermission, async ({ params, signal }) => {
        if (!observer) throw new Error("ACP permission request has no connected observer.");
        const answers = Object.fromEntries(
          params.options.map((option) => [option.optionId, { optionId: option.optionId }]),
        );
        const answer = await observer.interact(
          {
            id: params.toolCall.toolCallId,
            title: params.toolCall.title ?? "Permission",
            schema: {
              type: "object",
              properties: {
                optionId: { type: "string", enum: params.options.map((option) => option.optionId) },
              },
              required: ["optionId"],
              additionalProperties: false,
            },
            permission: { toolCall: params.toolCall, options: params.options, answers },
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
        if (!observer || params.mode !== "form")
          throw new Error("ACP form has no connected observer.");
        const metadata = z
          .object({ interactionId: z.string().optional() })
          .parse(params._meta?.swarmx ?? {});
        const answer = await observer.interact(
          {
            id: metadata.interactionId ?? randomUUID(),
            title: params.message,
            schema: z.record(z.string(), z.unknown()).parse(params.requestedSchema),
          },
          signal,
        );
        if (signal.aborted || answer === undefined) return { action: "cancel" };
        return { action: "accept", content: z.record(z.string(), z.unknown()).parse(answer) };
      });
    // Allocate inside the caller's AsyncLocalStorage scope; never share a reader across authorities.
    const connection = connect(client);
    connections.add(connection);
    try {
      const initialized = await connection.agent.request(acp.methods.agent.initialize, {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {
          elicitation: { form: {} },
          _meta: { swarmx: { version: 2, permissions: true } },
        },
      });
      if (initialized.protocolVersion !== acp.PROTOCOL_VERSION)
        throw new Error("Unsupported ACP version.");
      SwarmxCapability.parse(initialized.agentCapabilities?._meta?.swarmx);
      return await operation(connection.agent, permissions);
    } finally {
      connections.delete(connection);
      connection.close();
    }
  }
  return {
    name,
    capabilities,
    models: (sessionId) =>
      run(
        async (agent): Promise<ModelCatalog> =>
          catalogSchema.parse(await agent.request("_swarmx/models", { sessionId })),
      ),
    permissions: (sessionId) =>
      run(async (agent, permissions) => {
        const result = z
          .object({ _meta: z.record(z.string(), z.unknown()) })
          .parse(await agent.request("_swarmx/session/permissions", { sessionId }));
        return acknowledgedPermissions(result._meta, permissions);
      }),
    list: () =>
      run(async (agent) =>
        (await agent.request(acp.methods.agent.session.list, {})).sessions.map(
          ({ sessionId, title, updatedAt }) => ({
            sessionId,
            ...(title == null ? {} : { title }),
            ...(updatedAt == null ? {} : { updatedAt }),
          }),
        ),
      ),
    create: (options) =>
      run(async (agent, permissions) => {
        if (options?.instructions !== undefined)
          throw new Error("Instructions belong to the native leaf boundary.");
        const result = await agent.request(acp.methods.agent.session.new, {
          cwd,
          mcpServers: [],
          _meta: { swarmx: { version: 2, permissions: options?.permissions } },
        });
        acknowledgedPermissions(result._meta, narrowPermissions(permissions, options?.permissions));
        return result.sessionId;
      }),
    read: (sessionId, observer) =>
      run(async (agent, permissions) => {
        const result = await agent.request(acp.methods.agent.session.load, {
          sessionId,
          cwd,
          mcpServers: [],
        });
        acknowledgedPermissions(result._meta, permissions);
      }, observer),
    async start(sessionId, text, observer, options) {
      const preparation = new AbortController();
      const preparations = preparing.get(sessionId) ?? new Set<AbortController>();
      preparations.add(preparation);
      preparing.set(sessionId, preparations);
      const release = () => {
        preparations.delete(preparation);
        if (preparing.get(sessionId) === preparations && preparations.size === 0)
          preparing.delete(sessionId);
      };
      try {
        return await run(async (agent, permissions) => {
          for (const [configId, value] of [
            ["model", options?.model],
            ["effort", options?.effort],
            ["mode", options?.mode],
          ] as const)
            if (value !== undefined)
              await agent.request(acp.methods.agent.session.setConfigOption, {
                sessionId,
                configId,
                value,
              });
          if (preparation.signal.aborted) return { stopReason: "cancelled" };
          release();
          const result = await agent.request(acp.methods.agent.session.prompt, {
            sessionId,
            prompt: [{ type: "text", text }],
            _meta: { swarmx: { version: 2, permissions: options?.permissions } },
          });
          acknowledgedPermissions(
            result._meta,
            narrowPermissions(permissions, options?.permissions),
          );
          const { _meta, ...response } = result;
          return response;
        }, observer);
      } finally {
        release();
      }
    },
    steer: (sessionId, text, expectedRunId) =>
      run(async (agent) => {
        await agent.request("_swarmx/session/steer", { sessionId, text, expectedRunId });
      }),
    async interrupt(sessionId, expectedRunId) {
      if (expectedRunId === undefined)
        for (const preparation of preparing.get(sessionId) ?? []) preparation.abort();
      await run(async (agent) => {
        await agent.notify(acp.methods.agent.session.cancel, {
          sessionId,
          ...(expectedRunId ? { _meta: { swarmx: { version: 2, expectedRunId } } } : {}),
        });
        // A request keeps this operation's connection open until the notification is delivered.
        await agent.request("_swarmx/session/permissions", { sessionId });
      });
    },
    async dispose() {
      for (const connection of connections) connection.close();
    },
  };
}

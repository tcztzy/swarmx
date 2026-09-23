import { randomUUID } from "node:crypto";
import { type AGUIEvent, EventType } from "@ag-ui/core";
import type { RunResult } from "@swarmx/swarm";
import { z } from "zod";
import type { NativeAgent, Observer } from "../agents/types.js";
import { ActivitySchema } from "../message-activity.js";
import type { ExecutionContext, ExecutionJournal } from "./execution-journal.js";
import type { AgentMemory } from "./memory.js";

/** One durable observation boundary shared by browser, ACP, A2A and delegation. */
export function recordedAgent(
  journal: ExecutionJournal,
  harness: string,
  agent: NativeAgent,
  memory?: AgentMemory,
): NativeAgent {
  const pending = new Set<Promise<RunResult>>();
  const interrupted = new Set<string>();
  const globalCatalog = harness === "hermes" || harness === "openclaw" || harness === "dsh";
  const assertSession = (id: string) => {
    if (globalCatalog && !journal.sessionIds().includes(id))
      throw new Error("Session does not belong to this directory.");
  };
  const control = async (
    sessionId: string,
    name: string,
    value: unknown,
    execute: () => Promise<void>,
  ) => {
    assertSession(sessionId);
    const context = journal.activeSession(sessionId);
    const request = context && journal.append(context, { type: EventType.CUSTOM, name, value });
    try {
      await execute();
      if (request && context)
        journal.append(
          { ...context, causedBy: request.id },
          {
            type: EventType.CUSTOM,
            name: "swarmx.control.completed",
            value: {},
          },
        );
    } catch (error) {
      if (request && context)
        journal.append(
          { ...context, causedBy: request.id },
          {
            type: EventType.CUSTOM,
            name: "swarmx.control.failed",
            value: {
              message: error instanceof Error ? error.message : String(error),
            },
          },
        );
      throw error;
    }
  };
  const wrapped: NativeAgent = {
    name: agent.name,
    capabilities: agent.capabilities,
    models: async (id) => {
      if (id) assertSession(id);
      const catalog = await agent.models(harness === "dsh" ? undefined : id);
      const mode = id && harness !== "hermes" ? journal.sessionMode(id) : undefined;
      return mode ? { ...catalog, current: { ...catalog.current, mode } } : catalog;
    },
    async list() {
      if (harness === "dsh")
        return journal
          .sessionIds()
          .filter((id) => id.startsWith("dsh:"))
          .map((sessionId) => ({ sessionId }));
      const sessions = await agent.list();
      if (!globalCatalog) return sessions;
      const owned = new Set(journal.sessionIds());
      return sessions.filter(({ sessionId }) => owned.has(sessionId));
    },
    async create() {
      const instructions = await memory?.context();
      const id = await agent.create(instructions === undefined ? undefined : { instructions });
      const parent = journal.scope.getStore();
      if (globalCatalog || parent?.permissions || harness === "claude")
        journal.append(
          {
            sessionId: id,
            runId: randomUUID(),
            causedBy: parent?.causedBy ?? null,
            attributes: { "swarmx.harness.name": harness },
          },
          {
            type: EventType.CUSTOM,
            name: "swarmx.session.created",
            value: parent?.permissions ? { permissions: parent.permissions } : {},
          },
        );
      await memory?.snapshot(id, instructions);
      return id;
    },
    read: async (id, observer) => {
      assertSession(id);
      if (harness === "dsh") {
        if (!id.startsWith("dsh:")) throw new Error("Session does not belong to DSH.");
        const tools = new Map<string, { name: string; input: unknown }>();
        let after = 0;
        for (;;) {
          const page = journal.read({ session: id, after });
          for (const { event, attributes } of page.events) {
            if (event.type === EventType.RUN_STARTED) {
              for (const message of event.input?.messages ?? [])
                if (message.role === "user" && typeof message.content === "string")
                  observer.text(message.id, message.content, "user");
            } else if (
              event.type === EventType.TEXT_MESSAGE_CHUNK ||
              event.type === EventType.REASONING_MESSAGE_CHUNK
            ) {
              const chunk = z
                .object({
                  messageId: z.string(),
                  delta: z.string(),
                  role: z.enum(["user", "assistant"]).optional(),
                })
                .parse(event);
              observer.text(
                chunk.messageId,
                chunk.delta,
                event.type === EventType.REASONING_MESSAGE_CHUNK ? "reasoning" : chunk.role,
              );
            } else if (event.type === EventType.TOOL_CALL_CHUNK) {
              const chunk = z
                .object({ toolCallName: z.string(), toolCallId: z.string(), delta: z.string() })
                .parse(event);
              const call = {
                name: chunk.toolCallName,
                input: JSON.parse(chunk.delta) as unknown,
              };
              tools.set(chunk.toolCallId, call);
              observer.tool(chunk.toolCallId, call.name, call.input);
            } else if (event.type === EventType.TOOL_CALL_RESULT) {
              const call = tools.get(event.toolCallId);
              if (call)
                observer.tool(event.toolCallId, call.name, call.input, JSON.parse(event.content));
            } else if (event.type === EventType.RAW) observer.raw(event.event, attributes);
            else if (event.type === EventType.CUSTOM && event.name === "swarmx.activity")
              observer.activity?.(ActivitySchema.parse(event.value));
          }
          if (!page.events.length) return;
          after = page.nextAfter;
        }
      }
      return agent.read(id, {
        tool: observer.tool.bind(observer),
        raw: observer.raw.bind(observer),
        activity: (event) => observer.activity?.(event),
        interact: observer.interact.bind(observer),
        text: (messageId, text, role = "assistant") =>
          observer.text(
            messageId,
            role === "user" ? (memory?.visibleText(id, text) ?? text) : text,
            role,
          ),
      });
    },
    start(sessionId, text, observer, selection) {
      const execute = async () => {
        assertSession(sessionId);
        if (journal.activeSession(sessionId)) throw new Error("Session is busy.");
        if (
          harness === "dsh" &&
          (!sessionId.startsWith("dsh:") || !journal.emptySessions("dsh").includes(sessionId))
        )
          throw new Error(
            "DSH tasks execute once. Create a new task; previous output remains in the execution log.",
          );
        const mode =
          selection?.mode ?? (harness === "hermes" ? undefined : journal.sessionMode(sessionId));
        if (mode) selection = { ...selection, mode };
        const runId = randomUUID();
        const interactions = new AbortController();
        const cancelInteractions = () => interactions.abort();
        const parent = journal.scope.getStore();
        const permissions = parent?.permissions;
        const context: ExecutionContext = {
          ...(permissions ? { permissions } : {}),
          sessionId,
          runId,
          causedBy: journal.scope.getStore()?.causedBy ?? null,
          agent: wrapped,
          interact: observer.interact.bind(observer),
          cancelInteractions,
          attributes: {
            ...Object.fromEntries(
              Object.entries(parent?.attributes ?? {}).filter(
                ([key]) =>
                  key.startsWith("swarmx.work.") ||
                  key === "swarmx.execution.purpose" ||
                  key === "swarmx.memory.review.source_run_ids",
              ),
            ),
            "swarmx.execution.parent_run_id": parent?.sessionId ? parent.runId : null,
            "swarmx.harness.name": harness,
            "swarmx.harness.version": null,
            "gen_ai.agent.name": agent.name,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": sessionId,
            "gen_ai.request.model": selection?.model ?? null,
            "gen_ai.request.reasoning.level": selection?.effort ?? null,
            "swarmx.memory.review_eligible": Boolean(
              memory?.automatic &&
                (permissions === undefined || permissions.tools.includes("memory.write")) &&
                permissions?.delegation !== false,
            ),
            "swarmx.memory.review_permissions": memory
              ? JSON.stringify(memory.reviewPermissions(permissions))
              : null,
          },
        };
        const started = journal.append(context, {
          type: EventType.RUN_STARTED,
          threadId: sessionId,
          runId,
          input: {
            threadId: sessionId,
            runId,
            messages: [{ id: randomUUID(), role: "user", content: text }],
            tools: [],
            context: [],
            state: {},
            forwardedProps: Object.fromEntries(
              Object.entries({ ...selection, permissions: context.permissions }).filter(
                ([, value]) => value !== undefined,
              ),
            ),
          },
        });
        const scope = {
          ...context,
          causedBy: started.id,
          pendingInteractions: 0,
        };
        journal.activate(scope);
        const send = (event: AGUIEvent) => journal.append(scope, event);
        const tools = new Set<string>();
        const recorded: Observer = {
          executionId: runId,
          text(id, delta, role = "assistant") {
            send(
              role === "reasoning"
                ? { type: EventType.REASONING_MESSAGE_CHUNK, messageId: id, delta }
                : { type: EventType.TEXT_MESSAGE_CHUNK, messageId: id, role, delta },
            );
            observer.text(id, delta, role);
          },
          tool(id, name, input, output) {
            if (!tools.has(id)) {
              send({
                type: EventType.TOOL_CALL_CHUNK,
                toolCallId: id,
                toolCallName: name,
                delta: JSON.stringify(input ?? null),
              });
              tools.add(id);
            }
            if (output !== undefined)
              send({
                type: EventType.TOOL_CALL_RESULT,
                toolCallId: id,
                messageId: randomUUID(),
                content: JSON.stringify(output),
              });
            observer.tool(id, name, input, output);
          },
          raw(event, attributes) {
            journal.append(scope, { type: EventType.RAW, source: harness, event }, attributes);
            for (const key of [
              "swarmx.harness.version",
              "swarmx.agent.version",
              "swarmx.agent.mode",
              "swarmx.native.mode",
              "gen_ai.provider.name",
              "swarmx.model.version",
              "gen_ai.usage.input_tokens",
              "gen_ai.usage.output_tokens",
              "swarmx.usage.cost_usd",
              "swarmx.usage.basis",
              "swarmx.usage.cached_input_tokens",
              "swarmx.usage.reasoning_output_tokens",
              "swarmx.usage.coverage",
              "swarmx.usage.cost_source",
              "swarmx.usage.scope",
            ])
              if (attributes?.[key] !== undefined) scope.attributes[key] = attributes[key];
            const nativeRun = attributes?.["swarmx.native.run_id"];
            if (typeof nativeRun === "string") scope.attributes["swarmx.native.run_id"] = nativeRun;
            observer.raw(event, attributes);
          },
          activity(event) {
            send({ type: EventType.CUSTOM, name: "swarmx.activity", value: event });
            observer.activity?.(event);
          },
          async interact(request, signal) {
            const lifetime = signal
              ? AbortSignal.any([signal, interactions.signal])
              : interactions.signal;
            if (lifetime.aborted) return undefined;
            send({ type: EventType.CUSTOM, name: "swarmx.interaction.requested", value: request });
            scope.pendingInteractions += 1;
            let answer: unknown;
            const cancelled = Promise.withResolvers<undefined>();
            const cancel = () => cancelled.resolve(undefined);
            lifetime.addEventListener("abort", cancel, { once: true });
            try {
              answer = await Promise.race([
                observer.interact(request, lifetime),
                cancelled.promise,
              ]);
            } finally {
              lifetime.removeEventListener("abort", cancel);
              scope.pendingInteractions -= 1;
            }
            send({
              type: EventType.CUSTOM,
              name: "swarmx.interaction.answered",
              value: {
                id: request.id,
                status: answer === undefined ? "cancelled" : "answered",
                ...(request.sensitive ? { redacted: true } : { answer: answer ?? null }),
              },
            });
            return answer;
          },
        };
        try {
          observer.execution?.({
            runId,
            parentRunId: parent?.sessionId ? parent.runId : null,
            causedBy: context.causedBy,
            ...(permissions ? { permissions } : {}),
          });
          const result = await journal.scope.run(scope, async () => {
            const instructions = await memory?.snapshot(sessionId);
            if (memory && context.attributes["swarmx.memory.review_eligible"]) {
              const resources = await memory.resources
                .snapshot(interactions.signal)
                .catch((error: unknown) => {
                  if (!interrupted.has(sessionId)) throw error;
                  return [];
                });
              if (resources.length)
                send({
                  type: EventType.CUSTOM,
                  name: "swarmx.learning.resources",
                  value: resources.map(({ id, kind, path, expectedRevision }) => ({
                    id,
                    kind,
                    path,
                    revision: expectedRevision,
                  })),
                });
            }
            if (interrupted.has(sessionId)) return { stopReason: "cancelled" as const };
            return agent.start(sessionId, text, recorded, {
              ...selection,
              ...(instructions &&
              (["pi", "codex", "claude"].includes(harness) ||
                journal.completedTurns(sessionId) === 0)
                ? { instructions }
                : {}),
            });
          });
          journal.append(
            scope,
            {
              type: EventType.RUN_FINISHED,
              threadId: sessionId,
              runId,
              result: { ...result, interruptionRequested: interrupted.has(sessionId) },
            },
            { "swarmx.memory.tool_calls": tools.size },
          );
          return result;
        } catch (error) {
          send({
            type: EventType.RUN_ERROR,
            message: error instanceof Error ? error.message : String(error),
          });
          throw error;
        } finally {
          cancelInteractions();
          journal.deactivate(sessionId);
          interrupted.delete(sessionId);
          memory?.resume();
        }
      };
      const operation = execute();
      pending.add(operation);
      void operation.then(
        () => pending.delete(operation),
        () => pending.delete(operation),
      );
      return operation;
    },
    steer: (id, text) => control(id, "swarmx.input.steered", { text }, () => agent.steer(id, text)),
    interrupt: (id) =>
      control(id, "swarmx.run.interrupt_requested", {}, async () => {
        const active = journal.activeSession(id);
        if (active) {
          interrupted.add(id);
          active.cancelInteractions?.();
        }
        return agent.interrupt(id);
      }),
    async dispose() {
      for (const active of journal.activeRuns()) {
        if (active.agent === wrapped) active.cancelInteractions?.();
      }
      await agent.dispose();
      await Promise.allSettled([...pending]);
    },
  };
  return wrapped;
}

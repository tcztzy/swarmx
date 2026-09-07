import { randomUUID } from "node:crypto";
import { type AGUIEvent, EventType } from "@ag-ui/core";
import type { PromptResponse } from "@swarmx/swarm";
import type { NativeAgent, Observer } from "../agents/types.js";
import type { ExecutionContext, ExecutionJournal } from "./execution-journal.js";
import type { AgentMemory } from "./memory.js";

/** One durable observation boundary shared by browser, ACP, A2A and delegation. */
export function recordedAgent(
  journal: ExecutionJournal,
  harness: string,
  agent: NativeAgent,
  memory?: AgentMemory,
): NativeAgent {
  const pending = new Set<Promise<PromptResponse>>();
  const interrupted = new Set<string>();
  const globalCatalog = harness === "hermes" || harness === "openclaw";
  const assertSession = (id: string) => {
    if (globalCatalog && !journal.sessionIds().includes(id))
      throw new Error("Session does not belong to this project.");
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
      const catalog = await agent.models(id);
      const mode = id ? journal.sessionMode(id) : undefined;
      return mode ? { ...catalog, current: { ...catalog.current, mode } } : catalog;
    },
    async list() {
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
      return agent.read(id, {
        tool: observer.tool.bind(observer),
        raw: observer.raw.bind(observer),
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
        const mode = selection?.mode ?? journal.sessionMode(sessionId);
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
            "swarmx.harness.name": harness,
            "swarmx.harness.version": null,
            "gen_ai.agent.name": agent.name,
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": sessionId,
            "gen_ai.request.model": selection?.model ?? null,
            "gen_ai.request.reasoning.level": selection?.effort ?? null,
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
            const nativeRun = attributes?.["swarmx.native.run_id"];
            if (typeof nativeRun === "string") scope.attributes["swarmx.native.run_id"] = nativeRun;
            observer.raw(event, attributes);
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
                answer: answer ?? null,
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
            if (interrupted.has(sessionId)) return { stopReason: "cancelled" as const };
            return agent.start(sessionId, text, recorded, {
              ...selection,
              ...(instructions &&
              (["codex", "claude"].includes(harness) || journal.completedTurns(sessionId) === 0)
                ? { instructions }
                : {}),
            });
          });
          send({
            type: EventType.RUN_FINISHED,
            threadId: sessionId,
            runId,
            result: { ...result, interruptionRequested: interrupted.has(sessionId) },
          });
          if (
            result.stopReason === "end_turn" &&
            !interrupted.has(sessionId) &&
            (context.permissions === undefined ||
              context.permissions.tools.includes("memory.write")) &&
            context.permissions?.delegation !== false
          )
            memory?.completed(sessionId, tools.size);
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

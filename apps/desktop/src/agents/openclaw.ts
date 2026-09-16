import { randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";
import { GatewayClient, readAssistantStreamSegmentIdentity } from "@openclaw/gateway-client";
import {
  type QuestionGetResult,
  QuestionGetResultSchema,
  QuestionResolveResultSchema,
  validateApprovalGetResult,
  validateApprovalResolveResult,
} from "@openclaw/gateway-protocol";
import {
  GATEWAY_CLIENT_CAPS,
  GATEWAY_CLIENT_MODES,
  GATEWAY_CLIENT_NAMES,
} from "@openclaw/gateway-protocol/client-info";
import { PROTOCOL_VERSION } from "@openclaw/gateway-protocol/version";
import type { ModelCatalog, RunResult } from "@swarmx/swarm";
import { z } from "zod";
import manifest from "../../package.json" with { type: "json" };
import { requestApproval } from "./approval.js";
import { openClawAuth } from "./openclaw-auth.js";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

const modeSchema = z.enum(["read-only", "guarded", "workspace", "full"]);
const questionSnapshot = z.fromJSONSchema(
  QuestionGetResultSchema as unknown as Parameters<typeof z.fromJSONSchema>[0],
) as z.ZodType<QuestionGetResult>;
const questionResolution = z.fromJSONSchema(
  QuestionResolveResultSchema as unknown as Parameters<typeof z.fromJSONSchema>[0],
);
const rowSchema = z.looseObject({
  key: z.string(),
  displayName: z.string().optional(),
  derivedTitle: z.string().optional(),
  label: z.string().optional(),
  updatedAt: z.number().nullable().optional(),
  model: z.string().optional(),
  modelProvider: z.string().optional(),
  thinkingLevel: z.string().optional(),
  permissionMode: modeSchema.optional(),
});
const messageSchema = z.looseObject({
  role: z.string(),
  content: z.union([z.string(), z.array(z.looseObject({ type: z.string() }))]),
  id: z.string().optional(),
  toolCallId: z.string().optional(),
  toolName: z.string().optional(),
  isError: z.boolean().optional(),
});
const modelId = (provider: string | undefined, model: string) =>
  provider && !model.startsWith(`${provider}/`) ? `${provider}/${model}` : model;

function stream(id: string) {
  return {
    text: "",
    messageId: id,
    thought: "",
    seq: -1,
    toolSeq: -1,
    finished: false,
    lifetime: new AbortController(),
    tools: new Map<string, { name: string; args: unknown }>(),
  };
}
type OpenClawRun = {
  id: string;
  key: string;
  observer: Observer;
  done: ReturnType<typeof Promise.withResolvers<RunResult | Error>>;
  stopped: boolean;
  submitted: boolean;
  finished: boolean;
  result: RunResult;
  lifetime: AbortController;
  streams: Map<string, ReturnType<typeof stream>>;
};

export async function createOpenClaw(options: AgentOptions): Promise<NativeAgent> {
  if (options.reviewOnly) throw new Error("OpenClaw does not support restricted memory reviews.");
  const ready = Promise.withResolvers<void>();
  const legacySessions = new Map<string, string>();
  const running = new Map<string, OpenClawRun>();
  function finish(run: OpenClawRun, native: ReturnType<typeof stream>, result: RunResult | Error) {
    native.finished = true;
    native.lifetime.abort();
    if (result instanceof Error) run.done.resolve(result);
    else if (run.result.stopReason === "end_turn") run.result = result;
    if ([...run.streams.values()].every((native) => native.finished)) {
      run.finished = true;
      run.lifetime.abort();
      run.done.resolve(run.result);
    }
  }
  async function cancel(run: OpenClawRun) {
    await Promise.all(
      [...run.streams]
        .filter(([, native]) => !native.finished)
        .map(([runId]) =>
          client.request("chat.abort", { sessionKey: run.key, runId, preserveSideRuns: true }),
        ),
    );
  }
  async function send(
    run: OpenClawRun,
    runId: string,
    message: string,
    params: { thinking?: string; queueMode?: "steer" },
  ) {
    const ack = z
      .object({
        status: z.enum(["started", "in_flight", "accepted", "ok", "error", "timeout", "aborted"]),
        runId: z.string(),
        summary: z.string().optional(),
      })
      .parse(
        await client.request(
          "chat.send",
          {
            sessionKey: run.key,
            idempotencyKey: runId,
            message,
            ...params,
            deliver: false,
          },
          { timeoutMs: null },
        ),
      );
    if (ack.runId !== runId) throw new Error("OpenClaw changed the submitted run identity.");
    const aborted =
      ack.status === "aborted" || (ack.status === "timeout" && ack.summary === "aborted");
    if (ack.status === "error" || (ack.status === "timeout" && !aborted))
      throw new Error(ack.summary ?? `OpenClaw prompt failed: ${ack.status}`);
    if (ack.status === "ok" || aborted) {
      const native = run.streams.get(runId);
      if (native && !native.finished)
        finish(run, native, { stopReason: aborted ? "cancelled" : "end_turn" });
    }
  }
  const approvals = new Map<string, AbortController>();
  const questions = new Map<string, AbortController>();
  async function ask(id: string, runId: string, run: OpenClawRun, lifetime: AbortSignal) {
    if (questions.has(id) || lifetime.aborted) return;
    const control = new AbortController();
    questions.set(id, control);
    try {
      const { question: request } = questionSnapshot.parse(
        await client.request("question.get", { id }),
      );
      if (request.id !== id || request.runId !== runId || request.sessionKey !== run.key)
        throw new Error("OpenClaw question identity changed.");
      if (
        request.status !== "pending" ||
        request.expiresAtMs <= Date.now() ||
        control.signal.aborted ||
        lifetime.aborted
      )
        return;
      const fields = request.questions.flatMap((question, index) => {
        const title = [
          question.question,
          question.url,
          ...(question.secretStore
            ? [
                `${question.secretStoreExisting ? "Replace" : "Store"} OpenClaw secret: ${question.secretStore.name}`,
                `Allowed hosts: ${JSON.stringify(question.secretStore.allowedHosts ?? [])}`,
                question.secretStore.reason,
              ]
            : []),
        ]
          .filter(Boolean)
          .join("\n");
        const choices = {
          type: "string",
          oneOf: question.options.map((option) => ({
            const: option.label,
            title: option.description ? `${option.label} — ${option.description}` : option.label,
          })),
        };
        const values = z.enum(question.options.map((option) => option.label));
        return [
          ...(question.options.length
            ? [
                {
                  key: `choice_${index}`,
                  schema: {
                    ...(question.multiSelect ? { type: "array", items: choices } : choices),
                    title,
                  },
                  validate: (question.multiSelect ? z.array(values).min(1) : values).optional(),
                  required: !question.isOther,
                },
              ]
            : []),
          ...(!question.options.length || question.isOther
            ? [
                {
                  key: `other_${index}`,
                  schema: {
                    type: "string",
                    title: question.options.length ? `${title} — Other answer` : title,
                    ...(question.isSecret ? { format: "password" } : {}),
                  },
                  validate: z.string().min(1).optional(),
                  required: !question.options.length,
                },
              ]
            : []),
        ];
      });
      const signal = AbortSignal.any([
        control.signal,
        lifetime,
        AbortSignal.timeout(Math.max(0, request.expiresAtMs - Date.now())),
      ]);
      const answer = await run.observer.interact(
        {
          id,
          title: "OpenClaw needs input",
          sensitive: request.questions.some((question) => question.isSecret),
          schema: {
            type: "object",
            properties: Object.fromEntries(fields.map((field) => [field.key, field.schema])),
            required: fields.filter((field) => field.required).map((field) => field.key),
            additionalProperties: false,
          },
        },
        signal,
      );
      if (signal.aborted || request.expiresAtMs <= Date.now()) return;
      let resolution:
        | { id: string; cancel: true }
        | { id: string; answers: { answers: Record<string, string[]> } };
      if (answer === undefined) resolution = { id, cancel: true };
      else {
        const parsed = z
          .strictObject(Object.fromEntries(fields.map((field) => [field.key, field.validate])))
          .parse(answer);
        const answers = Object.fromEntries(
          request.questions.map((question, index) => {
            const selected = parsed[`choice_${index}`];
            const other = parsed[`other_${index}`];
            const values = [
              ...(Array.isArray(selected) ? selected : selected ? [selected] : []),
              ...(typeof other === "string" ? [other] : []),
            ];
            if (
              !values.length ||
              (!question.multiSelect && values.length !== 1) ||
              values.some((value) => !question.isSecret && !value.trim())
            )
              throw new Error(`Invalid answer for OpenClaw question: ${question.questionId}`);
            return [question.questionId, values];
          }),
        );
        resolution = { id, answers: { answers } };
      }
      questionResolution.parse(await client.request("question.resolve", resolution));
    } finally {
      questions.delete(id);
    }
  }
  async function approve(id: string, observer: Observer, lifetime: AbortSignal) {
    if (approvals.has(id) || lifetime.aborted) return;
    const control = new AbortController();
    approvals.set(id, control);
    try {
      const result: unknown = await client.request("approval.get", { id });
      if (!validateApprovalGetResult(result))
        throw new Error("Invalid OpenClaw approval snapshot.");
      const { approval } = result;
      if (approval.id !== id) throw new Error("OpenClaw approval identity changed.");
      if (
        approval.status !== "pending" ||
        approval.presentation.kind !== "exec" ||
        approval.expiresAtMs <= Date.now() ||
        control.signal.aborted ||
        lifetime.aborted
      )
        return;
      const presentation = approval.presentation;
      const signal = AbortSignal.any([
        control.signal,
        lifetime,
        AbortSignal.timeout(Math.max(0, approval.expiresAtMs - Date.now())),
      ]);
      const answer = await requestApproval(
        observer,
        approval.id,
        [
          presentation.commandText,
          presentation.warningText,
          presentation.scope ? JSON.stringify(presentation.scope) : undefined,
        ]
          .filter(Boolean)
          .join("\n"),
        approval.id,
        presentation.allowedDecisions.map((decision) => ({
          id: decision,
          label: decision,
          kind:
            decision === "deny"
              ? "reject_once"
              : decision === "allow-once"
                ? "allow_once"
                : "allow_always",
          answer: decision,
        })),
        signal,
      );
      if (signal.aborted || approval.expiresAtMs <= Date.now()) return;
      const resolved: unknown = await client.request("approval.resolve", {
        id: approval.id,
        kind: "exec",
        decision: answer ?? "deny",
      });
      if (!validateApprovalResolveResult(resolved))
        throw new Error("Invalid OpenClaw approval resolution.");
    } finally {
      approvals.delete(id);
    }
  }
  let disposed = false;
  let closing: Promise<void> | undefined;
  const fail = (error: Error) => {
    ready.reject(error);
    for (const run of running.values()) {
      run.lifetime.abort();
      run.done.resolve(error);
    }
  };
  const url = process.env.OPENCLAW_GATEWAY_URL ?? "ws://127.0.0.1:18789";
  const client = new GatewayClient({
    url,
    hostDeps: openClawAuth(
      options.productHome ?? process.env.SWARMX_HOME ?? join(homedir(), ".swarmx"),
      url,
    ),
    ...(process.env.OPENCLAW_GATEWAY_TOKEN ? { token: process.env.OPENCLAW_GATEWAY_TOKEN } : {}),
    ...(process.env.OPENCLAW_GATEWAY_PASSWORD
      ? { password: process.env.OPENCLAW_GATEWAY_PASSWORD }
      : {}),
    clientName: GATEWAY_CLIENT_NAMES.GATEWAY_CLIENT,
    clientDisplayName: "SwarmX",
    clientVersion: manifest.version,
    mode: GATEWAY_CLIENT_MODES.BACKEND,
    role: "operator",
    scopes: ["operator.read", "operator.write", "operator.approvals", "operator.questions"],
    caps: [GATEWAY_CLIENT_CAPS.TOOL_EVENTS, GATEWAY_CLIENT_CAPS.EXEC_APPROVALS],
    minProtocol: PROTOCOL_VERSION,
    maxProtocol: PROTOCOL_VERSION,
    onHelloOk: () => ready.resolve(),
    onConnectError: fail,
    onClose: () => {
      if (!disposed) fail(new Error("OpenClaw Gateway disconnected before a terminal outcome."));
    },
    onGap: () => fail(new Error("OpenClaw Gateway event sequence has a gap.")),
    async onEvent(event) {
      let owner: OpenClawRun | undefined;
      try {
        if (event.event === "question.resolved") {
          const { id } = z.object({ id: z.string() }).parse(event.payload);
          questions.get(id)?.abort();
          return;
        }
        if (event.event === "question.requested") {
          const request = z
            .object({
              id: z.string(),
              runId: z.string().optional(),
              sessionKey: z.string().optional(),
            })
            .parse(event.payload);
          const runId = request.runId;
          if (!runId) return;
          const run = [...running.values()].find((run) => run.streams.has(runId));
          const native = run?.streams.get(runId);
          if (!run || !native || native.finished || request.sessionKey !== run.key) return;
          owner = run;
          await ask(
            request.id,
            runId,
            run,
            AbortSignal.any([run.lifetime.signal, native.lifetime.signal]),
          );
          return;
        }
        if (event.event === "exec.approval.resolved") {
          const { id } = z.object({ id: z.string() }).parse(event.payload);
          approvals.get(id)?.abort();
          return;
        }
        if (event.event === "exec.approval.requested") {
          const request = z
            .object({
              id: z.string(),
              request: z.object({ runId: z.string().nullish(), sessionKey: z.string().nullish() }),
            })
            .parse(event.payload);
          const runId = request.request.runId;
          if (!runId) return;
          const run = [...running.values()].find((run) => run.streams.has(runId));
          const native = run?.streams.get(runId);
          if (
            !run ||
            !native ||
            native.finished ||
            (request.request.sessionKey && request.request.sessionKey !== run.key)
          )
            return;
          owner = run;
          run.observer.raw(event, { "swarmx.native.run_id": request.request.runId });
          await approve(
            request.id,
            run.observer,
            AbortSignal.any([run.lifetime.signal, native.lifetime.signal]),
          );
          return;
        }
        if (event.event !== "chat" && event.event !== "agent") return;
        const identity = z.object({ runId: z.string() }).parse(event.payload);
        const run = [...running.values()].find((run) => run.streams.has(identity.runId));
        const native = run?.streams.get(identity.runId);
        if (!run || !native || native.finished) return;
        owner = run;
        const payload = z
          .looseObject({ sessionKey: z.string().optional(), seq: z.number().int().nonnegative() })
          .parse(event.payload);
        if (payload.sessionKey !== undefined && payload.sessionKey !== run.key) return;
        run.observer.raw(event, { "swarmx.native.run_id": identity.runId });
        if (event.event === "chat") {
          const chat = z
            .object({
              state: z.enum(["status", "delta", "final", "aborted", "error"]),
              message: messageSchema.optional(),
              deltaText: z.string().optional(),
              replace: z.boolean().optional(),
              stopReason: z.string().optional(),
              errorMessage: z.string().optional(),
              errorKind: z.string().optional(),
            })
            .parse(payload);
          // Native final snapshots can share the last delta's sequence number.
          if (payload.seq <= native.seq && (chat.state === "delta" || chat.state === "status"))
            return;
          native.seq = payload.seq;
          if (chat.message) {
            const messageId =
              readAssistantStreamSegmentIdentity(chat.message)?.itemId ?? identity.runId;
            if (native.messageId !== messageId) {
              native.messageId = messageId;
              native.text = "";
              native.thought = "";
            }
            const content = chat.message.content;
            if (Array.isArray(content)) {
              const thought = content
                .filter((block) => block.type === "thinking")
                .map((block) => z.string().parse(block.thinking))
                .join("\n");
              if (thought.startsWith(native.thought) && thought.length > native.thought.length)
                run.observer.text(
                  `${native.messageId}:reasoning`,
                  thought.slice(native.thought.length),
                  "reasoning",
                );
              native.thought = thought;
            }
            const text =
              typeof content === "string"
                ? content
                : content
                    .filter((block) => block.type === "text")
                    .map((block) => z.string().parse(block.text))
                    .join("");
            if (!text.startsWith(native.text))
              throw new Error("OpenClaw replaced already-streamed text.");
            if (text.length > native.text.length)
              run.observer.text(native.messageId, text.slice(native.text.length), "assistant");
            native.text = text;
          } else if (chat.state === "delta" && chat.deltaText) {
            if (chat.replace && native.text)
              throw new Error("OpenClaw replaced already-streamed text.");
            native.text += chat.deltaText;
            run.observer.text(native.messageId, chat.deltaText, "assistant");
          }
          if (chat.state === "final" || chat.state === "aborted" || chat.state === "error") {
            finish(
              run,
              native,
              chat.state === "error" && chat.errorKind !== "refusal"
                ? new Error(chat.errorMessage ?? "OpenClaw native run failed.")
                : {
                    stopReason:
                      chat.state === "aborted"
                        ? "cancelled"
                        : chat.errorKind === "refusal"
                          ? "refusal"
                          : chat.stopReason === "max_tokens"
                            ? "max_tokens"
                            : "end_turn",
                  },
            );
          }
        } else {
          if (payload.seq <= native.toolSeq) return;
          native.toolSeq = payload.seq;
          const agentEvent = z
            .object({ stream: z.string(), data: z.record(z.string(), z.unknown()) })
            .parse(payload);
          if (agentEvent.stream === "approval") {
            const request = z
              .object({ approvalId: z.string(), phase: z.string(), kind: z.string() })
              .parse(agentEvent.data);
            if (
              request.phase !== "requested" ||
              request.kind !== "exec" ||
              approvals.has(request.approvalId)
            )
              return;
            await approve(
              request.approvalId,
              run.observer,
              AbortSignal.any([run.lifetime.signal, native.lifetime.signal]),
            );
            return;
          }
          if (agentEvent.stream === "reasoning") {
            const delta = z.object({ delta: z.string() }).parse(agentEvent.data);
            run.observer.text(`${identity.runId}:reasoning`, delta.delta, "reasoning");
          }
          if (agentEvent.stream !== "tool") return;
          const tool = z
            .object({
              toolCallId: z.string(),
              phase: z.enum(["start", "update", "result"]),
              name: z.string().optional(),
              args: z.unknown().optional(),
              result: z.unknown().optional(),
              partialResult: z.unknown().optional(),
              isError: z.boolean().optional(),
            })
            .parse(agentEvent.data);
          const previous = native.tools.get(tool.toolCallId);
          const name = tool.name ?? previous?.name;
          if (!name) throw new Error("OpenClaw tool event has no name.");
          const args = tool.args ?? previous?.args ?? {};
          native.tools.set(tool.toolCallId, { name, args });
          run.observer.tool(
            tool.toolCallId,
            name,
            args,
            tool.phase === "result" ? tool.result : undefined,
          );
          run.observer.activity?.({
            type: "tool",
            toolCallId: tool.toolCallId,
            status:
              tool.phase === "result" ? (tool.isError ? "failed" : "completed") : "in_progress",
          });
        }
      } catch (error) {
        const cause = error instanceof Error ? error : new Error(String(error));
        if (owner) {
          owner.lifetime.abort();
          owner.done.resolve(cause);
        } else fail(cause);
      }
    },
  });
  client.start();
  try {
    await ready.promise;
  } catch (error) {
    disposed = true;
    await client.stopAndWait();
    throw error;
  }
  const available = () => {
    if (disposed || closing) throw new Error("OpenClaw Agent is disposed.");
  };
  const list = async () => {
    available();
    const sessions: z.infer<typeof rowSchema>[] = [];
    let offset = 0;
    while (true) {
      const result = z
        .object({
          sessions: z.array(rowSchema),
          hasMore: z.boolean().optional(),
          nextOffset: z.number().int().nonnegative().nullish(),
        })
        .parse(
          await client.request("sessions.list", { limit: 100, offset, includeDerivedTitles: true }),
        );
      sessions.push(...result.sessions);
      if (!result.hasMore) return sessions;
      if (result.nextOffset == null || result.nextOffset <= offset)
        throw new Error("OpenClaw session pagination did not advance.");
      offset = result.nextOffset;
    }
  };
  const agent: NativeAgent = {
    name: "openclaw",
    capabilities: HARNESS_CAPABILITIES.openclaw,
    async list() {
      return (await list()).flatMap((row) => {
        const title = row.displayName || row.derivedTitle || row.label;
        const session = {
          sessionId: row.key,
          ...(title ? { title } : {}),
          ...(row.updatedAt == null ? {} : { updatedAt: new Date(row.updatedAt).toISOString() }),
        };
        // Previous ACP-created conversations persisted the bridge UUID in the Host journal.
        const legacyId = /(?:^|:)acp-bridge:([0-9a-f-]{36})$/u.exec(row.key)?.[1];
        if (!legacyId || !z.uuid().safeParse(legacyId).success) return [session];
        legacySessions.set(legacyId, row.key);
        return [session, { ...session, sessionId: legacyId }];
      });
    },
    async create() {
      available();
      const result = z.object({ ok: z.literal(true), key: z.string() }).parse(
        await client.request("sessions.create", {
          idempotencyKey: randomUUID(),
          cwd: options.cwd,
        }),
      );
      return result.key;
    },
    async models(id) {
      available();
      const catalog = z
        .object({
          models: z.array(
            z.object({
              id: z.string(),
              provider: z.string(),
              name: z.string(),
              thinkingLevels: z.array(z.object({ id: z.string(), label: z.string() })).optional(),
            }),
          ),
        })
        .parse(await client.request("models.list", {}));
      const session =
        id === undefined
          ? undefined
          : (await list()).find((row) => row.key === (legacySessions.get(id) ?? id));
      if (id !== undefined && !session) throw new Error("OpenClaw session not found.");
      return {
        models: catalog.models.map((model) => ({
          id: modelId(model.provider, model.id),
          name: model.name,
          efforts: (model.thinkingLevels ?? []).map(({ id, label }) => ({ id, name: label })),
        })),
        modes: modeSchema.options.map((id) => ({ id, name: id })),
        current: {
          ...(session?.model ? { model: modelId(session.modelProvider, session.model) } : {}),
          ...(session?.thinkingLevel ? { effort: session.thinkingLevel } : {}),
          ...(session?.permissionMode ? { mode: session.permissionMode } : {}),
        },
      } satisfies ModelCatalog;
    },
    async read(id, observer) {
      available();
      let offset = 0;
      const pages: z.infer<typeof messageSchema>[][] = [];
      while (true) {
        const result = z
          .object({
            messages: z.array(messageSchema),
            hasMore: z.boolean().optional(),
            nextOffset: z.number().int().nonnegative().nullish(),
          })
          .parse(
            await client.request("chat.history", {
              sessionKey: legacySessions.get(id) ?? id,
              limit: 1000,
              offset,
            }),
          );
        pages.unshift(result.messages);
        if (!result.hasMore) break;
        if (result.nextOffset == null || result.nextOffset <= offset)
          throw new Error("OpenClaw history pagination did not advance.");
        offset = result.nextOffset;
      }
      for (const [index, message] of pages.flat().entries()) {
        observer.raw(message);
        const key = message.id ?? `${id}:${index}`;
        if (message.role === "toolResult") {
          const toolCallId = z.string().parse(message.toolCallId);
          observer.activity?.({
            type: "tool",
            toolCallId,
            status: message.isError ? "failed" : "completed",
          });
          observer.tool(toolCallId, message.toolName ?? "tool", {}, message);
          continue;
        }
        if (message.role !== "assistant" && message.role !== "user") continue;
        if (typeof message.content === "string") observer.text(key, message.content, message.role);
        else
          for (const block of message.content) {
            if (block.type === "text")
              observer.text(key, z.string().parse(block.text), message.role);
            if (block.type === "thinking")
              observer.text(`${key}:reasoning`, z.string().parse(block.thinking), "reasoning");
            if (block.type === "toolCall")
              observer.tool(
                z.string().parse(block.id),
                z.string().parse(block.name),
                block.arguments,
              );
          }
      }
    },
    async start(id, text, observer, selection) {
      available();
      if ([...running.values()].some((run) => run.key === (legacySessions.get(id) ?? id)))
        throw new Error("OpenClaw session is busy.");
      const run: OpenClawRun = {
        id: randomUUID(),
        key: legacySessions.get(id) ?? id,
        observer,
        done: Promise.withResolvers<RunResult | Error>(),
        stopped: false,
        submitted: false,
        finished: false,
        result: { stopReason: "end_turn" },
        lifetime: new AbortController(),
        streams: new Map(),
      };
      run.streams.set(run.id, stream(run.id));
      running.set(id, run);
      try {
        if (selection?.mode !== undefined) modeSchema.parse(selection.mode);
        if (selection?.model !== undefined || selection?.mode !== undefined) {
          await client.request("sessions.patch", {
            key: run.key,
            ...(selection.model === undefined ? {} : { model: selection.model }),
            ...(selection.mode === undefined ? {} : { permissionMode: selection.mode }),
          });
        }
        if (run.stopped) return { stopReason: "cancelled" };
        run.submitted = true;
        await send(
          run,
          run.id,
          text + (selection?.instructions ? memoryContextSuffix(selection.instructions) : ""),
          selection?.effort === undefined ? {} : { thinking: selection.effort },
        );
        const result = await run.done.promise;
        if (result instanceof Error) throw result;
        return result;
      } catch (error) {
        if (run.stopped && !run.submitted) return { stopReason: "cancelled" };
        if (run.submitted && !run.finished) {
          try {
            await cancel(run);
          } catch (abortError) {
            throw new AggregateError(
              [error, abortError],
              "OpenClaw run failed and cancellation could not be confirmed.",
            );
          }
        }
        throw error;
      } finally {
        run.lifetime.abort();
        run.done.resolve(new Error("OpenClaw run closed without a terminal outcome."));
        running.delete(id);
      }
    },
    async steer(id, text) {
      const run = running.get(id);
      if (!run?.submitted || run.stopped || run.finished)
        throw new Error("No running OpenClaw session.");
      const runId = randomUUID();
      run.streams.set(runId, stream(runId));
      try {
        await send(run, runId, text, { queueMode: "steer" });
      } catch (error) {
        run.done.resolve(error instanceof Error ? error : new Error(String(error)));
        throw error;
      }
    },
    async interrupt(id) {
      const run = running.get(id);
      if (!run || run.stopped || run.finished) return;
      run.stopped = true;
      run.lifetime.abort();
      if (run.submitted) await cancel(run);
      else run.done.resolve({ stopReason: "cancelled" });
    },
    async dispose() {
      closing ??= (async () => {
        const turns = [...running.values()];
        try {
          await Promise.all([...running.keys()].map((id) => agent.interrupt(id)));
          await Promise.all(turns.map((run) => run.done.promise));
        } finally {
          disposed = true;
          fail(new Error("OpenClaw connection closed without a terminal outcome."));
          await client.stopAndWait();
        }
      })();
      return closing;
    },
  };
  return agent;
}

import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ModelCatalog, RunResult } from "@swarmx/swarm";
import { z } from "zod";
import { requestApproval } from "./approval.js";
import { rpcProcess } from "./rpc-process.js";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

const sessionSchema = z.object({
  session_id: z.string(),
  stored_session_id: z.string().optional(),
  session_key: z.string().optional(),
  info: z
    .object({
      model: z.string().optional(),
      provider: z.string().optional(),
      cwd: z.string().optional(),
      reasoning_effort: z.string().optional(),
      running: z.boolean().optional(),
    })
    .passthrough(),
});
const eventSchema = z.object({
  type: z.string(),
  session_id: z.string().optional(),
  payload: z.record(z.string(), z.unknown()).optional(),
});
const historySchema = z.object({
  messages: z.array(
    z.looseObject({
      role: z.string(),
      text: z.string().optional(),
      name: z.string().optional(),
      args: z.unknown().optional(),
      row_id: z.number().optional(),
      reasoning: z.string().optional(),
      reasoning_content: z.string().optional(),
    }),
  ),
});
const effortSchema = z.enum(["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]);
const questionSchema = z.object({
  question: z.string(),
  choices: z.array(z.string()),
  multi_select: z.boolean().optional(),
});
const catalogSchema = z.object({
  model: z.string(),
  provider: z.string(),
  providers: z.array(
    z.object({
      slug: z.string(),
      name: z.string(),
      models: z.array(z.string()),
      capabilities: z
        .record(
          z.string(),
          z
            .object({ reasoning: z.boolean(), can_disable_reasoning: z.boolean().optional() })
            .passthrough(),
        )
        .optional(),
    }),
  ),
});
const modelId = (provider: string | undefined, model: string) =>
  provider ? `${provider}:${model}` : model;

function hermesPython() {
  if (process.env.SWARMX_HERMES_PYTHON) return process.env.SWARMX_HERMES_PYTHON;
  for (const directory of (process.env.PATH ?? "").split(delimiter)) {
    const path = join(directory, "hermes");
    if (!existsSync(path)) continue;
    const interpreter = /^#!(\/[^\s]+)\s*$/u.exec(
      readFileSync(path, "utf8").split("\n")[0] ?? "",
    )?.[1];
    if (interpreter) return interpreter;
    break;
  }
  throw new Error(
    "Set SWARMX_HERMES_PYTHON to the Python interpreter of your Hermes installation.",
  );
}

export async function createHermes(options: AgentOptions): Promise<NativeAgent> {
  if (options.reviewOnly) throw new Error("Hermes does not support restricted memory reviews.");
  const python = hermesPython();
  const fresh = new Map<string, ReturnType<typeof open>>();
  const opened = new Set<ReturnType<typeof open>>();
  const running = new Map<
    string,
    { peer?: ReturnType<typeof open>; stopped: boolean; submitted: boolean }
  >();
  let disposed = false;
  const available = () => {
    if (disposed) throw new Error("Hermes Agent is disposed.");
  };

  function open(withTools = false) {
    const token = randomUUID();
    const endpoint = withTools ? options.registerMcp?.(token) : undefined;
    let session: (z.infer<typeof sessionSchema> & { stored_session_id: string }) | undefined;
    let observer: Observer | undefined;
    let messageId = randomUUID();
    let text = "";
    let accepting = false;
    let terminal: RunResult | Error | undefined;
    let acknowledged = false;
    let requiredTerminals = 1;
    let completedTerminals = 0;
    const done = Promise.withResolvers<RunResult | Error>();
    let idle: Promise<unknown> | undefined;
    let steering = 0;
    const settle = () => {
      if (
        !acknowledged ||
        steering > 0 ||
        completedTerminals < requiredTerminals ||
        !terminal ||
        session?.info.running !== false
      )
        return;
      idle ??= rpc.request("swarmx.session.wait", { session_id: session.session_id });
      void idle
        .then((response) => {
          if (steering > 0) return;
          if (
            z.object({ running: z.boolean() }).parse(response).running ||
            !terminal ||
            completedTerminals < requiredTerminals ||
            session?.info.running !== false
          )
            throw new Error("Hermes execution did not settle with a terminal outcome.");
          done.resolve(terminal);
        })
        .catch((error) => done.resolve(error));
    };
    const prompts = new Map<string, AbortController>();
    const tools = new Map<string, { name: string; args: unknown }>();
    const earlyEvents: (() => Promise<unknown>)[] = [];
    const rpc = rpcProcess(
      python,
      [fileURLToPath(new URL("../../resources/hermes-native.py", import.meta.url))],
      options.cwd,
      async function receive(request) {
        if (request.method !== "event")
          throw new Error(`Unknown Hermes request: ${request.method}`);
        const event = eventSchema.parse(request.params);
        if (!session) {
          earlyEvents.push(() => receive(request));
          return;
        }
        if (event.session_id !== session.session_id) return;
        const data = event.payload ?? {};
        if (event.type === "session.info") session.info = sessionSchema.shape.info.parse(data);
        if (!observer || !accepting) return;
        observer.raw(request);
        if (event.type === "session.info") {
          settle();
          return;
        }
        if (event.type.endsWith(".expire")) {
          const { request_id: id } = z.object({ request_id: z.string() }).parse(data);
          prompts.get(id)?.abort();
          prompts.delete(id);
          return;
        }
        switch (event.type) {
          case "message.start":
            messageId = randomUUID();
            text = "";
            terminal = undefined;
            session.info.running = true;
            break;
          case "message.delta": {
            const delta = z.string().parse(data.text);
            text += delta;
            observer.text(messageId, delta, "assistant");
            break;
          }
          case "message.interim": {
            const interim = z
              .object({ text: z.string(), already_streamed: z.boolean() })
              .parse(data);
            if (!interim.already_streamed) observer.text(messageId, interim.text, "assistant");
            messageId = randomUUID();
            text = "";
            break;
          }
          case "thinking.delta":
          case "reasoning.delta":
            observer.text(`${messageId}:reasoning`, z.string().parse(data.text), "reasoning");
            break;
          case "tool.start":
          case "tool.progress":
          case "tool.complete": {
            const tool = z
              .object({
                tool_id: z.string(),
                name: z.string().optional(),
                args: z.unknown().optional(),
                result: z.unknown().optional(),
              })
              .parse(data);
            const previous = tools.get(tool.tool_id);
            const name = tool.name ?? previous?.name;
            if (!name) throw new Error("Hermes tool event has no tool name.");
            const args = tool.args ?? previous?.args ?? {};
            tools.set(tool.tool_id, { name, args });
            observer.activity?.({
              type: "tool",
              toolCallId: tool.tool_id,
              status:
                event.type !== "tool.complete"
                  ? "in_progress"
                  : z.object({ error: z.string().min(1) }).safeParse(tool.result).success
                    ? "failed"
                    : "completed",
            });
            observer.tool(
              tool.tool_id,
              name,
              args,
              event.type === "tool.complete" ? data : undefined,
            );
            break;
          }
          case "approval.request": {
            const approval = z
              .object({
                request_id: z.string(),
                command: z.string().optional(),
                description: z.string().optional(),
                choices: z.array(z.enum(["once", "session", "always", "deny"])),
              })
              .parse(data);
            const control = new AbortController();
            prompts.set(approval.request_id, control);
            try {
              const choice = await requestApproval(
                observer,
                approval.request_id,
                approval.command ?? approval.description ?? "Hermes approval",
                approval.request_id,
                approval.choices.map((choice) => ({
                  id: choice,
                  label: choice,
                  kind:
                    choice === "deny"
                      ? "reject_once"
                      : choice === "once"
                        ? "allow_once"
                        : "allow_always",
                  answer: choice,
                })),
                AbortSignal.any([control.signal, rpc.signal]),
              );
              if (!control.signal.aborted && !rpc.signal.aborted) {
                const result = z.object({ resolved: z.boolean() }).parse(
                  await rpc.request("approval.respond", {
                    session_id: event.session_id,
                    request_id: approval.request_id,
                    choice: choice ?? "deny",
                  }),
                );
                if (!result.resolved)
                  throw new Error("Hermes did not resolve the approval request.");
              }
            } finally {
              prompts.delete(approval.request_id);
            }
            break;
          }
          case "clarify.request": {
            const input = z
              .union([
                questionSchema.extend({ request_id: z.string() }),
                z.object({
                  request_id: z.string(),
                  questions: z.array(questionSchema.extend({ qid: z.string() })).min(1),
                }),
              ])
              .parse(data);
            const questions = "questions" in input ? input.questions : [input];
            const fields = questions.flatMap((question, index) => {
              const choices = { type: "string", enum: question.choices };
              return [
                ...(question.choices.length
                  ? [
                      {
                        key: `answer_${index}`,
                        schema: {
                          ...(question.multi_select ? { type: "array", items: choices } : choices),
                          title: question.question,
                        },
                        validate: (question.multi_select
                          ? z.array(z.enum(question.choices))
                          : z.enum(question.choices)
                        ).optional(),
                      },
                    ]
                  : []),
                {
                  key: `other_${index}`,
                  schema: {
                    type: "string",
                    title: question.choices.length
                      ? `${question.question} — Other answer`
                      : question.question,
                  },
                  validate: z.string().optional(),
                },
              ];
            });
            const control = new AbortController();
            prompts.set(input.request_id, control);
            try {
              const answer = await observer.interact(
                {
                  id: input.request_id,
                  title: "Hermes needs input",
                  schema: {
                    type: "object",
                    properties: Object.fromEntries(
                      fields.map((field) => [field.key, field.schema]),
                    ),
                    additionalProperties: false,
                  },
                },
                AbortSignal.any([control.signal, rpc.signal]),
              );
              if (control.signal.aborted || rpc.signal.aborted) break;
              const params = { session_id: event.session_id, request_id: input.request_id };
              if (answer === undefined) {
                await rpc.request("clarify.respond", { ...params, answer: "" });
                break;
              }
              const parsed = z
                .strictObject(
                  Object.fromEntries(fields.map((field) => [field.key, field.validate])),
                )
                .parse(answer);
              // Validate the whole form before locking any native batch answer.
              const answers = questions.map((question, index) => {
                const selected = parsed[`answer_${index}`];
                const other = parsed[`other_${index}`];
                const values = [
                  ...(Array.isArray(selected) ? selected : selected ? [selected] : []),
                  ...(other ? [other] : []),
                ];
                if (!values.length)
                  throw new Error(`Hermes question requires an answer: ${question.question}`);
                return {
                  ...params,
                  ...("qid" in question ? { question_id: question.qid } : {}),
                  answer: question.multi_select ? JSON.stringify(values) : values.join(", "),
                };
              });
              for (const response of answers) {
                if (control.signal.aborted || rpc.signal.aborted) break;
                const result = z
                  .object({ status: z.enum(["ok", "expired"]) })
                  .parse(await rpc.request("clarify.respond", response));
                if (result.status === "expired") break;
              }
            } finally {
              prompts.delete(input.request_id);
            }
            break;
          }
          case "sudo.request":
          case "secret.request": {
            const input = z
              .object({
                request_id: z.string(),
                question: z.string().optional(),
                prompt: z.string().optional(),
                choices: z.array(z.string()).optional(),
              })
              .parse(data);
            const control = new AbortController();
            prompts.set(input.request_id, control);
            try {
              const answer = await observer.interact(
                {
                  id: input.request_id,
                  title: input.question ?? input.prompt ?? event.type,
                  sensitive: true,
                  schema: {
                    type: "object",
                    properties: {
                      answer: {
                        type: "string",
                        format: "password",
                      },
                    },
                    required: ["answer"],
                    additionalProperties: false,
                  },
                },
                AbortSignal.any([control.signal, rpc.signal]),
              );
              if (!control.signal.aborted && !rpc.signal.aborted)
                await rpc.request(event.type.replace(".request", ".respond"), {
                  session_id: event.session_id,
                  request_id: input.request_id,
                  [event.type === "sudo.request" ? "password" : "value"]:
                    answer === undefined
                      ? ""
                      : z.strictObject({ answer: z.string() }).parse(answer).answer,
                });
            } finally {
              prompts.delete(input.request_id);
            }
            break;
          }
          case "message.complete": {
            const result = z
              .object({
                text: z.string(),
                status: z.enum(["complete", "interrupted", "error"]),
                error: z.string().optional(),
              })
              .parse(data);
            completedTerminals += 1;
            if (result.status === "error") {
              terminal = new Error(result.error ?? result.text);
              break;
            }
            if (result.text.startsWith(text)) {
              if (result.text.length > text.length)
                observer.text(messageId, result.text.slice(text.length), "assistant");
            } else if (result.text) observer.text(randomUUID(), result.text, "assistant");
            terminal = {
              stopReason: result.status === "interrupted" ? "cancelled" : "end_turn",
            };
            break;
          }
          case "error":
            done.resolve(new Error(z.string().parse(data.message)));
            break;
        }
      },
      (error) => done.resolve(error),
      {
        ...process.env,
        SWARMX_HERMES_MCP: JSON.stringify(
          withTools
            ? {
                swarmx: {
                  command: options.mcp.command,
                  args: [...options.mcp.args],
                  env: endpoint ? { ...options.mcp.env, SWARMX_MCP_TOKEN: token } : options.mcp.env,
                },
              }
            : {},
        ),
      },
    );
    const peer = {
      rpc,
      done,
      get session() {
        return session;
      },
      observe(value: Observer) {
        observer = value;
        accepting = true;
      },
      async submit(text: string) {
        if (!session) throw new Error("No Hermes session.");
        const priorTurnEnded = terminal !== undefined && session.info.running !== false;
        const observed = completedTerminals;
        requiredTerminals = observed + 1;
        const result = z.object({ status: z.string() }).parse(
          await rpc.request("prompt.submit", {
            session_id: session.session_id,
            text,
          }),
        );
        if (result.status === "queued") {
          // Native busy-submit can interrupt/finish the resumed turn before draining this input.
          if (!priorTurnEnded) requiredTerminals += 1;
        }
        acknowledged = true;
        settle();
      },
      async steer(text: string) {
        if (!session || terminal) throw new Error("No running Hermes session.");
        steering += 1;
        try {
          const result = z
            .object({ status: z.enum(["queued", "rejected"]) })
            .parse(await rpc.request("session.steer", { session_id: session.session_id, text }));
          if (result.status !== "queued") throw new Error("Hermes rejected steering input.");
        } finally {
          steering -= 1;
          settle();
        }
      },
      async interrupt() {
        if (!session) throw new Error("No Hermes session.");
        await rpc.request("session.interrupt", { session_id: session.session_id });
        requiredTerminals = 0;
        settle();
      },
      bind(id: string, value: Observer) {
        if (endpoint) {
          if (!value.executionId)
            throw new Error("Hermes tool access requires a bound Host execution.");
          endpoint.bind(`hermes:${id}`, value.executionId);
        }
      },
      async connectSession(id?: string, lazy = false) {
        const response = sessionSchema.parse(
          await rpc.request(
            id === undefined ? "session.create" : "session.resume",
            id === undefined ? { cwd: options.cwd } : { session_id: id, ...(lazy ? { lazy } : {}) },
          ),
        );
        session = {
          ...response,
          stored_session_id: z
            .string()
            .parse(id === undefined ? response.stored_session_id : response.session_key),
        };
        for (const receive of earlyEvents.splice(0)) await receive();
        return session;
      },
      async close() {
        if (!opened.delete(peer)) return;
        accepting = false;
        for (const controller of prompts.values()) controller.abort();
        endpoint?.dispose();
        await rpc.dispose();
        done.resolve(new Error("Hermes runtime closed without a terminal outcome."));
      },
    };
    opened.add(peer);
    return peer;
  }

  return {
    name: "hermes",
    capabilities: HARNESS_CAPABILITIES.hermes,
    async list() {
      available();
      const peer = open();
      try {
        const { sessions } = z
          .object({
            sessions: z.array(
              z.object({
                id: z.string(),
                title: z.string(),
                preview: z.string(),
                started_at: z.number(),
              }),
            ),
          })
          .parse(await peer.rpc.request("session.list", { limit: Number.MAX_SAFE_INTEGER }));
        return [
          ...sessions.map((row) => ({
            sessionId: row.id,
            title: row.title || row.preview,
            updatedAt: new Date(row.started_at * 1000).toISOString(),
          })),
          ...[...fresh.keys()]
            .filter((id) => !sessions.some((row) => row.id === id))
            .map((sessionId) => ({ sessionId })),
        ];
      } finally {
        await peer.close();
      }
    },
    async create() {
      available();
      const peer = open(true);
      try {
        const session = await peer.connectSession();
        fresh.set(session.stored_session_id, peer);
        return session.stored_session_id;
      } catch (error) {
        await peer.close();
        throw error;
      }
    },
    async models(id) {
      available();
      const existing = id === undefined ? undefined : (fresh.get(id) ?? running.get(id)?.peer);
      const peer = existing ?? open();
      try {
        const session =
          id === undefined
            ? undefined
            : (existing?.session ?? (await peer.connectSession(id, true)));
        const params = session ? { session_id: session.session_id } : {};
        const data = catalogSchema.parse(await peer.rpc.request("model.options", params));
        const effort = z
          .object({ value: z.string() })
          .parse(await peer.rpc.request("config.get", { ...params, key: "reasoning" }));
        return {
          models: data.providers.flatMap((provider) =>
            provider.models.map((model) => ({
              id: modelId(provider.slug, model),
              name: model,
              description: provider.name,
              efforts:
                provider.capabilities?.[model]?.reasoning === false
                  ? []
                  : effortSchema.options
                      .filter(
                        (id) =>
                          id !== "none" ||
                          provider.capabilities?.[model]?.can_disable_reasoning !== false,
                      )
                      .map((id) => ({ id, name: id })),
            })),
          ),
          current:
            id !== undefined && !existing
              ? {}
              : { model: modelId(data.provider, data.model), effort: effort.value },
        } satisfies ModelCatalog;
      } finally {
        if (!existing) await peer.close();
      }
    },
    async read(id, observer) {
      available();
      if (fresh.has(id)) return;
      const existing = running.get(id)?.peer;
      const peer = existing ?? open();
      try {
        const session = existing?.session ?? (await peer.connectSession(id, true));
        const history = historySchema.parse(
          await peer.rpc.request("session.history", { session_id: session.session_id }),
        );
        for (const [index, row] of history.messages.entries()) {
          observer.raw(row);
          const key = `${id}:${row.row_id ?? index}`;
          if (row.role === "assistant" || row.role === "user") {
            if (row.text) observer.text(key, row.text, row.role);
            const reasoning = row.reasoning ?? row.reasoning_content;
            if (reasoning) observer.text(`${key}:reasoning`, reasoning, "reasoning");
          }
          if (row.role === "tool") observer.tool(key, row.name ?? "tool", row.args ?? {}, row);
        }
      } finally {
        if (!existing) await peer.close();
      }
    },
    async start(id, text, observer, selection) {
      available();
      if (selection?.mode !== undefined)
        throw new Error(
          "Hermes TUI Gateway uses its native approval settings; ACP edit modes are unsupported.",
        );
      if (running.has(id)) throw new Error("Hermes session is busy.");
      const run: { peer?: ReturnType<typeof open>; stopped: boolean; submitted: boolean } = {
        stopped: false,
        submitted: false,
      };
      running.set(id, run);
      try {
        const peer = fresh.get(id) ?? open(true);
        run.peer = peer;
        peer.bind(id, observer);
        peer.observe(observer);
        const session = peer.session ?? (await peer.connectSession(id));
        fresh.delete(id);
        if (run.stopped) return { stopReason: "cancelled" };
        for (const [key, value] of [
          ["model", selection?.model],
          ["reasoning", selection?.effort],
        ] as const) {
          if (value === undefined) continue;
          if (key === "reasoning") effortSchema.parse(value);
          const result = z
            .object({
              confirm_required: z.boolean().optional(),
              confirm_message: z.string().optional(),
            })
            .passthrough()
            .parse(
              await peer.rpc.request("config.set", {
                session_id: session.session_id,
                key,
                value,
                scope: "session",
              }),
            );
          if (result.confirm_required) {
            const answer = await observer.interact(
              {
                id: randomUUID(),
                title: result.confirm_message ?? "Confirm Hermes model selection",
                schema: {
                  type: "object",
                  properties: { confirm: { type: "boolean", title: "Use this model" } },
                  required: ["confirm"],
                  additionalProperties: false,
                },
              },
              peer.rpc.signal,
            );
            if (run.stopped) return { stopReason: "cancelled" };
            if (
              answer === undefined ||
              !z.strictObject({ confirm: z.boolean() }).parse(answer).confirm
            )
              throw new Error("Hermes model selection declined.");
            const confirmed = z.object({ confirm_required: z.boolean().optional() }).parse(
              await peer.rpc.request("config.set", {
                session_id: session.session_id,
                key,
                value,
                scope: "session",
                confirm_expensive_model: true,
              }),
            );
            if (confirmed.confirm_required)
              throw new Error("Hermes did not accept model confirmation.");
          }
        }
        if (selection?.model !== undefined) {
          const applied = catalogSchema.parse(
            await peer.rpc.request("model.options", { session_id: session.session_id }),
          );
          const actual = modelId(applied.provider, applied.model);
          if (actual !== selection.model)
            throw new Error(`Hermes model changed from "${selection.model}" to "${actual}".`);
        }
        if (selection?.effort !== undefined) {
          const applied = z.object({ value: z.string() }).parse(
            await peer.rpc.request("config.get", {
              session_id: session.session_id,
              key: "reasoning",
            }),
          );
          if (applied.value !== selection.effort)
            throw new Error(
              `Hermes effort changed from "${selection.effort}" to "${applied.value}".`,
            );
        }
        if (run.stopped) return { stopReason: "cancelled" };
        run.submitted = true;
        await peer.submit(
          text + (selection?.instructions ? memoryContextSuffix(selection.instructions) : ""),
        );
        const result = await peer.done.promise;
        if (result instanceof Error) throw result;
        return result;
      } catch (error) {
        if (run.stopped && run.peer?.rpc.signal.aborted) return { stopReason: "cancelled" };
        throw error;
      } finally {
        await run.peer?.close();
        running.delete(id);
      }
    },
    async steer(id, text) {
      const run = running.get(id);
      if (!run?.submitted || run.stopped || !run.peer?.session)
        throw new Error("No running Hermes session.");
      await run.peer.steer(text);
    },
    async interrupt(id) {
      const run = running.get(id);
      if (!run || run.stopped) return;
      run.stopped = true;
      if (run.submitted && run.peer?.session) await run.peer.interrupt();
      else await run.peer?.close();
    },
    async dispose() {
      disposed = true;
      for (const run of running.values()) run.stopped = true;
      await Promise.all([...opened].map((peer) => peer.close()));
      fresh.clear();
    },
  };
}

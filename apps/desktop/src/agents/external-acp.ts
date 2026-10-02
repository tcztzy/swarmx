import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { isAbsolute } from "node:path";
import { Readable, Writable } from "node:stream";
import * as acp from "@agentclientprotocol/sdk";
import type { ModelCatalog, RunResult } from "@swarmx/swarm";
import { z } from "zod";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  type NativeAgent,
  type NativeRunOptions,
  type Observer,
} from "./types.js";

const HOST_CREDENTIAL_KEYS = new Set(["SWARMX_API_TOKEN", "SWARMX_MCP_TOKEN"]);

const EndpointSchema = z.strictObject({
  id: z.string().regex(/^[a-z][a-z0-9_-]*$/u),
  command: z.string().refine(isAbsolute, "ACP command must be an absolute executable path."),
  args: z.array(z.string()).default([]),
  env: z
    .record(z.string(), z.string())
    .refine(
      (env) => Object.keys(env).every((name) => !HOST_CREDENTIAL_KEYS.has(name.toUpperCase())),
      "ACP endpoint env cannot supply Host credentials.",
    )
    .default({}),
});
type Endpoint = z.infer<typeof EndpointSchema>;
type Settings = Pick<acp.LoadSessionResponse, "configOptions" | "modes">;
type Select = Extract<acp.SessionConfigOption, { type: "select" }>;
type Run = {
  observer: Observer;
  tools: Map<string, { name: string; title: string | null | undefined; input: unknown }>;
  abort: AbortController;
  done: Promise<void>;
  finish(): void;
  dispatched: boolean;
  runtime?: Runtime;
};

function selector(settings: Settings, category: string): Select | undefined {
  const matches = settings.configOptions?.filter((option) => option.category === category) ?? [];
  if (matches.length > 1) throw new Error(`ACP advertised ambiguous ${category} selectors.`);
  const value = matches[0];
  return value?.type === "select" ? value : undefined;
}
function values(option?: Select): acp.SessionConfigSelectOption[] {
  return option?.options.flatMap((value) => ("group" in value ? value.options : [value])) ?? [];
}
function catalog(settings: Settings): ModelCatalog {
  const model = selector(settings, "model");
  const effort = selector(settings, "thought_level");
  const mode = selector(settings, "mode");
  return {
    models: values(model).map((value) => ({
      id: value.value,
      name: value.name,
      ...(value.description ? { description: value.description } : {}),
      efforts: values(effort).map((choice) => ({ id: choice.value, name: choice.name })),
      ...(effort ? { defaultEffort: effort.currentValue } : {}),
    })),
    ...(mode
      ? { modes: values(mode).map((value) => ({ id: value.value, name: value.name })) }
      : settings.modes
        ? { modes: settings.modes.availableModes.map(({ id, name }) => ({ id, name })) }
        : {}),
    current: {
      ...(model ? { model: model.currentValue } : {}),
      ...(effort ? { effort: effort.currentValue } : {}),
      ...(mode
        ? { mode: mode.currentValue }
        : settings.modes
          ? { mode: settings.modes.currentModeId }
          : {}),
    },
  };
}

/** Only translates observations. Native history is never used to synthesize a new remote session. */
function projection(observer: Observer, tools: Run["tools"] = new Map()) {
  let message = { kind: "", id: randomUUID() };
  return (notification: acp.SessionNotification) => {
    observer.raw(notification);
    const update = notification.update;
    if (
      update.sessionUpdate === "user_message_chunk" ||
      update.sessionUpdate === "agent_message_chunk" ||
      update.sessionUpdate === "agent_thought_chunk"
    ) {
      if (message.kind !== update.sessionUpdate)
        message = { kind: update.sessionUpdate, id: randomUUID() };
      if (update.content.type === "text")
        observer.text(
          update.messageId ?? message.id,
          update.content.text,
          update.sessionUpdate === "user_message_chunk"
            ? "user"
            : update.sessionUpdate === "agent_thought_chunk"
              ? "reasoning"
              : "assistant",
        );
    } else if (
      update.sessionUpdate === "tool_call" ||
      update.sessionUpdate === "tool_call_update"
    ) {
      message = { kind: "", id: randomUUID() };
      const previous = tools.get(update.toolCallId);
      const tool = {
        name: update.name ?? previous?.name ?? update.title ?? update.toolCallId,
        title: update.title === undefined ? previous?.title : update.title,
        input: update.rawInput === undefined ? previous?.input : update.rawInput,
      };
      tools.set(update.toolCallId, tool);
      observer.tool(
        update.toolCallId,
        tool.name,
        tool.input,
        update.rawOutput ?? (update.content?.length ? update.content : undefined),
      );
      observer.activity?.({
        type: "tool",
        toolCallId: update.toolCallId,
        ...(update.kind ? { kind: update.kind } : {}),
        ...(update.status ? { status: update.status } : {}),
      });
    }
  };
}

class Runtime {
  history = false;
  readonly settings = new Map<string, Settings>();
  readonly observers = new Map<string, ReturnType<typeof projection>>();
  readonly connection: acp.ClientConnection;
  readonly ready: Promise<void>;
  readonly exited: Promise<void>;
  private disposing?: Promise<void>;
  private dead = false;
  private readonly child;

  constructor(
    readonly cwd: string,
    endpoint: Endpoint,
    runs: Map<string, Run>,
  ) {
    const detached = process.platform !== "win32";
    const inheritedEnv = Object.fromEntries(
      Object.entries(process.env).filter(([name]) => !HOST_CREDENTIAL_KEYS.has(name.toUpperCase())),
    );
    this.child = spawn(endpoint.command, endpoint.args, {
      cwd,
      env: { ...inheritedEnv, ...endpoint.env, ELECTRON_RUN_AS_NODE: "1" },
      stdio: ["pipe", "pipe", "inherit"],
      detached,
    });
    this.exited = new Promise((resolve) => this.child.once("close", () => resolve()));
    this.connection = acp
      .client()
      .onNotification("session/update", ({ params }) => {
        const update = params.update;
        const state = this.settings.get(params.sessionId);
        if (state && update.sessionUpdate === "config_option_update")
          state.configOptions = update.configOptions;
        if (state?.modes && update.sessionUpdate === "current_mode_update")
          state.modes.currentModeId = update.currentModeId;
        this.observers.get(params.sessionId)?.(params);
      })
      .onRequest("session/request_permission", async ({ params }) => {
        const run = runs.get(`${endpoint.id}:${params.sessionId}`);
        const cancelled: acp.RequestPermissionResponse = { outcome: { outcome: "cancelled" } };
        if (!run || run.runtime !== this || !run.dispatched || run.abort.signal.aborted)
          return cancelled;
        const observed = run.tools.get(params.toolCall.toolCallId);
        const title = params.toolCall.title === undefined ? observed?.title : params.toolCall.title;
        const input =
          params.toolCall.rawInput === undefined ? observed?.input : params.toolCall.rawInput;
        const { promise, resolve } = Promise.withResolvers<undefined>();
        const abort = () => resolve(undefined);
        run.abort.signal.addEventListener("abort", abort, { once: true });
        try {
          const answer = await Promise.race([
            promise,
            run.observer.interact(
              {
                id: randomUUID(),
                title: title ?? "External agent permission",
                schema: {
                  type: "object",
                  properties: {
                    optionId: {
                      type: "string",
                      oneOf: params.options.map((option) => ({
                        const: option.optionId,
                        title: `${option.name} (${option.kind})`,
                      })),
                    },
                  },
                  required: ["optionId"],
                  additionalProperties: false,
                },
                approval: {
                  toolId: params.toolCall.toolCallId,
                  ...(input === undefined ? {} : { input }),
                  choices: params.options.map((option) => ({
                    id: option.optionId,
                    label: option.name,
                    kind: option.kind,
                    answer: { optionId: option.optionId },
                  })),
                },
              },
              run.abort.signal,
            ),
          ]);
          if (run.abort.signal.aborted) return cancelled;
          const selected = z.strictObject({ optionId: z.string() }).safeParse(answer);
          if (
            !selected.success ||
            !params.options.some((option) => option.optionId === selected.data.optionId)
          )
            return cancelled;
          return { outcome: { outcome: "selected", optionId: selected.data.optionId } };
        } finally {
          run.abort.signal.removeEventListener("abort", abort);
        }
      })
      .connect(
        acp.ndJsonStream(Writable.toWeb(this.child.stdin), Readable.toWeb(this.child.stdout)),
      );
    const fail = (error: Error) => {
      this.dead = true;
      this.connection.close(error);
    };
    this.child.once("error", fail);
    this.child.stdin.on("error", fail);
    this.child.once("exit", (code, signal) =>
      fail(new Error(`External ACP agent exited (${signal ?? code}).`)),
    );
    this.ready = this.connection.agent
      .request("initialize", {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientInfo: { name: "swarmx", version: "1" },
        clientCapabilities: {},
      })
      .then((response) => {
        if (response.protocolVersion !== acp.PROTOCOL_VERSION)
          throw new Error("External ACP protocol version is unsupported.");
        const capabilities = response.agentCapabilities;
        if (!capabilities?.sessionCapabilities?.list || !capabilities.sessionCapabilities.resume)
          throw new Error("External ACP agent must advertise native list and resume support.");
        this.history = capabilities.loadSession === true;
      });
  }
  get closed() {
    return this.dead || this.connection.signal.aborted;
  }
  dispose(): Promise<void> {
    if (this.disposing) return this.disposing;
    this.disposing = (async () => {
      this.connection.close();
      this.child.stdin.end();
      const terminate = (signal: NodeJS.Signals) => {
        try {
          if (process.platform !== "win32" && this.child.pid !== undefined)
            process.kill(-this.child.pid, signal);
          else this.child.kill(signal);
        } catch {
          /* Owned process group has already exited. */
        }
      };
      const graceful = setTimeout(() => terminate("SIGTERM"), 500);
      const force = setTimeout(() => terminate("SIGKILL"), 3000);
      graceful.unref();
      force.unref();
      try {
        await this.exited;
      } finally {
        clearTimeout(graceful);
        clearTimeout(force);
      }
      this.dead = true;
    })();
    return this.disposing;
  }
}

export async function createExternalAcp(options: AgentOptions): Promise<NativeAgent> {
  const path = process.env.SWARMX_ACP_AGENT;
  if (!path || !isAbsolute(path))
    throw new Error("Set SWARMX_ACP_AGENT to an absolute endpoint JSON path.");
  const endpoint = EndpointSchema.parse(JSON.parse(await readFile(path, "utf8")));
  const runtimes = new Map<string, Runtime>();
  const runs = new Map<string, Run>();
  const readers = new Set<string>();
  let closed = false;
  let disposing: Promise<void> | undefined;
  const assertOpen = () => {
    if (closed) throw new Error("External ACP agent is closed.");
  };
  const assertAdmission = () => {
    const allowed = options.executionPolicy?.().harnesses?.acp;
    if (allowed === undefined || allowed?.length === 0)
      throw new Error(
        'Harness "acp" requires an explicit Host policy grant before process launch.',
      );
  };
  const nativeId = (id: string) => {
    if (!id.startsWith(`${endpoint.id}:`) || id.length === endpoint.id.length + 1)
      throw new Error("Session does not belong to this external ACP endpoint.");
    return id.slice(endpoint.id.length + 1);
  };
  const runtime = async () => {
    assertOpen();
    assertAdmission();
    const cwd = await realpath(options.cwd);
    assertOpen();
    assertAdmission();
    let current = runtimes.get(cwd);
    if (current?.closed) {
      await current.dispose();
      assertOpen();
      assertAdmission();
      if (runtimes.get(cwd) === current) runtimes.delete(cwd);
      current = runtimes.get(cwd);
    }
    if (!current) {
      assertOpen();
      assertAdmission();
      current = new Runtime(cwd, endpoint, runs);
      runtimes.set(cwd, current);
    }
    try {
      await current.ready;
      assertOpen();
      return current;
    } catch (error) {
      await current.dispose();
      throw error;
    }
  };
  const stopped = (run?: Run) => closed || run?.abort.signal.aborted;
  const resume = async (connection: Runtime, id: string, run?: Run) => {
    if (!connection.settings.has(id) && !stopped(run)) {
      const settings = await connection.connection.agent.request("session/resume", {
        sessionId: id,
        cwd: connection.cwd,
        mcpServers: [],
      });
      connection.settings.set(id, settings);
    }
    return connection.settings.get(id) ?? {};
  };
  const unsupported = (
    selection?: Pick<NativeRunOptions, "instructions" | "permissions"> & Partial<NativeRunOptions>,
  ) => {
    if (selection?.instructions)
      throw new Error(
        "External ACP owns its system instructions; Host instruction injection is unsupported.",
      );
    if (selection?.permissions !== undefined)
      throw new Error("Host permissions cannot be forwarded as external native permissions.");
    if (selection?.profile !== undefined || selection?.budgetUsd !== undefined)
      throw new Error("External ACP does not advertise Host profiles or budgets.");
  };
  const interrupt = async (id: string) => {
    nativeId(id);
    const run = runs.get(id);
    if (!run) return;
    run.abort.abort();
    if (run.dispatched && run.runtime) {
      await run.runtime.connection.agent.notify("session/cancel", { sessionId: nativeId(id) });
      await run.done;
    }
  };
  return {
    name: `ACP (${endpoint.id})`,
    capabilities: {
      ...HARNESS_CAPABILITIES.acp,
      get history() {
        return (
          runtimes.size > 0 &&
          [...runtimes.values()].every((current) => !current.closed && current.history)
        );
      },
    },
    async models(id) {
      assertOpen();
      assertAdmission();
      if (id === undefined) return { models: [], current: {} };
      const key = nativeId(id);
      const connection = await runtime();
      return catalog(await resume(connection, key));
    },
    async list() {
      const connection = await runtime();
      const sessions = [];
      let cursor: string | undefined;
      const cursors = new Set<string>();
      do {
        const page = await connection.connection.agent.request("session/list", {
          cwd: connection.cwd,
          ...(cursor ? { cursor } : {}),
        });
        for (const session of page.sessions) {
          if ((await realpath(session.cwd)) !== connection.cwd) continue;
          sessions.push({
            sessionId: `${endpoint.id}:${session.sessionId}`,
            ...(session.title ? { title: session.title } : {}),
          });
        }
        cursor = page.nextCursor ?? undefined;
        if (cursor && cursors.has(cursor))
          throw new Error("External ACP session list repeated its cursor.");
        if (cursor) cursors.add(cursor);
      } while (cursor);
      return sessions;
    },
    async create(selection) {
      unsupported(selection);
      const connection = await runtime();
      const response = await connection.connection.agent.request("session/new", {
        cwd: connection.cwd,
        mcpServers: [],
      });
      connection.settings.set(response.sessionId, response);
      return `${endpoint.id}:${response.sessionId}`;
    },
    async read(id, observer) {
      const key = nativeId(id);
      if (runs.has(id) || readers.has(id)) throw new Error("External ACP session is busy.");
      readers.add(id);
      try {
        const connection = await runtime();
        if (!connection.history)
          throw new Error("External ACP agent does not support native history replay.");
        connection.observers.set(key, projection(observer));
        try {
          const response = await connection.connection.agent.request("session/load", {
            sessionId: key,
            cwd: connection.cwd,
            mcpServers: [],
          });
          connection.settings.set(key, response);
        } finally {
          connection.observers.delete(key);
        }
      } finally {
        readers.delete(id);
      }
    },
    async start(id, text, observer, selection) {
      assertOpen();
      unsupported(selection);
      const key = nativeId(id);
      if (runs.has(id) || readers.has(id)) throw new Error("External ACP session is busy.");
      const { promise, resolve } = Promise.withResolvers<void>();
      const run: Run = {
        observer,
        tools: new Map(),
        abort: new AbortController(),
        done: promise,
        finish: resolve,
        dispatched: false,
      };
      runs.set(id, run);
      try {
        const connection = await runtime();
        run.runtime = connection;
        if (stopped(run)) return { stopReason: "cancelled" };
        let settings = await resume(connection, key, run);
        if (stopped(run)) return { stopReason: "cancelled" };
        for (const [category, value] of [
          ["model", selection?.model],
          ["thought_level", selection?.effort],
          ["mode", selection?.mode],
        ] as const) {
          if (value === undefined) continue;
          const option = selector(settings, category);
          if (option) {
            if (!values(option).some((choice) => choice.value === value))
              throw new Error(`Select an advertised external ACP ${category}.`);
            if (option.currentValue !== value) {
              const response = await connection.connection.agent.request(
                "session/set_config_option",
                {
                  sessionId: key,
                  configId: option.id,
                  value,
                },
              );
              settings = { ...settings, configOptions: response.configOptions };
              connection.settings.set(key, settings);
              if (stopped(run)) return { stopReason: "cancelled" };
              if (selector(settings, category)?.currentValue !== value)
                throw new Error(`External ACP did not acknowledge requested ${category}.`);
            }
          } else if (
            category === "mode" &&
            settings.modes?.availableModes.some((mode) => mode.id === value)
          ) {
            await connection.connection.agent.request("session/set_mode", {
              sessionId: key,
              modeId: value,
            });
            if (stopped(run)) return { stopReason: "cancelled" };
            settings.modes.currentModeId = value;
          } else throw new Error(`Select an advertised external ACP ${category}.`);
        }
        if (stopped(run)) return { stopReason: "cancelled" };
        connection.observers.set(key, projection(observer, run.tools));
        run.dispatched = true;
        const response = await connection.connection.agent.request("session/prompt", {
          sessionId: key,
          prompt: [{ type: "text", text }],
        });
        observer.raw(response);
        return response as RunResult;
      } finally {
        run.abort.abort();
        run.runtime?.observers.delete(key);
        runs.delete(id);
        run.finish();
      }
    },
    async steer() {
      throw new Error("External ACP does not advertise interactive steering.");
    },
    interrupt,
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = (async () => {
        const active = [...runs.entries()];
        const controls = active.map(async ([id, run]) => {
          run.abort.abort();
          if (run.dispatched && run.runtime) await interrupt(id);
        });
        const deadline = setTimeout(() => {
          for (const connection of runtimes.values()) void connection.dispose();
        }, 3000);
        deadline.unref();
        try {
          await Promise.allSettled(controls);
          await Promise.all([...runtimes.values()].map((connection) => connection.dispose()));
          await Promise.all(active.map(([, run]) => run.done));
        } finally {
          clearTimeout(deadline);
          runtimes.clear();
        }
      })();
      return disposing;
    },
  };
}

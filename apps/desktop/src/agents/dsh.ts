import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  DeepSeekHarness,
  type DeepSeekHarnessOptions,
  type HarnessSession,
} from "@deepseek-ai/dsh-sdk-client";
import type { RunResult } from "@swarmx/swarm";
import { z } from "zod";
import {
  type AgentOptions,
  HARNESS_CAPABILITIES,
  memoryContextSuffix,
  type NativeAgent,
  type Observer,
} from "./types.js";

export async function createDsh(options: AgentOptions): Promise<NativeAgent> {
  if (options.reviewOnly) throw new Error("DSH does not support restricted memory reviews.");
  const fresh = new Set<string>();
  const running = new Map<string, { harness: DeepSeekHarness; stopped: boolean }>();
  let disposed = false;
  return {
    name: "dsh",
    capabilities: HARNESS_CAPABILITIES.dsh,
    models: async () => ({ models: [], current: {} }),
    list: async () => [...fresh].map((sessionId) => ({ sessionId })),
    async create() {
      if (disposed) throw new Error("DSH Agent is disposed.");
      const id = randomUUID();
      fresh.add(id);
      return id;
    },
    async read(id) {
      if (!fresh.has(id)) throw new Error("DSH history is available in the Host execution log.");
    },
    async start(id, text, observer, selection) {
      if (disposed) throw new Error("DSH Agent is disposed.");
      if (selection?.mode !== undefined)
        throw new Error("DSH does not support permission mode selection.");
      const route = z
        .string()
        .regex(/^[^/\s]+\/\S+$/u, "DSH model must use provider/model.")
        .transform((value) => {
          const separator = value.indexOf("/");
          return { provider: value.slice(0, separator), model: value.slice(separator + 1) };
        })
        .optional()
        .parse(selection?.model);
      const profile = z.enum(["sdk", "sdk-minimal"]).optional().parse(selection?.profile);
      if (!fresh.delete(id)) throw new Error("DSH tasks execute once. Create a new task.");
      const token = randomUUID();
      const endpoint = options.registerMcp?.(token);
      const directory = mkdtempSync(join(tmpdir(), "swarmx-dsh-"));
      let run: { harness: DeepSeekHarness; stopped: boolean } | undefined;
      try {
        if (endpoint) {
          if (!observer.executionId) throw new Error("DSH tools require a bound Host execution.");
          endpoint.bind(`dsh:${id}`, observer.executionId);
        }
        const patch = join(directory, "mcp.cordis.yml");
        writeFileSync(
          patch,
          JSON.stringify([
            {
              insert: [
                {
                  id: "mcp-swarmx",
                  name: "@deepseek-ai/dsh-mcp-client",
                  config: {
                    serverName: "swarmx",
                    transport: "stdio",
                    command: options.mcp.command,
                    args: [...options.mcp.args],
                    env: endpoint
                      ? { ...options.mcp.env, SWARMX_MCP_TOKEN: token }
                      : options.mcp.env,
                    cwd: options.cwd,
                    failOnStartupError: true,
                  },
                },
              ],
            },
          ]),
          { mode: 0o600 },
        );
        run = {
          harness: new DeepSeekHarness({
            cwd: options.cwd,
            processCwd: options.cwd,
            patches: [patch],
            ...route,
            ...(profile === undefined ? {} : { profile }),
            ...(selection?.effort === undefined
              ? {}
              : {
                  reasoningEffort: selection.effort as NonNullable<
                    DeepSeekHarnessOptions["reasoningEffort"]
                  >,
                }),
          }),
          stopped: false,
        };
        running.set(id, run);
        return await runDshSession(
          run.harness.session(id),
          text + (selection?.instructions ? memoryContextSuffix(selection.instructions) : ""),
          observer,
        );
      } catch (error) {
        if (run?.stopped) return { stopReason: "cancelled" };
        throw error;
      } finally {
        try {
          await run?.harness.close();
        } finally {
          endpoint?.dispose();
          rmSync(directory, { recursive: true, force: true });
          running.delete(id);
        }
      }
    },
    async steer() {
      throw new Error("DSH supports independent executions; steering is unavailable.");
    },
    async interrupt(id) {
      const run = running.get(id);
      if (!run) return;
      run.stopped = true;
      await run.harness.close();
    },
    async dispose() {
      disposed = true;
      fresh.clear();
      await Promise.all(
        [...running.values()].map(async (run) => {
          run.stopped = true;
          await run.harness.close();
        }),
      );
    },
  };
}

const eventSchema = z.object({
  type: z.string(),
  seq: z.number().int().nonnegative(),
  data: z.record(z.string(), z.unknown()),
});
const terminalSchema = z.object({
  turn: z.number().int().nonnegative(),
  reason: z.discriminatedUnion("kind", [
    z.object({ kind: z.literal("completed") }),
    z.object({ kind: z.literal("aborted") }),
    z.object({ kind: z.literal("max-tokens") }),
    z.object({ kind: z.literal("blocked") }),
    z.object({ kind: z.literal("interrupted") }),
    z.object({ kind: z.literal("error"), error: z.object({ message: z.string() }) }),
  ]),
});

/** The caller owns the SDK session and runtime; this boundary owns one receipt-to-idle result. */
export async function runDshSession(
  session: Pick<HarnessSession, "id" | "run">,
  prompt: string,
  observer: Observer,
): Promise<RunResult> {
  let outcome: RunResult | undefined;
  let failure: Error | undefined;
  const openTurns = new Set<number>();
  const tools = new Map<string, { name: string; arguments: string }>();
  await session.run(prompt, {
    onNotification(notification) {
      observer.raw(notification, {
        "swarmx.native.session_id":
          typeof notification.params.sessionId === "string"
            ? notification.params.sessionId
            : undefined,
      });
      if (notification.method !== "session.event" || notification.params.sessionId !== session.id)
        return;
      const event = eventSchema.parse(notification.params.event);
      switch (event.type) {
        case "turn/start":
          openTurns.add(z.number().int().nonnegative().parse(event.data.turn));
          break;
        case "turn/end": {
          const { turn, reason } = terminalSchema.parse(event.data);
          openTurns.delete(turn);
          if (reason.kind === "error") failure ??= new Error(reason.error.message);
          else if (reason.kind === "blocked" || reason.kind === "interrupted")
            failure ??= new Error(`DSH turn ${reason.kind}.`);
          else if (!outcome || outcome.stopReason === "end_turn")
            outcome = {
              stopReason:
                reason.kind === "aborted"
                  ? "cancelled"
                  : reason.kind === "max-tokens"
                    ? "max_tokens"
                    : "end_turn",
            };
          break;
        }
        case "assistant/message": {
          const { message } = z
            .object({
              message: z.object({ content: z.array(z.looseObject({ type: z.string() })) }),
            })
            .parse(event.data);
          for (const block of message.content) {
            if (block.type === "text" || block.type === "reasoning")
              observer.text(
                `${session.id}:${event.seq}${block.type === "reasoning" ? ":reasoning" : ""}`,
                z.string().parse(block.text),
                block.type === "reasoning" ? "reasoning" : "assistant",
              );
          }
          break;
        }
        case "tool/call": {
          const call = z
            .object({ callId: z.string(), name: z.string(), arguments: z.string() })
            .parse(event.data);
          tools.set(call.callId, call);
          observer.tool(call.callId, call.name, call.arguments);
          observer.activity?.({ type: "tool", toolCallId: call.callId, status: "in_progress" });
          break;
        }
        case "tool/result": {
          const { message } = z
            .object({
              message: z.object({
                content: z.array(
                  z.object({
                    type: z.literal("tool-result"),
                    toolCallId: z.string(),
                    content: z.array(z.looseObject({ type: z.string() })),
                    isError: z.boolean().optional(),
                  }),
                ),
              }),
            })
            .parse(event.data);
          for (const block of message.content) {
            const call = tools.get(block.toolCallId);
            if (!call) throw new Error("DSH tool result has no observed call.");
            observer.tool(block.toolCallId, call.name, call.arguments, block.content);
            observer.activity?.({
              type: "tool",
              toolCallId: block.toolCallId,
              status: block.isError ? "failed" : "completed",
            });
          }
          break;
        }
      }
    },
  });
  if (failure) throw failure;
  if (!outcome || openTurns.size) throw new Error("DSH became idle without a terminal outcome.");
  return outcome;
}

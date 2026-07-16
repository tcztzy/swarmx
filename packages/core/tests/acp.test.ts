import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  AcpClient,
  RequestCancelledError,
  cancelAcpRequest,
  currentRequestSignal,
  withAcpRequest,
} from "../src/acp.js";

const coreRoot = fileURLToPath(new URL("..", import.meta.url));

describe("request-scoped cancellation", () => {
  it("applies only models advertised by the ACP session", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt({ ...agentOptions("models"), model: "verified-model" }, "hello"),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "model:verified-model" })],
    });

    const rejected = new AcpClient();
    await expect(
      rejected.prompt({ ...agentOptions("models"), model: "invented-model" }, "hello"),
    ).rejects.toThrow('cannot run configured model "invented-model"');
  });

  it("applies stable model config before refreshed reasoning effort config", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt(
        { ...agentOptions("stable-config"), model: "verified-model", effort: "high" },
        "hello",
      ),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "config:verified-model:high:model,effort" })],
    });
  });

  it("supports grouped and category-less stable config options", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt(
        { ...agentOptions("grouped-config"), model: "verified-model", effort: "High" },
        "hello",
      ),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "config:verified-model:high:model,effort" })],
    });
  });

  it("sends ACP session/cancel before process fallback", async () => {
    const client = new AcpClient();
    const started = deferred<void>();

    const run = withAcpRequest("protocol-cancel", () =>
      client.prompt(agentOptions("cooperative"), "hello", undefined, undefined, (chunk) => {
        if (chunk.content === "started") started.resolve();
      }),
    );

    await started.promise;
    await expect(cancelAcpRequest("protocol-cancel")).resolves.toBe(true);
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(cancelAcpRequest("protocol-cancel")).resolves.toBe(false);
  });

  it("terminates an ACP process that ignores protocol cancellation", async () => {
    const client = new AcpClient();
    const started = deferred<void>();
    const run = withAcpRequest("fallback-cancel", () =>
      client.prompt(agentOptions("ignore"), "hello", undefined, undefined, (chunk) => {
        if (chunk.content === "started") started.resolve();
      }),
    );

    await started.promise;
    await expect(cancelAcpRequest("fallback-cancel")).resolves.toBe(true);
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(cancelAcpRequest("fallback-cancel")).resolves.toBe(false);
  });

  it("terminates descendants in the ACP process group on POSIX", async () => {
    if (process.platform === "win32") return;
    const client = new AcpClient();
    const grandchildPid = deferred<number>();
    const run = withAcpRequest("tree-cancel", () =>
      client.prompt(agentOptions("tree"), "hello", undefined, undefined, (chunk) => {
        if (chunk.content.startsWith("grandchild:")) {
          grandchildPid.resolve(Number(chunk.content.slice("grandchild:".length)));
        }
      }),
    );

    const pid = await grandchildPid.promise;
    await expect(cancelAcpRequest("tree-cancel")).resolves.toBe(true);
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    await waitForProcessExit(pid);
  });

  it("cleans request state after normal completion and failure", async () => {
    const completeClient = new AcpClient();
    await expect(
      withAcpRequest("normal-complete", () =>
        completeClient.prompt(agentOptions("complete"), "hello"),
      ),
    ).resolves.toMatchObject({ stopReason: "end_turn" });
    await expect(cancelAcpRequest("normal-complete")).resolves.toBe(false);

    const failingClient = new AcpClient();
    await expect(
      withAcpRequest("normal-failure", () =>
        failingClient.prompt(agentOptions("failure"), "hello"),
      ),
    ).rejects.toThrow("Internal error");
    await expect(cancelAcpRequest("normal-failure")).resolves.toBe(false);

    const missingClient = new AcpClient();
    await expect(
      withAcpRequest("spawn-failure", () =>
        missingClient.prompt({ command: "swarmx-command-that-does-not-exist", args: [] }, "hello"),
      ),
    ).rejects.toMatchObject({ code: "ENOENT" });
    await expect(cancelAcpRequest("spawn-failure")).resolves.toBe(false);
  });

  it("refuses optional session methods that the ACP backend did not advertise", async () => {
    const listClient = new AcpClient();
    await expect(listClient.listSessions(agentOptions("complete"))).rejects.toThrow(
      "does not advertise session/list",
    );

    const loadClient = new AcpClient();
    await expect(
      loadClient.loadSession(agentOptions("complete"), "test-session", coreRoot),
    ).rejects.toThrow("does not advertise session/load");
  });

  it("records cancellation before an ACP child can spawn", async () => {
    const gate = deferred<void>();
    const client = new AcpClient();
    const run = withAcpRequest("early-cancel", async () => {
      await gate.promise;
      return client.prompt(agentOptions("complete"), "hello");
    });

    await expect(cancelAcpRequest("early-cancel")).resolves.toBe(true);
    await expect(cancelAcpRequest("early-cancel")).resolves.toBe(true);
    gate.resolve();
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    expect(client.stderrOutput()).toBe("");
  });

  it("keeps rapid requests isolated and rejects concurrent ID reuse", async () => {
    const firstGate = deferred<void>();
    const secondGate = deferred<void>();

    const first = withAcpRequest("rapid-first", async () => {
      await firstGate.promise;
      return "first";
    });
    const second = withAcpRequest("rapid-second", async () => {
      await secondGate.promise;
      return "second";
    });

    await expect(withAcpRequest("rapid-first", async () => "duplicate")).rejects.toThrow(
      "already active",
    );
    await expect(cancelAcpRequest("rapid-first")).resolves.toBe(true);
    expect(currentRequestSignal()).toBeUndefined();

    secondGate.resolve();
    await expect(second).resolves.toBe("second");
    firstGate.resolve();
    await expect(first).rejects.toBeInstanceOf(RequestCancelledError);
    await expect(cancelAcpRequest("rapid-first")).resolves.toBe(false);
    await expect(cancelAcpRequest("rapid-second")).resolves.toBe(false);
  });
});

type AgentMode =
  | "cooperative"
  | "ignore"
  | "tree"
  | "complete"
  | "failure"
  | "models"
  | "stable-config"
  | "grouped-config";

function agentOptions(mode: AgentMode) {
  return {
    command: process.execPath,
    args: ["--input-type=module", "--eval", agentScript(mode)],
    cwd: coreRoot,
  };
}

function agentScript(mode: AgentMode): string {
  return `
    import { AgentSideConnection, ndJsonStream } from "@agentclientprotocol/sdk";
    import { spawn } from "node:child_process";
    import { Readable, Writable } from "node:stream";

    if (${JSON.stringify(mode)} === "ignore") process.on("SIGTERM", () => {});

    let finishPrompt;
    let selectedModel = "default-model";
    let selectedEffort = "low";
    const configChanges = [];
    const configOptions = () => {
      const modelValues = ${JSON.stringify(mode)} === "grouped-config"
        ? [{ group: "recommended", name: "Recommended", options: [
            { value: "default-model", name: "Default" },
            { value: "verified-model", name: "Verified" },
          ] }]
        : [
            { value: "default-model", name: "Default" },
            { value: "verified-model", name: "Verified" },
          ];
      return [
        {
          id: "model",
          name: "Model",
          ...(${JSON.stringify(mode)} === "stable-config" ? { category: "model" } : {}),
          type: "select",
          currentValue: selectedModel,
          options: modelValues,
        },
        {
          id: "reasoning-effort",
          name: "Reasoning Effort",
          ...(${JSON.stringify(mode)} === "stable-config" ? { category: "thought_level" } : {}),
          type: "select",
          currentValue: selectedEffort,
          options: [
            { value: "low", name: "Low" },
            { value: "high", name: "High" },
          ],
        },
      ];
    };
    new AgentSideConnection((connection) => ({
      async initialize(params) {
        return { protocolVersion: params.protocolVersion, agentCapabilities: {}, authMethods: [] };
      },
      async newSession() {
        return {
          sessionId: "test-session",
          ...(["stable-config", "grouped-config"].includes(${JSON.stringify(mode)}) ? {
            configOptions: configOptions(),
          } : {}),
          ...(${JSON.stringify(mode)} === "models" ? {
            models: {
              currentModelId: selectedModel,
              availableModels: [
                { modelId: "default-model", name: "Default" },
                { modelId: "verified-model", name: "Verified" },
              ],
            },
          } : {}),
        };
      },
      async unstable_setSessionModel(params) {
        selectedModel = params.modelId;
        return {};
      },
      async setSessionConfigOption(params) {
        if (params.configId === "model") selectedModel = params.value;
        if (params.configId === "reasoning-effort") selectedEffort = params.value;
        configChanges.push(params.configId === "model" ? "model" : "effort");
        return { configOptions: configOptions() };
      },
      async prompt() {
        if (${JSON.stringify(mode)} === "tree") {
          const grandchild = spawn(process.execPath, [
            "--input-type=module",
            "--eval",
            "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)",
          ], { stdio: "ignore" });
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "grandchild:" + grandchild.pid },
            },
          });
          return new Promise(() => {});
        }
        if (${JSON.stringify(mode)} === "models") {
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "model:" + selectedModel },
            },
          });
          return { stopReason: "end_turn" };
        }
        if (["stable-config", "grouped-config"].includes(${JSON.stringify(mode)})) {
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: {
                type: "text",
                text: "config:" + selectedModel + ":" + selectedEffort + ":" + configChanges.join(","),
              },
            },
          });
          return { stopReason: "end_turn" };
        }
        await connection.sessionUpdate({
          sessionId: "test-session",
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "started" },
          },
        });
        if (${JSON.stringify(mode)} === "complete") return { stopReason: "end_turn" };
        if (${JSON.stringify(mode)} === "failure") throw new Error("agent failed");
        return new Promise((resolve) => { finishPrompt = resolve; });
      },
      async cancel() {
        if (${JSON.stringify(mode)} !== "cooperative") return;
        await connection.sessionUpdate({
          sessionId: "test-session",
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "cancel-ack" },
          },
        });
        finishPrompt?.({ stopReason: "cancelled" });
      },
    }), ndJsonStream(Writable.toWeb(process.stdout), Readable.toWeb(process.stdin)));
  `;
}

async function waitForProcessExit(pid: number): Promise<void> {
  const deadline = Date.now() + 3_000;
  while (Date.now() < deadline) {
    try {
      process.kill(pid, 0);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ESRCH") return;
      throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error(`ACP grandchild process ${pid} was not terminated.`);
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
} {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

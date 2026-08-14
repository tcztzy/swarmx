import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  AcpClient,
  AcpSessionUnavailableError,
  cancelAcpRequest,
  currentRequestSignal,
  RequestCancelledError,
  withAcpRequest,
} from "../src/acp.js";

const coreRoot = fileURLToPath(new URL("..", import.meta.url));

describe("request-scoped cancellation", () => {
  it("applies only models advertised by the ACP session config", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt({ ...agentOptions("stable-config"), model: "verified-model" }, "hello"),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "config:verified-model:low:model" })],
    });

    const rejected = new AcpClient();
    await expect(
      rejected.prompt({ ...agentOptions("stable-config"), model: "invented-model" }, "hello"),
    ).rejects.toThrow('cannot run configured model "invented-model"');
  }, 15_000);

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

  it("applies a preferred ACP mode when the session advertises it", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt({ ...agentOptions("session-mode"), preferredMode: "plan" }, "hello"),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "mode:plan" })],
    });
  });

  it("reports a new ACP Session before its first prompt", async () => {
    const events: string[] = [];
    const client = new AcpClient();
    const result = await client.prompt(
      {
        ...agentOptions("complete"),
        onSessionId: (sessionId) => events.push(`session:${sessionId}`),
      },
      "hello",
      undefined,
      undefined,
      (chunk) => events.push(`chunk:${chunk.content}`),
    );

    expect(result.sessionId).toBe("test-session");
    expect(events).toEqual(["session:test-session", "chunk:started"]);
  });

  it("ignores ACP updates that arrive after the terminal prompt response", async () => {
    const chunks: string[] = [];
    const client = new AcpClient();

    const result = await client.prompt(
      agentOptions("late-update"),
      "hello",
      undefined,
      undefined,
      (chunk) => chunks.push(chunk.content),
    );
    expect(result.messages.map((message) => message.content)).toEqual(["settled"]);

    await new Promise((resolve) => setTimeout(resolve, 75));

    expect(chunks).toEqual(["settled"]);
  });

  it("rejects an ACP terminal response that overtakes a pending permission", async () => {
    const permissionStarted = deferred<void>();
    const permission = deferred<{ outcome: { outcome: "cancelled" } }>();
    const client = new AcpClient();
    const prompt = client.prompt(
      {
        ...agentOptions("unsettled-permission"),
        requestPermission: async () => {
          permissionStarted.resolve();
          return permission.promise;
        },
      },
      "hello",
    );
    const assertion = expect(prompt).rejects.toThrow(/terminal.*permission.*unsettled/i);

    await permissionStarted.promise;
    await new Promise((resolve) => setTimeout(resolve, 50));
    permission.resolve({ outcome: { outcome: "cancelled" } });
    await assertion;
  });

  it("suppresses loaded history while continuing an existing ACP Session", async () => {
    const client = new AcpClient();
    await expect(
      client.prompt(agentOptions("load"), "hello", undefined, "stored-session"),
    ).resolves.toMatchObject({
      sessionId: "stored-session",
      messages: [expect.objectContaining({ content: "current" })],
    });

    const unsupported = new AcpClient();
    await expect(
      unsupported.prompt(agentOptions("complete"), "hello", undefined, "stored-session"),
    ).rejects.toBeInstanceOf(AcpSessionUnavailableError);
  });

  it("keeps ACP tool updates correlated without exposing their ids as content", async () => {
    const client = new AcpClient();
    const chunks: Array<{
      content: string;
      kind: string;
      render?: { invocationId?: string; status?: string };
    }> = [];

    const result = await client.prompt(
      agentOptions("tools"),
      "hello",
      undefined,
      undefined,
      (chunk) => {
        chunks.push(chunk);
      },
    );
    expect(result).toMatchObject({ stopReason: "end_turn" });

    expect(chunks).toEqual([
      expect.objectContaining({
        content: JSON.stringify({ path: "README.md" }),
        kind: "tool_call",
        render: { invocationId: "call_readme_1", status: "running" },
      }),
      expect.objectContaining({
        content: "first line\n",
        kind: "tool_progress",
        render: { invocationId: "call_readme_1", status: "running" },
        structuredContent: expect.objectContaining({ mode: "append", stream: "combined" }),
      }),
      expect.objectContaining({
        content: JSON.stringify({ progress: "reading" }),
        kind: "tool_result",
        render: { invocationId: "call_readme_1", status: "running" },
      }),
      expect.objectContaining({
        content: JSON.stringify({ path: "README.md" }),
        kind: "tool_result",
        render: { invocationId: "call_readme_1", status: "succeeded" },
      }),
      expect.objectContaining({ content: "done", kind: "message" }),
    ]);
    expect(chunks.map((chunk) => chunk.content)).not.toContain("call_readme_1");
    expect(result.messages.some((chunk) => chunk.kind === "tool_progress")).toBe(false);
  });

  it("cancels ACP permission by default and returns only an offered handled option", async () => {
    const cancelled = new AcpClient();
    await expect(cancelled.prompt(agentOptions("permission"), "hello")).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "permission:cancelled" })],
    });

    const requests: unknown[] = [];
    const selected = new AcpClient();
    await expect(
      selected.prompt(
        {
          ...agentOptions("permission"),
          requestPermission: async (request) => {
            requests.push(request);
            return { outcome: { outcome: "selected", optionId: "allow-once" } };
          },
        },
        "hello",
      ),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "permission:selected:allow-once" })],
    });
    expect(requests).toEqual([
      expect.objectContaining({
        sessionId: "test-session",
        options: expect.arrayContaining([
          expect.objectContaining({ optionId: "allow-once", kind: "allow_once" }),
        ]),
      }),
    ]);

    const forged = new AcpClient();
    await expect(
      forged.prompt(
        {
          ...agentOptions("permission"),
          requestPermission: async () => ({
            outcome: { outcome: "selected", optionId: "not-offered" },
          }),
        },
        "hello",
      ),
    ).resolves.toMatchObject({
      messages: [expect.objectContaining({ content: "permission:cancelled" })],
    });
  }, 15_000);

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
  | "grouped-config"
  | "session-mode"
  | "tools"
  | "permission"
  | "late-update"
  | "unsettled-permission"
  | "load";

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

    if (["ignore", "late-update"].includes(${JSON.stringify(mode)})) {
      process.on("SIGTERM", () => {});
    }

    let finishPrompt;
    let selectedModel = "default-model";
    let selectedEffort = "low";
    let selectedMode = "default";
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
        return {
          protocolVersion: params.protocolVersion,
          agentCapabilities: {
            ...(${JSON.stringify(mode)} === "load" ? { loadSession: true } : {}),
          },
          authMethods: [],
        };
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
          ...(${JSON.stringify(mode)} === "session-mode" ? {
            modes: {
              currentModeId: selectedMode,
              availableModes: [
                { id: "default", name: "Default" },
                { id: "plan", name: "Plan" },
              ],
            },
          } : {}),
        };
      },
      async unstable_setSessionModel(params) {
        selectedModel = params.modelId;
        return {};
      },
      async loadSession(params) {
        if (${JSON.stringify(mode)} !== "load") throw new Error("load unsupported");
        await connection.sessionUpdate({
          sessionId: params.sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "historical" },
          },
        });
        return {};
      },
      async setSessionConfigOption(params) {
        if (params.configId === "model") selectedModel = params.value;
        if (params.configId === "reasoning-effort") selectedEffort = params.value;
        configChanges.push(params.configId === "model" ? "model" : "effort");
        return { configOptions: configOptions() };
      },
      async setSessionMode(params) {
        selectedMode = params.modeId;
        return {};
      },
      async prompt() {
        if (${JSON.stringify(mode)} === "unsettled-permission") {
          void connection.requestPermission({
            sessionId: "test-session",
            toolCall: {
              toolCallId: "call_unsettled_permission_1",
              title: "Late approval",
              kind: "execute",
              rawInput: { command: "false" },
            },
            options: [{ optionId: "reject-once", name: "Reject", kind: "reject_once" }],
          }).catch(() => {});
          await new Promise((resolve) => setTimeout(resolve, 10));
          return { stopReason: "end_turn" };
        }
        if (${JSON.stringify(mode)} === "late-update") {
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "settled" },
            },
          });
          setTimeout(() => {
            void connection.sessionUpdate({
              sessionId: "test-session",
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "late" },
              },
            }).catch(() => {});
          }, 25);
          return { stopReason: "end_turn" };
        }
        if (${JSON.stringify(mode)} === "permission") {
          const response = await connection.requestPermission({
            sessionId: "test-session",
            toolCall: {
              toolCallId: "call_permission_1",
              title: "Run tests",
              kind: "execute",
              rawInput: { command: "pnpm test", token: "must-not-cross-desktop" },
            },
            options: [
              { optionId: "reject-once", name: "Reject", kind: "reject_once" },
              { optionId: "allow-once", name: "Allow once", kind: "allow_once" },
            ],
          });
          const suffix = response.outcome.outcome === "selected"
            ? ":" + response.outcome.optionId
            : "";
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "permission:" + response.outcome.outcome + suffix },
            },
          });
          return { stopReason: "end_turn" };
        }
        if (${JSON.stringify(mode)} === "tools") {
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "tool_call",
              toolCallId: "call_readme_1",
              title: "workspace_read_file",
              rawInput: { path: "README.md" },
            },
          });
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: "call_readme_1",
              _meta: {
                terminal_output_delta: {
                  data: "first line\\n",
                  terminal_id: "call_readme_1",
                },
              },
            },
          });
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: "call_readme_1",
              title: "workspace_read_file",
              status: "in_progress",
              rawOutput: { progress: "reading" },
            },
          });
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "tool_call_update",
              toolCallId: "call_readme_1",
              title: "workspace_read_file",
              status: "completed",
              rawOutput: { path: "README.md" },
            },
          });
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "done" },
            },
          });
          return { stopReason: "end_turn" };
        }
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
        if (${JSON.stringify(mode)} === "session-mode") {
          await connection.sessionUpdate({
            sessionId: "test-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "mode:" + selectedMode },
            },
          });
          return { stopReason: "end_turn" };
        }
        if (${JSON.stringify(mode)} === "load") {
          await connection.sessionUpdate({
            sessionId: "stored-session",
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "current" },
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

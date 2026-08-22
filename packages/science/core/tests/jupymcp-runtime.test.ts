import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type JupyMcpPeer,
  type JupyMcpPeerFactory,
  JupyMcpRuntime,
  type JupyMcpToolResult,
} from "../src/jupymcp-runtime.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { force: true, recursive: true });
});

function workspace(): string {
  const root = mkdtempSync(join(tmpdir(), "swarmx-jupymcp-"));
  roots.push(root);
  return root;
}

function runtime(
  factory: JupyMcpPeerFactory,
  maxOutputBytes = 64 * 1024,
  maxNotebookBytes = 5 * 1024 * 1024,
): JupyMcpRuntime {
  return new JupyMcpRuntime(
    {
      args: [],
      command: "jupymcp",
      maxNotebookBytes,
      maxOutputBytes,
      requestTimeoutMs: 30_000,
    },
    factory,
  );
}

function peer(
  execute: (
    args: Readonly<Record<string, unknown>>,
    signal?: AbortSignal,
  ) => Promise<JupyMcpToolResult>,
): JupyMcpPeer & { readonly calls: { name: string; args: Readonly<Record<string, unknown>> }[] } {
  const calls: { name: string; args: Readonly<Record<string, unknown>> }[] = [];
  let kernelCount = 0;
  return {
    calls,
    async callTool(name, args, signal) {
      calls.push({ name, args });
      if (name === "start_kernel") {
        kernelCount += 1;
        return { content: [{ type: "text", text: `kernel-${kernelCount}` }] };
      }
      if (name === "execute") return execute(args, signal);
      return { content: [] };
    },
    async close() {},
    async readResource() {
      return { contents: [] };
    },
  };
}

describe("T29 JupyMCP Notebook Controller", () => {
  it("V72/V76 shares one workspace peer while routing each Notebook to its own kernel", async () => {
    const created: { cwd: string; peer: ReturnType<typeof peer> }[] = [];
    const factory: JupyMcpPeerFactory = async ({ cwd }) => {
      const next = peer(async (args) => ({
        content: [
          {
            type: "text",
            text: `ran:${String(args.code)}`,
            _meta: { name: "stdout" },
          },
        ],
      }));
      created.push({ cwd, peer: next });
      return next;
    };
    const current = runtime(factory);
    const root = workspace();

    await current.execute({
      notebookId: "notebook-a",
      source: "value = 40",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });
    await current.execute({
      notebookId: "notebook-a",
      source: "print(value + 2)",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });
    await current.execute({
      notebookId: "notebook-b",
      source: "print('isolated')",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });

    expect(created).toHaveLength(1);
    expect(created.map((entry) => entry.cwd)).toEqual([root]);
    expect(created[0]?.peer.calls).toEqual([
      { name: "start_kernel", args: { kernel_name: "python3" } },
      {
        name: "execute",
        args: expect.objectContaining({
          code: "value = 40",
          kernel_id: "kernel-1",
          path: ".dsh-science-notebook-notebook-a.ipynb",
          type: "ipynb",
        }),
      },
      {
        name: "execute",
        args: expect.objectContaining({
          code: "print(value + 2)",
          kernel_id: "kernel-1",
          path: ".dsh-science-notebook-notebook-a.ipynb",
          type: "ipynb",
        }),
      },
      { name: "start_kernel", args: { kernel_name: "python3" } },
      {
        name: "execute",
        args: expect.objectContaining({
          code: "print('isolated')",
          kernel_id: "kernel-2",
          path: ".dsh-science-notebook-notebook-b.ipynb",
          type: "ipynb",
        }),
      },
    ]);

    await current.close();
    expect(created[0]?.peer.calls.slice(-3)).toEqual([
      {
        name: "shutdown_kernel",
        args: { kernel_id: "kernel-1", now: true },
      },
      {
        name: "shutdown_kernel",
        args: { kernel_id: "kernel-2", now: true },
      },
      { name: "shutdown_all", args: { now: true } },
    ]);
  });

  it("V72 never shares an MCP peer across workspaces", async () => {
    const peers: ReturnType<typeof peer>[] = [];
    const current = runtime(async () => {
      const next = peer(async () => ({ content: [] }));
      peers.push(next);
      return next;
    });

    for (const workspaceRoot of [workspace(), workspace()]) {
      await current.execute({
        notebookId: "notebook-a",
        source: "pass",
        workspaceKey: workspaceRoot,
        workspaceRoot,
      });
    }

    expect(peers).toHaveLength(2);
    await current.close();
  });

  it("V73 normalizes ordered canonical MCP blocks into bounded Notebook outputs", async () => {
    const current = runtime(
      async () =>
        peer(async () => ({
          content: [
            { type: "text", text: "hello\n", _meta: { name: "stdout" } },
            { type: "text", text: "42" },
            { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
            {
              type: "resource",
              resource: {
                uri: "notebook://cell",
                mimeType: "text/html",
                text: "<strong>result</strong>",
              },
            },
          ],
        })),
      1_024,
    );

    const result = await current.execute({
      notebookId: "notebook-a",
      source: "display(42)",
      workspaceKey: "workspace",
      workspaceRoot: workspace(),
    });

    expect(result.status).toBe("succeeded");
    expect(result.stdout).toEqual({ text: "hello\n", truncated: false });
    expect(result.stderr).toEqual({ text: "", truncated: false });
    expect(result.outputs).toEqual([
      { type: "stream", name: "stdout", text: "hello\n", truncated: false },
      {
        type: "execute_result",
        data: [{ mime: "text/plain", data: "42", encoding: "utf8", truncated: false }],
      },
      {
        type: "display_data",
        data: [{ mime: "image/png", data: "aGVsbG8=", encoding: "base64", truncated: false }],
      },
      {
        type: "display_data",
        data: [
          {
            mime: "text/html",
            data: "<strong>result</strong>",
            encoding: "utf8",
            truncated: false,
          },
        ],
      },
    ]);

    await current.close();
  });

  it("V71/V73 prefers the canonical ipynb output type when MCP text metadata is absent", async () => {
    const current = runtime(async () => ({
      async callTool(name) {
        if (name === "start_kernel") {
          return { content: [{ type: "text", text: "kernel-1" }] };
        }
        return name === "execute" ? { content: [{ type: "text", text: "40\n" }] } : { content: [] };
      },
      async close() {},
      async readResource() {
        return {
          contents: [
            {
              uri: "notebook://current",
              text: JSON.stringify({
                cells: [
                  {
                    cell_type: "code",
                    outputs: [{ output_type: "stream", name: "stdout", text: "40\n" }],
                    source: ["print(40)"],
                  },
                ],
                metadata: {},
                nbformat: 4,
                nbformat_minor: 5,
              }),
            },
          ],
        };
      },
    }));

    const result = await current.execute({
      notebookId: "notebook-a",
      source: "print(40)",
      workspaceKey: "workspace",
      workspaceRoot: workspace(),
    });

    expect(result.outputs).toEqual([
      { type: "stream", name: "stdout", text: "40\n", truncated: false },
    ]);
    expect(result.stdout).toEqual({ text: "40\n", truncated: false });

    await current.close();
  });

  it("V77 bounds canonical Notebook resources separately from one cell output", async () => {
    const notebookDocument = JSON.stringify({
      cells: [{ cell_type: "code", source: ["x".repeat(256)] }],
      metadata: {},
      nbformat: 4,
      nbformat_minor: 5,
    });
    const current = runtime(
      async () => ({
        async callTool() {
          return { content: [] };
        },
        async close() {},
        async readResource() {
          return { contents: [{ uri: "notebook://current", text: notebookDocument }] };
        },
      }),
      32,
      2_048,
    );

    await expect(
      current.readNotebook({
        notebookId: "notebook-a",
        workspaceKey: "workspace",
        workspaceRoot: workspace(),
      }),
    ).resolves.toBe(notebookDocument);

    await current.close();
  });

  it("V67 installs artifact variables silently around only the visible execution", async () => {
    const currentPeer = peer(async () => ({ content: [] }));
    const current = runtime(async () => currentPeer);
    const root = workspace();

    await current.execute({
      inputEnvironment: {
        DSH_SCIENCE_INPUT_0: join(root, "staging", "input.csv"),
      },
      notebookId: "notebook-a",
      source: "print(open(__import__('os').environ['DSH_SCIENCE_INPUT_0']).read())",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });

    expect(currentPeer.calls.map((call) => call.name)).toEqual([
      "start_kernel",
      "execute",
      "execute",
    ]);
    expect(currentPeer.calls[1]?.args).toMatchObject({
      kernel_id: "kernel-1",
      path: ":memory:",
      silent: true,
      store_history: false,
    });
    expect(currentPeer.calls[1]?.args.code).toContain("DSH_SCIENCE_INPUT_0");
    expect(currentPeer.calls[1]?.args.code).toContain('events.register("post_run_cell"');
    expect(currentPeer.calls[2]?.args).toMatchObject({
      kernel_id: "kernel-1",
      path: ".dsh-science-notebook-notebook-a.ipynb",
      silent: false,
      store_history: true,
    });
    await current.close();
  });

  it("V67 rejects a failed silent artifact setup before visible execution", async () => {
    const currentPeer = peer(async () => ({
      content: [{ type: "text", text: "kernel setup failed" }],
      isError: true,
    }));
    const current = runtime(async () => currentPeer);
    const root = workspace();

    await expect(
      current.execute({
        inputEnvironment: { DSH_SCIENCE_INPUT_0: join(root, "input.csv") },
        notebookId: "notebook-a",
        source: "print('must not run')",
        workspaceKey: "workspace",
        workspaceRoot: root,
      }),
    ).rejects.toMatchObject({ code: "JUPYMCP_UNAVAILABLE" });
    expect(currentPeer.calls).toHaveLength(3);
    expect(currentPeer.calls[1]?.args).toMatchObject({ kernel_id: "kernel-1", silent: true });
    expect(currentPeer.calls[2]).toEqual({
      name: "shutdown_kernel",
      args: { kernel_id: "kernel-1", now: true },
    });

    await current.close();
  });

  it("V67 serializes temporary input setup across Notebook controllers", async () => {
    let releaseFirst: (() => void) | undefined;
    let markFirstStarted: (() => void) | undefined;
    const firstStarted = new Promise<void>((resolve) => {
      markFirstStarted = resolve;
    });
    const firstBlocked = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const currentPeer = peer(async (args) => {
      if (args.code === "first-visible") {
        markFirstStarted?.();
        await firstBlocked;
      }
      return { content: [] };
    });
    const current = runtime(async () => currentPeer);
    const root = workspace();

    const first = current.execute({
      inputEnvironment: { DSH_FIRST: join(root, "first.csv") },
      notebookId: "notebook-a",
      source: "first-visible",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });
    await firstStarted;
    const second = current.execute({
      inputEnvironment: { DSH_SECOND: join(root, "second.csv") },
      notebookId: "notebook-b",
      source: "second-visible",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(currentPeer.calls.some((call) => call.args.code === "second-visible")).toBe(false);
    expect(currentPeer.calls.some((call) => String(call.args.code).includes("DSH_SECOND"))).toBe(
      false,
    );

    releaseFirst?.();
    await Promise.all([first, second]);
    expect(currentPeer.calls.some((call) => call.args.code === "second-visible")).toBe(true);

    await current.close();
  });

  it("V73/V74 aborts without kernel reuse while retaining the workspace peer", async () => {
    let executionCount = 0;
    let markStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const currentPeer = peer(async (_args, signal) => {
      executionCount += 1;
      if (executionCount === 1) {
        markStarted?.();
        return new Promise((_resolve, reject) => {
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      }
      return { content: [] };
    });
    vi.spyOn(currentPeer, "close");
    const factory = vi.fn<JupyMcpPeerFactory>(async () => currentPeer);
    const current = runtime(factory);
    const root = workspace();
    const controller = new AbortController();
    const pending = current.execute(
      {
        notebookId: "notebook-a",
        source: "while True: pass",
        workspaceKey: "workspace",
        workspaceRoot: root,
      },
      controller.signal,
    );
    await started;
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
    expect(currentPeer.calls).toContainEqual({
      name: "shutdown_kernel",
      args: { kernel_id: "kernel-1", now: true },
    });
    expect(currentPeer.close).not.toHaveBeenCalled();

    await current.execute({
      notebookId: "notebook-a",
      source: "print('fresh kernel')",
      workspaceKey: "workspace",
      workspaceRoot: root,
    });
    expect(factory).toHaveBeenCalledOnce();
    expect(currentPeer.calls.filter((call) => call.name === "start_kernel")).toHaveLength(2);

    await current.close();
  });

  it("V74 settles an in-flight execution before shutting down its MCP peer", async () => {
    let settle: ((result: JupyMcpToolResult) => void) | undefined;
    let markStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });
    const currentPeer = peer(
      async () =>
        new Promise<JupyMcpToolResult>((resolve) => {
          settle = resolve;
          markStarted?.();
        }),
    );
    const current = runtime(async () => currentPeer);
    const execution = current.execute({
      notebookId: "notebook-a",
      source: "while True: pass",
      workspaceKey: "workspace",
      workspaceRoot: workspace(),
    });
    const executionError = execution.then(
      () => undefined,
      (error: unknown) => error,
    );
    await started;
    const closing = current.close();
    await Promise.resolve();
    await Promise.resolve();

    expect(currentPeer.calls.map((call) => call.name)).toEqual(["start_kernel", "execute"]);

    settle?.({ content: [] });
    await expect(executionError).resolves.toMatchObject({ code: "SCIENCE_CLOSED" });
    await closing;
    expect(currentPeer.calls.map((call) => call.name)).toEqual([
      "start_kernel",
      "execute",
      "shutdown_kernel",
      "shutdown_all",
    ]);
  });

  it("V74 disposes every Notebook session and workspace peer exactly once", async () => {
    const peers: ReturnType<typeof peer>[] = [];
    const current = runtime(async () => {
      const next = peer(async () => ({ content: [] }));
      vi.spyOn(next, "close");
      peers.push(next);
      return next;
    });
    const root = workspace();
    for (const notebookId of ["notebook-a", "notebook-b"]) {
      await current.execute({
        notebookId,
        source: "pass",
        workspaceKey: "workspace",
        workspaceRoot: root,
      });
    }

    await current.close();
    await current.close();

    expect(peers).toHaveLength(1);
    expect(peers[0]?.calls.filter((call) => call.name === "shutdown_kernel")).toHaveLength(2);
    expect(peers[0]?.calls.filter((call) => call.name === "shutdown_all")).toHaveLength(1);
    expect(peers[0]?.close).toHaveBeenCalledOnce();
  });
});

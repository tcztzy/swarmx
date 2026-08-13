import { describe, expect, it, vi } from "vitest";
import { cancelAcpRequest, RequestCancelledError, withAcpRequest } from "../src/acp.js";
import type { McpConnectionResult } from "../src/mcp.js";
import { localToolResult, McpManager } from "../src/mcp.js";

describe("McpManager cancellation", () => {
  it("passes the active request signal to an in-flight MCP tool call", async () => {
    const manager = new McpManager();
    const started = deferred<void>();
    const callTool = vi.fn(
      (_params: unknown, _schema: unknown, options: { signal?: AbortSignal } | undefined) =>
        new Promise<never>((_resolve, reject) => {
          started.resolve();
          options?.signal?.addEventListener("abort", () => reject(options.signal?.reason), {
            once: true,
          });
        }),
    );
    const clients = new Map([["test", { callTool }]]);
    Object.assign(manager as unknown as { clients: typeof clients; tools: unknown[] }, {
      clients,
      tools: [
        {
          name: "mcp__test__slow",
          inputSchema: { type: "object" },
          serverName: "test",
          remoteName: "slow",
        },
      ],
    });

    const run = withAcpRequest("mcp-cancel", () => manager.callTool("mcp__test__slow", {}));
    await started.promise;
    await expect(cancelAcpRequest("mcp-cancel")).resolves.toBe(true);

    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    expect(callTool.mock.calls[0]?.[2]?.signal?.aborted).toBe(true);
  });

  it("bounds direct host-side MCP tool calls with a timeout", async () => {
    const manager = new McpManager();
    const callTool = vi.fn(
      (_params: unknown, _schema: unknown, options: { signal?: AbortSignal } | undefined) =>
        new Promise<never>((_resolve, reject) => {
          options?.signal?.addEventListener("abort", () => reject(options.signal?.reason), {
            once: true,
          });
        }),
    );
    const clients = new Map([["registry", { callTool }]]);
    Object.assign(manager as unknown as { clients: typeof clients; tools: unknown[] }, {
      clients,
      tools: [
        {
          name: "mcp__registry__project_bootstrap",
          inputSchema: { type: "object" },
          serverName: "registry",
          remoteName: "project_bootstrap",
          hostOnly: true,
        },
      ],
    });

    await expect(
      manager.callServerTool("registry", "project_bootstrap", {}, { timeoutMs: 5 }),
    ).rejects.toThrow();
    expect(callTool.mock.calls[0]?.[2]?.signal?.aborted).toBe(true);
  });

  it("closes connected clients and clears reusable state", async () => {
    const manager = new McpManager();
    const close = vi.fn(async () => undefined);
    const clients = new Map([["test", { close }]]);
    Object.assign(manager as unknown as { clients: typeof clients; tools: unknown[] }, {
      clients,
      tools: [{ name: "tool", inputSchema: {} }],
    });

    await manager.close();

    expect(close).toHaveBeenCalledTimes(1);
    expect(clients.size).toBe(0);
    expect(manager.toolsForOpenai()).toEqual([]);
  });

  it("separates model text, structured output, and local disposal", async () => {
    const manager = new McpManager();
    const dispose = vi.fn(async () => undefined);
    manager.addLocalTools([
      {
        name: "read",
        inputSchema: { type: "object" },
        dispose,
        call: async () =>
          localToolResult("model-facing text", {
            type: "text",
            file: { filePath: "README.md", content: "structured" },
          }),
      },
    ]);

    await expect(manager.callTool("read", {})).resolves.toEqual({
      content: "model-facing text",
      structuredContent: {
        type: "text",
        file: { filePath: "README.md", content: "structured" },
      },
      isError: false,
    });
    await manager.close();
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it("conditionally projects paginated Claude MCP resources", async () => {
    const disconnected = new McpManager();
    disconnected.addClaudeMcpResourceTools();
    expect(disconnected.toolsForOpenai()).toEqual([]);

    const manager = new McpManager();
    const listResources = vi
      .fn()
      .mockResolvedValueOnce({
        resources: [
          {
            uri: "docs://one",
            name: "One",
            mimeType: "text/plain",
            description: "First resource",
          },
        ],
        nextCursor: "page-2",
      })
      .mockResolvedValueOnce({ resources: [{ uri: "docs://two", name: "Two" }] });
    const clients = new Map([["docs", { listResources, readResource: vi.fn(), close: vi.fn() }]]);
    Object.assign(manager as unknown as { clients: typeof clients }, { clients });
    manager.addClaudeMcpResourceTools();

    expect(manager.toolsForOpenai().map((tool) => tool.function.name)).toEqual([
      "ListMcpResourcesTool",
      "ReadMcpResourceTool",
    ]);
    await expect(manager.callTool("ListMcpResourcesTool", { server: "docs" })).resolves.toEqual({
      content: "docs: docs://one (One)\ndocs: docs://two (Two)",
      structuredContent: [
        {
          uri: "docs://one",
          name: "One",
          mimeType: "text/plain",
          description: "First resource",
          server: "docs",
        },
        { uri: "docs://two", name: "Two", server: "docs" },
      ],
      isError: false,
    });
    expect(listResources).toHaveBeenNthCalledWith(1, undefined, undefined);
    expect(listResources).toHaveBeenNthCalledWith(2, { cursor: "page-2" }, undefined);
    await expect(manager.callTool("ListMcpResourcesTool", { server: "missing" })).rejects.toThrow(
      /not connected/i,
    );
  });

  it("reads Claude MCP text resources with request cancellation context", async () => {
    const manager = new McpManager();
    const readResource = vi
      .fn()
      .mockResolvedValueOnce({
        contents: [{ uri: "docs://one", mimeType: "text/plain", text: "resource text" }],
      })
      .mockResolvedValueOnce({
        contents: [{ uri: "docs://image", mimeType: "image/png", blob: "aW1hZ2U=" }],
      });
    const clients = new Map([["docs", { listResources: vi.fn(), readResource, close: vi.fn() }]]);
    Object.assign(manager as unknown as { clients: typeof clients }, { clients });
    manager.addClaudeMcpResourceTools();

    await expect(
      withAcpRequest("mcp-resource-read", () =>
        manager.callTool("ReadMcpResourceTool", { server: "docs", uri: "docs://one" }),
      ),
    ).resolves.toEqual({
      content: "resource text",
      structuredContent: {
        contents: [{ uri: "docs://one", mimeType: "text/plain", text: "resource text" }],
      },
      isError: false,
    });
    expect(readResource.mock.calls[0]?.[1]?.signal).toBeInstanceOf(AbortSignal);
    await expect(
      manager.callTool("ReadMcpResourceTool", { server: "docs", uri: "docs://image" }),
    ).resolves.toMatchObject({
      content: expect.stringContaining("binary MCP resource item is unsupported"),
      structuredContent: {
        contents: [],
        error: expect.stringContaining("binary MCP resource item is unsupported"),
      },
      isError: true,
    });
  });

  it("starts MCP connections concurrently and aborts pending work on close", async () => {
    const aborted: string[] = [];
    const connectServer = vi.fn(
      (name: string, _config: unknown, signal: AbortSignal) =>
        new Promise<McpConnectionResult>((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => {
              aborted.push(name);
              reject(signal.reason);
            },
            { once: true },
          );
        }),
    );
    const manager = new McpManager({ connectServer });

    manager.startServer("one", stdioConfig());
    manager.startServer("two", stdioConfig());
    expect(connectServer).toHaveBeenCalledTimes(2);

    await manager.close();

    expect(aborted.sort()).toEqual(["one", "two"]);
    expect(manager.toolsForOpenai()).toEqual([]);
  });

  it("searches and activates only matching deferred MCP schemas", async () => {
    const manager = new McpManager({
      connectServer: async () =>
        fakeConnection([
          { name: "list_issues", description: "List repository issues" },
          { name: "create_issue", description: "Create a repository issue" },
          { name: "read_repository", description: "Read repository metadata" },
        ]),
    });
    manager.startServer("github", stdioConfig());
    manager.addClaudeMcpDiscoveryTools();
    await manager.addServer("github", stdioConfig());

    expect(toolNames(manager)).toEqual(["ToolSearch"]);
    await expect(
      manager.callTool("ToolSearch", {
        query: "select:mcp__github__create_issue,mcp__github__missing",
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        matches: ["mcp__github__create_issue"],
        query: "select:mcp__github__create_issue,mcp__github__missing",
        total_deferred_tools: 3,
      },
      isError: false,
    });
    expect(toolNames(manager)).toEqual(["ToolSearch", "mcp__github__create_issue"]);
    await expect(manager.callTool("mcp__github__list_issues", {})).rejects.toThrow(/deferred/i);

    await expect(
      manager.callTool("ToolSearch", { query: "issues", max_results: 1 }),
    ).resolves.toMatchObject({
      structuredContent: {
        matches: ["mcp__github__list_issues"],
        total_deferred_tools: 3,
      },
    });
    expect(toolNames(manager)).toEqual([
      "ToolSearch",
      "mcp__github__list_issues",
      "mcp__github__create_issue",
    ]);
    await expect(
      manager.callTool("ToolSearch", { query: "issues", max_results: 21 }),
    ).rejects.toThrow(/1 to 20/);
  });

  it("reports pending MCP servers when no deferred schema matches", async () => {
    const manager = new McpManager({
      connectServer: (_name, _config, signal) =>
        new Promise<McpConnectionResult>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        }),
    });
    manager.startServer("slow", stdioConfig());
    manager.addClaudeMcpDiscoveryTools();

    await expect(manager.callTool("ToolSearch", { query: "issues" })).resolves.toMatchObject({
      structuredContent: {
        matches: [],
        query: "issues",
        total_deferred_tools: 0,
        pending_mcp_servers: ["slow"],
      },
    });
    await manager.close();
  });

  it("waits for named servers and returns every Claude status bucket", async () => {
    const manager = new McpManager({
      waitTimeoutMs: 10,
      connectServer: (name, _config, signal) => {
        if (name === "ready") {
          return Promise.resolve(fakeConnection([{ name: "lookup", description: "Lookup" }]));
        }
        if (name === "broken") return Promise.reject(new Error("connect failed"));
        return new Promise<McpConnectionResult>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      },
    });
    for (const name of ["ready", "broken", "slow"]) manager.startServer(name, stdioConfig());
    manager.addClaudeMcpDiscoveryTools();
    await manager.addServer("ready", stdioConfig());
    await expect(manager.addServer("broken", stdioConfig())).rejects.toThrow("connect failed");

    await expect(
      manager.callTool("WaitForMcpServers", {
        servers: ["ready", "broken", "slow", "missing"],
      }),
    ).resolves.toMatchObject({
      structuredContent: {
        ready: false,
        connected: ["ready"],
        failed: ["broken"],
        stillPending: ["slow"],
        needsAuth: [],
        disabled: [],
        unknown: ["missing"],
      },
      isError: false,
    });
    expect(toolNames(manager)).toContain("mcp__ready__lookup");
    await manager.close();
  });

  it("namespaces equal MCP names and dispatches only to the owning client", async () => {
    const alphaCall = vi.fn().mockResolvedValue({ content: [{ type: "text", text: "alpha" }] });
    const betaCall = vi.fn().mockResolvedValue({ content: [{ type: "text", text: "beta" }] });
    const manager = new McpManager({
      connectServer: async (name) =>
        fakeConnection(
          [{ name: "lookup", description: `${name} lookup` }],
          name === "alpha" ? alphaCall : betaCall,
        ),
    });
    manager.startServer("alpha", stdioConfig());
    manager.startServer("beta", stdioConfig());
    manager.addClaudeMcpDiscoveryTools();
    await manager.addServer("alpha", stdioConfig());
    await manager.addServer("beta", stdioConfig());
    await manager.callTool("ToolSearch", { query: "select:mcp__alpha__lookup" });

    await expect(manager.callTool("mcp__alpha__lookup", { id: 7 })).resolves.toMatchObject({
      content: "alpha",
      isError: false,
    });
    expect(alphaCall).toHaveBeenCalledWith(
      { name: "lookup", arguments: { id: 7 } },
      undefined,
      undefined,
    );
    expect(betaCall).not.toHaveBeenCalled();
  });

  it("verifies a required server identity and exact tool surface before publishing tools", async () => {
    const close = vi.fn().mockResolvedValue(undefined);
    const manager = new McpManager({
      connectServer: async () => ({
        serverInfo: { name: "wrong-project-service", version: "1" },
        client: {
          callTool: vi.fn(),
          close,
        } as unknown as McpConnectionResult["client"],
        tools: [{ name: "project_bootstrap", inputSchema: { type: "object" } }],
        close,
      }),
    });
    await expect(
      manager.addServer("registry", stdioConfig(), {
        name: "biology-project-service",
        version: "1",
        tools: ["project_bootstrap"],
      }),
    ).rejects.toThrow(/identity/i);
    expect(manager.toolsForNative()).toEqual([]);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("rejects a verification contract added after tools were published", async () => {
    const close = vi.fn().mockResolvedValue(undefined);
    const manager = new McpManager({
      connectServer: async () => ({
        serverInfo: { name: "biology-project-service", version: "1" },
        client: {
          callTool: vi.fn(),
          close,
        } as unknown as McpConnectionResult["client"],
        tools: [{ name: "project_bootstrap", inputSchema: { type: "object" } }],
        close,
      }),
    });
    await manager.addServer("registry", stdioConfig());
    expect(manager.toolsForNative()).not.toEqual([]);

    await expect(
      manager.addServer("registry", stdioConfig(), {
        name: "biology-project-service",
        version: "1",
        tools: ["project_bootstrap"],
        hostOnlyTools: ["project_bootstrap"],
      }),
    ).rejects.toThrow(/different verification contract/i);
    expect(manager.toolsForNative()).toEqual([]);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("tracks an invalidated server close rejection until manager shutdown", async () => {
    const closeStarted = deferred<void>();
    let rejectClose!: (error: Error) => void;
    const closeResult = new Promise<void>((_resolve, reject) => {
      rejectClose = reject;
    });
    const close = vi.fn(() => {
      closeStarted.resolve();
      return closeResult;
    });
    const manager = new McpManager({
      connectServer: async () => ({
        serverInfo: { name: "biology-project-service", version: "1" },
        client: {
          callTool: vi.fn(),
          close: vi.fn().mockResolvedValue(undefined),
        } as unknown as McpConnectionResult["client"],
        tools: [{ name: "project_bootstrap", inputSchema: { type: "object" } }],
        close,
      }),
    });
    await manager.addServer("registry", stdioConfig());
    await expect(
      manager.addServer("registry", stdioConfig(), {
        name: "biology-project-service",
        version: "1",
        tools: ["project_bootstrap"],
      }),
    ).rejects.toThrow(/different verification contract/i);
    await closeStarted.promise;

    let shutdownSettled = false;
    const shutdown = manager.close().then(() => {
      shutdownSettled = true;
    });
    await Promise.resolve();
    expect(shutdownSettled).toBe(false);

    rejectClose(new Error("close failed"));
    await expect(shutdown).resolves.toBeUndefined();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("times out a required-style connection that never initializes", async () => {
    const manager = new McpManager({
      connectServer: async (_name, _config, signal) =>
        new Promise<never>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        }),
    });

    await expect(
      manager.addServer("registry", stdioConfig(), undefined, { timeoutMs: 5 }),
    ).rejects.toThrow(/timed out/i);
    expect(manager.toolsForNative()).toEqual([]);
    await expect(manager.close()).resolves.toBeUndefined();
  });

  it("calls a verified server tool directly even when Claude schemas are deferred", async () => {
    const snapshot = { schemaVersion: 1, projectId: "project-1" };
    const callTool = vi.fn().mockResolvedValue({
      content: [{ type: "text", text: JSON.stringify(snapshot) }],
      structuredContent: snapshot,
    });
    const manager = new McpManager({
      connectServer: async () => ({
        ...fakeConnection([{ name: "project_bootstrap" }], callTool),
        serverInfo: { name: "biology-project-service", version: "1" },
      }),
    });
    manager.startServer("registry", stdioConfig(), {
      name: "biology-project-service",
      version: "1",
      tools: ["project_bootstrap"],
      hostOnlyTools: ["project_bootstrap"],
    });
    manager.addClaudeMcpDiscoveryTools();
    await manager.addServer("registry", stdioConfig(), {
      name: "biology-project-service",
      version: "1",
      tools: ["project_bootstrap"],
      hostOnlyTools: ["project_bootstrap"],
    });

    expect(toolNames(manager)).toEqual([]);
    await expect(manager.callTool("mcp__registry__project_bootstrap", {})).rejects.toThrow(
      /host-side/i,
    );
    await expect(
      manager.callServerTool(
        "registry",
        "project_bootstrap",
        { projectId: "project-1" },
        { timeoutMs: 5_000 },
      ),
    ).resolves.toEqual({
      content: JSON.stringify(snapshot),
      structuredContent: snapshot,
      isError: false,
      rawMcpContentBlocks: [{ type: "text", text: JSON.stringify(snapshot) }],
    });
    expect(callTool).toHaveBeenCalledWith(
      { name: "project_bootstrap", arguments: { projectId: "project-1" } },
      undefined,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });
});

function stdioConfig() {
  return { type: "stdio" as const, command: "unused" };
}

function fakeConnection(
  tools: Array<{ name: string; description?: string }>,
  callTool = vi.fn(),
): McpConnectionResult {
  return {
    serverInfo: { name: "test-server", version: "1" },
    client: {
      callTool,
      close: vi.fn().mockResolvedValue(undefined),
    } as unknown as McpConnectionResult["client"],
    tools: tools.map((tool) => ({ ...tool, inputSchema: { type: "object" } })),
    close: vi.fn().mockResolvedValue(undefined),
  };
}

function toolNames(manager: McpManager): string[] {
  return manager.toolsForOpenai().map((tool) => tool.function.name);
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value?: T | PromiseLike<T>) => void;
} {
  let resolve!: (value?: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

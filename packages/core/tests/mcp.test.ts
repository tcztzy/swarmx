import { describe, expect, it, vi } from "vitest";
import { RequestCancelledError, cancelAcpRequest, withAcpRequest } from "../src/acp.js";
import { McpManager } from "../src/mcp.js";

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
    Object.assign(manager as unknown as { clients: typeof clients }, { clients });

    const run = withAcpRequest("mcp-cancel", () => manager.callTool("slow", {}));
    await started.promise;
    await expect(cancelAcpRequest("mcp-cancel")).resolves.toBe(true);

    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
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
});

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

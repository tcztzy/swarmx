import { describe, expect, it, vi } from "vitest";
import { MemoryRuntimeHost } from "./memory-runtime-host.js";

const launch = {
  program: "/Applications/SwarmX.app/Contents/Resources/mem-runtime/swarmx-mem",
  args: ["serve", "--root", "/tmp/memory", "--stdio"],
  cwd: "/tmp",
  env: { HOME: "/tmp", PATH: "/usr/bin", RUST_BACKTRACE: "0" },
  binaryDigest: `sha256:${"a".repeat(64)}`,
  protocolVersion: 1 as const,
  runtimeVersion: "0.1.0",
  memoryRoot: "/tmp/memory",
};

describe("MemoryRuntimeHost", () => {
  it("keeps the sidecar private and validates operation-matched responses", async () => {
    const callTool = vi.fn(async ({ arguments: arguments_ }) => {
      const structuredContent = {
        protocolVersion: 1,
        operation: arguments_.operation,
        ok: true,
        result: { pages: [] },
      };
      return {
        content: [{ type: "text" as const, text: JSON.stringify(structuredContent) }],
        structuredContent,
        isError: false,
      };
    });
    const close = vi.fn(async () => {});
    const connect = vi.fn(async () => ({
      serverInfo: () => ({ name: "swarmx-mem", version: "0.1.0" }),
      listTools: async () => [{ name: "swarmx_memory" }],
      callTool,
      close,
    }));
    const host = new MemoryRuntimeHost({ launch, connect });

    await expect(host.request({ protocolVersion: 1, operation: "list" })).resolves.toEqual({
      pages: [],
    });
    expect(connect).toHaveBeenCalledOnce();
    expect(callTool).toHaveBeenCalledWith({
      name: "swarmx_memory",
      arguments: { protocolVersion: 1, operation: "list" },
    });

    await host.close();
    expect(close).toHaveBeenCalledOnce();
  });

  it("rejects malformed, mismatched, oversized, and text-leaking failures", async () => {
    const responses = [
      { protocolVersion: 2, operation: "list", ok: true, result: { pages: [] } },
      { protocolVersion: 1, operation: "get", ok: true, result: { page: null } },
      {
        protocolVersion: 1,
        operation: "list",
        ok: false,
        error: { code: "internal", message: "private Markdown body" },
      },
    ];
    const connect = vi.fn(async () => ({
      serverInfo: () => ({ name: "swarmx-mem", version: "0.1.0" }),
      listTools: async () => [{ name: "swarmx_memory" }],
      callTool: vi.fn(async () => {
        const structuredContent = responses.shift();
        return {
          content: [{ type: "text" as const, text: JSON.stringify(structuredContent) }],
          structuredContent,
          isError: false,
        };
      }),
      close: vi.fn(async () => {}),
    }));
    const host = new MemoryRuntimeHost({ launch, connect });

    await expect(host.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      /invalid response/i,
    );
    await expect(host.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      /operation mismatch/i,
    );
    await expect(host.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      "Memory runtime internal failed",
    );

    const oversized = new MemoryRuntimeHost({
      launch,
      connect: async () => {
        const structuredContent = { padding: "x".repeat(12 * 1_048_576 + 1) };
        return {
          serverInfo: () => ({ name: "swarmx-mem", version: "0.1.0" }),
          listTools: async () => [{ name: "swarmx_memory" }],
          callTool: async () => ({
            content: [{ type: "text" as const, text: JSON.stringify(structuredContent) }],
            structuredContent,
            isError: false,
          }),
          close: async () => {},
        };
      },
    });
    await expect(oversized.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      /exceeds/i,
    );
  });

  it("fails closed after a connection error and can be disposed idempotently", async () => {
    const connect = vi.fn(async () => {
      throw new Error("spawn failed with secret stderr");
    });
    const host = new MemoryRuntimeHost({ launch, connect });

    await expect(host.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      "Memory runtime is unavailable",
    );
    await host.close();
    await host.close();
    expect(connect).toHaveBeenCalledOnce();
  });

  it("rejects a server with a different identity or tool surface", async () => {
    const close = vi.fn(async () => {});
    const host = new MemoryRuntimeHost({
      launch,
      connect: async () => ({
        serverInfo: () => ({ name: "other-memory", version: "0.1.0" }),
        listTools: async () => [{ name: "swarmx_memory" }],
        callTool: async () => ({ content: [] }),
        close,
      }),
    });

    await expect(host.request({ protocolVersion: 1, operation: "list" })).rejects.toThrow(
      "Memory runtime is unavailable",
    );
    expect(close).toHaveBeenCalledOnce();
  });
});

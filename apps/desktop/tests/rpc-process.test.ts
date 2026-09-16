import { expect, it, vi } from "vitest";
import { rpcProcess } from "../src/agents/rpc-process.js";

it("rejects pending calls on process exit and reaps the owned child", async () => {
  const failed = vi.fn();
  const rpc = rpcProcess(
    process.execPath,
    ["-e", "process.stdin.once('data', () => process.exit(7))"],
    process.cwd(),
    async () => {},
    failed,
  );
  await expect(rpc.request("pending", {})).rejects.toThrow(/exited.*7/);
  expect(rpc.signal.aborted).toBe(true);
  expect(failed).toHaveBeenCalledOnce();
  await rpc.dispose();
});

it("rejects malformed native frames instead of leaving a call pending", async () => {
  const rpc = rpcProcess(
    process.execPath,
    ["-e", "process.stdin.once('data', () => console.log('not-json'))"],
    process.cwd(),
    async () => {},
    () => {},
  );
  try {
    await expect(rpc.request("pending", {})).rejects.toThrow();
    expect(rpc.signal.aborted).toBe(true);
  } finally {
    await rpc.dispose();
  }
});

it("reaches a descendant that keeps the inherited stdout pipe open", async () => {
  const rpc = rpcProcess(
    process.execPath,
    [
      "-e",
      "const { spawn } = require('node:child_process');" +
        "spawn(process.execPath, ['-e', 'setTimeout(() => process.exit(0), 15000)']," +
        " { stdio: ['ignore', 'inherit', 'inherit'] });" +
        "setInterval(() => {}, 1000)",
    ],
    process.cwd(),
    async () => {},
    () => {},
  );
  const timeout = new Promise((resolve) => setTimeout(() => resolve("timeout"), 3_000));
  await expect(Promise.race([rpc.dispose(), timeout])).resolves.toBeUndefined();
});

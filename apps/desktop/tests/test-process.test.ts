import { afterEach, expect, it } from "vitest";
import { spawnProcess } from "../../../scripts/test-process.js";

const processes: ReturnType<typeof spawnProcess>[] = [];

function run(source: string, options: { signal?: AbortSignal } = {}) {
  const child = spawnProcess({
    argv: [process.execPath, "-e", source],
    graceMs: 100,
    stdin: "ignore",
    stdout: { maxBytes: 5 },
    stderr: { maxBytes: 64 },
    ...options,
  });
  processes.push(child);
  return child;
}

afterEach(async () => {
  for (const child of processes.splice(0)) {
    child.terminate();
    await child.done;
  }
});

it("collects bounded subprocess output and preserves the exit outcome", async () => {
  const child = run('process.stdout.write("0123456789"); process.stderr.write("diagnostic");');
  await expect(child.done).resolves.toEqual({ exitCode: 0, signal: null });
  expect(child.stdout?.readFrom(0)).toEqual({ text: "56789", nextOffset: 10, lossy: true });
  expect(child.stdout?.readFrom(7)).toEqual({ text: "789", nextOffset: 10, lossy: false });
  expect(child.stderr?.readFrom(0)).toEqual({
    text: "diagnostic",
    nextOffset: 10,
    lossy: false,
  });
});

it("rejects a pre-aborted process before spawning", () => {
  const controller = new AbortController();
  controller.abort(new Error("stop before spawn"));
  expect(() => run('throw new Error("must not run")', { signal: controller.signal })).toThrow(
    "stop before spawn",
  );
  expect(processes).toHaveLength(0);
});

it.skipIf(process.platform === "win32")(
  "terminates an aborting subprocess even when it ignores SIGTERM",
  async () => {
    const controller = new AbortController();
    const child = run(
      'process.on("SIGTERM", () => {}); process.stdout.write("ready"); setInterval(() => {}, 1000);',
      { signal: controller.signal },
    );
    await expect.poll(() => child.stdout?.readFrom(0).text).toBe("ready");
    controller.abort();
    await expect(child.done).resolves.toEqual({ exitCode: null, signal: "SIGKILL" });
  },
);

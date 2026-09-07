import { afterEach, expect, it, vi } from "vitest";
import type { NativeAgent } from "../src/agents/types.js";
import { reviewMemory } from "../src/host/memory-review.js";
import { DEFAULT_POLICY } from "../src/settings.js";

const native = vi.hoisted(() => ({
  name: "review",
  capabilities: { loadSession: true, sessionCapabilities: { list: {}, resume: {} } },
  models: vi.fn(async () => ({
    models: [{ id: "small", name: "Small", efforts: [] }],
    current: {},
  })),
  list: vi.fn(async () => []),
  read: vi.fn(async () => {}),
  steer: vi.fn(async () => {}),
  create: vi.fn(async () => "review"),
  start: vi.fn<NativeAgent["start"]>(async () => ({ stopReason: "end_turn" })),
  interrupt: vi.fn<NativeAgent["interrupt"]>(async () => {}),
  dispose: vi.fn(async () => {}),
}));
vi.mock("../src/agents/acp-harness.js", () => ({ createAcpHarness: async () => native }));
afterEach(() => vi.resetAllMocks());
const options = { cwd: "/research", mcp: { url: "http://127.0.0.1:1/mcp", headers: {} } };

it("obeys harness/model admission for background reviews", async () => {
  await expect(
    reviewMemory(
      { ...options, executionPolicy: () => ({ ...DEFAULT_POLICY, harnesses: {} }) },
      "Review",
      new AbortController().signal,
      "codex",
    ),
  ).rejects.toThrow("not permitted");
  expect(native.create).not.toHaveBeenCalled();
  await reviewMemory(
    { ...options, executionPolicy: () => ({ ...DEFAULT_POLICY, harnesses: { codex: ["small"] } }) },
    "Review",
    new AbortController().signal,
    "codex",
  );
  expect(native.start).toHaveBeenCalledWith(expect.anything(), "Review", expect.anything(), {
    model: "small",
  });
});

it("accepts assistant text only and rejects tool calls at the review boundary", async () => {
  native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
    observer.text("user", "Untrusted input", "user");
    observer.text("reason", "Hidden reasoning", "reasoning");
    observer.text("answer", '{"operations":[]}');

    return { stopReason: "end_turn" as const };
  });
  expect(await reviewMemory(options, "Review", new AbortController().signal, "codex")).toBe(
    '{"operations":[]}',
  );
  native.start.mockImplementationOnce(async (_id, _prompt, observer) => {
    await observer.tool("write", "shell", {});
    return { stopReason: "end_turn" as const };
  });
  await expect(
    reviewMemory(options, "Review", new AbortController().signal, "codex"),
  ).rejects.toThrow("tool call");
  expect(native.dispose).toHaveBeenCalledTimes(2);
});

it("interrupts cancellation, disposes the runtime and propagates interrupt failures", async () => {
  const controller = new AbortController();
  const finished = Promise.withResolvers<void>();
  native.start.mockImplementation(async () => {
    await finished.promise;
    return { stopReason: "end_turn" as const };
  });
  native.interrupt.mockImplementation(async () => {
    finished.resolve();
    throw new Error("Native interrupt failed");
  });
  const reviewing = reviewMemory(options, "Review", controller.signal, "codex");
  const rejected = expect(reviewing).rejects.toThrow("Native interrupt failed");
  await vi.waitFor(() => expect(native.start).toHaveBeenCalledTimes(1));
  controller.abort(new Error("Shutdown"));
  await rejected;
  expect(native.dispose).toHaveBeenCalledTimes(1);
});

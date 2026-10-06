import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, expect, it, vi } from "vitest";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
  vi.restoreAllMocks();
});
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-wiki-sdk-"));
  const brain = join(root, "brain");
  await mkdir(join(brain, "wiki"), { recursive: true });
  const command = join(root, "wiki-memory.mjs");
  await writeFile(
    command,
    `#!${process.execPath}\n${await readFile(new URL("./fixtures/wiki-memory-server.mjs", import.meta.url), "utf8")}`,
    { mode: 0o700 },
  );
  const log = join(root, "calls.jsonl");
  await writeFile(log, "");
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
    wikiMemory: {
      command,
      brain,
      scopes: [root],
      roots: [join(brain, "wiki")],
      env: { HOST_WIKI_TEST_DISPATCH_LOG: log },
    },
  });
  cleanups.push(async () => {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  return { products, log };
}
const call = (query: string, signal = new AbortController().signal) =>
  [
    "memory",
    { action: "search_wiki_memory", request: { query } },
    { actorId: "test", callId: "sdk", signal },
  ] as const;

it("closes a warmed real SDK connection without throwing from settled request abort listeners", async () => {
  const { products, log } = await fixture();
  const send = vi.spyOn(StdioClientTransport.prototype, "send");
  const result = await products.callTool(...call("synthetic"));
  expect(result).toMatchObject({
    data: {
      status: "available",
      records: [expect.objectContaining({ documentId: "knowledge/synthetic.md" })],
    },
  });
  const sent = send.mock.calls.length;
  await products.dispose();
  await new Promise((resolve) => setImmediate(resolve));
  expect(
    send.mock.calls
      .slice(sent)
      .every(([message]) => "method" in message && message.method === "notifications/cancelled"),
  ).toBe(true);
  expect((await readFile(log, "utf8")).trim().split("\n")).toHaveLength(1);
  await products.dispose();
});

it("settles an in-flight real SDK request on caller cancellation and reuses the process", async () => {
  const { products, log } = await fixture();
  await products.callTool(...call("synthetic"));
  const stopped = new AbortController();
  const pending = products.callTool(...call("wait", stopped.signal));
  const rejected = expect(pending).rejects.toThrow("cancelled");
  await vi.waitFor(async () =>
    expect((await readFile(log, "utf8")).trim().split("\n")).toHaveLength(2),
  );
  stopped.abort();
  await rejected;
  expect(products.busy).toBe(false);
  await products.callTool(...call("synthetic"));
  const requests = (await readFile(log, "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  expect(requests).toHaveLength(3);
  expect(requests.every((request) => request.name === "search_memory")).toBe(true);
});

it("settles pending real SDK work and releases the subprocess on Host shutdown", async () => {
  const { products, log } = await fixture();
  const start = StdioClientTransport.prototype.start;
  let transport: StdioClientTransport | undefined;
  vi.spyOn(StdioClientTransport.prototype, "start").mockImplementation(function (
    this: StdioClientTransport,
  ) {
    transport = this;
    return start.call(this);
  });
  const pending = products.callTool(...call("wait"));
  const rejected = expect(pending).rejects.toThrow("cancelled");
  await vi.waitFor(async () => expect((await readFile(log, "utf8")).trim()).not.toBe(""));
  const pid = transport?.pid;
  expect(pid).toBeTypeOf("number");
  await products.dispose();
  await rejected;
  expect(products.busy).toBe(false);
  expect(() => process.kill(pid as number, 0)).toThrow();
});

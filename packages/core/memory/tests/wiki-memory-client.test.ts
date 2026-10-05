import { expect, it, vi } from "vitest";
import { WikiMemoryClient, WikiMemoryError } from "../src/wiki-memory-client.js";

const signal = () => new AbortController().signal;
const envelope = (data: unknown) => ({ content: [{ type: "text", text: JSON.stringify(data) }] });
const record = {
  datasetId: "knowledge",
  documentId: "knowledge/synthetic.md",
  documentName: "synthetic.md",
  score: 0.75,
  priority: "P1",
  content: "Synthetic untrusted reference text",
};
const search = { query: "synthetic", totalRecords: 1, records: [record] };
const write = {
  target: "brain",
  write: {
    datasetId: "knowledge",
    name: "synthetic.md",
    text: "This is only a synthetic memory fixture.",
  },
};

function setup(response: unknown = envelope(search)) {
  const callTool = vi.fn().mockResolvedValue(response);
  const client = new WikiMemoryClient({ scopes: ["/synthetic/workspace"] }, { callTool });
  return { client, callTool };
}

it("uses explicit cloned scopes and bounded MCP excerpts without configuring a backend", async () => {
  const scopes = ["/synthetic/workspace"];
  const callTool = vi.fn().mockResolvedValue(envelope(search));
  const client = new WikiMemoryClient({ scopes }, { callTool });
  scopes.push("/unrelated");
  const abort = signal();
  expect(await client.search({ query: " synthetic " }, abort)).toEqual(search);
  expect(callTool).toHaveBeenCalledExactlyOnceWith(
    {
      name: "search_memory",
      arguments: {
        scopes: ["/synthetic/workspace"],
        query: "synthetic",
        maxResults: 8,
        maxChars: 600,
        fullContent: false,
      },
    },
    abort,
  );
  await client.read({ query: "synthetic", maxResults: 1, maxChars: 2_000 }, abort);
  expect(callTool.mock.calls[1]?.[0].arguments).toMatchObject({
    fullContent: false,
    maxResults: 1,
    maxChars: 2_000,
    sections: ["frontmatter", "body"],
  });
});

it("keeps read excerpts bounded by the server clamp", async () => {
  const callTool = vi.fn().mockImplementation(async ({ arguments: request }) => {
    const body = "synthetic ".repeat(500);
    return envelope({
      ...search,
      truncated: body.length > request.maxChars,
      records: [
        { ...record, content: request.fullContent ? body : body.slice(0, request.maxChars) },
      ],
    });
  });
  const client = new WikiMemoryClient({ scopes: ["/synthetic"] }, { callTool });
  const result = await client.read({ query: "synthetic", maxChars: 80 }, signal());
  expect(result.records[0]?.content?.length).toBeLessThanOrEqual(80);
  expect(result.truncated).toBe(true);
});

it("accepts omitted content only for an explicit frontmatter-only view", async () => {
  const { content: _content, ...frontmatter } = record;
  const result = { ...search, records: [{ ...frontmatter, brief: { title: "Synthetic" } }] };
  const { client, callTool } = setup(envelope(result));
  for (const method of ["search", "read"] as const) {
    expect(
      await client[method]({ query: "synthetic", sections: ["frontmatter"] }, signal()),
    ).toEqual(result);
    await expect(client[method]({ query: "synthetic" }, signal())).rejects.toMatchObject({
      outcome: "unknown",
    });
  }
  expect(callTool).toHaveBeenCalledTimes(4);
});

it("retains partial and truncation diagnostics instead of claiming a complete read", async () => {
  const result = {
    ...search,
    errors: [{ datasetId: "another", message: "Synthetic unavailable category" }],
    partial: { omitted: 3 },
    records: [
      { ...record, contentTruncated: true, originalChars: 4_000, resolvedRoot: "/synthetic" },
    ],
  };
  const { client } = setup(envelope(result));
  expect(await client.read({ query: "synthetic" }, signal())).toEqual(result);
});

it("preserves exact absolute scope spelling instead of trimming filesystem identities", async () => {
  const callTool = vi.fn().mockResolvedValue(envelope(search));
  const client = new WikiMemoryClient({ scopes: ["/synthetic/workspace "] }, { callTool });
  await client.search({ query: "synthetic" }, signal());
  expect(callTool.mock.calls[0]?.[0].arguments.scopes).toEqual(["/synthetic/workspace "]);
});

it("requires nonempty absolute scopes, target and supported bounded fields before dispatch", async () => {
  for (const scopes of [[], ["relative"], ["/unsafe\0path"]]) {
    expect(() => new WikiMemoryClient({ scopes }, { callTool: vi.fn() })).toThrow();
  }
  const { client, callTool } = setup();
  for (const request of [
    { query: "" },
    { query: "x", maxResults: 51 },
    { query: "x", maxChars: 79 },
  ]) {
    await expect(client.search(request, signal())).rejects.toThrow();
  }
  await expect(client.write({ ...write, target: " " }, signal())).rejects.toThrow();
  await expect(
    client.write({ ...write, write: { ...write.write, text: "short" } }, signal()),
  ).rejects.toThrow();
  await expect(
    client.write({ ...write, write: { ...write.write, allowDuplicate: true } } as never, signal()),
  ).rejects.toThrow();
  expect(callTool).not.toHaveBeenCalled();
});

it("never synthesizes consent or quality/duplicate bypasses and preserves write receipts", async () => {
  const receipt = { ok: true, documentId: record.documentId, replaced: false };
  const { client, callTool } = setup(envelope(receipt));
  expect(await client.write(write, signal())).toEqual(receipt);
  expect(callTool.mock.calls[0]?.[0]).toEqual({
    name: "write_memory",
    arguments: { ...write, scopes: ["/synthetic/workspace"] },
  });
  await client.write({ ...write, userRequested: false }, signal());
  expect(callTool.mock.calls[1]?.[0].arguments.gate).toEqual({ userRequested: false });
  await client.write({ ...write, userRequested: true }, signal());
  expect(callTool.mock.calls[2]?.[0].arguments.gate).toEqual({ userRequested: true });
});

it.each([
  "write-gate-refused",
  "quality-judge-unavailable",
  "quality-judge-rejected",
  "duplicate-suspected",
  "inline-body-too-large",
])("surfaces %s without retrying", async (error) => {
  const refusal = { ok: false, error, message: "Synthetic refusal" };
  const { client, callTool } = setup(envelope(refusal));
  await expect(client.write(write, signal())).rejects.toMatchObject({
    outcome: "refused",
    response: refusal,
  });
  expect(callTool).toHaveBeenCalledTimes(1);
});

it.each([
  { isError: true, content: [{ type: "text", text: "MCP error: synthetic refusal" }] },
  { content: [{ type: "text", text: "invalid JSON" }] },
  envelope({ ok: true }),
  { ...envelope({ ok: true, documentId: record.documentId }), isError: true },
  envelope({ ok: false, error: "synthetic-unknown-failure" }),
  { content: [{ type: "image", data: "synthetic" }] },
  {
    content: [
      { type: "text", text: "{}" },
      { type: "text", text: "{}" },
    ],
  },
  { content: [{ type: "text", text: "x".repeat(2 * 1024 * 1024 + 1) }] },
])("marks invalid or ambiguous responses as unknown without retrying", async (response) => {
  const { client, callTool } = setup(response);
  const error = await client.write(write, signal()).catch((error: unknown) => error);
  expect(error).toBeInstanceOf(WikiMemoryError);
  expect(error).toMatchObject({ outcome: "unknown" });
  expect(callTool).toHaveBeenCalledTimes(1);
});

it("makes no call for pre-aborted reads and writes", async () => {
  const controller = new AbortController();
  controller.abort();
  const { client, callTool } = setup();
  await expect(client.read({ query: "synthetic" }, controller.signal)).rejects.toThrow();
  await expect(client.write(write, controller.signal)).rejects.toThrow();
  expect(callTool).not.toHaveBeenCalled();
});

it("passes cancellation to its caller-owned transport and marks post-dispatch writes unknown", async () => {
  const controller = new AbortController();
  let release = (_value: unknown) => {};
  const callTool = vi.fn().mockImplementation((_request, passedSignal) => {
    expect(passedSignal).toBe(controller.signal);
    return new Promise((resolve) => {
      release = resolve;
    });
  });
  const client = new WikiMemoryClient({ scopes: ["/synthetic"] }, { callTool });
  const pending = client.write(write, controller.signal);
  controller.abort();
  release(envelope({ ok: true, documentId: record.documentId }));
  await expect(pending).rejects.toMatchObject({ outcome: "unknown" });
  expect(callTool).toHaveBeenCalledTimes(1);
});

it("preserves uncertain transport failure rather than claiming no write occurred", async () => {
  const { client, callTool } = setup();
  callTool.mockRejectedValue(new Error("Synthetic lost receipt"));
  await expect(client.write(write, signal())).rejects.toMatchObject({ outcome: "unknown" });
  expect(callTool).toHaveBeenCalledTimes(1);
});

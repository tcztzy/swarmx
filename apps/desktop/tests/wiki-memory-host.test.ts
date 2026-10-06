import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { afterEach, expect, it, vi } from "vitest";
import { ProductServices } from "../src/host/product-services.js";

const cleanups: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
  vi.restoreAllMocks();
});
const call = (request: unknown = { query: "synthetic" }) => ({
  action: "search_wiki_memory",
  request,
});
const context = (signal = new AbortController().signal) => ({
  actorId: "test",
  callId: "wiki-test",
  signal,
});
function envelope(records: unknown[] = [], extra: Record<string, unknown> = {}) {
  return {
    content: [
      {
        type: "text" as const,
        text: JSON.stringify({
          query: "synthetic",
          totalRecords: records.length,
          records,
          ...extra,
        }),
      },
    ],
  };
}
function hit(extra: Record<string, unknown> = {}) {
  return {
    datasetId: "knowledge",
    documentId: "knowledge/synthetic.md",
    documentName: "synthetic.md",
    score: 0.5,
    priority: "P2",
    content: "A synthetic excerpt",
    ...extra,
  };
}
async function fixture(configured = true) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-wiki-host-"));
  const brain = join(root, "brain");
  const wiki = join(brain, "wiki");
  const other = join(root, "repo", "wiki");
  await mkdir(wiki, { recursive: true });
  await mkdir(other, { recursive: true });
  const connect = vi.spyOn(Client.prototype, "connect").mockResolvedValue();
  const dispatch = vi.spyOn(Client.prototype, "callTool").mockResolvedValue(envelope([hit()]));
  const close = vi.spyOn(Client.prototype, "close").mockResolvedValue();
  const transportClose = vi.spyOn(StdioClientTransport.prototype, "close").mockResolvedValue();
  const configuration = {
    command: process.execPath,
    brain,
    scopes: [root],
    roots: [wiki, other],
    env: { MEMORY_DATA_DIR: "/override-forbidden", MEMORY_WORKSPACE_DIR: "/override-forbidden" },
  };
  const products = await ProductServices.create({
    productHome: join(root, "product"),
    cwd: root,
    ...(configured ? { wikiMemory: configuration } : {}),
  });
  cleanups.push(async () => {
    await products.dispose();
    await rm(root, { recursive: true, force: true });
  });
  return {
    root,
    brain,
    wiki,
    other,
    products,
    configuration,
    connect,
    dispatch,
    close,
    transportClose,
  };
}

it("does not connect or dispatch for unconfigured, disabled, denied or pre-aborted calls", async () => {
  const plain = await fixture(false);
  expect(await plain.products.callTool("memory", call(), context())).toMatchObject({
    action: "search_wiki_memory",
    data: { status: "unconfigured", records: [] },
  });
  expect(plain.connect).not.toHaveBeenCalled();
  const configured = await fixture();
  configured.products.settings.writeMemory({ enabled: false });
  expect(await configured.products.callTool("memory", call(), context())).toMatchObject({
    data: { status: "disabled" },
  });
  configured.products.settings.writeMemory({ enabled: true });
  configured.products.updatePolicy({ ...configured.products.settings.read().policy, tools: [] });
  await expect(configured.products.callTool("memory", call(), context())).rejects.toThrow(
    "memory.read",
  );
  const stopped = new AbortController();
  stopped.abort();
  await expect(
    configured.products.callTool("memory", call(), context(stopped.signal)),
  ).rejects.toThrow();
  expect(configured.connect).not.toHaveBeenCalled();
  expect(configured.dispatch).not.toHaveBeenCalled();
});

it("rejects launch, root, write and unbounded request fields before connecting", async () => {
  const { products, connect, dispatch } = await fixture();
  for (const request of [
    { query: "x", scopes: ["/escape"] },
    { query: "x", roots: ["/escape"] },
    { query: "x", command: "/escape" },
    { query: "x", env: {} },
    { query: "x", fullContent: true },
    { query: "x", target: "brain" },
    { query: "x", filters: {} },
    { query: "x", maxResults: 21 },
    { query: "x", maxChars: 2001 },
    { query: "x", sections: [] },
  ]) {
    await expect(products.callTool("memory", call(request), context())).rejects.toThrow();
  }
  await expect(
    products.callTool("memory", { ...call(), unexpected: true }, context()),
  ).rejects.toThrow();
  expect(connect).not.toHaveBeenCalled();
  expect(dispatch).not.toHaveBeenCalled();
});

it("allows wiki search with only memory.read and never asks for write authority", async () => {
  const { products, dispatch } = await fixture();
  products.updatePolicy({ ...products.settings.read().policy, tools: ["memory.read"] });
  expect(await products.callTool("memory", call(), context())).toMatchObject({
    data: { status: "available" },
  });
  expect(dispatch).toHaveBeenCalledOnce();
  expect(dispatch.mock.calls[0]?.[0].name).toBe("search_memory");
});

it("caps multibyte aggregate response bytes and total excerpt characters", async () => {
  const { products, dispatch } = await fixture();
  dispatch.mockResolvedValue(
    envelope(
      Array.from({ length: 20 }, (_, index) =>
        hit({ documentId: `knowledge/${index}.md`, content: "🧪".repeat(2_000) }),
      ),
    ),
  );
  const result = (await products.callTool(
    "memory",
    call({ query: "synthetic", maxResults: 20, maxChars: 2_000 }),
    context(),
  )) as { data: { records: { content: string }[]; truncated: boolean } };
  expect(
    result.data.records.reduce((count, record) => count + record.content.length, 0),
  ).toBeLessThanOrEqual(16_000);
  expect(Buffer.byteLength(JSON.stringify(result))).toBeLessThanOrEqual(32 * 1024);
  expect(result.data.truncated).toBe(true);
});

it("qualifies equal relative IDs with stable opaque sources and omits unapproved origins", async () => {
  const { products, wiki, other, dispatch, configuration } = await fixture();
  dispatch.mockResolvedValue(
    envelope([
      hit({ resolvedRoot: wiki }),
      hit({ resolvedRoot: other }),
      hit({ resolvedRoot: "/unapproved/wiki" }),
      hit({ documentId: "../escape.md" }),
    ]),
  );
  configuration.scopes[0] = "/changed-after-startup";
  configuration.roots.push("/unapproved/wiki");
  const result = (await products.callTool("memory", call(), context())) as {
    data: {
      records: { sourceId: string; documentId: string }[];
      diagnostics: unknown[];
      partial: boolean;
    };
  };
  expect(result.data.records).toHaveLength(2);
  expect(new Set(result.data.records.map((record) => record.sourceId)).size).toBe(2);
  expect(new Set(result.data.records.map((record) => record.documentId)).size).toBe(1);
  expect(result.data.partial).toBe(true);
  expect(result.data.diagnostics).toEqual(
    expect.arrayContaining([expect.objectContaining({ code: "origin-omitted" })]),
  );
  expect(JSON.stringify(result)).not.toContain(wiki);
  expect(JSON.stringify(result)).not.toContain(other);
  expect(dispatch.mock.calls[0]?.[0]).toMatchObject({
    name: "search_memory",
    arguments: { scopes: [products.options.cwd], fullContent: false },
  });
  const repeated = (await products.callTool("memory", call(), context())) as typeof result;
  expect(repeated.data.records).toEqual(result.data.records);
});

it("bounds UTF-16 excerpts, result bytes and diagnostics while retaining native flags", async () => {
  const { products, wiki, dispatch } = await fixture();
  const native = {
    resolvedRoot: wiki,
    content: "🧪".repeat(7_000),
    truncated: true,
    fullChars: 123_456,
    projectModule: `file://${wiki}`,
    nested: { root: wiki },
  };
  dispatch.mockResolvedValue(
    envelope(
      Array.from({ length: 50 }, (_, index) =>
        hit({ ...native, documentId: `knowledge/${index}.md` }),
      ),
      {
        truncated: true,
        partial: { reason: "bounded", root: wiki },
        errors: Array.from({ length: 80 }, () => ({
          message: `Failed at ${wiki} and C:\\Users\\private\\wiki; file://${wiki}`,
        })),
      },
    ),
  );
  const result = (await products.callTool(
    "memory",
    call({ query: "synthetic", maxResults: 20, maxChars: 80 }),
    context(),
  )) as {
    data: {
      records: { content: string; fullChars: number; truncated: boolean }[];
      diagnostics: unknown[];
      truncated: boolean;
      partial: boolean;
    };
  };
  expect(result.data.records.length).toBeLessThanOrEqual(20);
  for (const record of result.data.records) {
    expect(record.content.length).toBeLessThanOrEqual(80);
    expect(record.fullChars).toBe(123_456);
    expect(record.truncated).toBe(true);
  }
  expect(Buffer.byteLength(JSON.stringify(result))).toBeLessThanOrEqual(32 * 1024);
  expect(result.data.diagnostics.length).toBeLessThanOrEqual(16);
  expect(result.data).toMatchObject({ truncated: true, partial: true });
  expect(JSON.stringify(result)).not.toContain(wiki);
  expect(JSON.stringify(result)).not.toContain("C:\\Users");
  expect(JSON.stringify(result)).not.toContain("projectModule");
  expect(JSON.stringify(result)).not.toContain("nested");
});

it("redacts paths in excerpt and metadata text and supports frontmatter-only results", async () => {
  const { products, wiki, dispatch } = await fixture();
  dispatch.mockResolvedValue(
    envelope([
      hit({
        content: `Host ${wiki} file://${wiki} C:\\Users\\someone\\file`,
        documentName: wiki,
        datasetId: wiki,
        priority: wiki,
      }),
    ]),
  );
  const result = await products.callTool("memory", call(), context());
  expect(JSON.stringify(result)).not.toContain(wiki);
  expect(JSON.stringify(result)).not.toContain("C:\\Users");
  const { content: _content, ...frontmatter } = hit();
  dispatch.mockResolvedValue(envelope([frontmatter]));
  const frontmatterResult = (await products.callTool(
    "memory",
    call({ query: "synthetic", sections: ["frontmatter"] }),
    context(),
  )) as { data: { records: unknown[] } };
  expect(frontmatterResult.data.records[0]).not.toHaveProperty("content");
});

it("withholds quoted paths with spaces and Windows root-relative paths from results and journal", async () => {
  const { products, dispatch } = await fixture();
  const paths = [
    '"C:\\Users\\Jane Doe\\Documents\\secrets.md"',
    "\\Users\\private\\wiki",
    '"/Users/Jane Doe/Documents/secrets.md"',
  ];
  dispatch.mockResolvedValue(
    envelope([hit({ content: paths.join("\n") })], {
      errors: paths.map((message) => ({ message })),
    }),
  );
  const result = await products.callTool("memory", call(), context());
  for (const value of [JSON.stringify(result), JSON.stringify(products.journal.read({}))]) {
    expect(value).not.toContain("Users");
    expect(value).not.toContain("Doe");
    expect(value).not.toContain("secrets.md");
    expect(value).toContain("[path withheld]");
  }
});

it("cancels an initialization waiter before dispatch and rechecks disabled settings", async () => {
  const { products, connect, dispatch } = await fixture();
  let release!: () => void;
  connect.mockImplementation(
    () =>
      new Promise<void>((resolve) => {
        release = resolve;
      }),
  );
  const stopped = new AbortController();
  const pending = products.callTool("memory", call(), context(stopped.signal));
  expect(products.busy).toBe(true);
  await vi.waitFor(() => expect(connect).toHaveBeenCalledOnce());
  stopped.abort();
  await expect(pending).rejects.toThrow("cancelled");
  expect(products.busy).toBe(false);
  release();
  await Promise.resolve();
  expect(dispatch).not.toHaveBeenCalled();
  products.settings.writeMemory({ enabled: false });
  expect(await products.callTool("memory", call(), context())).toMatchObject({
    data: { status: "disabled" },
  });
  products.settings.writeMemory({ enabled: true });
  await products.callTool("memory", call(), context());
  expect(dispatch).toHaveBeenCalledOnce();
});

it("does not dispatch when Memory is disabled while initialization is pending", async () => {
  const { products, connect, dispatch } = await fixture();
  let release!: () => void;
  connect.mockImplementation(
    () =>
      new Promise<void>((resolve) => {
        release = resolve;
      }),
  );
  const pending = products.callTool("memory", call(), context());
  await vi.waitFor(() => expect(connect).toHaveBeenCalledOnce());
  products.settings.writeMemory({ enabled: false });
  release();
  expect(await pending).toMatchObject({ data: { status: "disabled" } });
  expect(dispatch).not.toHaveBeenCalled();
});

it("shutdown cancels in-flight search and settles tracked work before journal closure", async () => {
  const { products, dispatch, close, transportClose } = await fixture();
  dispatch.mockImplementation(
    (_request, _schema, options) =>
      new Promise((_resolve, reject) =>
        options?.signal?.addEventListener(
          "abort",
          () => reject(new Error("cancelled native search")),
          { once: true },
        ),
      ),
  );
  const pending = products.callTool("memory", call(), context());
  await vi.waitFor(() => expect(dispatch).toHaveBeenCalledOnce());
  const rejected = expect(pending).rejects.toThrow("cancelled");
  await products.dispose();
  await rejected;
  expect(products.busy).toBe(false);
  expect(close).toHaveBeenCalled();
  expect(transportClose).toHaveBeenCalled();
  await expect(products.callTool("memory", call(), context())).rejects.toThrow("closed");
});

it("sanitizes invalid backend responses and transport failures in tool results and the journal", async () => {
  const { products, root, dispatch } = await fixture();
  dispatch.mockRejectedValue(new Error(`Transport failed at ${root}`));
  await expect(products.callTool("memory", call(), context())).rejects.toThrow(
    "unavailable or returned an invalid response",
  );
  dispatch.mockResolvedValue({ content: [{ type: "text", text: "invalid" }] });
  await expect(products.callTool("memory", call(), context())).rejects.toThrow(
    "unavailable or returned an invalid response",
  );
  const events = products.journal.recall({});
  expect(JSON.stringify(events)).not.toContain(`Transport failed at ${root}`);
  expect((await products.learning.core.read()).content).toBe("");
});

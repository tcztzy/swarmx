import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type SessionEvent, type SessionHeader, SessionId } from "@deepseek-ai/dsh-session";
import {
  type SessionEventReadRequest,
  type SessionEventResultFilter,
  type SessionEventSearchDocument,
  type SessionEventWindow,
  SessionQueryError,
  type SessionRecord,
  type SessionSearchHit,
  type SessionSearchPage,
  type SessionSearchRequest,
} from "@deepseek-ai/dsh-session-query";
import { afterEach, describe, expect, it } from "vitest";
import { ConversationArchive, type ConversationSessionQuery, PkbVault } from "../src/index.js";

const roots: string[] = [];

function header(id: string, cwd: string, createdAt: number): SessionHeader {
  return { createdAt, cwd, id: SessionId(id), version: 0 };
}

function userEvent(seq: number, text: string, time: number): SessionEvent {
  return {
    data: { content: [{ text, type: "text" }], source: "human" },
    seq,
    surfaceOp: "append",
    time,
    type: "user/message",
  } as SessionEvent;
}

class FakeQuery implements ConversationSessionQuery {
  onRead?: () => void;
  searchError?: unknown;

  constructor(
    readonly records: SessionRecord[],
    readonly documents: Map<string, SessionEventSearchDocument[]>,
    readonly events: Map<string, SessionEvent>,
  ) {}

  async listSessions(): Promise<SessionRecord[]> {
    return this.records;
  }

  async searchSessions(
    request: SessionSearchRequest,
  ): Promise<SessionSearchPage<SessionSearchHit>> {
    if (this.searchError !== undefined) throw this.searchError;
    const ids = new Set(
      request.sessionFilters?.find((filter) => filter.kind === "id")?.values ?? [],
    );
    const items: SessionSearchHit[] = [];
    for (const record of this.records) {
      if (!ids.has(record.header.id)) continue;
      const match = this.documents
        .get(record.header.id)
        ?.find((document) => document.text.toLowerCase().includes(request.query.toLowerCase()));
      if (match === undefined) continue;
      items.push({
        ...record,
        bestMatch: { ...match, snippet: match.text },
      });
    }
    return { items: items.slice(0, request.limit) };
  }

  async filterEvents(
    sessionId: SessionId,
    filters: readonly SessionEventResultFilter[],
  ): Promise<SessionEventSearchDocument[]> {
    const text = filters.find((filter) => filter.kind === "text")?.text.toLowerCase();
    return (this.documents.get(sessionId) ?? []).filter(
      (document) => text === undefined || document.text.toLowerCase().includes(text),
    );
  }

  async readEvent(request: SessionEventReadRequest): Promise<SessionEventWindow> {
    const target = this.events.get(`${request.sessionId}:${String(request.seq)}`);
    const session = this.records.find((record) => record.header.id === request.sessionId)?.header;
    if (target === undefined || session === undefined) throw new Error("missing event");
    this.onRead?.();
    return {
      endSeq: request.seq,
      events: [target],
      session,
      startSeq: request.seq,
      target,
    };
  }
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-pkb-conversation-"));
  roots.push(root);
  const current = join(root, "one", "project");
  const other = join(root, "two", "project");
  await Promise.all([mkdir(current, { recursive: true }), mkdir(other, { recursive: true })]);
  const currentHeader = header("session-current", current, 2_000);
  const otherHeader = header("session-other", other, 1_000);
  const currentEvent = userEvent(7, "我们决定 PKB 使用 Markdown。", 2_100);
  const otherEvent = userEvent(3, "另一个项目也提到了 Markdown。", 1_100);
  const documents = new Map<string, SessionEventSearchDocument[]>([
    [
      currentHeader.id,
      [
        {
          seq: 7,
          sessionId: currentHeader.id,
          surface: "current",
          text: "我们决定 PKB 使用 Markdown。",
          time: 2_100,
          type: "user/message",
        },
      ],
    ],
    [
      otherHeader.id,
      [
        {
          seq: 3,
          sessionId: otherHeader.id,
          surface: "current",
          text: "另一个项目也提到了 Markdown。",
          time: 1_100,
          type: "user/message",
        },
      ],
    ],
  ]);
  const query = new FakeQuery(
    [
      { header: currentHeader, live: false, persisted: true },
      { header: otherHeader, live: false, persisted: true },
    ],
    documents,
    new Map([
      [`${currentHeader.id}:7`, currentEvent],
      [`${otherHeader.id}:3`, otherEvent],
    ]),
  );
  const vault = new PkbVault({ root: join(root, "vault") });
  const archive = new ConversationArchive(vault, query);
  return { archive, current, other, query, root, vault };
}

afterEach(async () => {
  const { rm } = await import("node:fs/promises");
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("ConversationArchive", () => {
  it("V132 V133: searches the current canonical workspace by default", async () => {
    const { archive, current } = await fixture();

    const result = await archive.search(current, { query: "Markdown" });

    expect(result.items).toHaveLength(1);
    expect(result.items[0]).toMatchObject({
      locator: { seq: 7, sessionId: "session-current" },
      trust: "untrusted-evidence",
    });
    expect(JSON.stringify(result)).not.toContain(current);
  });

  it("V133: requires explicit authorization before searching every workspace", async () => {
    const { archive, current } = await fixture();

    await expect(
      archive.search(current, { query: "Markdown", scope: "all" }),
    ).rejects.toMatchObject({ code: "AUTHORIZATION_REQUIRED" });

    const result = await archive.search(current, {
      allAuthorized: true,
      query: "Markdown",
      scope: "all",
    });
    expect(result.items.map((item) => item.locator.sessionId)).toEqual([
      "session-current",
      "session-other",
    ]);
  });

  it("V133: falls back to literal CJK scanning when full-text search is disabled", async () => {
    const { archive, current, query } = await fixture();
    query.searchError = new SessionQueryError("search disabled", "SESSION_QUERY_SEARCH_DISABLED");

    const result = await archive.search(current, { query: "决定" });

    expect(result.items).toEqual([
      expect.objectContaining({ locator: { seq: 7, sessionId: "session-current" } }),
    ]);
    expect(result.diagnostics).toContain("Full-text index unavailable; used literal scan.");
  });

  it("V134: captures an exact, untrusted conversation excerpt without a host path", async () => {
    const { archive, current, vault } = await fixture();

    const captured = await archive.capture(current, { seq: 7, sessionId: "session-current" });

    expect(captured.id).toMatch(/^workspaces\/project--[a-f0-9]{12}\/references\/conversations\//u);
    expect(captured.source.resource).toBe(`../references/conversations/${captured.filename}`);
    const source = await readFile(join(vault.root, captured.id), "utf8");
    expect(source).toContain('trust: "untrusted-evidence"');
    expect(source).toContain('session_id: "session-current"');
    expect(source).toContain("seq: 7");
    expect(source).toContain("start_seq: 7");
    expect(source).toContain("end_seq: 7");
    expect(source).toContain("我们决定 PKB 使用 Markdown。");
    expect(source).not.toContain(current);
  });

  it("B147: cancellation after the native read publishes no conversation excerpt", async () => {
    const { archive, current, query, vault } = await fixture();
    const controller = new AbortController();
    query.onRead = () => controller.abort(new Error("capture cancelled after read"));

    await expect(
      archive.capture(current, { seq: 7, sessionId: "session-current" }, controller.signal),
    ).rejects.toThrow("capture cancelled after read");

    const workspace = await vault.resolveWorkspace(current);
    const locatorHash = createHash("sha256").update("session-current:7").digest("hex").slice(0, 16);
    const target = join(
      vault.root,
      workspace.directory,
      "references",
      "conversations",
      `conversation--${locatorHash}--seq-7.md`,
    );
    await expect(readFile(target, "utf8")).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("V134: expands one exact event as bounded untrusted evidence", async () => {
    const { archive, current } = await fixture();

    const evidence = await archive.read(current, { seq: 7, sessionId: "session-current" });

    expect(evidence).toMatchObject({
      eventType: "user/message",
      locator: { seq: 7, sessionId: "session-current" },
      text: "我们决定 PKB 使用 Markdown。",
      trust: "untrusted-evidence",
    });
    expect(JSON.stringify(evidence)).not.toContain(current);
  });

  it("V132 V134: refuses to capture a conversation from another workspace", async () => {
    const { archive, current } = await fixture();

    await expect(
      archive.capture(current, { seq: 3, sessionId: "session-other" }),
    ).rejects.toMatchObject({ code: "UNSAFE_PATH" });
  });
});

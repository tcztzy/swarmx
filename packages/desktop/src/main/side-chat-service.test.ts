import * as fs from "node:fs";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { createSession, deleteSession, loadSession, saveSession } from "@swarmx/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { SideChatService } from "./side-chat-service.js";

const sessionsDir = fs.mkdtempSync(path.join(tmpdir(), "swarmx-side-chat-service-"));
const originalSessionsDir = process.env.SWARMX_SESSIONS_DIR;

describe("SideChatService", () => {
  const savedIds: string[] = [];

  beforeAll(() => {
    process.env.SWARMX_SESSIONS_DIR = sessionsDir;
  });

  afterAll(() => {
    for (const id of savedIds) deleteSession(id);
    if (originalSessionsDir === undefined) {
      Reflect.deleteProperty(process.env, "SWARMX_SESSIONS_DIR");
    } else {
      process.env.SWARMX_SESSIONS_DIR = originalSessionsDir;
    }
    fs.rmSync(sessionsDir, { recursive: true, force: true });
  });

  it("keeps tabs, drafts, context, visibility, and transcripts isolated by parent and id", () => {
    const parent = createSession("agent", "swarmx", "model-1");
    savedIds.push(parent.id);
    parent.messages = [
      { role: "user", content: "Parent request", kind: "message" },
      { role: "assistant", content: "Parent reply", kind: "message" },
    ];
    saveSession(parent);
    const service = new SideChatService();
    const first = service.create({
      parentSessionId: parent.id,
      throughMessageIndex: 1,
      expectedMessages: parent.messages,
    });
    const second = service.create({
      parentSessionId: parent.id,
      throughMessageIndex: 1,
      expectedMessages: parent.messages,
    });

    service.update({
      parentSessionId: parent.id,
      sideChatId: first.id,
      draft: "First draft @src",
      attachments: ["/workspace/src"],
    });
    service.addContext(parent.id, second.id, "selected parent text");
    service.beginRun(parent.id, first.id, "request-side-1", "Independent question");
    const running = service.beginRun(parent.id, second.id, "request-side-2", "Explain this");
    service.markStopping(parent.id, first.id, "request-side-1");
    expect(service.list(parent.id).chats.find((chat) => chat.id === first.id)?.runState).toBe(
      "stopping",
    );
    service.markRunning(parent.id, first.id, "request-side-1");
    expect(service.modelMessages(parent.id, second.id).at(-1)?.content).toContain(
      "selected parent text",
    );
    const completed = service.finishRun(
      parent.id,
      second.id,
      "request-side-2",
      [{ role: "assistant", content: "Side answer", kind: "message" }],
      { unread: true },
    );
    expect(service.list(parent.id).chats.find((chat) => chat.id === first.id)).toMatchObject({
      runState: "running",
      requestId: "request-side-1",
    });
    service.finishRun(
      parent.id,
      first.id,
      "request-side-1",
      [{ role: "assistant", content: "Independent answer", kind: "message" }],
      { unread: false },
    );
    service.setPaneHidden(parent.id, true);

    expect(running.runState).toBe("running");
    expect(completed).toMatchObject({ runState: "idle", unread: true });
    expect(service.list(parent.id)).toMatchObject({
      parentSessionId: parent.id,
      activeSideChatId: second.id,
      paneHidden: true,
      chats: [
        expect.objectContaining({
          id: first.id,
          draft: "",
          attachments: [],
          messages: [
            expect.objectContaining({ content: "Independent question" }),
            expect.objectContaining({ content: "Independent answer" }),
          ],
        }),
        expect.objectContaining({
          id: second.id,
          unread: true,
          messages: [
            expect.objectContaining({ content: "Explain this" }),
            expect.objectContaining({ content: "Side answer" }),
          ],
        }),
      ],
    });
    expect(loadSession(parent.id)?.messages).toEqual(parent.messages);
  });

  it("restores each parent's in-memory tabs without merging their anchors", () => {
    const firstParent = createSession("agent", "swarmx");
    const secondParent = createSession("agent", "swarmx");
    savedIds.push(firstParent.id, secondParent.id);
    firstParent.messages = [{ role: "user", content: "First parent", kind: "message" }];
    secondParent.messages = [{ role: "user", content: "Second parent", kind: "message" }];
    saveSession(firstParent);
    saveSession(secondParent);
    const service = new SideChatService();
    const first = service.create({
      parentSessionId: firstParent.id,
      throughMessageIndex: 0,
      expectedMessages: firstParent.messages,
    });
    const second = service.create({
      parentSessionId: secondParent.id,
      throughMessageIndex: 0,
      expectedMessages: secondParent.messages,
    });
    service.update({
      parentSessionId: firstParent.id,
      sideChatId: first.id,
      draft: "Return to this draft",
    });

    expect(service.list(secondParent.id).chats[0]).toMatchObject({
      id: second.id,
      anchorMessages: [expect.objectContaining({ content: "Second parent" })],
    });
    expect(service.list(firstParent.id).chats[0]).toMatchObject({
      id: first.id,
      draft: "Return to this draft",
      anchorMessages: [expect.objectContaining({ content: "First parent" })],
    });
  });

  it("distinguishes hiding from deletion and promotes only on explicit request", () => {
    const parent = createSession("agent", "swarmx");
    savedIds.push(parent.id);
    parent.messages = [
      { role: "user", content: "Parent request", kind: "message" },
      { role: "assistant", content: "Parent reply", kind: "message" },
    ];
    saveSession(parent);
    const service = new SideChatService();
    const side = service.create({
      parentSessionId: parent.id,
      throughMessageIndex: 1,
      expectedMessages: parent.messages,
    });
    service.beginRun(parent.id, side.id, "request-promote", "Side request");
    service.finishRun(
      parent.id,
      side.id,
      "request-promote",
      [{ role: "assistant", content: "Side reply", kind: "message" }],
      { unread: false },
    );

    service.setPaneHidden(parent.id, true);
    expect(service.list(parent.id).chats).toHaveLength(1);
    const promoted = service.promote(parent.id, side.id);
    savedIds.push(promoted.id);
    expect(loadSession(promoted.id)?.messages).toHaveLength(4);
    expect(loadSession(parent.id)?.messages).toEqual(parent.messages);
    expect(service.delete(parent.id, side.id).chats).toEqual([]);
  });
});

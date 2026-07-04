import { describe, expect, it } from "vitest";
import {
  conversationJsonlLine,
  createConversationEvent,
  parseConversationEvent,
  parseConversationJsonl,
  replayConversationEvents,
} from "../src/conversation.js";

const baseSession = {
  sessionId: "conv_local_1",
  title: "Analysis run",
  agentName: "analysis lead",
  harnessId: "geepilot-codex",
  model: "gpt-5",
  createdAt: "2026-07-03T00:00:00.000Z",
  updatedAt: "2026-07-03T00:00:00.000Z",
  contextStrategy: "checkpoint_tail" as const,
  storage: {
    root: "desktop" as const,
    rootRef: "$GEEPILOT_DESKTOP_ROOT",
    indexPath: "conversations.index.jsonl",
    rolloutPath: "conversations/conv_local_1.jsonl",
  },
};

describe("conversation ledger primitives", () => {
  it("creates deterministic conversation events and replays appended messages", () => {
    const created = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "session_created",
      actor: "host",
      session: baseSession,
    });
    const duplicate = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "session_created",
      actor: "host",
      session: baseSession,
    });
    const appended = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:01:00.000Z",
      eventType: "message_appended",
      actor: "user",
      message: {
        role: "user",
        kind: "message",
        content: "Plan an analysis workflow.",
      },
      renderEventIds: ["rne_123"],
    });

    expect(created.eventId).toBe(duplicate.eventId);
    expect(created.eventId).toMatch(/^cev_/);

    const state = replayConversationEvents([created, appended]);
    expect(state.rejectedEvents).toEqual([]);
    expect(state.sessions.conv_local_1).toMatchObject({
      title: "Analysis run",
      messageCount: 1,
      updatedAt: "2026-07-03T00:01:00.000Z",
    });
    expect(state.messagesBySession.conv_local_1).toEqual([
      {
        role: "user",
        kind: "message",
        content: "Plan an analysis workflow.",
      },
    ]);
  });

  it("round-trips append-only JSONL lines", () => {
    const created = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "session_created",
      session: baseSession,
    });
    const appended = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:01:00.000Z",
      eventType: "messages_appended",
      actor: "assistant",
      messages: [
        { role: "assistant", kind: "thinking", content: "Inspecting context." },
        { role: "assistant", kind: "message", content: "Use the benchmark harness." },
      ],
    });

    const jsonl = conversationJsonlLine(created) + conversationJsonlLine(appended);
    expect(parseConversationJsonl(jsonl)).toEqual([created, appended]);
    expect(() => parseConversationJsonl(`${jsonl}{bad json}\n`)).toThrow(
      /Invalid conversation JSONL line 3/,
    );
  });

  it("rejects invalid replay order without mutating session state", () => {
    const orphan = createConversationEvent({
      sessionId: "conv_missing",
      timestamp: "2026-07-03T00:01:00.000Z",
      eventType: "message_appended",
      actor: "assistant",
      message: {
        role: "assistant",
        kind: "message",
        content: "No session yet.",
      },
    });
    const created = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "session_created",
      session: baseSession,
    });
    const duplicateCreated = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:02:00.000Z",
      eventType: "session_created",
      session: { ...baseSession, title: "Duplicate" },
    });

    const state = replayConversationEvents([orphan, created, duplicateCreated]);
    expect(state.rejectedEvents.map((event) => event.eventId)).toEqual([
      orphan.eventId,
      duplicateCreated.eventId,
    ]);
    expect(state.sessions.conv_local_1.title).toBe("Analysis run");
  });

  it("applies title and archive events", () => {
    const created = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:00:00.000Z",
      eventType: "session_created",
      session: baseSession,
    });
    const renamed = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:02:00.000Z",
      eventType: "session_title_updated",
      actor: "user",
      title: "Reviewed analysis run",
    });
    const archived = createConversationEvent({
      sessionId: "conv_local_1",
      timestamp: "2026-07-03T00:03:00.000Z",
      eventType: "session_archived",
      actor: "user",
    });

    const state = replayConversationEvents([created, renamed, archived]);
    expect(state.sessions.conv_local_1).toMatchObject({
      title: "Reviewed analysis run",
      archived: true,
      updatedAt: "2026-07-03T00:03:00.000Z",
    });
  });

  it("rejects secret-looking structured fields while allowing secret references", () => {
    expect(() =>
      parseConversationEvent({
        eventId: "cev_secret",
        sessionId: "conv_local_1",
        timestamp: "2026-07-03T00:00:00.000Z",
        eventType: "artifact_linked",
        payload: {
          apiKey: "sk-test",
        },
      }),
    ).toThrow(/inline secret field.*apiKey/);

    expect(() =>
      parseConversationEvent({
        eventId: "cev_secret_ref",
        sessionId: "conv_local_1",
        timestamp: "2026-07-03T00:00:00.000Z",
        eventType: "artifact_linked",
        payload: {
          secretRef: "keychain:openai",
        },
      }),
    ).not.toThrow();
  });
});

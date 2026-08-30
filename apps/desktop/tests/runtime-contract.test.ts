import { describe, expect, it, vi } from "vitest";
import {
  ApprovalRegistry,
  ConversationController,
  type ConversationRuntime,
  type ConversationSnapshot,
  type UserMessageItem,
} from "../src/runtime/index.js";
import {
  parseScienceCarrierConfig,
  projectScienceCarrierConfig,
  serializeScienceCarrierConfig,
} from "../src/runtime/science-config.js";

describe("Science carrier configuration", () => {
  it("projects only bounded runtime settings and rejects malformed MCP input", () => {
    const projected = projectScienceCarrierConfig({
      root: "/private/product/science",
      embedArtifactMetadata: false,
      maxArtifactBytes: 4_096,
      notebookRuntime: "isolated",
      typstCommand: "/opt/typst",
    });
    expect(projected).toEqual({
      embedArtifactMetadata: false,
      maxArtifactBytes: 4_096,
      notebookRuntime: "isolated",
      typstCommand: "/opt/typst",
    });
    expect(serializeScienceCarrierConfig(projected)).not.toContain("/private/product/science");
    expect(parseScienceCarrierConfig(serializeScienceCarrierConfig(projected))).toEqual(projected);
    expect(() => parseScienceCarrierConfig('{"unknown":true}')).toThrow("SWARMX_SCIENCE_CONFIG");
    expect(() =>
      projectScienceCarrierConfig({
        root: "/private/product/science",
        jupymcpArgs: Array.from({ length: 20 }, () => "x".repeat(4_096)),
      }),
    ).toThrow("Science carrier configuration exceeds");
  });
});

const firstUser: UserMessageItem = {
  type: "user_message",
  id: "user-1",
  turnId: "turn-1",
  text: "first prompt",
  createdAt: 1,
};

const secondUser: UserMessageItem = {
  type: "user_message",
  id: "user-2",
  turnId: "turn-2",
  text: "second prompt",
  createdAt: 2,
};

function snapshot(): ConversationSnapshot {
  return {
    runtime: "codex",
    conversationId: "codex:source",
    workspace: { id: "workspace-1", label: "swarmx" },
    title: "Source",
    archived: false,
    turns: [
      { id: "turn-1", status: "completed", items: [firstUser] },
      { id: "turn-2", status: "completed", items: [secondUser] },
    ],
  };
}

function runtime(): ConversationRuntime {
  return {
    kind: "codex",
    list: vi.fn(async () => []),
    create: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "codex:new",
      workspace: { id: "workspace-1", label: "swarmx" },
      title: "New conversation",
      archived: false,
      updatedAt: 1,
    })),
    read: vi.fn(async () => snapshot()),
    start: vi.fn(async () => ({ turnId: "child-turn" })),
    steer: vi.fn(async () => undefined),
    interrupt: vi.fn(async () => undefined),
    revise: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "codex:revised",
      workspace: { id: "workspace-1", label: "swarmx" },
      title: "Source",
      archived: false,
      updatedAt: 3,
    })),
    fork: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "codex:child",
      workspace: { id: "workspace-1", label: "swarmx" },
      title: "Source (1)",
      archived: false,
      updatedAt: 2,
    })),
    archive: vi.fn(async () => undefined),
    subscribe: vi.fn(() => () => undefined),
    respondToApproval: vi.fn(async () => undefined),
    dispose: vi.fn(async () => undefined),
  };
}

describe("conversation controller", () => {
  it("retries by submitting the exact original text through runtime revision", async () => {
    const adapter = runtime();
    const result = await new ConversationController(adapter).retry("codex:source", "user-2");

    expect(adapter.revise).toHaveBeenCalledWith({
      conversationId: "codex:source",
      beforeTurnId: "turn-2",
      text: "second prompt",
    });
    expect(adapter.fork).not.toHaveBeenCalled();
    expect(adapter.start).not.toHaveBeenCalled();
    expect(result.conversationId).toBe("codex:revised");
  });

  it("edits only after an explicit replacement is submitted", async () => {
    const adapter = runtime();
    const result = await new ConversationController(adapter).edit(
      "codex:source",
      "user-1",
      "replacement prompt",
    );

    expect(adapter.revise).toHaveBeenCalledWith({
      conversationId: "codex:source",
      beforeTurnId: "turn-1",
      text: "replacement prompt",
    });
    expect(adapter.fork).not.toHaveBeenCalled();
    expect(adapter.start).not.toHaveBeenCalled();
    expect(result).toMatchObject({ conversationId: "codex:revised" });
  });

  it("rejects non-user and missing source items without creating a child", async () => {
    const adapter = runtime();
    await expect(
      new ConversationController(adapter).retry("codex:source", "missing"),
    ).rejects.toThrow('User message "missing"');
    expect(adapter.revise).not.toHaveBeenCalled();
  });
});

describe("approval registry", () => {
  it("requires the complete scoped identity and resolves once", async () => {
    const approvals = new ApprovalRegistry();
    const request = {
      runtime: "codex" as const,
      conversationId: "codex:source",
      turnId: "turn-1",
      itemId: "item-1",
      approvalId: "approval-1",
      kind: "command" as const,
      prompt: "Run tests?",
      choices: ["accept", "decline"] as const,
    };
    const pending = approvals.request(request);

    expect(approvals.list("codex", "codex:source")).toEqual([request]);

    expect(() => approvals.respond({ ...request, itemId: "wrong", decision: "accept" })).toThrow(
      "does not match",
    );
    approvals.respond({ ...request, decision: "accept" });
    await expect(pending).resolves.toMatchObject({ decision: "accept" });
    expect(approvals.list("codex", "codex:source")).toEqual([]);
    expect(() => approvals.respond({ ...request, decision: "decline" })).toThrow("not pending");
  });

  it("rejects one exact pending request without clearing another approval", async () => {
    const approvals = new ApprovalRegistry();
    const first = {
      runtime: "codex" as const,
      conversationId: "codex:source",
      turnId: "turn-1",
      itemId: "item-1",
      approvalId: "approval-1",
      kind: "command" as const,
      prompt: "Run tests?",
      choices: ["accept", "decline"] as const,
    };
    const second = { ...first, itemId: "item-2", approvalId: "approval-2" };
    const firstPending = approvals.request(first);
    const secondPending = approvals.request(second);

    expect(approvals.reject(first, "server cleared request")).toBe(true);
    await expect(firstPending).rejects.toThrow("server cleared request");
    expect(approvals.list("codex", "codex:source")).toEqual([second]);
    approvals.respond({ ...second, decision: "decline" });
    await expect(secondPending).resolves.toMatchObject({ approvalId: "approval-2" });
  });

  it("keeps delimiter-containing approval identities distinct", async () => {
    const approvals = new ApprovalRegistry();
    const first = {
      runtime: "codex" as const,
      conversationId: "codex:source",
      turnId: "turn-1",
      itemId: "item\u0000approval",
      approvalId: "tail",
      kind: "command" as const,
      prompt: "Run first command?",
      choices: ["accept", "decline"] as const,
    };
    const second = {
      ...first,
      itemId: "item",
      approvalId: "approval\u0000tail",
      prompt: "Run second command?",
    };
    const firstPending = approvals.request(first);
    const secondPending = approvals.request(second);
    void firstPending.catch(() => undefined);
    void secondPending.catch(() => undefined);

    try {
      expect(approvals.list("codex", "codex:source")).toEqual([first, second]);
      approvals.respond({ ...first, decision: "accept" });
      approvals.respond({ ...second, decision: "decline" });
      await expect(firstPending).resolves.toMatchObject({ itemId: first.itemId });
      await expect(secondPending).resolves.toMatchObject({ itemId: second.itemId });
    } finally {
      approvals.dispose();
      await Promise.allSettled([firstPending, secondPending]);
    }
  });

  it("rejects every pending request on disposal", async () => {
    const approvals = new ApprovalRegistry();
    const pending = approvals.request({
      runtime: "codex",
      conversationId: "codex:source",
      turnId: "turn-1",
      itemId: "item-1",
      approvalId: "approval-1",
      kind: "file_change",
      prompt: "Apply patch?",
      choices: ["accept", "decline"],
    });
    approvals.dispose();
    await expect(pending).rejects.toThrow("disposed");
  });
});

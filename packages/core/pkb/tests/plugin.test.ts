import { mkdir, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ToolRunContext } from "@deepseek-ai/dsh-tools";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createFrozenPkbIndexProvider,
  createPkbToolDefinition,
  executePkbOperation,
  PkbVault,
} from "../src/index.js";

const roots: string[] = [];

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-pkb-plugin-"));
  roots.push(root);
  const cwd = join(root, "project");
  await mkdir(cwd, { recursive: true });
  const vault = new PkbVault({ root: join(root, "vault") });
  await vault.initialize();
  return { cwd, vault };
}

function execution(cwd: string): ToolRunContext {
  return {
    agent: { session: { header: { cwd } } },
    callId: "call-pkb",
    signal: new AbortController().signal,
  } as unknown as ToolRunContext;
}

afterEach(async () => {
  const { rm } = await import("node:fs/promises");
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("pkb tool", () => {
  it("V135: exposes no delete action and reads without approval", async () => {
    const { cwd, vault } = await fixture();
    const approval = { request: vi.fn() };
    const archive = { capture: vi.fn(), read: vi.fn(), search: vi.fn() };
    const tool = createPkbToolDefinition({ approval, archive, vault });

    const action = (tool.parameters.properties.action as { enum: string[] }).enum;
    expect(action).not.toContain("delete_knowledge");
    expect(action).toContain("read_conversation");
    const output = await tool.execute(
      { action: "search_knowledge", request: { query: "anything" } },
      execution(cwd),
    );

    expect(output).toMatchObject({ action: "search_knowledge" });
    expect(approval.request).not.toHaveBeenCalled();

    await expect(
      executePkbOperation(
        { archive, vault },
        { action: "search_knowledge", request: { query: "anything" } },
        {
          actorId: "codex:thread-1",
          callId: "mcp-call-1",
          workspaceRoot: cwd,
          signal: new AbortController().signal,
          approve: vi.fn(),
        },
      ),
    ).resolves.toMatchObject({ action: "search_knowledge" });
  });

  it("V133 V135: fails closed before mutations and all-session reads", async () => {
    const { cwd, vault } = await fixture();
    const approval = { request: vi.fn().mockResolvedValue("rejected") };
    const archive = { capture: vi.fn(), read: vi.fn(), search: vi.fn() };
    const tool = createPkbToolDefinition({ approval, archive, vault });

    await expect(
      tool.execute(
        {
          action: "create_knowledge",
          request: {
            body: "# Memory\n\nDenied.",
            description: "Denied write.",
            scope: "workspace",
            title: "Denied",
            type: "Decision",
          },
        },
        execution(cwd),
      ),
    ).rejects.toMatchObject({ code: "AUTHORIZATION_REQUIRED" });
    expect((await vault.search(cwd, { query: "Denied" })).items).toEqual([]);

    await expect(
      tool.execute(
        {
          action: "search_conversations",
          request: { query: "旧账", scope: "all" },
        },
        execution(cwd),
      ),
    ).rejects.toMatchObject({ code: "AUTHORIZATION_REQUIRED" });
    expect(archive.search).not.toHaveBeenCalled();
  });

  it("V135: applies one approval to exactly one mutation", async () => {
    const { cwd, vault } = await fixture();
    const approval = { request: vi.fn().mockResolvedValueOnce("allowed-once") };
    const archive = { capture: vi.fn(), read: vi.fn(), search: vi.fn() };
    const tool = createPkbToolDefinition({ approval, archive, vault });

    await tool.execute(
      {
        action: "create_knowledge",
        request: {
          body: "# Memory\n\nApproved.",
          description: "Approved write.",
          scope: "workspace",
          title: "Approved",
          type: "Decision",
        },
      },
      execution(cwd),
    );

    expect((await vault.search(cwd, { query: "Approved" })).items).toHaveLength(1);
    expect(approval.request).toHaveBeenCalledTimes(1);
  });

  it("V135: does not mutate after approval returns to an aborted call", async () => {
    const { cwd } = await fixture();
    const controller = new AbortController();
    const createConcept = vi.fn();
    const approve = vi.fn(async () => {
      controller.abort();
      return "allowed-once";
    });

    await expect(
      executePkbOperation(
        {
          archive: { capture: vi.fn(), read: vi.fn(), search: vi.fn() },
          vault: {
            createConcept,
            deprecateConcept: vi.fn(),
            readConcept: vi.fn(),
            search: vi.fn(),
            updateConcept: vi.fn(),
          },
        },
        {
          action: "create_knowledge",
          request: {
            body: "# Cancelled\n\nMust not be written.",
            description: "Cancelled write.",
            title: "Cancelled",
            type: "Decision",
          },
        },
        {
          actorId: "codex:thread-1",
          callId: "mcp-call-aborted-after-approval",
          workspaceRoot: cwd,
          signal: controller.signal,
          approve,
        },
      ),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(createConcept).not.toHaveBeenCalled();
  });

  it("V134/V220: accepts every runtime-qualified native Thread locator it can return", async () => {
    const { cwd, vault } = await fixture();
    const sessionId = `codex:${"n".repeat(512)}`;
    const archive = {
      capture: vi.fn(),
      read: vi.fn().mockResolvedValue({ locator: { seq: 7, sessionId }, text: "evidence" }),
      search: vi.fn(),
    };

    await expect(
      executePkbOperation(
        { archive, vault },
        { action: "read_conversation", request: { seq: 7, sessionId } },
        {
          actorId: "codex:thread-1",
          callId: "mcp-call-long-locator",
          workspaceRoot: cwd,
          signal: new AbortController().signal,
          approve: vi.fn(),
        },
      ),
    ).resolves.toMatchObject({ action: "read_conversation", data: { text: "evidence" } });
    expect(archive.read).toHaveBeenCalledWith(cwd, { seq: 7, sessionId }, expect.anything());
  });

  it("V221: validates and defaults create scope before one approval", async () => {
    const { cwd, vault } = await fixture();
    const approve = vi.fn().mockResolvedValue("allowed-once");
    const archive = { capture: vi.fn(), read: vi.fn(), search: vi.fn() };

    await expect(
      executePkbOperation(
        { archive, vault },
        {
          action: "create_knowledge",
          request: {
            body: "# Default\n\nApproved.",
            description: "Approved default workspace write.",
            title: "Default workspace",
            type: "Decision",
          },
        },
        {
          actorId: "codex:thread-1",
          callId: "mcp-call-221",
          workspaceRoot: cwd,
          signal: new AbortController().signal,
          approve,
        },
      ),
    ).resolves.toMatchObject({
      action: "create_knowledge",
      data: { id: expect.stringMatching(/^workspaces\//u) },
    });
    expect(approve).toHaveBeenCalledOnce();

    approve.mockClear();
    await expect(
      executePkbOperation(
        { archive, vault },
        {
          action: "create_knowledge",
          request: { body: "# Invalid", description: "Missing title and type." },
        },
        {
          actorId: "codex:thread-1",
          callId: "mcp-call-invalid",
          workspaceRoot: cwd,
          signal: new AbortController().signal,
          approve,
        },
      ),
    ).rejects.toMatchObject({ code: "INVALID_REQUEST" });
    expect(approve).not.toHaveBeenCalled();
  });
});

describe("PKB prompt index", () => {
  it("V138: freezes one bounded index snapshot for each agent", async () => {
    const { cwd, vault } = await fixture();
    await vault.createConcept(cwd, {
      body: "# First\n\nInitial memory.",
      description: "Initial memory.",
      scope: "workspace",
      title: "First memory",
      type: "Reference",
    });
    const provider = createFrozenPkbIndexProvider(vault);
    const firstAgent = { session: { header: { cwd } } };
    const firstSnapshot = provider({ agent: firstAgent } as never);

    await vault.createConcept(cwd, {
      body: "# Second\n\nLater memory.",
      description: "Later memory.",
      scope: "workspace",
      title: "Second memory",
      type: "Reference",
    });

    expect(provider({ agent: firstAgent } as never)).toBe(firstSnapshot);
    expect(firstSnapshot).not.toContain("Second memory");
    expect(provider({ agent: { session: { header: { cwd } } } } as never)).toContain(
      "Second memory",
    );
    expect(firstSnapshot.length).toBeLessThanOrEqual(32_000);
  });
});

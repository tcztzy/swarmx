import { mkdir, mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ToolRunContext } from "@deepseek-ai/dsh-tools";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createFrozenPkbIndexProvider, createPkbToolDefinition, PkbVault } from "../src/index.js";

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

import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { CustomAgentService } from "./custom-agents.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("CustomAgentService", () => {
  it("persists Agent = versioned Harness recipe + Model and retains recipe history", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-agents-"));
    roots.push(root);
    const store = new DesktopSettingsStore({ path: join(root, "settings.json") });
    const service = new CustomAgentService(store, () => "2026-07-14T10:00:00.000Z");
    const first = await service.save(agentInput(["paper-review"]));
    const second = await service.save(agentInput(["paper-review", "memory"]));

    expect(first.harnessRecipe?.revisionId).not.toBe("draft");
    expect(second).toMatchObject({
      id: "researcher",
      harnessId: "research-harness",
      modelId: "gpt-5",
      harnessRecipe: { softwareId: "codex" },
    });
    expect(second.harnessRecipe?.revisionId).not.toBe(first.harnessRecipe?.revisionId);
    expect(second.harnessRecipeHistory).toEqual([first.harnessRecipe]);
    expect(await service.list()).toHaveLength(1);
  });

  it("rejects extension-owned ids, incomplete agents, and inline secrets", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-agents-"));
    roots.push(root);
    const service = new CustomAgentService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
    );

    await expect(
      service.save(agentInput([]), { reservedAgentIds: ["researcher"] }),
    ).rejects.toThrow(/Extension.*read-only/);
    await expect(service.save({ id: "bad", name: "Bad" })).rejects.toThrow(/Harness recipe/);
    await expect(
      service.save({ ...agentInput([]), providerApiKey: "do-not-store" }),
    ).rejects.toThrow(/secret/i);
  });

  it("removes only user-managed profiles", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-agents-"));
    roots.push(root);
    const service = new CustomAgentService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
    );
    await service.save(agentInput([]));
    await expect(service.remove("researcher")).resolves.toBe(true);
    await expect(service.remove("researcher")).resolves.toBe(false);
  });

  it("discovers Codex and Claude Code Agents read-only with project precedence", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-native-agents-"));
    roots.push(root);
    const home = join(root, "home");
    const workspace = join(root, "workspace");
    const userCodex = join(home, ".codex", "agents");
    const userClaude = join(home, ".claude", "agents");
    const projectCodex = join(workspace, ".codex", "agents");
    const projectClaude = join(workspace, ".claude", "agents");
    await Promise.all(
      [userCodex, userClaude, projectCodex, projectClaude].map((directory) =>
        mkdir(directory, { recursive: true }),
      ),
    );
    await Promise.all([
      writeFile(
        join(userCodex, "reviewer.toml"),
        codexAgent("reviewer", "inherit", "User instructions."),
      ),
      writeFile(
        join(projectCodex, "reviewer.toml"),
        codexAgent("reviewer", "gpt-5.6-sol", "Project instructions."),
      ),
      writeFile(
        join(userClaude, "reviewer.md"),
        claudeAgent("reviewer", "sonnet", "Claude instructions."),
      ),
      writeFile(
        join(userClaude, "analyst.md"),
        claudeAgent("analyst", "inherit", "Analyze the task."),
      ),
      writeFile(join(projectClaude, "broken.md"), "---\nname: broken\n---\nMissing description"),
    ]);
    const store = new DesktopSettingsStore({ path: join(root, "settings.json") });
    const service = new CustomAgentService(store);

    const result = await service.discoverNative({ workspaceRoot: workspace, homeDir: home });

    expect(result.agents).toHaveLength(3);
    expect(result.agents.map((agent) => agent.id)).toEqual([
      "native:claude_code:analyst",
      "native:claude_code:reviewer",
      "native:codex:reviewer",
    ]);
    expect(result.agents.find((agent) => agent.id === "native:codex:reviewer")).toMatchObject({
      harnessId: "codex",
      modelId: "gpt-5.6-sol",
      nativeModel: "gpt-5.6-sol",
      instructions: "Project instructions.",
      readOnly: true,
      definition: {
        kind: "project",
        host: "codex",
        format: "codex",
        readOnly: true,
      },
    });
    expect(result.agents.find((agent) => agent.id === "native:claude_code:analyst")).toMatchObject({
      harnessId: "claude_code",
      nativeModel: "inherit",
      readOnly: true,
    });
    expect(
      result.agents.find((agent) => agent.id === "native:claude_code:analyst")?.modelId,
    ).toBeUndefined();
    expect(result.warnings).toEqual([
      expect.objectContaining({ source: join(projectClaude, "broken.md") }),
    ]);
    expect(await service.list()).toEqual([]);
  });

  it("isolates secret-bearing native Agent files as discovery warnings", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-native-agent-secret-"));
    roots.push(root);
    const agentsDirectory = join(root, "workspace", ".codex", "agents");
    await mkdir(agentsDirectory, { recursive: true });
    await writeFile(
      join(agentsDirectory, "unsafe.toml"),
      `${codexAgent("unsafe", "inherit", "Do work.")}\napi_key = "do-not-import"\n`,
    );
    const service = new CustomAgentService(
      new DesktopSettingsStore({ path: join(root, "settings.json") }),
    );

    const result = await service.discoverNative({
      workspaceRoot: join(root, "workspace"),
      homeDir: join(root, "home"),
    });

    expect(result.agents).toEqual([]);
    expect(result.warnings[0]?.message).toMatch(/inline secret field/i);
  });
});

function agentInput(skills: string[]) {
  return {
    id: "researcher",
    name: "Researcher",
    harnessId: "research-harness",
    harnessRecipe: {
      id: "research-harness",
      revisionId: "draft",
      softwareId: "codex",
      skillBindings: skills.map((skillId) => ({ skillId, mode: "auto" })),
      mcpServerIds: ["zotero"],
    },
    modelId: "gpt-5",
  };
}

function codexAgent(name: string, model: string, instructions: string): string {
  return `name = "${name}"
description = "${name} description"
developer_instructions = "${instructions}"
model = "${model}"
`;
}

function claudeAgent(name: string, model: string, instructions: string): string {
  return `---
name: ${name}
description: ${name} description
model: ${model}
---

${instructions}
`;
}

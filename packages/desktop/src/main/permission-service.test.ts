import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type DesktopSettingsDocument, createDefaultDesktopSettings } from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import { PermissionService } from "./permission-service.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("PermissionService", () => {
  it("V450 resolves managed, Project, personal, and Agent authority ceilings", async () => {
    const { root, service } = await fixture({
      SWARMX_MANAGED_PERMISSION_POLICY: JSON.stringify({
        mode: "default",
        allowedTools: ["exec_command"],
      }),
    });
    await mkdir(join(root, ".swarmx"), { recursive: true });
    await writeFile(
      join(root, ".swarmx", "permissions.json"),
      JSON.stringify({ mode: "restricted", deniedTools: ["Bash"] }),
    );
    await service.savePersonalPolicy({ mode: "trusted", allowedTools: ["Write"] });

    const status = await service.status({
      cwd: root,
      agentId: "researcher",
      agentPolicy: { mode: "trusted", allowedTools: ["Bash"], deniedTools: [] },
    });

    expect(status.blocked).toBe(false);
    expect(status.layers.map((layer) => [layer.source, layer.configured])).toEqual([
      ["managed", true],
      ["project", true],
      ["personal", true],
      ["agent", true],
    ]);
    expect(status.effective).toMatchObject({
      policy: {
        mode: "restricted",
        allowedTools: ["Write", "exec_command"],
        deniedTools: ["Bash"],
      },
      modeSourceIds: ["project"],
      deniedToolSources: { Bash: ["project"] },
    });
  });

  it("V451 blocks execution on malformed managed or Project policy", async () => {
    const managed = await fixture({ SWARMX_MANAGED_PERMISSION_POLICY: "{" });
    await expect(
      managed.service.resolve({
        cwd: managed.root,
        agentPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      }),
    ).rejects.toThrow(/Managed policy is invalid/i);

    const project = await fixture();
    await mkdir(join(project.root, ".swarmx"), { recursive: true });
    await writeFile(
      join(project.root, ".swarmx", "permissions.json"),
      JSON.stringify({ allowedTools: ["Write"] }),
    );
    await expect(
      project.service.resolve({
        cwd: project.root,
        agentPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
      }),
    ).rejects.toThrow(/Project policy is invalid:[\s\S]*cannot pre-approve/i);
  });

  it("V458 applies a conversation mode without bypassing ceilings or explicit denials", async () => {
    const managed = await fixture({
      SWARMX_MANAGED_PERMISSION_POLICY: JSON.stringify({
        mode: "default",
        deniedTools: ["Write"],
      }),
    });
    await managed.service.savePersonalPolicy({
      mode: "plan",
      allowedTools: ["exec_command"],
      deniedTools: ["Bash"],
    });

    const inherited = await managed.service.resolve({
      cwd: managed.root,
      agentId: "builder",
      agentPolicy: { mode: "restricted", allowedTools: ["Edit"], deniedTools: ["WebFetch"] },
      sessionPermissionMode: "inherit",
    });
    expect(inherited.policy.mode).toBe("plan");

    const overridden = await managed.service.resolve({
      cwd: managed.root,
      agentId: "builder",
      agentPolicy: { mode: "restricted", allowedTools: ["Edit"], deniedTools: ["WebFetch"] },
      sessionPermissionMode: "trusted",
    });
    expect(overridden.policy).toEqual({
      mode: "default",
      allowedTools: ["Edit", "exec_command"],
      deniedTools: ["Bash", "WebFetch", "Write"],
    });
    expect(overridden.modeSourceIds).toEqual(["managed"]);
    expect(overridden.layers.at(-1)).toMatchObject({
      id: "session",
      source: "session",
      mode: "trusted",
    });

    const personalDefault = await fixture();
    await personalDefault.service.savePersonalPolicy({ mode: "trusted" });
    await expect(
      personalDefault.service.resolve({
        agentPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
        agentModeDeclared: false,
      }),
    ).resolves.toMatchObject({ policy: { mode: "trusted" }, modeSourceIds: ["personal"] });

    await mkdir(join(personalDefault.root, ".swarmx"), { recursive: true });
    await writeFile(
      join(personalDefault.root, ".swarmx", "permissions.json"),
      JSON.stringify({ mode: "plan" }),
    );
    await expect(
      personalDefault.service.resolve({
        cwd: personalDefault.root,
        agentPolicy: { mode: "default", allowedTools: [], deniedTools: [] },
        agentModeDeclared: false,
        sessionPermissionMode: "trusted",
      }),
    ).resolves.toMatchObject({ policy: { mode: "plan" }, modeSourceIds: ["project"] });
  });

  it("V463-V464 persists profile availability and safely degrades disabled modes", async () => {
    const { service } = await fixture();
    await service.savePersonalPolicy({
      mode: "auto",
      allowedTools: ["Write"],
      deniedTools: ["Bash"],
    });
    await service.saveProfileAvailability({ default: true, auto: false, trusted: false });

    const inherited = await service.status({
      agentPolicy: { mode: "trusted", allowedTools: [], deniedTools: [] },
      agentModeDeclared: false,
    });
    expect(inherited.profileAvailability).toEqual({
      default: true,
      auto: false,
      trusted: false,
    });
    expect(inherited.personalPolicy).toMatchObject({
      mode: "auto",
      allowedTools: ["Write"],
      deniedTools: ["Bash"],
    });
    expect(inherited.effective).toMatchObject({ policy: { mode: "plan" } });

    await expect(
      service.resolve({
        agentPolicy: { mode: "trusted", allowedTools: [], deniedTools: [] },
        sessionPermissionMode: "trusted",
      }),
    ).resolves.toMatchObject({ policy: { mode: "plan" }, modeSourceIds: ["session"] });
  });

  it("V452 persists structured personal policy and sanitized newest-first receipts", async () => {
    const { service } = await fixture(undefined, {
      now: () => "2026-07-18T12:00:00.000Z",
      id: () => "prm_12345678",
    });
    await service.savePersonalPolicy({ mode: "plan", deniedTools: ["Write"] });
    await service.recordDecision({
      source: "direct",
      toolName: "Write",
      toolKind: "write",
      decision: "rejected",
      optionKind: "reject_once",
      policySourceIds: ["personal"],
    });

    const status = await service.status();
    expect(status.personalPolicy).toMatchObject({
      id: "personal",
      source: "personal",
      mode: "plan",
      deniedTools: ["Write"],
      readOnly: false,
    });
    expect(status.approvalReceipts).toEqual([
      {
        id: "prm_12345678",
        createdAt: "2026-07-18T12:00:00.000Z",
        source: "direct",
        toolName: "Write",
        toolKind: "write",
        decision: "rejected",
        optionKind: "reject_once",
        policySourceIds: ["personal"],
      },
    ]);
    await expect(
      service.recordDecision({
        source: "direct",
        toolName: "Write",
        decision: "allowed",
        policySourceIds: ["api_key=secret"],
      }),
    ).rejects.toThrow(/secret/i);
  });

  it("V452 bounds approval history to the newest 200 receipts", async () => {
    let sequence = 0;
    const service = new PermissionService(new MemorySettingsStore(), {
      now: () => "2026-07-18T12:00:00.000Z",
      id: () => `prm_${String(++sequence).padStart(8, "0")}`,
    });
    for (let index = 0; index < 205; index += 1) {
      await service.recordDecision({
        source: "direct",
        toolName: `tool-${index}`,
        decision: "allowed",
      });
    }

    const receipts = (await service.status()).approvalReceipts;
    expect(receipts).toHaveLength(200);
    expect(receipts[0]?.toolName).toBe("tool-204");
    expect(receipts.at(-1)?.toolName).toBe("tool-5");
  });
});

class MemorySettingsStore {
  #settings = createDefaultDesktopSettings();

  async read(): Promise<DesktopSettingsDocument> {
    return this.#settings;
  }

  async update(
    mutation: (
      current: DesktopSettingsDocument,
    ) => DesktopSettingsDocument | Promise<DesktopSettingsDocument>,
  ): Promise<DesktopSettingsDocument> {
    this.#settings = await mutation(this.#settings);
    return this.#settings;
  }
}

async function fixture(
  env?: Record<string, string | undefined>,
  options?: ConstructorParameters<typeof PermissionService>[1],
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-permissions-"));
  roots.push(root);
  const store = new DesktopSettingsStore({ path: join(root, "settings.json") });
  return { root, service: new PermissionService(store, { env: env ?? {}, ...options }) };
}

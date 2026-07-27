import { access, mkdir, mkdtemp, readFile, rm, stat, utimes, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { AgentCompositionPlan, SessionData } from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import {
  createEphemeralCodexHome,
  createExternalAcpSessionBinding,
  externalAcpSessionIdentity,
  latestUserMessageHasAttachments,
  matchingExternalAcpSessionId,
  resolveCodexHome,
} from "./acp-session-runtime.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("desktop ACP Session runtime", () => {
  it("V565 matches bindings only for the exact execution identity", () => {
    const plan = {
      id: "desktop-gpt-5",
      agentId: "codex:gpt-5",
      agentProfileId: "reviewer",
      harnessId: "custom-codex",
      modelId: "gpt-5",
      modelSupplyId: "gpt-5-openai",
    } as AgentCompositionPlan;
    const identity = externalAcpSessionIdentity(plan, "codex", "/workspace/project");
    if (!identity) throw new Error("identity was not created");
    const binding = createExternalAcpSessionBinding(
      identity,
      "external-session",
      undefined,
      "2026-07-27T00:00:00.000Z",
    );
    const session = { externalAcpSession: binding } as SessionData;

    expect(matchingExternalAcpSessionId(session, identity)).toBe("external-session");
    expect(
      matchingExternalAcpSessionId(session, { ...identity, modelId: "gpt-5-mini" }),
    ).toBeUndefined();
    expect(
      matchingExternalAcpSessionId(session, { ...identity, cwd: "/workspace/other" }),
    ).toBeUndefined();
    expect(
      matchingExternalAcpSessionId(session, { ...identity, agentProfileId: "other" }),
    ).toBeUndefined();
  });

  it("V567 recognizes only the latest canonical user turn attachments", () => {
    const attached = {
      messages: [
        {
          role: "user",
          kind: "message",
          content: "old",
          attachments: [
            {
              id: "old-media",
              name: "old.png",
              kind: "image",
              mimeType: "image/png",
              sizeBytes: 8,
              uri: "file:///tmp/old.png",
              source: "user",
            },
          ],
        },
        { role: "assistant", kind: "message", content: "answer" },
        { role: "user", kind: "message", content: "current" },
      ],
    } as SessionData;
    expect(latestUserMessageHasAttachments(attached)).toBe(false);

    attached.messages[2] = {
      role: "user",
      kind: "message",
      content: "current",
      attachments: [
        {
          id: "new-media",
          name: "new.png",
          kind: "image",
          mimeType: "image/png",
          sizeBytes: 8,
          uri: "file:///tmp/new.png",
          source: "user",
        },
      ],
    };
    expect(latestUserMessageHasAttachments(attached)).toBe(true);
  });

  it("V568 isolates Codex inputs and removes the exact temporary home", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-acp-runtime-"));
    roots.push(root);
    const sourceHome = path.join(root, "source");
    const storageRoot = path.join(root, "ephemeral");
    await mkdir(path.join(sourceHome, "skills", "inspect"), { recursive: true });
    await writeFile(path.join(sourceHome, "auth.json"), '{"token":"secret"}', { mode: 0o600 });
    await writeFile(path.join(sourceHome, "config.toml"), 'model = "gpt-5"', { mode: 0o600 });
    await writeFile(path.join(sourceHome, "skills", "inspect", "SKILL.md"), "read only skill");

    const isolated = await createEphemeralCodexHome({ sourceHome, storageRoot });
    expect(isolated.env).toEqual({
      CODEX_HOME: isolated.path,
      APP_SERVER_LOGS: path.join(isolated.path, "logs"),
    });
    expect(await readFile(path.join(isolated.path, "auth.json"), "utf8")).toContain("secret");
    expect(await readFile(path.join(isolated.path, "skills", "inspect", "SKILL.md"), "utf8")).toBe(
      "read only skill",
    );
    expect(
      (await stat(path.join(isolated.path, "skills", "inspect", "SKILL.md"))).mode & 0o777,
    ).toBe(0o400);
    await mkdir(path.join(isolated.path, "sessions"));
    await writeFile(path.join(isolated.path, "sessions", "rollout.jsonl"), "base64-payload");

    await isolated.cleanup();
    await expect(access(isolated.path)).rejects.toThrow();
    expect(await readFile(path.join(sourceHome, "auth.json"), "utf8")).toContain("secret");
    await isolated.cleanup();
  });

  it("V568 removes stale crash residue before creating another temporary home", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-acp-stale-"));
    roots.push(root);
    const storageRoot = path.join(root, "ephemeral");
    const stale = path.join(storageRoot, "codex-attachment-stale");
    await mkdir(stale, { recursive: true });
    const old = new Date("2026-07-25T00:00:00.000Z");
    await utimes(stale, old, old);

    const isolated = await createEphemeralCodexHome({
      sourceHome: path.join(root, "missing-source"),
      storageRoot,
      now: () => new Date("2026-07-27T00:00:00.000Z").getTime(),
    });
    await expect(access(stale)).rejects.toThrow();
    await isolated.cleanup();
  });

  it("resolves default, tilde, and absolute Codex homes", () => {
    expect(resolveCodexHome(undefined, "/Users/test")).toBe("/Users/test/.codex");
    expect(resolveCodexHome("~/custom", "/Users/test")).toBe("/Users/test/custom");
    expect(resolveCodexHome("/var/codex", "/Users/test")).toBe("/var/codex");
  });
});

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  BuiltinToolSettingsService,
  preferredBuiltinToolStyleForProvider,
  resolveRunBuiltinTools,
} from "./builtin-tool-settings.js";
import { DesktopSettingsStore } from "./settings-store.js";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("BuiltinToolSettingsService", () => {
  it("updates built-in tool style without replacing unrelated Settings", async () => {
    const root = await mkdtemp(join(tmpdir(), "swarmx-builtin-tools-"));
    roots.push(root);
    const store = new DesktopSettingsStore({ path: join(root, "settings.json") });
    const service = new BuiltinToolSettingsService(store);
    await store.update((settings) => ({
      ...settings,
      extensions: { ...settings.extensions, enabledPluginIds: ["paper-tools"] },
    }));

    await expect(service.get()).resolves.toEqual({ style: "auto" });
    await expect(service.save({ style: "kimi_code" })).resolves.toEqual({
      style: "kimi_code",
    });
    await expect(service.save({ style: "invalid" })).rejects.toThrow();
    expect((await store.read()).extensions.enabledPluginIds).toEqual(["paper-tools"]);
  });

  it("resolves and preserves a Session binding before changed Settings", () => {
    expect(
      resolveRunBuiltinTools({
        settings: { style: "auto" },
        model: { preferredBuiltinToolStyle: "kimi_code" },
      }),
    ).toEqual({ style: "kimi_code", revision: 1, source: "model" });
    expect(
      resolveRunBuiltinTools({
        settings: { style: "claude_code" },
        model: { preferredBuiltinToolStyle: "claude_code" },
        session: {
          builtinTools: { style: "kimi_code", revision: 1, source: "model" },
        },
      }),
    ).toEqual({ style: "kimi_code", revision: 1, source: "model" });
  });

  it("maps only explicit first-party Provider endpoints to automatic styles", () => {
    expect(preferredBuiltinToolStyleForProvider("https://api.moonshot.cn/v1", undefined)).toBe(
      "kimi_code",
    );
    expect(preferredBuiltinToolStyleForProvider("https://api.anthropic.com", undefined)).toBe(
      "claude_code",
    );
    expect(
      preferredBuiltinToolStyleForProvider(
        "https://chatgpt.com/backend-api/codex",
        "codex_app_server",
      ),
    ).toBe("codex");
    expect(
      preferredBuiltinToolStyleForProvider("https://gateway.example.test/v1", undefined),
    ).toBeUndefined();
  });
});

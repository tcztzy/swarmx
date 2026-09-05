import { existsSync, readdirSync, readFileSync } from "node:fs";
import { extname, join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const root = process.cwd();

function files(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory() && !["dist", "lib", "node_modules", "generated"].includes(entry.name)) {
      return files(path);
    }
    return entry.isFile() ? [path] : [];
  });
}

describe("architecture boundaries", () => {
  it("contains no old runtime or deleted UI paths", () => {
    for (const path of [
      "apps/desktop/src/runtime",
      "apps/desktop/src/agents/dsh",
      "packages/core/agent",
      "packages/core/wikiskill",
      "packages/client",
    ]) {
      expect(existsSync(join(root, path)), path).toBe(false);
    }
    const source = files(join(root, "apps/desktop/src"))
      .filter((path) => [".ts", ".tsx", ".json"].includes(extname(path)))
      .map((path) => readFileSync(path, "utf8"))
      .join("\n");
    expect(source).not.toMatch(
      /@deepseek-ai|codex-acp|claude-agent-acp|ConversationController|cordis/iu,
    );
    const renderer = files(join(root, "apps/desktop/src/renderer"))
      .map((path) => readFileSync(path, "utf8"))
      .join("\\n");
    expect(renderer).not.toMatch(
      /@openai|@anthropic|@openclaw|@agentclientprotocol|@a2a-js|Retry|Fork|Revision|@theme/u,
    );
  });

  it("keeps public packages free of providers and UI protocols", () => {
    const forbidden =
      /@deepseek-ai|cordis|@openai|@anthropic|@openclaw|@agentclientprotocol|@a2a-js|@ag-ui|assistant-ui|electron/iu;
    const offenders = [
      "packages/core/swarm",
      "packages/core/dvc",
      "packages/core/knowledge-base",
      "packages/science/core",
    ].flatMap((directory) =>
      files(join(root, directory))
        .filter((path) => [".ts", ".tsx", ".json"].includes(extname(path)))
        .filter((path) => forbidden.test(readFileSync(path, "utf8")))
        .map((path) => relative(root, path)),
    );
    expect(offenders).toEqual([]);
  });

  it("ships native integrations, official protocols and unconfigured Tailwind", () => {
    const manifest = JSON.parse(readFileSync(join(root, "apps/desktop/package.json"), "utf8")) as {
      dependencies?: Record<string, string>;
      devDependencies?: Record<string, string>;
    };
    const names = Object.keys(manifest.dependencies ?? {});
    expect(names).not.toContain("@openai/codex");
    expect(names.filter((name) => name.startsWith("@deepseek-ai/"))).toEqual([]);
    expect(names).not.toContain("@swarmx/agent");
    expect(names).not.toContain("assistant-cloud");
    expect(names).not.toContain("@deepseek-ai/cordis");
    expect(manifest.dependencies).toMatchObject({
      "@a2a-js/sdk": "1.1.0",
      "@anthropic-ai/claude-agent-sdk": expect.any(String),
      "@openclaw/gateway-client": expect.any(String),
      "@agentclientprotocol/sdk": "1.4.0",
      "@assistant-ui/react-o11y": "0.0.42",
      "@assistant-ui/store": "0.3.12",
    });
    expect(manifest.devDependencies).toMatchObject({
      "@tailwindcss/vite": "4.3.3",
      tailwindcss: "4.3.3",
    });
    expect(existsSync(join(root, "apps/desktop/src/renderer/app.module.css"))).toBe(false);
    expect(files(join(root, "apps/desktop")).some((path) => /tailwind\.config\./u.test(path))).toBe(
      false,
    );
    const swarm = JSON.parse(
      readFileSync(join(root, "packages/core/swarm/package.json"), "utf8"),
    ) as { dependencies?: Record<string, string> };
    expect(swarm.dependencies).toBeUndefined();
    expect(
      names.some((name) => /codex-acp|claude-agent-acp|dsh|cordis|kimi|zcode/u.test(name)),
    ).toBe(false);
  });

  it.runIf(existsSync(join(root, "apps/desktop/dist/renderer")))(
    "production Renderer contains no provider or deleted UI code",
    () => {
      const bundle = files(join(root, "apps/desktop/dist/renderer"));
      expect(bundle.length).toBeGreaterThan(0);
      expect(bundle.map((path) => readFileSync(path, "utf8")).join("\n")).not.toMatch(
        /@deepseek-ai|dsh-web|dsh-client-ui|cordis-web|tui_gateway|codex-acp/u,
      );
    },
  );
});

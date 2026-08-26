import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

describe("dsh-swarm published artifacts", () => {
  it("retains every public entry and shared runtime chunk", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as {
      exports: Record<string, { default?: string } | string>;
      files: string[];
    };
    expect(manifest.files).toContain("lib/errors.js");
    for (const value of Object.values(manifest.exports)) {
      const target = typeof value === "string" ? value : value.default;
      if (!target?.startsWith("./lib/")) continue;
      expect(manifest.files).toContain(target.slice(2));
    }
  });
});

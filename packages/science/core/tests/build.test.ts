import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import buildConfig from "../tsdown.config.js";

describe("Science package build", () => {
  it("uses one deterministic shared chunk and publishes only current runtime entries", () => {
    expect(buildConfig).toMatchObject({
      clean: false,
      outputOptions: { chunkFileNames: "[name].js" },
    });
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as { readonly files: readonly string[] };

    expect(packageJson.files).toEqual([
      "lib/index.js",
      "lib/contracts.js",
      "lib/typert.js",
      "lib/remote.js",
      "lib/remote-contract.js",
      "lib/tools.js",
      "lib/preset.js",
      "lib/types/**/*.d.ts",
      "bin",
      "src",
      "demo",
      "config/agent-presets",
      "cordis.patch.yml",
    ]);
  });
});

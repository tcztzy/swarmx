import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import buildConfig from "../tsdown.config.js";

describe("Git UI client build", () => {
  it("uses the shared single-file browser bundle", () => {
    const configs = buildConfig({ env: { DSH_BUILD_FACE: "client" } });
    expect(configs[1]?.outputOptions).toMatchObject({ codeSplitting: false });
    expect(configs[1]?.outputOptions).not.toHaveProperty("inlineDynamicImports");
  });

  it("injects the renderer without removed client runtime compatibility", () => {
    const packageJson = JSON.parse(
      readFileSync(new URL("../package.json", import.meta.url), "utf8"),
    ) as {
      readonly dsh: { readonly client: { readonly inject: readonly string[] } };
      readonly devDependencies: Readonly<Record<string, string>>;
      readonly peerDependencies: Readonly<Record<string, string>>;
    };

    expect(packageJson.dsh.client.inject).toContain("@deepseek-ai/dsh-client-ui-renderer");
    expect(packageJson.dsh.client.inject).not.toContain("@deepseek-ai/dsh-client-runtime");
    expect(packageJson.dsh.client.inject).not.toContain("@deepseek-ai/dsh-client-ui-slots");
    expect(packageJson.devDependencies).not.toHaveProperty("@deepseek-ai/dsh-client-runtime");
    expect(packageJson.peerDependencies).not.toHaveProperty("@deepseek-ai/dsh-client-runtime");
    expect(packageJson.devDependencies).toHaveProperty("@deepseek-ai/dsh-client-ui-renderer");
    expect(packageJson.peerDependencies).toHaveProperty("@deepseek-ai/dsh-client-ui-renderer");
  });
});

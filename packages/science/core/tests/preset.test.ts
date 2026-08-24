import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const standardPath = join(
  dirname(require.resolve("@deepseek-ai/dsh/package.json")),
  "config",
  "agent-presets",
  "standard",
  "agent.cordis.yml",
);
const sciencePath = new URL(
  "../config/agent-presets/dsh-science/agent.cordis.yml",
  import.meta.url,
);

describe("V121/V122/V123 dsh-science preset", () => {
  it("extends the complete locked standard composition with only Science rows", () => {
    const standard = readFileSync(standardPath, "utf8").trimEnd();
    const science = readFileSync(sciencePath, "utf8");

    expect(science.startsWith(`${standard}\n\n`)).toBe(true);
    expect(science.slice(standard.length)).toBe(`

# ── SwarmX Science ───────────────────────────────────────────────────────────

- id: swarmx-science-tools
  name: '@swarmx/dsh-science/tools'

- id: swarmx-science-preset-contract
  name: '@swarmx/dsh-science/preset'
`);
  });

  it("publishes ordered system-preset metadata", () => {
    const metadata = readFileSync(
      new URL("../config/agent-presets/dsh-science/preset.yml", import.meta.url),
      "utf8",
    );

    expect(metadata).toBe(
      "name: 科学模式\ndescription: 具备标准模式的全部能力，并提供本地优先的科学研究工具、文献检索、注释理解与托管 Typst 工作流。\norder: 5\n",
    );
  });

  it("keeps Host services and browser UI global while removing model tools", () => {
    const patch = readFileSync(new URL("../cordis.patch.yml", import.meta.url), "utf8");

    expect(patch).toContain("name: '@swarmx/dsh-science'");
    expect(patch).toContain("name: '@swarmx/dsh-ui-science'");
    expect(patch).not.toContain("name: '@swarmx/dsh-science/tools'");
  });
});

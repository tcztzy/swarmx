import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const science = new URL(
  "../../../science/core/config/agent-presets/dsh-science/agent.cordis.yml",
  import.meta.url,
);
const swarm = new URL("../config/agent-presets/dsh-swarm/agent.cordis.yml", import.meta.url);

describe("V170 dsh-swarm preset", () => {
  it("extends the exact locked dsh-science composition with only Swarm rows", () => {
    const scienceConfig = readFileSync(science, "utf8").trimEnd();
    const swarmConfig = readFileSync(swarm, "utf8");
    expect(swarmConfig.startsWith(`${scienceConfig}\n\n`)).toBe(true);
    expect(swarmConfig.slice(scienceConfig.length)).toBe(`

# ── SwarmX Swarm ─────────────────────────────────────────────────────────────────────────

- id: swarmx-swarm-tools
  name: '@swarmx/dsh-swarm/tools'
`);
  });

  it("publishes ordered system metadata without mounting tools globally", () => {
    const metadata = readFileSync(
      new URL("../config/agent-presets/dsh-swarm/preset.yml", import.meta.url),
      "utf8",
    );
    const patch = readFileSync(new URL("../cordis.patch.yml", import.meta.url), "utf8");
    expect(metadata).toBe(
      "name: 团队模式\ndescription: 在科学模式的完整能力上，增加持久化多 Agent 任务、消息与安全调度。\norder: 6\n",
    );
    expect(patch).toContain("name: '@swarmx/dsh-swarm'");
    expect(patch).not.toContain("name: '@swarmx/dsh-swarm/tools'");
  });
});

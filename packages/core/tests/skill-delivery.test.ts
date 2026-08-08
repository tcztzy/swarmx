import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { Agent } from "../src/agent.js";
import {
  buildDeliveredInstructions,
  loadSkillFragmentContent,
  SkillDeliveryError,
  SkillInstructionDeliverySchema,
  sha256Hex,
} from "../src/skill-delivery.js";
import type { SwarmConfig } from "../src/types.js";

const temporaryRoots: string[] = [];
afterEach(async () => {
  while (temporaryRoots.length > 0) {
    const root = temporaryRoots.pop();
    if (root) await rmTree(root);
  }
});

function delivery(content: string, overrides: Partial<Record<string, string>> = {}) {
  return SkillInstructionDeliverySchema.parse({
    skillId: "math-coach",
    variantId: "math-coach:default",
    revisionId: `r_${"b".repeat(64)}`,
    contentDigest: `sha256:${sha256Hex(content)}`,
    mode: "prompt_fragment",
    content,
    ...overrides,
  });
}

describe("loadSkillFragmentContent", () => {
  it("reads and verifies the digest of a content-addressed artifact", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-delivery-"));
    temporaryRoots.push(root);
    const filePath = path.join(root, "skill.md");
    const content = "# Skill\n\nInstructions.\n";
    await writeFile(filePath, content, "utf8");
    const loaded = await loadSkillFragmentContent(filePath, {
      expectedDigest: `sha256:${sha256Hex(content)}`,
    });
    expect(loaded).toBe(content);
  });

  it("rejects a digest mismatch", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-delivery-"));
    temporaryRoots.push(root);
    const filePath = path.join(root, "skill.md");
    await writeFile(filePath, "# Skill\n", "utf8");
    await expect(
      loadSkillFragmentContent(filePath, {
        expectedDigest: `sha256:${"0".repeat(64)}`,
      }),
    ).rejects.toMatchObject({ code: "digest_mismatch" });
  });

  it("rejects an oversized artifact", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-delivery-"));
    temporaryRoots.push(root);
    const filePath = path.join(root, "skill.md");
    const content = "# Skill\n";
    await writeFile(filePath, content, "utf8");
    await expect(
      loadSkillFragmentContent(filePath, {
        expectedDigest: `sha256:${sha256Hex(content)}`,
        maxBytes: 4,
      }),
    ).rejects.toMatchObject({ code: "oversized_artifact" });
  });
});

describe("SkillInstructionDeliverySchema", () => {
  it("rejects content whose digest does not match", () => {
    expect(() => delivery("content-a", { contentDigest: `sha256:${"0".repeat(64)}` })).toThrow();
  });

  it("rejects non prompt_fragment modes", () => {
    expect(() =>
      delivery("content", { mode: "host_native_plugin" as "prompt_fragment" }),
    ).toThrow();
  });
});

describe("buildDeliveredInstructions", () => {
  it("appends the verified fragment to the base instructions", () => {
    const result = buildDeliveredInstructions("Base instructions.", [
      delivery("# Skill A\n\nDo A."),
    ]);
    expect(result).toContain("Base instructions.");
    expect(result).toContain("# Skill: math-coach");
    expect(result).toContain("content digest: sha256:");
    expect(result).toContain("Do A.");
  });

  it("returns base instructions unchanged without deliveries", () => {
    expect(buildDeliveredInstructions("Base.", [])).toBe("Base.");
  });
});

describe("instruction delivery reaches the model-visible system message", () => {
  it("changes what a stub model observes for baseline vs candidate", () => {
    const observed: string[] = [];
    const config: SwarmConfig = {
      name: "delivery",
      root: "agent",
      nodes: {
        agent: {
          kind: "agent",
          agent: { name: "agent", instructions: "You are a helpful assistant." },
        },
      },
      edges: [],
    };
    const baselineAgent = new Agent(config.nodes.agent.agent as never, { skillInstructions: [] });
    const candidateAgent = new Agent(config.nodes.agent.agent as never, {
      skillInstructions: [delivery("# Skill A\n\nAlways say YES.")],
    });
    observed.push(baselineAgent.instructions, candidateAgent.instructions);
    expect(observed[0]).not.toContain("Always say YES.");
    expect(observed[1]).toContain("Always say YES.");
    expect(observed[1]).toContain(`sha256:${sha256Hex("# Skill A\n\nAlways say YES.")}`);
  });
});

describe("SkillDeliveryError", () => {
  it("carries a stable error code", () => {
    const error = new SkillDeliveryError("unsupported_delivery", "nope");
    expect(error.code).toBe("unsupported_delivery");
    expect(error.name).toBe("SkillDeliveryError");
  });
});

async function rmTree(root: string): Promise<void> {
  const { rm } = await import("node:fs/promises");
  await rm(root, { recursive: true, force: true });
}

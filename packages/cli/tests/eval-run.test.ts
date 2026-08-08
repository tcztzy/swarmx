import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { SwarmConfig } from "@swarmx/core";
import { describe, expect, it } from "vitest";
import { buildEvalArguments, evalSwarmOptions, runEval } from "../src/eval-run.js";

describe("eval-run skill delivery binding", () => {
  it("refuses a multi-agent config without --skill-delivery-agent", async () => {
    const config: SwarmConfig = {
      name: "two",
      root: "a",
      nodes: {
        a: { kind: "agent", agent: { name: "a" } },
        b: { kind: "agent", agent: { name: "b" } },
      },
      edges: [],
    };
    await expect(
      evalSwarmOptions(
        {
          skillDelivery: JSON.stringify({
            skillId: "s",
            variantId: "s:v",
            revisionId: "r_v",
            contentDigest:
              "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            mode: "prompt_fragment",
          }),
          skillContentPath: "/tmp/skill.md",
        },
        config,
      ),
    ).rejects.toThrow(/requires --skill-delivery-agent/i);
  });

  it("binds an explicit delivery to a single named agent", async () => {
    const config: SwarmConfig = {
      name: "two",
      root: "a",
      nodes: {
        a: { kind: "agent", agent: { name: "a" } },
        b: { kind: "agent", agent: { name: "b" } },
      },
      edges: [],
    };
    const { mkdtemp, writeFile } = await import("node:fs/promises");
    const { tmpdir } = await import("node:os");
    const root = await mkdtemp(`${tmpdir()}/swarmx-eval-agent-`);
    const content = "# Skill\n";
    const { createHash } = await import("node:crypto");
    const digest = createHash("sha256").update(content).digest("hex");
    await writeFile(`${root}/skill.md`, content, "utf8");
    const options = await evalSwarmOptions(
      {
        skillDelivery: JSON.stringify({
          skillId: "s",
          variantId: "s:v",
          revisionId: "r_v",
          contentDigest: `sha256:${digest}`,
          mode: "prompt_fragment",
        }),
        skillContentPath: `${root}/skill.md`,
        skillDeliveryAgent: "b",
      },
      config,
    );
    expect(options.agent?.skillInstructionsByAgent).toBeDefined();
    expect(Object.keys(options.agent?.skillInstructionsByAgent ?? {})).toEqual(["b"]);
  });
});

describe("eval-run helpers", () => {
  it("builds chat arguments from a message", () => {
    expect(buildEvalArguments("hello", {})).toEqual({
      messages: [{ role: "user", content: "hello" }],
    });
  });

  it("prefers structured input JSON over the positional message", () => {
    expect(
      buildEvalArguments("ignored", {
        inputJson: '{"messages":[{"role":"user","content":"from json"}],"caseId":"case-1"}',
      }),
    ).toEqual({
      messages: [{ role: "user", content: "from json" }],
      caseId: "case-1",
    });
  });

  it("returns a schema-valid JSON result when Swarm execution fails", async () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-eval-run-"));
    const configPath = join(dir, "swarm.json");
    writeFileSync(
      configPath,
      JSON.stringify({
        name: "bad_eval",
        root: "missing",
        nodes: {},
        edges: [],
      }),
    );

    const result = await runEval("hello", { config: configPath });

    expect(result.output).toBe("");
    expect(result.messages).toEqual([]);
    expect(result.trace).toEqual([]);
    expect(result.error).toMatch(/Root node/);
    expect(result.metrics).toEqual({
      steps: 0,
      messages: 0,
      toolCalls: 0,
      toolResults: 0,
      contextTokens: 0,
    });
  });

  it("runs deterministic echo backend samples without model credentials", async () => {
    const dir = mkdtempSync(join(tmpdir(), "swarmx-echo-eval-run-"));
    const configPath = join(dir, "swarm.json");
    writeFileSync(
      configPath,
      JSON.stringify({
        name: "echo_eval",
        root: "echo_agent",
        nodes: {
          echo_agent: {
            kind: "agent",
            agent: {
              name: "echo_agent",
              backend: { type: "echo" },
            },
          },
        },
        edges: [],
      }),
    );

    const result = await runEval("deterministic answer", { config: configPath });

    expect(result.error).toBeNull();
    expect(result.output).toBe("deterministic answer");
    expect(result.messages).toHaveLength(1);
    expect(result.trace).toMatchObject([
      {
        swarm: "echo_eval",
        node: "echo_agent",
        kind: "agent",
        step: 1,
        status: "completed",
        messageCount: 1,
      },
    ]);
    expect(result.metrics).toEqual({
      steps: 1,
      messages: 1,
      toolCalls: 0,
      toolResults: 0,
      contextTokens: 0,
    });
  });
});

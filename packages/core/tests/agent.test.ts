import OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { Agent, HookRef } from "../src/agent.js";
import type { AgentConfig } from "../src/types.js";

describe("Agent", () => {
  it("constructs with minimal config", () => {
    const agent = new Agent({ name: "test" });
    expect(agent.name).toBe("test");
    expect(agent.description).toBeUndefined();
    expect(agent.model).toBe("gpt-4o");
    expect(agent.instructions).toBe("");
    expect(agent instanceof Agent).toBe(true);
  });

  it("creates OpenAI client from config", () => {
    const agent = new Agent({
      name: "test",
      client: { apiKey: "sk-test", baseUrl: "https://api.test.com/v1" },
    });
    expect(agent.client).toBeInstanceOf(OpenAI);
  });

  it("generates swarm config", () => {
    const agent = new Agent({
      name: "helper",
      description: "A helper agent",
      model: "claude-3",
      instructions: "Be helpful",
    });
    const config = agent.toSwarmConfig();
    expect(config.name).toBe("helper");
    expect(config.root).toBe("helper");
    expect(config.nodes).toHaveProperty("helper");
    expect(config.edges).toEqual([]);
  });

  it("rejects invalid agent name", () => {
    expect(() => new Agent({ name: "123bad" })).toThrow();
    expect(() => new Agent({ name: "bad-name" })).toThrow();
    expect(() => new Agent({ name: "" })).toThrow();
  });

  it("validates McpServer discriminated union", () => {
    const agent = new Agent({
      name: "test",
      mcpServers: {
        fs: { type: "stdio", command: "npx", args: ["-y", "server"] },
        web: { type: "sse", url: "http://localhost:8080" },
      },
    });
    expect(agent.mcpServers.size).toBe(2);
  });

  it("rejects invalid McpServer missing required fields", () => {
    const invalidMcpServers = {
      bad: { type: "stdio" },
    } as unknown as AgentConfig["mcpServers"];

    expect(
      () =>
        new Agent({
          name: "test",
          mcpServers: invalidMcpServers,
        }),
    ).toThrow();
  });

  it("accepts MCP servers and hooks", () => {
    const agent = new Agent({
      name: "test",
      mcpServers: {
        filesystem: {
          type: "stdio",
          command: "npx",
          args: ["-y", "@modelcontextprotocol/server-filesystem"],
        },
      },
      hooks: [{ onStart: "echo start" }],
    });
    expect(agent.mcpServers.size).toBe(1);
    expect(agent.hooks).toHaveLength(1);
    expect(agent.hooks[0].onStart).toBe("echo start");
  });

  it("uses model from config over default", () => {
    const agent = new Agent({ name: "test", model: "gpt-5-mini" });
    expect(agent.model).toBe("gpt-5-mini");
  });

  it("uses OPENAI_MODEL env var", () => {
    const previousModel = process.env.OPENAI_MODEL;
    try {
      process.env.OPENAI_MODEL = "env-model";
      const agent = new Agent({ name: "test" });
      expect(agent.model).toBe("env-model");
    } finally {
      process.env.OPENAI_MODEL = previousModel;
    }
  });
});

describe("HookRef", () => {
  it("constructs with hook config", () => {
    const hook = new HookRef({
      onStart: "start",
      onEnd: "end",
    });
    expect(hook.onStart).toBe("start");
    expect(hook.onEnd).toBe("end");
    expect(hook.onHandoff).toBeUndefined();
    expect(hook.onChunk).toBeUndefined();
  });
});

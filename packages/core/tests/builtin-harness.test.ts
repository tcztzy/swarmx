import { describe, expect, it, vi } from "vitest";
import { Agent } from "../src/agent.js";
import { BuiltinHarness } from "../src/builtin-harness.js";
import type { LocalTool } from "../src/local-tool-contracts.js";

const tool = (name: string): LocalTool => ({
  name,
  inputSchema: { type: "object" },
  async call() {
    return name;
  },
});

describe("BuiltinHarness", () => {
  it("composes ordered prompt, tool, and loop contributions", async () => {
    const harness = new BuiltinHarness();
    const observed: string[] = [];

    harness.mount({
      id: "first",
      setup(context) {
        context.addPromptSection("rules", () => "First rules");
        context.addTool(tool("first_tool"));
        context.useAgentLoop("observe", async (turn, next) => {
          observed.push(`before:${turn.instructions}`);
          const result = await next();
          observed.push("after");
          return result;
        });
      },
    });
    harness.mount({
      id: "second",
      setup(context) {
        context.addPromptSection("context", () => "Second context");
        context.addTool(tool("second_tool"));
      },
    });

    const result = await harness.run(
      {
        agentName: "agent",
        mode: "call",
        arguments: {},
        runtimeContext: {},
        instructions: "Base",
        tools: [tool("base_tool")],
      },
      async (turn) => {
        expect(turn.instructions).toBe("Base\n\n---\n\nFirst rules\n\n---\n\nSecond context");
        expect(turn.tools.map((item) => item.name)).toEqual([
          "base_tool",
          "first_tool",
          "second_tool",
        ]);
        return { messages: [] };
      },
    );

    expect(result).toEqual({ messages: [] });
    expect(observed).toEqual([
      "before:Base\n\n---\n\nFirst rules\n\n---\n\nSecond context",
      "after",
    ]);
  });

  it("unmounts every scoped effect together", async () => {
    const harness = new BuiltinHarness();
    const cleanup = vi.fn();
    const unmount = harness.mount({
      id: "temporary",
      setup(context) {
        context.addPromptSection("temporary", () => "temporary");
        context.addTool(tool("temporary_tool"));
        context.useAgentLoop("temporary_loop", async (_turn, next) => next());
        return cleanup;
      },
    });

    unmount();
    const fallback = vi.fn(async () => ({ messages: [] }));
    await harness.run(
      {
        agentName: "agent",
        mode: "call",
        arguments: {},
        runtimeContext: {},
        instructions: "Base",
        tools: [],
      },
      fallback,
    );

    expect(cleanup).toHaveBeenCalledOnce();
    expect(fallback).toHaveBeenCalledWith(
      expect.objectContaining({ instructions: "Base", tools: [] }),
    );
  });

  it("cleans up every plugin before reporting aggregate cleanup failures", () => {
    const harness = new BuiltinHarness();
    const firstCleanup = vi.fn();
    const secondCleanup = vi.fn(() => {
      throw new Error("second cleanup failed");
    });
    harness.mount({ id: "first", setup: () => firstCleanup });
    harness.mount({ id: "second", setup: () => secondCleanup });

    expect(() => harness.close()).toThrow(AggregateError);
    expect(secondCleanup).toHaveBeenCalledOnce();
    expect(firstCleanup).toHaveBeenCalledOnce();
    expect(() => harness.mount({ id: "second", setup() {} })).not.toThrow();
  });

  it("rejects duplicate identities and repeated waterfall delegation", async () => {
    const harness = new BuiltinHarness();
    harness.mount({
      id: "plugin",
      setup(context) {
        context.addTool(tool("shared"));
        context.useAgentLoop("twice", async (_turn, next) => {
          await next();
          return next();
        });
      },
    });

    expect(() => harness.mount({ id: "plugin", setup() {} })).toThrow(/already mounted/);
    expect(() =>
      harness.mount({
        id: "duplicate-tool",
        setup(context) {
          context.addTool(tool("shared"));
        },
      }),
    ).toThrow(/tool "shared"/);

    await expect(
      harness.run(
        {
          agentName: "agent",
          mode: "call",
          arguments: {},
          runtimeContext: {},
          instructions: "",
          tools: [],
        },
        async () => ({ messages: [] }),
      ),
    ).rejects.toThrow(/more than once/);
  });

  it("lets a direct Harness plugin replace the default Agent loop", async () => {
    const harness = new BuiltinHarness();
    harness.mount({
      id: "replacement",
      setup(context) {
        context.addPromptSection("identity", ({ agentName }) => `Agent: ${agentName}`);
        context.addTool(tool("plugin_tool"));
        context.useAgentLoop("replacement", async (turn) => ({
          messages: [
            {
              role: "assistant",
              kind: "message",
              agent: turn.agentName,
              content: `${turn.instructions}|${turn.tools.map((item) => item.name).join(",")}`,
            },
          ],
        }));
      },
    });
    const agent = new Agent(
      { name: "native_agent", instructions: "Base" },
      { builtinHarness: harness },
    );

    await expect(agent.call({ messages: [{ role: "user", content: "hello" }] })).resolves.toEqual({
      messages: [
        {
          role: "assistant",
          kind: "message",
          agent: "native_agent",
          content: "Base\n\n---\n\nAgent: native_agent|plugin_tool",
        },
      ],
    });
  });

  it("lets stream middleware wrap emitted chunks", async () => {
    const harness = new BuiltinHarness();
    harness.mount({
      id: "stream-wrapper",
      setup(context) {
        context.useAgentLoop("wrap", async (turn, next) => {
          const emit = turn.emitChunk;
          return next({
            emitChunk: (chunk) => emit?.({ ...chunk, content: `wrapped:${chunk.content}` }),
          });
        });
        context.useAgentLoop("replacement", async (turn) => {
          const message = {
            role: "assistant" as const,
            kind: "message" as const,
            agent: turn.agentName,
            content: "streamed",
          };
          turn.emitChunk?.(message);
          return { messages: [message] };
        });
      },
    });
    const agent = new Agent({ name: "native_agent" }, { builtinHarness: harness });
    const chunks: string[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "hello" }] },
      (chunk) => chunks.push(chunk.content),
    );

    expect(chunks).toEqual(["wrapped:streamed"]);
    expect(result.messages[0]?.content).toBe("streamed");
  });

  it("rebuilds the Agent tool surface from plugins admitted on a later turn", async () => {
    const harness = new BuiltinHarness();
    const agent = new Agent(
      {
        name: "native_agent",
        model: "test-model",
        client: { apiProtocol: "openai_responses" },
      },
      { builtinHarness: harness },
    );
    const create = vi.fn(async () => ({
      id: "response",
      status: "completed",
      error: null,
      output: [
        {
          id: "message",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "done", annotations: [] }],
        },
      ],
    }));
    Object.defineProperty(agent.client.responses, "create", { value: create });

    await agent.call({ messages: [{ role: "user", content: "first" }] });
    harness.mount({
      id: "later-tool",
      setup(context) {
        context.addTool(tool("later_tool"));
      },
    });
    await agent.call({ messages: [{ role: "user", content: "second" }] });

    expect(create.mock.calls[0]?.[0]?.tools ?? []).not.toContainEqual(
      expect.objectContaining({ name: "later_tool" }),
    );
    expect(create.mock.calls[1]?.[0]?.tools).toContainEqual(
      expect.objectContaining({ name: "later_tool" }),
    );
  });

  it("never injects the built-in kernel into external ACP Harnesses", () => {
    expect(
      () =>
        new Agent(
          {
            name: "external",
            backend: { type: "custom", program: "external-acp" },
          },
          { builtinHarness: new BuiltinHarness() },
        ),
    ).toThrow(/direct SwarmX backend/);
  });
});

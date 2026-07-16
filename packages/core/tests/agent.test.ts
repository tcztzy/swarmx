import OpenAI from "openai";
import { describe, expect, it, vi } from "vitest";
import { RequestCancelledError, cancelAcpRequest, withAcpRequest } from "../src/acp.js";
import { Agent, HookRef } from "../src/agent.js";
import type { AgentConfig, MessageChunk } from "../src/types.js";

describe("Agent", () => {
  it("uses composition-scoped provider environment for the native client", () => {
    const agent = new Agent({
      name: "provider_agent",
      process: {
        env: {
          OPENAI_API_KEY: "scoped-key",
          OPENAI_BASE_URL: "https://provider.example/v1",
        },
      },
    });

    expect(agent.client.apiKey).toBe("scoped-key");
    expect(agent.client.baseURL).toBe("https://provider.example/v1");
  });

  it("never sends an ambient API key to an explicit third-party runtime", () => {
    const previousKey = process.env.OPENAI_API_KEY;
    const previousBaseUrl = process.env.OPENAI_BASE_URL;
    try {
      process.env.OPENAI_API_KEY = "ambient-openai-key";
      process.env.OPENAI_BASE_URL = "https://api.openai.example/v1";
      const agent = new Agent({
        name: "third_party_agent",
        process: {
          env: { OPENAI_BASE_URL: "https://third-party.example/v1" },
          clearEnv: true,
        },
      });

      expect(agent.client.apiKey).toBe("sk-no-key");
      expect(agent.client.baseURL).toBe("https://third-party.example/v1");
    } finally {
      if (previousKey === undefined) Reflect.deleteProperty(process.env, "OPENAI_API_KEY");
      else process.env.OPENAI_API_KEY = previousKey;
      if (previousBaseUrl === undefined) Reflect.deleteProperty(process.env, "OPENAI_BASE_URL");
      else process.env.OPENAI_BASE_URL = previousBaseUrl;
    }
  });

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

  it("echo backend returns the latest user message without a model call", async () => {
    const agent = new Agent({ name: "test", backend: { type: "echo" } });

    const result = await agent.call({
      messages: [
        { role: "system", content: "ignore" },
        { role: "user", content: "first" },
        { role: "assistant", content: "middle" },
        { role: "user", content: "last" },
      ],
    });

    expect(result.messages).toEqual([
      {
        role: "assistant",
        content: "last",
        kind: "message",
        agent: "test",
      },
    ]);
  });

  it("custom backend delegates prompts to the ACP client", async () => {
    let seen:
      | {
          opts: {
            command: string;
            args: string[];
            cwd?: string;
            env?: Record<string, string>;
            clearEnv?: boolean;
          };
          prompt: string;
        }
      | undefined;
    const streamed: MessageChunk[] = [];
    const agent = new Agent(
      {
        name: "codex_agent",
        instructions: "Plan with evidence.",
        backend: { type: "custom", program: "codex", args: ["acp"] },
        process: {
          currentDir: "/tmp/project",
          env: { OPENAI_MODEL: "gpt-5" },
          clearEnv: true,
        },
      },
      {
        createAcpClient: () => ({
          async prompt(opts, prompt, _swarmConfig, _sessionId, onChunk) {
            seen = { opts, prompt };
            const chunk: MessageChunk = {
              role: "assistant",
              content: "working",
              kind: "thinking",
              agent: "codex_agent",
            };
            onChunk?.(chunk);
            return {
              messages: [
                {
                  role: "assistant",
                  content: "done",
                  kind: "message",
                  agent: "codex_agent",
                },
              ],
            };
          },
        }),
      },
    );

    const result = await agent.callStream(
      {
        messages: [
          { role: "user", content: "first request" },
          { role: "assistant", content: "middle" },
          { role: "user", content: "latest request" },
        ],
      },
      (chunk) => streamed.push(chunk),
    );

    expect(seen).toEqual({
      opts: {
        command: "codex",
        args: ["acp"],
        cwd: "/tmp/project",
        env: { OPENAI_MODEL: "gpt-5" },
        clearEnv: true,
      },
      prompt: "Agent instructions:\nPlan with evidence.\n\nUser request:\nlatest request",
    });
    expect(agent.model).toBeUndefined();
    expect(
      (agent.toSwarmConfig().nodes as Record<string, { model?: string }>).codex_agent?.model,
    ).toBeUndefined();
    expect(streamed).toEqual([
      { role: "assistant", content: "working", kind: "thinking", agent: "codex_agent" },
    ]);
    expect(result.messages).toEqual([
      { role: "assistant", content: "done", kind: "message", agent: "codex_agent" },
    ]);
  });

  it("does not continue after a custom backend confirms cancellation", async () => {
    const agent = new Agent(
      {
        name: "cancelled_agent",
        backend: { type: "custom", program: "test-acp" },
      },
      {
        createAcpClient: () => ({
          async prompt() {
            await cancelAcpRequest("cooperative-agent-cancel");
            return { messages: [] };
          },
        }),
      },
    );

    await expect(
      withAcpRequest("cooperative-agent-cancel", () =>
        agent.call({ messages: [{ role: "user", content: "stop" }] }),
      ),
    ).rejects.toBeInstanceOf(RequestCancelledError);
  });

  it("passes the request AbortSignal to native OpenAI calls", async () => {
    const agent = new Agent({ name: "native_agent", model: "gpt-5" });
    let receivedSignal: AbortSignal | undefined;
    let markStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      markStarted = resolve;
    });

    const create = vi.fn(
      (_body: unknown, options?: { signal?: AbortSignal }) =>
        new Promise<never>((_resolve, reject) => {
          receivedSignal = options?.signal;
          markStarted();
          options?.signal?.addEventListener("abort", () => reject(new Error("OpenAI aborted")), {
            once: true,
          });
        }),
    );
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    const run = withAcpRequest("native-openai-cancel", () =>
      agent.call({ messages: [{ role: "user", content: "wait" }] }),
    );
    await started;

    await expect(cancelAcpRequest("native-openai-cancel")).resolves.toBe(true);
    expect(receivedSignal?.aborted).toBe(true);
    await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
  });

  it("V283 executes OpenAI Responses natively with reasoning and MCP continuation", async () => {
    const agent = new Agent({
      name: "responses_agent",
      model: "gpt-5",
      client: { apiProtocol: "openai_responses" },
      parameters: {
        reasoning: {
          control: "effort_enum",
          effort: "high",
          parameterMapping: { api: "openai.responses", path: "reasoning.effort" },
        },
      },
    });
    const callTool = vi.fn().mockResolvedValue({ forecast: "sunny" });
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "resp_1",
        status: "completed",
        error: null,
        output: [
          {
            id: "reason_1",
            type: "reasoning",
            summary: [{ type: "summary_text", text: "Check the weather tool." }],
          },
          {
            id: "msg_1",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "I will check.", annotations: [] }],
          },
          {
            id: "fc_1",
            type: "function_call",
            call_id: "call_1",
            name: "weather",
            arguments: '{"city":"Shanghai"}',
            status: "completed",
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "resp_2",
        status: "completed",
        error: null,
        output: [
          {
            id: "msg_2",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: "It is sunny.", annotations: [] }],
          },
        ],
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiProtocol).toBe("openai_responses");
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "gpt-5",
      reasoning: { effort: "high" },
      tools: [
        {
          type: "function",
          name: "weather",
          parameters: { type: "object" },
          strict: false,
        },
      ],
    });
    expect(create.mock.calls[1]?.[0]?.input).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: "function_call", call_id: "call_1" }),
        {
          type: "function_call_output",
          call_id: "call_1",
          output: '{"forecast":"sunny"}',
        },
      ]),
    );
    expect(callTool).toHaveBeenCalledWith("weather", { city: "Shanghai" });
    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "Check the weather tool." }),
        expect.objectContaining({ kind: "tool_call", toolName: "weather" }),
        expect.objectContaining({ kind: "tool_result", content: '{"forecast":"sunny"}' }),
        expect.objectContaining({ kind: "message", content: "It is sunny." }),
      ]),
    );
  });

  it("V339 executes host-injected Project tools through the native Responses loop", async () => {
    const readProjectFile = vi.fn().mockResolvedValue({
      path: "README.md",
      content: "# Project-aware SwarmX",
      truncated: false,
    });
    const agent = new Agent(
      {
        name: "project_agent",
        model: "gpt-5.6-luna",
        client: { apiProtocol: "openai_responses" },
      },
      {
        localTools: [
          {
            name: "workspace_read_file",
            description: "Read a Project file.",
            inputSchema: {
              type: "object",
              properties: { path: { type: "string" } },
              required: ["path"],
            },
            call: readProjectFile,
          },
        ],
      },
    );
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "resp_project_tool",
        status: "completed",
        error: null,
        output: [
          {
            id: "fc_project_read",
            type: "function_call",
            call_id: "call_project_read",
            name: "workspace_read_file",
            arguments: '{"path":"README.md"}',
            status: "completed",
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "resp_project_answer",
        status: "completed",
        error: null,
        output: [
          {
            id: "msg_project_answer",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [
              { type: "output_text", text: "This is Project-aware SwarmX.", annotations: [] },
            ],
          },
        ],
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({
      messages: [{ role: "user", content: "Introduce this Project." }],
    });

    expect(create.mock.calls[0]?.[0]?.tools).toContainEqual(
      expect.objectContaining({ name: "workspace_read_file" }),
    );
    expect(readProjectFile).toHaveBeenCalledWith({ path: "README.md" });
    expect(create.mock.calls[1]?.[0]?.input).toContainEqual({
      type: "function_call_output",
      call_id: "call_project_read",
      output: expect.stringContaining("Project-aware SwarmX"),
    });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "This is Project-aware SwarmX." }),
    );
  });

  it("V354 recovers a streamed Project tool call omitted from response.completed", async () => {
    const readProjectFile = vi.fn().mockResolvedValue({
      path: "README.md",
      content: "# Streamed Project",
      truncated: false,
    });
    const agent = new Agent(
      {
        name: "streamed_project_agent",
        model: "gpt-5.4",
        client: { apiProtocol: "openai_responses" },
      },
      {
        localTools: [
          {
            name: "workspace_read_file",
            description: "Read a Project file.",
            inputSchema: {
              type: "object",
              properties: { path: { type: "string" } },
              required: ["path"],
            },
            call: readProjectFile,
          },
        ],
      },
    );
    const functionCall = {
      id: "fc_streamed_read",
      type: "function_call",
      call_id: "call_streamed_read",
      name: "workspace_read_file",
      arguments: '{"path":"README.md"}',
      status: "completed",
    };
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield {
            type: "response.reasoning_summary_text.delta",
            delta: "**Inspecting README**",
          };
          yield { type: "response.output_item.done", output_index: 1, item: functionCall };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_tool",
              status: "completed",
              error: null,
              output: [
                {
                  id: "reason_streamed_tool",
                  type: "reasoning",
                  summary: [{ type: "summary_text", text: "**Inspecting README**" }],
                },
              ],
            },
          };
        },
      })
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield {
            type: "response.reasoning_summary_text.delta",
            delta: "**Preparing final summary**",
          };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_reasoning_only",
              status: "completed",
              error: null,
              output: [
                {
                  id: "reason_streamed_summary",
                  type: "reasoning",
                  summary: [{ type: "summary_text", text: "**Preparing final summary**" }],
                },
              ],
            },
          };
        },
      })
      .mockResolvedValueOnce({
        async *[Symbol.asyncIterator]() {
          yield { type: "response.output_text.delta", delta: "This is the streamed Project." };
          yield {
            type: "response.completed",
            response: {
              id: "resp_streamed_answer",
              status: "completed",
              error: null,
              output: [
                {
                  id: "msg_streamed_answer",
                  type: "message",
                  role: "assistant",
                  status: "completed",
                  content: [
                    {
                      type: "output_text",
                      text: "This is the streamed Project.",
                      annotations: [],
                    },
                  ],
                },
              ],
            },
          };
        },
      });
    Object.defineProperty(agent.client.responses, "create", { value: create });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Introduce this Project." }] },
      (chunk) => streamed.push(chunk),
    );

    expect(readProjectFile).toHaveBeenCalledWith({ path: "README.md" });
    expect(create.mock.calls[1]?.[0]?.input).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "function_call_output",
          call_id: "call_streamed_read",
          output: expect.stringContaining("Streamed Project"),
        }),
      ]),
    );
    expect(create).toHaveBeenCalledTimes(3);
    expect(streamed).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "**Inspecting README**" }),
        expect.objectContaining({ kind: "tool_call", toolName: "workspace_read_file" }),
        expect.objectContaining({ kind: "tool_result", toolName: "workspace_read_file" }),
        expect.objectContaining({ kind: "message", content: "This is the streamed Project." }),
      ]),
    );
    expect(result.messages.at(-1)).toEqual(
      expect.objectContaining({ kind: "message", content: "This is the streamed Project." }),
    );
  });

  it("executes Codex Responses with SwarmX context and stateless encrypted continuation", async () => {
    const accessToken = fakeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "account-123" },
    });
    const agent = new Agent({
      name: "codex_responses_agent",
      model: "gpt-5.4",
      instructions: "Use only the SwarmX agent instructions.",
      client: { api_mode: "codex_responses", access_token: accessToken },
    });
    const callTool = vi.fn().mockResolvedValue({ forecast: "sunny" });
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce(
        openAIResponseStream({
          id: "resp_1",
          status: "completed",
          error: null,
          output: [
            {
              id: "reason_1",
              type: "reasoning",
              encrypted_content: "encrypted-state",
              summary: [{ type: "summary_text", text: "Use the weather tool." }],
              status: "completed",
            },
            {
              id: "fc_1",
              type: "function_call",
              call_id: "call_1",
              name: "weather",
              arguments: '{"city":"Shanghai"}',
              status: "completed",
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        openAIResponseStream({
          id: "resp_2",
          status: "completed",
          error: null,
          output: [
            {
              id: "msg_2",
              type: "message",
              role: "assistant",
              status: "completed",
              content: [{ type: "output_text", text: "It is sunny.", annotations: [] }],
            },
          ],
        }),
      );
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiMode).toBe("codex_responses");
    expect(agent.apiProtocol).toBe("openai_responses");
    expect(agent.client.apiKey).toBe(accessToken);
    expect(agent.client.baseURL).toBe("https://chatgpt.com/backend-api/codex");
    const clientOptions = agent.client as unknown as {
      _options: { defaultHeaders?: Record<string, string> };
    };
    expect(clientOptions._options.defaultHeaders).toMatchObject({
      originator: "swarmx",
      "ChatGPT-Account-ID": "account-123",
    });
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "gpt-5.4",
      instructions: "Use only the SwarmX agent instructions.",
      store: false,
      include: ["reasoning.encrypted_content"],
      reasoning: { summary: "auto" },
      tool_choice: "auto",
      parallel_tool_calls: true,
      stream: true,
    });
    const replayInput = create.mock.calls[1]?.[0]?.input;
    expect(replayInput).toEqual(
      expect.arrayContaining([
        {
          type: "reasoning",
          encrypted_content: "encrypted-state",
          summary: [{ type: "summary_text", text: "Use the weather tool." }],
        },
        {
          type: "function_call",
          call_id: "call_1",
          name: "weather",
          arguments: '{"city":"Shanghai"}',
        },
        {
          type: "function_call_output",
          call_id: "call_1",
          output: '{"forecast":"sunny"}',
        },
      ]),
    );
    expect(JSON.stringify(replayInput)).not.toContain("reason_1");
    expect(JSON.stringify(replayInput)).not.toContain("fc_1");
    expect(callTool).toHaveBeenCalledWith("weather", { city: "Shanghai" });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "It is sunny." }),
    );
  });

  it("V338 streams Codex subscription Responses without a consumer callback", async () => {
    const agent = new Agent({
      name: "codex_unary_caller",
      model: "gpt-5.6-luna",
      client: { apiMode: "codex_responses", accessToken: fakeJwt({}) },
    });
    const response = {
      id: "resp_codex_stream",
      status: "completed",
      error: null,
      output: [],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield { type: "response.output_text.delta", delta: "OK" };
        yield { type: "response.completed", response };
      },
    };
    const create = vi.fn((request: { stream?: boolean }) => {
      if (request.stream !== true) throw new Error("Codex subscription requires streaming.");
      return Promise.resolve(stream);
    });
    Object.defineProperty(agent.client.responses, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Reply OK" }] });

    expect(create.mock.calls[0]?.[0]).toMatchObject({ model: "gpt-5.6-luna", stream: true });
    expect(result.messages).toContainEqual(
      expect.objectContaining({ kind: "message", content: "OK" }),
    );
  });

  it("rejects codex_responses with a non-Responses protocol", () => {
    expect(
      () =>
        new Agent({
          name: "invalid_codex_agent",
          client: { apiProtocol: "openai_chat", apiMode: "codex_responses" },
        }),
    ).toThrow(/requires apiProtocol "openai_responses"/);
  });

  it("V283 executes Anthropic Messages natively with auth-token and tool-result blocks", async () => {
    const agent = new Agent({
      name: "anthropic_agent",
      model: "claude-sonnet-4-6",
      instructions: "Be concise.",
      client: { apiProtocol: "anthropic" },
      process: {
        env: {
          ANTHROPIC_AUTH_TOKEN: "scoped-token",
          ANTHROPIC_BASE_URL: "https://gateway.example/anthropic",
          ANTHROPIC_MODEL: "claude-sonnet-4-6",
        },
      },
      parameters: {
        reasoning: {
          control: "effort_enum",
          effort: "high",
          parameterMapping: { api: "anthropic.messages", path: "output_config.effort" },
        },
      },
    });
    const callTool = vi.fn().mockResolvedValue({ forecast: "cloudy" });
    installMockMcp(agent, callTool);
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-6",
        stop_reason: "tool_use",
        stop_sequence: null,
        container: null,
        usage: {},
        content: [
          { type: "thinking", thinking: "Use the weather tool.", signature: "sig" },
          { type: "text", text: "Checking." },
          {
            type: "tool_use",
            id: "toolu_1",
            name: "weather",
            input: { city: "Shanghai" },
          },
        ],
      })
      .mockResolvedValueOnce({
        id: "msg_2",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-6",
        stop_reason: "end_turn",
        stop_sequence: null,
        container: null,
        usage: {},
        content: [{ type: "text", text: "It is cloudy." }],
      });
    Object.defineProperty(agent.anthropicClient.messages, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "Weather?" }] });

    expect(agent.apiProtocol).toBe("anthropic");
    expect(agent.anthropicClient.authToken).toBe("scoped-token");
    expect(agent.anthropicClient.baseURL).toBe("https://gateway.example/anthropic");
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      model: "claude-sonnet-4-6",
      system: "Be concise.",
      output_config: { effort: "high" },
      stream: false,
      tools: [
        {
          name: "weather",
          description: "Weather lookup",
          input_schema: { type: "object" },
        },
      ],
    });
    expect(create.mock.calls[1]?.[0]?.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ role: "assistant" }),
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "toolu_1",
              content: '{"forecast":"cloudy"}',
            },
          ],
        },
      ]),
    );
    expect(callTool).toHaveBeenCalledWith("weather", { city: "Shanghai" });
    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "Use the weather tool." }),
        expect.objectContaining({ kind: "tool_call", toolName: "weather" }),
        expect.objectContaining({ kind: "message", content: "It is cloudy." }),
      ]),
    );
  });

  it("V283 streams typed OpenAI Responses events without Chat conversion", async () => {
    const agent = new Agent({
      name: "responses_stream_agent",
      model: "gpt-5",
      client: { apiProtocol: "openai_responses" },
    });
    const response = {
      id: "resp_stream",
      status: "completed",
      error: null,
      output: [
        {
          id: "reason_stream",
          type: "reasoning",
          summary: [{ type: "summary_text", text: "Brief thought." }],
        },
        {
          id: "msg_stream",
          type: "message",
          role: "assistant",
          status: "completed",
          content: [{ type: "output_text", text: "Hello", annotations: [] }],
        },
      ],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield { type: "response.reasoning_summary_text.delta", delta: "Brief thought." };
        yield { type: "response.output_text.delta", delta: "Hello" };
        yield { type: "response.completed", response };
      },
    };
    const create = vi.fn().mockResolvedValue(stream);
    Object.defineProperty(agent.client.responses, "create", { value: create });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Hello" }] },
      (chunk) => streamed.push(chunk),
    );

    expect(create.mock.calls[0]?.[0]).toMatchObject({ model: "gpt-5", stream: true });
    expect(streamed).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
    expect(result.messages).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
  });

  it("V283 streams Anthropic content blocks without OpenAI conversion", async () => {
    const agent = new Agent({
      name: "anthropic_stream_agent",
      model: "claude-sonnet-4-6",
      client: { apiProtocol: "anthropic" },
      process: { env: { ANTHROPIC_API_KEY: "scoped-key" } },
    });
    const finalMessage = {
      id: "msg_stream",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-6",
      stop_reason: "end_turn",
      stop_sequence: null,
      container: null,
      usage: {},
      content: [
        { type: "thinking", thinking: "Brief thought.", signature: "sig" },
        { type: "text", text: "Hello" },
      ],
    };
    const stream = {
      async *[Symbol.asyncIterator]() {
        yield {
          type: "content_block_delta",
          index: 0,
          delta: { type: "thinking_delta", thinking: "Brief thought." },
        };
        yield {
          type: "content_block_delta",
          index: 1,
          delta: { type: "text_delta", text: "Hello" },
        };
      },
      finalMessage: vi.fn().mockResolvedValue(finalMessage),
    };
    const createStream = vi.fn().mockReturnValue(stream);
    Object.defineProperty(agent.anthropicClient.messages, "stream", { value: createStream });
    const streamed: MessageChunk[] = [];

    const result = await agent.callStream(
      { messages: [{ role: "user", content: "Hello" }] },
      (chunk) => streamed.push(chunk),
    );

    expect(createStream.mock.calls[0]?.[0]).toMatchObject({
      model: "claude-sonnet-4-6",
      max_tokens: 8192,
    });
    expect(streamed).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
    expect(result.messages).toEqual([
      expect.objectContaining({ kind: "thinking", content: "Brief thought." }),
      expect.objectContaining({ kind: "message", content: "Hello" }),
    ]);
  });

  it.each(["anthropic", "openai_responses"] as const)(
    "V283 passes request cancellation to native %s calls",
    async (apiProtocol) => {
      const agent = new Agent({
        name: `${apiProtocol}_cancel_agent`,
        model: apiProtocol === "anthropic" ? "claude-sonnet-4-6" : "gpt-5",
        client: { apiProtocol },
        process: {
          env:
            apiProtocol === "anthropic"
              ? { ANTHROPIC_API_KEY: "scoped-key" }
              : { OPENAI_API_KEY: "scoped-key" },
        },
      });
      let receivedSignal: AbortSignal | undefined;
      let markStarted!: () => void;
      const started = new Promise<void>((resolve) => {
        markStarted = resolve;
      });
      const create = vi.fn(
        (_body: unknown, options?: { signal?: AbortSignal }) =>
          new Promise<never>((_resolve, reject) => {
            receivedSignal = options?.signal;
            markStarted();
            options?.signal?.addEventListener("abort", () => reject(new Error("aborted")), {
              once: true,
            });
          }),
      );
      if (apiProtocol === "anthropic") {
        Object.defineProperty(agent.anthropicClient.messages, "create", { value: create });
      } else {
        Object.defineProperty(agent.client.responses, "create", { value: create });
      }
      const requestId = `native-${apiProtocol}-cancel`;
      const run = withAcpRequest(requestId, () =>
        agent.call({ messages: [{ role: "user", content: "wait" }] }),
      );
      await started;

      await expect(cancelAcpRequest(requestId)).resolves.toBe(true);
      expect(receivedSignal?.aborted).toBe(true);
      await expect(run).rejects.toBeInstanceOf(RequestCancelledError);
    },
  );

  it("preserves provider reasoning content across tool-call continuation", async () => {
    const agent = new Agent({ name: "deepseek_agent", model: "deepseek-v4-pro" });
    const create = vi
      .fn()
      .mockResolvedValueOnce({
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              reasoning_content: "verified reasoning state",
              tool_calls: [
                {
                  id: "tool-1",
                  type: "function",
                  function: { name: "lookup", arguments: "{}" },
                },
              ],
            },
          },
        ],
      })
      .mockResolvedValueOnce({
        choices: [{ message: { role: "assistant", content: "done" } }],
      });
    Object.defineProperty(agent.client.chat.completions, "create", { value: create });

    const result = await agent.call({ messages: [{ role: "user", content: "reason" }] });

    expect(result.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "thinking", content: "verified reasoning state" }),
      ]),
    );
    expect(create.mock.calls[1]?.[0]?.messages).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          role: "assistant",
          reasoning_content: "verified reasoning state",
          tool_calls: expect.any(Array),
        }),
      ]),
    );
  });
});

function installMockMcp(agent: Agent, callTool: ReturnType<typeof vi.fn>): void {
  Object.defineProperty(agent, "mcp", {
    configurable: true,
    writable: true,
    value: {
      toolsForOpenai: () => [
        {
          type: "function",
          function: {
            name: "weather",
            description: "Weather lookup",
            parameters: { type: "object" },
          },
        },
      ],
      callTool,
      close: vi.fn().mockResolvedValue(undefined),
    },
  });
}

function fakeJwt(claims: Record<string, unknown>): string {
  return `header.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.signature`;
}

function openAIResponseStream(response: unknown) {
  return {
    async *[Symbol.asyncIterator]() {
      yield { type: "response.completed", response };
    },
  };
}

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

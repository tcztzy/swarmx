import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { setImmediate } from "node:timers/promises";
import { runInNewContext } from "node:vm";
import ts from "typescript";
import { expect, it, vi } from "vitest";

const require = createRequire(new URL("../package.json", import.meta.url));
const codex = ts.createSourceFile(
  "codex.js",
  readFileSync(require.resolve("@agentclientprotocol/codex-acp/dist/index.js"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.JS,
);
const claude = ts.createSourceFile(
  "claude.js",
  readFileSync(require.resolve("@agentclientprotocol/claude-agent-acp/dist/acp-agent.js"), "utf8"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.JS,
);
function classNode(source: ts.SourceFile, name: string) {
  let found: ts.ClassExpression | ts.ClassDeclaration | undefined;
  const visit = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === name &&
      node.initializer &&
      ts.isClassExpression(node.initializer)
    )
      found = node.initializer;
    if (ts.isClassDeclaration(node) && node.name?.text === name) found = node;
    ts.forEachChild(node, visit);
  };
  visit(source);
  if (!found) throw new Error(`Upstream class moved: ${name}`);
  return found;
}
function method(
  source: ts.SourceFile,
  owner: string,
  name: string,
  globals: Record<string, unknown> = {},
) {
  const member = classNode(source, owner).members.find(
    (entry) => ts.isMethodDeclaration(entry) && entry.name.getText(source) === name,
  );
  if (!member) throw new Error(`Upstream method moved: ${owner}.${name}`);
  return runInNewContext(`({ ${member.getText(source)} })`, globals)[name] as (
    this: unknown,
    ...args: unknown[]
  ) => Promise<unknown>;
}

it("retains the upstream Ask for approval mode without relabelling it as filesystem isolation", () => {
  const modes = runInNewContext(`(${classNode(codex, "AgentMode").getText(codex)})`);
  expect(modes.ReadOnly.name).toBe("Ask for approval");
  expect(modes.ReadOnly.sandboxPolicy.type).toBe("workspaceWrite");
  expect(modes.ReadOnly.sandboxMode).toBe("workspace-write");
});

it.each([
  { disabled: false, source: "unset", calls: 1 },
  { disabled: false, source: "explicit", calls: 0 },
  { disabled: true, source: "unset", calls: 0 },
])(
  "Codex title model calls: disabled=$disabled, title source=$source",
  async ({ disabled, source, calls }) => {
    const Titles = runInNewContext(`(${classNode(codex, "TitleGenerator").getText(codex)})`, {
      process: { env: disabled ? { ACP_DISABLE_TITLE_GENERATION: "1" } : {} },
      SYSTEM_PROMPT: "Generate a title.",
      TITLE_MODEL: "title-model",
      TITLE_OUTPUT_SCHEMA: {},
      extractTitle: () => undefined,
    });
    const client = {
      threadStart: vi.fn(async () => ({ thread: { id: "title-thread" } })),
      runTurn: vi.fn(async () => ({ turn: {} })),
    };
    const title = new Titles(client, "session", "/workspace", () => source);
    title.onTurnCompleted("summarize this project");
    await setImmediate();
    expect(client.threadStart).toHaveBeenCalledTimes(calls);
    expect(client.runTurn).toHaveBeenCalledTimes(calls);
  },
);

it.each([
  { name: "default", env: {}, settingsEnv: {}, customTitle: undefined, calls: 1 },
  { name: "stored title", env: {}, settingsEnv: {}, customTitle: "Memory review", calls: 0 },
  { name: "native env", env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" }, calls: 0 },
  { name: "native true", env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "TrUe" }, calls: 0 },
  { name: "native zero", env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "0" }, calls: 1 },
  { name: "native false", env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "false" }, calls: 1 },
  {
    name: "settings disable inherited generation",
    env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "0" },
    settingsEnv: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
    calls: 0,
  },
  {
    name: "settings enable inherited generation",
    env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
    settingsEnv: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "0" },
    calls: 1,
  },
  {
    name: "nonessential zero is enabled",
    env: { CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "0" },
    calls: 0,
  },
  {
    name: "session settings override file settings",
    env: {},
    settingsEnv: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "0" },
    flagEnv: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
    calls: 0,
  },
])("Claude title model calls: $name", async ({ env, settingsEnv, flagEnv, customTitle, calls }) => {
  const source = ts.createSourceFile(
    "titles.js",
    readFileSync(
      require.resolve("@agentclientprotocol/claude-agent-acp/dist/session-titles.js"),
      "utf8",
    ),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.JS,
  );
  const declaration = classNode(source, "SessionTitles")
    .getText(source)
    .replace(/^export /, "");
  const Titles = runInNewContext(`(${declaration})`, {
    process: { env: { CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" } },
    resolveSettings: async () => ({
      effective: { env: { FROM_FILE: "kept in file", ...settingsEnv } },
      sources: [],
    }),
    getSessionInfo: async () => (customTitle ? { customTitle, lastModified: 0 } : undefined),
    supportsTitleGeneration: () => true,
    MIN_TITLE_CONTEXT_LENGTH: 10,
    sanitizeTitle: (title: string) => title,
  });
  const generateSessionTitle = vi.fn(async () => "Generated review title");
  const query = { generateSessionTitle };
  const session = { cwd: "/workspace", query };
  const agent = {
    sessions: { session },
    client: { sessionUpdate: vi.fn() },
    logger: { error: vi.fn() },
  };
  const title = new Titles(agent, "session");
  const settings = { disableAllHooks: true, env: { FROM_FLAG: "kept", ...flagEnv } };
  const options = { cwd: session.cwd, env, settings, settingSources: [] };
  await title.initialize(options);
  expect(options.env).toEqual({ ...env, CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" });
  expect(options.settings).toEqual({
    ...settings,
    env: { ...settings.env, CLAUDE_CODE_DISABLE_TERMINAL_TITLE: "1" },
  });
  expect(options.env).not.toBe(env);
  expect(options.settings).not.toBe(settings);
  title.context = "This is a background memory review.";
  await title.onTurnEnd(session);
  await setImmediate();
  await title.onTurnEnd(session);
  await setImmediate();
  expect(generateSessionTitle).toHaveBeenCalledTimes(calls);
  expect(agent.client.sessionUpdate).toHaveBeenCalledTimes(customTitle || calls ? 1 : 0);
  expect(agent.logger.error).not.toHaveBeenCalled();
});

it.each([false, true])(
  "Codex turn dispatch applies review restrictions only when requested: %s",
  async (review) => {
    for (const sandboxPolicy of [
      { type: "readOnly", networkAccess: false },
      { type: "workspaceWrite" },
      { type: "dangerFullAccess" },
    ]) {
      const runTurn = vi.fn(async (params: unknown) => params);
      const addDirectories = vi.fn((policy: unknown) => policy);
      const send = method(codex, "CodexAcpClient", "sendPrompt", {
        process: { env: review ? { SWARMX_MEMORY_REVIEW: "1" } : {} },
        buildPromptItems: () => [],
        addAdditionalDirectoriesToSandboxPolicy: addDirectories,
      });
      await send.call(
        { refreshSkills: async () => {}, codexClient: { runTurn } },
        { sessionId: "session", prompt: [] },
        { approvalPolicy: "on-request", approvalsReviewer: "auto_review", sandboxPolicy },
        { model: "test", effort: "high" },
        null,
        false,
        "/workspace",
        ["/outside"],
        undefined,
        undefined,
      );
      expect(runTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          approvalPolicy: review ? "never" : "on-request",
          approvalsReviewer: review ? "user" : "auto_review",
          sandboxPolicy: review ? { type: "readOnly", networkAccess: false } : sandboxPolicy,
        }),
        undefined,
      );
      expect(addDirectories).toHaveBeenCalledTimes(review ? 0 : 1);
    }
  },
);

it.each([false, true])(
  "Codex disables ambient MCP only for background reviews: %s",
  async (review) => {
    const configRead = vi.fn(async () => ({
      config: { mcp_servers: { outside: {}, swarmx: {} } },
    }));
    const createConfig = method(codex, "CodexAcpClient", "createSessionConfig", {
      process: { env: review ? { SWARMX_MEMORY_REVIEW: "1" } : {} },
      logger: { log() {} },
      mergeGatewayConfig: (config: unknown) => config,
      mergeSandboxWorkspaceWriteRoots: (config: unknown) => config,
    });
    const configured = { sandbox_mode: "danger-full-access", approval_policy: "never" };
    const result = await createConfig.call(
      {
        config: configured,
        gatewayConfig: null,
        codexClient: { configRead },
        getNativeProviderConfig: () => ({}),
        getModelProvider: () => "test",
      },
      "/workspace",
      [],
      [],
    );
    expect(result).toEqual({
      ...configured,
      projects: { "/workspace": { trust_level: "trusted" } },
      ...(review
        ? { mcp_servers: { outside: { enabled: false }, swarmx: { enabled: false } } }
        : {}),
    });
    expect(configRead).toHaveBeenCalledTimes(review ? 1 : 0);
  },
);

it.each([false, true])("Codex forwards ephemeral session intent: %s", async (ephemeral) => {
  const threadStart = vi.fn(async () => ({ thread: { id: "session" }, model: "test" }));
  const create = method(codex, "CodexAcpClient", "newSession", {
    readAdditionalDirectories: () => [],
  });
  await create.call(
    {
      refreshSkills: async () => {},
      codexClient: { threadStart },
      createSessionConfig: async () => ({}),
      getModelProvider: () => "test",
      fetchAvailableModels: async () => ["test"],
      createModelId: () => "test",
      getCollaborationMode: () => null,
    },
    { cwd: "/workspace", mcpServers: [], ...(ephemeral ? { _meta: { ephemeral } } : {}) },
  );
  expect(threadStart).toHaveBeenCalledWith(expect.objectContaining({ ephemeral }));
});

it("patched Claude newSession forwards an empty reservation without marking it as a resume", async () => {
  const createSession = vi.fn(async () => ({ sessionId: "reserved" }));
  const create = method(claude, "ClaudeAcpAgent", "newSession", { setTimeout() {} });
  await create.call(
    { createSession },
    {
      cwd: "/workspace",
      mcpServers: [],
      _meta: { claudeCode: { options: { sessionId: "reserved" } } },
    },
  );
  expect(createSession).toHaveBeenCalledWith(expect.anything(), {
    resume: undefined,
    reuseSessionId: "reserved",
  });
});

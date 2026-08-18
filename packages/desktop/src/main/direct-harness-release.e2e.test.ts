import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const MODEL_ID = "swarmx-release-e2e-model";
const PLACEHOLDER_CREDENTIAL = "release-e2e-placeholder-not-a-real-key";
const USER_PROMPT = "Return the deterministic release acceptance response.";
const FINAL_RESPONSE = "release golden path";
const AMBIENT_TRIPWIRES = {
  OPENAI_API_KEY: "ambient-openai-must-not-be-used",
  ANTHROPIC_API_KEY: "ambient-anthropic-must-not-be-used",
  DEEPSEEK_API_KEY: "ambient-deepseek-must-not-be-used",
  OLLAMA_HOST: "http://127.0.0.1:1",
} as const;

const temporaryRoots: string[] = [];
const originalEnvironment = new Map<string, string | undefined>();
let activeServer: Server | undefined;

afterEach(async () => {
  if (activeServer?.listening) await closeServer(activeServer);
  activeServer = undefined;
  for (const [key, value] of originalEnvironment) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  originalEnvironment.clear();
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("SwarmX direct Harness release acceptance", () => {
  it("recovers an explicit Provider, cached Model, and persisted Session after services restart", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-direct-release-e2e-"));
    temporaryRoots.push(root);
    const home = path.join(root, "home");
    const swarmxHome = path.join(home, ".swarmx");
    const settingsPath = path.join(swarmxHome, "settings.json");
    const authPath = path.join(swarmxHome, "provider-auth.json");
    const cachePath = path.join(swarmxHome, "model-catalog-cache.json");
    const sessionsDir = path.join(swarmxHome, "projects");
    await mkdir(swarmxHome, { recursive: true });
    isolateEnvironment(home, sessionsDir);

    const core = await import("@swarmx/core");
    const runtime = await core.createCoreRuntime();
    const { ModelCatalogService } = await import("./model-catalog.js");
    const { FileProviderAuthStore } = await import("./provider-auth.js");
    const { DesktopSettingsStore } = await import("./settings-store.js");
    const baseInventory = core.createExtensionInventory([core.builtInExtensionBundle()]);
    const authStore = new FileProviderAuthStore({ path: authPath });
    const settingsStore = new DesktopSettingsStore({ path: settingsPath });
    const catalogService = new ModelCatalogService({
      settingsStore,
      cachePath,
      authStore,
      includeCodex: false,
      env: { ...AMBIENT_TRIPWIRES },
    });

    const pristine = await catalogService.list(baseInventory);
    expect(pristine.providers).toEqual([]);
    expect(pristine.models).toEqual([]);
    expect(pristine.modelSupplies).toEqual([]);

    const provider = await startFakeProvider();
    activeServer = provider.server;
    const configured = await catalogService.saveProvider(baseInventory, {
      label: "Release E2E Provider",
      kind: "openai_chat",
      baseUrl: `${provider.origin}/v1`,
      authMode: "api_key",
      secret: PLACEHOLDER_CREDENTIAL,
    });
    expect(configured.modelCatalog.userProviderIds).toEqual(["swarmx.user.release-e2e-provider"]);

    const catalog = await catalogService.refresh(baseInventory);
    const supply = catalog.modelSupplies.find((candidate) => candidate.modelId === MODEL_ID);
    expect(provider.catalogRequests).toBe(1);
    expect(supply).toBeDefined();
    expect(catalog.models).toContainEqual(expect.objectContaining({ id: MODEL_ID }));
    expect(catalog.modelCatalog.providers).toEqual([
      expect.objectContaining({ status: "ready", modelCount: 1 }),
    ]);

    const composition = {
      id: "release-e2e-composition",
      harnessId: "swarmx",
      modelId: MODEL_ID,
      modelSupplyId: supply?.id,
      host: "local" as const,
      skills: [],
      mcpServers: [],
      pluginIds: [],
      plugins: [],
    };
    const plan = core.resolveAgentCompositionPlan(composition, catalog);
    expect(plan).toMatchObject({
      status: "ready",
      agentId: `swarmx:${MODEL_ID}`,
      harnessId: "swarmx",
      modelId: MODEL_ID,
      modelSupplyId: supply?.id,
      apiProtocol: "openai_chat",
    });

    const session = core.createSession("release-e2e-agent", "swarmx", MODEL_ID);
    session.messages = [{ role: "user", kind: "message", content: USER_PROMPT }];
    core.saveSession(session);
    const providerSecrets = await catalogService.runtimeSecretsForSupply(
      catalog,
      supply?.id ?? "missing",
    );
    const streamed: Array<{ kind?: string; content: string }> = [];
    const usages: Array<{ inputTokens: number; outputTokens: number; totalTokens: number }> = [];
    const finalMessages = await core.executeAgentComposition(composition, session.messages, {
      runtime,
      inventory: catalog,
      env: { ...AMBIENT_TRIPWIRES },
      providerSecrets,
      onChunk: (chunk) => streamed.push({ kind: chunk.kind, content: chunk.content }),
      onUsage: (usage) => usages.push(usage),
    });
    expect(core.appendMessages(session.id, finalMessages)).toBe(true);

    expect(provider.completionRequests).toBe(1);
    expect(provider.authorizationHeaders).toEqual([
      `Bearer ${PLACEHOLDER_CREDENTIAL}`,
      `Bearer ${PLACEHOLDER_CREDENTIAL}`,
    ]);
    expect(provider.completionBodies).toEqual([
      expect.objectContaining({ model: MODEL_ID, stream: true }),
    ]);
    expect(streamed.filter((chunk) => chunk.kind === "message")).toEqual([
      expect.objectContaining({ content: "release golden " }),
      expect.objectContaining({ content: "path" }),
    ]);
    expect(finalMessages).toContainEqual(
      expect.objectContaining({ role: "assistant", kind: "message", content: FINAL_RESPONSE }),
    );
    expect(usages).toEqual([
      expect.objectContaining({ inputTokens: 7, outputTokens: 3, totalTokens: 10 }),
    ]);
    expect(core.loadSession(session.id)?.messages.at(-1)).toMatchObject({
      role: "assistant",
      kind: "message",
      content: FINAL_RESPONSE,
    });

    const settingsText = await readFile(settingsPath, "utf8");
    const authText = await readFile(authPath, "utf8");
    const cacheText = await readFile(cachePath, "utf8");
    expect(settingsText).not.toContain(PLACEHOLDER_CREDENTIAL);
    expect(cacheText).not.toContain(PLACEHOLDER_CREDENTIAL);
    expect(authText).toContain(PLACEHOLDER_CREDENTIAL);
    expect([settingsText, authText, cacheText].join("\n")).not.toContain(
      AMBIENT_TRIPWIRES.OPENAI_API_KEY,
    );
    expect(await authStore.fileMode()).toBe(0o600);

    await closeServer(provider.server);
    activeServer = undefined;
    await rm(path.join(sessionsDir, "__recents__", "sessions-index.json"), { force: true });
    vi.resetModules();

    const restartedCore = await import("@swarmx/core");
    const { ModelCatalogService: RestartedModelCatalogService } = await import(
      "./model-catalog.js"
    );
    const { FileProviderAuthStore: RestartedFileProviderAuthStore } = await import(
      "./provider-auth.js"
    );
    const { DesktopSettingsStore: RestartedDesktopSettingsStore } = await import(
      "./settings-store.js"
    );
    const offlineFetch = vi.fn(async () => {
      throw new Error("Provider discovery must remain offline during restart recovery.");
    });
    const restartedAuthStore = new RestartedFileProviderAuthStore({ path: authPath });
    const restartedCatalogService = new RestartedModelCatalogService({
      settingsStore: new RestartedDesktopSettingsStore({ path: settingsPath }),
      cachePath,
      authStore: restartedAuthStore,
      includeCodex: false,
      env: { ...AMBIENT_TRIPWIRES },
      fetch: offlineFetch,
    });
    const restartedInventory = restartedCore.createExtensionInventory([
      restartedCore.builtInExtensionBundle(),
    ]);
    const recoveredCatalog = await restartedCatalogService.list(restartedInventory);
    const recoveredPlan = restartedCore.resolveAgentCompositionPlan(composition, recoveredCatalog);
    const recoveredSession = restartedCore.loadSession(session.id);

    expect(offlineFetch).not.toHaveBeenCalled();
    expect(provider.catalogRequests).toBe(1);
    expect(recoveredCatalog.modelCatalog.userProviderIds).toEqual([
      "swarmx.user.release-e2e-provider",
    ]);
    expect(recoveredCatalog.modelCatalog.providers).toEqual([
      expect.objectContaining({ status: "cached", modelCount: 1 }),
    ]);
    expect(recoveredPlan).toMatchObject({
      status: "ready",
      agentId: `swarmx:${MODEL_ID}`,
      modelSupplyId: supply?.id,
    });
    await expect(
      restartedCatalogService.runtimeSecretsForSupply(recoveredCatalog, supply?.id ?? "missing"),
    ).resolves.toEqual({ "swarmx.user.release-e2e-provider": PLACEHOLDER_CREDENTIAL });
    expect(recoveredSession?.messages).toEqual([
      expect.objectContaining({ role: "user", content: USER_PROMPT }),
      expect.objectContaining({ role: "assistant", content: FINAL_RESPONSE }),
    ]);
    await runtime.dispose();
  }, 20_000);
});

function isolateEnvironment(home: string, sessionsDir: string): void {
  const isolated = {
    HOME: home,
    USERPROFILE: home,
    SWARMX_SESSIONS_DIR: sessionsDir,
    SWARMX_EXTENSION_PATHS: "",
    ...AMBIENT_TRIPWIRES,
  };
  for (const [key, value] of Object.entries(isolated)) {
    if (!originalEnvironment.has(key)) originalEnvironment.set(key, process.env[key]);
    process.env[key] = value;
  }
}

async function startFakeProvider(): Promise<{
  server: Server;
  origin: string;
  catalogRequests: number;
  completionRequests: number;
  authorizationHeaders: string[];
  completionBodies: unknown[];
}> {
  const state = {
    catalogRequests: 0,
    completionRequests: 0,
    authorizationHeaders: [] as string[],
    completionBodies: [] as unknown[],
  };
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      const authorization = request.headers.authorization ?? "";
      state.authorizationHeaders.push(authorization);
      if (authorization !== `Bearer ${PLACEHOLDER_CREDENTIAL}`) {
        response.writeHead(401, { "content-type": "application/json" });
        response.end(JSON.stringify({ error: { message: "unexpected credential" } }));
        return;
      }
      if (request.method === "GET" && url.pathname === "/v1/models") {
        state.catalogRequests += 1;
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ data: [{ id: MODEL_ID }] }));
        return;
      }
      if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
        state.completionRequests += 1;
        state.completionBodies.push(await readJsonBody(request));
        await writeStreamingCompletion(response);
        return;
      }
      response.writeHead(404, { "content-type": "application/json" });
      response.end(JSON.stringify({ error: { message: "unexpected request" } }));
    } catch (error) {
      response.writeHead(500, { "content-type": "application/json" });
      response.end(
        JSON.stringify({
          error: { message: error instanceof Error ? error.message : String(error) },
        }),
      );
    }
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
  const address = server.address() as AddressInfo;
  return {
    server,
    origin: `http://127.0.0.1:${address.port}`,
    get catalogRequests() {
      return state.catalogRequests;
    },
    get completionRequests() {
      return state.completionRequests;
    },
    authorizationHeaders: state.authorizationHeaders,
    completionBodies: state.completionBodies,
  };
}

async function readJsonBody(request: IncomingMessage): Promise<unknown> {
  let body = "";
  for await (const chunk of request) {
    body += chunk.toString();
    if (body.length > 1_000_000) throw new Error("Request body exceeded the E2E bound.");
  }
  return JSON.parse(body);
}

async function writeStreamingCompletion(response: ServerResponse): Promise<void> {
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  response.write(
    sseChunk({
      id: "release-e2e",
      object: "chat.completion.chunk",
      created: 0,
      model: MODEL_ID,
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "release golden " },
          finish_reason: null,
        },
      ],
    }),
  );
  await new Promise<void>((resolve) => setImmediate(resolve));
  response.write(
    sseChunk({
      id: "release-e2e",
      object: "chat.completion.chunk",
      created: 0,
      model: MODEL_ID,
      choices: [{ index: 0, delta: { content: "path" }, finish_reason: "stop" }],
    }),
  );
  response.write(
    sseChunk({
      id: "release-e2e",
      object: "chat.completion.chunk",
      created: 0,
      model: MODEL_ID,
      choices: [],
      usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
    }),
  );
  response.end("data: [DONE]\n\n");
}

function sseChunk(value: unknown): string {
  return `data: ${JSON.stringify(value)}\n\n`;
}

async function closeServer(server: Server): Promise<void> {
  server.closeIdleConnections();
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

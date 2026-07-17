import { EventEmitter } from "node:events";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import {
  ActivityStore,
  builtInExtensionBundle,
  createExtensionInventory,
  parseExtensionBundle,
  removeProject,
} from "@swarmx/core";
import type { SessionData } from "@swarmx/core";
import { describe, expect, it, vi } from "vitest";

const electron = vi.hoisted(() => ({
  handle: vi.fn(),
  showOpenDialog: vi.fn(),
  showSaveDialog: vi.fn(),
  showItemInFolder: vi.fn(),
  isEncryptionAvailable: vi.fn(() => true),
  encryptString: vi.fn((value: string) => Buffer.from(value)),
  decryptString: vi.fn((value: Buffer) => value.toString("utf8")),
}));

vi.mock("electron", () => ({
  ipcMain: { handle: electron.handle },
  dialog: {
    showOpenDialog: electron.showOpenDialog,
    showSaveDialog: electron.showSaveDialog,
  },
  safeStorage: {
    isEncryptionAvailable: electron.isEncryptionAvailable,
    encryptString: electron.encryptString,
    decryptString: electron.decryptString,
  },
  shell: { showItemInFolder: electron.showItemInFolder },
}));

const desktopMain = await import("./library.js");
const desktopIpc = await import("./ipc.js");

describe("desktop main library entry", () => {
  it("exports host integration without registering handlers on import", () => {
    expect(desktopMain.registerIpcHandlers).toBeTypeOf("function");
    expect(desktopMain.AgentInteractionBroker).toBeTypeOf("function");
    expect(desktopMain.DesktopRequestRegistry).toBeTypeOf("function");
    expect(desktopMain.HarnessEnvironmentService).toBeTypeOf("function");
    expect(desktopMain.HarnessDoctor).toBeTypeOf("function");
    expect(desktopMain.LspHost).toBeTypeOf("function");
    expect(desktopMain.ModelCatalogService).toBeTypeOf("function");
    expect(desktopMain.EncryptedFileProviderAuthStore).toBeTypeOf("function");
    expect(electron.handle).not.toHaveBeenCalled();
  });

  it("refuses a desktop send without an explicit Harness x Model composition", async () => {
    desktopMain.registerIpcHandlers();
    const registration = electron.handle.mock.calls.find(([channel]) => channel === "agent:send");
    const handler = registration?.[1];
    if (typeof handler !== "function") throw new Error("agent:send handler was not registered");
    const sender = new EventEmitter();
    Object.assign(sender, { id: 1 });

    await expect(
      handler(
        { sender },
        {
          requestId: "missing-provider-model",
          harnessId: "swarmx",
          userText: "hello",
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("requires an Agent Composition with an explicit Model"),
    });

    await expect(
      handler(
        { sender },
        {
          requestId: "inline-agent-config",
          harnessId: "swarmx",
          userText: "hello",
          agentConfig: { name: "legacy_agent", model: "gpt-5" },
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("Inline agentConfig is not accepted"),
    });

    await expect(
      handler(
        { sender },
        {
          requestId: "implicit-workflow-model",
          harnessId: "swarmx",
          userText: "hello",
          swarmConfig: {
            name: "implicit_workflow",
            root: "agent",
            nodes: {
              agent: { kind: "agent", agent: { name: "agent" } },
            },
            edges: [],
          },
        },
      ),
    ).resolves.toMatchObject({
      success: false,
      error: expect.stringContaining("requires an explicit Model"),
    });
  });

  it("records failed tasks and estimated token usage for the Profile summary", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-activity-"));
    const activityStore = new ActivityStore({ filePath: path.join(root, "activity.jsonl") });

    try {
      desktopMain.registerIpcHandlers({ activityStore });
      const sendRegistration = [...electron.handle.mock.calls]
        .reverse()
        .find(([channel]) => channel === "agent:send");
      const profileRegistration = [...electron.handle.mock.calls]
        .reverse()
        .find(([channel]) => channel === "activity:profile");
      const sendHandler = sendRegistration?.[1];
      const profileHandler = profileRegistration?.[1];
      if (typeof sendHandler !== "function" || typeof profileHandler !== "function") {
        throw new Error("Activity IPC handlers were not registered");
      }
      const sender = new EventEmitter();
      Object.assign(sender, { id: 41 });

      await expect(
        sendHandler(
          { sender },
          {
            requestId: "profile-failed-task",
            sessionId: "profile-session",
            harnessId: "swarmx",
            userText: "Record this failed request",
          },
        ),
      ).resolves.toMatchObject({ success: false });

      expect(profileHandler()).toMatchObject({
        lifetime: {
          totalTasks: 1,
          completedTasks: 0,
          totalTokens: expect.any(Number),
          estimatedTokens: expect.any(Number),
        },
      });
      expect(profileHandler().lifetime.estimatedTokens).toBeGreaterThan(0);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it("V346 resolves host coding tools from the runtime Harness adapter", () => {
    expect(
      desktopIpc.compositionRuntimeHarnessId(
        {
          harnesses: [
            {
              id: "custom-swarmx-harness",
              runtimeHarnessId: "swarmx",
            },
          ],
        },
        { harnessId: "custom-swarmx-harness" },
      ),
    ).toBe("swarmx");
    expect(
      desktopIpc.compositionRuntimeHarnessId(
        { harnesses: [{ id: "custom-codex-harness", runtimeHarnessId: "codex" }] },
        { harnessId: "custom-codex-harness" },
      ),
    ).toBe("codex");
  });

  it("V429 replays only persisted conversational messages into session activations", () => {
    const session = {
      messages: [
        { role: "user", content: "question", kind: "message" },
        { role: "assistant", content: "private reasoning", kind: "thinking" },
        { role: "assistant", content: "answer", kind: "message" },
        { role: "assistant", content: "{}", kind: "tool_call", toolName: "Read" },
        { role: "system", content: "scheduled event", kind: "message" },
      ],
    } as SessionData;

    expect(desktopIpc.sessionChatMessages(session)).toEqual([
      { role: "user", content: "question" },
      { role: "assistant", content: "answer" },
      { role: "system", content: "scheduled event" },
    ]);
  });

  it("V353/V355 scopes streamed chunks and rejects a reasoning-only terminal result", () => {
    const send = vi.fn();
    const publish = desktopIpc.agentChunkPublisher(
      { isDestroyed: () => false, send },
      "request-live-work",
    );
    const thought = { role: "assistant", kind: "thinking" as const, content: "Inspecting" };

    publish(thought);
    expect(send).toHaveBeenCalledWith("agent:chunk", {
      requestId: "request-live-work",
      chunk: thought,
    });
    expect(() => desktopIpc.assertFinalAssistantMessage([thought])).toThrow(
      /without a final assistant response/i,
    );
    expect(() =>
      desktopIpc.assertFinalAssistantMessage([
        thought,
        { role: "assistant", kind: "message", content: "Complete." },
      ]),
    ).not.toThrow();
  });

  it("opens the native file and folder picker only through an explicit IPC request", async () => {
    electron.showOpenDialog.mockResolvedValue({
      canceled: false,
      filePaths: ["/workspace/src/App.tsx", "/workspace/docs"],
    });
    desktopMain.registerIpcHandlers();
    const registration = electron.handle.mock.calls.find(
      ([channel]) => channel === "workspace:selectFilesAndFolders",
    );
    const handler = registration?.[1];
    if (typeof handler !== "function") throw new Error("file picker handler was not registered");

    await expect(handler()).resolves.toEqual(["/workspace/src/App.tsx", "/workspace/docs"]);
    expect(electron.showOpenDialog).toHaveBeenCalledWith(
      expect.objectContaining({
        properties: ["openFile", "openDirectory", "multiSelections"],
      }),
    );
  });

  it("V322 registers the canonical project only after the native folder picker confirms it", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-project-"));
    let projectId: string | undefined;
    try {
      electron.showOpenDialog.mockResolvedValue({ canceled: false, filePaths: [root] });
      desktopMain.registerIpcHandlers();
      const registration = electron.handle.mock.calls.find(
        ([channel]) => channel === "project:addExisting",
      );
      const handler = registration?.[1];
      if (typeof handler !== "function") {
        throw new Error("project picker handler was not registered");
      }

      const project = await handler();
      projectId = project?.id;
      expect(project).toMatchObject({ name: path.basename(root), cwd: await realpath(root) });
      expect(electron.showOpenDialog).toHaveBeenCalledWith(
        expect.objectContaining({ properties: ["openDirectory", "createDirectory"] }),
      );

      const projectHandler = (channel: string) =>
        electron.handle.mock.calls.filter(([registered]) => registered === channel).at(-1)?.[1];
      const pinHandler = projectHandler("project:setPinned");
      const renameHandler = projectHandler("project:rename");
      const revealHandler = projectHandler("project:reveal");
      const archiveHandler = projectHandler("project:archiveTasks");
      const removeHandler = projectHandler("project:remove");
      if (
        typeof pinHandler !== "function" ||
        typeof renameHandler !== "function" ||
        typeof revealHandler !== "function" ||
        typeof archiveHandler !== "function" ||
        typeof removeHandler !== "function" ||
        !projectId
      ) {
        throw new Error("project action handlers were not registered");
      }

      expect(pinHandler({}, { id: projectId, pinned: true })).toMatchObject({ pinned: true });
      expect(renameHandler({}, { id: projectId, name: "Renamed project" })).toMatchObject({
        name: "Renamed project",
      });
      expect(revealHandler({}, { id: projectId })).toBe(true);
      expect(electron.showItemInFolder).toHaveBeenCalledWith(await realpath(root));
      expect(archiveHandler({}, { id: projectId })).toBe(0);
      expect(removeHandler({}, { id: projectId })).toBe(true);
    } finally {
      if (projectId) removeProject(projectId);
      await rm(root, { recursive: true, force: true });
    }
  });

  it("exposes update state/install IPC and broadcasts service progress", async () => {
    const available = {
      phase: "available" as const,
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
    };
    const restarting = {
      phase: "restarting" as const,
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
      progress: 100,
    };
    let publish: ((state: typeof restarting) => void) | undefined;
    const updateService = {
      getState: vi.fn(() => available),
      check: vi.fn(async () => available),
      startUpdate: vi.fn(async () => restarting),
      subscribe: vi.fn((listener: (state: typeof restarting) => void) => {
        publish = listener;
        return () => undefined;
      }),
    };
    const broadcastUpdateState = vi.fn();
    desktopMain.registerIpcHandlers({ updateService, broadcastUpdateState });
    const stateHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:getState")
      .at(-1)?.[1];
    const installHandler = electron.handle.mock.calls
      .filter(([channel]) => channel === "appUpdate:install")
      .at(-1)?.[1];
    if (typeof stateHandler !== "function" || typeof installHandler !== "function") {
      throw new Error("update handlers were not registered");
    }

    expect(stateHandler()).toEqual(available);
    await expect(installHandler()).resolves.toEqual(restarting);
    expect(updateService.startUpdate).toHaveBeenCalledTimes(1);
    publish?.(restarting);
    expect(broadcastUpdateState).toHaveBeenCalledWith(restarting);
  });

  it("blocks extension agents whose runtime secret is unavailable", () => {
    const bundle = parseExtensionBundle({
      id: "runtime-readiness",
      name: "Runtime readiness",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "missing-secret-provider",
            label: "Missing secret provider",
            kind: "openai_chat",
            secretRef: { source: "env", key: "MISSING_TEST_API_KEY" },
          },
        ],
        modelSupplies: [
          {
            id: "gpt-5-missing-secret",
            modelId: "gpt-5",
            providerProfileId: "missing-secret-provider",
          },
        ],
        agents: [
          {
            id: "blocked-agent",
            name: "Blocked agent",
            harnessId: "swarmx",
            modelId: "gpt-5",
            modelSupplyId: "gpt-5-missing-secret",
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);
    const projected = desktopIpc.extensionInventoryWithPlans(inventory, {});

    expect(
      projected.providers.find((provider) => provider.id === "missing-secret-provider"),
    ).toMatchObject({
      runtimeReady: false,
      runtimeNote: expect.stringContaining("MISSING_TEST_API_KEY"),
    });
    expect(projected.agentPlans[0]).toMatchObject({
      status: "blocked",
      healthStatus: "blocked",
      requirements: expect.arrayContaining([
        expect.objectContaining({
          kind: "model_supply",
          status: "unavailable",
          id: "gpt-5-missing-secret",
        }),
      ]),
    });
  });

  it("translates protected supply routes without changing Model identity", () => {
    const bundle = parseExtensionBundle({
      id: "bridge-routes",
      name: "Bridge routes",
      version: "1.0.0",
      capabilities: {
        providers: [
          {
            id: "local-anthropic",
            label: "Local Anthropic",
            kind: "anthropic",
            baseUrl: "http://localhost:9000",
          },
        ],
        modelSupplies: [
          {
            id: "gpt-local",
            modelId: "gpt-5",
            providerProfileId: "local-anthropic",
            apiCompatibility: { mode: "bridge", baseUrl: "http://127.0.0.1:4000/v1" },
          },
        ],
      },
    });
    const inventory = createExtensionInventory([builtInExtensionBundle(), bundle]);

    const translated = desktopIpc.containerizeCompositionSupplyRoutes(inventory);

    expect(translated.models.find((model) => model.id === "gpt-5")?.id).toBe("gpt-5");
    expect(
      translated.providers.find((provider) => provider.id === "local-anthropic")?.baseUrl,
    ).toBe("http://host.docker.internal:9000");
    expect(
      translated.modelSupplies.find((supply) => supply.id === "gpt-local")?.apiCompatibility
        .baseUrl,
    ).toBe("http://host.docker.internal:4000/v1");
  });

  it("requires explicit opt-in before probing a ready native ACP harness", () => {
    const status = {
      checkedAt: "2026-07-11T00:00:00.000Z",
      path: "/usr/bin",
      ready: true,
      setupAvailable: false,
      containerRuntimes: [],
      protection: { mode: "protected" as const, ready: true, requiredHarnessIds: [] },
      requirements: [],
      harnesses: [
        harnessStatus("swarmx", "native"),
        harnessStatus("claude_code", "protected"),
        harnessStatus("opencode", "native"),
        harnessStatus("hermes", "native"),
        harnessStatus("openclaw", "native"),
      ],
    };

    expect(desktopIpc.sessionDiscoveryHarnessIds(status)).toEqual([]);
    expect(desktopIpc.sessionDiscoveryHarnessIds(status, ["codex", "hermes"])).toEqual(["hermes"]);
  });

  it("transforms queen and nested swarm agent backends", async () => {
    const config = {
      name: "outer",
      root: "root_agent",
      queen: { name: "queen", backend: { type: "custom" as const, program: "queen-acp" } },
      nodes: {
        root_agent: {
          kind: "agent" as const,
          agent: { name: "root_agent", backend: { type: "custom" as const, program: "root-acp" } },
        },
        nested: {
          kind: "swarm" as const,
          swarm: {
            name: "nested",
            root: "nested_agent",
            nodes: {
              nested_agent: {
                kind: "agent" as const,
                agent: {
                  name: "nested_agent",
                  backend: { type: "custom" as const, program: "nested-acp" },
                },
              },
            },
            edges: [],
          },
        },
      },
      edges: [],
    };

    const transformed = await desktopIpc.transformSwarmConfigAgentBackends(
      config,
      async (backend) =>
        backend.type === "custom"
          ? { ...backend, program: `protected-${backend.program}` }
          : backend,
    );

    expect(transformed.queen?.backend).toMatchObject({ program: "protected-queen-acp" });
    expect(transformed.nodes.root_agent).toMatchObject({
      agent: { backend: { program: "protected-root-acp" } },
    });
    expect(transformed.nodes.nested).toMatchObject({
      swarm: {
        nodes: {
          nested_agent: { agent: { backend: { program: "protected-nested-acp" } } },
        },
      },
    });
    expect(config.queen.backend.program).toBe("queen-acp");
  });
});

function harnessStatus(harnessId: string, executionMode: "native" | "protected") {
  return {
    harnessId,
    harnessLabel: harnessId,
    status: "ready" as const,
    requirements: [],
    executionMode,
    protectionRequired: executionMode === "protected",
  };
}

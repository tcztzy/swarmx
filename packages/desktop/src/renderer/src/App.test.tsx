/** @vitest-environment jsdom */

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SWRConfig } from "swr";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppProps } from "./App.js";

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: class {
    fit(): void {}
  },
}));

vi.mock("@xterm/xterm", () => ({
  Terminal: class {
    cols = 80;
    rows = 24;
    options: Record<string, unknown> = {};
    loadAddon(): void {}
    open(): void {}
    onData(): { dispose(): void } {
      return { dispose: () => undefined };
    }
    write(): void {}
    writeln(): void {}
    focus(): void {}
    reset(): void {}
    dispose(): void {}
  },
}));

type MessageKind = "message" | "thinking" | "tool_call" | "tool_result";

interface MessageChunk {
  role: string;
  content: string;
  kind: MessageKind;
  agent?: string;
  render?: {
    artifacts?: Array<{
      artifactId?: string;
      byteCount?: number;
      kind: string;
      path?: string;
      title?: string;
      truncated?: boolean;
    }>;
    durationMs?: number;
    endedAt?: string;
    invocationId?: string;
    provenance?: Record<string, string>;
    rawPayloadRef?: string;
    status?: string;
    startedAt?: string;
  };
  swarmEvent?: string;
  toolName?: string;
}

interface SessionData {
  id: string;
  title: string;
  projectId?: string;
  cwd?: string;
  agentName: string;
  harness: string;
  pinned?: boolean;
  messages: MessageChunk[];
  createdAt: string;
  updatedAt: string;
}

interface DiscoveredSession {
  id: string;
  title: string;
  projectId?: string;
  cwd: string;
  pinned?: boolean;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: "local" | "acp";
}

interface ProjectData {
  id: string;
  name: string;
  cwd: string;
  pinned: boolean;
  createdAt: string;
  updatedAt: string;
}

interface DesktopApiMock {
  initialProjects?: ProjectData[];
  sendMessage: ReturnType<typeof vi.fn>;
  onAgentChunk: ReturnType<typeof vi.fn>;
  onAgentInteraction: ReturnType<typeof vi.fn>;
  onSessionMessages: ReturnType<typeof vi.fn>;
  resolveAgentInteraction: ReturnType<typeof vi.fn>;
  cancelMessage: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  saveSession: ReturnType<typeof vi.fn>;
  loadSession: ReturnType<typeof vi.fn>;
  loadDiscoveredSession: ReturnType<typeof vi.fn>;
  listSessions: ReturnType<typeof vi.fn>;
  getActivityProfile: ReturnType<typeof vi.fn>;
  listProjects: ReturnType<typeof vi.fn>;
  addExistingProject: ReturnType<typeof vi.fn>;
  createScratchProject: ReturnType<typeof vi.fn>;
  setProjectPinned: ReturnType<typeof vi.fn>;
  renameProject: ReturnType<typeof vi.fn>;
  revealProject: ReturnType<typeof vi.fn>;
  archiveProjectTasks: ReturnType<typeof vi.fn>;
  removeProject: ReturnType<typeof vi.fn>;
  listGroupedSessions: ReturnType<typeof vi.fn>;
  deleteSession: ReturnType<typeof vi.fn>;
  renameSession: ReturnType<typeof vi.fn>;
  setSessionPinned: ReturnType<typeof vi.fn>;
  generateSessionTitle: ReturnType<typeof vi.fn>;
  appendMessages: ReturnType<typeof vi.fn>;
  importN8nWorkflow: ReturnType<typeof vi.fn>;
  listExtensions: ReturnType<typeof vi.fn>;
  getExtensionManagementState: ReturnType<typeof vi.fn>;
  saveExtensionSource: ReturnType<typeof vi.fn>;
  refreshExtensionSource: ReturnType<typeof vi.fn>;
  removeExtensionSource: ReturnType<typeof vi.fn>;
  applyExtensionAction: ReturnType<typeof vi.fn>;
  saveSkillEvolutionPolicy: ReturnType<typeof vi.fn>;
  listCustomAgents: ReturnType<typeof vi.fn>;
  saveCustomAgent: ReturnType<typeof vi.fn>;
  removeCustomAgent: ReturnType<typeof vi.fn>;
  workspaceRoot: ReturnType<typeof vi.fn>;
  createTerminal: ReturnType<typeof vi.fn>;
  writeTerminal: ReturnType<typeof vi.fn>;
  resizeTerminal: ReturnType<typeof vi.fn>;
  killTerminal: ReturnType<typeof vi.fn>;
  onTerminalData: ReturnType<typeof vi.fn>;
  onTerminalExit: ReturnType<typeof vi.fn>;
  selectFilesAndFolders: ReturnType<typeof vi.fn>;
  refreshModelCatalog: ReturnType<typeof vi.fn>;
  addManualModel: ReturnType<typeof vi.fn>;
  removeManualModel: ReturnType<typeof vi.fn>;
  saveProvider: ReturnType<typeof vi.fn>;
  removeProvider: ReturnType<typeof vi.fn>;
  refreshProviderUsage: ReturnType<typeof vi.fn>;
  getUpdateState: ReturnType<typeof vi.fn>;
  startUpdate: ReturnType<typeof vi.fn>;
  onUpdateState: ReturnType<typeof vi.fn>;
  getHarnessEnvironment: ReturnType<typeof vi.fn>;
  getHarnessVersion: ReturnType<typeof vi.fn>;
  inspectDoctor: ReturnType<typeof vi.fn>;
  fixDoctor: ReturnType<typeof vi.fn>;
  setupHarnessEnvironment: ReturnType<typeof vi.fn>;
  lspComplete: ReturnType<typeof vi.fn>;
  lspStop: ReturnType<typeof vi.fn>;
  loadImageDataUrl: ReturnType<typeof vi.fn>;
}

const localSession: SessionData = {
  id: "local-1",
  title: "Existing local run",
  agentName: "agent",
  harness: "swarmx",
  messages: [
    {
      role: "user",
      kind: "message",
      content: "Summarize local state",
    },
  ],
  createdAt: "2026-06-11T09:00:00.000Z",
  updatedAt: "2026-06-11T09:05:00.000Z",
};

const discoveredAcpSession: DiscoveredSession = {
  id: "acp-1",
  title: "ACP investigation",
  cwd: "/Users/tcztzy/swarmx",
  harnessId: "codex",
  harnessLabel: "Codex",
  source: "acp",
  updatedAt: "2026-06-11T10:00:00.000Z",
};

const swarmxProject: ProjectData = {
  id: "project-swarmx",
  name: "swarmx",
  cwd: "/Users/tcztzy/swarmx",
  pinned: false,
  createdAt: "2026-06-11T08:00:00.000Z",
  updatedAt: "2026-06-11T08:00:00.000Z",
};

const acpSessionDetail: SessionData = {
  id: "acp-1",
  acpSessionId: "acp-1",
  title: "ACP investigation",
  agentName: "agent",
  harness: "codex",
  messages: [
    {
      role: "user",
      kind: "message",
      content: "Inspect the failing UI path",
    },
    {
      role: "assistant",
      kind: "message",
      agent: "codex",
      content: "Previous ACP answer",
    },
  ],
  createdAt: "2026-06-11T10:00:00.000Z",
  updatedAt: "2026-06-11T10:05:00.000Z",
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.resetModules();
  Object.defineProperty(window, "swarmxAPI", {
    configurable: true,
    value: undefined,
  });
});

describe("App user workflow", () => {
  it("uses one accessible send/stop control and cancels only the active request", async () => {
    const reply = deferred<{ success: boolean; messages: MessageChunk[] }>();
    const api = createDesktopApiMock({
      sendMessage: vi.fn(() => reply.promise),
    });
    await renderApp(api);
    const user = userEvent.setup();

    const input = screen.getByRole("textbox");
    const emptySend = screen.getByRole("button", { name: "Send message" });
    expect(emptySend.hasAttribute("disabled")).toBe(true);

    fireEvent.change(input, { target: { value: "Stop this request" } });
    expect(emptySend.hasAttribute("disabled")).toBe(false);
    await user.click(emptySend);

    const stop = await screen.findByRole("button", { name: "Stop generating" });
    const requestId = api.sendMessage.mock.calls[0]?.[0]?.requestId;
    expect(requestId).toEqual(expect.any(String));
    await user.click(stop);
    expect(api.cancelMessage).toHaveBeenCalledWith(requestId);
    expect(stop.hasAttribute("disabled")).toBe(true);

    reply.resolve({ success: true, messages: [] });
    await screen.findByRole("button", { name: "Send message" });
  }, 15_000);

  it("does not fall back to an implicit model when no standalone Model is registered", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      models: [],
      modelSupplies: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    const send = await screen.findByRole("button", { name: "Send message" });
    fireEvent.change(screen.getByRole("textbox"), {
      target: { value: "Do not guess a model" },
    });

    expect(send.hasAttribute("disabled")).toBe(true);
    expect(send.getAttribute("title")).toMatch(/Register a compatible standalone Model/);
    expect(api.sendMessage).not.toHaveBeenCalled();
  }, 15_000);

  it("returns to running when cancellation is not accepted", async () => {
    const reply = deferred<{ success: boolean; messages: MessageChunk[] }>();
    const api = createDesktopApiMock({
      sendMessage: vi.fn(() => reply.promise),
      cancelMessage: vi.fn(async (requestId: string) => ({ requestId, canceled: false })),
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.type(screen.getByRole("textbox"), "Keep running");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await user.click(await screen.findByRole("button", { name: "Stop generating" }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Stop generating" }).hasAttribute("disabled")).toBe(
        false,
      ),
    );
    reply.resolve({ success: true, messages: [] });
    await screen.findByRole("button", { name: "Send message" });
  });

  it("honors Stop before the request reaches main", async () => {
    const saveGate = deferred<void>();
    const api = createDesktopApiMock({
      saveSession: vi.fn(() => saveGate.promise),
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.type(screen.getByRole("textbox"), "Cancel during persistence");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    await user.click(await screen.findByRole("button", { name: "Stop generating" }));

    expect(api.cancelMessage).not.toHaveBeenCalled();
    saveGate.resolve();
    await screen.findByRole("button", { name: "Send message" });
    expect(api.sendMessage).not.toHaveBeenCalled();
  });

  it("shows IPC transport failures without an unhandled rejection", async () => {
    const api = createDesktopApiMock({
      sendMessage: vi.fn(async () => {
        throw new Error("transport unavailable");
      }),
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.type(screen.getByRole("textbox"), "Handle transport failure");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect((await screen.findByRole("alert")).textContent).toContain("transport unavailable");
    expect(screen.getByRole("button", { name: "Send message" })).toBeTruthy();
  });

  it("V429 reloads the authoritative session when a background activation appends messages", async () => {
    let notifySession = (_event: { sessionId: string }): void => undefined;
    const api = createDesktopApiMock({
      onSessionMessages: vi.fn((listener: (event: { sessionId: string }) => void) => {
        notifySession = listener;
        return () => undefined;
      }),
    });
    await renderApp(api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /Existing local run/ }));
    await screen.findByText("Summarize local state");

    api.loadSession.mockResolvedValue({
      ...localSession,
      messages: [
        ...localSession.messages,
        {
          role: "assistant",
          kind: "message",
          content: "Background build finished successfully",
        },
      ],
    });
    notifySession({ sessionId: localSession.id });

    expect(await screen.findByText("Background build finished successfully")).toBeTruthy();
  });

  it("V394 shows only Harness, Model, and Effort while routing Supply internally", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_chat"],
        },
        {
          id: "codex",
          label: "Codex",
          modelControl: "session",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_responses"],
        },
      ],
      models: [
        {
          id: "chat-model",
          label: "chat-model",
          runtimeModel: "chat-model",
          apiProtocols: ["openai_chat"],
        },
        {
          id: "responses-model",
          label: "responses-model",
          runtimeModel: "responses-model",
          apiProtocols: ["openai_responses"],
        },
      ],
      providers: [{ id: "responses", label: "Responses", kind: "openai_responses" }],
      modelSupplies: [
        {
          id: "responses-supply",
          modelId: "responses-model",
          providerProfileId: "responses",
          runtimeModel: "responses-model-vendor-alias",
          apiCompatibility: { mode: "native" },
        },
      ],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const primary = screen.getByTestId("agent-picker-primary");
    const leftBefore = primary.getBoundingClientRect().left;
    await user.click(within(primary).getByRole("menuitem", { name: /^Harness/ }));
    const codexIcon = document.querySelector<HTMLImageElement>('[data-harness-icon="codex"]');
    expect(codexIcon?.getAttribute("src")).toMatch(/^data:image\/svg\+xml/);
    await user.click(screen.getByRole("menuitemradio", { name: "Codex" }));

    await waitFor(() =>
      expect(
        document.querySelector(".agent-picker__trigger")?.getAttribute("data-harness-id"),
      ).toBe("codex"),
    );
    await waitFor(() =>
      expect(document.querySelector(".agent-picker__trigger")?.textContent).toContain(
        "responses-model",
      ),
    );
    expect(primary.getBoundingClientRect().left).toBe(leftBefore);
    await user.click(within(primary).getByRole("menuitem", { name: /Modelresponses-model/ }));
    expect(screen.getByRole("menu", { name: "model options" })).toBeTruthy();
    expect(screen.queryByRole("menuitemradio", { name: /chat-model/ })).toBeNull();
    expect(within(primary).getAllByRole("menuitem")).toHaveLength(3);
    expect(within(primary).queryByRole("menuitem", { name: /^Supply/ })).toBeNull();
    expect(primary.getBoundingClientRect().left).toBe(leftBefore);

    await user.type(screen.getByRole("textbox"), "Use the selected composition");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(api.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        harnessId: "codex",
        agentComposition: expect.objectContaining({
          harnessId: "codex",
          modelId: "responses-model",
        }),
      }),
    );
    expect(api.sendMessage.mock.calls[0]?.[0]?.agentComposition).toEqual(
      expect.objectContaining({ modelSupplyId: "responses-supply" }),
    );
  }, 15_000);

  it("refreshes Provider Models and manages manual Models inside the Model menu", async () => {
    const api = createDesktopApiMock();
    const base = await api.listExtensions();
    const remoteModel = {
      id: "remote-provider-model",
      label: "Remote Provider Model",
      runtimeModel: "remote-provider-model",
      apiProtocols: ["openai_chat"],
      readOnly: true,
    };
    const refreshed = {
      ...base,
      models: [...base.models, remoteModel],
      modelCatalog: {
        manualModelIds: [],
        providers: [
          {
            providerProfileId: "explicit-openai",
            label: "OpenAI",
            status: "ready" as const,
            modelCount: 1,
          },
        ],
      },
    };
    const manualModel = {
      id: "manual-local-model",
      label: "Manual Local Model",
      runtimeModel: "runtime/manual-local-model",
      apiProtocols: ["openai_chat"],
      readOnly: false,
    };
    const withManual = {
      ...refreshed,
      models: [...refreshed.models, manualModel],
      modelCatalog: {
        ...refreshed.modelCatalog,
        manualModelIds: [manualModel.id],
      },
    };
    api.refreshModelCatalog.mockResolvedValue(refreshed);
    api.addManualModel.mockResolvedValue(withManual);
    api.removeManualModel.mockResolvedValue(refreshed);
    await renderApp(api);
    const user = userEvent.setup();

    expect(api.refreshModelCatalog).not.toHaveBeenCalled();
    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const primary = screen.getByTestId("agent-picker-primary");
    await user.click(within(primary).getByRole("menuitem", { name: /^Model/ }));
    const modelOptions = screen.getByRole("menu", { name: "model options" });
    expect(
      within(modelOptions).queryByRole("menuitemradio", { name: "Remote Provider Model" }),
    ).toBeNull();
    await user.click(within(modelOptions).getByRole("button", { name: "Refresh" }));
    await waitFor(() => expect(api.refreshModelCatalog).toHaveBeenCalledTimes(1));
    expect(
      await screen.findByRole("menuitemradio", { name: "Remote Provider Model" }),
    ).toBeTruthy();
    expect(within(primary).getAllByRole("menuitem")).toHaveLength(3);
    expect(within(primary).queryByRole("menuitem", { name: /^Supply/ })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Add model" }));
    await user.type(screen.getByLabelText("Model ID"), manualModel.id);
    await user.type(screen.getByLabelText("Runtime model"), manualModel.runtimeModel);
    await user.type(screen.getByLabelText("Display name"), manualModel.label);
    await user.selectOptions(screen.getByLabelText("API protocol"), "openai_chat");
    await user.click(screen.getByRole("button", { name: "Save model" }));

    expect(api.addManualModel).toHaveBeenCalledWith({
      id: manualModel.id,
      label: manualModel.label,
      runtimeModel: manualModel.runtimeModel,
      apiProtocol: "openai_chat",
    });
    expect(await screen.findByRole("menuitemradio", { name: /Manual Local Model/ })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: `Remove manual model ${manualModel.id}` }));
    expect(api.removeManualModel).toHaveBeenCalledWith(manualModel.id);
  });

  it("V282 reuses cached Provider Models across Renderer restarts without discovery", async () => {
    const api = createDesktopApiMock();
    const base = await api.listExtensions();
    const cachedModel = {
      id: "cached-provider-model",
      label: "Cached Provider Model",
      runtimeModel: "cached-provider-model",
      apiProtocols: ["openai_chat"],
      readOnly: true,
    };
    api.listExtensions.mockResolvedValue({
      ...base,
      models: [...base.models, cachedModel],
      modelCatalog: {
        manualModelIds: [],
        userProviderIds: ["explicit-openai"],
        providers: [
          {
            providerProfileId: "explicit-openai",
            label: "OpenAI",
            status: "cached",
            modelCount: 1,
            fetchedAt: "2026-07-12T08:00:00.000Z",
          },
        ],
        refreshedAt: "2026-07-12T08:00:00.000Z",
      },
    });
    api.listExtensions.mockClear();

    await renderApp(api);
    let user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    await user.click(
      within(screen.getByTestId("agent-picker-primary")).getByRole("menuitem", {
        name: /^Model/,
      }),
    );
    expect(
      await screen.findByRole("menuitemradio", { name: "Cached Provider Model" }),
    ).toBeTruthy();
    expect(api.refreshModelCatalog).not.toHaveBeenCalled();

    cleanup();
    await renderApp(api);
    user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    await user.click(
      within(screen.getByTestId("agent-picker-primary")).getByRole("menuitem", {
        name: /^Model/,
      }),
    );
    expect(
      await screen.findByRole("menuitemradio", { name: "Cached Provider Model" }),
    ).toBeTruthy();
    expect(api.listExtensions).toHaveBeenCalledTimes(2);
    expect(api.refreshModelCatalog).not.toHaveBeenCalled();
  });

  it("keeps a single Settings action in the account menu and hides the updater when current", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();

    const accountTrigger = await screen.findByRole("button", {
      name: "Open anonymous user menu",
    });
    expect(screen.queryByRole("button", { name: /Update SwarmX/ })).toBeNull();
    expect(screen.queryByText("Check for updates")).toBeNull();

    await user.click(accountTrigger);
    const accountMenu = screen.getByRole("menu", { name: "Anonymous user menu" });
    expect(within(accountMenu).getAllByRole("menuitem")).toHaveLength(1);
    expect(within(accountMenu).queryByRole("menuitem", { name: "Usage remaining" })).toBeNull();
    expect(within(accountMenu).getByRole("menuitem", { name: "Settings" })).toBeTruthy();
    expect(within(accountMenu).queryByText("Show pet")).toBeNull();
    expect(within(accountMenu).queryByText("Log out")).toBeNull();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Anonymous user menu" })).toBeNull();

    await user.click(accountTrigger);
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("menu", { name: "Anonymous user menu" })).toBeNull();

    await user.click(accountTrigger);
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    const settings = screen.getByRole("region", { name: "Settings" });
    const settingsNavigation = screen.getByRole("complementary", {
      name: "Settings navigation",
    });
    expect(screen.queryByRole("complementary", { name: "Sessions" })).toBeNull();
    expect(within(settingsNavigation).getByRole("button", { name: "Back to app" })).toBeTruthy();
    expect(
      within(settingsNavigation).getByRole("searchbox", { name: "Search settings" }),
    ).toBeTruthy();
    expect(
      within(settingsNavigation).getByRole("navigation", { name: "Settings sections" }),
    ).toBeTruthy();
    expect(within(settings).getByRole("heading", { name: "Anonymous user" })).toBeTruthy();
    expect(within(settings).getByText("Lifetime tokens")).toBeTruthy();
    expect(api.getActivityProfile).toHaveBeenCalledTimes(1);
    expect(api.refreshProviderUsage).not.toHaveBeenCalled();

    await user.type(
      within(settingsNavigation).getByRole("searchbox", { name: "Search settings" }),
      "providers",
    );
    expect(within(settingsNavigation).queryByRole("button", { name: "Usage" })).toBeNull();
    await user.click(within(settingsNavigation).getByRole("button", { name: "Providers" }));
    expect(screen.getByRole("heading", { name: "Providers" })).toBeTruthy();
    await waitFor(() => expect(api.refreshProviderUsage).toHaveBeenCalledTimes(1));
    expect(screen.getByText("Codex")).toBeTruthy();
    expect(screen.getByText("93% left")).toBeTruthy();
  });

  it("switches Profile activity aggregation and capability rankings", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));

    expect(await screen.findByText("1.3K")).toBeTruthy();
    expect(screen.getByText("paper-reviewer")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Weekly" }));
    expect(screen.getByRole("button", { name: "Weekly" }).getAttribute("aria-pressed")).toBe(
      "true",
    );
    await user.click(screen.getByRole("button", { name: "Tools" }));
    expect(screen.getByText("workspace_read_file")).toBeTruthy();
  });

  it("shows Codex, OpenAI, and DeepSeek as peers in one fixed Provider matrix", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    const openaiProviderId = "swarmx.user.openai";
    const providerId = "swarmx.user.deepseek";
    const minimaxProviderId = "swarmx.user.minimax";
    api.listExtensions.mockResolvedValue({
      ...inventory,
      providers: [
        {
          id: openaiProviderId,
          label: "OpenAI",
          kind: "openai_responses",
          baseUrl: "https://api.openai.com/v1",
          authMode: "api_key",
          runtimeReady: true,
          readOnly: false,
        },
        {
          id: providerId,
          label: "DeepSeek",
          kind: "openai_chat",
          baseUrl: "https://api.deepseek.com",
          authMode: "api_key",
          runtimeReady: true,
          readOnly: false,
        },
        {
          id: minimaxProviderId,
          label: "MiniMax",
          kind: "anthropic",
          baseUrl: "https://api.minimax.io/anthropic",
          authMode: "auth_token",
          runtimeReady: true,
          readOnly: true,
        },
      ],
      modelCatalog: {
        manualModelIds: [],
        userProviderIds: [openaiProviderId, providerId],
        providers: [
          {
            providerProfileId: openaiProviderId,
            label: "OpenAI",
            status: "ready",
            modelCount: 3,
          },
          {
            providerProfileId: providerId,
            label: "DeepSeek",
            status: "ready",
            modelCount: 2,
          },
          {
            providerProfileId: minimaxProviderId,
            label: "MiniMax",
            status: "ready",
            modelCount: 1,
          },
        ],
      },
    });
    api.refreshProviderUsage.mockResolvedValue({
      fetchedAt: "2026-07-12T12:00:00.000Z",
      providers: [
        {
          source: "provider",
          sourceId: openaiProviderId,
          providerProfileId: openaiProviderId,
          label: "OpenAI",
          adapterId: "openai_api",
          status: "unsupported",
          meters: [],
          detail:
            "OpenAI API keys do not expose Codex subscription quota; Codex login usage is shown separately.",
        },
        {
          source: "provider",
          sourceId: providerId,
          providerProfileId: providerId,
          label: "DeepSeek",
          adapterId: "deepseek",
          status: "ready",
          meters: [
            {
              kind: "balance",
              label: "CNY balance",
              currency: "CNY",
              total: "110.00",
              granted: "10.00",
              toppedUp: "100.00",
            },
          ],
        },
        {
          source: "provider",
          sourceId: minimaxProviderId,
          providerProfileId: minimaxProviderId,
          label: "MiniMax",
          adapterId: "minimax",
          status: "ready",
          meters: [
            {
              kind: "window",
              id: "weekly:MiniMax-M2.7",
              label: "MiniMax-M2.7 Weekly",
              usedPercent: 0,
              remainingPercent: 150,
            },
          ],
        },
      ],
      toolAccounts: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Providers" }));

    const providerList = screen.getByRole("region", { name: "Provider usage matrix" });
    expect(within(providerList).getByText("Codex")).toBeTruthy();
    expect(within(providerList).getByText("OpenAI")).toBeTruthy();
    expect(within(providerList).getByText("DeepSeek")).toBeTruthy();
    expect(providerList.querySelector(".settings-provider-matrix__header")?.textContent).toContain(
      "Credit & balance",
    );
    expect(within(providerList).queryByText("Tool accounts")).toBeNull();
    expect(within(providerList).queryByText("Model Providers")).toBeNull();
    const providerCard = within(providerList).getByRole("article", {
      name: "DeepSeek Provider",
    });
    expect(providerCard.textContent).toContain("OpenAI Chat + Anthropic · Preferred OpenAI Chat");
    const finance = within(providerCard).getByRole("button", {
      name: /DeepSeek credit and balance/,
    });
    expect(finance.textContent).toContain("110.00");
    expect(finance.querySelector(":scope > small")?.textContent).toBe("CNY balance");
    expect(within(finance).getByRole("tooltip").textContent).toContain("Granted");
    expect(within(finance).getByRole("tooltip").textContent).toContain("Paid");
    const minimaxCard = within(providerList).getByRole("article", {
      name: "MiniMax Provider",
    });
    expect(minimaxCard.textContent).toContain("150% left");
    expect(
      within(minimaxCard)
        .getByRole("progressbar", { name: "7-day remaining" })
        .getAttribute("aria-valuenow"),
    ).toBe("100");
    expect(providerCard).not.toBeNull();
    expect(
      within(providerCard).getByRole("button", {
        name: "Edit Provider DeepSeek",
      }),
    ).toBeTruthy();
    expect(
      within(providerCard).getByRole("button", {
        name: "Remove Provider DeepSeek",
      }),
    ).toBeTruthy();
    expect(within(providerList).getByRole("button", { name: "Edit Provider OpenAI" })).toBeTruthy();
    expect(
      within(providerList)
        .getByRole("article", { name: "OpenAI Provider" })
        .querySelector('img[src="./harness-icons/codex.svg"]'),
    ).toBeTruthy();
    expect(
      within(within(providerList).getByRole("article", { name: "OpenAI Provider" })).getAllByText(
        "Not provided",
      ),
    ).toHaveLength(4);
    expect(
      within(providerList).getByRole("article", { name: "Codex Provider" }).textContent,
    ).toContain("OpenAI official · Local account");
    expect(providerCard.querySelector('img[src="./provider-icons/deepseek.svg"]')).toBeTruthy();
    await user.click(within(providerCard).getByRole("button", { name: "Edit Provider DeepSeek" }));
    expect(screen.getByLabelText("Preferred API protocol")).toBeTruthy();
    expect(screen.getByText(/DeepSeek supports native OpenAI and Anthropic APIs/)).toBeTruthy();
  });

  it("refreshes only the selected Provider row and merges its targeted snapshot", async () => {
    const providerId = "swarmx.user.deepseek";
    const targeted = deferred<Record<string, unknown>>();
    const initialSnapshot = {
      fetchedAt: "2026-07-12T12:00:00.000Z",
      providers: [
        {
          source: "provider",
          sourceId: providerId,
          providerProfileId: providerId,
          label: "DeepSeek",
          adapterId: "deepseek",
          status: "ready",
          fetchedAt: "2026-07-12T12:00:00.000Z",
          meters: [
            {
              kind: "balance",
              label: "CNY balance",
              currency: "CNY",
              total: "10.00",
            },
          ],
        },
      ],
      toolAccounts: [
        {
          source: "tool_account",
          sourceId: "codex",
          label: "Codex",
          adapterId: "codex_app_server",
          status: "ready",
          meters: [
            {
              kind: "window",
              id: "five_hour",
              label: "5-hour",
              usedPercent: 7,
              remainingPercent: 93,
            },
          ],
        },
      ],
    };
    const api = createDesktopApiMock({
      refreshProviderUsage: vi.fn((target) =>
        target ? targeted.promise : Promise.resolve(initialSnapshot),
      ),
    });
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      providers: [
        {
          id: providerId,
          label: "DeepSeek",
          kind: "anthropic",
          baseUrl: "https://api.deepseek.com/anthropic",
          authMode: "auth_token",
          runtimeReady: true,
          readOnly: false,
        },
      ],
      modelCatalog: {
        manualModelIds: [],
        userProviderIds: [providerId],
        providers: [
          {
            providerProfileId: providerId,
            label: "DeepSeek",
            status: "ready",
            modelCount: 2,
          },
        ],
      },
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Providers" }));
    const refreshDeepSeek = await screen.findByRole("button", {
      name: "Refresh DeepSeek usage",
    });
    const refreshCodex = screen.getByRole("button", { name: "Refresh Codex usage" });
    await user.click(refreshDeepSeek);

    expect(api.refreshProviderUsage).toHaveBeenLastCalledWith({
      source: "provider",
      sourceId: providerId,
    });
    expect(refreshDeepSeek.hasAttribute("disabled")).toBe(true);
    expect(refreshCodex.hasAttribute("disabled")).toBe(false);
    expect(screen.getByRole("button", { name: "Refresh all" }).hasAttribute("disabled")).toBe(true);

    targeted.resolve({
      fetchedAt: "2026-07-12T12:05:00.000Z",
      providers: [
        {
          ...initialSnapshot.providers[0],
          fetchedAt: "2026-07-12T12:05:00.000Z",
          meters: [
            {
              kind: "balance",
              label: "CNY balance",
              currency: "CNY",
              total: "25.00",
            },
          ],
        },
      ],
      toolAccounts: [],
    });
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /DeepSeek credit and balance/ }).textContent,
      ).toContain("25.00"),
    );
    expect(
      within(screen.getByRole("article", { name: "Codex Provider" })).getByText("93% left"),
    ).toBeTruthy();
  });

  it("expands New API account and token summaries without aggregating token balances", async () => {
    const packyId = "swarmx.user.packy";
    const disconnectedId = "swarmx.user.new-api-disconnected";
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      providers: [
        {
          id: packyId,
          label: "packy cc sale",
          kind: "anthropic",
          baseUrl: "https://www.packyapi.com",
          authMode: "auth_token",
          usageAdapter: "new_api",
          newApiAccountUserId: "7",
          accountAccessReady: true,
          runtimeReady: true,
          readOnly: false,
        },
        {
          id: disconnectedId,
          label: "New API staging",
          kind: "openai_chat",
          baseUrl: "https://new-api.example.test",
          authMode: "api_key",
          usageAdapter: "new_api",
          accountAccessReady: false,
          runtimeReady: true,
          readOnly: false,
        },
      ],
      modelCatalog: {
        manualModelIds: [],
        userProviderIds: [packyId, disconnectedId],
        providers: [],
      },
    });
    api.refreshProviderUsage.mockResolvedValue({
      fetchedAt: "2026-07-12T12:00:00.000Z",
      providers: [
        {
          source: "provider",
          sourceId: packyId,
          providerProfileId: packyId,
          label: "packy cc sale",
          adapterId: "new_api",
          status: "ready",
          fetchedAt: "2026-07-12T12:00:00.000Z",
          meters: [
            {
              kind: "balance",
              label: "Account balance",
              currency: "USD",
              total: "9.50",
            },
            {
              kind: "credit",
              label: "Primary API token",
              remaining: "Unlimited",
              unit: "quota",
            },
          ],
          account: {
            kind: "new_api",
            status: "ready",
            displayName: "Packy user",
            group: "default",
            balance: { remaining: "9.50", used: "0.50", total: "10.00", unit: "USD" },
            tokens: [
              {
                id: "token-primary-123456",
                name: "primary",
                status: "active",
                remaining: "$2.00",
                used: "$1.00",
                total: "$3.00",
              },
              {
                id: "token-batch-654321",
                name: "batch",
                status: "exhausted",
                remaining: "$0.00",
                used: "$4.00",
                total: "$4.00",
              },
            ],
            totalTokens: 2,
          },
        },
      ],
      toolAccounts: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Providers" }));
    const packy = await screen.findByRole("article", { name: "packy cc sale Provider" });
    expect(packy.querySelector('img[src="./provider-icons/packy.svg"]')).toBeTruthy();
    expect(within(packy).getByRole("button", { name: /credit and balance: \$9.50/ })).toBeTruthy();
    await user.click(within(packy).getByText("Account & API tokens"));
    const tokens = within(packy).getByRole("region", { name: "New API tokens" });
    expect(within(tokens).getByText("primary")).toBeTruthy();
    expect(within(tokens).getByText("batch")).toBeTruthy();
    expect(within(tokens).getByText("$2.00")).toBeTruthy();
    expect(within(tokens).getByText("$0.00")).toBeTruthy();
    expect(packy.textContent).not.toContain("$11.50");

    const disconnected = screen.getByRole("article", { name: "New API staging Provider" });
    await user.click(within(disconnected).getByText("Account & API tokens"));
    expect(
      within(disconnected).getByText("Connect account access to see wallet and API tokens."),
    ).toBeTruthy();
    expect(
      within(disconnected).getByRole("button", { name: "Manage account access" }),
    ).toBeTruthy();
  });

  it("shows the Codex account-row updater only when available and renders live progress", async () => {
    const installGate = deferred<{
      phase: "restarting";
      currentVersion: string;
      latestVersion: string;
      progress: number;
    }>();
    let publishUpdate:
      | ((state: {
          phase: "available" | "downloading" | "installing" | "restarting";
          currentVersion: string;
          latestVersion: string;
          progress?: number;
        }) => void)
      | undefined;
    const api = createDesktopApiMock({
      getUpdateState: vi.fn(async () => ({
        phase: "available",
        currentVersion: "3.0.1",
        latestVersion: "3.0.2",
      })),
      startUpdate: vi.fn(() => installGate.promise),
      onUpdateState: vi.fn((listener) => {
        publishUpdate = listener;
        return vi.fn();
      }),
    });
    await renderApp(api);
    const user = userEvent.setup();

    const update = await screen.findByRole("button", { name: "Update SwarmX to 3.0.2" });
    expect(update.closest(".sidebar-account-row")).toBeTruthy();
    expect(screen.queryByText("Check for updates")).toBeNull();
    expect(within(update).getByText("Update")).toBeTruthy();

    await user.hover(update);
    await user.click(update);
    expect(api.startUpdate).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Downloading SwarmX 0%" })).toBeTruthy();

    act(() =>
      publishUpdate?.({
        phase: "downloading",
        currentVersion: "3.0.1",
        latestVersion: "3.0.2",
        progress: 42,
      }),
    );
    expect(screen.getByRole("button", { name: "Downloading SwarmX 42%" })).toBeTruthy();
    expect(screen.getByText("42%")).toBeTruthy();

    act(() =>
      publishUpdate?.({
        phase: "installing",
        currentVersion: "3.0.1",
        latestVersion: "3.0.2",
        progress: 100,
      }),
    );
    expect(screen.getByRole("button", { name: "Installing SwarmX 3.0.2" })).toBeTruthy();
    expect(screen.getByText("Installing")).toBeTruthy();

    act(() =>
      publishUpdate?.({
        phase: "restarting",
        currentVersion: "3.0.1",
        latestVersion: "3.0.2",
        progress: 100,
      }),
    );
    expect(screen.getByRole("button", { name: "Restarting SwarmX 3.0.2" })).toBeTruthy();
    installGate.resolve({
      phase: "restarting",
      currentVersion: "3.0.1",
      latestVersion: "3.0.2",
      progress: 100,
    });
  });

  it("V281 configures Providers with stable labels and keeps the Agent Picker model-only", async () => {
    const api = createDesktopApiMock();
    const base = await api.listExtensions();
    const emptyCatalog = {
      ...base,
      providers: [],
      modelCatalog: { manualModelIds: [], userProviderIds: [], providers: [] },
    };
    const providerId = "swarmx.user.anthropic-gateway";
    const configuredProvider = {
      id: providerId,
      label: "Anthropic Gateway",
      kind: "anthropic",
      baseUrl: "https://gateway.example.test/anthropic",
      authMode: "auth_token" as const,
      usageAdapter: "new_api" as const,
      newApiAccountUserId: "42",
      accountAccessReady: true,
      runtimeReady: true,
      readOnly: false,
    };
    const configuredCatalog = {
      ...emptyCatalog,
      providers: [configuredProvider],
      modelCatalog: {
        manualModelIds: [],
        userProviderIds: [providerId],
        providers: [
          {
            providerProfileId: providerId,
            label: configuredProvider.label,
            status: "ready" as const,
            modelCount: 2,
          },
        ],
      },
    };
    const renamedCatalog = {
      ...configuredCatalog,
      providers: [{ ...configuredProvider, label: "Team Anthropic" }],
    };
    api.listExtensions.mockResolvedValue(emptyCatalog);
    api.refreshModelCatalog.mockResolvedValue(emptyCatalog);
    api.saveProvider.mockResolvedValueOnce(configuredCatalog).mockResolvedValueOnce(renamedCatalog);
    api.removeProvider.mockResolvedValue(emptyCatalog);
    await renderApp(api);
    const user = userEvent.setup();

    expect(api.refreshModelCatalog).not.toHaveBeenCalled();
    await user.click(await screen.findByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Providers" }));
    expect(screen.getByRole("heading", { name: "Providers" })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Add Provider" }));
    fireEvent.change(screen.getByLabelText("Provider name"), {
      target: { value: configuredProvider.label },
    });
    await user.selectOptions(screen.getByLabelText("API protocol"), "anthropic");
    fireEvent.change(screen.getByLabelText("Base URL"), {
      target: { value: configuredProvider.baseUrl },
    });
    await user.selectOptions(screen.getByLabelText("Authentication"), "auth_token");
    await user.selectOptions(screen.getByLabelText("Usage API"), "new_api");
    fireEvent.change(screen.getByLabelText("Primary API token"), {
      target: { value: "user-entered-token" },
    });
    fireEvent.change(screen.getByLabelText("New API user ID"), { target: { value: "42" } });
    fireEvent.change(screen.getByLabelText("Account access token"), {
      target: { value: "account-access-token" },
    });
    expect(screen.getByText(/high-privilege management credential/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Save Provider" }));

    expect(api.saveProvider).toHaveBeenNthCalledWith(1, {
      label: configuredProvider.label,
      kind: "anthropic",
      baseUrl: configuredProvider.baseUrl,
      authMode: "auth_token",
      usageAdapter: "new_api",
      secret: "user-entered-token",
      accountAccessToken: "account-access-token",
      accountUserId: "42",
    });
    expect(await screen.findByText(/2 models/)).toBeTruthy();
    expect(document.body.textContent).not.toContain("user-entered-token");
    await waitFor(() => expect(api.refreshProviderUsage).toHaveBeenCalledTimes(2));

    await user.click(
      screen.getByRole("button", { name: `Edit Provider ${configuredProvider.label}` }),
    );
    expect((screen.getByLabelText("Primary API token") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("New API user ID") as HTMLInputElement).value).toBe("42");
    expect((screen.getByLabelText("Usage API") as HTMLSelectElement).value).toBe("new_api");
    fireEvent.change(screen.getByLabelText("Provider name"), {
      target: { value: "Team Anthropic" },
    });
    await user.click(screen.getByRole("button", { name: "Save Provider" }));
    expect(api.saveProvider).toHaveBeenNthCalledWith(2, {
      id: providerId,
      label: "Team Anthropic",
      kind: "anthropic",
      baseUrl: configuredProvider.baseUrl,
      authMode: "auth_token",
      usageAdapter: "new_api",
      accountUserId: "42",
    });
    await waitFor(() => expect(api.refreshProviderUsage).toHaveBeenCalledTimes(3));

    await user.click(await screen.findByRole("button", { name: "Remove Provider Team Anthropic" }));
    expect(api.removeProvider).toHaveBeenCalledWith(providerId);
    await waitFor(() => expect(api.refreshProviderUsage).toHaveBeenCalledTimes(4));
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Remove Provider Team Anthropic" })).toBeNull(),
    );

    await user.click(screen.getByRole("button", { name: "Back to app" }));
    expect(screen.getByRole("complementary", { name: "Sessions" })).toBeTruthy();
    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const primary = screen.getByTestId("agent-picker-primary");
    await user.click(within(primary).getByRole("menuitem", { name: /^Model/ }));
    expect(within(primary).getAllByRole("menuitem")).toHaveLength(3);
    expect(within(primary).queryByRole("menuitem", { name: /^Provider/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Providers" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Add connection" })).toBeNull();
  }, 15_000);

  it("keeps unconfigured OpenClaw visible but disabled in Harness selection", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_chat"],
        },
        {
          id: "codex",
          label: "Codex",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
        {
          id: "openclaw",
          label: "OpenClaw",
          modelControl: "unsupported",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
      ],
    });
    await renderApp(api);
    const user = userEvent.setup();

    const trigger = await screen.findByRole("button", { name: "Choose agent" });
    await user.click(trigger);
    await user.click(
      within(screen.getByTestId("agent-picker-primary")).getByRole("menuitem", {
        name: /^Harness/,
      }),
    );

    const openClaw = screen.getByRole("menuitemradio", { name: /OpenClaw/ });
    expect(openClaw.hasAttribute("disabled")).toBe(true);
    expect(openClaw.getAttribute("aria-disabled")).toBe("true");
    expect(openClaw.getAttribute("title")).toMatch(/Model switching is not configured/);
    expect(screen.getByRole("menuitemradio", { name: "Codex" }).hasAttribute("disabled")).toBe(
      false,
    );

    fireEvent.click(openClaw);
    expect(trigger.getAttribute("data-harness-id")).toBe("swarmx");
  });

  it("V204 keeps a long secondary menu separate from the primary menu", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    const modelIds = [
      "gpt-5.5",
      "gpt-5.4",
      "gpt-5.4-mini",
      "gpt-5.4-nano",
      "gpt-5.2",
      "gpt-5.1",
      "gpt-5",
      "gpt-5-mini",
      "gpt-5-nano",
      "gpt-4.1",
      "gpt-4.1-mini",
      "gpt-4o-mini",
    ];
    api.listExtensions.mockResolvedValue({
      ...inventory,
      models: modelIds.map((id) => ({
        id,
        label: id,
        runtimeModel: id,
        apiProtocols: ["openai_chat"],
      })),
      modelSupplies: [],
      providers: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const menu = screen.getByRole("menu", { name: "Agent composition" });
    const primary = screen.getByTestId("agent-picker-primary");
    await user.click(within(primary).getByRole("menuitem", { name: /Modelgpt-5\.5/i }));
    const secondary = screen.getByRole("menu", { name: "model options" });

    expect(within(secondary).getAllByRole("menuitemradio")).toHaveLength(12);
    expect(secondary.parentElement).toBe(menu);
    expect(primary.nextElementSibling).toBe(secondary);
  });

  it("V205 consumes current core model capabilities instead of stale build output", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      models: [
        {
          id: "gpt-5.5",
          label: "gpt-5.5",
          runtimeModel: "gpt-5.5",
          apiProtocols: ["openai_chat"],
        },
      ],
      modelSupplies: [],
      providers: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const effortRow = screen.getByRole("menuitem", { name: /EffortMedium/i });
    expect(effortRow.hasAttribute("disabled")).toBe(false);
    await user.click(effortRow);
    expect(screen.getByRole("menuitemradio", { name: "Extra High" })).toBeTruthy();
  });

  it("flips only the secondary panel at the viewport edge and supports menu keyboard exit", async () => {
    const originalInnerWidth = window.innerWidth;
    const originalGetBoundingClientRect = HTMLElement.prototype.getBoundingClientRect;
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 800 });
    const rectSpy = vi
      .spyOn(HTMLElement.prototype, "getBoundingClientRect")
      .mockImplementation(function () {
        if (this.classList.contains("agent-picker")) {
          return {
            x: 650,
            y: 0,
            width: 100,
            height: 32,
            top: 0,
            right: 750,
            bottom: 32,
            left: 650,
            toJSON: () => ({}),
          } as DOMRect;
        }
        return originalGetBoundingClientRect.call(this);
      });

    try {
      const api = createDesktopApiMock();
      await renderApp(api);
      const user = userEvent.setup();
      const trigger = await screen.findByRole("button", { name: "Choose agent" });

      await user.click(trigger);
      const menu = screen.getByRole("menu", { name: "Agent composition" });
      await waitFor(() => expect(menu.getAttribute("data-secondary-side")).toBe("left"));
      expect(menu.style.getPropertyValue("--agent-picker-inline-offset")).toBe("-58px");

      const harnessRow = within(screen.getByTestId("agent-picker-primary")).getByRole("menuitem", {
        name: /^Harness/,
      });
      harnessRow.focus();
      await user.keyboard("{ArrowRight}");
      await waitFor(() =>
        expect(document.activeElement?.closest(".agent-picker__secondary")).not.toBeNull(),
      );
      await user.keyboard("{ArrowLeft}");
      expect(document.activeElement).toBe(harnessRow);
      await user.keyboard("{Escape}");
      expect(screen.queryByRole("menu", { name: "Agent composition" })).toBeNull();
      expect(document.activeElement).toBe(trigger);
    } finally {
      rectSpy.mockRestore();
      Object.defineProperty(window, "innerWidth", {
        configurable: true,
        value: originalInnerWidth,
      });
    }
  });

  it("V206 closes from trigger or menu focus and always restores trigger focus", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();
    const trigger = await screen.findByRole("button", { name: "Choose agent" });

    await user.click(trigger);
    expect(screen.getByRole("menu", { name: "Agent composition" })).toBeTruthy();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Agent composition" })).toBeNull();
    expect(document.activeElement).toBe(trigger);

    await user.keyboard("{ArrowDown}");
    const primary = await screen.findByTestId("agent-picker-primary");
    const harnessRow = within(primary).getByRole("menuitem", { name: /^Harness/ });
    await waitFor(() => expect(document.activeElement).toBe(harnessRow));
    await user.keyboard("{ArrowRight}");
    await waitFor(() =>
      expect(document.activeElement?.closest(".agent-picker__secondary")).not.toBeNull(),
    );
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu", { name: "Agent composition" })).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it("uses a clean guided empty state and keeps workspace actions in the sidebar", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();
    const runtime = document.querySelector<HTMLElement>("main.runtime");
    expect(runtime).not.toBeNull();

    expect(within(runtime as HTMLElement).queryByRole("heading", { name: "SwarmX" })).toBeNull();
    expect(screen.getByRole("heading", { name: /What should we build in swarmx\?/i })).toBeTruthy();
    expect(screen.getByRole("button", { name: "New task" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Setup" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Open Doctor/ })).toBeNull();
    expect(screen.getByRole("button", { name: "Workflow" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Extensions" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    expect(screen.getByRole("button", { name: "Extensions" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Custom Agents" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Runtime" })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Back to app" }));

    await user.click(screen.getByRole("button", { name: "Explore and understand code" }));
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toMatch(
      /Explore this codebase/,
    );
    await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("textbox")));
    expect(api.createSession).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Search sessions" }));
    fireEvent.change(screen.getByRole("searchbox", { name: "Search sessions" }), {
      target: { value: "ACP" },
    });
    expect(screen.getByText("ACP investigation")).toBeTruthy();
    expect(screen.queryByText("Existing local run")).toBeNull();
  });

  it("adds an existing project and creates the next task inside its working directory", async () => {
    const project: ProjectData = {
      id: "project-demo",
      name: "demo",
      cwd: "/Users/tcztzy/demo",
      pinned: false,
      createdAt: "2026-07-15T08:00:00.000Z",
      updatedAt: "2026-07-15T08:00:00.000Z",
    };
    const api = createDesktopApiMock({
      addExistingProject: vi.fn(async () => project),
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Add project" }));
    await user.click(screen.getByRole("menuitem", { name: "Use an existing folder" }));

    expect(api.addExistingProject).toHaveBeenCalledTimes(1);
    expect(await screen.findByLabelText("demo")).toBeTruthy();
    expect(screen.getByRole("heading", { name: /What should we build in demo\?/i })).toBeTruthy();

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Build the demo" } });
    await user.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => {
      expect(api.createSession).toHaveBeenCalledWith(
        expect.objectContaining({
          projectId: "project-demo",
          cwd: "/Users/tcztzy/demo",
        }),
      );
      expect(api.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ cwd: "/Users/tcztzy/demo" }),
      );
    });
  });

  it("V331 renders persisted Projects on the first frame without a loading transition", async () => {
    const pendingProjects = new Promise<ProjectData[]>(() => undefined);
    const api = createDesktopApiMock({
      initialProjects: [swarmxProject],
      listProjects: vi.fn(() => pendingProjects),
    });

    await renderApp(api);

    expect(screen.getByRole("button", { name: "swarmx" })).toBeTruthy();
    expect(screen.queryByText("Loading projects")).toBeNull();
    expect(api.listProjects).not.toHaveBeenCalled();
  });

  it("V329 matches the per-project hover row, controls, and semantic detail card", async () => {
    const api = createDesktopApiMock({
      setProjectPinned: vi.fn(async (_id: string, pinned: boolean) => ({
        ...swarmxProject,
        pinned,
      })),
    });
    await renderApp(api);
    const user = userEvent.setup();
    const projectTrigger = await screen.findByRole("button", { name: "swarmx" });
    const projectRow = projectTrigger.closest(".project-group__header-row");
    expect(projectRow).toBeTruthy();
    expect(projectTrigger.querySelector(".lucide-chevron-down")).toBeNull();
    expect(screen.getByRole("button", { name: "Options for swarmx" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "New task in swarmx" })).toBeTruthy();

    fireEvent.pointerEnter(projectRow as Element);
    const preview = await screen.findByRole("dialog", { name: "swarmx project details" });
    expect(preview.tagName).toBe("DIALOG");
    expect(preview.hasAttribute("role")).toBe(false);
    expect(within(preview).getByText("swarmx")).toBeTruthy();
    expect(within(preview).getByText("1 thread")).toBeTruthy();
    expect(within(preview).getByText("~/swarmx")).toBeTruthy();

    await user.click(within(preview).getByRole("button", { name: "Pin swarmx" }));
    expect(api.setProjectPinned).toHaveBeenCalledWith("project-swarmx", true);

    fireEvent.pointerLeave(projectRow as Element);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("uses the Projects overflow for organization and sorting", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Project options" }));
    let menu = screen.getByRole("menu", { name: "Organize projects" });
    expect(within(menu).getByText("Organize")).toBeTruthy();
    expect(within(menu).getByText("Sort by")).toBeTruthy();
    expect(
      within(menu)
        .getAllByRole("menuitemradio")
        .map((item) => item.textContent),
    ).toEqual(["By project", "In one list", "Priority", "Last updated", "Manual order"]);
    expect(
      within(menu).getByRole("menuitemradio", { name: "By project" }).getAttribute("aria-checked"),
    ).toBe("true");
    expect(
      within(menu).getByRole("menuitemradio", { name: "Priority" }).getAttribute("aria-checked"),
    ).toBe("true");

    await user.click(within(menu).getByRole("menuitemradio", { name: "In one list" }));
    expect(screen.queryByRole("button", { name: "Options for swarmx" })).toBeNull();
    expect(screen.getByText("Existing local run")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Project options" }));
    menu = screen.getByRole("menu", { name: "Organize projects" });
    await user.click(within(menu).getByRole("menuitemradio", { name: "Last updated" }));
    await user.click(screen.getByRole("button", { name: "Project options" }));
    menu = screen.getByRole("menu", { name: "Organize projects" });
    expect(
      within(menu).getByRole("menuitemradio", { name: "In one list" }).getAttribute("aria-checked"),
    ).toBe("true");
    expect(
      within(menu)
        .getByRole("menuitemradio", { name: "Last updated" })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  it("V330 matches each project's overflow menu and wires all five actions", async () => {
    const api = createDesktopApiMock({
      setProjectPinned: vi.fn(async (_id: string, pinned: boolean) => ({
        ...swarmxProject,
        pinned,
      })),
      renameProject: vi.fn(async (_id: string, name: string) => ({
        ...swarmxProject,
        name,
      })),
      revealProject: vi.fn(async () => true),
      archiveProjectTasks: vi.fn(async () => 1),
      removeProject: vi.fn(async () => true),
    });
    await renderApp(api);
    const user = userEvent.setup();
    const openProjectMenu = async () => {
      await user.click(await screen.findByRole("button", { name: /^Options for / }));
      return screen.getByRole("menu", { name: /^Project actions for / });
    };

    let menu = await openProjectMenu();
    expect(
      within(menu)
        .getAllByRole("menuitem")
        .map((item) => item.textContent),
    ).toEqual(["Pin project", "Reveal in Finder", "Rename project", "Archive tasks", "Remove"]);
    await user.click(within(menu).getByRole("menuitem", { name: "Pin project" }));
    expect(api.setProjectPinned).toHaveBeenCalledWith("project-swarmx", true);

    menu = await openProjectMenu();
    await user.click(within(menu).getByRole("menuitem", { name: "Reveal in Finder" }));
    expect(api.revealProject).toHaveBeenCalledWith("project-swarmx");

    menu = await openProjectMenu();
    await user.click(within(menu).getByRole("menuitem", { name: "Rename project" }));
    const renameInput = screen.getByRole("textbox", { name: "Rename swarmx" });
    fireEvent.change(renameInput, { target: { value: "Renamed project" } });
    fireEvent.submit(renameInput.closest("form") as HTMLFormElement);
    await waitFor(() =>
      expect(api.renameProject).toHaveBeenCalledWith("project-swarmx", "Renamed project"),
    );

    menu = await openProjectMenu();
    await user.click(within(menu).getByRole("menuitem", { name: "Archive tasks" }));
    expect(api.archiveProjectTasks).toHaveBeenCalledWith("project-swarmx");

    menu = await openProjectMenu();
    await user.click(within(menu).getByRole("menuitem", { name: "Remove" }));
    expect(api.removeProject).toHaveBeenCalledWith("project-swarmx");
    await waitFor(() => expect(screen.queryByLabelText("Renamed project")).toBeNull());
  });

  it("keeps only the first two suggested tasks while the right panel is open", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();

    expect(screen.getByRole("button", { name: "Explore and understand code" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Build a new feature, app, or tool" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Review code and suggest changes" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Fix issues and failures" })).toBeTruthy();
    expect(
      screen
        .getByLabelText("Suggested tasks")
        .classList.contains("empty-run__suggestions--right-panel"),
    ).toBe(false);

    await user.click(screen.getByRole("button", { name: "Show right panel" }));

    expect(screen.getByRole("button", { name: "Explore and understand code" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Build a new feature, app, or tool" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Review code and suggest changes" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fix issues and failures" })).toBeNull();
    expect(
      screen
        .getByLabelText("Suggested tasks")
        .classList.contains("empty-run__suggestions--right-panel"),
    ).toBe(true);

    await user.click(screen.getByRole("button", { name: "Hide right panel" }));

    expect(screen.getByRole("button", { name: "Review code and suggest changes" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Fix issues and failures" })).toBeTruthy();
    expect(
      screen
        .getByLabelText("Suggested tasks")
        .classList.contains("empty-run__suggestions--right-panel"),
    ).toBe(false);
  });

  it("provides working sidebar, back, forward, pinned summary, bottom, and right toggles", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();
    const titlebar = screen.getByRole("banner", { name: "Window title bar" });
    const navigation = screen.getByLabelText("Window navigation");
    const back = screen.getByRole("button", { name: "Go back" });
    const forward = screen.getByRole("button", { name: "Go forward" });
    expect(within(navigation).getByRole("button", { name: "Collapse sidebar" })).toBeTruthy();
    expect(within(navigation).getByRole("button", { name: "Go back" })).toBe(back);
    expect(within(navigation).getByRole("button", { name: "Go forward" })).toBe(forward);
    expect(within(titlebar).getByRole("button", { name: "Show pinned summary" })).toBeTruthy();
    expect(within(titlebar).getByRole("button", { name: "Show bottom panel" })).toBeTruthy();
    expect(within(titlebar).getByRole("button", { name: "Show right panel" })).toBeTruthy();
    expect(back.hasAttribute("disabled")).toBe(true);
    expect(forward.hasAttribute("disabled")).toBe(true);

    await user.click(await screen.findByRole("button", { name: /Existing local run/ }));
    await screen.findByText("Summarize local state");
    await user.click(screen.getByRole("button", { name: /ACP investigation/ }));
    await screen.findByText("Previous ACP answer");

    expect(back.hasAttribute("disabled")).toBe(false);
    await user.click(back);
    await screen.findByText("Summarize local state");
    expect(forward.hasAttribute("disabled")).toBe(false);
    await user.click(forward);
    await screen.findByText("Previous ACP answer");

    await user.click(screen.getByRole("button", { name: "Show pinned summary" }));
    expect(screen.getByLabelText("Pinned summary")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Show bottom panel" }));
    expect(screen.getByLabelText("Bottom panel")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Show right panel" }));
    const rightPanel = screen.getByLabelText("Right panel");
    expect(rightPanel).toBeTruthy();
    expect(document.querySelector("main.runtime")?.classList.contains("runtime--right-panel")).toBe(
      true,
    );
    const workspaceTools = within(rightPanel).getByRole("navigation", {
      name: "Open workspace tool",
    });
    expect(within(workspaceTools).getByRole("button", { name: /Review/ })).toBeTruthy();
    expect(within(workspaceTools).getByRole("button", { name: /Terminal/ })).toBeTruthy();
    expect(within(workspaceTools).getByRole("button", { name: /Browser/ })).toBeTruthy();
    expect(within(workspaceTools).getByRole("button", { name: /Files/ })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    expect(screen.getByRole("button", { name: "Open sidebar" })).toBeTruthy();
  });

  it("V221 keeps collapsed macOS navigation clear of traffic lights", async () => {
    vi.spyOn(window.navigator, "userAgent", "get").mockReturnValue(
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    );
    await renderApp(createDesktopApiMock());
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));

    const titlebar = screen.getByRole("banner", { name: "Window title bar" });
    expect(titlebar.classList.contains("app-titlebar--macos")).toBe(true);
    expect(within(titlebar).getByRole("button", { name: "Open sidebar" })).toBeTruthy();
  });

  it("animates all four toggle-controlled panels in and out", async () => {
    await renderApp(createDesktopApiMock());
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    expect(document.querySelector(".app-shell")?.classList.contains("app-shell--collapsed")).toBe(
      true,
    );
    await user.click(screen.getByRole("button", { name: "Open sidebar" }));
    expect(document.querySelector(".app-shell")?.classList.contains("app-shell--collapsed")).toBe(
      false,
    );

    await user.click(screen.getByRole("button", { name: "Show pinned summary" }));
    expect(document.querySelector(".panel-transition--pinned")?.classList.contains("is-open")).toBe(
      true,
    );
    await user.click(screen.getByRole("button", { name: "Hide pinned summary" }));
    expect(document.querySelector(".panel-transition--pinned")?.classList.contains("is-open")).toBe(
      false,
    );

    await user.click(screen.getByRole("button", { name: "Show bottom panel" }));
    expect(document.querySelector(".panel-transition--bottom")?.classList.contains("is-open")).toBe(
      true,
    );
    await user.click(screen.getByRole("button", { name: "Hide bottom panel" }));
    expect(document.querySelector(".panel-transition--bottom")?.classList.contains("is-open")).toBe(
      false,
    );

    await user.click(screen.getByRole("button", { name: "Show right panel" }));
    expect(document.querySelector(".panel-transition--right")?.classList.contains("is-open")).toBe(
      true,
    );
    await user.click(screen.getByRole("button", { name: "Hide right panel" }));
    expect(document.querySelector(".panel-transition--right")?.classList.contains("is-open")).toBe(
      false,
    );
  });

  it("opens a persistent internal terminal below the conversation composer", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Show bottom panel" }));
    const panel = screen.getByLabelText("Bottom panel");
    const composer = document.querySelector<HTMLElement>(".composer-dock");
    if (!composer) throw new Error("Composer dock is missing");
    expect(composer.compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(panel).getByLabelText("Internal terminal")).toBeTruthy();
    expect(within(panel).getByRole("tab", { name: /swarmx/i })).toBeTruthy();
    await waitFor(() =>
      expect(api.createTerminal).toHaveBeenCalledWith(
        expect.objectContaining({ cwd: "/Users/tcztzy/swarmx", cols: 80, rows: 24 }),
      ),
    );

    await user.click(screen.getByRole("button", { name: "Hide bottom panel" }));
    expect(api.killTerminal).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Show bottom panel" }));
    expect(api.createTerminal).toHaveBeenCalledTimes(1);

    await user.click(within(panel).getByRole("button", { name: "New terminal" }));
    await waitFor(() => expect(api.createTerminal).toHaveBeenCalledTimes(2));
    expect(api.killTerminal).toHaveBeenCalledTimes(1);

    await user.click(within(panel).getByRole("button", { name: "Close bottom panel" }));
    expect(screen.getByRole("button", { name: "Show bottom panel" })).toBeTruthy();
    expect(api.killTerminal).toHaveBeenCalledTimes(1);
  });

  it("groups routed Models by Provider and optional Provider group", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["anthropic", "openai_responses", "openai_chat"],
        },
        {
          id: "codex",
          label: "Codex",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
      ],
      models: [
        {
          id: "gpt-5",
          label: "gpt-5",
          runtimeModel: "gpt-5",
          apiProtocols: ["openai_chat"],
        },
        {
          id: "gpt-5.5",
          label: "gpt-5.5",
          runtimeModel: "gpt-5.5",
          apiProtocols: ["openai_chat"],
        },
        {
          id: "claude-opus-4-6",
          label: "Claude Opus 4.6",
          runtimeModel: "claude-opus-4-6",
          apiProtocols: ["anthropic"],
        },
      ],
      modelSupplies: [
        {
          id: "openai:gpt-5",
          modelId: "gpt-5",
          providerProfileId: "openai",
          apiCompatibility: { mode: "native", targetApi: "openai_chat" },
        },
        {
          id: "openai:gpt-5.5",
          modelId: "gpt-5.5",
          providerProfileId: "openai",
          apiCompatibility: { mode: "native", targetApi: "openai_chat" },
        },
        {
          id: "anthropic:claude-opus-4-6:research",
          modelId: "claude-opus-4-6",
          providerProfileId: "anthropic",
          providerGroup: "Research",
          apiCompatibility: { mode: "native", targetApi: "anthropic" },
        },
        {
          id: "packy:claude-opus-4-6:premium",
          modelId: "claude-opus-4-6",
          providerProfileId: "packy",
          providerGroup: "Premium",
          apiCompatibility: { mode: "native", targetApi: "anthropic" },
        },
      ],
      providers: [
        {
          id: "openai",
          label: "OpenAI",
          kind: "openai_chat",
          apiEntrypoints: {},
        },
        {
          id: "anthropic",
          label: "Anthropic",
          kind: "anthropic",
          apiEntrypoints: {},
        },
        {
          id: "packy",
          label: "Packy",
          kind: "anthropic",
          baseUrl: "https://www.packyapi.com",
          apiEntrypoints: {},
          usageAdapter: "new_api",
        },
      ],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const primary = screen.getByTestId("agent-picker-primary");
    await user.click(
      within(primary).getByRole("menuitem", {
        name: /Modelgpt-5/,
      }),
    );

    const openAiGroup = screen.getByRole("group", { name: "OpenAI" });
    const anthropicGroup = screen.getByRole("group", { name: "Anthropic" });
    const packyGroup = screen.getByRole("group", { name: "Packy" });
    expect(
      within(openAiGroup)
        .getAllByRole("menuitemradio")
        .map((item) => item.textContent),
    ).toEqual(["gpt-5.5", "gpt-5"]);
    expect(within(anthropicGroup).getAllByRole("menuitemradio")).toHaveLength(1);
    expect(within(packyGroup).getAllByRole("menuitemradio")).toHaveLength(1);
    expect(screen.getByRole("group", { name: "Research" })).toBeTruthy();

    fireEvent.change(screen.getByRole("searchbox", { name: "Search models" }), {
      target: { value: "claude" },
    });
    expect(screen.queryByRole("group", { name: "OpenAI" })).toBeNull();
    expect(screen.getByRole("group", { name: "Anthropic" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Packy" })).toBeTruthy();
    await user.click(
      within(screen.getByRole("group", { name: "Anthropic" })).getByRole("menuitemradio", {
        name: "Claude Opus 4.6",
      }),
    );
    const trigger = screen.getByRole("button", { name: "Choose agent" });
    expect(trigger.getAttribute("data-harness-id")).toBe("swarmx");
    expect(within(trigger).getByText("Opus 4.6")).toBeTruthy();
    expect(trigger.querySelector('[data-model-brand="claude"]')?.getAttribute("src")).toBe(
      "./harness-icons/claude_code.svg",
    );
  });

  it("uses a stable fallback when a packaged harness icon fails", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_chat"],
        },
        {
          id: "codex",
          label: "Codex",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
      ],
    });
    await renderApp(api);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    const icon = document.querySelector<HTMLImageElement>('[data-harness-icon="codex"]');
    expect(icon?.getAttribute("src")).toMatch(/^data:image\/svg\+xml/);
    if (icon) fireEvent.error(icon);
    expect(document.querySelector('[data-harness-icon-fallback="codex"]')).not.toBeNull();
  });

  it("resets a failed packaged icon when the selected harness changes", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_chat"],
        },
        {
          id: "codex",
          label: "Codex",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
        {
          id: "opencode",
          label: "OpenCode",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: [],
        },
      ],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    await user.click(screen.getByRole("menuitem", { name: /^Harness/ }));
    await user.click(screen.getByRole("menuitemradio", { name: "Codex" }));

    const codexIcon = await waitFor(() => {
      const icon = document.querySelector<HTMLImageElement>('[data-harness-icon="codex"]');
      expect(icon).not.toBeNull();
      return icon;
    });
    if (codexIcon) fireEvent.error(codexIcon);
    expect(document.querySelector('[data-harness-icon-fallback="codex"]')).not.toBeNull();

    await user.click(screen.getByRole("menuitemradio", { name: "OpenCode" }));

    await waitFor(() => {
      const icon = document.querySelector<HTMLImageElement>('[data-harness-icon="opencode"]');
      expect(icon?.getAttribute("src")).toMatch(/^data:image\/svg\+xml/);
    });
  });

  it("derives effort from the selected Harness and Model", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      models: [
        {
          id: "gpt-5",
          label: "gpt-5",
          runtimeModel: "gpt-5",
          apiProtocols: ["openai_chat"],
        },
      ],
      modelSupplies: [],
      providers: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    const trigger = await screen.findByRole("button", { name: "Choose agent" });
    await waitFor(() => {
      expect(within(trigger).getByText("5")).toBeTruthy();
      expect(trigger.querySelector('[data-model-brand="gpt"]')?.getAttribute("src")).toBe(
        "./harness-icons/codex.svg",
      );
      expect(
        within(trigger).getByText("Medium").classList.contains("agent-picker__trigger-effort"),
      ).toBe(true);
    });

    await user.click(trigger);
    const primary = screen.getByTestId("agent-picker-primary");
    const leftBefore = primary.getBoundingClientRect().left;
    const effortRow = screen.getByRole("menuitem", { name: /EffortMedium/i });
    expect(effortRow.hasAttribute("disabled")).toBe(false);
    await user.click(effortRow);
    expect(screen.getByRole("menu", { name: "effort options" })).toBeTruthy();
    expect(primary.getBoundingClientRect().left).toBe(leftBefore);
    await user.click(screen.getByRole("menuitemradio", { name: "High" }));
    await waitFor(() =>
      expect(
        within(trigger).getByText("High").classList.contains("agent-picker__trigger-effort"),
      ).toBe(true),
    );
    expect(trigger.textContent).not.toContain("·");

    await user.type(screen.getByRole("textbox"), "Use verified high effort");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(api.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        agentComposition: expect.objectContaining({
          harnessId: "swarmx",
          modelId: "gpt-5",
          effort: "high",
        }),
      }),
    );
  });

  it("resets effort to the official default when the selected model changes", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      models: [
        {
          id: "gpt-5",
          label: "gpt-5",
          runtimeModel: "gpt-5",
          apiProtocols: ["openai_chat"],
        },
        {
          id: "deepseek-v4-pro",
          label: "deepseek-v4-pro",
          runtimeModel: "deepseek-v4-pro",
          apiProtocols: ["openai_chat"],
        },
      ],
      modelSupplies: [],
      providers: [],
    });
    await renderApp(api);
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: "Choose agent" }));
    await user.click(screen.getByRole("menuitem", { name: /EffortMedium/i }));
    await user.click(screen.getByRole("menuitemradio", { name: "Minimal" }));
    await user.click(screen.getByRole("menuitem", { name: /Modelgpt-5/ }));
    await user.click(screen.getByRole("menuitemradio", { name: /deepseek-v4-pro/ }));

    const trigger = screen.getByRole("button", { name: "Choose agent" });
    await waitFor(() => {
      expect(within(trigger).getByText("V4 Pro")).toBeTruthy();
      expect(trigger.querySelector('[data-model-brand="deepseek"]')?.getAttribute("src")).toBe(
        "./provider-icons/deepseek.svg",
      );
    });
    await waitFor(() => expect(screen.getByRole("menuitem", { name: /EffortHigh/i })).toBeTruthy());
    await user.click(screen.getByRole("menuitem", { name: /EffortHigh/i }));
    expect(screen.queryByRole("menuitemradio", { name: "Minimal" })).toBeNull();
    expect(screen.getByRole("menuitemradio", { name: "High" })).toBeTruthy();
    expect(screen.getByRole("menuitemradio", { name: "Max" })).toBeTruthy();

    await user.type(screen.getByRole("textbox"), "Use the corrected effort");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(api.sendMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        agentComposition: expect.objectContaining({
          modelId: "deepseek-v4-pro",
          effort: "high",
        }),
      }),
    );
  });

  it("renders host-registered GUI contribution components from extension metadata", async () => {
    const api = createDesktopApiMock();
    const user = userEvent.setup();

    await renderApp(api, {
      product: {
        name: "GEEPilot",
        subtitle: "analysis cockpit",
      },
      uiComponentRegistry: {
        "geepilot.ui.shell": ({ contribution, inventory, onSelectAgent }) => (
          <section aria-label="GEEPilot registered shell">
            <h2>Registered component {contribution.id}</h2>
            <p>agents {inventory?.agents.length ?? 0}</p>
            <button type="button" onClick={() => onSelectAgent("analysis-lead")}>
              Use analysis lead from custom UI
            </button>
          </section>
        ),
      },
    });

    expect(await screen.findByText("analysis cockpit")).toBeTruthy();
    expect(await screen.findByRole("button", { name: "Open GEEPilot navigation" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Open Analysis dashboard" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Open GEEPilot navigation" }));

    expect(screen.getByLabelText("GEEPilot navigation contribution")).toBeTruthy();
    expect(screen.getByLabelText("GEEPilot registered shell")).toBeTruthy();
    expect(screen.getByText("Registered component geepilot.nav")).toBeTruthy();
    expect(screen.getByText("agents 2")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Use analysis lead from custom UI" }));

    expect(await screen.findByPlaceholderText("Message analysis lead")).toBeTruthy();
    expect(screen.queryByLabelText("GEEPilot registered shell")).toBeNull();
  }, 40_000);

  it("shows extension marketplace metadata separately from harnesses and agent profiles", async () => {
    const api = createDesktopApiMock();
    const user = userEvent.setup();

    await renderApp(api);

    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Extensions" }));

    expect(await screen.findByLabelText("Extension inventory")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Plugin bundles" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Marketplace sources" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Plugin catalog" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Plugin components" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "GUI contributions" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Harnesses" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Agent profiles" })).toBeTruthy();
    expect(screen.getByText("SwarmX Built-ins")).toBeTruthy();
    expect(screen.getByText("GEEPilot")).toBeTruthy();
    expect(screen.getByText("Codex local marketplace")).toBeTruthy();
    expect(screen.getByText("./.agents/plugins/marketplace.json")).toBeTruthy();
    expect(screen.getByText("GEEPilot plugin")).toBeTruthy();
    expect(screen.getByText("runnable harness")).toBeTruthy();
    expect(screen.getAllByText("GEEPilot Codex").length).toBeGreaterThan(0);
    expect(screen.getByText("analysis lead")).toBeTruthy();
    expect(screen.getByText("ready")).toBeTruthy();
    expect(screen.getByText("@analysis-lead")).toBeTruthy();
    expect(screen.getAllByText("1 plugins").length).toBeGreaterThan(0);
    expect(screen.getAllByText("via geepilot").length).toBeGreaterThan(0);
    expect(screen.getAllByText("context thread_packet/auto").length).toBeGreaterThan(0);
    expect(screen.getByText("permissions mode plan / selected MCP / plan")).toBeTruthy();
    expect(screen.getByText("secret required")).toBeTruthy();
    expect(screen.getByText("blocked agent")).toBeTruthy();
    expect(screen.getByText("missing harness missing-harness")).toBeTruthy();
    expect(
      (
        screen.getByRole("button", {
          name: "Use agent profile blocked agent",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(screen.getByText("geepilot.memory")).toBeTruthy();
    expect(screen.getByText("Biosecurity")).toBeTruthy();
    expect(screen.getByText("canonical skills/biosecurity/SKILL.md")).toBeTruthy();
    expect(screen.getByText("codex plugin")).toBeTruthy();
    expect(screen.getByText("opencode rules_only")).toBeTruthy();
    expect(screen.getByText("gate geepilot.biosecurity")).toBeTruthy();
    expect(screen.getByText("manifest ./.claude-plugin/plugin.json")).toBeTruthy();
    expect(screen.getByText("project-fs")).toBeTruthy();
    expect(screen.getByText("permission plan")).toBeTruthy();
    expect(screen.getByText("memory readonly")).toBeTruthy();
    expect(screen.getByText("tool Read")).toBeTruthy();
    expect(screen.getByText("blocked Bash")).toBeTruthy();
    expect(screen.getByText("6 turns")).toBeTruthy();
    expect(screen.getByText("Refresh index")).toBeTruthy();
    expect(screen.getByText("command")).toBeTruthy();
    expect(screen.getByText("geepilot index refresh")).toBeTruthy();
    expect(screen.getByText("Pyright")).toBeTruthy();
    expect(screen.getByText("before_agent_run")).toBeTruthy();
    expect(screen.getByText("project-read")).toBeTruthy();
    expect(screen.getByText("auth policy")).toBeTruthy();
    expect(screen.getByText("GEEPilot navigation")).toBeTruthy();
    expect(screen.getByText("Analysis dashboard")).toBeTruthy();
    expect(screen.getByText("Review dataset")).toBeTruthy();
    expect(screen.getByText("/extensions/geepilot")).toBeTruthy();
    expect(screen.getByText("geepilot.ui.shell")).toBeTruthy();
    expect(screen.getByText("command geepilot.refresh-index")).toBeTruthy();
    expect(screen.getByText("setting geepilot.indexRoot")).toBeTruthy();
    expect(screen.getByText("permission project-read")).toBeTruthy();
    expect(screen.getByText("auth project-fs-auth")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Use agent profile analysis lead" }));

    expect(await screen.findByPlaceholderText("Message analysis lead")).toBeTruthy();
    expect(document.querySelector(".agent-picker__trigger")?.getAttribute("data-harness-id")).toBe(
      "geepilot-codex",
    );
    fireEvent.change(screen.getByPlaceholderText("Message analysis lead"), {
      target: { value: "Plan a GEEPilot analysis" },
    });
    await user.click(screen.getByRole("button", { name: /Send/i }));

    await waitFor(() => {
      expect(api.createSession).toHaveBeenCalledWith({
        agentName: "analysis lead",
        harness: "geepilot-codex",
        projectId: "project-swarmx",
        cwd: "/Users/tcztzy/swarmx",
      });
      expect(api.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          harnessId: "geepilot-codex",
          userText: "Plan a GEEPilot analysis",
          cwd: "/Users/tcztzy/swarmx",
          agentComposition: {
            id: "desktop-analysis-lead",
            agentProfileId: "analysis-lead",
            host: "local",
          },
        }),
      );
    });
  }, 40_000);

  it("composes a Custom Agent as Harness recipe plus Model in Settings", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.saveCustomAgent.mockResolvedValue(inventory);
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Custom Agents" }));

    expect(await screen.findByLabelText("Custom Agents settings")).toBeTruthy();
    expect(screen.getByText("Software + Skills + MCP + Context + Policy")).toBeTruthy();
    await user.type(screen.getByLabelText("Name"), "Researcher");
    await user.click(screen.getByRole("checkbox", { name: /Biosecurity/ }));
    await user.click(screen.getByRole("checkbox", { name: /Project Files/ }));
    await user.click(screen.getByRole("button", { name: "Save Agent" }));

    await waitFor(() => expect(api.saveCustomAgent).toHaveBeenCalledTimes(1));
    expect(api.saveCustomAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "researcher",
        name: "Researcher",
        harnessId: "researcher-harness",
        modelId: "gpt-5",
        harnessRecipe: expect.objectContaining({
          id: "researcher-harness",
          softwareId: "swarmx",
          skillBindings: [
            expect.objectContaining({ skillId: "geepilot.biosecurity", mode: "auto" }),
          ],
          mcpServerIds: ["project-fs"],
        }),
      }),
    );
  });

  it("groups Custom Agent Models by Provider and reuses canonical family ordering", async () => {
    const api = createDesktopApiMock();
    const base = await api.listExtensions();
    const gptModelIds = [
      "gpt-5.6-luna",
      "gpt-5.5-sol",
      "gpt-5.6-terra",
      "gpt-6-luna",
      "gpt-5.6-sol",
    ];
    const claudeModelIds = [
      "claude-haiku-6",
      "claude-sonnet-7",
      "claude-opus-4",
      "claude-fable-4",
      "claude-opus-5",
      "claude-mythos-4",
      "claude-sonnet-5",
    ];
    const routedInventory = {
      ...base,
      models: [...gptModelIds, ...claudeModelIds].map((id) => ({
        id,
        label: id,
        runtimeModel: id,
        apiProtocols: ["openai_chat"],
      })),
      providers: [
        { id: "packy", label: "Packy", kind: "openai_chat" },
        { id: "anthropic", label: "Anthropic", kind: "openai_chat" },
      ],
      modelSupplies: [
        ...gptModelIds.map((modelId) => ({
          id: `packy-${modelId}`,
          modelId,
          providerProfileId: "packy",
          runtimeModel: modelId,
        })),
        ...claudeModelIds.map((modelId) => ({
          id: `anthropic-${modelId}`,
          modelId,
          providerProfileId: "anthropic",
          runtimeModel: modelId,
        })),
      ],
    };
    api.listExtensions.mockResolvedValue(routedInventory);
    api.saveCustomAgent.mockResolvedValue(routedInventory);
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Custom Agents" }));

    const modelSelect = (await screen.findByLabelText("Model")) as HTMLSelectElement;
    const providerGroups = [...modelSelect.querySelectorAll("optgroup")];
    expect(providerGroups.map((group) => group.label)).toEqual(["Packy", "Anthropic"]);
    expect(modelSelect.value).toBe("swarmx:gpt-6-luna@packy-gpt-6-luna");
    expect(
      [...(providerGroups[0]?.querySelectorAll("option") ?? [])].map(
        (option) => option.textContent,
      ),
    ).toEqual(["gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5-sol"]);
    expect(
      [...(providerGroups[1]?.querySelectorAll("option") ?? [])].map(
        (option) => option.textContent,
      ),
    ).toEqual([
      "claude-mythos-4",
      "claude-fable-4",
      "claude-opus-5",
      "claude-opus-4",
      "claude-sonnet-7",
      "claude-sonnet-5",
      "claude-haiku-6",
    ]);

    await user.selectOptions(modelSelect, "swarmx:gpt-5.6-terra@packy-gpt-5.6-terra");
    await user.type(screen.getByLabelText("Name"), "Routed researcher");
    await user.click(screen.getByRole("button", { name: "Save Agent" }));

    await waitFor(() => expect(api.saveCustomAgent).toHaveBeenCalledTimes(1));
    expect(api.saveCustomAgent).toHaveBeenCalledWith(
      expect.objectContaining({
        modelId: "gpt-5.6-terra",
        modelSupplyId: "packy-gpt-5.6-terra",
      }),
    );
  });

  it("separates discovered native Agent definitions from persisted and Extension profiles", async () => {
    const api = createDesktopApiMock();
    const inventory = await api.listExtensions();
    api.listExtensions.mockResolvedValue({
      ...inventory,
      agents: [
        ...inventory.agents,
        {
          id: "native:codex:reviewer",
          name: "reviewer",
          harnessId: "codex",
          nativeModel: "inherit",
          readOnly: true,
          definition: {
            kind: "project",
            host: "codex",
            format: "codex",
            path: "/workspace/.codex/agents/reviewer.toml",
            readOnly: true,
          },
        },
        {
          id: "native:claude_code:researcher",
          name: "researcher",
          harnessId: "claude_code",
          modelId: "claude-sonnet-5",
          nativeModel: "claude-sonnet-5",
          readOnly: true,
          definition: {
            kind: "user",
            host: "claude_code",
            format: "claude_code",
            path: "/home/.claude/agents/researcher.md",
            readOnly: true,
          },
        },
      ],
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Custom Agents" }));

    expect(await screen.findByText("Native definitions · read-only")).toBeTruthy();
    expect(screen.getByText(/Codex · reviewer · inherit/)).toBeTruthy();
    expect(screen.getByText(/Claude Code · researcher · claude-sonnet-5/)).toBeTruthy();
    expect(screen.getByText("Extension profiles · read-only")).toBeTruthy();
  });

  it("keeps Node and Harness tools in Runtime with Doctor built in", async () => {
    const environment = readyHarnessEnvironment();
    const api = createDesktopApiMock({ getHarnessEnvironment: vi.fn(async () => environment) });
    const user = userEvent.setup();

    await renderApp(api);
    expect(screen.queryByRole("button", { name: /Open Doctor/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Open anonymous user menu" }));
    await user.click(screen.getByRole("menuitem", { name: "Settings" }));
    await user.click(screen.getByRole("button", { name: "Runtime" }));

    expect(await screen.findByLabelText("Runtime settings")).toBeTruthy();
    expect(screen.getByText("Node.js runtime")).toBeTruthy();
    expect(screen.queryByText("Bun runtime")).toBeNull();
    expect(screen.getByText("Apple Container")).toBeTruthy();
    expect(screen.getByText("/opt/homebrew/bin/node")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Harness tools" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Environment Doctor" })).toBeTruthy();
    expect(screen.getByText("Claude Code")).toBeTruthy();
    expect(screen.getByText("Codex")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Open Doctor" })).toBeNull();
    expect(api.inspectDoctor).toHaveBeenCalledWith();

    await user.click(screen.getByRole("button", { name: "Check Codex version again" }));
    await waitFor(() => {
      expect(api.getHarnessVersion).toHaveBeenLastCalledWith({
        harnessId: "codex",
        refresh: true,
      });
    });
  });

  it("renders harnesses while versions load and refreshes only the clicked version", async () => {
    const environment = readyHarnessEnvironment();
    const reportGate = deferred<ReturnType<typeof doctorReport>>();
    const versionGate = deferred<void>();
    const api = createDesktopApiMock({
      inspectDoctor: vi.fn(() => reportGate.promise),
      getHarnessVersion: vi.fn(async ({ harnessId }: { harnessId: string; refresh?: boolean }) => {
        await versionGate.promise;
        return {
          harnessId,
          version: environment.harnesses.find((harness) => harness.harnessId === harnessId)
            ?.version,
        };
      }),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.type(screen.getByRole("textbox"), "/doctor{Enter}");

    expect(await screen.findByRole("heading", { name: "Harnesses" })).toBeTruthy();
    expect(document.querySelectorAll(".doctor-harness")).toHaveLength(6);
    expect(screen.getByLabelText("Checking Codex version")).toBeTruthy();
    expect(screen.getByText("Checking environment")).toBeTruthy();

    versionGate.resolve();
    reportGate.resolve(doctorReport(environment));
    const codexVersion = await screen.findByRole("button", {
      name: "Check Codex version again",
    });
    expect(codexVersion.textContent).toBe("0.69.0");
    expect(api.getHarnessVersion).toHaveBeenCalledTimes(6);

    await user.click(screen.getByRole("button", { name: "Close Doctor" }));
    await user.type(screen.getByRole("textbox"), "/doctor{Enter}");
    await screen.findByRole("button", { name: "Check Codex version again" });
    expect(api.getHarnessVersion).toHaveBeenCalledTimes(6);

    await user.click(screen.getByRole("button", { name: "Check Codex version again" }));
    await waitFor(() => expect(api.getHarnessVersion).toHaveBeenCalledTimes(7));
    expect(api.getHarnessVersion).toHaveBeenLastCalledWith({
      harnessId: "codex",
      refresh: true,
    });
  });

  it("shows unavailable optional harnesses without global repairs", async () => {
    const before = doctorReport(missingHarnessEnvironment());
    const legacyBefore = {
      ...before,
      healthy: false,
      issues: [
        {
          id: "requirement:openclaw",
          severity: "error" as const,
          scope: "requirement" as const,
          targetId: "openclaw",
          message: "OpenClaw is missing.",
          repairActionId: "harness:openclaw",
        },
      ],
      repairActions: [
        {
          id: "harness:openclaw",
          label: "Set up OpenClaw",
          risk: "install" as const,
          request: { harnessId: "openclaw" },
        },
      ],
    };
    const after = doctorReport(readyHarnessEnvironment());
    const api = createDesktopApiMock({
      getHarnessEnvironment: vi.fn(async () => missingHarnessEnvironment()),
      getHarnessVersion: vi.fn(async ({ harnessId }: { harnessId: string }) => ({
        harnessId,
        version: missingHarnessEnvironment().harnesses.find(
          (harness) => harness.harnessId === harnessId,
        )?.version,
      })),
      inspectDoctor: vi.fn().mockResolvedValueOnce(legacyBefore).mockResolvedValue(after),
    });
    const user = userEvent.setup();

    await renderApp(api);
    expect(screen.queryByRole("button", { name: "Setup" })).toBeNull();
    await user.type(screen.getByRole("textbox"), "/doctor{Enter}");

    expect(await screen.findByLabelText("Doctor panel")).toBeTruthy();
    expect(screen.getByRole("textbox")).toBeTruthy();
    expect(screen.getByText("Environment ready")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Diagnostics" })).toBeNull();
    expect(screen.getByText("0.69.0").classList.contains("badge--active")).toBe(true);
    expect(screen.queryByText("Built in.")).toBeNull();
    expect(document.querySelectorAll(".doctor-harness__icon")).toHaveLength(6);
    expect(document.querySelector(".doctor-harness [data-harness-icon='codex']")).not.toBeNull();
    expect(
      screen.queryByText("Apple Container must be installed and its system service started."),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: "Fix issues" })).toBeNull();
    expect(api.fixDoctor).not.toHaveBeenCalled();

    const installOpenClaw = screen.getByRole("button", { name: "Install OpenClaw" });
    expect(installOpenClaw.textContent).toContain("Install");
    await user.click(installOpenClaw);
    await waitFor(() =>
      expect(api.setupHarnessEnvironment).toHaveBeenCalledWith({ harnessToolId: "openclaw" }),
    );
    await waitFor(() => {
      const row = screen
        .getByText("OpenClaw", { selector: ".doctor-harness strong" })
        .closest("li");
      expect(row).not.toBeNull();
      expect(within(row as HTMLElement).getByText("2026.6.11")).toBeTruthy();
    });
  });

  it("routes /doctor and /setup through the same panel without requiring a model", async () => {
    const before = doctorReport(missingHarnessEnvironment(), "codex");
    const api = createDesktopApiMock({
      inspectDoctor: vi.fn(async () => before),
    });
    const user = userEvent.setup();
    await renderApp(api);

    const composer = screen.getByRole("textbox");
    await user.type(composer, "/doctor --fix --harness codex{Enter}");

    expect(await screen.findByLabelText("Doctor panel")).toBeTruthy();
    expect(screen.getByText("Environment ready")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Diagnostics" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Fix issues" })).toBeNull();
    expect(api.inspectDoctor).toHaveBeenCalledWith({ harnessId: "codex" });
    expect(api.fixDoctor).not.toHaveBeenCalled();
    expect(api.createSession).not.toHaveBeenCalled();
    expect(api.sendMessage).not.toHaveBeenCalled();

    await user.clear(composer);
    await user.type(composer, "/setup codex{Enter}");
    expect(await screen.findByLabelText("Setup panel")).toBeTruthy();
    expect(screen.getByText("Environment ready")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Diagnostics" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Set up missing" })).toBeNull();
    expect(api.inspectDoctor).toHaveBeenLastCalledWith({ harnessId: "codex" });
    expect(api.fixDoctor).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Refresh diagnostics" }));
    await waitFor(() => expect(api.inspectDoctor).toHaveBeenCalledTimes(3));
  });

  it("renders ACP agents as harness plus model nodes and executes with the parsed swarm config", async () => {
    const api = createDesktopApiMock();
    const user = userEvent.setup();

    await renderApp(api);

    await user.click(await screen.findByRole("button", { name: /Workflow/i }));

    expect(screen.getByLabelText("Workflow canvas")).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Editor" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByRole("tab", { name: "Executions" })).toBeTruthy();
    expect(screen.getByRole("tab", { name: "JSON" })).toBeTruthy();
    expect(screen.getByLabelText("Canvas controls").textContent).toContain("100%");
    expect(screen.getByLabelText("Workflow inspector")).toBeTruthy();
    expect(screen.getByLabelText("Workflow JSON")).toBeTruthy();
    expect(screen.getByLabelText("Workflow node triage_agent codex root")).toBeTruthy();
    expect(screen.getByLabelText("Workflow node researcher_agent claude_code")).toBeTruthy();
    expect(screen.getByLabelText("Workflow node writer_agent codex")).toBeTruthy();
    expect(screen.getAllByText("Harness Codex")).toHaveLength(2);
    expect(screen.getByText("Harness Claude Code")).toBeTruthy();
    expect(screen.getAllByText("Model negotiated by harness")).toHaveLength(3);
    expect(screen.getByText("harness = software + MCPs + skills + project files")).toBeTruthy();
    expect(screen.getAllByText("Software codex-acp@1.1.2")).toHaveLength(2);
    expect(screen.getByText("Software claude-agent-acp@0.58.1")).toBeTruthy();
    expect(screen.getAllByText("MCPs filesystem")).toHaveLength(3);
    expect(screen.getAllByText("Skills test-driven-development, backprop")).toHaveLength(3);
    expect(screen.getAllByText("Project files AGENTS.md, CLAUDE.md")).toHaveLength(3);
    expect(screen.getByLabelText("Workflow edge triage_agent to researcher_agent")).toBeTruthy();
    expect(
      screen.getByLabelText("Workflow connector triage_agent to researcher_agent"),
    ).toBeTruthy();

    await user.click(screen.getByLabelText("Use workflow"));
    fireEvent.change(screen.getByPlaceholderText("Message SwarmX"), {
      target: { value: "Run the workflow" },
    });
    await user.click(screen.getByRole("button", { name: /Execute workflow/i }));

    await waitFor(() => {
      expect(api.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          harnessId: "swarmx",
          userText: "Run the workflow",
          swarmConfig: expect.objectContaining({
            name: "research_review",
            root: "triage_agent",
            nodes: expect.objectContaining({
              triage_agent: expect.objectContaining({
                kind: "agent",
                agent: expect.objectContaining({
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "npx",
                    args: expect.arrayContaining(["@agentclientprotocol/codex-acp@1.1.2"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "codex-acp",
                        version: "1.1.2",
                      }),
                      mcps: expect.arrayContaining([
                        expect.objectContaining({ name: "filesystem" }),
                      ]),
                      skills: expect.arrayContaining(["test-driven-development", "backprop"]),
                      projectFiles: expect.arrayContaining(["AGENTS.md", "CLAUDE.md"]),
                    }),
                  }),
                }),
              }),
              researcher_agent: expect.objectContaining({
                kind: "agent",
                agent: expect.objectContaining({
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "npx",
                    args: expect.arrayContaining(["@agentclientprotocol/claude-agent-acp@0.58.1"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "claude-agent-acp",
                        version: "0.58.1",
                      }),
                      mcps: expect.arrayContaining([
                        expect.objectContaining({ name: "filesystem" }),
                      ]),
                      skills: expect.arrayContaining(["test-driven-development", "backprop"]),
                      projectFiles: expect.arrayContaining(["AGENTS.md", "CLAUDE.md"]),
                    }),
                  }),
                }),
              }),
              writer_agent: expect.objectContaining({
                kind: "agent",
                agent: expect.objectContaining({
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "npx",
                    args: expect.arrayContaining(["@agentclientprotocol/codex-acp@1.1.2"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "codex-acp",
                        version: "1.1.2",
                      }),
                      mcps: expect.arrayContaining([
                        expect.objectContaining({ name: "filesystem" }),
                      ]),
                      skills: expect.arrayContaining(["test-driven-development", "backprop"]),
                      projectFiles: expect.arrayContaining(["AGENTS.md", "CLAUDE.md"]),
                    }),
                  }),
                }),
              }),
            }),
          }),
        }),
      );
    });
    const executedConfig = api.sendMessage.mock.calls.at(-1)?.[0]?.swarmConfig;
    expect(executedConfig?.nodes.triage_agent.agent).not.toHaveProperty("model");
    expect(executedConfig?.nodes.researcher_agent.agent).not.toHaveProperty("model");
    expect(executedConfig?.nodes.writer_agent.agent).not.toHaveProperty("model");
  }, 20_000);

  it("shows workflow JSON errors and does not fall back to an implicit agent", async () => {
    const api = createDesktopApiMock();
    const user = userEvent.setup();

    await renderApp(api);

    await user.click(await screen.findByRole("button", { name: /Workflow/i }));
    await user.click(screen.getByLabelText("Use workflow"));
    fireEvent.change(screen.getByLabelText("Workflow JSON"), {
      target: { value: "{{" },
    });

    expect((await screen.findByRole("alert")).textContent).toMatch(/Workflow JSON/i);

    fireEvent.change(screen.getByPlaceholderText("Message SwarmX"), {
      target: { value: "Run without config" },
    });
    await user.click(screen.getByRole("button", { name: /Execute workflow/i }));

    expect(api.sendMessage).not.toHaveBeenCalled();
  });

  it("imports n8n workflow JSON and executes the converted swarm config", async () => {
    const importedConfig = {
      name: "Imported n8n",
      root: "Manual_Trigger",
      nodes: {
        Manual_Trigger: {
          kind: "agent",
          agent: {
            name: "Manual_Trigger",
            model: "gpt-4o",
            backend: { type: "swarmx" },
            parameters: {
              harness: {
                software: { name: "n8n-import" },
                mcps: [],
                skills: [],
                projectFiles: [],
              },
              n8n: {
                name: "Manual Trigger",
                type: "n8n-nodes-base.manualTrigger",
                parameters: {},
              },
            },
          },
        },
        HTTP_Request: {
          kind: "agent",
          agent: {
            name: "HTTP_Request",
            model: "gpt-4o",
            backend: { type: "swarmx" },
            parameters: {
              harness: {
                software: { name: "n8n-import" },
                mcps: [],
                skills: [],
                projectFiles: [],
              },
              n8n: {
                name: "HTTP Request",
                type: "n8n-nodes-base.httpRequest",
                parameters: { url: "https://example.com" },
              },
            },
          },
        },
      },
      edges: [{ source: "Manual_Trigger", target: "HTTP_Request" }],
    };
    const api = createDesktopApiMock({
      getHarnessEnvironment: vi.fn(async () => missingHarnessEnvironment()),
      importN8nWorkflow: vi.fn(async () => ({
        success: true,
        config: importedConfig,
        warnings: ["Native n8n node execution is not imported."],
        nodeMap: { "Manual Trigger": "Manual_Trigger" },
      })),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /Workflow/i }));

    const file = new File(
      [
        JSON.stringify({
          name: "Imported n8n",
          nodes: [{ name: "Manual Trigger", type: "n8n-nodes-base.manualTrigger" }],
          connections: {},
        }),
      ],
      "workflow.json",
      { type: "application/json" },
    );
    fireEvent.change(screen.getByLabelText("n8n workflow JSON file"), {
      target: { files: [file] },
    });

    await waitFor(() => {
      expect(api.importN8nWorkflow).toHaveBeenCalledWith(expect.stringContaining("Imported n8n"));
    });
    expect((screen.getByLabelText("Workflow JSON") as HTMLTextAreaElement).value).toContain(
      '"name": "Imported n8n"',
    );
    expect((screen.getByLabelText("Use workflow") as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole("status").textContent).toContain(
      'Imported n8n workflow "Imported n8n".',
    );
    expect(screen.getByRole("status").textContent).toContain(
      "Native n8n node execution is not imported.",
    );
    expect(screen.getByLabelText("Workflow node Manual_Trigger swarmx gpt-4o root")).toBeTruthy();
    expect(screen.getByLabelText("Workflow edge Manual_Trigger to HTTP_Request")).toBeTruthy();

    fireEvent.change(screen.getByPlaceholderText("Message SwarmX"), {
      target: { value: "Run imported workflow" },
    });
    await user.click(screen.getByRole("button", { name: /Execute workflow/i }));

    await waitFor(() => {
      expect(api.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          harnessId: "swarmx",
          userText: "Run imported workflow",
          swarmConfig: importedConfig,
        }),
      );
    });
  });

  it("keeps the current workflow JSON when n8n import fails", async () => {
    const api = createDesktopApiMock({
      importN8nWorkflow: vi.fn(async () => ({
        success: false,
        error: "n8n workflow JSON needs a nodes array.",
      })),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /Workflow/i }));
    const originalJson = (screen.getByLabelText("Workflow JSON") as HTMLTextAreaElement).value;
    const file = new File([JSON.stringify({ name: "Broken n8n" })], "broken.json", {
      type: "application/json",
    });

    fireEvent.change(screen.getByLabelText("n8n workflow JSON file"), {
      target: { files: [file] },
    });

    expect((await screen.findByRole("alert")).textContent).toContain(
      "n8n workflow JSON needs a nodes array.",
    );
    expect((screen.getByLabelText("Workflow JSON") as HTMLTextAreaElement).value).toBe(
      originalJson,
    );
    expect((screen.getByLabelText("Use workflow") as HTMLInputElement).checked).toBe(false);
  });

  it("collapses completed work and leaves only the final summary visible", async () => {
    const completedSession: SessionData = {
      ...acpSessionDetail,
      messages: [
        {
          role: "user",
          kind: "message",
          content: "Investigate the renderer",
        },
        {
          role: "assistant",
          kind: "thinking",
          content: "Reviewing message boundaries",
          render: {
            startedAt: "2026-06-11T10:00:00.000Z",
          },
        },
        {
          role: "assistant",
          kind: "message",
          content: "I will inspect the current layout first.",
        },
        {
          role: "assistant",
          kind: "tool_call",
          toolName: "terminal",
          content: JSON.stringify({ command: "rg run-event App.tsx" }),
        },
        {
          role: "tool",
          kind: "tool_result",
          toolName: "terminal",
          content: JSON.stringify({ status: "succeeded", output: "run-event" }),
          render: {
            endedAt: "2026-06-11T10:00:41.000Z",
          },
        },
        {
          role: "assistant",
          kind: "message",
          content: "The final summary stays visible.",
        },
      ],
    };
    const api = createDesktopApiMock({
      loadDiscoveredSession: vi.fn(async () => completedSession),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /ACP investigation/i }));

    const work = await screen.findByRole("button", { name: "Worked for 41s" });
    expect(work.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByText("The final summary stays visible.")).toBeTruthy();
    expect(screen.queryByText("Reviewing message boundaries")).toBeNull();
    expect(screen.queryByText("I will inspect the current layout first.")).toBeNull();
    expect(screen.queryByText("terminal")).toBeNull();

    await user.click(work);
    expect(work.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Reviewing message boundaries")).toBeTruthy();
    expect(screen.getByText("I will inspect the current layout first.")).toBeTruthy();
    expect(screen.getAllByText("terminal").length).toBeGreaterThan(0);
    expect(screen.queryByText("Output")).toBeNull();
    expect(screen.getByText("The final summary stays visible.")).toBeTruthy();

    await user.click(work);
    expect(work.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("Reviewing message boundaries")).toBeNull();
    expect(screen.getByText("The final summary stays visible.")).toBeTruthy();
  });

  it("V353/V355 streams open work live, normalizes Thought emphasis, and collapses on completion", async () => {
    const reply = deferred<{ success: boolean; messages: MessageChunk[] }>();
    const unsubscribe = vi.fn();
    const api = createDesktopApiMock({
      sendMessage: vi.fn(() => reply.promise),
      onAgentChunk: vi.fn(() => unsubscribe),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.type(screen.getByRole("textbox"), "Inspect the message history");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    const working = await screen.findByRole("button", { name: "Working" });
    expect(working.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("status").textContent).toContain("Waiting for agent output");

    const requestId = api.sendMessage.mock.calls[0]?.[0]?.requestId as string;
    const emitChunk = api.onAgentChunk.mock.calls[0]?.[0] as (event: {
      requestId: string;
      chunk: MessageChunk;
    }) => void;
    await act(async () => {
      emitChunk({
        requestId,
        chunk: { role: "assistant", kind: "thinking", content: "**Inspecting " },
      });
      emitChunk({
        requestId,
        chunk: { role: "assistant", kind: "thinking", content: "README**" },
      });
      emitChunk({
        requestId,
        chunk: {
          role: "assistant",
          kind: "message",
          content: "I will read the Project files now.",
        },
      });
      emitChunk({
        requestId,
        chunk: {
          role: "assistant",
          kind: "tool_call",
          toolName: "workspace_read_file",
          content: JSON.stringify({ path: "README.md" }),
        },
      });
      emitChunk({
        requestId,
        chunk: {
          role: "tool",
          kind: "tool_result",
          toolName: "workspace_read_file",
          content: JSON.stringify({ path: "README.md", content: "# SwarmX" }),
        },
      });
    });

    const thought = screen.getByText("Inspecting README");
    expect(thought.closest(".run-event__markdown")?.querySelector("strong")).toBeNull();
    const commentary = screen.getByText("I will read the Project files now.");
    expect(commentary.closest(".run-event")?.querySelector(".run-event__card")).toBeNull();
    expect(screen.getAllByText("workspace_read_file").length).toBeGreaterThan(0);

    reply.resolve({
      success: true,
      messages: [
        {
          role: "assistant",
          kind: "thinking",
          content: "**Inspecting README**",
        },
        {
          role: "assistant",
          kind: "message",
          content: "I will read the Project files now.",
        },
        {
          role: "assistant",
          kind: "tool_call",
          toolName: "workspace_read_file",
          content: JSON.stringify({ path: "README.md" }),
        },
        {
          role: "tool",
          kind: "tool_result",
          toolName: "workspace_read_file",
          content: JSON.stringify({ path: "README.md", content: "# SwarmX" }),
        },
        {
          role: "assistant",
          kind: "message",
          content: "The completed answer remains open.",
        },
      ],
    });

    expect(await screen.findByText("The completed answer remains open.")).toBeTruthy();
    const worked = await screen.findByRole("button", { name: /Worked for \d+s/ });
    await waitFor(() => expect(worked.getAttribute("aria-expanded")).toBe("false"));
    expect(screen.queryByText("Inspecting README")).toBeNull();
    expect(screen.queryByText("I will read the Project files now.")).toBeNull();
    expect(screen.queryByText("workspace_read_file")).toBeNull();
    expect(unsubscribe).toHaveBeenCalledTimes(1);
  }, 15_000);

  it("V341/V352 persists timing and renders Worked reasoning as unboxed body text", async () => {
    const api = createDesktopApiMock({
      sendMessage: vi.fn(async () => ({
        success: true,
        messages: [
          {
            role: "assistant",
            kind: "thinking",
            agent: "swarmx_gpt_5_6_luna",
            content: "Reading the **active Project**",
          },
          { role: "assistant", kind: "message", content: "Project inspected." },
        ],
      })),
    });
    const user = userEvent.setup();
    await renderApp(api);

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "Inspect this Project" } });
    await user.click(screen.getByRole("button", { name: "Send message" }));

    const worked = await screen.findByRole("button", { name: /Worked for \d+s/ });
    await user.click(worked);
    const emphasized = screen.getByText("active Project");
    expect(emphasized.tagName).toBe("STRONG");
    const reasoning = emphasized.closest(".run-event");
    expect(reasoning).toBeTruthy();
    expect(reasoning?.querySelector(".run-event__card")).toBeNull();
    expect(reasoning?.querySelector(".run-event__header")).toBeNull();
    expect(screen.queryByText("Reasoning")).toBeNull();
    expect(screen.queryByText("thought")).toBeNull();
    expect(screen.queryByText("swarmx_gpt_5_6_luna")).toBeNull();
    const saved = api.saveSession.mock.calls.at(-1)?.[0] as SessionData;
    const timed = saved.messages.find((message) => message.kind === "thinking");
    expect(timed?.render).toMatchObject({
      startedAt: expect.any(String),
      endedAt: expect.any(String),
      durationMs: expect.any(Number),
    });
  });

  it("V343 generates a short title after the first successful task response", async () => {
    const api = createDesktopApiMock({
      sendMessage: vi.fn(async () => ({
        success: true,
        messages: [{ role: "assistant", kind: "message", content: "Implemented." }],
      })),
      generateSessionTitle: vi.fn(async () => ({
        title: "Fix Project context",
        updated: true,
      })),
    });
    const user = userEvent.setup();
    await renderApp(api);

    await user.type(screen.getByRole("textbox"), "Fix the Project context");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() =>
      expect(api.generateSessionTitle).toHaveBeenCalledWith("created-1", "Fix the Project context"),
    );
    expect(await screen.findByRole("heading", { name: "Fix Project context" })).toBeTruthy();
  });

  it("V344 opens a centered rename dialog from a local task double-click", async () => {
    const api = createDesktopApiMock();
    await renderApp(api);
    const task = (await screen.findByText("Existing local run")).closest("button");
    if (!task) throw new Error("local task button was not rendered");

    fireEvent.doubleClick(task);
    const dialog = screen.getByRole("dialog", { name: "Rename task" });
    expect(dialog.classList.contains("session-rename-dialog")).toBe(true);
    const input = within(dialog).getByRole("textbox", { name: "Task title" });
    expect((input as HTMLInputElement).value).toBe("Existing local run");
    fireEvent.change(input, { target: { value: "Renamed local task" } });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(api.renameSession).toHaveBeenCalledWith("local-1", "Renamed local task"),
    );
    expect(screen.queryByRole("dialog", { name: "Rename task" })).toBeNull();
  });

  it("V345 exposes pin, rename, and delete only in local task context menus", async () => {
    const api = createDesktopApiMock();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    await renderApp(api);
    const localTask = (await screen.findByText("Existing local run")).closest("button");
    const acpTask = screen.getByText("ACP investigation").closest("button");
    if (!localTask || !acpTask) throw new Error("sidebar tasks were not rendered");

    fireEvent.contextMenu(acpTask, { clientX: 40, clientY: 40 });
    expect(screen.queryByRole("menu", { name: /Task actions/ })).toBeNull();

    fireEvent.contextMenu(localTask, { clientX: 40, clientY: 40 });
    let menu = screen.getByRole("menu", { name: "Task actions for Existing local run" });
    expect(within(menu).getByRole("menuitem", { name: "Pin task" })).toBeTruthy();
    expect(within(menu).getByRole("menuitem", { name: "Rename task" })).toBeTruthy();
    expect(within(menu).getByRole("menuitem", { name: "Delete task" })).toBeTruthy();
    fireEvent.click(within(menu).getByRole("menuitem", { name: "Pin task" }));
    await waitFor(() => expect(api.setSessionPinned).toHaveBeenCalledWith("local-1", true));

    fireEvent.contextMenu(localTask, { clientX: 40, clientY: 40 });
    menu = screen.getByRole("menu", { name: "Task actions for Existing local run" });
    fireEvent.click(within(menu).getByRole("menuitem", { name: "Delete task" }));
    expect(confirm).toHaveBeenCalled();
    await waitFor(() => expect(api.deleteSession).toHaveBeenCalledWith("local-1"));
  });

  it("groups sessions by project and keeps discovered ACP history read-only", async () => {
    const api = createDesktopApiMock();
    const user = userEvent.setup();

    await renderApp(api);

    expect(await screen.findByText("Existing local run")).toBeTruthy();
    expect(screen.getByText("ACP investigation")).toBeTruthy();
    expect(screen.queryByLabelText("Harness")).toBeNull();
    expect(screen.queryByLabelText("Session grouping")).toBeNull();
    expect(screen.getByLabelText("swarmx")).toBeTruthy();
    expect(screen.getByLabelText("No project")).toBeTruthy();
    expect(api.listGroupedSessions).toHaveBeenCalledWith({ mode: "project" });

    await user.click(screen.getByRole("button", { name: /ACP investigation/i }));

    expect(await screen.findByRole("heading", { name: "ACP investigation" })).toBeTruthy();
    expect(await screen.findByText("Previous ACP answer")).toBeTruthy();
    expect(api.loadDiscoveredSession).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "acp-1",
        source: "acp",
      }),
    );

    const composer = screen.getByPlaceholderText(
      "ACP history is read-only until resume is supported",
    );
    expect(composer.hasAttribute("disabled")).toBe(true);
    const send = screen.getByRole("button", { name: "Send message" });
    expect(send.hasAttribute("disabled")).toBe(true);
    expect(send.getAttribute("title")).toMatch(/read-only until session resume/i);
    expect(api.sendMessage).not.toHaveBeenCalled();
    expect(api.saveSession).not.toHaveBeenCalledWith(expect.objectContaining({ id: "acp-1" }));
  });

  it("renders tool payloads through normalized redaction", async () => {
    const sessionWithToolTrace: SessionData = {
      ...acpSessionDetail,
      messages: [
        ...acpSessionDetail.messages,
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "terminal",
          render: {
            artifacts: [
              {
                artifactId: "art_terminal_log",
                kind: "log",
                path: "autonomy/runs/run_1/terminal.log",
                title: "terminal.log",
                truncated: true,
              },
            ],
            provenance: {
              adapter: "codex",
              host: "desktop",
              mcpServer: "filesystem",
              pluginId: "geepilot",
            },
            rawPayloadRef: "autonomy/runs/run_1/tool_result_terminal.json",
          },
          content: JSON.stringify({
            status: "failed",
            command: "curl",
            cwd: "/Users/tcztzy/swarmx",
            durationMs: 1200,
            exitCode: 1,
            stdout: "downloaded 0 bytes",
            stdoutTruncated: true,
            stderr: "curl failed",
            apiKey: "sk-test",
            error: "exit 1",
          }),
        },
      ],
    };
    const api = createDesktopApiMock({
      loadDiscoveredSession: vi.fn(async () => sessionWithToolTrace),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /ACP investigation/i }));
    await user.click(await screen.findByRole("button", { name: "Worked" }));

    expect((await screen.findAllByText("terminal")).length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(document.body.textContent).toContain("[redacted]");
    });
    await user.click(screen.getByRole("button", { name: "Show details" }));
    const terminalDetails = screen
      .getByRole("heading", { name: "Terminal" })
      .closest<HTMLElement>(".trace-card__special");
    if (!terminalDetails) throw new Error("Terminal details are missing");
    expect(within(terminalDetails).getByText("command")).toBeTruthy();
    expect(within(terminalDetails).getByText("cwd")).toBeTruthy();
    expect(within(terminalDetails).getByText("/Users/tcztzy/swarmx")).toBeTruthy();
    expect(within(terminalDetails).getByText("exit")).toBeTruthy();
    expect(screen.getAllByText("1").length).toBeGreaterThan(0);
    expect(screen.getByText("downloaded 0 bytes")).toBeTruthy();
    expect(screen.getAllByText("truncated").length).toBeGreaterThan(0);
    expect(screen.getByText("curl failed")).toBeTruthy();
    expect(screen.getByText("Output")).toBeTruthy();
    expect(screen.getByText("Artifacts")).toBeTruthy();
    expect(screen.getByText("terminal.log")).toBeTruthy();
    expect(screen.getByText("Raw payload ref")).toBeTruthy();
    expect(screen.getByText("autonomy/runs/run_1/tool_result_terminal.json")).toBeTruthy();
    expect(screen.getByText("mcpServer")).toBeTruthy();
    expect(screen.getByText("filesystem")).toBeTruthy();
    expect(document.body.textContent).not.toContain("sk-test");
    expect(document.querySelector('[data-render-status="failed"]')).toBeTruthy();
    expect(screen.queryByRole("button", { name: /rerun/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /open artifact/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /reveal raw/i })).toBeNull();
  });

  it("renders passive specialized trace presentations for common run artifacts", async () => {
    const sessionWithSpecializedTraces: SessionData = {
      ...acpSessionDetail,
      messages: [
        ...acpSessionDetail.messages,
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "apply_patch",
          render: {
            artifacts: [{ artifactId: "art_diff", kind: "diff", title: "change.patch" }],
          },
          content: JSON.stringify({
            diff: "@@ -1 +1 @@\n-old\n+new",
            path: "packages/core/src/rendering.ts",
          }),
        },
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "read_file",
          content: JSON.stringify({
            operation: "read",
            path: "README.md",
            lineStart: 10,
            lineEnd: 20,
            preview: "SwarmX overview",
          }),
        },
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "vitest",
          content: JSON.stringify({
            status: "failed",
            passed: 12,
            failed: 1,
            testCount: 13,
            failures: "expected true to equal false",
            output: "1 failed, 12 passed",
          }),
        },
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "mcp.call",
          render: {
            provenance: { mcpServer: "filesystem", pluginId: "swarmx" },
          },
          content: JSON.stringify({
            tool: "read_file",
            result: "Read README.md",
          }),
        },
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "playwright",
          render: {
            artifacts: [{ artifactId: "art_screen", kind: "screenshot", title: "screen.png" }],
          },
          content: JSON.stringify({
            action: "click",
            target: "https://example.test",
            summary: "button clicked",
          }),
        },
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "image_generation",
          render: {
            artifacts: [{ artifactId: "art_plot", kind: "image", title: "plot.png" }],
          },
          content: "generated plot",
        },
      ],
    };
    const api = createDesktopApiMock({
      loadDiscoveredSession: vi.fn(async () => sessionWithSpecializedTraces),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /ACP investigation/i }));
    await user.click(await screen.findByRole("button", { name: "Worked" }));
    for (const button of screen.getAllByRole("button", { name: "Show details" })) {
      await user.click(button);
    }

    expect(screen.getByRole("heading", { name: "Diff" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "File" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Test/check" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "MCP" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Automation" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Generated media" })).toBeTruthy();
    expect(screen.getByText("packages/core/src/rendering.ts")).toBeTruthy();
    expect(screen.getByText("README.md")).toBeTruthy();
    expect(screen.getByText("10-20")).toBeTruthy();
    expect(screen.getByText("expected true to equal false")).toBeTruthy();
    expect(screen.getByText("Read README.md")).toBeTruthy();
    expect(screen.getByText("https://example.test")).toBeTruthy();
    expect(screen.getByText("button clicked")).toBeTruthy();
    expect(screen.getByText("plot.png")).toBeTruthy();
    const transcript = document.querySelector<HTMLElement>(".transcript-scroll");
    expect(transcript).not.toBeNull();
    expect(within(transcript as HTMLElement).queryByRole("button", { name: /rerun/i })).toBeNull();
    expect(within(transcript as HTMLElement).queryByRole("button", { name: /open/i })).toBeNull();
  });

  it("falls back to the generic trace card for unknown tool payloads", async () => {
    const sessionWithUnknownTrace: SessionData = {
      ...acpSessionDetail,
      messages: [
        ...acpSessionDetail.messages,
        {
          role: "tool",
          kind: "tool_result",
          agent: "codex",
          toolName: "opaque",
          content: JSON.stringify({ note: "plain generic output" }),
        },
      ],
    };
    const api = createDesktopApiMock({
      loadDiscoveredSession: vi.fn(async () => sessionWithUnknownTrace),
    });
    const user = userEvent.setup();

    await renderApp(api);
    await user.click(await screen.findByRole("button", { name: /ACP investigation/i }));
    await user.click(await screen.findByRole("button", { name: "Worked" }));
    await user.click(screen.getByRole("button", { name: "Show details" }));

    expect(screen.getByText("Output")).toBeTruthy();
    expect(document.body.textContent).toContain("plain generic output");
    expect(screen.queryByText("Terminal")).toBeNull();
    expect(screen.queryByText("Diff")).toBeNull();
    expect(screen.queryByText("Test/check")).toBeNull();
    expect(screen.queryByText("MCP")).toBeNull();
    expect(screen.queryByText("Automation")).toBeNull();
    expect(screen.queryByText("Generated media")).toBeNull();
  });
});

async function renderApp(api: DesktopApiMock, appProps: AppProps = {}): Promise<void> {
  vi.resetModules();
  Object.defineProperty(window, "swarmxAPI", {
    configurable: true,
    value: api,
  });
  Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    configurable: true,
    value: vi.fn(),
  });

  const { App } = await import("./App.js");
  render(
    <SWRConfig value={{ provider: () => new Map(), dedupingInterval: 0 }}>
      <App {...appProps} />
    </SWRConfig>,
  );
}

function readyHarnessEnvironment() {
  return {
    checkedAt: "2026-07-08T00:00:00.000Z",
    path: "/Users/test/.npm-global/bin:/usr/bin",
    ready: true,
    setupAvailable: false,
    containerRuntimes: [
      {
        id: "apple_container",
        label: "Apple Container",
        command: "container",
        status: "ready",
        supported: true,
        installable: true,
        serviceReady: true,
        preferred: true,
        path: "/usr/local/bin/container",
        version: "1.1.0",
        note: "Apple Container system service is running.",
      },
    ],
    protection: {
      mode: "protected",
      ready: true,
      requiredHarnessIds: ["claude_code", "codex"],
      selectedRuntimeId: "apple_container",
      note: "Protected harness execution uses Apple Container.",
    },
    requirements: [
      {
        id: "node",
        label: "Node.js runtime",
        command: "node",
        status: "ready",
        installable: false,
        requiredBy: [],
        path: "/opt/homebrew/bin/node",
        version: "22.17.0",
      },
      {
        id: "claude_code",
        label: "Claude Code",
        command: "claude",
        status: "ready",
        installable: true,
        requiredBy: ["claude_code"],
        path: "/Users/test/.npm-global/bin/claude",
        version: "2.1.0",
      },
      {
        id: "codex",
        label: "Codex",
        command: "codex",
        status: "ready",
        installable: true,
        requiredBy: ["codex"],
        path: "/Users/test/.npm-global/bin/codex",
        version: "0.69.0",
      },
      {
        id: "opencode",
        label: "OpenCode CLI",
        command: "opencode",
        status: "ready",
        installable: true,
        requiredBy: ["opencode"],
        path: "/Users/test/.local/bin/opencode",
        version: "1.0.0",
      },
      {
        id: "hermes",
        label: "Hermes Agent CLI",
        command: "hermes",
        status: "ready",
        installable: true,
        requiredBy: ["hermes"],
        path: "/Users/test/.local/bin/hermes",
        version: "1.0.0",
      },
      {
        id: "openclaw",
        label: "OpenClaw CLI",
        command: "openclaw",
        status: "ready",
        installable: true,
        requiredBy: ["openclaw"],
        path: "/Users/test/.local/bin/openclaw",
        version: "2026.6.11",
      },
    ],
    harnesses: [
      {
        harnessId: "swarmx",
        harnessLabel: "SwarmX",
        command: "swarmx",
        installable: false,
        version: "3.0.1",
        status: "ready",
        requirements: [],
        executionMode: "native",
        protectionRequired: false,
        note: "Built in.",
      },
      {
        harnessId: "claude_code",
        harnessLabel: "Claude Code",
        command: "claude",
        installable: true,
        path: "/Users/test/.npm-global/bin/claude",
        version: "2.1.0",
        status: "ready",
        requirements: ["claude_code"],
        executionMode: "protected",
        protectionRequired: true,
        containerRuntimeId: "apple_container",
      },
      {
        harnessId: "codex",
        harnessLabel: "Codex",
        command: "codex",
        installable: true,
        path: "/Users/test/.npm-global/bin/codex",
        version: "0.69.0",
        status: "ready",
        requirements: ["codex"],
        executionMode: "protected",
        protectionRequired: true,
        containerRuntimeId: "apple_container",
      },
      {
        harnessId: "opencode",
        harnessLabel: "OpenCode",
        command: "opencode",
        installable: true,
        version: "1.0.0",
        status: "ready",
        requirements: ["opencode"],
        executionMode: "native",
        protectionRequired: false,
      },
      {
        harnessId: "hermes",
        harnessLabel: "Hermes",
        command: "hermes",
        installable: true,
        version: "1.0.0",
        status: "ready",
        requirements: ["hermes"],
        executionMode: "native",
        protectionRequired: false,
      },
      {
        harnessId: "openclaw",
        harnessLabel: "OpenClaw",
        command: "openclaw",
        installable: true,
        version: "2026.6.11",
        status: "ready",
        requirements: ["openclaw"],
        executionMode: "native",
        protectionRequired: false,
      },
    ],
  };
}

function missingHarnessEnvironment() {
  const ready = readyHarnessEnvironment();
  return {
    ...ready,
    ready: true,
    setupAvailable: false,
    containerRuntimes: ready.containerRuntimes.map((runtime) => ({
      ...runtime,
      status: "missing",
      serviceReady: false,
      path: undefined,
      version: undefined,
      note: "Apple Container must be installed and its system service started.",
    })),
    protection: {
      ...ready.protection,
      ready: false,
      note: "Apple Container must be installed and its system service started.",
    },
    requirements: ready.requirements.map((requirement) =>
      requirement.id === "openclaw"
        ? { ...requirement, status: "missing", path: undefined, version: undefined }
        : requirement,
    ),
    harnesses: ready.harnesses.map((harness) => {
      if (harness.harnessId === "openclaw") {
        return { ...harness, version: undefined, status: "needs_setup" };
      }
      return harness.protectionRequired ? { ...harness, status: "needs_setup" } : harness;
    }),
  };
}

function doctorReport(environment: ReturnType<typeof readyHarnessEnvironment>, harnessId?: string) {
  const selectedHarnesses = harnessId
    ? environment.harnesses.filter((harness) => harness.harnessId === harnessId)
    : environment.harnesses;
  const selectedProtectedHarnesses = selectedHarnesses.filter(
    (harness) => harness.executionMode === "protected",
  );
  const issues = [];
  const repairActions = [];

  const nativeRequirementIds = new Set(
    selectedHarnesses
      .filter((harness) => harness.executionMode === "native")
      .flatMap((harness) => harness.requirements),
  );
  const filteredEnvironment = harnessId
    ? {
        ...environment,
        containerRuntimes:
          selectedProtectedHarnesses.length > 0 ? environment.containerRuntimes : [],
        requirements: environment.requirements.filter((requirement) =>
          nativeRequirementIds.has(requirement.id),
        ),
        harnesses: selectedHarnesses,
      }
    : environment;

  return {
    checkedAt: environment.checkedAt,
    healthy: issues.length === 0,
    ...(harnessId ? { harnessId } : {}),
    summary: {
      readyHarnesses: selectedHarnesses.filter((harness) => harness.status === "ready").length,
      totalHarnesses: selectedHarnesses.length,
      issueCount: issues.length,
      fixableCount: repairActions.length,
    },
    issues,
    repairActions,
    environment: filteredEnvironment,
  };
}

function activityProfileFixture() {
  return {
    generatedAt: "2026-07-16T12:00:00.000Z",
    trackingSince: "2026-07-01T09:00:00.000Z",
    lifetime: {
      totalTokens: 1_250,
      inputTokens: 700,
      outputTokens: 400,
      reasoningTokens: 150,
      cachedInputTokens: 120,
      estimatedTokens: 100,
      peakDayTokens: 600,
      longestTaskMs: 3_600_000,
      currentStreakDays: 3,
      longestStreakDays: 7,
      totalTasks: 12,
      completedTasks: 11,
      toolCalls: 18,
      skillCalls: 9,
      skillsExplored: 4,
    },
    daily: [
      {
        date: "2026-07-15",
        tokens: 600,
        estimatedTokens: 0,
        tasks: 2,
        tools: 4,
        skills: 2,
      },
      {
        date: "2026-07-16",
        tokens: 650,
        estimatedTokens: 100,
        tasks: 3,
        tools: 5,
        skills: 2,
      },
    ],
    topTools: [{ name: "workspace_read_file", count: 8 }],
    topSkills: [{ name: "paper-reviewer", count: 5 }],
    reasoningEfforts: [{ name: "high", count: 8 }],
    models: [{ name: "gpt-5", count: 10 }],
  };
}

function createDesktopApiMock(overrides: Partial<DesktopApiMock> = {}): DesktopApiMock {
  return {
    initialProjects: [swarmxProject],
    sendMessage: vi.fn(async () => ({
      success: true,
      messages: [],
    })),
    onAgentChunk: vi.fn(() => () => undefined),
    onAgentInteraction: vi.fn(() => () => undefined),
    onSessionMessages: vi.fn(() => () => undefined),
    resolveAgentInteraction: vi.fn(
      async (params: { requestId: string; interactionId: string }) => ({
        requestId: params.requestId,
        interactionId: params.interactionId,
        resolved: true,
      }),
    ),
    cancelMessage: vi.fn(async (requestId: string) => ({ requestId, canceled: true })),
    createSession: vi.fn(async (params?: { projectId?: string; cwd?: string }) => ({
      ...localSession,
      id: "created-1",
      title: "Untitled",
      messages: [],
      ...(params?.projectId ? { projectId: params.projectId } : {}),
      ...(params?.cwd ? { cwd: params.cwd } : {}),
    })),
    saveSession: vi.fn(async () => undefined),
    loadSession: vi.fn(async () => localSession),
    loadDiscoveredSession: vi.fn(async (session: DiscoveredSession) =>
      session.id === "acp-1" ? acpSessionDetail : localSession,
    ),
    listSessions: vi.fn(async () => [localSession]),
    getActivityProfile: vi.fn(async () => activityProfileFixture()),
    listProjects: vi.fn(async () => [swarmxProject]),
    addExistingProject: vi.fn(async () => null),
    createScratchProject: vi.fn(async () => null),
    setProjectPinned: vi.fn(async (_id: string, pinned: boolean) => ({
      ...swarmxProject,
      pinned,
    })),
    renameProject: vi.fn(async (_id: string, name: string) => ({ ...swarmxProject, name })),
    revealProject: vi.fn(async () => true),
    archiveProjectTasks: vi.fn(async () => 0),
    removeProject: vi.fn(async () => true),
    listGroupedSessions: vi.fn(async () => ({
      mode: "project",
      groups: [
        {
          id: "codex",
          label: "Codex",
          sessions: [discoveredAcpSession],
        },
      ],
      errors: [],
    })),
    deleteSession: vi.fn(async () => true),
    renameSession: vi.fn(async (_id: string, title: string) => ({ ...localSession, title })),
    setSessionPinned: vi.fn(async (_id: string, pinned: boolean) => ({
      ...localSession,
      pinned,
    })),
    generateSessionTitle: vi.fn(async () => ({ title: "Generated task title", updated: false })),
    appendMessages: vi.fn(async () => true),
    importN8nWorkflow: vi.fn(async () => ({
      success: false,
      error: "No n8n workflow imported.",
    })),
    workspaceRoot: vi.fn(async () => "/Users/tcztzy/swarmx"),
    createTerminal: vi.fn(async ({ id }: { id: string }) => ({ id, pid: 42 })),
    writeTerminal: vi.fn(async () => ({ written: true })),
    resizeTerminal: vi.fn(async () => ({ resized: true })),
    killTerminal: vi.fn(async () => ({ killed: true })),
    onTerminalData: vi.fn(() => () => undefined),
    onTerminalExit: vi.fn(() => () => undefined),
    selectFilesAndFolders: vi.fn(async () => []),
    refreshModelCatalog: vi.fn(async () => null),
    addManualModel: vi.fn(async () => null),
    removeManualModel: vi.fn(async () => null),
    saveProvider: vi.fn(async () => null),
    removeProvider: vi.fn(async () => null),
    refreshProviderUsage: vi.fn(async () => ({
      fetchedAt: "2026-07-12T12:00:00.000Z",
      providers: [],
      toolAccounts: [
        {
          source: "tool_account",
          sourceId: "codex",
          label: "Codex",
          adapterId: "codex_app_server",
          status: "ready",
          plan: "pro",
          meters: [
            {
              kind: "window",
              id: "five_hour",
              label: "5-hour",
              usedPercent: 7,
              remainingPercent: 93,
            },
            {
              kind: "window",
              id: "weekly",
              label: "Weekly",
              usedPercent: 47,
              remainingPercent: 53,
            },
          ],
        },
      ],
    })),
    getUpdateState: vi.fn(async () => ({
      phase: "hidden",
      currentVersion: "3.0.1",
    })),
    startUpdate: vi.fn(async () => ({
      phase: "hidden",
      currentVersion: "3.0.1",
    })),
    onUpdateState: vi.fn(() => () => undefined),
    getHarnessEnvironment: vi.fn(async () => readyHarnessEnvironment()),
    getHarnessVersion: vi.fn(async ({ harnessId }: { harnessId: string; refresh?: boolean }) => ({
      harnessId,
      version: readyHarnessEnvironment().harnesses.find(
        (harness) => harness.harnessId === harnessId,
      )?.version,
    })),
    inspectDoctor: vi.fn(async (params: { harnessId?: string } = {}) =>
      doctorReport(readyHarnessEnvironment(), params.harnessId),
    ),
    fixDoctor: vi.fn(async (params: { harnessId?: string; confirmed: boolean }) => {
      const report = doctorReport(readyHarnessEnvironment(), params.harnessId);
      return {
        executed: false,
        before: report,
        plan: {
          actions: report.repairActions,
          requiresConfirmation: false,
          requiresAdmin: false,
        },
        setupResults: [],
        after: report,
      };
    }),
    setupHarnessEnvironment: vi.fn(async () => ({
      success: true,
      status: readyHarnessEnvironment(),
      installedRequirementIds: [],
      skippedRequirementIds: [],
      failedRequirementIds: [],
      installedContainerRuntimeIds: [],
      skippedContainerRuntimeIds: [],
      failedContainerRuntimeIds: [],
      log: ["ready"],
    })),
    getExtensionManagementState: vi.fn(async () => ({
      sources: [],
      installed: [],
      skillEvolutionEnabled: false,
      skillPromotionGate: "human",
    })),
    saveExtensionSource: vi.fn(async () => ({
      sources: [],
      installed: [],
      skillEvolutionEnabled: false,
      skillPromotionGate: "human",
    })),
    refreshExtensionSource: vi.fn(async () => ({
      sources: [],
      installed: [],
      skillEvolutionEnabled: false,
      skillPromotionGate: "human",
    })),
    removeExtensionSource: vi.fn(async () => ({
      sources: [],
      installed: [],
      skillEvolutionEnabled: false,
      skillPromotionGate: "human",
    })),
    applyExtensionAction: vi.fn(async () => ({
      receipt: { status: "applied", message: "applied" },
      state: {
        sources: [],
        installed: [],
        skillEvolutionEnabled: false,
        skillPromotionGate: "human",
      },
    })),
    saveSkillEvolutionPolicy: vi.fn(async (input) => ({
      sources: [],
      installed: [],
      skillEvolutionEnabled: input.enabled,
      skillPromotionGate: input.promotionGate,
    })),
    listCustomAgents: vi.fn(async () => ({ agents: [] })),
    saveCustomAgent: vi.fn(async () => ({ agents: [] })),
    removeCustomAgent: vi.fn(async () => ({ agents: [] })),
    listExtensions: vi.fn(async () => ({
      bundles: [
        {
          id: "swarmx.builtin",
          name: "SwarmX Built-ins",
          version: "3.0.1",
          trust: "builtin",
          readOnly: true,
          capabilities: {
            harnesses: [{ id: "swarmx" }, { id: "codex" }],
            agents: [],
            skills: [],
            mcpServers: [],
            uiContributions: [],
            marketplaceSources: [],
            pluginCatalog: [],
          },
        },
        {
          id: "geepilot",
          name: "GEEPilot",
          version: "0.1.0",
          trust: "local",
          capabilities: {
            harnesses: [{ id: "geepilot-codex" }],
            agents: [{ id: "analysis-lead" }],
            skills: [{ id: "geepilot.memory" }, { id: "geepilot.primer3" }],
            mcpServers: [{ id: "project-fs" }],
            commands: [{ id: "geepilot.refresh-index" }],
            lspServers: [{ id: "pyright" }],
            hooks: [{ id: "before-run" }],
            monitors: [{ id: "catalog-refresh" }],
            outputStyles: [{ id: "compact-review" }],
            uiContributions: [
              { id: "geepilot.nav" },
              { id: "geepilot.dashboard" },
              { id: "geepilot.composer.review" },
            ],
            settings: [{ id: "geepilot.indexRoot" }],
            assets: [{ id: "geepilot-icon" }],
            permissions: [{ id: "project-read" }],
            authPolicies: [{ id: "project-fs-auth" }],
            marketplaceSources: [{ id: "codex-local" }],
            pluginCatalog: [{ id: "geepilot" }],
          },
        },
      ],
      marketplaceSources: [
        {
          id: "codex-local",
          name: "Codex local marketplace",
          host: "codex",
          kind: "local_path",
          path: "./.agents/plugins/marketplace.json",
          trust: "local",
          readOnly: true,
        },
      ],
      pluginCatalog: [
        {
          id: "geepilot",
          name: "GEEPilot plugin",
          version: "0.1.0",
          marketplaceSourceId: "codex-local",
          bundleId: "geepilot",
          hosts: ["codex", "claude"],
          trust: "local",
          installState: "installed",
          updateState: "current",
          providesHarness: true,
          componentCounts: {
            commands: 1,
            skills: 2,
            mcpServers: 1,
            lspServers: 1,
            agents: 1,
            hooks: 1,
            monitors: 1,
            outputStyles: 1,
            uiContributions: 3,
            assets: 1,
            settings: 1,
            permissions: 1,
            authPolicies: 1,
          },
        },
      ],
      harnesses: [
        {
          id: "swarmx",
          label: "SwarmX",
          modelControl: "direct",
          modelCompatibility: "declared_apis",
          supportedModelApis: ["openai_chat"],
          software: { name: "swarmx" },
          readOnly: true,
        },
        {
          id: "geepilot-codex",
          label: "GEEPilot Codex",
          modelControl: "session",
          modelCompatibility: "any",
          supportedModelApis: ["openai_responses"],
          software: { name: "codex-acp", version: "0.22.0" },
        },
      ],
      models: [
        {
          id: "gpt-5",
          label: "gpt-5",
          runtimeModel: "gpt-5",
          apiProtocols: ["openai_chat", "openai_responses"],
        },
      ],
      modelSupplies: [],
      providers: [],
      agents: [
        {
          id: "analysis-lead",
          name: "analysis lead",
          harnessId: "geepilot-codex",
          modelId: "gpt-5",
          skills: ["geepilot.memory"],
          mcpServers: ["project-fs"],
          tools: ["Read", "Grep"],
          disallowedTools: ["Bash"],
          permissionMode: "plan",
          maxTurns: 6,
          memory: "readonly",
          effort: "high",
          isolation: "workspace",
        },
        {
          id: "blocked-agent",
          name: "blocked agent",
          harnessId: "missing-harness",
          modelId: "gpt-5",
        },
      ],
      agentPlans: [
        {
          id: "desktop-analysis-lead",
          agentId: "geepilot-codex:gpt-5",
          agentProfileId: "analysis-lead",
          displayName: "analysis lead",
          canonicalSelector: "@analysis-lead",
          host: "local",
          status: "ready",
          healthStatus: "ready",
          harnessId: "geepilot-codex",
          harnessLabel: "GEEPilot Codex",
          modelId: "gpt-5",
          runtimeModel: "gpt-5",
          definition: {
            source: "plugin",
            pluginId: "geepilot",
            path: "agents/analysis-lead.md",
            readOnly: true,
          },
          pluginIds: ["geepilot"],
          skills: [
            {
              id: "geepilot.memory",
              name: "Memory",
              sourcePluginId: "geepilot",
              status: "ok",
            },
          ],
          mcpServers: [
            {
              id: "project-fs",
              name: "Project Files",
              sourcePluginId: "geepilot",
              status: "ok",
            },
          ],
          context: { mode: "thread_packet", strategy: "auto", memory: "readonly" },
          permissions: {
            tools: "2 allowed tools",
            mcp: "selected",
            shell: "plan",
            mode: "plan",
            summary: "mode plan / selected MCP / plan",
          },
          visual: { label: "analysis lead" },
          requirements: [
            { kind: "harness", status: "ok", id: "geepilot-codex", message: "Resolved harness." },
            { kind: "model", status: "ok", id: "gpt-5", message: "Resolved model." },
            { kind: "plugin", status: "ok", id: "geepilot", message: "Resolved plugin." },
            { kind: "secret", status: "unknown", message: "Provider secret required." },
          ],
        },
        {
          id: "desktop-blocked-agent",
          agentId: "missing-harness:gpt-5",
          agentProfileId: "blocked-agent",
          displayName: "blocked agent",
          canonicalSelector: "@blocked-agent",
          host: "local",
          status: "blocked",
          healthStatus: "blocked",
          harnessId: "missing-harness",
          modelId: "gpt-5",
          runtimeModel: "gpt-5",
          definition: { source: "none" },
          pluginIds: [],
          skills: [],
          mcpServers: [],
          context: { mode: "thread_packet", strategy: "auto" },
          permissions: {
            tools: "inherit",
            mcp: "none",
            shell: "harness-policy",
            summary: "none MCP / harness-policy",
          },
          requirements: [
            {
              kind: "harness",
              status: "missing",
              id: "missing-harness",
              message: 'Unknown harness id "missing-harness".',
            },
          ],
        },
      ],
      skills: [
        {
          id: "geepilot.biosecurity",
          name: "Biosecurity",
          path: "skills/biosecurity/SKILL.md",
          canonicalPath: "skills/biosecurity/SKILL.md",
          governanceRef: "docs/skills-governance.md",
          hostExposures: [
            {
              host: "codex",
              status: "plugin",
              manifestPath: "./.codex-plugin/plugin.json",
              marketplaceSourceId: "codex-local",
            },
            { host: "opencode", status: "rules_only", rulesPath: "./AGENTS.md" },
          ],
          readOnly: true,
        },
        {
          id: "geepilot.primer3",
          name: "Primer3",
          path: "skills/primer3/SKILL.md",
          canonicalPath: "skills/primer3/SKILL.md",
          requiresGateSkillIds: ["geepilot.biosecurity"],
          hostExposures: [
            {
              host: "claude",
              status: "plugin",
              manifestPath: "./.claude-plugin/plugin.json",
            },
          ],
        },
      ],
      mcpServers: [{ id: "project-fs", name: "Project Files", scope: "project" }],
      appConnectors: [],
      uiContributions: [
        {
          id: "geepilot.nav",
          kind: "navigation_item",
          name: "GEEPilot navigation",
          placement: "sidebar",
          route: "/extensions/geepilot",
          componentRef: "geepilot.ui.shell",
          assetRef: "geepilot-icon",
          sourcePluginId: "geepilot",
          readOnly: true,
        },
        {
          id: "geepilot.dashboard",
          kind: "dashboard_widget",
          name: "Analysis dashboard",
          placement: "dashboard",
          componentRef: "geepilot.ui.dashboard",
          permissionIds: ["project-read"],
          authPolicyIds: ["project-fs-auth"],
        },
        {
          id: "geepilot.composer.review",
          kind: "composer_action",
          name: "Review dataset",
          placement: "composer",
          commandId: "geepilot.refresh-index",
          settingIds: ["geepilot.indexRoot"],
        },
      ],
      commands: [
        {
          id: "geepilot.refresh-index",
          name: "Refresh index",
          command: ["geepilot", "index", "refresh"],
          scope: "plugin",
        },
      ],
      lspServers: [
        {
          id: "pyright",
          name: "Pyright",
          languages: ["python"],
          command: ["pyright-langserver", "--stdio"],
          scope: "project",
        },
      ],
      hooks: [{ id: "before-run", name: "Before run", event: "before_agent_run" }],
      monitors: [{ id: "catalog-refresh", trigger: "schedule", schedule: "PT1H" }],
      outputStyles: [
        { id: "compact-review", name: "Compact Review", path: "output-styles/compact.md" },
      ],
      settings: [{ id: "geepilot.indexRoot", name: "Index root", valueType: "string" }],
      assets: [{ id: "geepilot-icon", kind: "icon", path: "assets/icon.png" }],
      permissions: [
        { id: "project-read", kind: "filesystem", access: "read", target: "workspace" },
      ],
      authPolicies: [
        {
          id: "project-fs-auth",
          kind: "env",
          secretRefs: [{ source: "env", key: "GEEPILOT_PROJECT_TOKEN" }],
        },
      ],
      warnings: [],
    })),
    lspComplete: vi.fn(async () => ({ serverId: "pyright", status: "ok", result: null })),
    lspStop: vi.fn(async () => ({ serverId: "pyright", stopped: true })),
    loadImageDataUrl: vi.fn(async () => null),
    ...overrides,
  };
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

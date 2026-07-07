/** @vitest-environment jsdom */

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SWRConfig } from "swr";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppProps } from "./App.js";

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
    provenance?: Record<string, string>;
    rawPayloadRef?: string;
    status?: string;
  };
  swarmEvent?: string;
  toolName?: string;
}

interface SessionData {
  id: string;
  title: string;
  agentName: string;
  harness: string;
  messages: MessageChunk[];
  createdAt: string;
  updatedAt: string;
}

interface DiscoveredSession {
  id: string;
  title: string;
  cwd: string;
  updatedAt?: string;
  harnessId: string;
  harnessLabel: string;
  source: "local" | "acp";
}

interface DesktopApiMock {
  sendMessage: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  saveSession: ReturnType<typeof vi.fn>;
  loadSession: ReturnType<typeof vi.fn>;
  loadDiscoveredSession: ReturnType<typeof vi.fn>;
  listSessions: ReturnType<typeof vi.fn>;
  listGroupedSessions: ReturnType<typeof vi.fn>;
  deleteSession: ReturnType<typeof vi.fn>;
  appendMessages: ReturnType<typeof vi.fn>;
  importN8nWorkflow: ReturnType<typeof vi.fn>;
  listExtensions: ReturnType<typeof vi.fn>;
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

const acpSessionDetail: SessionData = {
  id: "acp-1",
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

    await user.click(await screen.findByRole("button", { name: /Extensions/i }));

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
    fireEvent.change(screen.getByPlaceholderText("Message analysis lead"), {
      target: { value: "Plan a GEEPilot analysis" },
    });
    await user.click(screen.getByRole("button", { name: /Send/i }));

    await waitFor(() => {
      expect(api.createSession).toHaveBeenCalledWith({
        agentName: "analysis lead",
        harness: "geepilot-codex",
      });
      expect(api.sendMessage).toHaveBeenCalledWith({
        harnessId: "geepilot-codex",
        userText: "Plan a GEEPilot analysis",
        agentComposition: {
          id: "desktop-analysis-lead",
          agentProfileId: "analysis-lead",
          host: "local",
        },
      });
    });
  }, 40_000);

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
    expect(screen.getByLabelText("Workflow node triage_agent codex gpt-4o-mini root")).toBeTruthy();
    expect(
      screen.getByLabelText("Workflow node researcher_agent claude_code claude-sonnet-4-20250514"),
    ).toBeTruthy();
    expect(screen.getByLabelText("Workflow node writer_agent codex gpt-4o")).toBeTruthy();
    expect(screen.getAllByText("Harness Codex")).toHaveLength(2);
    expect(screen.getByText("Harness Claude Code")).toBeTruthy();
    expect(screen.getByText("Model gpt-4o-mini")).toBeTruthy();
    expect(screen.getByText("Model claude-sonnet-4-20250514")).toBeTruthy();
    expect(screen.getByText("harness = software + MCPs + skills + project files")).toBeTruthy();
    expect(screen.getAllByText("Software codex-acp@0.22.0")).toHaveLength(2);
    expect(screen.getByText("Software claude-agent-acp@0.22.0")).toBeTruthy();
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
                  model: "gpt-4o-mini",
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "bun",
                    args: expect.arrayContaining(["@agentclientprotocol/codex-acp@0.22.0"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "codex-acp",
                        version: "0.22.0",
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
                  model: "claude-sonnet-4-20250514",
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "bun",
                    args: expect.arrayContaining(["@agentclientprotocol/claude-agent-acp@0.22.0"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "claude-agent-acp",
                        version: "0.22.0",
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
                  model: "gpt-4o",
                  backend: expect.objectContaining({
                    type: "custom",
                    program: "bun",
                    args: expect.arrayContaining(["@agentclientprotocol/codex-acp@0.22.0"]),
                  }),
                  parameters: expect.objectContaining({
                    harness: expect.objectContaining({
                      software: expect.objectContaining({
                        name: "codex-acp",
                        version: "0.22.0",
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
  }, 20_000);

  it("shows workflow JSON errors and omits swarm config while invalid", async () => {
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

    await waitFor(() => {
      expect(api.sendMessage).toHaveBeenCalledWith({
        harnessId: "swarmx",
        userText: "Run without config",
      });
    });
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
      expect(api.sendMessage).toHaveBeenCalledWith({
        harnessId: "swarmx",
        userText: "Run imported workflow",
        swarmConfig: importedConfig,
      });
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

  it("loads a discovered session, switches grouping, sends a follow-up, and persists the reply", async () => {
    const reply = deferred<{
      success: true;
      messages: MessageChunk[];
    }>();
    const api = createDesktopApiMock({
      sendMessage: vi.fn(() => reply.promise),
    });
    const user = userEvent.setup();

    await renderApp(api);

    expect(await screen.findByText("Existing local run")).toBeTruthy();
    expect(screen.getByText("ACP investigation")).toBeTruthy();
    expect(screen.getByRole("tab", { name: "Harness" }).getAttribute("aria-selected")).toBe("true");

    await user.click(screen.getByRole("tab", { name: "Project" }));

    expect(screen.getByRole("tab", { name: "Project" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText("/Users/tcztzy/swarmx")).toBeTruthy();
    expect(screen.getByText("No project")).toBeTruthy();

    await user.click(screen.getByRole("tab", { name: "Harness" }));
    await user.click(screen.getByRole("button", { name: /ACP investigation/i }));

    expect(await screen.findByRole("heading", { name: "ACP investigation" })).toBeTruthy();
    expect(await screen.findByText("Previous ACP answer")).toBeTruthy();
    expect(api.loadDiscoveredSession).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "acp-1",
        source: "acp",
      }),
    );

    fireEvent.change(screen.getByPlaceholderText("Message Codex"), {
      target: { value: "Continue the investigation" },
    });
    await user.click(screen.getByRole("button", { name: /Send/i }));

    expect(api.sendMessage).toHaveBeenCalledWith({
      harnessId: "codex",
      userText: "Continue the investigation",
    });
    expect(await screen.findByText("Running")).toBeTruthy();
    expect(screen.getByText("Thinking")).toBeTruthy();
    expect(screen.getByText("Continue the investigation")).toBeTruthy();

    reply.resolve({
      success: true,
      messages: [
        {
          role: "assistant",
          kind: "message",
          agent: "codex",
          content: "Follow-up complete.",
        },
      ],
    });

    expect(await screen.findByText("Follow-up complete.")).toBeTruthy();
    expect(screen.getByText("4 events")).toBeTruthy();
    await waitFor(() => {
      expect(api.saveSession).toHaveBeenCalledWith(
        expect.objectContaining({
          id: "acp-1",
          messages: expect.arrayContaining([
            expect.objectContaining({ content: "Continue the investigation" }),
            expect.objectContaining({ content: "Follow-up complete." }),
          ]),
        }),
      );
    });
    expect(api.listSessions).toHaveBeenCalledTimes(2);
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

    expect(await screen.findByText("terminal")).toBeTruthy();
    await waitFor(() => {
      expect(document.body.textContent).toContain("[redacted]");
    });
    await user.click(screen.getByRole("button", { name: "Show details" }));
    expect(screen.getByRole("heading", { name: "Terminal" })).toBeTruthy();
    expect(screen.getByText("command")).toBeTruthy();
    expect(screen.getByText("cwd")).toBeTruthy();
    expect(screen.getByText("/Users/tcztzy/swarmx")).toBeTruthy();
    expect(screen.getByText("exit")).toBeTruthy();
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
    expect(screen.queryByRole("button", { name: /rerun/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /open/i })).toBeNull();
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

function createDesktopApiMock(overrides: Partial<DesktopApiMock> = {}): DesktopApiMock {
  return {
    sendMessage: vi.fn(async () => ({
      success: true,
      messages: [],
    })),
    createSession: vi.fn(async () => ({
      ...localSession,
      id: "created-1",
      title: "Untitled",
      messages: [],
    })),
    saveSession: vi.fn(async () => undefined),
    loadSession: vi.fn(async () => localSession),
    loadDiscoveredSession: vi.fn(async (session: DiscoveredSession) =>
      session.id === "acp-1" ? acpSessionDetail : localSession,
    ),
    listSessions: vi.fn(async () => [localSession]),
    listGroupedSessions: vi.fn(async () => ({
      mode: "harness",
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
    appendMessages: vi.fn(async () => true),
    importN8nWorkflow: vi.fn(async () => ({
      success: false,
      error: "No n8n workflow imported.",
    })),
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
          compatibleProviders: ["openai_chat"],
          software: { name: "swarmx" },
          readOnly: true,
        },
        {
          id: "geepilot-codex",
          label: "GEEPilot Codex",
          compatibleProviders: ["openai_responses"],
          software: { name: "codex-acp", version: "0.22.0" },
        },
      ],
      providers: [],
      agents: [
        {
          id: "analysis-lead",
          name: "analysis lead",
          harnessId: "geepilot-codex",
          model: "gpt-5",
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
          model: "gpt-5",
        },
      ],
      agentPlans: [
        {
          id: "desktop-analysis-lead",
          agentId: "analysis-lead",
          agentProfileId: "analysis-lead",
          displayName: "analysis lead",
          canonicalSelector: "@analysis-lead",
          host: "local",
          status: "ready",
          healthStatus: "ready",
          harnessId: "geepilot-codex",
          harnessLabel: "GEEPilot Codex",
          model: "gpt-5",
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
          agentId: "blocked-agent",
          agentProfileId: "blocked-agent",
          displayName: "blocked agent",
          canonicalSelector: "@blocked-agent",
          host: "local",
          status: "blocked",
          healthStatus: "blocked",
          harnessId: "missing-harness",
          model: "gpt-5",
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

import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  Swarm,
  appendMessages,
  createSession,
  deleteSession,
  executeAgentComposition,
  getHarness,
  importN8nWorkflow,
  listGroupedSessions,
  listSessions,
  loadDiscoveredSession,
  loadExtensionInventory,
  loadSession,
  resolveAgentCompositionPlan,
  saveSession,
} from "@swarmx/core";
import type {
  AgentComposition,
  AgentCompositionPlan,
  AgentConfig,
  DiscoveredSession,
  ExtensionInventory,
  ListGroupedSessionsOptions,
  MessageChunk,
  SessionData,
  SwarmConfig,
} from "@swarmx/core";
import { type IpcMainInvokeEvent, ipcMain } from "electron";
import { type LspCompletionRequest, LspHost, type LspStopRequest } from "./lsp-host.js";

const MAX_INLINE_IMAGE_BYTES = 25 * 1024 * 1024;
const lspHost = new LspHost();

export function registerIpcHandlers(): void {
  ipcMain.handle(
    "agent:send",
    async (
      _event: IpcMainInvokeEvent,
      params: {
        harnessId: string;
        userText: string;
        agentConfig?: AgentConfig;
        agentComposition?: AgentComposition;
        swarmConfig?: SwarmConfig;
        sessionId?: string;
      },
    ) => {
      try {
        let swarm: Swarm;

        if (params.swarmConfig) {
          swarm = new Swarm(params.swarmConfig);
        } else if (params.agentComposition) {
          const messages = await executeAgentComposition(params.agentComposition, [
            { role: "user", content: params.userText },
          ]);
          return { success: true, messages };
        } else if (params.agentConfig) {
          const harness = getHarness(params.harnessId);
          if (!harness) throw new Error(`Unknown harness: ${params.harnessId}`);
          swarm = singleAgentSwarm({
            ...params.agentConfig,
            backend: harness.backend,
          });
        } else {
          const harness = getHarness(params.harnessId);
          if (!harness) throw new Error(`Unknown harness: ${params.harnessId}`);
          swarm = new Swarm({
            name: "default",
            root: "agent",
            nodes: {
              agent: {
                kind: "agent",
                agent: {
                  name: "agent",
                  instructions: "You are a helpful AI assistant.",
                },
              },
            },
            edges: [],
          });
        }

        const result = await swarm.execute({
          messages: [{ role: "user", content: params.userText }],
        });

        return { success: true, messages: result };
      } catch (err) {
        return {
          success: false,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    },
  );

  ipcMain.handle(
    "session:create",
    (
      _event: IpcMainInvokeEvent,
      params: { agentName: string; harness: string; model?: string },
    ): SessionData => {
      return createSession(params.agentName, params.harness, params.model);
    },
  );

  ipcMain.handle("session:save", (_event: IpcMainInvokeEvent, session: SessionData): void => {
    saveSession(session);
  });

  ipcMain.handle("session:load", (_event: IpcMainInvokeEvent, id: string): SessionData | null => {
    return loadSession(id);
  });

  ipcMain.handle("session:list", (): SessionData[] => listSessions());

  ipcMain.handle(
    "session:listGrouped",
    (_event: IpcMainInvokeEvent, params?: ListGroupedSessionsOptions) =>
      listGroupedSessions(params ?? {}),
  );

  ipcMain.handle(
    "session:loadDiscovered",
    (_event: IpcMainInvokeEvent, session: DiscoveredSession): Promise<SessionData | null> =>
      loadDiscoveredSession(session),
  );

  ipcMain.handle("session:delete", (_event: IpcMainInvokeEvent, id: string): boolean =>
    deleteSession(id),
  );

  ipcMain.handle(
    "session:appendMessages",
    (_event: IpcMainInvokeEvent, params: { id: string; messages: MessageChunk[] }): boolean =>
      appendMessages(params.id, params.messages),
  );

  ipcMain.handle("workflow:importN8n", (_event: IpcMainInvokeEvent, params: { source: string }) => {
    try {
      const result = importN8nWorkflow(params.source);
      return {
        success: true,
        config: result.config,
        warnings: result.warnings,
        nodeMap: result.nodeMap,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  });

  ipcMain.handle("extension:list", async () => {
    const inventory = await loadExtensionInventory();
    return extensionInventoryWithPlans(inventory);
  });

  ipcMain.handle(
    "lsp:complete",
    async (_event: IpcMainInvokeEvent, params: LspCompletionRequest) => {
      const inventory = await loadExtensionInventory();
      return lspHost.complete(inventory, params);
    },
  );

  ipcMain.handle("lsp:stop", (_event: IpcMainInvokeEvent, params: LspStopRequest) =>
    lspHost.stop(params),
  );

  ipcMain.handle(
    "asset:imageDataUrl",
    async (_event: IpcMainInvokeEvent, source: string): Promise<string | null> =>
      loadImageDataUrl(source),
  );
}

function singleAgentSwarm(agentConfig: AgentConfig): Swarm {
  return new Swarm({
    name: agentConfig.name,
    root: agentConfig.name,
    nodes: {
      [agentConfig.name]: { kind: "agent", agent: agentConfig },
    },
    edges: [],
  });
}

function extensionInventoryWithPlans(
  inventory: ExtensionInventory,
): ExtensionInventory & { agentPlans: AgentCompositionPlan[] } {
  return {
    ...inventory,
    agentPlans: inventory.agents.map((agent) =>
      resolveAgentCompositionPlan(
        {
          id: `desktop-${agent.id}`,
          agentProfileId: agent.id,
          host: "local",
        },
        inventory,
      ),
    ),
  };
}

async function loadImageDataUrl(source: string): Promise<string | null> {
  try {
    const filePath = localFilePathFromSource(source);
    if (!filePath) return null;

    const fileStat = await stat(filePath);
    if (!fileStat.isFile() || fileStat.size > MAX_INLINE_IMAGE_BYTES) return null;

    const bytes = await readFile(filePath);
    if (bytes.byteLength > MAX_INLINE_IMAGE_BYTES) return null;

    const mimeType = detectImageMimeType(bytes);
    if (!mimeType) return null;

    return `data:${mimeType};base64,${bytes.toString("base64")}`;
  } catch {
    return null;
  }
}

function localFilePathFromSource(source: string): string | null {
  const trimmed = source.trim();
  if (!trimmed) return null;

  if (trimmed.startsWith("file://")) {
    try {
      return fileURLToPath(trimmed);
    } catch {
      return null;
    }
  }

  const decoded = safeDecodeUri(trimmed);
  return path.isAbsolute(decoded) ? decoded : null;
}

function safeDecodeUri(value: string): string {
  try {
    return decodeURI(value);
  } catch {
    return value;
  }
}

function detectImageMimeType(bytes: Buffer): string | null {
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return "image/png";
  }

  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }

  const gifHeader = bytes.subarray(0, 6).toString("ascii");
  if (gifHeader === "GIF87a" || gifHeader === "GIF89a") {
    return "image/gif";
  }

  if (
    bytes.length >= 12 &&
    bytes.subarray(0, 4).toString("ascii") === "RIFF" &&
    bytes.subarray(8, 12).toString("ascii") === "WEBP"
  ) {
    return "image/webp";
  }

  return null;
}

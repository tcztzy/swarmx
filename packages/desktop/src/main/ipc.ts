import {
  Agent,
  Swarm,
  appendMessages,
  createSession,
  deleteSession,
  getHarness,
  listGroupedSessions,
  listSessions,
  loadSession,
  saveSession,
} from "@swarmx/core";
import type {
  AgentConfig,
  ListGroupedSessionsOptions,
  MessageChunk,
  SessionData,
  SwarmConfig,
} from "@swarmx/core";
import { type IpcMainInvokeEvent, ipcMain } from "electron";

export function registerIpcHandlers(): void {
  ipcMain.handle(
    "agent:send",
    async (
      _event: IpcMainInvokeEvent,
      params: {
        harnessId: string;
        userText: string;
        agentConfig?: AgentConfig;
        swarmConfig?: SwarmConfig;
        sessionId?: string;
      },
    ) => {
      try {
        const harness = getHarness(params.harnessId);
        if (!harness) throw new Error(`Unknown harness: ${params.harnessId}`);

        let swarm: Swarm;

        if (params.swarmConfig) {
          swarm = new Swarm(params.swarmConfig);
        } else if (params.agentConfig) {
          const agent = new Agent({
            ...params.agentConfig,
            backend: harness.backend,
          });
          swarm = new Swarm({
            name: agent.name,
            root: agent.name,
            nodes: {
              [agent.name]: { kind: "agent", agent: params.agentConfig },
            },
            edges: [],
          });
        } else {
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

  ipcMain.handle("session:delete", (_event: IpcMainInvokeEvent, id: string): boolean =>
    deleteSession(id),
  );

  ipcMain.handle(
    "session:appendMessages",
    (_event: IpcMainInvokeEvent, params: { id: string; messages: MessageChunk[] }): boolean =>
      appendMessages(params.id, params.messages),
  );
}

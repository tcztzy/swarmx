import { EventSchemas } from "@ag-ui/core";
import { type Mock, vi } from "vitest";
import type { AgUiEventMessage } from "../src/bridge-contract.js";
import type { SwarmxBridge } from "../src/renderer/bridge.js";

export interface BridgeHarness {
  readonly bridge: SwarmxBridge;
  readonly bootstrap: Mock;
  readonly tool: Mock;
  readonly cancelTool: Mock;
  readonly settingsRead: Mock;
  readonly settingsUpdate: Mock;
  readonly languageWrite: Mock;
  readonly sessionsList: Mock;
  readonly sessionsCreate: Mock;
  readonly sessionsHistory: Mock;
  readonly modelsRead: Mock;
  readonly logsRead: Mock;
  readonly logsEvidence: Mock;
  readonly runsControl: Mock;
  readonly workRead: Mock;
  readonly workCommand: Mock;
  readonly aguiStart: Mock;
  readonly aguiCancel: Mock;
  emit(threadId: string, ...events: object[]): void;
  emitMessage(message: AgUiEventMessage): void;
}

/** Installs the Electron preload surface over vi mocks for renderer tests. */
export function installBridge(): BridgeHarness {
  const listeners = new Set<(message: AgUiEventMessage) => void>();
  const bootstrap = vi.fn();
  const tool = vi.fn();
  const cancelTool = vi.fn(async () => ({ cancelled: true }));
  const settingsRead = vi.fn();
  const settingsUpdate = vi.fn();
  const languageWrite = vi.fn(async () => ({}));
  const sessionsList = vi.fn(async () => []);
  const sessionsCreate = vi.fn();
  const sessionsHistory = vi.fn(async () => ({ supported: true, messages: [] }));
  const modelsRead = vi.fn(async () => ({ models: [], current: {} }));
  const logsRead = vi.fn(async () => ({ events: [], nextAfter: 0, activeRunIds: [] }));
  const logsEvidence = vi.fn();
  const runsControl = vi.fn(async () => ({}));
  const workRead = vi.fn(async () => ({
    cycles: [],
    snapshot: null,
    activeWorkIds: [],
    interactions: [],
  }));
  const workCommand = vi.fn(async () => ({
    cycles: [],
    snapshot: null,
    activeWorkIds: [],
    interactions: [],
  }));
  const aguiStart = vi.fn(async () => ({}));
  const aguiCancel = vi.fn(async () => ({}));
  const bridge = {
    bootstrap,
    tool,
    cancelTool,
    settings: { read: settingsRead, update: settingsUpdate },
    language: { write: languageWrite },
    sessions: { list: sessionsList, create: sessionsCreate, history: sessionsHistory },
    models: { read: modelsRead },
    logs: { read: logsRead, evidence: logsEvidence },
    runs: { control: runsControl },
    work: { read: workRead, command: workCommand },
    agui: {
      start: aguiStart,
      cancel: aguiCancel,
      subscribe: (listener: (message: AgUiEventMessage) => void) => {
        listeners.add(listener);
        return () => {
          listeners.delete(listener);
        };
      },
    },
  } as unknown as SwarmxBridge;
  (window as unknown as { swarmx?: SwarmxBridge }).swarmx = bridge;
  return {
    bridge,
    bootstrap,
    tool,
    cancelTool,
    settingsRead,
    settingsUpdate,
    languageWrite,
    sessionsList,
    sessionsCreate,
    sessionsHistory,
    modelsRead,
    logsRead,
    logsEvidence,
    runsControl,
    workRead,
    workCommand,
    aguiStart,
    aguiCancel,
    emit(threadId: string, ...events: object[]) {
      for (const event of events)
        this.emitMessage({
          threadId,
          event: EventSchemas.parse({ timestamp: Date.now(), ...event }),
        });
    },
    emitMessage(message: AgUiEventMessage) {
      for (const listener of [...listeners]) listener(message);
    },
  };
}

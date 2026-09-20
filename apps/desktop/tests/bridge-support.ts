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
  readonly environmentRead: Mock;
  readonly environmentAct: Mock;
  readonly sessionsList: Mock;
  readonly sessionsCreate: Mock;
  readonly sessionsHistory: Mock;
  readonly modelsRead: Mock;
  readonly logsRead: Mock;
  readonly logsEvidence: Mock;
  readonly runsControl: Mock;
  readonly scienceWorkspace: Mock;
  readonly scienceResearchObject: Mock;
  readonly scienceNotebookExecutions: Mock;
  readonly scienceArtifactPreview: Mock;
  readonly scienceArtifactContent: Mock;
  readonly scienceImport: Mock;
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
  const environmentRead = vi.fn();
  const environmentAct = vi.fn();
  const sessionsList = vi.fn(async () => []);
  const sessionsCreate = vi.fn();
  const sessionsHistory = vi.fn(async () => []);
  const modelsRead = vi.fn(async () => ({ models: [], current: {} }));
  const logsRead = vi.fn(async () => ({ events: [], nextAfter: 0, activeRunIds: [] }));
  const logsEvidence = vi.fn();
  const runsControl = vi.fn(async () => ({}));
  const scienceWorkspace = vi.fn();
  const scienceResearchObject = vi.fn();
  const scienceNotebookExecutions = vi.fn(async () => []);
  const scienceArtifactPreview = vi.fn();
  const scienceArtifactContent = vi.fn();
  const scienceImport = vi.fn();
  const aguiStart = vi.fn(async () => ({}));
  const aguiCancel = vi.fn(async () => ({}));
  const bridge = {
    bootstrap,
    tool,
    cancelTool,
    settings: { read: settingsRead, update: settingsUpdate },
    language: { write: languageWrite },
    environment: { read: environmentRead, act: environmentAct },
    sessions: { list: sessionsList, create: sessionsCreate, history: sessionsHistory },
    models: { read: modelsRead },
    logs: { read: logsRead, evidence: logsEvidence },
    runs: { control: runsControl },
    science: {
      workspace: scienceWorkspace,
      researchObject: scienceResearchObject,
      notebookExecutions: scienceNotebookExecutions,
      artifactPreview: scienceArtifactPreview,
      artifactContent: scienceArtifactContent,
      import: scienceImport,
    },
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
    environmentRead,
    environmentAct,
    sessionsList,
    sessionsCreate,
    sessionsHistory,
    modelsRead,
    logsRead,
    logsEvidence,
    runsControl,
    scienceWorkspace,
    scienceResearchObject,
    scienceNotebookExecutions,
    scienceArtifactPreview,
    scienceArtifactContent,
    scienceImport,
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

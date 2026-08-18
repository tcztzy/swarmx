import type {
  SessionInfo as AcpSessionInfo,
  RequestPermissionRequest,
  RequestPermissionResponse,
} from "@agentclientprotocol/sdk";
import type { MediaAttachment, MessageChunk } from "./types.js";

export interface HarnessLaunchRequest {
  command: string;
  args: readonly string[];
  /** Optional wire-transport selector; defaults to the command token. */
  transport?: string;
}

export interface HarnessLaunchSpec {
  command: string;
  args: string[];
  env: Record<string, string>;
}

export type HarnessLaunchResolver = (request: HarnessLaunchRequest) => HarnessLaunchSpec;

export type HarnessPermissionRequest = RequestPermissionRequest;
export type HarnessPermissionResponse = RequestPermissionResponse;
export type HarnessPermissionHandler = (
  request: HarnessPermissionRequest,
) => Promise<HarnessPermissionResponse>;

export type HarnessPromptInput =
  | string
  | {
      text: string;
      attachments?: readonly MediaAttachment[];
    };

export interface HarnessPromptClientOptions {
  command: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
  clearEnv?: boolean;
  model?: string;
  effort?: string;
  preferredMode?: string;
  requestPermission?: HarnessPermissionHandler;
  onSessionId?: (sessionId: string) => void | Promise<void>;
  signal?: AbortSignal;
}

export interface HarnessPromptClient {
  prompt(
    options: HarnessPromptClientOptions,
    input: HarnessPromptInput,
    swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<{ messages: MessageChunk[] }>;
  stderrOutput?(): string;
}

export interface HarnessSessionClient {
  listSessions(options: HarnessPromptClientOptions, cwd?: string): Promise<AcpSessionInfo[]>;
  loadSession(
    options: HarnessPromptClientOptions,
    sessionId: string,
    cwd: string,
  ): Promise<{ messages: MessageChunk[] }>;
  stderrOutput(): string;
  kill(): void;
}

export interface HarnessApprovalRequest {
  sessionId: string;
  toolCall: {
    toolCallId: string;
    kind: "execute" | "edit" | "other";
    status: "pending";
    title?: string;
    content?: { type: string; text: string }[];
    rawInput: unknown;
  };
  options: Array<{
    optionId: string;
    name: string;
    kind: "allow_once" | "allow_always" | "reject_once";
  }>;
}

export type HarnessApprovalOutcome =
  | { outcome: "cancelled" }
  | { outcome: "selected"; optionId: string }
  | { outcome: "approved" }
  | { outcome: "rejected" };

export type HarnessApprovalResolver = (
  request: HarnessApprovalRequest,
) => Promise<HarnessApprovalOutcome>;

export type HarnessClient = HarnessPromptClient & Partial<HarnessSessionClient>;

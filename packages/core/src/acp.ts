import { type ChildProcess, spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import type {
  SessionInfo as AcpSessionInfo,
  Client,
  ClientSideConnection,
  ListSessionsRequest,
  LoadSessionRequest,
  LoadSessionResponse,
  NewSessionRequest,
  PromptRequest,
  RequestPermissionRequest,
  RequestPermissionResponse,
  SessionNotification,
  SessionUpdate,
} from "@agentclientprotocol/sdk";
import { SWARMX_VERSION } from "./version.js";

let _acp: typeof import("@agentclientprotocol/sdk") | null = null;

async function loadAcp(): Promise<typeof import("@agentclientprotocol/sdk")> {
  if (!_acp) {
    _acp = await import("@agentclientprotocol/sdk");
  }
  return _acp;
}

export interface AcpClientOptions {
  command: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
  clearEnv?: boolean;
}

export interface AcpPromptResult {
  sessionId: string;
  messages: MessageChunk[];
  stopReason: string;
}

export interface MessageChunk {
  role: string;
  content: string;
  kind: "message" | "thinking" | "tool_call" | "tool_result";
  agent?: string;
  swarmEvent?: string;
  toolName?: string;
}

export class AcpClient {
  private child: ChildProcess | null = null;
  private stderr = "";

  private buildEnv(opts: AcpClientOptions): Record<string, string> {
    const env: Record<string, string> = {};
    if (!opts.clearEnv) {
      Object.assign(env, process.env as Record<string, string>);
    }
    if (opts.env) {
      Object.assign(env, opts.env);
    }
    return env;
  }

  private async spawnAndConnect(
    opts: AcpClientOptions,
    onSessionUpdate: (update: SessionUpdate) => void,
  ): Promise<{
    connection: ClientSideConnection;
    acp: typeof import("@agentclientprotocol/sdk");
  }> {
    const acp = await loadAcp();
    const env = this.buildEnv(opts);

    const child = spawn(opts.command, opts.args, {
      cwd: opts.cwd,
      env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    this.child = child;
    this.stderr = "";

    child.stderr?.setEncoding("utf-8");
    child.stderr?.on("data", (chunk: string) => {
      this.stderr = `${this.stderr}${chunk}`.slice(-4000);
    });

    if (!child.stdin || !child.stdout) {
      throw new Error("ACP child process stdio is unavailable");
    }

    const transport = acp.ndJsonStream(Writable.toWeb(child.stdin), Readable.toWeb(child.stdout));

    const clientStubs: Client = {
      async requestPermission(
        _request: RequestPermissionRequest,
      ): Promise<RequestPermissionResponse> {
        return { outcome: { outcome: "cancelled" } };
      },
      async sessionUpdate(notification: SessionNotification): Promise<void> {
        onSessionUpdate(notification.update);
      },
    };

    const connection = new acp.ClientSideConnection(() => clientStubs, transport);

    return { connection, acp };
  }

  async prompt(
    opts: AcpClientOptions,
    userText: string,
    swarmConfig?: unknown,
    sessionId?: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<AcpPromptResult> {
    const chunks: MessageChunk[] = [];

    const result = await this.spawnAndConnect(opts, (update) => {
      const msg = sessionUpdateToChunk(update);
      if (msg) {
        chunks.push(msg);
        onChunk?.(msg);
      }
    });
    const { connection, acp } = result;

    try {
      await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });

      let sid: string;
      if (sessionId) {
        sid = sessionId;
      } else {
        const resp = await connection.newSession({
          cwd: opts.cwd ?? process.cwd(),
          mcpServers: [],
        });
        sid = resp.sessionId;
      }

      const meta: Record<string, unknown> = {};
      if (swarmConfig) {
        meta.swarmConfig = swarmConfig;
      }

      const promptBlock: PromptRequest["prompt"][number] = {
        type: "text",
        text: userText,
        ...(Object.keys(meta).length > 0 ? { _meta: meta } : {}),
      };

      const promptReq: PromptRequest = {
        sessionId: sid,
        prompt: [promptBlock],
      };

      const promptResp = await connection.prompt(promptReq);
      await connection.closed;

      return {
        sessionId: sid,
        messages: mergeChunks(chunks),
        stopReason: promptResp.stopReason ?? "end_turn",
      };
    } finally {
      this.kill();
    }
  }

  async listSessions(opts: AcpClientOptions, cwd?: string): Promise<AcpSessionInfo[]> {
    const result = await this.spawnAndConnect(opts, () => {});
    const { connection, acp } = result;

    try {
      await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });

      const req: ListSessionsRequest = cwd ? { cwd } : {};
      const resp = await connection.listSessions(req);
      return resp.sessions ?? [];
    } finally {
      this.kill();
    }
  }

  async loadSession(
    opts: AcpClientOptions,
    sessionId: string,
    cwd: string,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<{ response: LoadSessionResponse; messages: MessageChunk[] }> {
    const chunks: MessageChunk[] = [];

    const result = await this.spawnAndConnect(opts, (update) => {
      const msg = sessionUpdateToChunk(update);
      if (msg) {
        chunks.push(msg);
        onChunk?.(msg);
      }
    });
    const { connection, acp } = result;

    try {
      await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });

      const req: LoadSessionRequest = {
        sessionId,
        cwd,
        mcpServers: [],
      };
      const resp = await connection.loadSession(req);
      return { response: resp, messages: mergeChunks(chunks) };
    } finally {
      this.kill();
    }
  }

  async newSession(opts: AcpClientOptions, cwd: string): Promise<string> {
    const result = await this.spawnAndConnect(opts, () => {});
    const { connection, acp } = result;

    try {
      await connection.initialize({
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: {},
        clientInfo: { name: "swarmx", title: "SwarmX", version: SWARMX_VERSION },
      });

      const req: NewSessionRequest = { cwd, mcpServers: [] };
      const resp = await connection.newSession(req);
      return resp.sessionId;
    } finally {
      this.kill();
    }
  }

  kill(): void {
    if (this.child) {
      this.child.kill();
      this.child = null;
    }
  }

  stderrOutput(): string {
    return this.stderr.trim();
  }
}

function sessionUpdateToChunk(update: SessionUpdate): MessageChunk | null {
  const u = update as Record<string, unknown>;
  const updateKind = stringValue(u.sessionUpdate) ?? stringValue(u.updateType);

  switch (updateKind) {
    case "user_message_chunk":
    case "agent_message_chunk": {
      const content = (u.content as Record<string, unknown> | undefined) ?? {};
      const text = String(content.text ?? "");
      if (!text) return null;
      const meta = recordValue(content._meta) ?? recordValue(content.meta) ?? {};
      return {
        role:
          stringValue(meta.role) ?? (updateKind === "user_message_chunk" ? "user" : "assistant"),
        content: text,
        kind: "message",
        agent: stringValue(meta.agent),
        swarmEvent: stringValue(meta.swarmEvent),
      };
    }
    case "agent_thought_chunk": {
      const content = (u.content as Record<string, unknown> | undefined) ?? {};
      const text = String(content.text ?? "");
      if (!text) return null;
      return { role: "assistant", content: text, kind: "thinking" };
    }
    case "tool_call": {
      const args = u.rawInput ? JSON.stringify(u.rawInput) : "";
      return {
        role: "assistant",
        content: args,
        kind: "tool_call",
        toolName: stringValue(u.title),
      };
    }
    case "tool_call_update": {
      const fields = recordValue(u.fields);
      const rawOutput = u.rawOutput ?? fields?.rawOutput;
      const status = u.status ?? fields?.status;
      const result =
        (rawOutput ? JSON.stringify(rawOutput) : "") || (status ? JSON.stringify(status) : "");
      if (!result) return null;
      return {
        role: "assistant",
        content: result,
        kind: "tool_result",
        toolName: stringValue(u.title) ?? stringValue(fields?.title) ?? "tool",
      };
    }
    default:
      return null;
  }
}

function recordValue(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function mergeChunks(chunks: MessageChunk[]): MessageChunk[] {
  function key(c: MessageChunk): string {
    return `${c.role ?? ""}|${c.agent ?? ""}|${c.swarmEvent ?? ""}|${c.kind}`;
  }

  const merged: MessageChunk[] = [];
  for (const chunk of chunks) {
    const ck = key(chunk);
    const last = merged[merged.length - 1];
    if (last && key(last) === ck) {
      last.content += chunk.content;
    } else {
      merged.push({ ...chunk });
    }
  }
  return merged;
}

import { randomUUID } from "node:crypto";
import http from "node:http";
import type { AuditInput } from "./audit.js";
import type { MessageChunk } from "./types.js";

export interface ServerSwarmExecution {
  readonly models: readonly { id: string; object: "model" }[];
  execute(
    arguments_: Record<string, unknown>,
    context?: Record<string, unknown>,
    onChunk?: (chunk: MessageChunk) => void,
  ): Promise<MessageChunk[]>;
  listAllSessions(cwd?: string): Promise<unknown[]>;
}

export interface ServerAuditWriter {
  append(input: AuditInput): unknown | Promise<unknown>;
}

export interface ServerOptions {
  port?: number;
  host?: string;
  apiToken?: string;
  allowedOrigins?: string[];
  allowNullOrigin?: boolean;
  audit?: ServerAuditWriter;
}

export function createServer(swarm: ServerSwarmExecution, opts: ServerOptions = {}): http.Server {
  const port = opts.port ?? 3000;
  const host = opts.host ?? "127.0.0.1";
  const boundary = resolveServerBoundary(opts, host);

  const server = http.createServer(async (req, res) => {
    const requestId = randomUUID();
    const startedAt = Date.now();
    const method = req.method ?? "UNKNOWN";
    const path = resolveAuditPath(req.url, host, port);

    res.setHeader("X-Request-ID", requestId);
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.setHeader("Vary", "Origin");

    if (
      !(await recordRequestAudit(opts.audit, {
        requestId,
        method,
        path,
        outcome: "attempted",
        startedAt,
      }))
    ) {
      writeAuditUnavailable(res);
      return;
    }

    const originPolicy = evaluateOriginPolicy(req, boundary);
    if (!originPolicy.allowed) {
      if (
        !(await recordRequestAudit(opts.audit, {
          requestId,
          method,
          path,
          outcome: "denied",
          statusCode: 403,
          startedAt,
        }))
      ) {
        writeAuditUnavailable(res);
        return;
      }
      writeJson(res, 403, { error: "Origin not allowed" });
      return;
    }
    if (originPolicy.responseOrigin) {
      res.setHeader("Access-Control-Allow-Origin", originPolicy.responseOrigin);
    }

    if (req.method === "OPTIONS") {
      if (
        !(await recordRequestAudit(opts.audit, {
          requestId,
          method,
          path,
          outcome: "completed",
          statusCode: 204,
          startedAt,
        }))
      ) {
        writeAuditUnavailable(res);
        return;
      }
      res.writeHead(204);
      res.end();
      return;
    }

    if (!isAuthorized(req, boundary)) {
      if (
        !(await recordRequestAudit(opts.audit, {
          requestId,
          method,
          path,
          outcome: "denied",
          statusCode: 401,
          startedAt,
        }))
      ) {
        writeAuditUnavailable(res);
        return;
      }
      writeJson(res, 401, { error: "Unauthorized" });
      return;
    }

    const url = new URL(req.url ?? "/", `http://${host}:${port}`);

    try {
      if (req.method === "GET" && url.pathname === "/models") {
        const data = listModels(swarm);
        if (
          !(await recordRequestAudit(opts.audit, {
            requestId,
            method,
            path,
            outcome: "completed",
            statusCode: 200,
            startedAt,
          }))
        ) {
          writeAuditUnavailable(res);
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ object: "list", data }));
        return;
      }

      if (req.method === "GET" && url.pathname === "/sessions") {
        const sessions = await swarm.listAllSessions();
        if (
          !(await recordRequestAudit(opts.audit, {
            requestId,
            method,
            path,
            outcome: "completed",
            statusCode: 200,
            startedAt,
          }))
        ) {
          writeAuditUnavailable(res);
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(sessions));
        return;
      }

      if (req.method === "POST" && url.pathname === "/chat/completions") {
        const body = await readBody(req);

        if (body.stream) {
          const streamCompleted = await handleStream(req, res, swarm, body);
          if (
            !(await recordRequestAudit(opts.audit, {
              requestId,
              method,
              path,
              outcome: streamCompleted ? "completed" : "failed",
              statusCode: 200,
              startedAt,
            }))
          ) {
            writeAuditUnavailable(res);
            return;
          }
          res.write("data: [DONE]\n\n");
          res.end();
          return;
        }

        const result = await handleChat(swarm, body);
        if (
          !(await recordRequestAudit(opts.audit, {
            requestId,
            method,
            path,
            outcome: "completed",
            statusCode: 200,
            startedAt,
          }))
        ) {
          writeAuditUnavailable(res);
          return;
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
        return;
      }

      if (
        !(await recordRequestAudit(opts.audit, {
          requestId,
          method,
          path,
          outcome: "completed",
          statusCode: 404,
          startedAt,
        }))
      ) {
        writeAuditUnavailable(res);
        return;
      }
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
    } catch (err) {
      if (
        !(await recordRequestAudit(opts.audit, {
          requestId,
          method,
          path,
          outcome: "failed",
          statusCode: 500,
          startedAt,
        }))
      ) {
        writeAuditUnavailable(res);
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      if (!res.headersSent) {
        res.writeHead(500, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: message }));
    }
  });

  server.listen(port, host);
  return server;
}

interface ServerBoundary {
  requiresAuth: boolean;
  apiToken?: string;
  allowedOrigins: Set<string>;
  allowNullOrigin: boolean;
}

function resolveServerBoundary(opts: ServerOptions, host: string): ServerBoundary {
  if (!isLoopbackHost(host) && !opts.apiToken) {
    throw new Error("Non-loopback SwarmX server bindings require opts.apiToken.");
  }
  if (opts.allowedOrigins?.includes("*")) {
    throw new Error(
      "SwarmX server allowedOrigins must be explicit; wildcard origins are rejected.",
    );
  }

  return {
    requiresAuth: !!opts.apiToken,
    apiToken: opts.apiToken,
    allowedOrigins: new Set(opts.allowedOrigins ?? []),
    allowNullOrigin: opts.allowNullOrigin ?? false,
  };
}

function evaluateOriginPolicy(
  req: http.IncomingMessage,
  boundary: ServerBoundary,
): { allowed: boolean; responseOrigin?: string } {
  const origin = req.headers.origin;
  if (!origin) return { allowed: true };

  if (origin === "null") {
    if (boundary.allowNullOrigin) {
      return { allowed: true, responseOrigin: "null" };
    }
    return { allowed: false };
  }

  if (boundary.allowedOrigins.has(origin)) {
    return { allowed: true, responseOrigin: origin };
  }

  return { allowed: false };
}

function isAuthorized(req: http.IncomingMessage, boundary: ServerBoundary): boolean {
  if (!boundary.requiresAuth) return true;

  const authorization = req.headers.authorization;
  return authorization === `Bearer ${boundary.apiToken}`;
}

interface RequestAuditDetails {
  requestId: string;
  method: string;
  path: string;
  outcome: "attempted" | "completed" | "denied" | "failed";
  statusCode?: number;
  startedAt: number;
}

async function recordRequestAudit(
  writer: ServerAuditWriter | undefined,
  details: RequestAuditDetails,
): Promise<boolean> {
  if (!writer) return true;

  try {
    const metadata: Record<string, string | number> = {
      method: details.method,
      path: details.path,
      durationMs: Math.max(0, Date.now() - details.startedAt),
    };
    if (details.statusCode !== undefined) metadata.statusCode = details.statusCode;

    await writer.append({
      category: "system",
      action: "http.request",
      outcome: details.outcome,
      actor: { kind: "service", id: "swarmx-http" },
      requestId: details.requestId,
      metadata,
    });
    return true;
  } catch {
    return false;
  }
}

function resolveAuditPath(rawUrl: string | undefined, host: string, port: number): string {
  try {
    const pathname = new URL(rawUrl ?? "/", `http://${host}:${port}`).pathname;
    if (["/models", "/sessions", "/chat/completions"].includes(pathname)) return pathname;
  } catch {
    // Malformed and unknown routes share one non-sensitive audit bucket.
  }
  return "/:unmatched";
}

function writeAuditUnavailable(res: http.ServerResponse): void {
  if (!res.headersSent && !res.writableEnded) {
    writeJson(res, 503, { error: "Audit log unavailable" });
    return;
  }
  if (!res.writableEnded) res.destroy();
}

function writeJson(res: http.ServerResponse, statusCode: number, body: unknown): void {
  res.writeHead(statusCode, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function isLoopbackHost(host: string): boolean {
  const normalized = host.toLowerCase().replace(/^\[(.*)\]$/, "$1");
  return (
    normalized === "localhost" ||
    normalized === "::1" ||
    normalized === "0:0:0:0:0:0:0:1" ||
    normalized.startsWith("127.")
  );
}

// ── SSE streaming ───────────────────────────────────────────────────────────

async function handleStream(
  _req: http.IncomingMessage,
  res: http.ServerResponse,
  swarm: ServerSwarmExecution,
  body: ChatCompletionRequest,
): Promise<boolean> {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const id = `chatcmpl-${Date.now()}`;
  const model = body.model ?? "swarmx";
  const created = Math.floor(Date.now() / 1000);

  try {
    await streamViaSwarm(id, created, model, res, swarm, body);

    return true;
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : String(err);
    if (!res.closed) {
      res.write(`data: ${JSON.stringify({ error: errorMsg })}\n\n`);
    }
    return false;
  }
}

async function streamViaSwarm(
  id: string,
  created: number,
  model: string,
  res: http.ServerResponse,
  swarm: ServerSwarmExecution,
  body: ChatCompletionRequest,
): Promise<void> {
  let streamedMessageChunks = 0;
  const messages = await swarm.execute({ messages: body.messages }, undefined, (chunk) => {
    if (res.closed || chunk.kind !== "message") return;
    streamedMessageChunks += 1;
    const streamChunk: ChatCompletionChunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [
        {
          index: 0,
          delta: { role: chunk.role === "user" ? "user" : "assistant", content: chunk.content },
          finish_reason: null,
        },
      ],
    };
    res.write(`data: ${JSON.stringify(streamChunk)}\n\n`);
  });

  if (streamedMessageChunks === 0) {
    for (const msg of messages) {
      if (msg.kind !== "message") continue;
      const chunk: ChatCompletionChunk = {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: [
          {
            index: 0,
            delta: { role: msg.role === "user" ? "user" : "assistant", content: msg.content },
            finish_reason: null,
          },
        ],
      };
      res.write(`data: ${JSON.stringify(chunk)}\n\n`);
    }
  }

  if (!res.closed) {
    const final: ChatCompletionChunk = {
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    };
    res.write(`data: ${JSON.stringify(final)}\n\n`);
  }
}

// ── Non-streaming ───────────────────────────────────────────────────────────

async function handleChat(
  swarm: ServerSwarmExecution,
  body: ChatCompletionRequest,
): Promise<ChatCompletionResponse> {
  const messages = await swarm.execute({ messages: body.messages });
  const content = messages.map((m) => m.content).join("\n");

  return {
    id: `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: body.model ?? "swarmx",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
  };
}

// ── Models ──────────────────────────────────────────────────────────────────

function listModels(swarm: ServerSwarmExecution): readonly { id: string; object: "model" }[] {
  return swarm.models;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function readBody(req: http.IncomingMessage): Promise<ChatCompletionRequest> {
  return new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => {
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(e);
      }
    });
    req.on("error", reject);
  });
}

// ── Types ───────────────────────────────────────────────────────────────────

interface ChatCompletionRequest {
  model?: string;
  messages: Array<{ role: string; content: string }>;
  stream?: boolean;
}

interface ChatCompletionResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: string; content: string };
    finish_reason: string;
  }>;
}

interface ChatCompletionChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: { role?: string; content?: string };
    finish_reason: string | null;
  }>;
}

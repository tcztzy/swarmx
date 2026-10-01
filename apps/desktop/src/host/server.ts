import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { realpath } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { ZodError, z } from "zod";
import type { AgentId } from "../agent.js";
import type { NativeAgent } from "../agents/types.js";
import { startMcpSocket } from "./mcp-socket.js";
import type { ProductServices } from "./product-services.js";

const HOST = "127.0.0.1";
const BODY_LIMIT = 1024 * 1024;
const A2ADirectory = z.object({
  params: z
    .object({
      message: z
        .object({
          metadata: z
            .object({
              swarmx: z.object({ directory: z.string().min(1).optional() }).optional(),
            })
            .optional(),
        })
        .optional(),
    })
    .optional(),
});

export interface StartHostOptions {
  readonly agent?: NativeAgent | undefined;
  readonly agentId?: AgentId | undefined;
  readonly products: ProductServices;
}

export interface SwarmXHost {
  readonly products: ProductServices;
  readonly signal: AbortSignal;
  readonly token: string;
  readonly origin: string;
  dispose(): Promise<void>;
}

export async function startHost(options: StartHostOptions): Promise<SwarmXHost> {
  const { products } = options;
  const token = process.env.SWARMX_API_TOKEN ?? secret();
  const shutdown = new AbortController();
  const operations = new Set<Promise<unknown>>();
  let closing = false;
  const server = createServer((request, response) => {
    const operation = route(request, response, products, token);
    operations.add(operation);
    void operation
      .catch((error: unknown) => sendError(response, error))
      .finally(() => operations.delete(operation));
  });
  await new Promise<void>((done, reject) => {
    server.once("error", reject);
    server.listen(0, HOST, done);
  });
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("Host has no TCP address.");
  const origin = `http://${HOST}:${String(address.port)}`;
  const socket = await (async () => {
    try {
      await products.attachAgents(origin, options.agent, options.agentId);
      const socket = await startMcpSocket(products.mcpSocket, {
        tools: products.toolManifest,
        known: (credential) =>
          credential === products.openclawToken || products.mcpExecutions.has(credential),
        invoke: (request, signal) => {
          const bound = products.resolveMcpCall(request);
          return products.callTool(request.tool, request.args, {
            actorId: bound.sessionId,
            callId: request.toolCallId ?? randomUUID(),
            signal,
            sessionId: bound.sessionId,
            runId: bound.runId,
          });
        },
      });
      await products.publishOpenClawBridge();
      return socket;
    } catch (error) {
      server.closeAllConnections();
      await new Promise<void>((done) => server.close(() => done()));
      throw error;
    }
  })();
  return {
    products,
    signal: shutdown.signal,
    token,
    origin,
    async dispose() {
      if (closing) return;
      closing = true;
      shutdown.abort(new Error("SwarmX Host is closing."));
      server.closeAllConnections();
      await new Promise<void>((done, reject) =>
        server.close((error) => (error === undefined ? done() : reject(error))),
      );
      const results = await Promise.allSettled([
        ...operations,
        products.dispose(),
        socket.dispose(),
      ]);
      for (const result of results.slice(-2)) if (result.status === "rejected") throw result.reason;
    },
  };
}

async function route(
  request: IncomingMessage,
  response: ServerResponse,
  products: ProductServices,
  token: string,
): Promise<void> {
  const url = new URL(request.url ?? "/", "http://127.0.0.1");
  const match = /^\/a2a\/([^/]+)(\/\.well-known\/agent-card\.json)?$/u.exec(url.pathname);
  if (!match?.[1]) throw new HttpError(404, "Not found.");
  const agentId = decodeURIComponent(match[1]);
  if (!products.hasA2A(agentId))
    throw new HttpError(404, `A2A Agent "${agentId}" is not available. Create it first.`);
  if (request.method === "GET" && match[2] === "/.well-known/agent-card.json") {
    sendJson(response, 200, products.a2aCard(agentId));
    return;
  }
  if (request.method !== "POST" || match[2] !== undefined)
    throw new HttpError(405, "Method not allowed.");
  authorizeBearer(request, token);
  const version = request.headers["a2a-version"];
  if (typeof version !== "string") throw new HttpError(400, "A2A-Version is required.");
  const body = z.record(z.string(), z.unknown()).parse(await readJson(request));
  const directory = A2ADirectory.parse(body).params?.message?.metadata?.swarmx?.directory;
  if (directory !== undefined && (await realpath(directory)) !== products.options.cwd)
    throw new HttpError(400, "The requested directory does not match the Host working directory.");
  sendJson(response, 200, await products.handleA2A(agentId, body, version));
}

async function readJson(request: IncomingMessage, limit = BODY_LIMIT): Promise<unknown> {
  if (!(request.headers["content-type"] ?? "").startsWith("application/json")) {
    throw new HttpError(415, "Expected an application/json body.");
  }
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.from(chunk);
    bytes += buffer.byteLength;
    if (bytes > limit) throw new HttpError(413, "Request body is too large.");
    chunks.push(buffer);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch (error) {
    throw new HttpError(400, "Request body is not valid JSON.", { cause: error });
  }
}

function authorizeBearer(request: IncomingMessage, token: string): void {
  if (!safeEqual(request.headers.authorization?.replace(/^Bearer /u, ""), token)) {
    throw new HttpError(403, "Invalid Host bearer token.");
  }
}

function safeEqual(left: string | undefined, right: string): boolean {
  if (left === undefined) return false;
  const leftBytes = Buffer.from(left);
  const rightBytes = Buffer.from(right);
  return leftBytes.length === rightBytes.length && timingSafeEqual(leftBytes, rightBytes);
}

function secret(): string {
  return randomBytes(32).toString("base64url");
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  if (response.headersSent || response.destroyed) return;
  response.statusCode = status;
  response.setHeader("content-type", "application/json; charset=utf-8");
  response.end(`${JSON.stringify(value)}\n`);
}

function sendError(response: ServerResponse, error: unknown): void {
  if (response.headersSent || response.destroyed) {
    if (!response.destroyed) response.end();
    return;
  }
  const status = error instanceof HttpError ? error.status : error instanceof ZodError ? 400 : 500;
  let message = error instanceof Error ? error.message : String(error);
  const detail = z
    .object({
      data: z.union([
        z.string(),
        z.object({ message: z.string() }).transform((value) => value.message),
        z.object({ details: z.string() }).transform((value) => value.details),
      ]),
    })
    .safeParse(error);
  if (detail.success && detail.data.data) message += `: ${detail.data.data}`;
  sendJson(response, status, {
    error: message,
    ...(error instanceof ZodError ? { issues: error.issues } : {}),
  });
}

export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

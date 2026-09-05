import { randomBytes, randomUUID, timingSafeEqual } from "node:crypto";
import { createReadStream, realpathSync, statSync } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { extname, relative, resolve, sep } from "node:path";
import { ZodError, z } from "zod";
import { loadAgUiHistory } from "./ag-ui.js";
import { handleMcp } from "./mcp.js";
import type { ProductServices, Workspace } from "./product-services.js";

const HOST = "127.0.0.1";
const COOKIE = "swarmx_session";
const BODY_LIMIT = 1024 * 1024;
const LAUNCH_TTL_MS = 60_000;
const DvcMutation = z.discriminatedUnion("action", [
  z.strictObject({
    action: z.literal("pull"),
    request: z.strictObject({
      remote: z.string().optional(),
      targets: z.array(z.string()).max(32).optional(),
    }),
  }),
  z.strictObject({
    action: z.literal("reproduce"),
    request: z.strictObject({
      pull: z.boolean().optional(),
      remote: z.string().optional(),
      targets: z.array(z.string()).max(32).optional(),
    }),
  }),
]);

const CONTENT_TYPES: Readonly<Record<string, string>> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

export interface StartHostOptions {
  readonly products: ProductServices;
  readonly rendererRoot: string;
  readonly workspace: Workspace;
}

export interface SwarmXHost {
  readonly internalToken: string;
  readonly internalUrl: string;
  issueLaunchUrl(): string;
  dispose(): Promise<void>;
}

export async function startHost(options: StartHostOptions): Promise<SwarmXHost> {
  const rendererRoot = realpathSync(options.rendererRoot);
  if (!statSync(rendererRoot).isDirectory())
    throw new Error("Renderer build directory is missing.");
  const sessionToken = secret();
  const internalToken = process.env.SWARMX_API_TOKEN ?? secret();
  const launchTokens = new Map<string, number>();
  const shutdown = new AbortController();
  const operations = new Set<Promise<void>>();
  let origin = "";
  let hostHeader = "";
  let closing = false;
  const server = createServer((request, response) => {
    securityHeaders(response);
    const operation = route(request, response, {
      ...options,
      rendererRoot,
      sessionToken,
      internalToken,
      launchTokens,
      origin,
      hostHeader,
      signal: shutdown.signal,
    });
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
  hostHeader = `${HOST}:${String(address.port)}`;
  origin = `http://${hostHeader}`;
  return {
    internalToken,
    internalUrl: origin,
    issueLaunchUrl() {
      if (closing) throw new Error("SwarmX Host is closing.");
      const token = secret();
      launchTokens.set(token, Date.now() + LAUNCH_TTL_MS);
      return `${origin}/?token=${encodeURIComponent(token)}`;
    },
    async dispose() {
      if (closing) return;
      closing = true;
      shutdown.abort(new Error("SwarmX Host is closing."));
      server.closeAllConnections();
      await new Promise<void>((done, reject) =>
        server.close((error) => (error === undefined ? done() : reject(error))),
      );
      await Promise.allSettled([...operations]);
      launchTokens.clear();
    },
  };
}

interface RouteContext extends StartHostOptions {
  readonly rendererRoot: string;
  readonly sessionToken: string;
  readonly internalToken: string;
  readonly launchTokens: Map<string, number>;
  readonly origin: string;
  readonly hostHeader: string;
  readonly signal: AbortSignal;
}

async function route(
  request: IncomingMessage,
  response: ServerResponse,
  context: RouteContext,
): Promise<void> {
  if (request.headers.host !== context.hostHeader) throw new HttpError(403, "Invalid Host header.");
  const url = new URL(request.url ?? "/", context.origin);
  if (url.pathname.startsWith("/a2a/")) {
    await a2aRoute(request, response, url, context);
    return;
  }
  if (url.pathname === "/mcp") {
    authorizeBearer(request, context.internalToken);
    await handleMcp(request, response, context.products.toolManifest, (name, args, signal) =>
      context.products.callTool(name, args, {
        actorId: "agent",
        callId: randomUUID(),
        signal,
      }),
    );
    return;
  }
  if (exchangeLaunchToken(request, response, url, context)) return;
  if (!safeEqual(cookie(request, COOKIE), context.sessionToken)) {
    throw new HttpError(401, "SwarmX browser session is required.");
  }
  if (!validOrigin(request, context.origin)) throw new HttpError(403, "Invalid Origin header.");
  if (url.pathname.startsWith("/api/")) {
    response.setHeader("cache-control", "no-store");
    await apiRoute(request, response, url, context);
    return;
  }
  await serveStatic(request, response, url, context.rendererRoot);
}

async function a2aRoute(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): Promise<void> {
  const match = /^\/a2a\/([^/]+)(?:\/\.well-known\/agent-card\.json)?$/u.exec(url.pathname);
  if (match?.[1] === undefined) throw new HttpError(404, "A2A Agent not found.");
  const id = decodeURIComponent(match[1]);
  if (request.method === "GET" && url.pathname.endsWith("/.well-known/agent-card.json")) {
    sendJson(response, 200, context.products.a2aCard(id));
    return;
  }
  if (request.method !== "POST" || url.pathname.endsWith("/.well-known/agent-card.json")) {
    throw new HttpError(405, "Method not allowed.");
  }
  authorizeBearer(request, context.internalToken);
  const version = request.headers["a2a-version"];
  if (typeof version !== "string") throw new HttpError(400, "A2A-Version is required.");
  const body = z.record(z.string(), z.unknown()).parse(await readJson(request));
  sendJson(response, 200, await context.products.handleA2A(id, body, version));
}

async function apiRoute(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): Promise<void> {
  if (request.method === "POST" && url.pathname === "/api/ag-ui") {
    await (await context.products.agUi(url.searchParams.get("agent") ?? "swarm")).handle(
      request,
      response,
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/bootstrap") {
    sendJson(response, 200, {
      agents: context.products.availableAgents,
      sessions: await context.products.rootAgent.list(),
      workspace: { id: context.workspace.id, label: context.workspace.label },
    });
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/sessions") {
    sendJson(
      response,
      200,
      await (await context.products.agent(url.searchParams.get("agent") ?? "swarm")).list(),
    );
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/v1/sessions") {
    sendJson(response, 200, {
      sessionId: await (
        await context.products.agent(url.searchParams.get("agent") ?? "swarm")
      ).create(),
    });
    return;
  }
  const session = /^\/api\/v1\/sessions\/([^/]+)$/u.exec(url.pathname)?.[1];
  if (request.method === "GET" && session !== undefined) {
    sendJson(
      response,
      200,
      await loadAgUiHistory(
        await context.products.agent(url.searchParams.get("agent") ?? "swarm"),
        decodeURIComponent(session),
      ),
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/science") {
    sendJson(response, 200, context.products.science.getWorkspace("renderer", context.signal));
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/swarm") {
    sendJson(response, 200, context.products.listSwarms());
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/git") {
    sendJson(
      response,
      200,
      (await context.products.dvc.inspect(context.workspace.root, context.signal)).git,
    );
    return;
  }
  if (request.method === "GET" && url.pathname === "/api/v1/dvc") {
    sendJson(
      response,
      200,
      await context.products.dvc.inspect(context.workspace.root, context.signal),
    );
    return;
  }
  if (request.method === "POST" && url.pathname === "/api/v1/dvc") {
    const input = DvcMutation.parse(await readJson(request));
    if (input.action === "pull") {
      sendJson(
        response,
        200,
        await context.products.dvc.pull(
          context.workspace.root,
          {
            ...(input.request.remote === undefined ? {} : { remote: input.request.remote }),
            ...(input.request.targets === undefined ? {} : { targets: input.request.targets }),
          },
          context.signal,
        ),
      );
      return;
    }
    const reproduction = await context.products.dvc.reproduce(
      context.workspace.root,
      {
        ...(input.request.pull === undefined ? {} : { pull: input.request.pull }),
        ...(input.request.remote === undefined ? {} : { remote: input.request.remote }),
        ...(input.request.targets === undefined ? {} : { targets: input.request.targets }),
      },
      context.signal,
    );
    try {
      sendJson(response, 200, {
        source: reproduction.source,
        result: reproduction.result,
        after: reproduction.after,
      });
    } finally {
      await reproduction.dispose();
    }
    return;
  }
  throw new HttpError(404, "Not found.");
}

function exchangeLaunchToken(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  context: RouteContext,
): boolean {
  if (request.method !== "GET" || url.pathname !== "/" || !url.searchParams.has("token")) {
    return false;
  }
  const token = url.searchParams.get("token") ?? "";
  const expiry = context.launchTokens.get(token);
  context.launchTokens.delete(token);
  if (expiry === undefined || expiry < Date.now())
    throw new HttpError(401, "Launch token expired.");
  response.statusCode = 303;
  response.setHeader("location", "/");
  response.setHeader(
    "set-cookie",
    `${COOKIE}=${context.sessionToken}; HttpOnly; SameSite=Strict; Path=/`,
  );
  response.end();
  return true;
}

async function serveStatic(
  request: IncomingMessage,
  response: ServerResponse,
  url: URL,
  rendererRoot: string,
): Promise<void> {
  if (request.method !== "GET" && request.method !== "HEAD") {
    throw new HttpError(405, "Method not allowed.");
  }
  let pathname: string;
  try {
    pathname = decodeURIComponent(url.pathname);
  } catch {
    throw new HttpError(400, "Malformed static path.");
  }
  if (pathname.includes("\0") || pathname.split("/").includes("..")) {
    throw new HttpError(404, "Not found.");
  }
  const candidate = resolve(rendererRoot, pathname === "/" ? "index.html" : pathname.slice(1));
  if (!contained(rendererRoot, candidate)) throw new HttpError(404, "Not found.");
  let canonical: string;
  try {
    canonical = await realpath(candidate);
  } catch {
    throw new HttpError(404, "Not found.");
  }
  if (!contained(rendererRoot, canonical) || !(await stat(canonical)).isFile()) {
    throw new HttpError(404, "Not found.");
  }
  response.statusCode = 200;
  response.setHeader(
    "content-type",
    CONTENT_TYPES[extname(canonical)] ?? "application/octet-stream",
  );
  if (request.method === "HEAD") {
    response.end();
    return;
  }
  await new Promise<void>((done, reject) => {
    const stream = createReadStream(canonical);
    stream.once("error", reject);
    response.once("finish", done);
    stream.pipe(response);
  });
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  if (!(request.headers["content-type"] ?? "").startsWith("application/json")) {
    throw new HttpError(415, "Expected an application/json body.");
  }
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.from(chunk);
    bytes += buffer.byteLength;
    if (bytes > BODY_LIMIT) throw new HttpError(413, "Request body is too large.");
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

function validOrigin(request: IncomingMessage, origin: string): boolean {
  return request.headers.origin === undefined
    ? request.method === "GET" || request.method === "HEAD"
    : request.headers.origin === origin;
}

function contained(root: string, candidate: string): boolean {
  const path = relative(root, candidate);
  return path === "" || (path !== ".." && !path.startsWith(`..${sep}`));
}

function cookie(request: IncomingMessage, name: string): string | undefined {
  for (const entry of (request.headers.cookie ?? "").split(";")) {
    const [key, ...value] = entry.trim().split("=");
    if (key === name) return value.join("=");
  }
  return undefined;
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
  sendJson(response, status, {
    error: error instanceof Error ? error.message : String(error),
    ...(error instanceof ZodError ? { issues: error.issues } : {}),
  });
}

function securityHeaders(response: ServerResponse): void {
  response.setHeader(
    "content-security-policy",
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
  );
  response.setHeader("referrer-policy", "no-referrer");
  response.setHeader("x-content-type-options", "nosniff");
  response.setHeader("x-frame-options", "DENY");
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

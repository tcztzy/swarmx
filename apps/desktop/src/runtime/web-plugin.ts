import type { IncomingMessage, ServerResponse } from "node:http";
import { type Context, Service } from "@deepseek-ai/cordis";
import type { WebServer } from "@deepseek-ai/dsh-host-webserver";
import { z } from "zod";
import type { ApprovalResponse, RuntimeKind, WorkspaceScope } from "./contracts.js";
import { ConversationController } from "./controller.js";
import { type ConversationRuntimeRegistry, RuntimeNotRegisteredError } from "./registry.js";

export const CONVERSATION_RUNTIME_PATH = "/api/swarmx/conversation-runtimes";

const MAX_BODY_BYTES = 1024 * 1024;
const DEFAULT_ROUTE_DRAIN_TIMEOUT_MS = 10_000;
const RuntimeKindSchema = z.enum(["dsh", "codex"]);
const IdSchema = z.string().min(1).max(2048);
const RuntimeSchema = z.object({ runtimeKind: RuntimeKindSchema }).strict();
const ConversationSchema = RuntimeSchema.extend({ conversationId: IdSchema }).strict();
const StartSchema = ConversationSchema.extend({
  text: z.string().min(1).max(MAX_BODY_BYTES),
}).strict();
const SteerSchema = StartSchema.extend({ turnId: IdSchema }).strict();
const InterruptSchema = ConversationSchema.extend({ turnId: IdSchema }).strict();
const ForkSchema = ConversationSchema.extend({ beforeTurnId: IdSchema }).strict();
const RerunSchema = ConversationSchema.extend({ userItemId: IdSchema }).strict();
const EditSchema = RerunSchema.extend({
  text: z.string().min(1).max(MAX_BODY_BYTES),
}).strict();
const ApprovalSchema = ConversationSchema.extend({
  turnId: IdSchema,
  itemId: IdSchema,
  approvalId: IdSchema,
  decision: z.enum(["accept", "accept_for_session", "decline", "cancel", "submit"]),
  answers: z.record(z.string(), z.array(z.string().max(100_000)).max(100)).optional(),
  form: z.record(z.string(), z.unknown()).optional(),
}).strict();

export interface ConversationRuntimeServiceConfig {
  registry: ConversationRuntimeRegistry;
  routeDrainTimeoutMs?: number;
  workspace: WorkspaceScope;
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    conversationRuntimes: ConversationRuntimeService;
  }
}

/** Host Cordis service mounting the neutral protocol on DSH Web's listener. */
export class ConversationRuntimeService extends Service {
  static inject = ["webServer"];

  private readonly controllers = new Map<RuntimeKind, ConversationController>();
  private readonly lifetime = new AbortController();
  private readonly routes = new Set<Promise<void>>();
  private readonly routeDrainTimeoutMs: number;
  private readonly streams = new Set<ServerResponse>();
  private shutdownResult: Promise<void> | undefined;
  private unregister = () => {};
  private unsubscribe = () => {};

  constructor(
    ctx: Context,
    private readonly config: ConversationRuntimeServiceConfig,
  ) {
    super(ctx, "conversationRuntimes");
    this.routeDrainTimeoutMs = config.routeDrainTimeoutMs ?? DEFAULT_ROUTE_DRAIN_TIMEOUT_MS;
    this.unsubscribe = config.registry.subscribe((event) => {
      const frame = `data: ${JSON.stringify(event)}\n\n`;
      for (const stream of this.streams) {
        if (stream.write(frame)) continue;
        this.streams.delete(stream);
        stream.destroy();
      }
    });
    this.unregister = ctx.webServer.register({
      kind: "prefix",
      path: CONVERSATION_RUNTIME_PATH,
      handler: (request, response) => {
        const route = this.route(request, response).catch((error: unknown) => {
          try {
            sendError(response, error);
          } catch (sendFailure) {
            response.destroy(
              sendFailure instanceof Error ? sendFailure : new Error(String(sendFailure)),
            );
          }
        });
        this.routes.add(route);
        void route.then(() => this.routes.delete(route));
      },
    });
    ctx.effect(() => () => this.shutdown(), "swarmx: conversation runtime protocol");
  }

  /** Stop admission and settle admitted requests before their runtime owner is disposed. */
  shutdown(): Promise<void> {
    if (this.shutdownResult === undefined) this.shutdownResult = this.shutdownOnce();
    return this.shutdownResult;
  }

  private controller(runtimeKind: RuntimeKind): ConversationController {
    let controller = this.controllers.get(runtimeKind);
    if (controller === undefined) {
      controller = new ConversationController(this.config.registry.runtime(runtimeKind));
      this.controllers.set(runtimeKind, controller);
    }
    return controller;
  }

  private async shutdownOnce(): Promise<void> {
    this.unregister();
    this.lifetime.abort(new Error("Conversation runtime service disposed."));
    this.unsubscribe();
    for (const stream of this.streams) stream.end();
    this.streams.clear();
    await drainRoutes(this.routes, this.routeDrainTimeoutMs);
  }

  private async route(request: IncomingMessage, response: ServerResponse): Promise<void> {
    const signal = AbortSignal.any([disconnectSignal(request, response), this.lifetime.signal]);
    const unbindLifetime = destroyOnAbort(request, response, this.lifetime.signal);
    try {
      signal.throwIfAborted();
      const origin = expectedOrigin(this.ctx.webServer);
      if (request.headers.host !== new URL(origin).host) {
        sendJson(response, 403, { error: "Invalid Host header." });
        return;
      }
      const url = new URL(request.url ?? CONVERSATION_RUNTIME_PATH, origin);
      if (request.method === "GET") {
        await this.routeGet(response, url, signal);
        return;
      }
      if (request.method !== "POST") {
        sendJson(response, 404, { error: "Not found." });
        return;
      }
      if (request.headers.origin !== origin) {
        sendJson(response, 403, { error: "Cross-origin mutation denied." });
        return;
      }
      const body = await readJson(request, signal);
      await this.routePost(response, url.pathname, body, signal);
    } finally {
      unbindLifetime();
    }
  }

  private async routeGet(response: ServerResponse, url: URL, signal: AbortSignal): Promise<void> {
    if (url.pathname === CONVERSATION_RUNTIME_PATH) {
      sendJson(response, 200, {
        defaultRuntimeKind: this.config.registry.defaultKind,
        runtimeKinds: this.config.registry.kinds(),
      });
      return;
    }
    if (url.pathname === `${CONVERSATION_RUNTIME_PATH}/conversations`) {
      const runtimeKind = RuntimeKindSchema.parse(url.searchParams.get("runtimeKind"));
      sendJson(response, 200, await this.config.registry.runtime(runtimeKind).list(signal));
      return;
    }
    if (url.pathname === `${CONVERSATION_RUNTIME_PATH}/conversation`) {
      const runtimeKind = RuntimeKindSchema.parse(url.searchParams.get("runtimeKind"));
      const conversationId = IdSchema.parse(url.searchParams.get("conversationId"));
      sendJson(
        response,
        200,
        await this.config.registry.runtime(runtimeKind).read(conversationId, signal),
      );
      return;
    }
    if (url.pathname === `${CONVERSATION_RUNTIME_PATH}/events`) {
      response.writeHead(200, {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache, no-transform",
        connection: "keep-alive",
        "x-content-type-options": "nosniff",
      });
      response.write(": connected\n\n");
      this.streams.add(response);
      response.once("close", () => this.streams.delete(response));
      return;
    }
    sendJson(response, 404, { error: "Not found." });
  }

  private async routePost(
    response: ServerResponse,
    pathname: string,
    body: unknown,
    signal: AbortSignal,
  ): Promise<void> {
    switch (pathname) {
      case `${CONVERSATION_RUNTIME_PATH}/create`: {
        const { runtimeKind } = RuntimeSchema.parse(body);
        sendJson(
          response,
          200,
          await this.config.registry
            .runtime(runtimeKind)
            .create({ workspace: this.config.workspace }, signal),
        );
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/start`: {
        const { runtimeKind, ...request } = StartSchema.parse(body);
        sendJson(
          response,
          200,
          await this.config.registry.runtime(runtimeKind).start(request, signal),
        );
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/steer`: {
        const { runtimeKind, ...request } = SteerSchema.parse(body);
        await this.config.registry.runtime(runtimeKind).steer(request, signal);
        sendJson(response, 200, {});
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/interrupt`: {
        const { runtimeKind, ...request } = InterruptSchema.parse(body);
        await this.config.registry.runtime(runtimeKind).interrupt(request, signal);
        sendJson(response, 200, {});
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/retry`: {
        const { runtimeKind, ...request } = RerunSchema.parse(body);
        sendJson(
          response,
          200,
          await this.controller(runtimeKind).retry(
            request.conversationId,
            request.userItemId,
            signal,
          ),
        );
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/edit`: {
        const { runtimeKind, ...request } = EditSchema.parse(body);
        sendJson(
          response,
          200,
          await this.controller(runtimeKind).edit(
            request.conversationId,
            request.userItemId,
            request.text,
            signal,
          ),
        );
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/fork`: {
        const { runtimeKind, ...request } = ForkSchema.parse(body);
        sendJson(
          response,
          200,
          await this.controller(runtimeKind).fork(
            request.conversationId,
            request.beforeTurnId,
            signal,
          ),
        );
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/archive`: {
        const { runtimeKind, conversationId } = ConversationSchema.parse(body);
        await this.config.registry.runtime(runtimeKind).archive(conversationId, signal);
        sendJson(response, 200, {});
        return;
      }
      case `${CONVERSATION_RUNTIME_PATH}/approval`: {
        const value = ApprovalSchema.parse(body);
        const approval: ApprovalResponse = {
          runtime: value.runtimeKind,
          conversationId: value.conversationId,
          turnId: value.turnId,
          itemId: value.itemId,
          approvalId: value.approvalId,
          decision: value.decision,
          ...(value.answers === undefined ? {} : { answers: value.answers }),
          ...(value.form === undefined ? {} : { form: value.form }),
        };
        await this.config.registry.runtime(value.runtimeKind).respondToApproval(approval);
        sendJson(response, 200, {});
        return;
      }
      default:
        sendJson(response, 404, { error: "Not found." });
    }
  }
}

function expectedOrigin(webServer: WebServer): string {
  return `http://${webServer.host}:${String(webServer.port)}`;
}

async function readJson(request: IncomingMessage, signal: AbortSignal): Promise<unknown> {
  if (request.headers["content-type"]?.split(";", 1)[0] !== "application/json") {
    throw new HttpError(415, "Mutation requests require application/json.");
  }
  signal.throwIfAborted();
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of request) {
    signal.throwIfAborted();
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > MAX_BODY_BYTES) throw new HttpError(413, "Request body is too large.");
    chunks.push(buffer);
  }
  signal.throwIfAborted();
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    throw new HttpError(400, "Request body is not valid JSON.");
  }
}

function disconnectSignal(request: IncomingMessage, response: ServerResponse): AbortSignal {
  const controller = new AbortController();
  const abort = () => controller.abort(new Error("HTTP request disconnected."));
  const finish = () => {
    request.off("aborted", abort);
    response.off("close", close);
  };
  const close = () => {
    request.off("aborted", abort);
    response.off("finish", finish);
    if (!response.writableFinished) abort();
  };
  request.once("aborted", abort);
  response.once("finish", finish);
  response.once("close", close);
  if (request.aborted || (response.destroyed && !response.writableFinished)) abort();
  return controller.signal;
}

function destroyOnAbort(
  request: IncomingMessage,
  response: ServerResponse,
  signal: AbortSignal,
): () => void {
  const destroy = () => {
    const reason =
      signal.reason instanceof Error ? signal.reason : new Error(String(signal.reason));
    if (!response.destroyed) response.destroy(reason);
    if (!request.destroyed) request.destroy(reason);
  };
  signal.addEventListener("abort", destroy, { once: true });
  if (signal.aborted) destroy();
  return () => signal.removeEventListener("abort", destroy);
}

async function drainRoutes(routes: ReadonlySet<Promise<void>>, timeoutMs: number): Promise<void> {
  if (routes.size === 0) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      Promise.all([...routes]),
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () =>
            reject(
              new Error(
                `${String(routes.size)} conversation runtime route(s) did not settle within ${String(timeoutMs)}ms.`,
              ),
            ),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  if (response.headersSent || response.destroyed || response.writableEnded) return;
  const body = `${JSON.stringify(value)}\n`;
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  response.end(body);
}

function sendError(response: ServerResponse, error: unknown): void {
  if (response.destroyed || response.writableEnded) return;
  if (response.headersSent) {
    response.destroy(error instanceof Error ? error : new Error(String(error)));
    return;
  }
  const status =
    error instanceof HttpError
      ? error.status
      : error instanceof RuntimeNotRegisteredError
        ? error.status
        : error instanceof z.ZodError
          ? 400
          : 500;
  sendJson(response, status, { error: error instanceof Error ? error.message : String(error) });
}

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export default ConversationRuntimeService;

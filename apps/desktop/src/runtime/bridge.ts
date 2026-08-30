import { randomUUID, timingSafeEqual } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { z } from "zod";
import type { ConversationRuntime, WorkspaceScope } from "./contracts.js";
import {
  SwarmRecoveryClaimConflictError,
  type SwarmRecoveryOwner,
} from "./swarm-recovery-owner.js";

const HOST = "127.0.0.1";
const MAX_BODY_BYTES = 1024 * 1024;
const MEMBER_CLEANUP_TIMEOUT_MS = 10_000;
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 10_000;
const IdSchema = z.string().min(1).max(2048);
const RequestSchema = z.discriminatedUnion("action", [
  z.object({ action: z.literal("list") }).strict(),
  z.object({ action: z.literal("read"), conversationId: IdSchema }).strict(),
  z.object({ action: z.literal("create"), model: z.string().min(1).max(512).optional() }).strict(),
  z
    .object({
      action: z.literal("create_member"),
      teamId: IdSchema,
      memberId: z.string().uuid(),
      model: z.string().min(1).max(512).optional(),
    })
    .strict(),
  z
    .object({
      action: z.literal("send"),
      conversationId: IdSchema,
      text: z.string().min(1).max(MAX_BODY_BYTES),
    })
    .strict(),
  z
    .object({
      action: z.literal("steer"),
      conversationId: IdSchema,
      turnId: IdSchema,
      text: z.string().min(1).max(MAX_BODY_BYTES),
    })
    .strict(),
  z.object({ action: z.literal("interrupt"), conversationId: IdSchema, turnId: IdSchema }).strict(),
  z
    .object({
      action: z.literal("archive"),
      conversationId: IdSchema,
      memberId: z.string().uuid(),
    })
    .strict(),
]);

export type RuntimeBridgeRequest = z.infer<typeof RequestSchema>;

export interface RuntimeBridge {
  url: string;
  token: string;
  attach(runtime: ConversationRuntime): void;
  dispose(): Promise<void>;
}

export interface RuntimeBridgeOptions {
  shutdownTimeoutMs?: number;
}

export class RuntimeBridgeMemberStartupError extends Error {
  constructor(
    message: string,
    readonly handleState: "absent" | "possible",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "RuntimeBridgeMemberStartupError";
  }
}

export async function startRuntimeBridge(
  workspace: WorkspaceScope,
  memberClaims?: Pick<
    SwarmRecoveryOwner,
    "claimCodexMember" | "settleCodexMemberArchive" | "settleCodexMemberCreationFailure"
  >,
  options: RuntimeBridgeOptions = {},
): Promise<RuntimeBridge> {
  const token = randomUUID();
  const shutdownTimeoutMs = z
    .number()
    .int()
    .positive()
    .max(60_000)
    .parse(options.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS);
  let runtime: ConversationRuntime | undefined;
  let closing = false;
  let disposal: Promise<void> | undefined;
  const shutdown = new AbortController();
  const operations = new Set<Promise<void>>();
  const memberRetirements = new Map<string, { memberId: string; operation: Promise<void> }>();
  const retireMember = (
    attached: ConversationRuntime,
    conversationId: string,
    memberId: string,
  ): Promise<void> => {
    const existing = memberRetirements.get(conversationId);
    if (existing !== undefined) {
      if (existing.memberId !== memberId) {
        throw new Error("Codex Thread retirement identity does not match its in-flight owner.");
      }
      return existing.operation;
    }
    const retirement = retireProvisionedMember(attached, conversationId, memberId).finally(() => {
      if (memberRetirements.get(conversationId)?.operation === retirement) {
        memberRetirements.delete(conversationId);
      }
    });
    memberRetirements.set(conversationId, { memberId, operation: retirement });
    return retirement;
  };
  const server = createServer((request, response) => {
    if (closing) {
      sendJson(response, 503, { error: "Runtime bridge is closing." });
      return;
    }
    const operation = route(
      request,
      response,
      token,
      workspace,
      runtime,
      memberClaims,
      retireMember,
      shutdown.signal,
    );
    operations.add(operation);
    void operation
      .catch((error: unknown) => {
        sendJson(response, 500, {
          error: error instanceof Error ? error.message : String(error),
          ...(error instanceof RuntimeBridgeMemberStartupError
            ? { memberHandleState: error.handleState }
            : {}),
        });
      })
      .finally(() => operations.delete(operation));
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, HOST, resolve);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    server.close();
    throw new Error("SwarmX runtime bridge did not expose a TCP address.");
  }
  return {
    url: `http://${HOST}:${String(address.port)}/`,
    token,
    attach(value) {
      if (runtime !== undefined) throw new Error("SwarmX runtime bridge is already attached.");
      runtime = value;
    },
    dispose() {
      disposal ??= (async () => {
        closing = true;
        shutdown.abort(new Error("Runtime bridge is closing."));
        const closed = new Promise<void>((resolve, reject) =>
          server.close((error) => (error === undefined ? resolve() : reject(error))),
        );
        const drained = await settlesWithin(
          Promise.allSettled([...operations]).then(() => undefined),
          shutdownTimeoutMs,
        );
        if (!drained) server.closeAllConnections();
        await settlesWithin(closed, shutdownTimeoutMs);
      })();
      return disposal;
    },
  };
}

async function route(
  request: IncomingMessage,
  response: ServerResponse,
  token: string,
  workspace: WorkspaceScope,
  runtime: ConversationRuntime | undefined,
  memberClaims:
    | Pick<
        SwarmRecoveryOwner,
        "claimCodexMember" | "settleCodexMemberArchive" | "settleCodexMemberCreationFailure"
      >
    | undefined,
  retireMember: (
    runtime: ConversationRuntime,
    conversationId: string,
    memberId: string,
  ) => Promise<void>,
  shutdownSignal: AbortSignal,
): Promise<void> {
  if (request.method !== "POST" || request.url !== "/") {
    sendJson(response, 404, { error: "Not found." });
    return;
  }
  if (!authorized(request.headers.authorization, token)) {
    sendJson(response, 403, { error: "Invalid runtime bridge token." });
    return;
  }
  if (runtime === undefined) {
    sendJson(response, 503, { error: "Runtime bridge is not attached yet." });
    return;
  }
  const input = RequestSchema.parse(await readJson(request));
  const signal = AbortSignal.any([disconnectSignal(request, response), shutdownSignal]);
  switch (input.action) {
    case "list":
      sendJson(response, 200, await runtime.list(signal));
      return;
    case "read":
      sendJson(response, 200, await runtime.read(input.conversationId, signal));
      return;
    case "create":
      sendJson(
        response,
        200,
        await runtime.create(
          { workspace, ...(input.model === undefined ? {} : { model: input.model }) },
          signal,
        ),
      );
      return;
    case "create_member": {
      if (runtime.createProvisionedMember === undefined) {
        throw new RuntimeBridgeMemberStartupError(
          "Selected runtime cannot durably provision Swarm members.",
          "absent",
        );
      }
      let conversation: Awaited<ReturnType<ConversationRuntime["create"]>>;
      try {
        conversation = await runtime.createProvisionedMember(
          {
            workspace,
            ...(input.model === undefined ? {} : { model: input.model }),
          },
          input.memberId,
        );
      } catch (cause) {
        throw new RuntimeBridgeMemberStartupError(
          cause instanceof Error ? cause.message : "Codex member creation failed",
          "possible",
          { cause },
        );
      }
      let claim: "archive_required" | "created" | "existing" | "unclaimed" = "unclaimed";
      if (memberClaims !== undefined) {
        try {
          claim = memberClaims.claimCodexMember({
            workspaceRoot: workspace.root,
            teamId: input.teamId,
            memberId: input.memberId,
            conversationId: conversation.conversationId,
          });
          if (claim === "archive_required") {
            await retireMember(runtime, conversation.conversationId, input.memberId);
            if (
              !memberClaims.settleCodexMemberArchive({
                workspaceRoot: workspace.root,
                teamId: input.teamId,
                memberId: input.memberId,
                conversationId: conversation.conversationId,
              })
            ) {
              throw new Error("Codex Swarm archive acknowledgement became stale.");
            }
            throw new RuntimeBridgeMemberStartupError(
              "Swarm archive started while the member Thread was provisioning.",
              "absent",
            );
          }
        } catch (cause) {
          if (claim === "archive_required") throw cause;
          if (cause instanceof SwarmRecoveryClaimConflictError && cause.kind === "handle") {
            memberClaims.settleCodexMemberCreationFailure({
              workspaceRoot: workspace.root,
              teamId: input.teamId,
              memberId: input.memberId,
            });
            throw new RuntimeBridgeMemberStartupError(cause.message, "absent", { cause });
          }
          try {
            await retireMember(runtime, conversation.conversationId, input.memberId);
            memberClaims.settleCodexMemberCreationFailure({
              workspaceRoot: workspace.root,
              teamId: input.teamId,
              memberId: input.memberId,
            });
          } catch (cleanupError) {
            throw new RuntimeBridgeMemberStartupError(
              "Codex Swarm root member claim and rollback failed",
              "possible",
              { cause: new AggregateError([cause, cleanupError]) },
            );
          }
          throw new RuntimeBridgeMemberStartupError(
            cause instanceof Error ? cause.message : "Codex Swarm root member claim failed",
            "absent",
            { cause },
          );
        }
      }
      sendJson(response, 200, { claim, conversation });
      return;
    }
    case "send":
      sendJson(response, 200, await runtime.start(input, signal));
      return;
    case "steer":
      await runtime.steer(input, signal);
      sendJson(response, 200, {});
      return;
    case "interrupt":
      await runtime.interrupt(input, signal);
      sendJson(response, 200, {});
      return;
    case "archive":
      await retireMember(runtime, input.conversationId, input.memberId);
      sendJson(response, 200, {});
  }
}

async function settlesWithin(operation: Promise<void>, timeoutMs: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<false>((resolve) => {
    timer = setTimeout(() => resolve(false), timeoutMs);
  });
  try {
    return await Promise.race([operation.then(() => true), timeout]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

async function retireProvisionedMember(
  runtime: ConversationRuntime,
  conversationId: string,
  memberId: string,
): Promise<void> {
  const signal = AbortSignal.timeout(MEMBER_CLEANUP_TIMEOUT_MS);
  if (runtime.retireProvisionedMember !== undefined) {
    await runtime.retireProvisionedMember(conversationId, memberId, signal);
    return;
  }
  const snapshot = await runtime.read(conversationId, signal);
  if (!snapshot.archived) await runtime.archive(conversationId, signal);
}

function authorized(header: string | undefined, token: string): boolean {
  if (header === undefined) return false;
  const expected = Buffer.from(`Bearer ${token}`);
  const actual = Buffer.from(header);
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    bytes += buffer.byteLength;
    if (bytes > MAX_BODY_BYTES) throw new Error("Runtime bridge request body is too large.");
    chunks.push(buffer);
  }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function disconnectSignal(request: IncomingMessage, response: ServerResponse): AbortSignal {
  const controller = new AbortController();
  const abort = () => {
    if (!response.writableEnded) {
      controller.abort(new Error("Runtime bridge caller disconnected."));
    }
  };
  request.once("aborted", abort);
  response.once("close", abort);
  return controller.signal;
}

function sendJson(response: ServerResponse, status: number, value: unknown): void {
  if (response.headersSent) return;
  response.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
  });
  response.end(JSON.stringify(value));
}

export class RuntimeBridgeClient {
  constructor(
    private readonly url: string,
    private readonly token: string,
  ) {}

  async request<T>(request: RuntimeBridgeRequest, signal?: AbortSignal): Promise<T> {
    const response = await fetch(this.url, {
      method: "POST",
      headers: {
        authorization: `Bearer ${this.token}`,
        "content-type": "application/json",
      },
      body: JSON.stringify(request),
      ...(signal === undefined ? {} : { signal }),
    });
    const value = (await response.json()) as {
      error?: unknown;
      memberHandleState?: unknown;
    };
    if (!response.ok) {
      const message = typeof value.error === "string" ? value.error : "Runtime bridge failed.";
      if (value.memberHandleState === "absent" || value.memberHandleState === "possible") {
        throw new RuntimeBridgeMemberStartupError(message, value.memberHandleState);
      }
      throw new Error(message);
    }
    return value as T;
  }
}

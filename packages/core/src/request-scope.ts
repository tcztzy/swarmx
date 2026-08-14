import { AsyncLocalStorage } from "node:async_hooks";

export interface RequestParticipant {
  cancel(): Promise<void>;
  cleanup(): void;
}

export interface RequestParticipantRegistration {
  requestId: string;
  signal: AbortSignal;
  unregister(): void;
}

interface RequestContext {
  id: string;
  controller: AbortController;
  participants: Set<RequestParticipant>;
  cancelPromise: Promise<void> | null;
}

let requestScope: AsyncLocalStorage<RequestContext> | null = null;
let activeRequests: Map<string, RequestContext> | null = null;

export class RequestCancelledError extends Error {
  constructor(requestId?: string) {
    super(requestId ? `Request "${requestId}" was cancelled.` : "Request was cancelled.");
    this.name = "RequestCancelledError";
  }
}

/** Run one exclusive request inside a cooperative cancellation scope. */
export async function withRequestScope<T>(requestId: string, run: () => Promise<T>): Promise<T> {
  const id = validateRequestId(requestId);
  const requests = getActiveRequests();
  if (requests.has(id)) throw new Error(`Request "${id}" is already active.`);

  const context: RequestContext = {
    id,
    controller: new AbortController(),
    participants: new Set(),
    cancelPromise: null,
  };
  requests.set(id, context);

  return getRequestScope().run(context, async () => {
    try {
      const result = await run();
      throwIfRequestCancelled(context);
      return result;
    } catch (error) {
      if (context.controller.signal.aborted) throw cancellationReason(context);
      throw error;
    } finally {
      for (const participant of [...context.participants]) participant.cleanup();
      requests.delete(id);
    }
  });
}

/** Abort an active request and settle every participant cancellation once. */
export async function cancelRequest(requestId: string): Promise<boolean> {
  const context = getActiveRequests().get(requestId);
  if (!context) return false;

  if (!context.cancelPromise) {
    context.controller.abort(new RequestCancelledError(context.id));
    context.cancelPromise = Promise.allSettled(
      [...context.participants].map(cancelParticipant),
    ).then(() => undefined);
  }

  await context.cancelPromise;
  return true;
}

/** Register protocol/process cleanup without coupling the scope to an adapter. */
export function registerCurrentRequestParticipant(
  participant: RequestParticipant,
): RequestParticipantRegistration | undefined {
  const context = getRequestScope().getStore();
  if (!context) return undefined;
  context.participants.add(participant);
  return {
    requestId: context.id,
    signal: context.controller.signal,
    unregister: () => context.participants.delete(participant),
  };
}

/** The signal for the request currently executing in this async context. */
export function currentRequestSignal(): AbortSignal | undefined {
  return getRequestScope().getStore()?.controller.signal;
}

/** Throw the request's stable cancellation reason at cooperative boundaries. */
export function throwIfCurrentRequestCancelled(): void {
  const signal = currentRequestSignal();
  if (signal?.aborted) {
    throw signal.reason instanceof Error ? signal.reason : new RequestCancelledError();
  }
}

function throwIfRequestCancelled(context: RequestContext): void {
  if (context.controller.signal.aborted) throw cancellationReason(context);
}

function cancellationReason(context: RequestContext): Error {
  return context.controller.signal.reason instanceof Error
    ? context.controller.signal.reason
    : new RequestCancelledError(context.id);
}

function cancelParticipant(participant: RequestParticipant): Promise<void> {
  try {
    return participant.cancel();
  } catch (error) {
    return Promise.reject(error);
  }
}

function validateRequestId(requestId: string): string {
  if (typeof requestId !== "string" || requestId.trim().length === 0) {
    throw new Error("Request ID must be a non-empty string.");
  }
  if (requestId.length > 256) throw new Error("Request ID must be at most 256 characters.");
  return requestId;
}

function getRequestScope(): AsyncLocalStorage<RequestContext> {
  requestScope ??= new AsyncLocalStorage<RequestContext>();
  return requestScope;
}

function getActiveRequests(): Map<string, RequestContext> {
  activeRequests ??= new Map<string, RequestContext>();
  return activeRequests;
}

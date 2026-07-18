import { randomUUID } from "node:crypto";
import { currentRequestSignal } from "@swarmx/core";

export interface ClaudeQuestionOption {
  label: string;
  description: string;
  preview?: string;
}

export interface ClaudeQuestion {
  question: string;
  header: string;
  options: ClaudeQuestionOption[];
  multiSelect: boolean;
}

export interface ToolApprovalOption {
  optionId: string;
  name: string;
  kind: "allow_once" | "allow_always" | "reject_once" | "reject_always";
}

export type ClaudeInteractionRequest =
  | { kind: "questions"; questions: ClaudeQuestion[] }
  | { kind: "plan_approval"; plan: string; filePath: string }
  | {
      kind: "tool_approval";
      title: string;
      toolKind?: string;
      summary: string;
      options: ToolApprovalOption[];
    };

export type ClaudeInteractionResponse =
  | { kind: "questions"; answers: Record<string, string> }
  | { kind: "plan_approval"; approved: boolean; feedback?: string }
  | { kind: "tool_approval"; optionId: string };

export type DesktopAgentInteractionEvent = ClaudeInteractionRequest & {
  requestId: string;
  interactionId: string;
};

export interface DesktopAgentInteractionResolution {
  requestId: string;
  interactionId: string;
  response: ClaudeInteractionResponse;
}

export interface AgentInteractionOwner {
  readonly id: number;
  isDestroyed(): boolean;
  send(channel: string, value: unknown): void;
  once(event: "destroyed", listener: () => void): unknown;
  removeListener(event: "destroyed", listener: () => void): unknown;
}

interface PendingInteraction {
  owner: AgentInteractionOwner;
  request: ClaudeInteractionRequest;
  requestId: string;
  interactionId: string;
  signal?: AbortSignal;
  onAbort: () => void;
  onDestroyed: () => void;
  resolve: (response: ClaudeInteractionResponse) => void;
  reject: (error: Error) => void;
}

/** Bridges request-scoped tool calls to the renderer without granting cross-window authority. */
export class AgentInteractionBroker {
  private readonly pending = new Map<string, PendingInteraction>();

  request(
    owner: AgentInteractionOwner,
    requestId: string,
    request: ClaudeInteractionRequest,
  ): Promise<ClaudeInteractionResponse> {
    if (owner.isDestroyed()) return Promise.reject(new Error("Renderer window was closed."));
    const interactionId = randomUUID();
    const signal = currentRequestSignal();
    if (signal?.aborted) return Promise.reject(abortReason(signal));

    return new Promise((resolve, reject) => {
      const entry: PendingInteraction = {
        owner,
        request,
        requestId,
        interactionId,
        ...(signal ? { signal } : {}),
        onAbort: () => this.reject(interactionId, abortReason(signal)),
        onDestroyed: () =>
          this.reject(interactionId, new Error("Renderer window was closed during interaction.")),
        resolve,
        reject,
      };
      this.pending.set(interactionId, entry);
      owner.once("destroyed", entry.onDestroyed);
      signal?.addEventListener("abort", entry.onAbort, { once: true });
      try {
        owner.send("agent:interaction", {
          ...request,
          requestId,
          interactionId,
        } satisfies DesktopAgentInteractionEvent);
      } catch (error) {
        this.reject(interactionId, error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  resolve(owner: AgentInteractionOwner, resolution: DesktopAgentInteractionResolution): boolean {
    if (
      !isRecord(resolution) ||
      typeof resolution.requestId !== "string" ||
      typeof resolution.interactionId !== "string" ||
      !("response" in resolution)
    ) {
      throw new Error("Interaction resolution is invalid.");
    }
    const entry = this.pending.get(resolution.interactionId);
    if (
      !entry ||
      entry.owner !== owner ||
      entry.requestId !== resolution.requestId ||
      entry.interactionId !== resolution.interactionId
    ) {
      return false;
    }
    const response = validateResponse(entry.request, resolution.response);
    this.finish(entry);
    entry.resolve(response);
    return true;
  }

  cancelOwner(owner: AgentInteractionOwner): number {
    const entries = [...this.pending.values()].filter((entry) => entry.owner === owner);
    for (const entry of entries) {
      this.reject(entry.interactionId, new Error("Renderer window was closed during interaction."));
    }
    return entries.length;
  }

  cancelRequest(owner: AgentInteractionOwner, requestId: string): number {
    const entries = [...this.pending.values()].filter(
      (entry) => entry.owner === owner && entry.requestId === requestId,
    );
    for (const entry of entries) {
      this.reject(entry.interactionId, new Error(`Request "${requestId}" interaction closed.`));
    }
    return entries.length;
  }

  get size(): number {
    return this.pending.size;
  }

  private reject(interactionId: string, error: Error): void {
    const entry = this.pending.get(interactionId);
    if (!entry) return;
    this.finish(entry);
    entry.reject(error);
  }

  private finish(entry: PendingInteraction): void {
    if (this.pending.get(entry.interactionId) !== entry) return;
    this.pending.delete(entry.interactionId);
    entry.owner.removeListener("destroyed", entry.onDestroyed);
    entry.signal?.removeEventListener("abort", entry.onAbort);
  }
}

function validateResponse(
  request: ClaudeInteractionRequest,
  response: unknown,
): ClaudeInteractionResponse {
  if (!isRecord(response) || typeof response.kind !== "string") {
    throw new Error("Interaction response is invalid.");
  }
  if (request.kind !== response.kind) throw new Error("Interaction response kind does not match.");
  if (request.kind === "plan_approval") {
    if (response.kind !== "plan_approval" || typeof response.approved !== "boolean") {
      throw new Error("Plan approval response is invalid.");
    }
    if (response.feedback !== undefined && typeof response.feedback !== "string") {
      throw new Error("Plan approval feedback must be text.");
    }
    return {
      kind: "plan_approval",
      approved: response.approved,
      ...(response.feedback?.trim() ? { feedback: response.feedback.trim() } : {}),
    };
  }
  if (request.kind === "tool_approval") {
    if (response.kind !== "tool_approval" || typeof response.optionId !== "string") {
      throw new Error("Tool approval response is invalid.");
    }
    if (!request.options.some((option) => option.optionId === response.optionId)) {
      throw new Error("Tool approval option was not offered.");
    }
    return { kind: "tool_approval", optionId: response.optionId };
  }
  if (response.kind !== "questions" || !isRecord(response.answers)) {
    throw new Error("Question answers are invalid.");
  }
  const answers: Record<string, string> = {};
  const expected = new Set(request.questions.map((question) => question.question));
  for (const question of request.questions) {
    const answer = response.answers[question.question];
    if (typeof answer !== "string" || !answer.trim()) {
      throw new Error(`Question "${question.question}" requires an answer.`);
    }
    answers[question.question] = answer.trim();
  }
  for (const key of Object.keys(response.answers)) {
    if (!expected.has(key)) throw new Error(`Unexpected answer for question "${key}".`);
  }
  return { kind: "questions", answers };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function abortReason(signal: AbortSignal | undefined): Error {
  return signal?.reason instanceof Error ? signal.reason : new Error("Request was cancelled.");
}

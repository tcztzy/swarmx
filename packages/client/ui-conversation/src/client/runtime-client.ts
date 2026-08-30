const BASE_PATH = "/api/swarmx/conversation-runtimes";

export type RuntimeKind = "dsh" | "codex";
export type TurnStatus = "running" | "completed" | "interrupted" | "failed";
export type ApprovalDecision = "accept" | "accept_for_session" | "decline" | "cancel" | "submit";
export type ApprovalInputValue = string | boolean | readonly string[];

export interface ApprovalResponseFields {
  readonly answers?: Readonly<Record<string, readonly string[]>>;
  readonly form?: Readonly<Record<string, unknown>>;
}

export interface RuntimeMetadata {
  readonly defaultRuntimeKind: RuntimeKind;
  readonly runtimeKinds: readonly RuntimeKind[];
}

export interface ConversationSummary {
  readonly runtime: RuntimeKind;
  readonly conversationId: string;
  readonly workspace: { readonly id: string; readonly label: string };
  readonly title: string;
  readonly archived: boolean;
  readonly updatedAt: number;
}

export type ConversationItem =
  | {
      readonly type: "user_message" | "assistant_message" | "reasoning";
      readonly id: string;
      readonly turnId: string;
      readonly text: string;
      readonly createdAt: number;
      readonly provisional?: true;
    }
  | {
      readonly type: "tool";
      readonly id: string;
      readonly turnId: string;
      readonly name: string;
      readonly status: "running" | "completed" | "failed";
      readonly summary?: string;
      readonly createdAt: number;
    }
  | {
      readonly type: "error";
      readonly id: string;
      readonly turnId: string;
      readonly message: string;
      readonly createdAt: number;
    };

export interface ConversationTurn {
  readonly id: string;
  readonly status: TurnStatus;
  readonly items: readonly ConversationItem[];
}

export interface ConversationSnapshot {
  readonly runtime: RuntimeKind;
  readonly conversationId: string;
  readonly workspace: { readonly id: string; readonly label: string };
  readonly title: string;
  readonly archived: boolean;
  readonly turns: readonly ConversationTurn[];
  readonly approvals?: readonly ApprovalRequest[];
}

export interface ApprovalQuestion {
  readonly id: string;
  readonly prompt: string;
  readonly header?: string;
  readonly type?: "string" | "boolean" | "number" | "integer" | "string_array";
  readonly required?: boolean;
  readonly defaultValue?: string | boolean | number | readonly string[];
  readonly options?: readonly string[];
  readonly multiSelect?: boolean;
  readonly minimum?: number;
  readonly maximum?: number;
  readonly minLength?: number;
  readonly maxLength?: number;
  readonly minItems?: number;
  readonly maxItems?: number;
  readonly format?: "date" | "uri" | "email" | "date-time";
}

export interface ApprovalRequest {
  readonly seq: number;
  readonly runtime: RuntimeKind;
  readonly conversationId: string;
  readonly type: "approval_requested";
  readonly turnId: string;
  readonly itemId: string;
  readonly approvalId: string;
  readonly kind: "command" | "file_change" | "permissions" | "elicitation" | "user_input";
  readonly prompt: string;
  readonly choices: readonly ApprovalDecision[];
  readonly questions?: readonly ApprovalQuestion[];
}

export interface ApprovalResolvedEvent {
  readonly seq: number;
  readonly runtime: RuntimeKind;
  readonly conversationId: string;
  readonly type: "approval_resolved";
  readonly turnId: string;
  readonly itemId: string;
  readonly approvalId: string;
}

export type ApprovalIdentity = Pick<
  ApprovalRequest,
  "runtime" | "conversationId" | "turnId" | "itemId" | "approvalId"
>;

export type RuntimeEvent =
  | ApprovalRequest
  | ApprovalResolvedEvent
  | {
      readonly seq: number;
      readonly runtime: RuntimeKind;
      readonly conversationId: string;
      readonly type: "item_delta";
      readonly turnId: string;
      readonly itemId: string;
      readonly delta: string;
      readonly itemType?: "assistant_message" | "reasoning";
    }
  | {
      readonly seq: number;
      readonly runtime: RuntimeKind;
      readonly conversationId: string;
      readonly type: "item_completed";
      readonly turnId: string;
      readonly item: ConversationItem;
    }
  | {
      readonly seq: number;
      readonly runtime: RuntimeKind;
      readonly conversationId: string;
      readonly type: "turn_status";
      readonly turnId: string;
      readonly status: TurnStatus;
    };

export interface RuntimeEventSource {
  onerror: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent<string>) => void) | null;
  close(): void;
}

type RuntimeFetch = (input: string, init?: RequestInit) => Promise<Response>;
type EventSourceFactory = (url: string) => RuntimeEventSource;

export class ConversationRuntimeClient {
  constructor(
    private readonly fetcher: RuntimeFetch = (input, init) => fetch(input, init),
    private readonly eventSource: EventSourceFactory = (url) => new EventSource(url),
  ) {}

  metadata(signal?: AbortSignal): Promise<RuntimeMetadata> {
    return this.get(BASE_PATH, signal);
  }

  list(runtimeKind: RuntimeKind, signal?: AbortSignal): Promise<ConversationSummary[]> {
    return this.get(`${BASE_PATH}/conversations?${query({ runtimeKind })}`, signal);
  }

  read(
    runtimeKind: RuntimeKind,
    conversationId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSnapshot> {
    return this.get(`${BASE_PATH}/conversation?${query({ runtimeKind, conversationId })}`, signal);
  }

  create(runtimeKind: RuntimeKind, signal?: AbortSignal): Promise<ConversationSummary> {
    return this.post("create", { runtimeKind }, signal);
  }

  start(
    runtimeKind: RuntimeKind,
    conversationId: string,
    text: string,
    signal?: AbortSignal,
  ): Promise<{ turnId: string }> {
    return this.post("start", { runtimeKind, conversationId, text }, signal);
  }

  steer(
    runtimeKind: RuntimeKind,
    conversationId: string,
    turnId: string,
    text: string,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.post("steer", { runtimeKind, conversationId, turnId, text }, signal);
  }

  interrupt(
    runtimeKind: RuntimeKind,
    conversationId: string,
    turnId: string,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.post("interrupt", { runtimeKind, conversationId, turnId }, signal);
  }

  retry(
    runtimeKind: RuntimeKind,
    conversationId: string,
    userItemId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.post("retry", { runtimeKind, conversationId, userItemId }, signal);
  }

  edit(
    runtimeKind: RuntimeKind,
    conversationId: string,
    userItemId: string,
    text: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.post("edit", { runtimeKind, conversationId, userItemId, text }, signal);
  }

  fork(
    runtimeKind: RuntimeKind,
    conversationId: string,
    beforeTurnId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary> {
    return this.post("fork", { runtimeKind, conversationId, beforeTurnId }, signal);
  }

  archive(runtimeKind: RuntimeKind, conversationId: string, signal?: AbortSignal): Promise<void> {
    return this.post("archive", { runtimeKind, conversationId }, signal);
  }

  approve(
    request: ApprovalRequest,
    decision: ApprovalDecision,
    fields?: ApprovalResponseFields,
    signal?: AbortSignal,
  ): Promise<void> {
    return this.post(
      "approval",
      {
        runtimeKind: request.runtime,
        conversationId: request.conversationId,
        turnId: request.turnId,
        itemId: request.itemId,
        approvalId: request.approvalId,
        decision,
        ...fields,
      },
      signal,
    );
  }

  subscribe(listener: (event: RuntimeEvent) => void, onError?: () => void): () => void {
    const source = this.eventSource(`${BASE_PATH}/events`);
    let lastSeq = 0;
    source.onmessage = (message) => {
      let event: RuntimeEvent;
      try {
        event = JSON.parse(message.data) as RuntimeEvent;
        if (!Number.isSafeInteger(event.seq) || event.seq <= lastSeq) {
          throw new Error(
            `Runtime event sequence ${String(event.seq)} must be greater than ${String(lastSeq)}.`,
          );
        }
        lastSeq = event.seq;
      } catch {
        onError?.();
        return;
      }
      listener(event);
    };
    source.onerror = () => onError?.();
    return () => source.close();
  }

  private async get<Value>(path: string, signal?: AbortSignal): Promise<Value> {
    return responseValue<Value>(
      await this.fetcher(path, signal === undefined ? undefined : { signal }),
    );
  }

  private async post<Value>(action: string, body: object, signal?: AbortSignal): Promise<Value> {
    return responseValue<Value>(
      await this.fetcher(`${BASE_PATH}/${action}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        ...(signal === undefined ? {} : { signal }),
      }),
    );
  }
}

export function approvalResponseFields(
  request: ApprovalRequest,
  decision: ApprovalDecision,
  values: Readonly<Record<string, ApprovalInputValue>>,
): ApprovalResponseFields | undefined {
  if (request.kind === "user_input" && decision === "submit") {
    return {
      answers: Object.fromEntries(
        (request.questions ?? []).map((question) => {
          const value = Object.hasOwn(values, question.id) ? values[question.id] : "";
          return [question.id, Array.isArray(value) ? value : [String(value)]];
        }),
      ),
    };
  }
  if (
    request.kind !== "elicitation" ||
    (decision !== "accept" && decision !== "accept_for_session")
  ) {
    return undefined;
  }
  const form: Array<readonly [string, unknown]> = [];
  for (const question of request.questions ?? []) {
    const value =
      (Object.hasOwn(values, question.id) ? values[question.id] : undefined) ??
      question.defaultValue ??
      (question.type === "boolean" && question.required === true ? false : undefined);
    if (value === undefined || value === "") {
      if (question.required !== true) continue;
      throw new Error(`"${question.header ?? question.prompt}" is required.`);
    }
    switch (question.type) {
      case "boolean":
        if (typeof value !== "boolean") {
          throw new Error(`"${question.header ?? question.prompt}" must be true or false.`);
        }
        form.push([question.id, value]);
        break;
      case "number":
      case "integer": {
        const numeric = typeof value === "string" && value !== "" ? Number(value) : value;
        if (
          typeof numeric !== "number" ||
          !Number.isFinite(numeric) ||
          (question.type === "integer" && !Number.isInteger(numeric))
        ) {
          throw new Error(
            `"${question.header ?? question.prompt}" must be a valid ${question.type}.`,
          );
        }
        form.push([question.id, numeric]);
        break;
      }
      case "string_array":
        if (!Array.isArray(value)) {
          throw new Error(`"${question.header ?? question.prompt}" must be a selection list.`);
        }
        form.push([question.id, value]);
        break;
      default:
        form.push([question.id, String(value)]);
    }
  }
  return { form: Object.fromEntries(form) };
}

function query(values: Record<string, string>): string {
  return new URLSearchParams(values).toString();
}

async function responseValue<Value>(response: Response): Promise<Value> {
  const value = (await response.json()) as unknown;
  if (!response.ok) {
    const message =
      typeof value === "object" &&
      value !== null &&
      "error" in value &&
      typeof value.error === "string"
        ? value.error
        : `Conversation runtime request failed (${String(response.status)}).`;
    throw new Error(message);
  }
  return value as Value;
}

/** Apply one disposable native event projection; native read remains the durable truth. */
export function projectRuntimeEvent(
  snapshot: ConversationSnapshot,
  event: RuntimeEvent,
): ConversationSnapshot {
  if (
    event.runtime !== snapshot.runtime ||
    event.conversationId !== snapshot.conversationId ||
    event.type === "approval_requested" ||
    event.type === "approval_resolved"
  ) {
    return snapshot;
  }
  if (event.type === "item_completed" || event.type === "item_delta") {
    if (event.type === "item_completed" && event.item.turnId !== event.turnId) return snapshot;
    const itemId = event.type === "item_completed" ? event.item.id : event.itemId;
    const priorItem = snapshot.turns
      .flatMap((turn) => turn.items)
      .find((item) => item.id === itemId);
    if (priorItem !== undefined) {
      if (priorItem.turnId !== event.turnId) return snapshot;
      const provisional =
        (priorItem.type === "assistant_message" || priorItem.type === "reasoning") &&
        priorItem.provisional === true;
      const runningToolCompletion =
        event.type === "item_completed" &&
        priorItem.type === "tool" &&
        priorItem.status === "running";
      if (!provisional && !runningToolCompletion) return snapshot;
    }
  }
  const turns = snapshot.turns.map((turn) => ({ ...turn, items: [...turn.items] }));
  let turn = turns.find((candidate) => candidate.id === event.turnId);
  if (turn === undefined) {
    turn = { id: event.turnId, status: "running", items: [] };
    turns.push(turn);
  }
  if (event.type === "turn_status") {
    turn.status = event.status;
  } else if (event.type === "item_completed") {
    const index = turn.items.findIndex((item) => item.id === event.item.id);
    if (index === -1) turn.items.push(event.item);
    else turn.items[index] = event.item;
  } else {
    const index = turn.items.findIndex((item) => item.id === event.itemId);
    const existing = index === -1 ? undefined : turn.items[index];
    if (
      existing !== undefined &&
      (existing.type === "assistant_message" || existing.type === "reasoning") &&
      existing.provisional === true
    ) {
      turn.items[index] = { ...existing, text: existing.text + event.delta, provisional: true };
    } else if (existing === undefined) {
      turn.items.push({
        type: event.itemType ?? "assistant_message",
        id: event.itemId,
        turnId: event.turnId,
        text: event.delta,
        createdAt: Date.now(),
        provisional: true,
      });
    }
  }
  return { ...snapshot, turns };
}

export function projectApprovalEvent(
  approvals: readonly ApprovalRequest[],
  event: RuntimeEvent,
): readonly ApprovalRequest[] {
  if (event.type === "approval_requested") {
    return [...removeApprovalRequest(approvals, event), event];
  }
  if (event.type === "approval_resolved") {
    return removeApprovalRequest(approvals, event);
  }
  return approvals;
}

export function approvalIdentityKey(value: ApprovalIdentity): string {
  return JSON.stringify(approvalIdentityParts(value));
}

export function approvalAnswerKey(value: ApprovalIdentity, questionId: string): string {
  return JSON.stringify([...approvalIdentityParts(value), questionId]);
}

export function removeApprovalRequest(
  approvals: readonly ApprovalRequest[],
  resolved: ApprovalIdentity,
): readonly ApprovalRequest[] {
  const resolvedKey = approvalIdentityKey(resolved);
  return approvals.filter((approval) => approvalIdentityKey(approval) !== resolvedKey);
}

export function removeApprovalAnswers(
  answers: Readonly<Record<string, ApprovalInputValue>>,
  resolved: ApprovalIdentity,
): Readonly<Record<string, ApprovalInputValue>> {
  const prefix = `${JSON.stringify(approvalIdentityParts(resolved)).slice(0, -1)},`;
  return Object.fromEntries(Object.entries(answers).filter(([key]) => !key.startsWith(prefix)));
}

function approvalIdentityParts(
  value: ApprovalIdentity,
): [RuntimeKind, string, string, string, string] {
  return [value.runtime, value.conversationId, value.turnId, value.itemId, value.approvalId];
}

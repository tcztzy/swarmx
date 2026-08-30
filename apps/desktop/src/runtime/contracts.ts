export type RuntimeKind = "dsh" | "codex";

export interface WorkspaceSummary {
  id: string;
  label: string;
}

export interface WorkspaceScope extends WorkspaceSummary {
  root: string;
  token: string;
}

export interface ConversationSummary {
  runtime: RuntimeKind;
  conversationId: string;
  workspace: WorkspaceSummary;
  title: string;
  archived: boolean;
  updatedAt: number;
}

interface ConversationItemBase {
  id: string;
  turnId: string;
  createdAt: number;
}

export interface UserMessageItem extends ConversationItemBase {
  type: "user_message";
  text: string;
}

export interface AssistantMessageItem extends ConversationItemBase {
  type: "assistant_message";
  text: string;
  provisional?: true;
}

export interface ReasoningItem extends ConversationItemBase {
  type: "reasoning";
  text: string;
  provisional?: true;
}

export interface ToolItem extends ConversationItemBase {
  type: "tool";
  name: string;
  status: "running" | "completed" | "failed";
  summary?: string;
}

export interface ErrorItem extends ConversationItemBase {
  type: "error";
  message: string;
}

export type ConversationItem =
  | UserMessageItem
  | AssistantMessageItem
  | ReasoningItem
  | ToolItem
  | ErrorItem;

export type TurnStatus = "running" | "completed" | "interrupted" | "failed";

export interface ConversationTurn {
  id: string;
  status: TurnStatus;
  items: ConversationItem[];
}

export interface ConversationSnapshot {
  runtime: RuntimeKind;
  conversationId: string;
  workspace: WorkspaceSummary;
  title: string;
  archived: boolean;
  turns: ConversationTurn[];
  approvals?: ApprovalRequest[];
}

interface RuntimeEventBase {
  seq: number;
  runtime: RuntimeKind;
  conversationId: string;
}

export interface ItemDeltaEvent extends RuntimeEventBase {
  type: "item_delta";
  turnId: string;
  itemId: string;
  delta: string;
  itemType?: "assistant_message" | "reasoning";
}

export interface ItemCompletedEvent extends RuntimeEventBase {
  type: "item_completed";
  turnId: string;
  item: ConversationItem;
}

export interface TurnStatusEvent extends RuntimeEventBase {
  type: "turn_status";
  turnId: string;
  status: TurnStatus;
}

export type ApprovalKind = "command" | "file_change" | "permissions" | "elicitation" | "user_input";

export type ApprovalDecision = "accept" | "accept_for_session" | "decline" | "cancel" | "submit";

export interface ApprovalQuestion {
  id: string;
  prompt: string;
  header?: string;
  type?: "string" | "boolean" | "number" | "integer" | "string_array";
  required?: boolean;
  defaultValue?: string | boolean | number | readonly string[];
  options?: readonly string[];
  multiSelect?: boolean;
  minimum?: number;
  maximum?: number;
  minLength?: number;
  maxLength?: number;
  minItems?: number;
  maxItems?: number;
  format?: "date" | "uri" | "email" | "date-time";
}

export interface ApprovalRequest {
  runtime: RuntimeKind;
  conversationId: string;
  turnId: string;
  itemId: string;
  approvalId: string;
  kind: ApprovalKind;
  prompt: string;
  choices: readonly ApprovalDecision[];
  questions?: readonly ApprovalQuestion[];
}

export interface ApprovalRequestedEvent extends RuntimeEventBase, ApprovalRequest {
  type: "approval_requested";
}

export interface ApprovalResolvedEvent extends RuntimeEventBase {
  type: "approval_resolved";
  turnId: string;
  itemId: string;
  approvalId: string;
}

export type RuntimeEvent =
  | ItemDeltaEvent
  | ItemCompletedEvent
  | TurnStatusEvent
  | ApprovalRequestedEvent
  | ApprovalResolvedEvent;

export interface ApprovalResponse {
  runtime: RuntimeKind;
  conversationId: string;
  turnId: string;
  itemId: string;
  approvalId: string;
  decision: ApprovalDecision;
  answers?: Readonly<Record<string, readonly string[]>>;
  form?: Readonly<Record<string, unknown>>;
}

export interface CreateConversationRequest {
  workspace: WorkspaceScope;
  model?: string;
}

export interface StartTurnRequest {
  conversationId: string;
  text: string;
}

export interface SteerTurnRequest extends StartTurnRequest {
  turnId: string;
}

export interface InterruptTurnRequest {
  conversationId: string;
  turnId: string;
}

export interface ForkConversationRequest {
  conversationId: string;
  beforeTurnId: string;
}

export interface ReviseConversationRequest extends ForkConversationRequest {
  text: string;
}

export type RuntimeEventListener = (event: RuntimeEvent) => void;

export interface ConversationRuntime {
  readonly kind: RuntimeKind;
  list(signal?: AbortSignal): Promise<ConversationSummary[]>;
  create(request: CreateConversationRequest, signal?: AbortSignal): Promise<ConversationSummary>;
  createProvisionedMember?(
    request: CreateConversationRequest,
    provisioningId: string,
    signal?: AbortSignal,
  ): Promise<ConversationSummary>;
  retireProvisionedMember?(
    conversationId: string,
    provisioningId: string,
    signal?: AbortSignal,
  ): Promise<void>;
  read(conversationId: string, signal?: AbortSignal): Promise<ConversationSnapshot>;
  start(request: StartTurnRequest, signal?: AbortSignal): Promise<{ turnId: string }>;
  steer(request: SteerTurnRequest, signal?: AbortSignal): Promise<void>;
  interrupt(request: InterruptTurnRequest, signal?: AbortSignal): Promise<void>;
  revise(request: ReviseConversationRequest, signal?: AbortSignal): Promise<ConversationSummary>;
  fork(request: ForkConversationRequest, signal?: AbortSignal): Promise<ConversationSummary>;
  archive(conversationId: string, signal?: AbortSignal): Promise<void>;
  subscribe(listener: RuntimeEventListener): () => void;
  respondToApproval(response: ApprovalResponse): Promise<void>;
  dispose(): Promise<void>;
}

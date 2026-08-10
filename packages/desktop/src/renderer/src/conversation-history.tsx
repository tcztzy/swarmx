import {
  type NormalizedRenderEvent,
  type NormalizeMessageChunkOptions,
  normalizeMessageChunk,
  type RenderArtifactReference,
  type RenderProvenance,
} from "@swarmx/core/rendering";
import { cva, type VariantProps } from "class-variance-authority";
import {
  Bot,
  Brain,
  Check,
  ChevronRight,
  Code2,
  GitFork,
  Loader2,
  type LucideIcon,
  Minus,
  Pencil,
  Plus,
  RefreshCw,
  Sparkles,
  Terminal as TerminalIcon,
  User,
  XCircle,
} from "lucide-react";
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type {
  DesktopMediaAttachment,
  DesktopMessageChunk as MessageChunk,
} from "../../shared/desktop-api.js";
import { mergeToolProgress, messageKey } from "./conversation-messages.js";
import { MessageAttachments } from "./message-attachments.js";
import { MESSAGE_ACTION_CLASS_NAME, MessageContent, MessageCopyButton } from "./message-content.js";
import { formatFullMessageTimestamp, formatMessageTimestamp, isRecord } from "./text-utils.js";
import { cx } from "./ui-primitives.js";

interface ProviderErrorNotice {
  type: "provider_error";
  code: "overloaded" | "rate_limited" | "temporarily_unavailable";
  title: string;
  message: string;
  retryable: boolean;
}

interface ConversationTurn {
  id: string;
  userMessage: MessageChunk | null;
  userMessageIndex: number | null;
  workMessages: MessageChunk[];
  finalMessage: MessageChunk | null;
  finalMessageIndex: number | null;
}

export interface MessageEditState {
  messageIndex: number;
  draft: string;
  error: string | null;
  expectedMessages: MessageChunk[];
}

interface ToolActivity {
  call?: MessageChunk;
  progress?: MessageChunk;
  result?: MessageChunk;
  sourceIndex: number;
}

type WorkActivity =
  | { kind: "message"; message: MessageChunk; sourceIndex: number }
  | { kind: "tool"; activity: ToolActivity };

const runEventVariants = cva(
  "run-event group [display:grid] [grid-template-columns:34px_minmax(0,_1fr)] [gap:12px] [align-items:start] [animation:event-enter_var(--duration-med)_var(--ease-out)_both] max-680:[grid-template-columns:28px_minmax(0,_1fr)] max-680:[gap:9px]",
  {
    variants: {
      tone: {
        user: String.raw`run-event--user [&_.run-event\_\_header]:[color:rgba(9,_9,_11,_0.58)]`,
        system: "run-event--system",
        thinking: "run-event--thinking",
        tool: "run-event--tool",
        assistant: "run-event--assistant",
      },
      compact: {
        true: "run-event--compact",
        false: null,
      },
    },
    defaultVariants: {
      compact: false,
    },
  },
);

type RunEventTone = NonNullable<VariantProps<typeof runEventVariants>["tone"]>;

const traceStatusVariants = cva(
  "trace-card__status [flex:0_0_auto] [padding:2px_6px] [border:1px_solid_var(--border-subtle)] [border-radius:999px] [font-family:var(--font-mono)] [font-size:10px] [text-transform:none]",
  {
    variants: {
      status: {
        queued:
          "trace-card__status--queued [color:var(--muted)] [background:rgba(255,_255,_255,_0.05)]",
        running:
          "trace-card__status--running [color:var(--muted)] [background:rgba(255,_255,_255,_0.05)]",
        succeeded:
          "trace-card__status--succeeded [color:var(--success)] [background:rgba(52,_211,_153,_0.1)] [border-color:rgba(52,_211,_153,_0.22)]",
        failed:
          "trace-card__status--failed [color:var(--danger)] [background:var(--danger-muted)] [border-color:rgba(248,_113,_113,_0.24)]",
        canceled:
          "trace-card__status--canceled [color:var(--muted)] [background:rgba(255,_255,_255,_0.05)]",
        skipped:
          "trace-card__status--skipped [color:var(--muted)] [background:rgba(255,_255,_255,_0.05)]",
        completed:
          "trace-card__status--completed [color:var(--success)] [background:rgba(52,_211,_153,_0.1)] [border-color:rgba(52,_211,_153,_0.22)]",
      },
    },
  },
);

function RunEvent({
  actionsDisabled = false,
  createdAt,
  compact = false,
  msg,
  onChangeModel,
  onContinueInNewChat,
  onEdit,
  onPreviewAttachment,
  onRetry,
}: {
  actionsDisabled?: boolean;
  createdAt?: string;
  compact?: boolean;
  msg: MessageChunk;
  onChangeModel?: () => void;
  onContinueInNewChat?: () => void;
  onEdit?: () => void;
  onPreviewAttachment?: (attachment: DesktopMediaAttachment) => void;
  onRetry?: () => void;
}) {
  const providerNotice = providerErrorNotice(msg);
  if (providerNotice && !compact) {
    return (
      <ProviderErrorEvent
        notice={providerNotice}
        actionsDisabled={actionsDisabled}
        onRetry={onRetry}
        onChangeModel={onChangeModel}
      />
    );
  }
  const renderEvent = normalizeMessageChunk(msg, normalizeOptionsFromMessage(msg));
  const {
    icon: Icon,
    label,
    tone,
    meta,
  } = compact ? compactWorkPresentation(msg) : messagePresentation(msg);
  const content = renderEventContent(msg, renderEvent);
  const showTraceCard = isTraceCardEvent(renderEvent);
  const plainNarrative = compact && (msg.kind === "thinking" || msg.kind === "message");
  const showMessageCopy =
    !compact &&
    msg.kind === "message" &&
    (msg.role === "user" || msg.role === "assistant") &&
    content.length > 0;
  const showMessageEdit =
    !compact && msg.kind === "message" && msg.role === "user" && onEdit !== undefined;
  const showContinueInNewChat =
    !compact &&
    msg.kind === "message" &&
    msg.role === "assistant" &&
    onContinueInNewChat !== undefined;
  const messageTimestamp = createdAt ?? messageCreatedAt(msg);
  const displayContent = msg.kind === "thinking" ? normalizeThoughtMarkdown(content) : content;

  return (
    <article
      className={runEventVariants({ tone, compact })}
      data-render-event-id={renderEvent.eventId}
      data-render-kind={renderEvent.kind}
      data-render-status={renderEvent.status}
    >
      {plainNarrative ? (
        <div className="run-event__content [min-width:0] [white-space:pre-wrap] [overflow-wrap:anywhere] [font-size:14px] [line-height:1.58] [.run-event--tool_&]:[color:#d4d4d8] [.run-event--tool_&]:[font-family:var(--font-mono)] [.run-event--tool_&]:[font-size:12.5px] [.run-event--tool_&]:[line-height:1.55]">
          <MessageContent kind={msg.kind} content={displayContent} />
        </div>
      ) : (
        <>
          <div className="run-event__rail [position:relative] [width:34px] [height:34px] [display:grid] [place-items:center] [color:var(--muted)] [background:var(--card)] [border:1px_solid_var(--border)] [border-radius:var(--radius)] [box-shadow:var(--shadow-inset)] [-webkit-backdrop-filter:saturate(140%)_blur(16px)] [&_svg]:[width:16px] [&_svg]:[height:16px] max-680:[width:28px] max-680:[height:28px]">
            <Icon aria-hidden="true" />
          </div>
          <div className="run-event__card [min-width:0] [padding:14px_15px] [color:var(--foreground)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow-inset),_0_12px_34px_rgba(0,_0,_0,_0.18)] [-webkit-backdrop-filter:saturate(140%)_blur(18px)] [transition:border-color_var(--duration-fast)_var(--ease-out),_background-color_var(--duration-fast)_var(--ease-out),_box-shadow_var(--duration-fast)_var(--ease-out)] [.run-event--system_&]:[color:var(--danger)] [.run-event--system_&]:[background:var(--danger-muted)] [.run-event--system_&]:[border-color:rgba(248,_113,_113,_0.28)] [.run-event--thinking_&]:[color:var(--muted)] [.run-event--thinking_&]:[background:rgba(255,_255,_255,_0.04)] [.run-event--tool_&]:[background:rgba(8,_10,_14,_0.84)] [.run-event--tool_&]:[border-color:var(--border)] max-680:[padding:11px]">
            <div className="run-event__header [margin-bottom:8px] [display:flex] [align-items:center] [justify-content:space-between] [gap:10px] [color:var(--muted-foreground)] [font-size:11px] [font-weight:700] [line-height:1.2] [text-transform:uppercase]">
              <span>{label}</span>
              <span>{renderEvent.status === "completed" ? meta : renderEvent.status}</span>
            </div>
            {msg.toolName && (
              <div className="run-event__tool [margin-bottom:8px] [display:inline-flex] [max-width:100%] [align-items:center] [gap:6px] [color:var(--muted)] [background:rgba(255,_255,_255,_0.048)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [padding:4px_7px] [font-family:var(--font-mono)] [font-size:12px] [line-height:1.25] [&_svg]:[flex:0_0_auto] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_span]:[overflow:hidden] [&_span]:[text-overflow:ellipsis] [&_span]:[white-space:nowrap]">
                <Code2 aria-hidden="true" />
                <span>{msg.toolName}</span>
                <span className="run-event__tool-status [color:var(--muted-foreground)]">
                  {renderEvent.status}
                </span>
              </div>
            )}
            {msg.swarmEvent && (
              <div className="run-event__event [margin-bottom:8px] [display:inline-flex] [max-width:100%] [align-items:center] [gap:6px] [color:var(--muted)] [background:rgba(255,_255,_255,_0.048)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [padding:4px_7px] [font-family:var(--font-mono)] [font-size:12px] [line-height:1.25] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
                {msg.swarmEvent}
              </div>
            )}
            <div className="run-event__content [min-width:0] [white-space:pre-wrap] [overflow-wrap:anywhere] [font-size:14px] [line-height:1.58] [.run-event--tool_&]:[color:#d4d4d8] [.run-event--tool_&]:[font-family:var(--font-mono)] [.run-event--tool_&]:[font-size:12.5px] [.run-event--tool_&]:[line-height:1.55]">
              <MessageContent kind={msg.kind} content={content} />
            </div>
            {msg.attachments && msg.attachments.length > 0 && (
              <MessageAttachments attachments={msg.attachments} onPreview={onPreviewAttachment} />
            )}
            {showTraceCard && <TraceCard event={renderEvent} />}
          </div>
          {(showMessageCopy || showMessageEdit || showContinueInNewChat) && (
            <div className="run-event__actions pointer-events-none opacity-0 transition-opacity duration-[var(--duration-fast)] ease-[var(--ease-out)] group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100">
              {msg.role === "user" && messageTimestamp && (
                <MessageTimestamp createdAt={messageTimestamp} />
              )}
              {showMessageCopy && <MessageCopyButton content={content} />}
              {showMessageEdit && (
                <button
                  aria-label="Edit message"
                  className={MESSAGE_ACTION_CLASS_NAME}
                  disabled={actionsDisabled}
                  onClick={onEdit}
                  title="Edit message"
                  type="button"
                >
                  <Pencil aria-hidden="true" />
                </button>
              )}
              {showContinueInNewChat && (
                <button
                  aria-label="Continue in new chat"
                  className={MESSAGE_ACTION_CLASS_NAME}
                  disabled={actionsDisabled}
                  onClick={onContinueInNewChat}
                  title="Continue in new chat"
                  type="button"
                >
                  <GitFork aria-hidden="true" />
                </button>
              )}
              {msg.role === "assistant" && messageTimestamp && (
                <MessageTimestamp createdAt={messageTimestamp} />
              )}
            </div>
          )}
        </>
      )}
    </article>
  );
}

function MessageTimestamp({ createdAt }: { createdAt: string }) {
  const label = formatMessageTimestamp(createdAt);
  const fullLabel = formatFullMessageTimestamp(createdAt);
  if (!label || !fullLabel) return null;

  return (
    <time
      aria-label={`Created ${fullLabel}`}
      className="run-event__timestamp [min-width:max-content] [color:var(--muted-foreground)] [font-size:11.5px] [line-height:28px] [opacity:0] [visibility:hidden] [white-space:nowrap] [transition:opacity_var(--duration-fast)_var(--ease-out)]"
      dateTime={createdAt}
      title={fullLabel}
    >
      {label}
    </time>
  );
}

function EditableUserMessage({
  actionsDisabled,
  draft,
  error,
  hasAttachments,
  onCancel,
  onChange,
  onSubmit,
}: {
  actionsDisabled: boolean;
  draft: string;
  error: string | null;
  hasAttachments: boolean;
  onCancel: () => void;
  onChange: (draft: string) => void;
  onSubmit: () => void;
}) {
  const noteId = useId();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const canSubmit = !actionsDisabled && (draft.trim().length > 0 || hasAttachments);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  return (
    <article
      className={runEventVariants({
        tone: "user",
        className: "run-event--editing",
      })}
    >
      <div className="run-event__card [min-width:0] [padding:14px_15px] [color:var(--foreground)] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow-inset),_0_12px_34px_rgba(0,_0,_0,_0.18)] [-webkit-backdrop-filter:saturate(140%)_blur(18px)] [transition:border-color_var(--duration-fast)_var(--ease-out),_background-color_var(--duration-fast)_var(--ease-out),_box-shadow_var(--duration-fast)_var(--ease-out)] [.run-event--system_&]:[color:var(--danger)] [.run-event--system_&]:[background:var(--danger-muted)] [.run-event--system_&]:[border-color:rgba(248,_113,_113,_0.28)] [.run-event--thinking_&]:[color:var(--muted)] [.run-event--thinking_&]:[background:rgba(255,_255,_255,_0.04)] [.run-event--tool_&]:[background:rgba(8,_10,_14,_0.84)] [.run-event--tool_&]:[border-color:var(--border)] max-680:[padding:11px]">
        <form
          className="message-editor [min-width:0] [&_textarea]:[width:100%] [&_textarea]:[min-height:92px] [&_textarea]:[max-height:260px] [&_textarea]:[resize:vertical] [&_textarea]:[padding:10px_11px] [&_textarea]:[color:var(--foreground)] [&_textarea]:[background:var(--input)] [&_textarea]:[border:1px_solid_var(--border)] [&_textarea]:[border-radius:11px] [&_textarea]:[font-size:14.5px] [&_textarea]:[line-height:1.55]"
          onSubmit={(event) => {
            event.preventDefault();
            if (canSubmit) onSubmit();
          }}
        >
          <textarea
            aria-describedby={noteId}
            aria-label="Edit message"
            disabled={actionsDisabled}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                if (!actionsDisabled) onCancel();
                return;
              }
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                if (canSubmit) onSubmit();
              }
            }}
            rows={3}
            ref={textareaRef}
            value={draft}
          />
          <div className="message-editor__footer [margin-top:9px] [display:flex] [align-items:center] [justify-content:space-between] [gap:9px_14px] [flex-wrap:wrap]">
            <p
              id={noteId}
              className="message-editor__note [flex:1_1_240px] [color:var(--muted)] [margin:0] [font-size:11.5px] [line-height:1.45]"
            >
              This replaces the latest turn and generates a new reply.
            </p>
            <div className="message-editor__actions [display:flex] [align-items:center] [justify-content:flex-end] [gap:7px] [&_button]:[min-height:30px] [&_button]:[padding:5px_10px] [&_button]:[display:inline-flex] [&_button]:[align-items:center] [&_button]:[justify-content:center] [&_button]:[gap:6px] [&_button]:[color:var(--foreground)] [&_button]:[background:transparent] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:8px] [&_button]:[cursor:pointer] [&_button]:[font-size:12px] [&_button]:[font-weight:620] [&_svg]:[width:13px] [&_svg]:[height:13px] [&_svg]:[animation:spin_900ms_linear_infinite]">
              <button type="button" disabled={actionsDisabled} onClick={onCancel}>
                Cancel
              </button>
              <button type="submit" disabled={!canSubmit}>
                {actionsDisabled && <Loader2 aria-hidden="true" />}
                Save &amp; resend
              </button>
            </div>
          </div>
          {error && (
            <p
              className="message-editor__error [margin-top:8px] [color:var(--danger)] [margin:0] [font-size:11.5px] [line-height:1.45]"
              role="alert"
            >
              {error}
            </p>
          )}
        </form>
      </div>
    </article>
  );
}

function ProviderErrorEvent({
  actionsDisabled,
  notice,
  onChangeModel,
  onRetry,
}: {
  actionsDisabled: boolean;
  notice: ProviderErrorNotice;
  onChangeModel?: () => void;
  onRetry?: () => void;
}) {
  return (
    <article
      className="provider-error-notice [min-width:0] [padding:18px] [display:grid] [grid-template-columns:36px_minmax(0,_1fr)] [gap:13px] [color:var(--foreground)] [background:color-mix(in_srgb,_var(--accent)_5%,_var(--card))] [border:1px_solid_color-mix(in_srgb,_var(--accent)_26%,_var(--border-subtle))] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow-inset),_0_12px_34px_rgba(0,_0,_0,_0.12)] [&_h3]:[margin:0] [&_p]:[margin:0] [&_h3]:[font-size:15px] [&_h3]:[font-weight:650] [&_h3]:[line-height:1.35] [&_p]:[margin-top:6px] [&_p]:[color:var(--muted)] [&_p]:[font-size:13.5px] [&_p]:[line-height:1.55] max-680:[padding:15px] max-680:[grid-template-columns:32px_minmax(0,_1fr)] max-680:[gap:11px]"
      aria-label={notice.title}
      data-provider-error-code={notice.code}
      role="alert"
    >
      <div
        className="provider-error-notice__icon [width:36px] [height:36px] [display:grid] [place-items:center] [color:var(--accent)] [background:var(--accent-muted)] [border-radius:10px] [&_svg]:[width:17px] [&_svg]:[height:17px] max-680:[width:32px] max-680:[height:32px]"
        aria-hidden="true"
      >
        <RefreshCw />
      </div>
      <div className="provider-error-notice__body [min-width:0]">
        <span className="provider-error-notice__eyebrow [display:block] [margin:1px_0_4px] [color:var(--muted-foreground)] [font-size:10px] [font-weight:720] [letter-spacing:0.09em] [text-transform:uppercase]">
          Provider issue
        </span>
        <h3>{notice.title}</h3>
        <p>{notice.message}</p>
        {(onChangeModel || (notice.retryable && onRetry)) && (
          <div className="provider-error-notice__actions [margin-top:14px] [display:flex] [flex-wrap:wrap] [gap:8px] [&_button]:[min-height:32px] [&_button]:[padding:0_11px] [&_button]:[display:inline-flex] [&_button]:[align-items:center] [&_button]:[gap:7px] [&_button]:[color:var(--foreground)] [&_button]:[background:var(--card-hover)] [&_button]:[border:1px_solid_var(--border)] [&_button]:[border-radius:8px] [&_button]:[cursor:pointer] [&_button]:[font:inherit] [&_button]:[font-size:12.5px] [&_button]:[font-weight:590] [&_svg]:[width:14px] [&_svg]:[height:14px] max-680:[align-items:stretch] max-680:[flex-direction:column] max-680:[&_button]:[justify-content:center]">
            {notice.retryable && onRetry && (
              <button type="button" disabled={actionsDisabled} onClick={onRetry}>
                <RefreshCw aria-hidden="true" />
                Try again
              </button>
            )}
            {onChangeModel && (
              <button type="button" disabled={actionsDisabled} onClick={onChangeModel}>
                <Sparkles aria-hidden="true" />
                Change model
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  );
}

export function ConversationHistory({
  activeRunStartedAt,
  actionsDisabled,
  messageEdit,
  messages,
  onBeginEdit,
  onCancelEdit,
  onChangeModel,
  onContinue,
  onContinueInNewChat,
  onEditDraftChange,
  onPreviewAttachment,
  onRetry,
  onSubmitEdit,
  running,
}: {
  activeRunStartedAt: number | null;
  actionsDisabled: boolean;
  messageEdit: MessageEditState | null;
  messages: MessageChunk[];
  onBeginEdit?: (messageIndex: number, content: string) => void;
  onCancelEdit: () => void;
  onChangeModel?: () => void;
  onContinue?: () => void;
  onContinueInNewChat?: (messageIndex: number) => void;
  onEditDraftChange: (draft: string) => void;
  onPreviewAttachment?: (attachment: DesktopMediaAttachment) => void;
  onRetry: (userText: string, attachments?: DesktopMediaAttachment[]) => void;
  onSubmitEdit: (messageIndex: number, content: string) => void;
  running: boolean;
}) {
  const turns = useMemo(() => groupConversationTurns(messages), [messages]);

  return turns.map((turn, index) => {
    const latest = index === turns.length - 1;
    const active = running && latest;
    const visibleTurn =
      active && turn.finalMessage
        ? {
            ...turn,
            workMessages: [...turn.workMessages, turn.finalMessage],
            finalMessage: null,
          }
        : turn;
    const interrupted =
      !active &&
      !visibleTurn.finalMessage &&
      (visibleTurn.userMessage !== null || visibleTurn.workMessages.length > 0);
    return (
      <ConversationTurnView
        active={active}
        activeRunStartedAt={active ? activeRunStartedAt : null}
        actionsDisabled={actionsDisabled}
        interrupted={interrupted}
        key={turn.id}
        latest={latest}
        messageEdit={messageEdit}
        onBeginEdit={onBeginEdit}
        onCancelEdit={onCancelEdit}
        onChangeModel={onChangeModel}
        onContinue={onContinue}
        onContinueInNewChat={onContinueInNewChat}
        onEditDraftChange={onEditDraftChange}
        onPreviewAttachment={onPreviewAttachment}
        onRetry={onRetry}
        onSubmitEdit={onSubmitEdit}
        turn={visibleTurn}
      />
    );
  });
}

function ConversationTurnView({
  active,
  activeRunStartedAt,
  actionsDisabled,
  interrupted,
  latest,
  messageEdit,
  onBeginEdit,
  onCancelEdit,
  onChangeModel,
  onContinue,
  onContinueInNewChat,
  onEditDraftChange,
  onPreviewAttachment,
  onRetry,
  onSubmitEdit,
  turn,
}: {
  active: boolean;
  activeRunStartedAt: number | null;
  actionsDisabled: boolean;
  interrupted: boolean;
  latest: boolean;
  messageEdit: MessageEditState | null;
  onBeginEdit?: (messageIndex: number, content: string) => void;
  onCancelEdit: () => void;
  onChangeModel?: () => void;
  onContinue?: () => void;
  onContinueInNewChat?: (messageIndex: number) => void;
  onEditDraftChange: (draft: string) => void;
  onPreviewAttachment?: (attachment: DesktopMediaAttachment) => void;
  onRetry: (userText: string, attachments?: DesktopMediaAttachment[]) => void;
  onSubmitEdit: (messageIndex: number, content: string) => void;
  turn: ConversationTurn;
}) {
  const memoryReceipts = turn.workMessages.flatMap((message, sourceIndex) =>
    isPersonalMemoryReceipt(message) ? [{ message, sourceIndex }] : [],
  );
  const collapsibleWork = turn.workMessages.filter((message) => !isPersonalMemoryReceipt(message));
  const hasWork = active || interrupted || collapsibleWork.length > 0;
  const retryText = turn.userMessage?.content ?? "";
  const retryAttachments = turn.userMessage?.attachments;
  const canRetry = Boolean(retryText || retryAttachments?.length);
  const turnStatus = active ? "running" : interrupted ? "interrupted" : "completed";
  const editing =
    turn.userMessageIndex !== null && messageEdit?.messageIndex === turn.userMessageIndex;
  const userCreatedAt = turn.userMessage ? turnUserMessageCreatedAt(turn) : undefined;

  return (
    <section
      className={String.raw`conversation-turn [min-width:0] [display:flow-root] [animation:event-enter_var(--duration-med)_var(--ease-out)_both] [&_+_.conversation-turn]:[margin-top:48px] [&_>_.run-event--user_+_.work-disclosure]:[margin-top:32px] [&_>_.run-event--user_+_.run-event]:[margin-top:32px] [&_>_.work-disclosure_+_.run-event]:[margin-top:26px] [&_>_.run-event--user]:[display:flex] [&_>_.run-event--user]:[align-items:flex-end] [&_>_.run-event--user]:[flex-direction:column] [&_>_.run-event--assistant]:[display:block] [&_>_.run-event--system]:[display:block] [&_>_.run-event_>_.run-event\_\_rail]:[display:none] [&_>_.run-event--assistant_>_.run-event\_\_card]:[padding:0] [&_>_.run-event--assistant_>_.run-event\_\_card]:[background:transparent] [&_>_.run-event--assistant_>_.run-event\_\_card]:[border:0] [&_>_.run-event--assistant_>_.run-event\_\_card]:[border-radius:0] [&_>_.run-event--assistant_>_.run-event\_\_card]:[box-shadow:none] [&_>_.run-event--assistant_>_.run-event\_\_card]:[-webkit-backdrop-filter:none] [&_>_.run-event--assistant_.run-event\_\_header]:[display:none] [&_>_.run-event--user_.run-event\_\_header]:[display:none] [&_>_.run-event--assistant_.run-event\_\_content]:[font-size:15px] [&_>_.run-event--assistant_.run-event\_\_content]:[line-height:1.72] [&_>_.run-event--user_.run-event\_\_card]:[width:fit-content] [&_>_.run-event--user_.run-event\_\_card]:[max-width:min(76%,_680px)] [&_>_.run-event--user_.run-event\_\_card]:[padding:10px_16px] [&_>_.run-event--user_.run-event\_\_card]:[color:var(--user-message-foreground)] [&_>_.run-event--user_.run-event\_\_card]:[background:var(--user-message-background)] [&_>_.run-event--user_.run-event\_\_card]:[border:0] [&_>_.run-event--user_.run-event\_\_card]:[border-radius:22px] [&_>_.run-event--user_.run-event\_\_card]:[box-shadow:none] [&_>_.run-event--user_.run-event\_\_card]:[-webkit-backdrop-filter:none] [&_>_.run-event--user_.run-event\_\_content]:[font-size:14.5px] [&_>_.run-event--user_.run-event\_\_content]:[line-height:1.55] [&_>_.run-event--assistant_.run-event\_\_actions]:[margin-top:5px] [&_>_.run-event--assistant_.run-event\_\_actions]:[display:flex] [&_>_.run-event--assistant_.run-event\_\_actions]:[align-items:center] [&_>_.run-event--assistant_.run-event\_\_actions]:[gap:4px] [&_>_.run-event--user_.run-event\_\_actions]:[margin-top:5px] [&_>_.run-event--user_.run-event\_\_actions]:[display:flex] [&_>_.run-event--user_.run-event\_\_actions]:[align-items:center] [&_>_.run-event--user_.run-event\_\_actions]:[gap:4px] [&_>_.run-event--assistant_.run-event\_\_actions]:[justify-content:flex-start] [&_>_.run-event--user_.run-event\_\_actions]:[justify-content:flex-end] [&_>_.run-event--editing_.run-event\_\_card]:[width:min(100%,_680px)] [&_>_.run-event--editing_.run-event\_\_card]:[max-width:88%] [&_>_.run-event--editing_.run-event\_\_card]:[padding:12px] [&_>_.run-event--editing_.run-event\_\_card]:[border-radius:16px] max-680:[&_+_.conversation-turn]:[margin-top:38px] max-680:[&_>_.run-event--user_+_.work-disclosure]:[margin-top:26px] max-680:[&_>_.run-event--user_+_.run-event]:[margin-top:26px] max-680:[&_>_.run-event--user_.run-event\_\_card]:[max-width:88%]`}
      data-turn-status={turnStatus}
    >
      {turn.userMessage &&
        (editing ? (
          <EditableUserMessage
            actionsDisabled={actionsDisabled}
            draft={messageEdit.draft}
            error={messageEdit.error}
            hasAttachments={Boolean(turn.userMessage.attachments?.length)}
            onCancel={onCancelEdit}
            onChange={onEditDraftChange}
            onSubmit={() => onSubmitEdit(turn.userMessageIndex as number, messageEdit.draft)}
          />
        ) : (
          <RunEvent
            actionsDisabled={actionsDisabled}
            createdAt={userCreatedAt}
            msg={turn.userMessage}
            onPreviewAttachment={onPreviewAttachment}
            onEdit={
              latest && onBeginEdit && turn.userMessageIndex !== null
                ? () =>
                    onBeginEdit(turn.userMessageIndex as number, turn.userMessage?.content ?? "")
                : undefined
            }
          />
        ))}
      {memoryReceipts.map(({ message, sourceIndex }) => (
        <RunEvent key={`${messageKey(message)}\u001fmemory-${sourceIndex}`} msg={message} />
      ))}
      {hasWork && (
        <WorkDisclosure
          active={active}
          activeRunStartedAt={activeRunStartedAt}
          actionsDisabled={actionsDisabled}
          interrupted={interrupted}
          latest={latest}
          messages={collapsibleWork}
          onContinue={onContinue}
          turnId={turn.id}
        />
      )}
      {turn.finalMessage && (
        <RunEvent
          actionsDisabled={actionsDisabled}
          msg={turn.finalMessage}
          onChangeModel={onChangeModel}
          onContinueInNewChat={
            onContinueInNewChat && turn.finalMessageIndex !== null
              ? () => onContinueInNewChat(turn.finalMessageIndex as number)
              : undefined
          }
          onPreviewAttachment={onPreviewAttachment}
          onRetry={canRetry ? () => onRetry(retryText, retryAttachments) : undefined}
        />
      )}
    </section>
  );
}

function isPersonalMemoryReceipt(message: MessageChunk): boolean {
  return message.render?.source === "personal_memory_receipt";
}

function WorkDisclosure({
  active,
  activeRunStartedAt,
  actionsDisabled,
  interrupted,
  latest,
  messages,
  onContinue,
  turnId,
}: {
  active: boolean;
  activeRunStartedAt: number | null;
  actionsDisabled: boolean;
  interrupted: boolean;
  latest: boolean;
  messages: MessageChunk[];
  onContinue?: () => void;
  turnId: string;
}) {
  const prominent = active || (interrupted && latest);
  const [expanded, setExpanded] = useState(prominent);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const detailsRef = useRef<HTMLDivElement>(null);
  const wasProminent = useRef(prominent);
  const workRevision =
    messages.length > 0
      ? `${messages.length}:${messageKey(messages[messages.length - 1] as MessageChunk)}`
      : "";
  const previousWorkRevision = useRef(workRevision);
  const detailsId = `${turnId}-work-details`;
  const duration = workDurationMs(messages);
  const activeDuration = useLiveWorkDuration(active, activeRunStartedAt);
  const activities = groupWorkActivities(messages);
  const label = active
    ? `Working for ${formatLiveWorkDuration(activeDuration)}`
    : interrupted
      ? duration
        ? `Interrupted after ${formatWorkDuration(duration)}`
        : "Interrupted"
      : duration
        ? `Worked for ${formatWorkDuration(duration)}`
        : "Worked";

  useLayoutEffect(() => {
    if (active) {
      if (!wasProminent.current || previousWorkRevision.current !== workRevision) setExpanded(true);
    } else if (prominent && !wasProminent.current) {
      setExpanded(true);
    } else if (!prominent && wasProminent.current) {
      if (detailsRef.current?.contains(document.activeElement)) toggleRef.current?.focus();
      setExpanded(false);
    }
    wasProminent.current = prominent;
    previousWorkRevision.current = workRevision;
  }, [active, prominent, workRevision]);

  return (
    <section
      className={cx(
        String.raw`work-disclosure [min-width:0] [border-bottom:1px_solid_var(--border-subtle)] [&.is-open_.work-disclosure\_\_toggle_svg]:[transform:rotate(90deg)] [&.is-active_.work-disclosure\_\_toggle]:[color:var(--muted)] [&.is-interrupted_.work-disclosure\_\_toggle]:[color:var(--muted)] [&.is-active_.work-disclosure\_\_label]:[color:transparent] [&.is-active_.work-disclosure\_\_label]:[background:linear-gradient(_100deg,_var(--muted-foreground)_25%,_var(--foreground)_50%,_var(--muted-foreground)_75%_)] [&.is-active_.work-disclosure\_\_label]:[background-size:220%_100%] [&.is-active_.work-disclosure\_\_label]:[-webkit-background-clip:text] [&.is-active_.work-disclosure\_\_label]:[background-clip:text] [&.is-active_.work-disclosure\_\_label]:[animation:work-label-shimmer_1.8s_linear_infinite] [&.is-active_.run-event--thinking_.run-event\_\_rail_svg]:[animation:spin_900ms_linear_infinite]`,
        active && "is-active",
        interrupted && "is-interrupted",
        expanded && "is-open",
      )}
    >
      <div className="work-disclosure__bar [min-width:0] [display:flex] [align-items:flex-start] [justify-content:space-between] [gap:12px]">
        <button
          type="button"
          className="work-disclosure__toggle [min-width:0] [min-height:38px] [padding:0_2px_12px] [display:inline-flex] [align-items:center] [gap:6px] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:5px] [cursor:pointer] [font-size:13.5px] [font-weight:520] [line-height:1.2] [text-align:left] [&_svg]:[width:16px] [&_svg]:[height:16px] [&_svg]:[transition:transform_var(--duration-fast)_var(--ease-out)]"
          aria-controls={detailsId}
          aria-expanded={expanded}
          onClick={() => setExpanded((value) => !value)}
          ref={toggleRef}
        >
          <span className="work-disclosure__label [display:inline-block] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
            {label}
          </span>
          <ChevronRight aria-hidden="true" />
        </button>
        {interrupted && latest && onContinue && (
          <button
            type="button"
            className="work-disclosure__continue [min-height:28px] [padding:4px_9px] [display:inline-flex] [flex:0_0_auto] [align-items:center] [gap:6px] [color:var(--muted)] [background:transparent] [border:1px_solid_var(--border)] [border-radius:7px] [cursor:pointer] [font-size:12px] [font-weight:560] [line-height:1.2] [&_svg]:[width:13px] [&_svg]:[height:13px]"
            disabled={actionsDisabled}
            onClick={onContinue}
          >
            <RefreshCw aria-hidden="true" />
            Continue
          </button>
        )}
      </div>
      {expanded && (
        <div
          className="work-disclosure__details [min-width:0] [padding:3px_0_17px]"
          id={detailsId}
          ref={detailsRef}
        >
          {interrupted && (
            <p className="work-disclosure__interruption [margin:0_0_11px] [color:var(--muted-foreground)] [font-size:12px] [line-height:1.5]">
              This run ended before a final response. Unfinished tools were not resumed.
            </p>
          )}
          {messages.length > 0 ? (
            <div
              className={String.raw`work-disclosure__events [min-width:0] [display:grid] [gap:10px] [&_.run-event]:[grid-template-columns:24px_minmax(0,_1fr)] [&_.run-event]:[gap:10px] [&_.run-event]:[animation:none] [&_.run-event\_\_rail]:[width:24px] [&_.run-event\_\_rail]:[height:24px] [&_.run-event\_\_rail]:[color:var(--muted-foreground)] [&_.run-event\_\_rail]:[background:transparent] [&_.run-event\_\_rail]:[border:0] [&_.run-event\_\_rail]:[border-radius:0] [&_.run-event\_\_rail]:[box-shadow:none] [&_.run-event\_\_rail]:[-webkit-backdrop-filter:none] [&_.run-event\_\_rail_svg]:[width:14px] [&_.run-event\_\_rail_svg]:[height:14px] [&_.run-event\_\_card]:[padding:10px_12px] [&_.run-event\_\_card]:[background:rgba(255,_255,_255,_0.026)] [&_.run-event\_\_card]:[border-color:var(--border-subtle)] [&_.run-event\_\_card]:[border-radius:10px] [&_.run-event\_\_card]:[box-shadow:none] [&_.run-event\_\_card]:[-webkit-backdrop-filter:none] [&_.run-event--compact]:[grid-template-columns:minmax(0,_1fr)] [&_.run-event--compact_.run-event\_\_rail]:[display:none] [&_.run-event--compact_.run-event\_\_card]:[padding:10px_12px_11px] [&_.run-event--compact_.run-event\_\_card]:[border-left:2px_solid_var(--border)] [&_.run-event--compact_.run-event\_\_card]:[border-radius:7px] [&_.run-event--compact.run-event--thinking]:[display:block] [&_.run-event--compact.run-event--assistant]:[display:block] [&_.run-event--compact.run-event--thinking_.run-event\_\_content]:[color:var(--foreground)] [&_.run-event--compact.run-event--thinking_.run-event\_\_content]:[font-size:15px] [&_.run-event--compact.run-event--thinking_.run-event\_\_content]:[line-height:1.72] [&_.run-event--compact.run-event--assistant_.run-event\_\_content]:[color:var(--foreground)] [&_.run-event--compact.run-event--assistant_.run-event\_\_content]:[font-size:15px] [&_.run-event--compact.run-event--assistant_.run-event\_\_content]:[line-height:1.72] [&_.run-event--compact_.run-event\_\_header]:[margin-bottom:5px] [&_.run-event--compact_.run-event\_\_header]:[text-transform:none] [&_.run-event--compact_.run-event\_\_header]:[letter-spacing:0] [&_.run-event--tool_.run-event\_\_card]:[background:var(--input)] [&_.run-event--tool_.run-event\_\_card]:[border-color:var(--border-subtle)] [&_.run-event\_\_header]:[margin-bottom:7px] [&_.run-event\_\_header]:[font-size:10px] [&_.run-event\_\_content]:[font-size:13px] [&_.run-event\_\_content]:[line-height:1.55] [&_.run-event--tool_.run-event\_\_content]:[color:var(--foreground)] max-680:[&_.run-event]:[grid-template-columns:22px_minmax(0,_1fr)] max-680:[&_.run-event]:[gap:8px] max-680:[&_.run-event\_\_rail]:[width:22px] max-680:[&_.run-event\_\_rail]:[height:22px]`}
            >
              {activities.map((activity) =>
                activity.kind === "message" ? (
                  <RunEvent
                    compact
                    key={`${messageKey(activity.message)}\u001f${activity.sourceIndex}`}
                    msg={activity.message}
                  />
                ) : (
                  <ToolActivityEvent
                    activity={activity.activity}
                    interrupted={interrupted}
                    key={toolActivityKey(activity.activity)}
                  />
                ),
              )}
            </div>
          ) : (
            <output className="work-disclosure__pending [min-height:34px] [display:flex] [align-items:center] [gap:8px] [color:var(--muted-foreground)] [font-size:12.5px] [&_svg]:[width:15px] [&_svg]:[height:15px] [&_svg]:[animation:spin_900ms_linear_infinite]">
              <Loader2 aria-hidden="true" />
              <span>Waiting for agent output</span>
            </output>
          )}
        </div>
      )}
    </section>
  );
}

function useLiveWorkDuration(active: boolean, startedAt: number | null): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!active || startedAt === null) return undefined;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [active, startedAt]);

  return startedAt === null ? 0 : Math.max(0, now - startedAt);
}

function groupWorkActivities(messages: readonly MessageChunk[]): WorkActivity[] {
  const activities: WorkActivity[] = [];
  const tools: ToolActivity[] = [];

  messages.forEach((message, sourceIndex) => {
    if (message.kind === "tool_call") {
      const activity = { call: message, sourceIndex };
      activities.push({ kind: "tool", activity });
      tools.push(activity);
      return;
    }
    if (message.kind === "tool_progress") {
      const invocationId = message.render?.invocationId;
      const exactMatch = invocationId
        ? tools.find(
            (activity) =>
              activity.call?.render?.invocationId === invocationId ||
              activity.progress?.render?.invocationId === invocationId,
          )
        : undefined;
      const fallbackMatch = tools.find(
        (activity) => activity.call?.toolName === message.toolName && !activity.result,
      );
      const match = exactMatch ?? fallbackMatch;
      if (match) {
        match.progress = mergeToolProgress(match.progress, message);
        return;
      }
      const activity = { progress: message, sourceIndex };
      activities.push({ kind: "tool", activity });
      tools.push(activity);
      return;
    }
    if (message.kind !== "tool_result") {
      activities.push({ kind: "message", message, sourceIndex });
      return;
    }

    const invocationId = message.render?.invocationId;
    const exactMatch = invocationId
      ? tools.find((activity) => activity.call?.render?.invocationId === invocationId)
      : undefined;
    const fallbackMatch = tools.find(
      (activity) =>
        activity.call?.toolName === message.toolName &&
        (!activity.result || !isTerminalToolResult(activity.result)),
    );
    const match = exactMatch ?? fallbackMatch;
    if (match) {
      match.result = message;
      return;
    }

    const activity = { result: message, sourceIndex };
    activities.push({ kind: "tool", activity });
    tools.push(activity);
  });

  return activities;
}

function isTerminalToolResult(message: MessageChunk): boolean {
  const status = normalizeMessageChunk(message, normalizeOptionsFromMessage(message)).status;
  return !["queued", "running"].includes(status);
}

function toolActivityKey(activity: ToolActivity): string {
  const message = activity.call ?? activity.progress ?? activity.result;
  const invocationId = message?.render?.invocationId;
  return invocationId
    ? `tool:${invocationId}`
    : `tool:${activity.sourceIndex}:${message ? messageKey(message) : "unknown"}`;
}

function ToolActivityEvent({
  activity,
  interrupted,
}: {
  activity: ToolActivity;
  interrupted: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const mergedEvent = mergedToolActivityEvent(activity);
  const interruptedTool =
    interrupted && ["queued", "running", "canceled"].includes(mergedEvent.status);
  const event = interruptedTool ? { ...mergedEvent, status: "canceled" as const } : mergedEvent;
  const summary = describeToolActivity(event, interruptedTool);
  const transcript = toolActivityTranscript(event);
  const failure = toolFailureSummary(event);
  const hasDetails = !!transcript || event.artifacts.length > 0 || !!failure;
  const detailsId = `${event.eventId}-activity-details`;
  const failed = !interruptedTool && (event.status === "failed" || event.status === "canceled");
  const running = event.status === "queued" || event.status === "running";
  const hasProgress = Boolean(activity.progress?.content);
  const autoOpened = useRef(false);
  const wasRunning = useRef(running);
  const StatusIcon = running ? Loader2 : failed ? XCircle : Check;

  useEffect(() => {
    if (running && hasProgress && !autoOpened.current) {
      autoOpened.current = true;
      setExpanded(true);
    }
    if (wasRunning.current && !running && autoOpened.current) setExpanded(false);
    wasRunning.current = running;
  }, [hasProgress, running]);

  return (
    <article
      className={cx(
        String.raw`tool-activity [min-width:0] [color:var(--muted)] [&.is-running_.tool-activity\_\_status]:[color:var(--muted-foreground)] [&.is-running_.tool-activity\_\_status_svg]:[animation:spin_900ms_linear_infinite] [&.is-open_.tool-activity\_\_chevron]:[transform:rotate(90deg)]`,
        running && "is-running",
        failed && "is-failed",
        interruptedTool && "is-interrupted",
        expanded && "is-open",
      )}
      data-render-event-id={event.eventId}
      data-render-kind="tool_activity"
      data-render-status={event.status}
    >
      <button
        type="button"
        className="tool-activity__summary [width:100%] [min-width:0] [min-height:28px] [padding:2px_0] [display:flex] [align-items:center] [gap:8px] [color:inherit] [background:transparent] [border:0] [border-radius:6px] [font-size:13px] [line-height:1.45] [text-align:left]"
        aria-controls={hasDetails ? detailsId : undefined}
        aria-expanded={hasDetails ? expanded : undefined}
        aria-label={`${summary}, ${humanToolStatus(event.status, interruptedTool)}`}
        disabled={!hasDetails}
        onClick={() => hasDetails && setExpanded((value) => !value)}
      >
        <span
          className="tool-activity__status [width:16px] [height:16px] [flex:0_0_auto] [display:grid] [place-items:center] [color:var(--success)] [&_svg]:[width:14px] [&_svg]:[height:14px]"
          aria-hidden="true"
        >
          <StatusIcon />
        </span>
        <span className="tool-activity__label [min-width:0] [flex:1_1_auto] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
          {summary}
        </span>
        {hasDetails && (
          <ChevronRight
            className="tool-activity__chevron [width:14px] [height:14px] [flex:0_0_auto] [color:var(--muted-foreground)] [transition:transform_var(--duration-fast)_var(--ease-out)]"
            aria-hidden="true"
          />
        )}
      </button>
      {expanded && hasDetails && (
        <div
          className="tool-activity__details [min-width:0] [margin:6px_0_4px_24px] [display:grid] [gap:8px]"
          id={detailsId}
        >
          {transcript && <ToolActivityTranscript live={running} transcript={transcript} />}
          {failure &&
            !(transcript?.stderr ?? transcript?.output ?? "").includes(failure) &&
            transcript?.command !== failure && (
              <p className="tool-activity__error [margin:0] [color:var(--danger)] [font-size:12px] [line-height:1.5]">
                {failure}
              </p>
            )}
          <TraceArtifacts artifacts={event.artifacts} />
        </div>
      )}
    </article>
  );
}

interface ToolTranscript {
  command?: string;
  output?: string;
  stderr?: string;
  truncated?: boolean;
}

function ToolActivityTranscript({
  live,
  transcript,
}: {
  live: boolean;
  transcript: ToolTranscript;
}) {
  const outputRef = useRef<HTMLPreElement>(null);
  const followOutput = useRef(true);

  useLayoutEffect(() => {
    if (!live || !followOutput.current || !outputRef.current) return;
    outputRef.current.scrollTop = outputRef.current.scrollHeight;
  });

  return (
    <div
      className={cx(
        "tool-activity__terminal [position:relative] [min-width:0] [background:var(--input)] [border:1px_solid_var(--border-subtle)] [border-radius:8px] [&_pre]:[min-width:0] [&_pre]:[max-height:260px] [&_pre]:[margin:0] [&_pre]:[padding:10px_12px] [&_pre]:[overflow:auto] [&_pre]:[color:var(--foreground)] [&_pre]:[font:12px_/_1.55_var(--font-mono)] [&_pre]:[white-space:pre-wrap] [&_pre]:[overflow-wrap:anywhere]",
        live && "is-live",
      )}
    >
      {transcript.truncated && (
        <em className="tool-activity__truncated [position:absolute] [top:6px] [right:10px] [color:var(--muted-foreground)] [font-size:10px] [font-style:normal]">
          truncated
        </em>
      )}
      <pre
        onScroll={(event) => {
          const output = event.currentTarget;
          followOutput.current = output.scrollHeight - output.scrollTop - output.clientHeight <= 8;
        }}
        ref={outputRef}
      >
        {transcript.command && (
          <span className="tool-activity__prompt [color:var(--success)] [user-select:none]">
            {"$ "}
          </span>
        )}
        {transcript.command}
        {transcript.command && (transcript.output || transcript.stderr) ? "\n" : null}
        {transcript.output}
        {transcript.output && transcript.stderr ? "\n" : null}
        {transcript.stderr && (
          <span className="tool-activity__stderr [color:var(--danger)]">{transcript.stderr}</span>
        )}
        {live && (
          <span
            className="tool-activity__cursor [display:inline-block] [width:7px] [height:14px] [margin-left:2px] [vertical-align:-2px] [background:var(--muted-foreground)] [animation:tool-cursor-blink_1.1s_steps(2,_start)_infinite]"
            aria-hidden="true"
          />
        )}
      </pre>
    </div>
  );
}

function toolActivityTranscript(event: NormalizedRenderEvent): ToolTranscript | null {
  const payload = tracePayloadRecord(event);
  if (!payload) return null;
  const stream = stringValue(payload, ["stream"]);
  const streamedOutput = stringValue(payload, ["output", "outputText", "formatted_output"]);
  const command = stringValue(payload, ["command", "cmd"]);
  const stdout =
    stringValue(payload, ["stdout", "stdoutText"]) ??
    (stream === "stderr" ? undefined : streamedOutput);
  const stderr =
    stringValue(payload, ["stderr", "stderrText", "error"]) ??
    (stream === "stderr" ? streamedOutput : undefined);
  const output =
    stdout ??
    stringValue(payload, ["content", "result", "preview", "diff", "patch", "text", "summary"]);
  const subject = stringValue(payload, [
    "path",
    "filePath",
    "file_path",
    "notebook_path",
    "targetPath",
    "target_path",
    "query",
    "pattern",
    "search",
    "search_query",
    "url",
    "target",
    "tool",
  ]);
  const commandLine =
    command ??
    (output || stderr
      ? [event.toolName, subject].filter(Boolean).join(" ") || undefined
      : undefined);
  if (!commandLine && !output && !stderr) return null;
  return {
    command: commandLine,
    output,
    stderr,
    truncated: booleanValue(payload, ["stdoutTruncated", "truncated", "outputTruncated"]) ?? false,
  };
}

function mergedToolActivityEvent(activity: ToolActivity): NormalizedRenderEvent {
  const call = activity.call
    ? normalizeMessageChunk(activity.call, normalizeOptionsFromMessage(activity.call))
    : undefined;
  const result = activity.result
    ? normalizeMessageChunk(activity.result, normalizeOptionsFromMessage(activity.result))
    : undefined;
  const progress = activity.progress
    ? normalizeMessageChunk(activity.progress, normalizeOptionsFromMessage(activity.progress))
    : undefined;
  const base = call ?? progress ?? result;
  if (!base) throw new Error("Tool activity requires a call, progress, or result.");
  const resultIsTerminal = result && !["queued", "running"].includes(result.status);

  return {
    ...base,
    eventId: call?.eventId ?? result?.eventId ?? base.eventId,
    invocationId: call?.invocationId ?? progress?.invocationId ?? result?.invocationId,
    kind: result ? "tool_result" : progress ? "tool_progress" : "tool_call",
    status: result?.status ?? progress?.status ?? call?.status ?? base.status,
    title: call?.title ?? progress?.title ?? result?.title ?? base.title,
    summary: result?.summary ?? call?.summary ?? progress?.summary ?? base.summary,
    toolName: call?.toolName ?? progress?.toolName ?? result?.toolName,
    input: call?.input,
    output: resultIsTerminal ? result.output : (progress?.output ?? result?.output),
    artifacts: uniqueArtifacts([
      ...(call?.artifacts ?? []),
      ...(progress?.artifacts ?? []),
      ...(result?.artifacts ?? []),
    ]),
    provenance: {
      ...(call?.provenance ?? {}),
      ...(progress?.provenance ?? {}),
      ...(result?.provenance ?? {}),
    },
    startedAt: call?.startedAt ?? progress?.startedAt ?? result?.startedAt,
    endedAt: result?.endedAt ?? call?.endedAt,
    durationMs: result?.durationMs ?? call?.durationMs,
    rawPayloadRef: undefined,
  };
}

function describeToolActivity(event: NormalizedRenderEvent, interrupted = false): string {
  const payload = tracePayloadRecord(event) ?? {};
  const tool = (event.toolName ?? "tool").toLowerCase();
  const path = compactToolValue(
    stringValue(payload, [
      "path",
      "filePath",
      "file_path",
      "notebook_path",
      "targetPath",
      "target_path",
    ]),
  );
  const command = compactToolValue(stringValue(payload, ["command", "cmd"]));
  const query = compactToolValue(
    stringValue(payload, ["query", "pattern", "search", "search_query"]),
  );
  const running = event.status === "queued" || event.status === "running";
  const failed = event.status === "failed" || event.status === "canceled";

  if (interrupted) {
    if (/apply[_-]?patch|notebookedit|(^|[_-])(edit|patch)([_-]|$)/.test(tool)) {
      return interruptedToolSummary("Edit interrupted", path);
    }
    if (/(^|[_-])(write|create)([_-]|$)/.test(tool)) {
      return interruptedToolSummary("Write interrupted", path);
    }
    if (/(^|[_-])(read|open)([_-]|$)/.test(tool)) {
      return interruptedToolSummary("Read interrupted", path);
    }
    if (/(^|[_-])(glob|grep|search|find)([_-]|$)|toolsearch/.test(tool)) {
      return interruptedToolSummary("Search interrupted", query ? `“${query}”` : path);
    }
    if (/(^|[_-])(test|check|vitest|jest|pytest)([_-]|$)/.test(tool)) {
      return "Tests interrupted";
    }
    if (/(^|[_-])(bash|shell|terminal|exec_command|powershell)([_-]|$)/.test(tool)) {
      return interruptedToolSummary("Command interrupted", command);
    }
    if (/(^|[_-])(browser|chrome|playwright|computer[_-]?use)([_-]|$)/.test(tool)) {
      return "Browser action interrupted";
    }
    if (/(^|[_-])(imagegen|image_generation|generate[_-]?image)([_-]|$)/.test(tool)) {
      return "Image generation interrupted";
    }
    return `${humanizeToolName(event.toolName ?? "tool")} interrupted`;
  }

  if (/apply[_-]?patch|notebookedit|(^|[_-])(edit|patch)([_-]|$)/.test(tool)) {
    return toolActionSummary("Editing", "Edited", "Couldn’t edit", path, running, failed);
  }
  if (/(^|[_-])(write|create)([_-]|$)/.test(tool)) {
    return toolActionSummary("Writing", "Wrote", "Couldn’t write", path, running, failed);
  }
  if (/(^|[_-])(read|open)([_-]|$)/.test(tool)) {
    return toolActionSummary("Reading", "Read", "Couldn’t read", path, running, failed);
  }
  if (/(^|[_-])(glob|grep|search|find)([_-]|$)|toolsearch/.test(tool)) {
    const subject = query ? `“${query}”` : path;
    return toolActionSummary(
      "Searching for",
      "Searched for",
      "Couldn’t search for",
      subject,
      running,
      failed,
      "Searching files",
      "Searched files",
      "File search failed",
    );
  }
  if (/(^|[_-])(test|check|vitest|jest|pytest)([_-]|$)/.test(tool)) {
    return running ? "Running tests" : failed ? "Tests failed" : "Ran tests";
  }
  if (/(^|[_-])(bash|shell|terminal|exec_command|powershell)([_-]|$)/.test(tool)) {
    return toolActionSummary(
      "Running",
      "Ran",
      "Command failed:",
      command,
      running,
      failed,
      "Running command",
      "Ran command",
      "Command failed",
    );
  }
  if (/(^|[_-])(browser|chrome|playwright|computer[_-]?use)([_-]|$)/.test(tool)) {
    return running ? "Using the browser" : failed ? "Browser action failed" : "Used the browser";
  }
  if (/(^|[_-])(imagegen|image_generation|generate[_-]?image)([_-]|$)/.test(tool)) {
    return running
      ? "Generating an image"
      : failed
        ? "Image generation failed"
        : "Generated an image";
  }

  const label = humanizeToolName(event.toolName ?? "tool");
  return running ? `Using ${label}` : failed ? `${label} failed` : `Used ${label}`;
}

function interruptedToolSummary(label: string, subject: string | undefined): string {
  return subject ? `${label}: ${subject}` : label;
}

function toolActionSummary(
  pendingVerb: string,
  completedVerb: string,
  failedVerb: string,
  subject: string | undefined,
  running: boolean,
  failed: boolean,
  pendingFallback = pendingVerb,
  completedFallback = completedVerb,
  failedFallback = failedVerb,
): string {
  if (!subject) return running ? pendingFallback : failed ? failedFallback : completedFallback;
  return `${running ? pendingVerb : failed ? failedVerb : completedVerb} ${subject}`;
}

function compactToolValue(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > 96 ? `${compact.slice(0, 93)}…` : compact;
}

function humanizeToolName(toolName: string): string {
  const publicName = toolName.includes("__") ? (toolName.split("__").at(-1) ?? toolName) : toolName;
  return publicName
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^\w/, (character) => character.toUpperCase());
}

function humanToolStatus(status: NormalizedRenderEvent["status"], interrupted = false): string {
  if (interrupted) return "interrupted";
  if (status === "queued" || status === "running") return "in progress";
  if (status === "failed") return "failed";
  if (status === "canceled") return "canceled";
  return "complete";
}

function toolFailureSummary(event: NormalizedRenderEvent): string | undefined {
  if (event.status !== "failed" && event.status !== "canceled") return undefined;
  const payload = tracePayloadRecord(event);
  return compactToolValue(
    payload
      ? stringValue(payload, ["error", "message", "stderr", "failure", "failures"])
      : undefined,
  );
}

function uniqueArtifacts(artifacts: RenderArtifactReference[]): RenderArtifactReference[] {
  const seen = new Set<string>();
  return artifacts.filter((artifact, index) => {
    const key =
      artifact.artifactId ??
      [artifact.kind, artifact.path ?? "", artifact.title ?? "", index].join(":");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function groupConversationTurns(messages: MessageChunk[]): ConversationTurn[] {
  const grouped: Array<{
    id: string;
    userMessage: MessageChunk | null;
    userMessageIndex: number | null;
    responseMessages: MessageChunk[];
    responseMessageIndices: number[];
  }> = [];
  let current: (typeof grouped)[number] | null = null;

  messages.forEach((message, index) => {
    if (message.role === "user") {
      current = {
        id: `conversation-turn-${index}`,
        userMessage: message,
        userMessageIndex: index,
        responseMessages: [],
        responseMessageIndices: [],
      };
      grouped.push(current);
      return;
    }

    if (!current) {
      current = {
        id: "conversation-turn-opening",
        userMessage: null,
        userMessageIndex: null,
        responseMessages: [],
        responseMessageIndices: [],
      };
      grouped.push(current);
    }
    current.responseMessages.push(message);
    current.responseMessageIndices.push(index);
  });

  return grouped.map(
    ({ id, userMessage, userMessageIndex, responseMessages, responseMessageIndices }) => {
      let finalMessageIndex = -1;
      responseMessages.forEach((message, index) => {
        if (isFinalResponse(message)) finalMessageIndex = index;
      });
      const finalWasSuperseded =
        finalMessageIndex >= 0 &&
        responseMessages.slice(finalMessageIndex + 1).some((message) => {
          if (
            message.kind === "thinking" ||
            message.kind === "tool_call" ||
            message.kind === "tool_progress"
          ) {
            return true;
          }
          return message.kind === "tool_result" && !isTerminalToolResult(message);
        });
      if (finalWasSuperseded) finalMessageIndex = -1;
      const finalMessage =
        finalMessageIndex >= 0 ? (responseMessages[finalMessageIndex] ?? null) : null;
      return {
        id,
        userMessage,
        userMessageIndex,
        workMessages:
          finalMessageIndex >= 0
            ? responseMessages.filter((_message, index) => index !== finalMessageIndex)
            : responseMessages,
        finalMessage,
        finalMessageIndex:
          finalMessageIndex >= 0 ? (responseMessageIndices[finalMessageIndex] ?? null) : null,
      };
    },
  );
}

function messageCreatedAt(message: MessageChunk): string | undefined {
  return (
    message.createdAt ??
    (message.role === "user"
      ? (message.render?.startedAt ?? message.render?.endedAt)
      : (message.render?.endedAt ?? message.render?.startedAt))
  );
}

function turnUserMessageCreatedAt(turn: ConversationTurn): string | undefined {
  if (!turn.userMessage) return undefined;
  const direct = messageCreatedAt(turn.userMessage);
  if (direct) return direct;

  for (const response of [...turn.workMessages, turn.finalMessage]) {
    if (!response) continue;
    const timestamp = response.createdAt ?? response.render?.startedAt ?? response.render?.endedAt;
    if (timestamp) return timestamp;
  }
  return undefined;
}

function isFinalResponse(message: MessageChunk): boolean {
  return message.kind === "message" && message.role !== "user" && message.role !== "tool";
}

function workDurationMs(messages: MessageChunk[]): number | undefined {
  const starts = messages
    .map((message) => parseRenderTime(message.render?.startedAt))
    .filter((value): value is number => value !== undefined);
  const ends = messages
    .map((message) => parseRenderTime(message.render?.endedAt))
    .filter((value): value is number => value !== undefined);
  if (starts.length > 0 && ends.length > 0) {
    const elapsed = Math.max(...ends) - Math.min(...starts);
    if (elapsed > 0) return elapsed;
  }

  const explicitDurations = messages
    .map((message) => message.render?.durationMs)
    .filter((duration): duration is number => duration !== undefined && duration > 0);
  return explicitDurations.length === 1 ? explicitDurations[0] : undefined;
}

function parseRenderTime(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function formatWorkDuration(durationMs: number): string {
  return formatWorkSeconds(Math.max(1, Math.round(durationMs / 1000)));
}

function formatLiveWorkDuration(durationMs: number): string {
  return formatWorkSeconds(Math.max(0, Math.floor(durationMs / 1000)));
}

function formatWorkSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder === 0 ? `${minutes}m` : `${minutes}m ${remainder}s`;
}

function normalizeThoughtMarkdown(content: string): string {
  const match = /^(\s*)\*\*([^\n]+)\*\*(\s*)$/.exec(content);
  return match ? `${match[1]}${match[2]}${match[3]}` : content;
}

function TraceCard({ event }: { event: NormalizedRenderEvent }) {
  const [expanded, setExpanded] = useState(false);
  const detailsId = `${event.eventId}-details`;
  const specialized = specializedTracePresentation(event);

  return (
    <section
      className="trace-card [display:grid] [gap:10px] [margin-top:12px] [padding-top:12px] [border-top:1px_solid_var(--border-subtle)]"
      aria-label={`${event.title} trace details`}
    >
      <div className="trace-card__summary [display:grid] [gap:5px] [min-width:0] [&_p]:[margin:0] [&_p]:[color:var(--muted)] [&_p]:[font-size:12.5px] [&_p]:[line-height:1.45] [&_p]:[overflow-wrap:anywhere]">
        <div className="trace-card__title [display:flex] [align-items:center] [justify-content:space-between] [gap:10px] [min-width:0] [color:var(--muted-foreground)] [font-size:11px] [font-weight:750] [letter-spacing:0] [text-transform:uppercase]">
          <span>{event.title}</span>
          <span className={traceStatusVariants({ status: event.status })}>{event.status}</span>
        </div>
        <p>{event.summary || event.status}</p>
      </div>
      <button
        type="button"
        className="trace-card__toggle [display:inline-flex] [width:fit-content] [align-items:center] [gap:6px] [padding:5px_8px] [color:var(--muted)] [background:rgba(255,_255,_255,_0.045)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-size:12px] [font-weight:680] [&_svg]:[width:13px] [&_svg]:[height:13px]"
        aria-controls={detailsId}
        aria-expanded={expanded}
        onClick={() => setExpanded((value) => !value)}
      >
        {expanded ? <Minus aria-hidden="true" /> : <Plus aria-hidden="true" />}
        {expanded ? "Hide details" : "Show details"}
      </button>
      {expanded && (
        <div className="trace-card__details [display:grid] [gap:10px] [min-width:0]" id={detailsId}>
          {specialized && <SpecializedTracePresentation presentation={specialized} />}
          {event.input !== undefined && <TraceDetailBlock title="Input" payload={event.input} />}
          {event.output !== undefined && <TraceDetailBlock title="Output" payload={event.output} />}
          <TraceMetadata title="Provenance" entries={metadataEntries(event.provenance)} />
          <TraceArtifacts artifacts={event.artifacts} />
          {event.rawPayloadRef && (
            <TraceMetadata title="Raw payload ref" entries={[["ref", event.rawPayloadRef]]} />
          )}
        </div>
      )}
    </section>
  );
}

interface TracePresentationBlock {
  content: string;
  title: string;
  truncated?: boolean;
}

interface SpecializedTracePresentationModel {
  blocks: TracePresentationBlock[];
  fields: Array<[string, string]>;
  live?: boolean;
  title: string;
}

interface SpecializedTracePresentationDraft {
  blocks: Array<TracePresentationBlock | null>;
  fields: Array<[string, string] | null>;
  live?: boolean;
  title: string;
}

function SpecializedTracePresentation({
  presentation,
}: {
  presentation: SpecializedTracePresentationModel;
}) {
  return (
    <div className="trace-card__special [display:grid] [gap:8px] [min-width:0] [padding:9px] [background:rgba(255,_255,_255,_0.035)] [border:1px_solid_var(--border-subtle)] [border-radius:9px] [&_h4]:[margin:0] [&_h4]:[color:var(--foreground)] [&_h4]:[font-size:12px] [&_h4]:[font-weight:760] [&_h4]:[letter-spacing:0]">
      <h4>{presentation.title}</h4>
      {presentation.fields.length > 0 && (
        <div className="trace-card__field-grid [display:grid] [grid-template-columns:repeat(auto-fit,_minmax(132px,_1fr))] [gap:6px] [min-width:0] max-680:[grid-template-columns:1fr]">
          {presentation.fields.map(([label, value]) => (
            <div
              className="trace-card__field [display:grid] [gap:2px] [min-width:0] [padding:6px_7px] [background:rgba(0,_0,_0,_0.18)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [&_span]:[color:var(--muted-foreground)] [&_span]:[font-size:10px] [&_span]:[font-weight:720] [&_span]:[letter-spacing:0] [&_span]:[text-transform:uppercase] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-family:var(--font-mono)] [&_strong]:[font-size:11px] [&_strong]:[font-weight:620] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap]"
              key={`${label}:${value}`}
            >
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      )}
      {presentation.blocks.map((block) => (
        <TracePresentationBlockView block={block} key={block.title} live={presentation.live} />
      ))}
    </div>
  );
}

function TracePresentationBlockView({
  block,
  live = false,
}: {
  block: TracePresentationBlock;
  live?: boolean;
}) {
  const outputRef = useRef<HTMLPreElement>(null);
  const followOutput = useRef(true);

  useLayoutEffect(() => {
    if (!live || !followOutput.current || !outputRef.current) return;
    outputRef.current.scrollTop = outputRef.current.scrollHeight;
  });

  return (
    <div
      className={cx(
        "trace-card__excerpt [display:grid] [gap:5px] [min-width:0] [&_pre]:[max-width:100%] [&_pre]:[max-height:260px] [&_pre]:[margin:0] [&_pre]:[overflow:auto] [&_pre]:[padding:8px_9px] [&_pre]:[color:#d4d4d8] [&_pre]:[background:rgba(0,_0,_0,_0.28)] [&_pre]:[border:1px_solid_var(--border-subtle)] [&_pre]:[border-radius:8px] [&_pre]:[font-family:var(--font-mono)] [&_pre]:[font-size:12px] [&_pre]:[line-height:1.45] [&_pre]:[white-space:pre] [&.is-live_pre]:[border-color:color-mix(in_srgb,_var(--accent)_32%,_var(--border-subtle))]",
        live && "is-live",
      )}
    >
      <div className="trace-card__excerpt-title [display:flex] [align-items:center] [gap:6px] [color:var(--muted-foreground)] [font-size:10px] [font-weight:720] [letter-spacing:0] [text-transform:uppercase] [&_em]:[color:var(--accent)] [&_em]:[font-style:normal] [&_em]:[text-transform:none]">
        <span>{block.title}</span>
        {block.truncated && <em>truncated</em>}
        {live && <em>live</em>}
      </div>
      <pre
        onScroll={(event) => {
          const output = event.currentTarget;
          followOutput.current = output.scrollHeight - output.scrollTop - output.clientHeight <= 8;
        }}
        ref={outputRef}
      >
        {block.content}
      </pre>
    </div>
  );
}

function TraceDetailBlock({ payload, title }: { payload: unknown; title: string }) {
  return (
    <div className="trace-card__detail [min-width:0] [&_h4]:[margin:0_0_6px] [&_h4]:[color:var(--muted-foreground)] [&_h4]:[font-size:11px] [&_h4]:[font-weight:740] [&_h4]:[letter-spacing:0] [&_h4]:[text-transform:uppercase] [&_pre]:[max-width:100%] [&_pre]:[margin:0] [&_pre]:[overflow-x:auto] [&_pre]:[padding:9px_10px] [&_pre]:[color:#d4d4d8] [&_pre]:[background:rgba(0,_0,_0,_0.26)] [&_pre]:[border:1px_solid_var(--border-subtle)] [&_pre]:[border-radius:8px] [&_pre]:[font-family:var(--font-mono)] [&_pre]:[font-size:12px] [&_pre]:[line-height:1.45] [&_pre]:[white-space:pre]">
      <h4>{title}</h4>
      <pre>{stringifyRenderPayload(payload)}</pre>
    </div>
  );
}

function TraceMetadata({ entries, title }: { entries: Array<[string, string]>; title: string }) {
  if (entries.length === 0) return null;

  return (
    <div className="trace-card__metadata [min-width:0] [&_h4]:[margin:0_0_6px] [&_h4]:[color:var(--muted-foreground)] [&_h4]:[font-size:11px] [&_h4]:[font-weight:740] [&_h4]:[letter-spacing:0] [&_h4]:[text-transform:uppercase]">
      <h4>{title}</h4>
      <div className="trace-card__chips [display:flex] [flex-wrap:wrap] [gap:6px] [min-width:0]">
        {entries.map(([key, value]) => (
          <span
            className="trace-card__chip [display:inline-flex] [max-width:100%] [align-items:center] [gap:5px] [padding:4px_6px] [color:var(--muted)] [background:rgba(255,_255,_255,_0.045)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-family:var(--font-mono)] [font-size:11px] [line-height:1.25] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-style:normal] [&_strong]:[font-weight:650] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_em]:[min-width:0] [&_em]:[overflow:hidden] [&_em]:[font-style:normal] [&_em]:[font-weight:650] [&_em]:[text-overflow:ellipsis] [&_em]:[white-space:nowrap] [&_em]:[color:var(--accent)]"
            key={`${key}:${value}`}
          >
            <span>{key}</span>
            <strong>{value}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function TraceArtifacts({ artifacts }: { artifacts: RenderArtifactReference[] }) {
  if (artifacts.length === 0) return null;

  return (
    <div className="trace-card__metadata [min-width:0] [&_h4]:[margin:0_0_6px] [&_h4]:[color:var(--muted-foreground)] [&_h4]:[font-size:11px] [&_h4]:[font-weight:740] [&_h4]:[letter-spacing:0] [&_h4]:[text-transform:uppercase]">
      <h4>Artifacts</h4>
      <div className="trace-card__chips [display:flex] [flex-wrap:wrap] [gap:6px] [min-width:0]">
        {artifacts.map((artifact, index) => (
          <span
            className="trace-card__chip trace-card__chip--artifact [display:inline-flex] [max-width:100%] [align-items:center] [gap:5px] [padding:4px_6px] [color:var(--muted)] [background:rgba(255,_255,_255,_0.045)] [border:1px_solid_var(--border-subtle)] [border-radius:7px] [font-family:var(--font-mono)] [font-size:11px] [line-height:1.25] [&_strong]:[min-width:0] [&_strong]:[overflow:hidden] [&_strong]:[color:var(--foreground)] [&_strong]:[font-style:normal] [&_strong]:[font-weight:650] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_em]:[min-width:0] [&_em]:[overflow:hidden] [&_em]:[font-style:normal] [&_em]:[font-weight:650] [&_em]:[text-overflow:ellipsis] [&_em]:[white-space:nowrap] [&_em]:[color:var(--accent)]"
            key={artifact.artifactId ?? `${artifact.kind}:${artifact.path ?? index}`}
          >
            <span>{artifact.kind}</span>
            <strong>{artifact.title ?? artifact.path ?? artifact.artifactId ?? "artifact"}</strong>
            {artifact.truncated && <em>truncated</em>}
          </span>
        ))}
      </div>
    </div>
  );
}

function specializedTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  return (
    terminalTracePresentation(event) ??
    diffTracePresentation(event) ??
    testTracePresentation(event) ??
    fileTracePresentation(event) ??
    mcpTracePresentation(event) ??
    automationTracePresentation(event) ??
    mediaTracePresentation(event)
  );
}

function terminalTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !payload ||
    (!tool.match(/\b(terminal|shell|bash|zsh|command)\b|exec[_-]?command|write[_-]?stdin/) &&
      !hasAnyKey(payload, ["command", "cmd", "stdout", "stderr", "exitCode", "exit_code"]))
  ) {
    return null;
  }

  const output = stringValue(payload, ["output", "outputText", "formatted_output"]);
  const stream = stringValue(payload, ["stream"]);
  const stdout =
    stringValue(payload, ["stdout", "stdoutText"]) ?? (stream === "stderr" ? undefined : output);
  const stderr =
    stringValue(payload, ["stderr", "stderrText", "error"]) ??
    (stream === "stderr" ? output : undefined);

  return compactPresentation({
    blocks: [
      traceBlock(stream === "combined" ? "Output" : "Stdout", stdout, {
        truncated: booleanValue(payload, ["stdoutTruncated", "truncated"]),
      }),
      traceBlock("Stderr", stderr, {
        truncated: booleanValue(payload, ["stderrTruncated"]),
      }),
    ],
    fields: [
      traceField("command", stringValue(payload, ["command", "cmd"])),
      traceField("cwd", stringValue(payload, ["cwd", "workdir", "workingDirectory"])),
      traceField("status", stringValue(payload, ["status"]) ?? event.status),
      traceField("exit", stringValue(payload, ["exitCode", "exit_code", "code"])),
      traceField("duration", durationText(payload, event)),
    ],
    live: event.status === "queued" || event.status === "running",
    title: "Terminal",
  });
}

function diffTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  const hasDiffArtifact = event.artifacts.some((artifact) => artifact.kind === "diff");
  const diff = payload ? stringValue(payload, ["diff", "patch", "unifiedDiff"]) : undefined;
  if (!diff && !hasDiffArtifact && !tool.match(/\b(diff|patch)\b/)) return null;

  return compactPresentation({
    blocks: [traceBlock("Diff", diff)],
    fields: [
      traceField(
        "path",
        payload ? stringValue(payload, ["path", "filePath", "targetPath"]) : undefined,
      ),
      traceField("artifacts", hasDiffArtifact ? String(event.artifacts.length) : undefined),
    ],
    title: "Diff",
  });
}

function testTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !payload ||
    (!tool.match(/\b(test|check|vitest|jest|pytest|pnpm)\b/) &&
      !hasAnyKey(payload, ["passed", "failed", "testCount", "failures"]))
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("Failures", stringValue(payload, ["failures", "failure", "failureExcerpt"])),
      traceBlock("Output", stringValue(payload, ["output", "stdout"])),
    ],
    fields: [
      traceField("command", stringValue(payload, ["command", "cmd"])),
      traceField("status", stringValue(payload, ["status"]) ?? event.status),
      traceField("passed", stringValue(payload, ["passed", "passedCount"])),
      traceField("failed", stringValue(payload, ["failed", "failedCount"])),
      traceField("tests", stringValue(payload, ["testCount", "tests"])),
    ],
    title: "Test/check",
  });
}

function fileTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  if (!payload) return null;
  const path = stringValue(payload, ["path", "filePath", "targetPath"]);
  const operation = stringValue(payload, ["operation", "action", "mode"]);
  if (
    !path ||
    (!operation && !hasAnyKey(payload, ["lineStart", "lineEnd", "byteCount", "content"]))
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("Preview", stringValue(payload, ["preview", "content", "excerpt"]), {
        truncated: booleanValue(payload, ["truncated"]),
      }),
    ],
    fields: [
      traceField("operation", operation),
      traceField("path", path),
      traceField("range", rangeText(payload)),
      traceField("bytes", stringValue(payload, ["byteCount", "bytes"])),
    ],
    title: "File",
  });
}

function mcpTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const toolText = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  const hasMcpShape =
    !!event.provenance.mcpServer ||
    toolText.includes("mcp") ||
    !!(payload && hasAnyKey(payload, ["mcpServer", "server", "tool", "toolName"]));
  if (!hasMcpShape) return null;

  const server =
    event.provenance.mcpServer ?? (payload && stringValue(payload, ["mcpServer", "server"]));
  const tool =
    (payload && stringValue(payload, ["tool", "toolName", "name"])) ??
    (toolText.includes("mcp") ? event.toolName : undefined);
  if (!server && !tool) return null;

  return compactPresentation({
    blocks: [
      traceBlock(
        "Result",
        payload ? stringValue(payload, ["result", "summary", "output"]) : undefined,
      ),
    ],
    fields: [
      traceField("server", server),
      traceField("tool", tool),
      traceField("plugin", event.provenance.pluginId),
    ],
    title: "MCP",
  });
}

function automationTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const payload = tracePayloadRecord(event);
  const tool = `${event.toolName ?? ""} ${event.title}`.toLowerCase();
  if (
    !tool.match(/\b(browser|chrome|playwright|automation|app)\b/) &&
    !hasAutomationArtifact(event)
  ) {
    return null;
  }

  return compactPresentation({
    blocks: [
      traceBlock("DOM", payload ? stringValue(payload, ["dom", "excerpt", "summary"]) : undefined),
    ],
    fields: [
      traceField("target", payload ? stringValue(payload, ["target", "url", "app"]) : undefined),
      traceField("action", payload ? stringValue(payload, ["action", "operation"]) : undefined),
      traceField(
        "artifacts",
        hasAutomationArtifact(event) ? String(event.artifacts.length) : undefined,
      ),
    ],
    title: "Automation",
  });
}

function mediaTracePresentation(
  event: NormalizedRenderEvent,
): SpecializedTracePresentationModel | null {
  const mediaArtifacts = event.artifacts.filter((artifact) =>
    ["image", "screenshot", "table", "html"].includes(artifact.kind),
  );
  if (mediaArtifacts.length === 0) return null;

  return compactPresentation({
    blocks: [],
    fields: [
      traceField("artifacts", String(mediaArtifacts.length)),
      traceField(
        "types",
        Array.from(new Set(mediaArtifacts.map((artifact) => artifact.kind))).join(", "),
      ),
    ],
    title: "Generated media",
  });
}

function compactPresentation(
  presentation: SpecializedTracePresentationDraft,
): SpecializedTracePresentationModel | null {
  const fields = presentation.fields.filter((field): field is [string, string] => !!field);
  const blocks = presentation.blocks.filter(
    (block): block is TracePresentationBlock => block !== null,
  );
  if (fields.length === 0 && blocks.length === 0) return null;
  return { ...presentation, blocks, fields };
}

function tracePayloadRecord(event: NormalizedRenderEvent): Record<string, unknown> | undefined {
  const input = isRecord(event.input) ? event.input : undefined;
  const output = isRecord(event.output) ? event.output : undefined;
  if (input && output) return { ...input, ...output };
  return output ?? input;
}

function traceField(label: string, value: string | undefined): [string, string] | null {
  return value ? [label, value] : null;
}

function traceBlock(
  title: string,
  content: string | undefined,
  options: { truncated?: boolean } = {},
): TracePresentationBlock | null {
  if (!content) return null;
  return { content, title, truncated: options.truncated };
}

function stringValue(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value) return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value) && value.length > 0) return value.map(String).join(", ");
  }
  return undefined;
}

function booleanValue(record: Record<string, unknown>, keys: string[]): boolean | undefined {
  for (const key of keys) {
    if (typeof record[key] === "boolean") return record[key];
  }
  return undefined;
}

function hasAnyKey(record: Record<string, unknown>, keys: string[]): boolean {
  return keys.some((key) => record[key] !== undefined);
}

function durationText(
  record: Record<string, unknown>,
  event: NormalizedRenderEvent,
): string | undefined {
  const value = stringValue(record, ["durationMs", "duration"]);
  if (value) return value.endsWith("ms") ? value : `${value}ms`;
  return event.durationMs === undefined ? undefined : `${event.durationMs}ms`;
}

function rangeText(record: Record<string, unknown>): string | undefined {
  const start = stringValue(record, ["lineStart", "startLine"]);
  const end = stringValue(record, ["lineEnd", "endLine"]);
  if (!start && !end) return undefined;
  return `${start ?? "?"}-${end ?? "?"}`;
}

function hasAutomationArtifact(event: NormalizedRenderEvent): boolean {
  return event.artifacts.some(
    (artifact) => artifact.kind === "screenshot" || artifact.kind === "html",
  );
}

function providerErrorNotice(msg: MessageChunk): ProviderErrorNotice | null {
  if (msg.role !== "system" || !isRecord(msg.structuredContent)) return null;
  const value = msg.structuredContent;
  if (
    value.type !== "provider_error" ||
    !["overloaded", "rate_limited", "temporarily_unavailable"].includes(String(value.code)) ||
    typeof value.title !== "string" ||
    value.title.length === 0 ||
    value.title.length > 80 ||
    typeof value.message !== "string" ||
    value.message.length === 0 ||
    value.message.length > 280 ||
    typeof value.retryable !== "boolean"
  ) {
    return null;
  }
  return value as unknown as ProviderErrorNotice;
}

function normalizeOptionsFromMessage(msg: MessageChunk): NormalizeMessageChunkOptions {
  return {
    artifacts: msg.render?.artifacts,
    durationMs: msg.render?.durationMs,
    endedAt: msg.render?.endedAt,
    invocationId: msg.render?.invocationId,
    parentMessageId: msg.render?.parentMessageId,
    provenance: msg.render?.provenance,
    rawPayloadRef: msg.render?.rawPayloadRef,
    source: msg.render?.source,
    startedAt: msg.render?.startedAt,
    status: msg.render?.status,
  };
}

function isTraceCardEvent(event: NormalizedRenderEvent): boolean {
  return event.kind === "tool_call" || event.kind === "tool_result" || event.kind === "trace";
}

function metadataEntries(provenance: RenderProvenance): Array<[string, string]> {
  return Object.entries(provenance)
    .filter((entry): entry is [string, string] => typeof entry[1] === "string" && !!entry[1])
    .filter(([key]) => key !== "agent")
    .sort(([left], [right]) => left.localeCompare(right));
}

function messagePresentation(msg: MessageChunk): {
  icon: LucideIcon;
  label: string;
  tone: RunEventTone;
  meta: string;
} {
  if (msg.role === "user") {
    return { icon: User, label: "you", tone: "user", meta: "message" };
  }
  if (msg.render?.source === "personal_memory_receipt") {
    return { icon: Brain, label: "personal memory", tone: "system", meta: "context" };
  }
  if (msg.role === "system") {
    return { icon: XCircle, label: "system", tone: "system", meta: msg.kind };
  }
  if (msg.kind === "thinking") {
    return { icon: Loader2, label: msg.agent ?? "agent", tone: "thinking", meta: "thinking" };
  }
  if (msg.kind === "tool_call") {
    return { icon: TerminalIcon, label: msg.agent ?? "tool", tone: "tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { icon: Code2, label: msg.agent ?? "tool", tone: "tool", meta: "result" };
  }
  return { icon: Bot, label: msg.agent ?? "assistant", tone: "assistant", meta: "message" };
}

function compactWorkPresentation(msg: MessageChunk): ReturnType<typeof messagePresentation> {
  const presentation = messagePresentation(msg);
  if (msg.kind === "thinking") return { ...presentation, label: "Reasoning", meta: "thought" };
  if (msg.kind === "tool_call") {
    return { ...presentation, label: msg.toolName ?? "Tool", meta: "call" };
  }
  if (msg.kind === "tool_result") {
    return { ...presentation, label: msg.toolName ?? "Tool", meta: "result" };
  }
  return { ...presentation, label: "Progress" };
}

function renderEventContent(
  msg: MessageChunk,
  event: ReturnType<typeof normalizeMessageChunk>,
): string {
  if (isTraceCardEvent(event)) {
    return event.summary || event.title;
  }
  return msg.content;
}

function stringifyRenderPayload(payload: unknown): string {
  return typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
}

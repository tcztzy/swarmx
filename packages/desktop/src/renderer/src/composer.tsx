import { ArrowUp, FolderOpen, Paperclip, Plus, Square, X } from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import type { DesktopMediaAttachment, DesktopMediaImport } from "../../shared/desktop-api.js";
import { AttachmentIcon, formatMediaBytes } from "./message-attachments.js";
import { cx } from "./ui-primitives.js";

const COMPOSER_MIN_HEIGHT = 48;
const COMPOSER_MAX_HEIGHT = 240;
const COMPLETION_DEBOUNCE_MS = 180;

export interface MentionServer {
  id: string;
  name?: string;
  description?: string;
  mentionPrefixes?: string[];
}

export interface ComposerProps {
  value: string;
  placeholder: string;
  disabled: boolean;
  running: boolean;
  sendDisabled: boolean;
  sendTitle?: string;
  workspaceRoot?: string;
  textareaRef?: React.RefObject<HTMLTextAreaElement | null>;
  mentionServers: MentionServer[];
  completeMention: (params: {
    serverId: string;
    workspaceRoot: string;
    text: string;
    position: { line: number; character: number };
    languageId: string;
    triggerCharacter: string;
    timeoutMs: number;
  }) => Promise<{ result: unknown }>;
  selectFilesAndFolders: () => Promise<string[]>;
  selectMediaAttachments?: (
    existingAttachments: readonly DesktopMediaAttachment[],
  ) => Promise<DesktopMediaAttachment[]>;
  importMediaAttachments?: (
    files: DesktopMediaImport[],
    existingAttachments: readonly DesktopMediaAttachment[],
  ) => Promise<DesktopMediaAttachment[]>;
  attachments?: DesktopMediaAttachment[];
  onAttachmentsChange?: (attachments: DesktopMediaAttachment[]) => void;
  onPreviewAttachment?: (attachment: DesktopMediaAttachment) => void;
  onFilesSelected?: (paths: string[]) => void;
  onContextError?: (error: unknown) => void;
  error?: string | null;
  onChange: (value: string) => void;
  onFocus?: () => void;
  onSubmit: () => void | Promise<void>;
  onStop: () => void | Promise<void>;
  children: React.ReactNode;
}

interface MentionCompletionItem {
  id: string;
  label: string;
  detail?: string;
  documentation?: string;
  insertText: string;
}

type MentionMenuState = "idle" | "loading" | "ready" | "empty";

interface MentionContext {
  start: number;
  token: string;
  trigger: "@" | "$";
}

export function Composer({
  value,
  placeholder,
  disabled,
  running,
  sendDisabled,
  sendTitle,
  workspaceRoot,
  textareaRef: providedTextareaRef,
  mentionServers,
  completeMention,
  selectFilesAndFolders,
  selectMediaAttachments,
  importMediaAttachments,
  attachments = [],
  onAttachmentsChange,
  onPreviewAttachment,
  onFilesSelected,
  onContextError,
  error,
  onChange,
  onFocus,
  onSubmit,
  onStop,
  children,
}: ComposerProps): React.JSX.Element {
  const internalTextareaRef = useRef<HTMLTextAreaElement>(null);
  const requestIdRef = useRef(0);
  const [cursorOffset, setCursorOffset] = useState(value.length);
  const [isComposing, setIsComposing] = useState(false);
  const [mentionItems, setMentionItems] = useState<MentionCompletionItem[]>([]);
  const [mentionMenuState, setMentionMenuState] = useState<MentionMenuState>("idle");
  const [activeMentionIndex, setActiveMentionIndex] = useState(0);
  const [contextMenuOpen, setContextMenuOpen] = useState(false);
  const [textareaElement, setTextareaElement] = useState<HTMLTextAreaElement | null>(null);
  const mentionListId = useId();
  const mentionContext = getMentionContext(value, cursorOffset);
  const mentionStart = mentionContext?.start;
  const mentionToken = mentionContext?.token;
  const mentionTrigger = mentionContext?.trigger;
  const mentionMenuOpen = mentionMenuState !== "idle";
  const assignTextareaRef = useCallback(
    (element: HTMLTextAreaElement | null) => {
      internalTextareaRef.current = element;
      if (providedTextareaRef) providedTextareaRef.current = element;
      setTextareaElement(element);
    },
    [providedTextareaRef],
  );

  useLayoutEffect(() => {
    if (!textareaElement) return;
    textareaElement.style.height = "0px";
    textareaElement.style.height = `${Math.min(
      Math.max(textareaElement.scrollHeight, COMPOSER_MIN_HEIGHT),
      COMPOSER_MAX_HEIGHT,
    )}px`;
  });

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    if (mentionStart === undefined || !mentionToken || !mentionTrigger || disabled || isComposing) {
      setMentionItems([]);
      setMentionMenuState("idle");
      return;
    }

    if (mentionToken === "@") {
      const items = prefixMentionItems(mentionServers, "@");
      setMentionItems(items);
      setMentionMenuState(items.length > 0 ? "ready" : "empty");
      setActiveMentionIndex(0);
      return;
    }

    if (!workspaceRoot) {
      setMentionItems([]);
      setMentionMenuState("empty");
      return;
    }

    const matchingServers = mentionServers.flatMap((server) => {
      const matchingPrefix = server.mentionPrefixes
        ?.filter((prefix) => mentionToken.startsWith(prefix))
        .sort((left, right) => right.length - left.length)[0];
      return matchingPrefix ? [{ server, prefixLength: matchingPrefix.length }] : [];
    });
    const longestPrefix = Math.max(...matchingServers.map((match) => match.prefixLength));
    const servers = matchingServers
      .filter((match) => match.prefixLength === longestPrefix)
      .map((match) => match.server);
    if (servers.length === 0) {
      setMentionItems([]);
      setMentionMenuState("empty");
      return;
    }

    setMentionItems([]);
    setMentionMenuState("loading");
    const timeout = window.setTimeout(() => {
      void Promise.allSettled(
        servers.map((server) =>
          completeMention({
            serverId: server.id,
            workspaceRoot,
            text: value,
            position: positionAtOffset(value, cursorOffset),
            languageId: "plaintext",
            triggerCharacter: mentionTrigger,
            timeoutMs: 1_500,
          }),
        ),
      ).then((responses) => {
        if (requestIdRef.current !== requestId) return;
        const items = uniqueMentionItems(
          responses.flatMap((response) =>
            response.status === "fulfilled" ? parseMentionItems(response.value.result) : [],
          ),
        );
        setMentionItems(items);
        setMentionMenuState(items.length > 0 ? "ready" : "empty");
        setActiveMentionIndex(0);
      });
    }, COMPLETION_DEBOUNCE_MS);

    return () => window.clearTimeout(timeout);
  }, [
    completeMention,
    cursorOffset,
    disabled,
    isComposing,
    mentionStart,
    mentionToken,
    mentionTrigger,
    mentionServers,
    value,
    workspaceRoot,
  ]);

  const syncCursor = useCallback((textarea: HTMLTextAreaElement) => {
    setCursorOffset(textarea.selectionStart);
  }, []);

  const focusAt = useCallback(
    (offset: number) => {
      window.requestAnimationFrame(() => {
        if (!textareaElement) return;
        textareaElement.focus();
        textareaElement.setSelectionRange(offset, offset);
        setCursorOffset(offset);
      });
    },
    [textareaElement],
  );

  const insertMention = useCallback(
    (item: MentionCompletionItem) => {
      const context = getMentionContext(value, cursorOffset);
      if (!context) return;
      const nextValue = `${value.slice(0, context.start)}${item.insertText}${value.slice(cursorOffset)}`;
      const nextCursor = context.start + item.insertText.length;
      onChange(nextValue);
      requestIdRef.current += 1;
      setMentionItems([]);
      setMentionMenuState("idle");
      focusAt(nextCursor);
    },
    [cursorOffset, focusAt, onChange, value],
  );

  const addFilesAndFolders = useCallback(async () => {
    try {
      const paths = await selectFilesAndFolders();
      if (paths.length === 0) return;
      onFilesSelected?.(paths);

      const beforeCursor = value.slice(0, cursorOffset);
      const prefix = beforeCursor.length > 0 && !/\s$/.test(beforeCursor) ? " " : "";
      const references = paths.map(fileReference).join(" ");
      const nextValue = `${beforeCursor}${prefix}${references}${value.slice(cursorOffset)}`;
      const nextCursor = beforeCursor.length + prefix.length + references.length;
      onChange(nextValue);
      focusAt(nextCursor);
    } catch (error) {
      onContextError?.(error);
    } finally {
      setContextMenuOpen(false);
    }
  }, [
    cursorOffset,
    focusAt,
    onChange,
    onContextError,
    onFilesSelected,
    selectFilesAndFolders,
    value,
  ]);

  const appendAttachments = useCallback(
    (next: DesktopMediaAttachment[]) => {
      onAttachmentsChange?.(mergeAttachments(attachments, next));
    },
    [attachments, onAttachmentsChange],
  );

  const addMediaFiles = useCallback(async () => {
    if (!selectMediaAttachments) return;
    try {
      appendAttachments(await selectMediaAttachments(attachments));
    } catch (error) {
      onContextError?.(error);
    } finally {
      setContextMenuOpen(false);
    }
  }, [appendAttachments, attachments, onContextError, selectMediaAttachments]);

  const importBrowserFiles = useCallback(
    async (files: readonly globalThis.File[]) => {
      if (files.length === 0 || !importMediaAttachments) return;
      try {
        let pending = attachments;
        for (const file of files) {
          const payload: DesktopMediaImport = {
            name: file.name,
            mimeType: file.type || undefined,
            bytes: new Uint8Array(await file.arrayBuffer()),
          };
          pending = mergeAttachments(pending, await importMediaAttachments([payload], pending));
          onAttachmentsChange?.(pending);
        }
      } catch (error) {
        onContextError?.(error);
      }
    },
    [attachments, importMediaAttachments, onAttachmentsChange, onContextError],
  );

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (isComposing || event.nativeEvent.isComposing) return;

      if (mentionMenuOpen) {
        if (event.key === "Escape") {
          event.preventDefault();
          requestIdRef.current += 1;
          setMentionItems([]);
          setMentionMenuState("idle");
          return;
        }
        if (mentionItems.length === 0) return;
        if (event.key === "ArrowDown") {
          event.preventDefault();
          setActiveMentionIndex((index) => (index + 1) % mentionItems.length);
          return;
        }
        if (event.key === "ArrowUp") {
          event.preventDefault();
          setActiveMentionIndex((index) => (index - 1 + mentionItems.length) % mentionItems.length);
          return;
        }
        if (event.key === "Enter" || event.key === "Tab") {
          event.preventDefault();
          const item = mentionItems[activeMentionIndex];
          if (item) insertMention(item);
          return;
        }
      }

      if (event.key === "Escape" && contextMenuOpen) {
        event.preventDefault();
        setContextMenuOpen(false);
        return;
      }

      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        if (!sendDisabled) void onSubmit();
      }
    },
    [
      activeMentionIndex,
      insertMention,
      isComposing,
      mentionItems,
      mentionMenuOpen,
      onSubmit,
      sendDisabled,
      contextMenuOpen,
    ],
  );

  return (
    <div
      className={cx(
        "composer [position:relative] [width:min(100%,_960px)] [margin:0_auto] [padding:9px_10px_8px] [background:var(--card-strong)] [border:1px_solid_var(--border)] [border-radius:var(--radius-xl)] [box-shadow:var(--shadow),_var(--shadow-inset)] [-webkit-backdrop-filter:saturate(155%)_blur(var(--glass-blur))] [transition:border-color_var(--duration-med)_var(--ease-out),_box-shadow_var(--duration-med)_var(--ease-out),_transform_var(--duration-med)_var(--ease-out)] max-680:[padding:8px]",
        attachments.length > 0 && "composer--with-attachments",
      )}
      onDragOver={(event) => {
        if (importMediaAttachments && event.dataTransfer.types.includes("Files")) {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }
      }}
      onDrop={(event) => {
        if (!importMediaAttachments) return;
        const files = [...event.dataTransfer.files];
        if (files.length === 0) return;
        event.preventDefault();
        void importBrowserFiles(files);
      }}
    >
      {mentionMenuOpen && (
        <div
          className="composer__mentions [position:absolute] [z-index:30] [bottom:calc(100%_+_9px)] [left:0] [width:min(440px,_calc(100vw_-_28px))] [max-height:264px] [padding:5px] [overflow-y:auto] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:var(--radius-lg)] [box-shadow:var(--shadow),_var(--shadow-inset)]"
          id={mentionListId}
          aria-label="Mention suggestions"
        >
          {mentionItems.length > 0 ? (
            mentionItems.map((item, index) => (
              <button
                key={item.id}
                id={`${mentionListId}-option-${index}`}
                type="button"
                data-active={index === activeMentionIndex}
                className="composer__mention [width:100%] [display:grid] [grid-template-columns:minmax(0,_1fr)_auto] [gap:2px_10px] [padding:8px_9px] [color:var(--foreground)] [text-align:left] [background:transparent] [border:0] [border-radius:9px] [cursor:pointer]"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => insertMention(item)}
              >
                <span className="composer__mention-label [font-size:13px] [font-weight:650] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
                  {item.label}
                </span>
                {item.detail && (
                  <span className="composer__mention-detail [align-self:center] [color:var(--muted-foreground)] [font-size:11px] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
                    {item.detail}
                  </span>
                )}
                {item.documentation && (
                  <span className="composer__mention-documentation [grid-column:1_/_-1] [color:var(--muted)] [font-size:11px] [min-width:0] [overflow:hidden] [text-overflow:ellipsis] [white-space:nowrap]">
                    {item.documentation}
                  </span>
                )}
              </button>
            ))
          ) : (
            <output className="composer__mention-status [padding:10px_11px] [color:var(--muted-foreground)] [font-size:12px]">
              {mentionMenuState === "loading" ? "Loading options…" : "No matching options"}
            </output>
          )}
        </div>
      )}
      {attachments.length > 0 && (
        <div
          className="composer__attachments [margin:1px_2px_6px] [display:flex] [gap:7px] [overflow-x:auto] [scrollbar-width:thin]"
          aria-label="Attached files"
        >
          {attachments.map((attachment) => (
            <div
              className="composer-attachment [position:relative] [min-width:184px] [max-width:250px] [flex:0_0_auto] [display:flex] [align-items:stretch] [background:var(--card)] [border:1px_solid_var(--border-subtle)] [border-radius:10px]"
              key={attachment.id}
            >
              <button
                type="button"
                className="composer-attachment__preview [min-width:0] [flex:1] [padding:8px_28px_8px_9px] [display:flex] [align-items:center] [gap:8px] [color:var(--foreground)] [text-align:left] [background:transparent] [border:0] [border-radius:inherit] [cursor:pointer] [&_>_svg]:[width:18px] [&_>_svg]:[height:18px] [&_>_svg]:[flex:0_0_18px] [&_>_svg]:[color:var(--muted)] [&_>_span]:[min-width:0] [&_>_span]:[display:flex] [&_>_span]:[flex-direction:column] [&_strong]:[overflow:hidden] [&_strong]:[text-overflow:ellipsis] [&_strong]:[white-space:nowrap] [&_small]:[overflow:hidden] [&_small]:[text-overflow:ellipsis] [&_small]:[white-space:nowrap] [&_strong]:[font-size:11.5px] [&_strong]:[font-weight:600] [&_small]:[color:var(--muted-foreground)] [&_small]:[font-size:9.5px]"
                onClick={() => onPreviewAttachment?.(attachment)}
                disabled={!onPreviewAttachment}
                aria-label={`Preview ${attachment.name}`}
              >
                <AttachmentIcon attachment={attachment} />
                <span>
                  <strong>{attachment.name}</strong>
                  <small>{formatMediaBytes(attachment.sizeBytes)}</small>
                </span>
              </button>
              <button
                type="button"
                className="composer-attachment__remove [position:absolute] [top:5px] [right:5px] [width:20px] [height:20px] [display:grid] [place-items:center] [color:var(--muted-foreground)] [background:transparent] [border:0] [border-radius:6px] [cursor:pointer] [&_svg]:[width:12px] [&_svg]:[height:12px]"
                onClick={() =>
                  onAttachmentsChange?.(
                    attachments.filter((candidate) => candidate.id !== attachment.id),
                  )
                }
                aria-label={`Remove ${attachment.name}`}
              >
                <X aria-hidden="true" />
              </button>
            </div>
          ))}
        </div>
      )}
      <textarea
        ref={assignTextareaRef}
        value={value}
        onChange={(event) => {
          onChange(event.target.value);
          syncCursor(event.target);
        }}
        onSelect={(event) => syncCursor(event.currentTarget)}
        onClick={(event) => syncCursor(event.currentTarget)}
        onFocus={(event) => {
          syncCursor(event.currentTarget);
          onFocus?.();
        }}
        onCompositionStart={() => setIsComposing(true)}
        onCompositionEnd={(event) => {
          setIsComposing(false);
          syncCursor(event.currentTarget);
        }}
        onPaste={(event) => {
          if (!importMediaAttachments) return;
          const files = [...event.clipboardData.files];
          if (files.length === 0) return;
          event.preventDefault();
          void importBrowserFiles(files);
        }}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="composer__textarea [display:block] [width:100%] [min-height:48px] [max-height:240px] [padding:5px_7px_9px] [resize:none] [overflow-y:auto] [color:var(--foreground)] [background:transparent] [border:0] [outline:0] [font-size:15px] [line-height:1.5] max-680:[min-height:40px] max-680:[max-height:112px] max-680:[padding-bottom:6px]"
        rows={1}
        disabled={disabled}
      />
      {error && (
        <div
          className="composer__error [margin:0_4px_6px] [color:var(--danger)] [font-size:11px] [line-height:1.35]"
          role="alert"
        >
          {error}
        </div>
      )}
      <div className="composer__footer [display:flex] [align-items:center] [gap:7px] [min-height:36px]">
        <button
          type="button"
          className="composer__context [flex:0_0_auto] [color:var(--muted)] [background:transparent] [border-radius:9px] [display:inline-grid] [width:36px] [height:36px] [place-items:center] [border:0] [cursor:pointer] [transition:background_var(--duration-fast)_var(--ease-out),_transform_var(--duration-fast)_var(--ease-out),_opacity_var(--duration-fast)_var(--ease-out)] [&_svg]:[width:19px] [&_svg]:[height:19px]"
          onClick={() => setContextMenuOpen((open) => !open)}
          disabled={disabled}
          aria-label="Add context"
          aria-expanded={contextMenuOpen}
        >
          <Plus aria-hidden="true" />
        </button>
        {contextMenuOpen && (
          <section
            className="composer__context-menu [position:absolute] [z-index:31] [bottom:52px] [left:10px] [min-width:260px] [padding:7px] [background:var(--card-solid)] [border:1px_solid_var(--border)] [border-radius:10px] [box-shadow:var(--shadow),_var(--shadow-inset)] [&_button]:[width:100%] [&_button]:[display:flex] [&_button]:[align-items:center] [&_button]:[gap:10px] [&_button]:[min-height:34px] [&_button]:[padding:7px_9px] [&_button]:[color:var(--foreground)] [&_button]:[text-align:left] [&_button]:[background:transparent] [&_button]:[border:0] [&_button]:[border-radius:7px] [&_button]:[cursor:pointer] [&_button]:[font-size:15px] [&_button_svg]:[width:18px] [&_button_svg]:[height:18px] [&_button_svg]:[color:var(--muted)]"
            aria-label="Add context"
          >
            <p className="composer__context-menu-title [margin:1px_8px_6px] [color:var(--muted-foreground)] [font-size:13px] [font-weight:560]">
              Add
            </p>
            {selectMediaAttachments && (
              <button type="button" onClick={() => void addMediaFiles()}>
                <Paperclip aria-hidden="true" />
                <span>Files and photos</span>
              </button>
            )}
            <button type="button" onClick={() => void addFilesAndFolders()}>
              <FolderOpen aria-hidden="true" />
              <span>File or folder context</span>
            </button>
          </section>
        )}
        <div className="composer__controls [min-width:0] [display:flex] [flex:1_1_auto] [align-items:center] [justify-content:flex-end] [gap:4px]">
          {children}
        </div>
        <button
          type="button"
          className="composer__submit [flex:0_0_auto] [color:var(--primary-foreground)] [background:var(--primary)] [border-radius:999px] [box-shadow:0_6px_16px_rgba(0,_0,_0,_0.2),_inset_0_1px_0_rgba(255,_255,_255,_0.22)] [display:inline-grid] [width:36px] [height:36px] [place-items:center] [border:0] [cursor:pointer] [transition:background_var(--duration-fast)_var(--ease-out),_transform_var(--duration-fast)_var(--ease-out),_opacity_var(--duration-fast)_var(--ease-out)] [&_svg]:[width:16px] [&_svg]:[height:16px]"
          onClick={() => void (running ? onStop() : onSubmit())}
          disabled={sendDisabled}
          title={sendTitle}
          aria-label={running ? "Stop generating" : "Send message"}
        >
          {running ? <Square aria-hidden="true" /> : <ArrowUp aria-hidden="true" />}
        </button>
      </div>
    </div>
  );
}

function mergeAttachments(
  existing: readonly DesktopMediaAttachment[],
  next: readonly DesktopMediaAttachment[],
): DesktopMediaAttachment[] {
  const byId = new Map(existing.map((attachment) => [attachment.id, attachment]));
  for (const attachment of next) byId.set(attachment.id, attachment);
  return [...byId.values()];
}

function getMentionContext(text: string, cursorOffset: number): MentionContext | null {
  const beforeCursor = text.slice(0, cursorOffset);
  const tokenStart =
    Math.max(
      beforeCursor.lastIndexOf(" "),
      beforeCursor.lastIndexOf("\n"),
      beforeCursor.lastIndexOf("\t"),
      beforeCursor.lastIndexOf("("),
      beforeCursor.lastIndexOf("["),
      beforeCursor.lastIndexOf("{"),
      beforeCursor.lastIndexOf(","),
    ) + 1;
  const token = beforeCursor.slice(tokenStart);
  const trigger = token[0];
  return trigger === "@" || trigger === "$" ? { start: tokenStart, token, trigger } : null;
}

function positionAtOffset(text: string, offset: number): { line: number; character: number } {
  const beforeCursor = text.slice(0, offset);
  const line = beforeCursor.split("\n").length - 1;
  const lineStart = beforeCursor.lastIndexOf("\n") + 1;
  return { line, character: beforeCursor.length - lineStart };
}

function parseMentionItems(result: unknown): MentionCompletionItem[] {
  const rawItems = Array.isArray(result)
    ? result
    : isRecord(result) && Array.isArray(result.items)
      ? result.items
      : [];
  return rawItems.flatMap((item, index) => {
    if (!isRecord(item)) return [];
    const label = typeof item.label === "string" ? item.label : undefined;
    const textEdit = isRecord(item.textEdit) ? item.textEdit : undefined;
    const insertText =
      typeof textEdit?.newText === "string"
        ? textEdit.newText
        : typeof item.insertText === "string"
          ? item.insertText
          : label;
    if (!label || !insertText) return [];
    return [
      {
        id: `${label}:${insertText}:${index}`,
        label,
        detail: typeof item.detail === "string" ? item.detail : undefined,
        documentation: documentationText(item.documentation),
        insertText,
      },
    ];
  });
}

function documentationText(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  if (isRecord(value) && typeof value.value === "string") return value.value;
  return undefined;
}

function prefixMentionItems(servers: MentionServer[], trigger: "@" | "$"): MentionCompletionItem[] {
  return uniqueMentionItems(
    servers.flatMap((server) =>
      (server.mentionPrefixes ?? [])
        .filter((prefix) => prefix.startsWith(trigger) && prefix !== trigger)
        .map((prefix) => ({
          id: `prefix:${server.id}:${prefix}`,
          label: prefix,
          detail: server.name ?? server.id,
          documentation: server.description,
          insertText: prefix,
        })),
    ),
  );
}

function uniqueMentionItems(items: MentionCompletionItem[]): MentionCompletionItem[] {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = `${item.label}\0${item.insertText}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function fileReference(path: string): string {
  return /\s/.test(path) ? `@"${path.replaceAll('"', '\\"')}"` : `@${path}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

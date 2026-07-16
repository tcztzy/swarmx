import { ArrowUp, Paperclip, Plus, Square } from "lucide-react";
import type React from "react";
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";

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
  onContextError?: (error: unknown) => void;
  error?: string | null;
  onChange: (value: string) => void;
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
  onContextError,
  error,
  onChange,
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
  }, [cursorOffset, focusAt, onChange, onContextError, selectFilesAndFolders, value]);

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
    <div className="composer">
      {mentionMenuOpen && (
        <div className="composer__mentions" id={mentionListId} aria-label="Mention suggestions">
          {mentionItems.length > 0 ? (
            mentionItems.map((item, index) => (
              <button
                key={item.id}
                id={`${mentionListId}-option-${index}`}
                type="button"
                data-active={index === activeMentionIndex}
                className="composer__mention"
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => insertMention(item)}
              >
                <span className="composer__mention-label">{item.label}</span>
                {item.detail && <span className="composer__mention-detail">{item.detail}</span>}
                {item.documentation && (
                  <span className="composer__mention-documentation">{item.documentation}</span>
                )}
              </button>
            ))
          ) : (
            <output className="composer__mention-status">
              {mentionMenuState === "loading" ? "Loading options…" : "No matching options"}
            </output>
          )}
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
        onFocus={(event) => syncCursor(event.currentTarget)}
        onCompositionStart={() => setIsComposing(true)}
        onCompositionEnd={(event) => {
          setIsComposing(false);
          syncCursor(event.currentTarget);
        }}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="composer__textarea"
        rows={1}
        disabled={disabled}
      />
      {error && (
        <div className="composer__error" role="alert">
          {error}
        </div>
      )}
      <div className="composer__footer">
        <button
          type="button"
          className="composer__context"
          onClick={() => setContextMenuOpen((open) => !open)}
          disabled={disabled}
          aria-label="Add context"
          aria-expanded={contextMenuOpen}
        >
          <Plus aria-hidden="true" />
        </button>
        {contextMenuOpen && (
          <section className="composer__context-menu" aria-label="Add context">
            <p className="composer__context-menu-title">Add</p>
            <button type="button" onClick={() => void addFilesAndFolders()}>
              <Paperclip aria-hidden="true" />
              <span>Files and folders</span>
            </button>
          </section>
        )}
        <div className="composer__controls">{children}</div>
        <button
          type="button"
          className="composer__submit"
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

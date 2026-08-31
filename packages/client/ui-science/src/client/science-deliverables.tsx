import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-remotes/client";
import type {} from "@deepseek-ai/dsh-api-session-controller/client";
import type {
  ChatFileMentions,
  ChatSnapshot,
  TurnTailOwnerProps,
} from "@deepseek-ai/dsh-client-ui-chat/client";
import type { ConversationNodeDefinition } from "@deepseek-ai/dsh-client-ui-conversation/client";
import type { MarkdownFileMentions } from "@deepseek-ai/dsh-client-ui-primitives";
import type {} from "@deepseek-ai/dsh-client-ui-renderer/client";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import { typstRelativePathSchema } from "@swarmx/dsh-science/types";
import type { SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import { useMemo } from "react";
import css from "./science-deliverables.module.css";

interface ProducedPath {
  readonly seq: number;
  readonly path: string;
}

export interface ScienceDeliverablesTurnData {
  readonly produced: readonly ProducedPath[];
  readonly referenced: readonly ProducedPath[];
}

declare module "@deepseek-ai/dsh-client-ui-conversation/client" {
  interface ConversationTurnDataMap {
    deliverables: ScienceDeliverablesTurnData;
  }
}

interface DeliverablesState extends ScienceDeliverablesTurnData {
  readonly turn: number;
  readonly calls: ReadonlyMap<string, string | null>;
}

interface TurnFilesProps {
  readonly paths: readonly string[];
  readonly openFile: (path: string) => void;
  readonly openTypst: (path: string) => void;
}

type ScienceConversationFilesProps = PropsRuntime<"conversation.chat.turnTail.items"> &
  Omit<TurnFilesProps, "paths">;

function isAppendEvent(event: unknown): boolean {
  return (
    typeof event === "object" &&
    event !== null &&
    "surfaceOp" in event &&
    event.surfaceOp === "append"
  );
}

function resolveWorkspacePath(cwd: string | undefined, path: string): string {
  if (path.startsWith("/") || /^[A-Za-z]:[\\/]/u.test(path) || path.startsWith("\\\\")) {
    return path;
  }
  if (cwd === undefined || cwd === "") return path;
  return `${cwd.replace(/[/\\]+$/u, "")}/${path.replace(/^[/\\]+/u, "")}`;
}

function mutationPath(name: string, argsRaw: string): string | null {
  let args: unknown;
  try {
    args = JSON.parse(argsRaw);
  } catch {
    return null;
  }
  if (typeof args !== "object" || args === null || Array.isArray(args)) return null;
  const input = args as Record<string, unknown>;
  const path = (value: unknown) =>
    typeof value === "string" && value.trim().length > 0 ? value : null;
  if (name === "write") return typeof input.content === "string" ? path(input.file_path) : null;
  if (name === "edit") {
    return typeof input.old_string === "string" &&
      input.old_string.length > 0 &&
      typeof input.new_string === "string" &&
      input.old_string !== input.new_string &&
      (input.replace_all === undefined || typeof input.replace_all === "boolean")
      ? path(input.file_path)
      : null;
  }
  if (name !== "str_replace_editor") return null;
  const editorPath = path(input.path);
  if (editorPath === null) return null;
  if (input.command === "create") {
    return typeof input.file_text === "string" ? editorPath : null;
  }
  if (input.command === "str_replace") {
    return typeof input.old_str === "string" &&
      input.old_str.length > 0 &&
      (input.new_str === undefined || typeof input.new_str === "string")
      ? editorPath
      : null;
  }
  return input.command === "insert" &&
    typeof input.insert_line === "number" &&
    Number.isInteger(input.insert_line) &&
    input.insert_line >= 0 &&
    typeof input.new_str === "string"
    ? editorPath
    : null;
}

function normalizedReferencedTypstPath(value: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURI(value);
  } catch {
    return null;
  }
  const normalized = decoded.replaceAll("\\", "/").replace(/^\.\//u, "");
  return safeRelativePath(normalized) && isTypstPaperPath(normalized) ? normalized : null;
}

/** Recover explicit Typst paths without treating absolute, URL, or traversal text as a locator. */
export function referencedTypstPaths(text: string): readonly string[] {
  const paths: string[] = [];
  const seen = new Set<string>();
  const add = (value: string) => {
    const path = normalizedReferencedTypstPath(value);
    if (path === null || seen.has(path)) return;
    seen.add(path);
    paths.push(path);
  };
  const withoutMarkdownLinks = text.replace(
    /\[[^\]\r\n]*\]\(\s*<?([^\s()<>]+?\.(?:typst|typ))>?(?:\s+["'][^)\r\n]*["'])?\s*\)/giu,
    (_match, destination: string) => {
      add(destination);
      return " ";
    },
  );
  const pathPattern =
    /(?:^|[\s"'`([{<:,=])((?:\.\/)?(?:[\p{L}\p{N}_+@%.-]+\/)*[\p{L}\p{N}_+@%.-]+\.(?:typst|typ))(?=$|[\s"'`\])}>:,;!，。；！])/giu;
  for (const match of withoutMarkdownLinks.matchAll(pathPattern)) {
    const value = match[1];
    if (value !== undefined) add(value);
  }
  return paths;
}

function assistantMessageText(message: unknown): string {
  if (typeof message !== "object" || message === null || !("content" in message)) return "";
  if (!Array.isArray(message.content)) return "";
  return message.content
    .map((block) => {
      if (
        typeof block !== "object" ||
        block === null ||
        !("type" in block) ||
        block.type !== "text" ||
        !("text" in block) ||
        typeof block.text !== "string"
      ) {
        return "";
      }
      return block.text;
    })
    .filter((value) => value.length > 0)
    .join("\n");
}

/** Mutation-derived file accumulator with a public Typst opener. */
export const scienceDeliverablesDefinition: ConversationNodeDefinition<DeliverablesState> = {
  kind: "deliverables",
  match: (event) => {
    if (event.type === "turn/start") return { id: String(event.data.turn), role: "start" };
    if (event.type === "tool/call") return { id: String(event.data.turn), role: "update" };
    if (event.type === "assistant/message") {
      return { id: String(event.data.turn), role: "update" };
    }
    if (event.type === "tool/result" && isAppendEvent(event)) {
      return { id: String(event.data.turn), role: "update" };
    }
    return null;
  },
  start: (_context, match) => {
    if (match.event.type !== "turn/start") {
      throw new Error("science deliverables start requires turn/start");
    }
    return { turn: match.event.data.turn, calls: new Map(), produced: [], referenced: [] };
  },
  update: (context, match) => {
    if (match.event.type === "tool/call") {
      const calls = new Map(context.state.calls);
      calls.set(
        String(match.event.data.callId),
        mutationPath(match.event.data.name, match.event.data.arguments),
      );
      const referenced = referencedTypstPaths(match.event.data.arguments).map((path) => ({
        seq: match.event.seq,
        path,
      }));
      return {
        ...context.state,
        calls,
        referenced: [...context.state.referenced, ...referenced],
      };
    }
    if (match.event.type === "assistant/message") {
      const referenced = referencedTypstPaths(assistantMessageText(match.event.data.message)).map(
        (path) => ({ seq: match.event.seq, path }),
      );
      return referenced.length === 0
        ? context.state
        : {
            ...context.state,
            referenced: [...context.state.referenced, ...referenced],
          };
    }
    if (match.event.type !== "tool/result") return context.state;
    if (match.event.data.message.content[0]?.isError === true) return context.state;
    const callId = String(match.event.data.message.source.callId);
    const path = context.state.calls.get(callId);
    return path === null || path === undefined
      ? context.state
      : {
          ...context.state,
          produced: [...context.state.produced, { seq: match.event.seq, path }],
        };
  },
  buildLocationData: (context, scope) =>
    scope !== "turn" || context.state === undefined
      ? null
      : {
          kind: "turn",
          turn: context.state.turn,
          key: "deliverables",
          value: {
            produced: context.state.produced,
            referenced: context.state.referenced,
          },
        },
};

export function basename(path: string): string {
  const at = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return at === -1 ? path : path.slice(at + 1);
}

export function isTypstPaperPath(path: string): boolean {
  return /\.(?:typ|typst)$/iu.test(path);
}

function safeRelativePath(path: string): boolean {
  return (
    path.length > 0 &&
    !path.startsWith("/") &&
    !/^[a-z]:[\\/]/iu.test(path) &&
    !path.includes("\\") &&
    !path.includes("\0") &&
    path.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..")
  );
}

export function workspaceRelativeTypstPath(path: string, cwd?: string): string | null {
  const normalized = path.replaceAll("\\", "/");
  if (typstRelativePathSchema.safeParse(normalized).success) return normalized;
  if (cwd === undefined) return null;
  const root = cwd.replaceAll("\\", "/").replace(/\/$/u, "");
  const prefix = `${root}/`;
  if (!normalized.startsWith(prefix)) return null;
  const relativePath = normalized.slice(prefix.length);
  return typstRelativePathSchema.safeParse(relativePath).success ? relativePath : null;
}

export function typstPaperSideViewEntry(relativePath: string): SideViewEntry {
  const parsed = typstRelativePathSchema.safeParse(relativePath);
  if (!parsed.success) {
    throw new TypeError("Typst Side View path must be workspace-relative");
  }
  return {
    id: `science-typst:${parsed.data}`,
    kind: "science-typst",
    title: basename(parsed.data),
    mode: "workbench",
    payload: { relativePath: parsed.data },
  };
}

function samePathEvidence(left: string, right: string): boolean {
  const a = left.replaceAll("\\", "/").replace(/^\.\//u, "");
  const b = right.replaceAll("\\", "/").replace(/^\.\//u, "");
  return a === b || a.endsWith(`/${b}`) || b.endsWith(`/${a}`);
}

export function filesForClosing(
  data: Readonly<ScienceDeliverablesTurnData> | undefined,
  seq = Number.POSITIVE_INFINITY,
): readonly string[] {
  if (data === undefined) return [];
  const paths: string[] = [];
  const evidence = [
    ...data.produced.map((item) => ({ ...item, priority: 0 })),
    ...data.referenced.map((item) => ({ ...item, priority: 1 })),
  ].sort((left, right) => left.seq - right.seq || left.priority - right.priority);
  for (const item of evidence) {
    if (item.seq > seq || paths.some((path) => samePathEvidence(path, item.path))) continue;
    paths.push(item.path);
  }
  return paths;
}

function selectTurnFiles(owner: TurnTailOwnerProps): readonly string[] | null {
  const paths = filesForClosing(owner.turn.data.get("deliverables"), owner.seq);
  return paths.length === 0 ? null : paths;
}

function onlyPathWithBasename(paths: readonly string[], value: string): string | undefined {
  const matches = paths.filter((path) => basename(path) === value);
  return matches.length === 1 ? matches[0] : undefined;
}

export function scienceTurnFileMentions(
  paths: readonly string[],
  openTypst: (path: string) => void,
  openFile: (path: string) => void,
  label: (path: string) => string,
): MarkdownFileMentions {
  const mention = (path: string) => ({
    open: () => (isTypstPaperPath(path) ? openTypst(path) : openFile(path)),
    label: label(path),
    title: path,
  });
  return {
    resolve(value) {
      const path = paths.includes(value) ? value : onlyPathWithBasename(paths, value);
      return path === undefined ? undefined : mention(path);
    },
  };
}

function ScienceTurnFiles({ paths, openFile, openTypst }: TurnFilesProps) {
  const visible = paths.slice(0, 6);
  const hidden = paths.length - visible.length;
  return (
    <section className={css.root} aria-label="Files">
      <span className={css.label}>Files</span>
      <span className={css.row}>
        {visible.map((path) => (
          <button
            key={path}
            type="button"
            className={css.file}
            title={path}
            aria-label={`${isTypstPaperPath(path) ? "Open paper preview" : "Open file"}: ${path}`}
            onClick={() => (isTypstPaperPath(path) ? openTypst(path) : openFile(path))}
          >
            {basename(path)}
          </button>
        ))}
        {hidden > 0 && <span className={css.more}>+ {hidden} files</span>}
      </span>
      {hidden > 0 && (
        <button type="button" className={css.showFolder} onClick={() => openFile(".")}>
          Show in folder
        </button>
      )}
    </section>
  );
}

/** Rebuild one completed Turn's persistent Files row from public conversation evidence. */
export function ScienceConversationFiles({
  turn,
  useChat,
  openFile,
  openTypst,
}: ScienceConversationFilesProps) {
  const data = useChat((snapshot: ChatSnapshot) =>
    snapshot.timeline.turns.get(turn)?.data.get("deliverables"),
  );
  const paths = useMemo(() => filesForClosing(data), [data]);
  return paths.length === 0 ? null : (
    <ScienceTurnFiles paths={paths} openFile={openFile} openTypst={openTypst} />
  );
}

function sessionCwd(ctx: Context, sessionId: SessionId): string | undefined {
  return ctx.sessions.list.getSnapshot().byId[sessionId]?.cwd;
}

async function openWorkspacePath(ctx: Context, sessionId: SessionId, path: string): Promise<void> {
  const result = await ctx.remote.session.openWorkspacePath({
    path: resolveWorkspacePath(sessionCwd(ctx, sessionId), path),
  });
  if (!result.ok) throw result.error;
}

/** Register Typst-aware deliverable reconstruction and routing. */
export function registerScienceDeliverables(ctx: Context): void {
  ctx.uiConversation.events.register(scienceDeliverablesDefinition);
  const openTypst = (sessionId: SessionId, path: string): boolean => {
    const relativePath = workspaceRelativeTypstPath(path, sessionCwd(ctx, sessionId));
    if (relativePath === null) return false;
    ctx.sideView.open(sessionId, typstPaperSideViewEntry(relativePath));
    return true;
  };
  const actions = (sessionId: SessionId) => ({
    openFile: (path: string) => {
      void openWorkspacePath(ctx, sessionId, path);
    },
    openTypst: (path: string) => {
      if (!openTypst(sessionId, path)) {
        void openWorkspacePath(ctx, sessionId, path);
      }
    },
  });
  ctx.slots.inject("conversation.chat.turnTail.items", () =>
    ctx.slots.register(
      {
        name: "conversation.chat.turnTail.items",
        id: "science-files",
        inject: actions,
      },
      ScienceConversationFiles,
    ),
  );
  const fileMentions: ChatFileMentions = {
    forClosing(owner) {
      const paths = selectTurnFiles(owner);
      const sessionId = ctx.sessions.list.getSnapshot().current;
      if (paths === null || sessionId === undefined) return undefined;
      return scienceTurnFileMentions(
        paths,
        (path) => {
          if (!openTypst(sessionId, path)) owner.openFile(path);
        },
        owner.openFile,
        (path) => `Open ${path}`,
      );
    },
  };
  ctx.provide("chatFileMentions", fileMentions);
}

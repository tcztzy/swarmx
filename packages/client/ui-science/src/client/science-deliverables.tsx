import type { Context } from "@deepseek-ai/cordis";
import type {
  ConversationNodeDefinition,
  SessionId,
  ToolResultNode,
} from "@deepseek-ai/dsh-client-runtime/client";
import type {
  ChatFileMentions,
  TurnTailOwnerProps,
} from "@deepseek-ai/dsh-client-ui-conversation/client";
import type { MarkdownFileMentions } from "@deepseek-ai/dsh-client-ui-primitives";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
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

declare module "@deepseek-ai/dsh-client-runtime/client" {
  interface ConversationTurnDataMap {
    deliverables: ScienceDeliverablesTurnData;
  }
}

interface DeliverablesState extends ScienceDeliverablesTurnData {
  readonly turn: number;
  readonly calls: ReadonlyMap<string, ToolResultNode["callView"]>;
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

function producedPaths(view: ToolResultNode["callView"]): readonly string[] {
  if (view === null) return [];
  if (view.card === "diff") return (view.locations ?? []).map(({ path }) => path);
  if (view.card === "generic" && view.kind === "edit") {
    return (view.locations ?? []).map(({ path }) => path);
  }
  return [];
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

/** Behavior-equivalent rc.2 mutation-derived file accumulator with a public Typst opener. */
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
        match.view?.for === "call" ? match.view.view : null,
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
    const additions = producedPaths(context.state.calls.get(callId) ?? null).map((path) => ({
      seq: match.event.seq,
      path,
    }));
    return additions.length === 0
      ? context.state
      : { ...context.state, produced: [...context.state.produced, ...additions] };
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

function workspaceRelativeProducedPath(path: string, cwd?: string): string | null {
  const normalized = path.replaceAll("\\", "/").replace(/^\.\//u, "");
  if (safeRelativePath(normalized)) return normalized;
  if (cwd === undefined) return null;
  const root = cwd.replaceAll("\\", "/").replace(/\/$/u, "");
  const prefix = `${root}/`;
  if (!normalized.startsWith(prefix)) return null;
  const relativePath = normalized.slice(prefix.length);
  return safeRelativePath(relativePath) ? relativePath : null;
}

function markdownFileDestination(value: string): string | null {
  let decoded: string;
  try {
    decoded = decodeURI(value);
  } catch {
    return null;
  }
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/u.test(decoded) || /[?#]/u.test(decoded)) return null;
  const relativePath = decoded.replace(/^\.\//u, "");
  return safeRelativePath(relativePath) ? relativePath : null;
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
  cwd?: string,
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
    resolveLink(value) {
      const destination = markdownFileDestination(value);
      if (destination === null) return undefined;
      const exact = paths.filter(
        (path) => workspaceRelativeProducedPath(path, cwd) === destination,
      );
      if (exact.length === 1 && exact[0] !== undefined) return mention(exact[0]);
      if (exact.length > 1 || destination.includes("/")) return undefined;
      const path = onlyPathWithBasename(paths, destination);
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
  useSession,
  openFile,
  openTypst,
}: ScienceConversationFilesProps) {
  const data = useSession((snapshot) =>
    snapshot.chat.timeline.turns.get(turn)?.data.get("deliverables"),
  );
  const paths = useMemo(() => filesForClosing(data), [data]);
  return paths.length === 0 ? null : (
    <ScienceTurnFiles paths={paths} openFile={openFile} openTypst={openTypst} />
  );
}

function sessionCwd(ctx: Context, sessionId: SessionId): string | undefined {
  return ctx.sessions.list.getSnapshot().byId[sessionId]?.cwd;
}

/** Replace rc.2's non-decoratable deliverables client with behavior-equivalent Typst routing. */
export function registerScienceDeliverables(ctx: Context): void {
  ctx.conversationEvents.register(scienceDeliverablesDefinition);
  const openTypst = (sessionId: SessionId, path: string): boolean => {
    const relativePath = workspaceRelativeTypstPath(path, sessionCwd(ctx, sessionId));
    if (relativePath === null) return false;
    ctx.sideView.open(sessionId, typstPaperSideViewEntry(relativePath));
    return true;
  };
  const actions = (sessionId: SessionId) => ({
    openFile: (path: string) => {
      void ctx.workspaces.openPath(resolveWorkspacePath(sessionCwd(ctx, sessionId), path));
    },
    openTypst: (path: string) => {
      if (!openTypst(sessionId, path)) {
        void ctx.workspaces.openPath(resolveWorkspacePath(sessionCwd(ctx, sessionId), path));
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
        sessionCwd(ctx, sessionId),
      );
    },
  };
  ctx.provide("chatFileMentions", fileMentions);
}

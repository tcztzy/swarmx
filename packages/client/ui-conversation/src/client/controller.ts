/** Per-session orchestration for failed-turn Retry and user-message Edit. */
import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-agent-presets/types";
import type {} from "@deepseek-ai/dsh-api-session-controller/client";
import type {} from "@deepseek-ai/dsh-api-workspace-controller/client";
import type { ChatSnapshot } from "@deepseek-ai/dsh-client-ui-chat/client";
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import { canRerunTurn } from "../fork-boundary.js";
import { prepareRerunSession, type RerunSessions, rerunTurn } from "../rerun.js";
import { turnTextOf } from "../turn-origin.js";

/** Create a source-adjacent session through the addressable alpha.2 object layer. */
export async function createSibling(ctx: Context, sourceId: SessionId): Promise<SessionId> {
  const source = ctx.sessions.list.getSnapshot().byId[sourceId];
  if (source === undefined) throw new Error(`source session ${String(sourceId)} is unavailable`);
  const workspaces = ctx.workspaces.list.getSnapshot().items;
  let workspace = workspaces.find((candidate) => candidate.sessionIds.includes(sourceId));
  if (workspace === undefined) {
    if (source.cwd === undefined) {
      throw new Error(`source session ${String(sourceId)} has no workspace or working directory`);
    }
    workspace = await ctx.workspaces.create({ path: source.cwd });
  }
  const agentPreset = source.projectionValues?.agentPreset;
  return ctx.sessions.create({
    workspaceId: workspace.workspaceId,
    ...(typeof agentPreset === "string" ? { agentPreset } : {}),
  });
}

/** What the controller needs from the host services. */
export interface ControllerDeps {
  readonly sessions: RerunSessions;
  readonly snapshot: (
    sessionId: SessionId,
  ) => { readonly chat: ChatSnapshot; readonly hasMore: boolean } | undefined;
  readonly setDraft: (sessionId: SessionId, text: string) => void;
}

/** One session's re-run actions. */
export class RerunController {
  constructor(
    private readonly deps: ControllerDeps,
    private readonly sessionId: SessionId,
  ) {}

  /** The live conversation snapshot, or undefined when the session is gone. */
  #snapshot() {
    return this.deps.snapshot(this.sessionId);
  }

  /** Whether a closed turn and its opening text are available in this window. */
  canRerun(turn: number): boolean {
    const snapshot = this.#snapshot();
    if (snapshot === undefined) return false;
    return (
      turnTextOf(snapshot.chat, turn) !== undefined &&
      canRerunTurn(snapshot.chat.timeline, turn, !snapshot.hasMore)
    );
  }

  /** Prepare before the failed turn and send its original user text. */
  async rerun(turn: number): Promise<void> {
    const snapshot = this.#snapshot();
    if (snapshot === undefined) return;
    const text = turnTextOf(snapshot.chat, turn);
    if (text === undefined || !canRerunTurn(snapshot.chat.timeline, turn, !snapshot.hasMore))
      return;
    await rerunTurn(this.deps.sessions, {
      sessionId: this.sessionId,
      turn,
      timeline: snapshot.chat.timeline,
      windowReachesStart: !snapshot.hasMore,
      text,
    });
  }

  /** Prepare before a user message and seed the new composer for revision. */
  async beginEdit(turn: number, text: string): Promise<void> {
    const snapshot = this.#snapshot();
    const normalized = text.trim();
    if (
      snapshot === undefined ||
      normalized === "" ||
      !canRerunTurn(snapshot.chat.timeline, turn, !snapshot.hasMore)
    ) {
      return;
    }
    const child = await prepareRerunSession(this.deps.sessions, {
      sessionId: this.sessionId,
      turn,
      timeline: snapshot.chat.timeline,
      windowReachesStart: !snapshot.hasMore,
    });
    if (child !== undefined) this.deps.setDraft(child, normalized);
  }
}

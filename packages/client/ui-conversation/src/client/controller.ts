/** Per-session orchestration for failed-turn Retry and user-message Edit. */
import type { SessionFace, SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import { canRerunTurn } from "../fork-boundary.js";
import { prepareRerunSession, type RerunSessions, rerunTurn } from "../rerun.js";
import { type LookupNode, turnTextOf } from "../turn-origin.js";

/** What the controller needs from the host services. */
export interface ControllerDeps {
  readonly sessions: RerunSessions & {
    binding(id: SessionId): { session: SessionFace } | undefined;
  };
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
    return this.deps.sessions.binding(this.sessionId)?.session.getSnapshot();
  }

  /** Whether a closed turn and its opening text are available in this window. */
  canRerun(turn: number): boolean {
    const snapshot = this.#snapshot();
    const location = snapshot?.chat.timeline.turns.get(turn);
    if (snapshot === undefined || location === undefined) return false;
    return (
      turnTextOf(snapshot.nodes as readonly LookupNode[], location) !== undefined &&
      canRerunTurn(snapshot.chat.timeline, turn, !snapshot.hasMore)
    );
  }

  /** Prepare before the failed turn and send its original user text. */
  async rerun(turn: number): Promise<void> {
    const snapshot = this.#snapshot();
    const location = snapshot?.chat.timeline.turns.get(turn);
    if (snapshot === undefined || location === undefined) return;
    const text = turnTextOf(snapshot.nodes as readonly LookupNode[], location);
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

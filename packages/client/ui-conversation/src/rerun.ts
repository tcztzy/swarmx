/**
 * Re-running a turn: prepare a session at the boundary before it, then prompt
 * that session. Later turns use a fork; the first uses a fresh sibling because
 * DSH's fork API cannot represent an empty completed-turn prefix.
 */
import type { SessionId } from "@deepseek-ai/dsh-session/types";
import { resolveForkBoundary, type Timeline } from "./fork-boundary.js";

/** The sessions-service subset a re-run needs. */
export interface RerunSessions {
  /** Create a fresh session carrying the source's workspace and agent preset. */
  createSibling(sessionId: SessionId): Promise<SessionId>;
  /** Fork a prefix of `sessionId` into a new child session. */
  fork(opts: { sessionId: SessionId; atSeq?: number; increaseTitle?: boolean }): Promise<SessionId>;
  /** Make a session the active one. */
  open(sessionId: SessionId): void;
  /** Send text into a session as one user turn. */
  prompt(sessionId: SessionId, text: string): Promise<void>;
}

/** What one re-run needs to know. */
export interface RerunRequest {
  /** Session holding the turn being re-run. */
  readonly sessionId: SessionId;
  /** Turn index being re-run. */
  readonly turn: number;
  /** The session's turn timeline. */
  readonly timeline: Timeline;
  /** Whether the loaded window reaches the session start (`!snapshot.hasMore`). */
  readonly windowReachesStart: boolean;
  /** Sole user text appended to the child; source failure metadata is never forwarded. */
  readonly text: string;
}

/** Prepare a session immediately before the requested turn and open it. */
export async function prepareRerunSession(
  sessions: Pick<RerunSessions, "createSibling" | "fork" | "open">,
  request: Omit<RerunRequest, "text">,
): Promise<SessionId | undefined> {
  const boundary = resolveForkBoundary(request.timeline, request.turn, request.windowReachesStart);
  if (boundary === undefined) return undefined;
  const child =
    boundary.kind === "fresh"
      ? await sessions.createSibling(request.sessionId)
      : await sessions.fork({
          sessionId: request.sessionId,
          atSeq: boundary.atSeq,
          increaseTitle: true,
        });
  sessions.open(child);
  return child;
}

/**
 * Prepare a session before `turn` and prompt it with `text`.
 *
 * The source session is never modified: its superseded prompt, retry records,
 * and terminal failure remain durable there. The child contains only the
 * completed prefix and this newly appended user text, so neither its Chat
 * projection nor the model request can include source failure metadata.
 * Preparation that succeeds before a failing prompt leaves the new session
 * open — silently discarding a session the host already created would lose
 * user-visible state.
 *
 * @param sessions - the sessions service.
 * @param request - the turn to re-run and the text to send.
 * @returns the child session id, or undefined when no boundary was resolvable.
 * @throws whatever session creation, `fork`, or `prompt` rejects with.
 */
export async function rerunTurn(
  sessions: RerunSessions,
  request: RerunRequest,
): Promise<SessionId | undefined> {
  const child = await prepareRerunSession(sessions, request);
  if (child === undefined) return undefined;
  await sessions.prompt(child, request.text);
  return child;
}

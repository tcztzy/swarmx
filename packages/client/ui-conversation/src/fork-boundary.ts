/**
 * Resolving the fork boundary for a re-run.
 *
 * Both actions in this package restart the conversation from just before a
 * chosen turn: editing stages changed text, retrying sends the original.
 * Neither mutates history — later turns fork a prefix of the source log, while
 * the first turn starts in a fresh sibling session.
 *
 * The host anchors the cut at a completed turn, searching *forward* from the
 * requested seq for the first `turn/end`:
 *
 * ```ts
 * events.find(e => e.type === 'turn/end' && e.seq >= atSeq)
 * ```
 *
 * That forward search is the whole subtlety. Passing the seq of the turn we
 * want to re-run would find that turn's own `turn/end`, keeping the turn we
 * meant to replace. The boundary must be the `turn/end` of the turn *before*
 * it. The first turn has no completed prefix, and DSH cannot express an empty
 * prefix through `session.fork`: an omitted `atSeq` retains the last completed
 * turn. That case must create a fresh sibling instead.
 */

/** The timeline subset this resolution reads, as published on the snapshot. */
export interface Timeline {
  /** Turn indices in ascending order. */
  readonly turnOrder: readonly number[];
  /** Turn index to its location record. */
  readonly turns: ReadonlyMap<number, TimelineTurn>;
}

/** One turn's boundaries. Mirrors the runtime's `TurnLocation`. */
export interface TimelineTurn {
  /** Turn index. */
  readonly turn: number;
  /** Opening event, absent when it is outside the loaded window. */
  readonly start?: { readonly seq: number } | undefined;
  /** Closing event, absent while the turn is open or out of window. */
  readonly end?: { readonly seq: number } | undefined;
  /** Whether the turn closed; `unknown` when the window cannot say. */
  readonly status: "open" | "closed" | "unknown";
}

/**
 * A resolved preparation strategy: fork a completed prefix, or start fresh.
 */
export type ForkBoundary =
  | { readonly kind: "fork"; readonly atSeq: number }
  | { readonly kind: "fresh" };

/**
 * Resolve the boundary that re-runs `targetTurn`.
 *
 * @param timeline - the session's turn timeline.
 * @param targetTurn - index of the turn being re-run.
 * @param windowReachesStart - whether the loaded history reaches the session
 *   start. Older pages load on demand, so a window beginning mid-history
 *   cannot tell "first turn" from "earlier turns not fetched yet", and forking
 *   from the start would discard history the user can still scroll to.
 * @returns the boundary, or undefined when no cut is possible.
 */
export function resolveForkBoundary(
  timeline: Timeline,
  targetTurn: number,
  windowReachesStart: boolean,
): ForkBoundary | undefined {
  const position = timeline.turnOrder.indexOf(targetTurn);
  if (position === -1) return undefined;
  // Walk back to the nearest turn that actually closed. An intervening turn
  // with no usable `turn/end` cannot anchor a cut, but an earlier one still
  // can, and cutting there re-runs the target along with those turns — which
  // is what re-running from before the target means.
  for (let index = position - 1; index >= 0; index -= 1) {
    const turn = timeline.turns.get(timeline.turnOrder[index] as number);
    const end = turn?.end;
    if (turn?.status === "closed" && end !== undefined) {
      return { kind: "fork", atSeq: end.seq };
    }
  }
  return windowReachesStart ? { kind: "fresh" } : undefined;
}

/**
 * Whether `targetTurn` may be re-run at all. The host rejects a prefix ending
 * inside an open turn, so a running turn is excluded here rather than left to
 * fail at the boundary.
 *
 * @param timeline - the session's turn timeline.
 * @param targetTurn - index of the turn being re-run.
 * @param windowReachesStart - as in {@link resolveForkBoundary}.
 * @returns whether an action targeting this turn should be offered.
 */
export function canRerunTurn(
  timeline: Timeline,
  targetTurn: number,
  windowReachesStart: boolean,
): boolean {
  if (timeline.turns.get(targetTurn)?.status === "open") return false;
  return resolveForkBoundary(timeline, targetTurn, windowReachesStart) !== undefined;
}

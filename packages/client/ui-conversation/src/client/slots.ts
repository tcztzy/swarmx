/**
 * Action contract injected into the user-message and failed-turn entries.
 */

/** The business face the strip entry receives, bound per session. */
export interface RerunActionsInjected {
  /** Whether the turn and its opening text are available for a safe re-run. */
  canRerun(turn: number): boolean;
  /** Prepare before a failed turn and re-send its original text. */
  rerun(turn: number): Promise<void>;
  /** Prepare before a user message and load its text for revision. */
  beginEdit(turn: number, text: string): Promise<void>;
}

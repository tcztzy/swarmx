/** Terminal failed-turn selection for the conversation turn-tail extension. */

import type { TurnTailOwnerProps } from "@deepseek-ai/dsh-client-ui-chat/client";

/** Failure data rendered beside the retry icon. */
export interface FailedTurn {
  readonly turn: number;
  readonly message: string;
  readonly code?: string;
}

/** Convert a durable failure into the same display-safe copy used by DSH. */
function failureMessage(failure: unknown): string {
  if (failure === null || typeof failure !== "object") return String(failure);
  const record = failure as Record<string, unknown>;
  if (record.code === "AUTH") return "API key is invalid";
  return typeof record.message === "string" ? record.message : JSON.stringify(failure);
}

/** Select only turns whose durable end reason is an error. */
export function selectFailedTurn(owner: TurnTailOwnerProps): FailedTurn | null {
  const reason = owner.turn.end?.data.reason;
  if (reason?.kind !== "error") return null;
  const code = "code" in reason.error ? reason.error.code : undefined;
  return {
    turn: owner.turn.turn,
    message: failureMessage(reason.error),
    ...(typeof code === "string" ? { code } : {}),
  };
}

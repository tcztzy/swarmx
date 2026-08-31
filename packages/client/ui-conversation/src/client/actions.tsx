/** Icon actions for user-authored messages and terminal failed turns. */

import type { ChatSnapshot } from "@deepseek-ai/dsh-client-ui-chat/client";
import { Tooltip } from "@deepseek-ai/dsh-client-ui-primitives";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import { type ReactNode, useCallback, useState } from "react";
import type { FailedTurn } from "../error-turn.js";
import css from "./actions.module.css";
import { conversationIcons } from "./icons.js";
import type { RerunActionsInjected } from "./slots.js";

const { edit: EditIcon, retry: RetryIcon } = conversationIcons;

interface AsyncIconActionProps {
  readonly label: string;
  readonly busyLabel: string;
  readonly run: () => Promise<void>;
  readonly children: ReactNode;
}

/** DSH-sized icon button with bounded busy and failure feedback. */
function AsyncIconAction({ label, busyLabel, run, children }: AsyncIconActionProps) {
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  const onClick = useCallback(() => {
    setBusy(true);
    setFailed(false);
    void run().then(
      () => setBusy(false),
      () => {
        setBusy(false);
        setFailed(true);
      },
    );
  }, [run]);
  const currentLabel = busy ? busyLabel : failed ? `${label} failed` : label;
  return (
    <Tooltip label={currentLabel} side="bottom">
      <button
        type="button"
        className={css.action}
        aria-label={currentLabel}
        disabled={busy}
        data-failed={failed || undefined}
        onClick={onClick}
      >
        {children}
      </button>
    </Tooltip>
  );
}

/** Props for the Chat node contributed immediately after a user message. */
type UserEditActionProps = PropsRuntime<"conversation.chat.node", "swarmx-user-edit"> &
  RerunActionsInjected;

/** Edit icon visually composed into the user message's existing action row. */
export function UserEditAction({ node, useChat, canRerun, beginEdit }: UserEditActionProps) {
  const { turn, text } = node.data;
  useChat((snapshot: ChatSnapshot) => snapshot.timeline.turns.get(turn)?.status);
  const run = useCallback(() => beginEdit(turn, text), [beginEdit, text, turn]);
  if (!canRerun(turn)) return null;
  return (
    <div className={css.userEditRow} data-user-edit-action>
      <AsyncIconAction label="Edit message" busyLabel="Opening editor" run={run}>
        <EditIcon />
      </AsyncIconAction>
    </div>
  );
}

/** Props for the failed-turn chain entry. */
type FailedTurnActionProps = PropsRuntime<"conversation.chat.turnTail"> &
  RerunActionsInjected & { readonly matched: FailedTurn };

/** Persistent terminal failure plus an icon that retries that exact turn. */
export function FailedTurnAction({ matched, useChat, canRerun, rerun }: FailedTurnActionProps) {
  useChat((snapshot: ChatSnapshot) => snapshot.timeline.turns.get(matched.turn)?.status);
  const run = useCallback(() => rerun(matched.turn), [matched.turn, rerun]);
  return (
    <div className={css.errorRow} role="status" data-failed-turn-action>
      <span className={css.errorDot} aria-hidden="true" />
      <span className={css.errorMessage}>{matched.message}</span>
      {matched.code !== undefined && <code className={css.errorCode}>{matched.code}</code>}
      {canRerun(matched.turn) && (
        <AsyncIconAction label="Retry failed turn" busyLabel="Retrying" run={run}>
          <RetryIcon />
        </AsyncIconAction>
      )}
    </div>
  );
}

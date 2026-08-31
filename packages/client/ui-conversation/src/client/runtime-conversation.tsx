import type { Context } from "@deepseek-ai/cordis";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ApprovalDecision,
  type ApprovalInputValue,
  type ApprovalQuestion,
  type ApprovalRequest,
  approvalAnswerKey,
  approvalIdentityKey,
  approvalResponseFields,
  ConversationRuntimeClient,
  type ConversationSnapshot,
  type ConversationSummary,
  projectApprovalEvent,
  projectRuntimeEvent,
  type RuntimeEvent,
  type RuntimeKind,
  removeApprovalAnswers,
  removeApprovalRequest,
} from "./runtime-client.js";
import css from "./runtime-conversation.module.css";

const SHADOW_PRIORITY = -10;

export interface TerminalTurnAction {
  readonly refreshList: true;
  readonly reloadConversationId?: string;
}

export function terminalTurnAction(
  event: RuntimeEvent,
  selectedConversationId: string | undefined,
): TerminalTurnAction | undefined {
  if (event.type !== "turn_status" || event.status === "running") return undefined;
  return event.conversationId === selectedConversationId
    ? { refreshList: true, reloadConversationId: event.conversationId }
    : { refreshList: true };
}

export class ConversationOpenFence {
  private generation = 0;
  private controller: AbortController | undefined;
  private committedId: string | undefined;
  private selectedId: string | undefined;

  get selectedConversationId(): string | undefined {
    return this.selectedId;
  }

  async open<Value>(
    conversationId: string,
    read: (signal: AbortSignal) => Promise<Value>,
    apply: (value: Value) => void,
  ): Promise<boolean> {
    const generation = ++this.generation;
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    this.selectedId = conversationId;
    try {
      const value = await read(controller.signal);
      if (
        controller.signal.aborted ||
        this.controller !== controller ||
        this.generation !== generation
      ) {
        return false;
      }
      apply(value);
      this.committedId = conversationId;
      return true;
    } catch (reason) {
      if (
        controller.signal.aborted ||
        this.controller !== controller ||
        this.generation !== generation
      ) {
        return false;
      }
      this.selectedId = this.committedId;
      throw reason;
    } finally {
      if (this.controller === controller) this.controller = undefined;
    }
  }

  clear(): void {
    this.generation += 1;
    this.controller?.abort();
    this.controller = undefined;
    this.committedId = undefined;
    this.selectedId = undefined;
  }
}

export class ConversationListFence {
  private generation = 0;
  private controller: AbortController | undefined;

  async refresh<Value>(
    load: (signal: AbortSignal) => Promise<Value>,
    apply: (value: Value) => void,
  ): Promise<Value | undefined> {
    const generation = ++this.generation;
    this.controller?.abort();
    const controller = new AbortController();
    this.controller = controller;
    try {
      const value = await load(controller.signal);
      if (
        controller.signal.aborted ||
        this.controller !== controller ||
        this.generation !== generation
      ) {
        return undefined;
      }
      apply(value);
      return value;
    } catch (reason) {
      if (
        controller.signal.aborted ||
        this.controller !== controller ||
        this.generation !== generation
      ) {
        return undefined;
      }
      throw reason;
    } finally {
      if (this.controller === controller) this.controller = undefined;
    }
  }

  clear(): void {
    this.generation += 1;
    this.controller?.abort();
    this.controller = undefined;
  }
}

export function initialConversationId(
  conversations: readonly ConversationSummary[] | undefined,
  selectedConversationId: string | undefined,
): string | undefined {
  return selectedConversationId === undefined ? conversations?.[0]?.conversationId : undefined;
}

export function runtimeControlState(
  busy: boolean,
  hasActiveTurn: boolean,
): { readonly archiveDisabled: boolean; readonly selectConversationDisabled: boolean } {
  return {
    archiveDisabled: busy || hasActiveTurn,
    selectConversationDisabled: busy,
  };
}

export class PendingOperationCounter {
  private count = 0;

  get busy(): boolean {
    return this.count > 0;
  }

  begin(): void {
    this.count += 1;
  }

  finish(): boolean {
    if (this.count === 0) throw new Error("no pending conversation operation to finish");
    this.count -= 1;
    return this.busy;
  }
}

/** Install the direct Conversation-slot boundary only for a non-DSH default runtime. */
export function registerPeerRuntimeConversation(ctx: Context): void {
  ctx.effect(() => {
    const controller = new AbortController();
    const client = new ConversationRuntimeClient();
    let unregister: (() => void) | undefined;
    void client.metadata(controller.signal).then(
      (metadata) => {
        if (controller.signal.aborted || metadata.defaultRuntimeKind === "dsh") return;
        const runtimeKind = metadata.defaultRuntimeKind;
        unregister = ctx.slots.inject("conversation", () =>
          ctx.slots.register(
            { name: "conversation", priority: SHADOW_PRIORITY },
            function PeerConversationEntry() {
              return <RuntimeConversation client={client} runtimeKind={runtimeKind} />;
            },
          ),
        );
      },
      (reason: unknown) => {
        if (controller.signal.aborted) return;
        const message = reason instanceof Error ? reason.message : String(reason);
        unregister = ctx.slots.inject("conversation", () =>
          ctx.slots.register(
            { name: "conversation", priority: SHADOW_PRIORITY },
            function RuntimeMetadataFailure() {
              return (
                <main className={css.root} aria-label="Conversation runtime unavailable">
                  <div className={css.error} role="alert">
                    Conversation runtime metadata failed: {message}
                  </div>
                </main>
              );
            },
          ),
        );
      },
    );
    return () => {
      controller.abort();
      unregister?.();
    };
  }, "swarmx: selected peer conversation");
}

interface RuntimeConversationProps {
  readonly client: ConversationRuntimeClient;
  readonly runtimeKind: RuntimeKind;
}

interface PendingEdit {
  readonly conversationId: string;
  readonly userItemId: string;
  readonly previousDraft: string;
}

function RuntimeConversation({ client, runtimeKind }: RuntimeConversationProps) {
  const openFence = useRef(new ConversationOpenFence());
  const listFence = useRef(new ConversationListFence());
  const pendingOperations = useRef(new PendingOperationCounter());
  const [conversations, setConversations] = useState<readonly ConversationSummary[]>([]);
  const [snapshot, setSnapshot] = useState<ConversationSnapshot>();
  const [draft, setDraft] = useState("");
  const [editing, setEditing] = useState<PendingEdit>();
  const [approvals, setApprovals] = useState<readonly ApprovalRequest[]>([]);
  const [answers, setAnswers] = useState<Readonly<Record<string, ApprovalInputValue>>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>();

  const refreshList = useCallback(
    () =>
      listFence.current.refresh(
        (signal) => client.list(runtimeKind, signal),
        (next) => setConversations(next),
      ),
    [client, runtimeKind],
  );

  const openConversation = useCallback(
    async (conversationId: string) => {
      await openFence.current.open(
        conversationId,
        (signal) => client.read(runtimeKind, conversationId, signal),
        (next) => {
          setSnapshot(next);
          setApprovals((current) => [
            ...current.filter((approval) => approval.conversationId !== conversationId),
            ...(next.approvals ?? []),
          ]);
        },
      );
    },
    [client, runtimeKind],
  );

  const run = useCallback(async (operation: () => Promise<void>) => {
    pendingOperations.current.begin();
    setBusy(true);
    setError(undefined);
    try {
      await operation();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(pendingOperations.current.finish());
    }
  }, []);

  useEffect(() => {
    let active = true;
    void refreshList().then(
      (items) => {
        if (!active) return;
        const conversationId = initialConversationId(
          items,
          openFence.current.selectedConversationId,
        );
        if (conversationId !== undefined) void openConversation(conversationId);
      },
      (reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      },
    );
    return () => {
      active = false;
      listFence.current.clear();
      openFence.current.clear();
    };
  }, [openConversation, refreshList]);

  useEffect(
    () =>
      client.subscribe(
        (event) => {
          if (event.runtime !== runtimeKind) return;
          if (event.type === "approval_requested") {
            setApprovals((current) => projectApprovalEvent(current, event));
          } else if (event.type === "approval_resolved") {
            setApprovals((current) => projectApprovalEvent(current, event));
            setAnswers((current) => removeApprovalAnswers(current, event));
          } else {
            setSnapshot((current) =>
              current === undefined ? current : projectRuntimeEvent(current, event),
            );
          }
          const action = terminalTurnAction(event, openFence.current.selectedConversationId);
          if (action !== undefined) {
            void refreshList();
            if (action.reloadConversationId !== undefined) {
              void openConversation(action.reloadConversationId);
            }
          }
        },
        () => setError("Conversation event stream disconnected."),
      ),
    [client, openConversation, refreshList, runtimeKind],
  );

  const activeTurn = useMemo(
    () => [...(snapshot?.turns ?? [])].reverse().find((turn) => turn.status === "running"),
    [snapshot],
  );
  const controls = runtimeControlState(busy, activeTurn !== undefined);

  const createConversation = () =>
    run(async () => {
      const created = await client.create(runtimeKind);
      setEditing(undefined);
      setDraft("");
      await refreshList();
      await openConversation(created.conversationId);
    });

  const submit = () =>
    run(async () => {
      const text = draft.trim();
      if (snapshot === undefined || text === "") return;
      if (editing !== undefined) {
        let revised: ConversationSummary;
        try {
          revised = await client.edit(
            runtimeKind,
            editing.conversationId,
            editing.userItemId,
            text,
          );
        } catch (reason) {
          try {
            await openConversation(editing.conversationId);
          } catch {
            // Preserve the revision failure; the draft remains available for deliberate retry.
          }
          throw reason;
        }
        setEditing(undefined);
        setDraft("");
        await refreshList();
        await openConversation(revised.conversationId);
        return;
      }
      if (activeTurn === undefined) {
        await client.start(runtimeKind, snapshot.conversationId, text);
      } else {
        await client.steer(runtimeKind, snapshot.conversationId, activeTurn.id, text);
      }
      setDraft("");
      await openConversation(snapshot.conversationId);
    });

  const selectChild = async (child: ConversationSummary) => {
    setEditing(undefined);
    setDraft("");
    await refreshList();
    await openConversation(child.conversationId);
  };

  const decide = (approval: ApprovalRequest, decision: ApprovalDecision) =>
    run(async () => {
      const values: Record<string, ApprovalInputValue> = {};
      for (const question of approval.questions ?? []) {
        const value = answers[approvalAnswerKey(approval, question.id)];
        if (value !== undefined) values[question.id] = value;
      }
      await client.approve(approval, decision, approvalResponseFields(approval, decision, values));
      setApprovals((current) => removeApprovalRequest(current, approval));
      setAnswers((current) => removeApprovalAnswers(current, approval));
    });

  return (
    <main className={css.root} aria-label={`${runtimeKind} conversations`}>
      <aside className={css.runtimeList}>
        <div className={css.runtimeHeader}>
          <span className={css.runtimeName}>{runtimeKind}</span>
          <button
            type="button"
            className={css.primaryButton}
            disabled={busy}
            onClick={createConversation}
          >
            New
          </button>
        </div>
        <div className={css.conversationList}>
          {conversations.map((conversation) => (
            <button
              type="button"
              className={css.conversationButton}
              data-active={conversation.conversationId === snapshot?.conversationId || undefined}
              disabled={controls.selectConversationDisabled}
              key={conversation.conversationId}
              onClick={() =>
                void run(async () => {
                  setEditing(undefined);
                  setDraft("");
                  await openConversation(conversation.conversationId);
                })
              }
            >
              <span>{conversation.title}</span>
              <small>{conversation.workspace.label}</small>
            </button>
          ))}
        </div>
      </aside>

      <section className={css.conversation}>
        <header className={css.header}>
          <div>
            <strong>{snapshot?.title ?? "Select or create a conversation"}</strong>
            {snapshot !== undefined && <small>{snapshot.workspace.label}</small>}
          </div>
          {snapshot !== undefined && (
            <button
              type="button"
              className={css.secondaryButton}
              disabled={controls.archiveDisabled}
              title={
                runtimeKind === "codex"
                  ? "Codex also archives native branches descended from this conversation."
                  : undefined
              }
              onClick={() =>
                void run(async () => {
                  await client.archive(runtimeKind, snapshot.conversationId);
                  setEditing(undefined);
                  setDraft("");
                  openFence.current.clear();
                  setSnapshot(undefined);
                  await refreshList();
                })
              }
            >
              {runtimeKind === "codex" ? "Archive conversation and branches" : "Archive"}
            </button>
          )}
        </header>

        <div className={css.transcript} aria-live="polite">
          {snapshot?.turns.map((turn) => (
            <article className={css.turn} key={turn.id}>
              <div className={css.turnHeader}>
                <span>{turn.status}</span>
                <button
                  type="button"
                  className={css.textButton}
                  disabled={busy || activeTurn !== undefined}
                  onClick={() =>
                    void run(async () => {
                      const child = await client.fork(
                        runtimeKind,
                        snapshot.conversationId,
                        turn.id,
                      );
                      await selectChild(child);
                    })
                  }
                >
                  Fork here
                </button>
              </div>
              {turn.items.map((item) => (
                <div className={css.item} data-kind={item.type} key={item.id}>
                  <span className={css.itemKind}>{item.type.replaceAll("_", " ")}</span>
                  <p>{itemText(item)}</p>
                  {item.type === "user_message" && (
                    <div className={css.itemActions}>
                      <button
                        type="button"
                        className={css.textButton}
                        disabled={busy || activeTurn !== undefined}
                        onClick={() =>
                          void run(async () => {
                            const child = await client.retry(
                              runtimeKind,
                              snapshot.conversationId,
                              item.id,
                            );
                            await selectChild(child);
                          })
                        }
                      >
                        Retry
                      </button>
                      <button
                        type="button"
                        className={css.textButton}
                        disabled={busy || activeTurn !== undefined}
                        onClick={() => {
                          setEditing({
                            conversationId: snapshot.conversationId,
                            userItemId: item.id,
                            previousDraft: editing?.previousDraft ?? draft,
                          });
                          setDraft(item.text);
                          setError(undefined);
                        }}
                      >
                        Edit
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </article>
          ))}

          {approvals
            .filter((approval) => approval.conversationId === snapshot?.conversationId)
            .map((approval) => (
              <section className={css.approval} key={approvalIdentityKey(approval)}>
                <strong>Approval requested</strong>
                <p>{approval.prompt}</p>
                {(approval.questions ?? []).map((question) => {
                  const key = approvalAnswerKey(approval, question.id);
                  const value = answers[key] ?? question.defaultValue;
                  return (
                    <label htmlFor={key} key={question.id}>
                      <span>{question.header ?? question.prompt}</span>
                      {question.type === "boolean" ? (
                        <input
                          id={key}
                          type="checkbox"
                          checked={value === true}
                          required={question.required}
                          onChange={(event) =>
                            setAnswers((current) => ({
                              ...current,
                              [key]: event.currentTarget.checked,
                            }))
                          }
                        />
                      ) : question.type === "string_array" ? (
                        <select
                          id={key}
                          multiple
                          value={Array.isArray(value) ? value : []}
                          required={question.required}
                          onChange={(event) =>
                            setAnswers((current) => ({
                              ...current,
                              [key]: [...event.currentTarget.selectedOptions].map(
                                (option) => option.value,
                              ),
                            }))
                          }
                        >
                          {(question.options ?? []).map((option) => (
                            <option value={option} key={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : question.options !== undefined ? (
                        <select
                          id={key}
                          value={typeof value === "string" ? value : ""}
                          required={question.required}
                          onChange={(event) =>
                            setAnswers((current) => ({
                              ...current,
                              [key]: event.currentTarget.value,
                            }))
                          }
                        >
                          <option value="" />
                          {question.options.map((option) => (
                            <option value={option} key={option}>
                              {option}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          id={key}
                          type={approvalInputType(question.type, question.format)}
                          value={
                            typeof value === "string" || typeof value === "number" ? value : ""
                          }
                          required={question.required}
                          min={question.minimum}
                          max={question.maximum}
                          minLength={question.minLength}
                          maxLength={question.maxLength}
                          step={question.type === "integer" ? 1 : undefined}
                          onChange={(event) =>
                            setAnswers((current) => ({
                              ...current,
                              [key]: event.currentTarget.value,
                            }))
                          }
                        />
                      )}
                    </label>
                  );
                })}
                <div className={css.approvalActions}>
                  {approval.choices.map((choice) => (
                    <button
                      type="button"
                      className={
                        choice === "accept" || choice === "submit"
                          ? css.primaryButton
                          : css.secondaryButton
                      }
                      disabled={busy}
                      key={choice}
                      onClick={() => void decide(approval, choice)}
                    >
                      {choice.replaceAll("_", " ")}
                    </button>
                  ))}
                </div>
              </section>
            ))}
        </div>

        <footer className={css.composer}>
          {error !== undefined && <div className={css.error}>{error}</div>}
          {editing !== undefined && (
            <div className={css.editNotice}>
              <span>Editing message</span>
              <button
                type="button"
                className={css.textButton}
                disabled={busy}
                onClick={() => {
                  setDraft(editing.previousDraft);
                  setEditing(undefined);
                  setError(undefined);
                }}
              >
                Cancel
              </button>
            </div>
          )}
          <textarea
            aria-label="Message"
            disabled={snapshot === undefined || busy}
            placeholder={
              editing !== undefined
                ? "Edit message"
                : activeTurn === undefined
                  ? "Message"
                  : "Steer the active turn"
            }
            rows={3}
            value={draft}
            onChange={(event) => setDraft(event.currentTarget.value)}
          />
          <div className={css.composerActions}>
            {activeTurn !== undefined && snapshot !== undefined && (
              <button
                type="button"
                className={css.secondaryButton}
                disabled={busy}
                onClick={() =>
                  void run(async () => {
                    await client.interrupt(runtimeKind, snapshot.conversationId, activeTurn.id);
                    await openConversation(snapshot.conversationId);
                  })
                }
              >
                Interrupt
              </button>
            )}
            <button
              type="button"
              className={css.primaryButton}
              disabled={snapshot === undefined || busy || draft.trim() === ""}
              onClick={() => void submit()}
            >
              {editing !== undefined ? "Save edit" : activeTurn === undefined ? "Send" : "Steer"}
            </button>
          </div>
        </footer>
      </section>
    </main>
  );
}

function itemText(item: ConversationSnapshot["turns"][number]["items"][number]): string {
  if (item.type === "tool") return `${item.name}: ${item.summary ?? item.status}`;
  if (item.type === "error") return item.message;
  return item.text;
}

function approvalInputType(
  type: ApprovalQuestion["type"],
  format: ApprovalQuestion["format"],
): "text" | "number" | "date" | "url" | "email" {
  if (type === "number" || type === "integer") return "number";
  if (format === "date") return "date";
  if (format === "uri") return "url";
  if (format === "email") return "email";
  return "text";
}

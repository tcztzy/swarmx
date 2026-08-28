import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SwarmUiSnapshot } from "@swarmx/dsh-swarm/contracts";
import { swarmUiSnapshotSchema } from "@swarmx/dsh-swarm/contracts";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import { useCallback, useSyncExternalStore } from "react";
import type { SwarmActivityState, SwarmActivityStore } from "./activity-store.js";
import { englishSwarmText, type SwarmLocaleKey, type SwarmTranslate } from "./swarm-locales.js";
import css from "./swarm-view.module.css";

export function swarmSideViewEntry(
  snapshot: SwarmUiSnapshot,
  t: SwarmTranslate = englishSwarmText,
): SideViewEntry {
  const parsed = swarmUiSnapshotSchema.parse(snapshot);
  return {
    id: "swarm-activity",
    kind: "swarm-activity",
    title: t("title.activity"),
    mode: "inspect",
    payload: parsed as SideViewEntry["payload"],
  };
}

function enumLabel(t: SwarmTranslate, group: string, value: string): string {
  return t(`${group}.${value}` as SwarmLocaleKey);
}

export function SwarmActivity({
  snapshot,
  t = englishSwarmText,
}: {
  readonly snapshot: SwarmUiSnapshot;
  readonly t?: SwarmTranslate;
}) {
  if (snapshot.kind === "inactive") {
    return <p className={css.empty}>{t("empty.inactive")}</p>;
  }
  return (
    <section className={css.panel} aria-label={t("title.activity")}>
      <header className={css.summary}>
        <div>
          <h3>{snapshot.name}</h3>
          <p>
            {snapshot.kind === "archived" ? t("state.archived") : t("state.active")} ·{" "}
            {t("summary.revision", { revision: snapshot.revision })}
          </p>
        </div>
        <span>
          {t("summary.running", {
            count: snapshot.tasks.filter((task) => task.status === "in_progress").length,
          })}
        </span>
      </header>
      <details className={css.group} open>
        <summary>{t("group.members", { count: snapshot.members.length })}</summary>
        <ul>
          {snapshot.members.map((member) => (
            <li key={member.name}>
              <div>
                <strong>{member.name}</strong>
                <p>
                  {enumLabel(t, "role", member.role)} · {member.modelLabel}
                </p>
                <p>{member.description}</p>
                <p data-budget={member.budgetState}>{enumLabel(t, "budget", member.budgetState)}</p>
              </div>
              <span data-status={member.status}>{enumLabel(t, "status", member.status)}</span>
            </li>
          ))}
        </ul>
      </details>
      <details className={css.group} open>
        <summary>{t("group.tasks", { count: snapshot.tasks.length })}</summary>
        {snapshot.tasks.length === 0 ? (
          <p className={css.empty}>{t("empty.tasks")}</p>
        ) : (
          <ul>
            {snapshot.tasks.map((task) => (
              <li key={task.id}>
                <div>
                  <strong>{task.subject}</strong>
                  <p>
                    {task.id} · {t(`task.${task.kind}`)}
                    {task.ownerName ? ` · ${task.ownerName}` : ""}
                  </p>
                  {task.verifierName && <p>{t("task.verifier", { name: task.verifierName })}</p>}
                  <p data-budget={task.budgetState}>{enumLabel(t, "budget", task.budgetState)}</p>
                  {task.usage && (
                    <p>
                      {t(task.usage.availability === "known" ? "usage.known" : "usage.unknown", {
                        input: task.usage.inputTokens,
                        output: task.usage.outputTokens,
                        turns: task.usage.turns,
                        wall: task.usage.wallMs,
                      })}
                    </p>
                  )}
                  {task.submission && (
                    <details className={css.detail}>
                      <summary>{t("task.submission")}</summary>
                      <p>{task.submission.summary}</p>
                      <p>
                        {t("task.artifacts", {
                          artifacts: task.submission.artifactCount,
                          evidence: task.submission.evidenceCount,
                        })}
                      </p>
                    </details>
                  )}
                  {task.verification && (
                    <details className={css.detail}>
                      <summary>
                        {t("task.verdict", {
                          verdict: enumLabel(t, "verdict", task.verification.verdict),
                        })}
                      </summary>
                      <p>{enumLabel(t, "verification", task.verification.mode)}</p>
                      <p>
                        {t("task.checks", {
                          passed: task.verification.checkResults.filter(
                            (check) => check.status === "pass",
                          ).length,
                          total: task.verification.checkResults.length,
                        })}
                      </p>
                      <p>{task.verification.rationale}</p>
                    </details>
                  )}
                  {task.escalationReason && (
                    <p>{t("task.escalation", { reason: task.escalationReason })}</p>
                  )}
                </div>
                <span data-status={task.status}>{enumLabel(t, "status", task.status)}</span>
              </li>
            ))}
          </ul>
        )}
      </details>
      {snapshot.findings.length > 0 && (
        <details className={css.group}>
          <summary>{t("group.findings", { count: snapshot.findings.length })}</summary>
          <ul>
            {snapshot.findings.slice(-20).map((finding) => (
              <li key={`${finding.code}:${finding.recordedAt}`}>
                <div>
                  <strong>{finding.code}</strong>
                  <p>{finding.summary}</p>
                </div>
                <span data-status={finding.severity}>
                  {enumLabel(t, "severity", finding.severity)}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {snapshot.pendingMessages > 0 && (
        <p className={css.notice}>{t("message.pending", { count: snapshot.pendingMessages })}</p>
      )}
    </section>
  );
}

function useActivity(store: SwarmActivityStore, sessionId: SessionId): SwarmActivityState {
  const subscribe = useCallback(
    (listener: () => void) => store.subscribe(sessionId, listener),
    [sessionId, store],
  );
  const snapshot = useCallback(() => store.getSnapshot(sessionId), [sessionId, store]);
  return useSyncExternalStore(subscribe, snapshot, snapshot);
}

type SwarmHeaderActionProps = PropsRuntime<"conversation.session.header.actions"> & {
  readonly store: SwarmActivityStore;
  readonly open: (snapshot: SwarmUiSnapshot) => void;
  readonly t: SwarmTranslate;
};

export function SwarmHeaderAction({ open, sessionId, store, t }: SwarmHeaderActionProps) {
  const activity = useActivity(store, sessionId);
  if (activity.kind !== "ready" || activity.snapshot.kind === "inactive") return null;
  const running = activity.snapshot.tasks.filter((task) => task.status === "in_progress").length;
  return (
    <button
      type="button"
      className={css.headerAction}
      aria-label={t("action.open")}
      onClick={() => open(activity.snapshot)}
    >
      {t("action.team")}
      {running > 0 ? ` ${running}` : ""}
    </button>
  );
}

type SwarmActivityViewProps = SideViewContentOwnerProps & {
  readonly sessionId: SessionId;
  readonly store: SwarmActivityStore;
  readonly t: SwarmTranslate;
};

export function SwarmActivityView({ entry, sessionId, store, t }: SwarmActivityViewProps) {
  const activity = useActivity(store, sessionId);
  if (activity.kind === "error") {
    return (
      <p className={css.error} role="alert">
        {activity.error.message}
      </p>
    );
  }
  if (activity.kind === "ready") return <SwarmActivity snapshot={activity.snapshot} t={t} />;
  const initial = swarmUiSnapshotSchema.safeParse(entry.payload);
  return initial.success ? (
    <SwarmActivity snapshot={initial.data} t={t} />
  ) : (
    <p className={css.empty}>{t("empty.loading")}</p>
  );
}

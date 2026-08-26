import type { SessionId } from "@deepseek-ai/dsh-client-runtime/client";
import type { PropsRuntime } from "@deepseek-ai/dsh-client-ui-slots";
import type { SwarmUiSnapshot } from "@swarmx/dsh-swarm/contracts";
import { swarmUiSnapshotSchema } from "@swarmx/dsh-swarm/contracts";
import type { SideViewContentOwnerProps, SideViewEntry } from "@swarmx/dsh-ui-conversation/client";
import { useCallback, useSyncExternalStore } from "react";
import type { SwarmActivityState, SwarmActivityStore } from "./activity-store.js";
import css from "./swarm-view.module.css";

export function swarmSideViewEntry(snapshot: SwarmUiSnapshot): SideViewEntry {
  const parsed = swarmUiSnapshotSchema.parse(snapshot);
  return {
    id: "swarm-activity",
    kind: "swarm-activity",
    title: "团队活动",
    mode: "inspect",
    payload: parsed as SideViewEntry["payload"],
  };
}

function statusLabel(status: string): string {
  const labels: Record<string, string> = {
    cancelled: "已取消",
    completed: "已完成",
    failed: "失败",
    idle: "空闲",
    inactive: "离线",
    in_progress: "进行中",
    needs_attention: "需确认",
    pending: "待处理",
    provisioning: "启动中",
    retired: "已退出",
    running: "运行中",
  };
  return labels[status] ?? status;
}

export function SwarmActivity({ snapshot }: { readonly snapshot: SwarmUiSnapshot }) {
  if (snapshot.kind === "inactive") {
    return <p className={css.empty}>当前会话尚未加入团队。</p>;
  }
  return (
    <section className={css.panel} aria-label="团队活动">
      <header className={css.summary}>
        <div>
          <h3>{snapshot.name}</h3>
          <p>
            {snapshot.kind === "archived" ? "已归档" : "活动中"} · 修订 {snapshot.revision}
          </p>
        </div>
        <span>
          {snapshot.tasks.filter((task) => task.status === "in_progress").length} 个进行中
        </span>
      </header>
      <details className={css.group} open>
        <summary>成员 ({snapshot.members.length})</summary>
        <ul>
          {snapshot.members.map((member) => (
            <li key={member.name}>
              <div>
                <strong>{member.name}</strong>
                <p>{member.description}</p>
              </div>
              <span data-status={member.status}>{statusLabel(member.status)}</span>
            </li>
          ))}
        </ul>
      </details>
      <details className={css.group} open>
        <summary>任务 ({snapshot.tasks.length})</summary>
        {snapshot.tasks.length === 0 ? (
          <p className={css.empty}>暂无任务。</p>
        ) : (
          <ul>
            {snapshot.tasks.map((task) => (
              <li key={task.id}>
                <div>
                  <strong>{task.subject}</strong>
                  <p>
                    {task.id} · {task.kind === "write" ? "写任务" : "读任务"}
                    {task.ownerName ? ` · ${task.ownerName}` : ""}
                  </p>
                </div>
                <span data-status={task.status}>{statusLabel(task.status)}</span>
              </li>
            ))}
          </ul>
        )}
      </details>
      {snapshot.pendingMessages > 0 && (
        <p className={css.notice}>有 {snapshot.pendingMessages} 条消息等待投递。</p>
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
};

export function SwarmHeaderAction({ open, sessionId, store }: SwarmHeaderActionProps) {
  const activity = useActivity(store, sessionId);
  if (activity.kind !== "ready" || activity.snapshot.kind === "inactive") return null;
  const running = activity.snapshot.tasks.filter((task) => task.status === "in_progress").length;
  return (
    <button
      type="button"
      className={css.headerAction}
      aria-label="打开团队活动"
      onClick={() => open(activity.snapshot)}
    >
      团队{running > 0 ? ` ${running}` : ""}
    </button>
  );
}

type SwarmActivityViewProps = SideViewContentOwnerProps & {
  readonly sessionId: SessionId;
  readonly store: SwarmActivityStore;
};

export function SwarmActivityView({ entry, sessionId, store }: SwarmActivityViewProps) {
  const activity = useActivity(store, sessionId);
  if (activity.kind === "error") {
    return (
      <p className={css.error} role="alert">
        {activity.error.message}
      </p>
    );
  }
  if (activity.kind === "ready") return <SwarmActivity snapshot={activity.snapshot} />;
  const initial = swarmUiSnapshotSchema.safeParse(entry.payload);
  return initial.success ? (
    <SwarmActivity snapshot={initial.data} />
  ) : (
    <p className={css.empty}>正在读取团队活动…</p>
  );
}

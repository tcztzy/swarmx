import {
  SWARM_PROVISIONING_INTERRUPTED_ERROR,
  SwarmJournal,
  type SwarmMember,
  type SwarmTeamState,
} from "@swarmx/dsh-swarm";
import type { ConversationRuntime, WorkspaceScope } from "../contracts.js";
import { CodexRpcError } from "./connection.js";
import { type CodexMemberBinding, CodexMemberBindingStore } from "./member-bindings.js";

export const CODEX_PROVISIONING_INTERRUPTED_ERROR = SWARM_PROVISIONING_INTERRUPTED_ERROR;

const CODEX_TEAM_PREFIXES = ["codex-mcp-thread:", "codex-mcp-session:"] as const;

export interface ReconcileCodexSwarmBindingsOptions {
  journalRoot: string;
  runtime: Pick<ConversationRuntime, "archive" | "read" | "retireProvisionedMember">;
  workspace: WorkspaceScope;
  signal?: AbortSignal;
}

export interface CodexSwarmBindingRecoveryResult {
  archived: number;
  failed: number;
  resumed: number;
}

/** Reconcile native Codex member claims once at the root runtime startup boundary. */
export async function reconcileCodexSwarmBindings(
  options: ReconcileCodexSwarmBindingsOptions,
): Promise<CodexSwarmBindingRecoveryResult> {
  const journal = new SwarmJournal(options.journalRoot);
  let archived = 0;
  let failed = 0;
  let resumed = 0;
  try {
    const workspaceKey = journal.workspaceKey(options.workspace.root);
    const bindings = new CodexMemberBindingStore(journal, workspaceKey);
    const signal = options.signal ?? new AbortController().signal;
    const activeBindingIds = new Set<string>();
    for (const team of journal.list()) {
      if (
        team.phase !== "active" ||
        team.workspaceKey !== workspaceKey ||
        !CODEX_TEAM_PREFIXES.some((prefix) => team.id.startsWith(prefix))
      ) {
        continue;
      }
      if (team.archiveStartedAt !== undefined) {
        await reconcileArchivingTeam(journal, options, bindings, team, signal);
        continue;
      }
      for (const member of team.members) {
        if (member.role === "lead") continue;
        const binding = bindings.get(member.id);
        const interrupted =
          member.phase === "provisioning" ||
          (member.phase === "failed" && member.error === SWARM_PROVISIONING_INTERRUPTED_ERROR);
        if (interrupted) {
          if (binding === undefined) {
            appendFailedMember(
              journal,
              team.id,
              member,
              "Native Codex Thread binding is unavailable after restart",
            );
            failed += 1;
            continue;
          }
          const snapshot = await readBinding(options.runtime, binding, signal);
          if (snapshot === undefined || snapshot.archived || snapshot.turns.length === 0) {
            appendFailedMember(
              journal,
              team.id,
              member,
              "Native Codex Thread did not complete initial member materialization",
            );
            await cleanupBinding(
              options.runtime,
              options.workspace,
              bindings,
              binding,
              signal,
              snapshot ?? null,
            );
            archived += 1;
            failed += 1;
            continue;
          }
          assertBindingWorkspace(options.workspace, binding, snapshot);
          const { error: _error, ...rest } = member;
          journal.append(team.id, {
            type: "member/updated",
            data: { ...rest, phase: "active" },
          });
          activeBindingIds.add(member.id);
          resumed += 1;
          continue;
        }
        if (member.phase === "active") {
          if (binding === undefined) {
            appendFailedMember(
              journal,
              team.id,
              member,
              "Native Codex Thread binding is unavailable after restart",
            );
            failed += 1;
          } else {
            const snapshot = await readBinding(options.runtime, binding, signal);
            if (snapshot === undefined || snapshot.archived || snapshot.turns.length === 0) {
              appendFailedMember(
                journal,
                team.id,
                member,
                "Native Codex Thread is unavailable or incomplete",
              );
              await cleanupBinding(
                options.runtime,
                options.workspace,
                bindings,
                binding,
                signal,
                snapshot ?? null,
              );
              archived += 1;
              failed += 1;
            } else {
              assertBindingWorkspace(options.workspace, binding, snapshot);
              activeBindingIds.add(member.id);
            }
          }
        }
      }
    }

    for (const binding of bindings.list()) {
      if (activeBindingIds.has(binding.id)) continue;
      await cleanupBinding(options.runtime, options.workspace, bindings, binding, signal);
      archived += 1;
    }
    return { archived, failed, resumed };
  } finally {
    journal.close();
  }
}

async function reconcileArchivingTeam(
  journal: SwarmJournal,
  options: ReconcileCodexSwarmBindingsOptions,
  bindings: CodexMemberBindingStore,
  team: SwarmTeamState,
  signal: AbortSignal,
): Promise<void> {
  journal.settleArchiveIntents(team.id, Date.now());
  for (const member of team.members.filter((candidate) => candidate.role !== "lead")) {
    const binding = bindings.get(member.id);
    if (binding === undefined) {
      if (member.phase === "provisioning") {
        journal.settleProvisioningMemberWithoutBinding(team.id, member.id);
      } else {
        journal.retireMemberForArchive(team.id, member.id);
      }
      continue;
    }
    const snapshot = await readBinding(options.runtime, binding, signal);
    if (snapshot !== undefined) {
      assertBindingWorkspace(options.workspace, binding, snapshot);
      await retireNativeMember(options.runtime, binding, signal, snapshot);
    }
    if (!bindings.retireForArchive(team.id, binding)) {
      throw new Error(`Codex Swarm archive claim became stale for member "${member.name}".`);
    }
  }
  journal.finishArchive(team.id, Date.now());
}

function appendFailedMember(
  journal: SwarmJournal,
  teamId: string,
  member: SwarmMember,
  error: string,
): void {
  if (member.phase === "failed" && member.error === error) return;
  journal.append(teamId, {
    type: "member/updated",
    data: { ...member, error, phase: "failed" },
  });
}

async function readBinding(
  runtime: Pick<ConversationRuntime, "read">,
  binding: CodexMemberBinding,
  signal: AbortSignal,
): Promise<Awaited<ReturnType<ConversationRuntime["read"]>> | undefined> {
  signal.throwIfAborted();
  try {
    const snapshot = await runtime.read(binding.conversationId, signal);
    signal.throwIfAborted();
    return snapshot;
  } catch (error) {
    signal.throwIfAborted();
    if (isMissingNativeThread(error, binding)) return undefined;
    throw error;
  }
}

function assertBindingWorkspace(
  workspace: WorkspaceScope,
  binding: CodexMemberBinding,
  snapshot: Awaited<ReturnType<ConversationRuntime["read"]>>,
): void {
  if (
    snapshot.conversationId !== binding.conversationId ||
    snapshot.workspace.id !== workspace.id
  ) {
    throw new Error(`Codex Swarm orphan "${binding.conversationId}" belongs to another workspace.`);
  }
}

async function cleanupBinding(
  runtime: Pick<ConversationRuntime, "archive" | "read" | "retireProvisionedMember">,
  workspace: WorkspaceScope,
  bindings: CodexMemberBindingStore,
  binding: CodexMemberBinding,
  signal: AbortSignal,
  observed?: Awaited<ReturnType<ConversationRuntime["read"]>> | null,
): Promise<void> {
  const snapshot = observed === undefined ? await readBinding(runtime, binding, signal) : observed;
  if (snapshot !== undefined && snapshot !== null) {
    assertBindingWorkspace(workspace, binding, snapshot);
    await retireNativeMember(runtime, binding, signal, snapshot);
  }
  bindings.release(binding);
}

async function retireNativeMember(
  runtime: Pick<ConversationRuntime, "archive" | "retireProvisionedMember">,
  binding: CodexMemberBinding,
  signal: AbortSignal,
  snapshot: Awaited<ReturnType<ConversationRuntime["read"]>>,
): Promise<void> {
  if (snapshot.archived) return;
  if (runtime.retireProvisionedMember !== undefined) {
    await runtime.retireProvisionedMember(binding.conversationId, binding.id, signal);
    return;
  }
  await runtime.archive(binding.conversationId, signal);
}

function isMissingNativeThread(error: unknown, binding: CodexMemberBinding): boolean {
  if (!(error instanceof CodexRpcError)) return false;
  const nativeThreadId = binding.conversationId.replace(/^codex:/u, "");
  return (
    error.code === -32600 &&
    error.data === undefined &&
    error.message === `thread not loaded: ${nativeThreadId}`
  );
}

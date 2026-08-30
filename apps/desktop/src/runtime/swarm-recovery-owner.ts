import { type ProvisioningMemberBindingClaim, SwarmJournal } from "@swarmx/dsh-swarm";

export class SwarmRecoveryClaimConflictError extends Error {
  constructor(
    message: string,
    readonly kind: "handle" | "member",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "SwarmRecoveryClaimConflictError";
  }
}

export interface SwarmRecoveryOwner {
  claimCodexMember(input: {
    workspaceRoot: string;
    teamId: string;
    memberId: string;
    conversationId: string;
  }): ProvisioningMemberBindingClaim;
  settleCodexMemberArchive(input: {
    workspaceRoot: string;
    teamId: string;
    memberId: string;
    conversationId: string;
  }): boolean;
  settleCodexMemberCreationFailure(input: {
    workspaceRoot: string;
    teamId: string;
    memberId: string;
  }): boolean;
  dispose(): Promise<void>;
}

/** Own the one cold/final recovery epoch for the complete desktop platform lifetime. */
export function startSwarmRecoveryOwner(journalRoot: string): SwarmRecoveryOwner {
  const journal = new SwarmJournal(journalRoot);
  let closed = false;
  try {
    recover(journal);
  } catch (error) {
    journal.close();
    throw error;
  }
  return {
    claimCodexMember(input) {
      if (closed) throw new Error("Swarm recovery owner is closed.");
      try {
        return journal.claimProvisioningMemberBinding(input.teamId, {
          workspaceKey: journal.workspaceKey(input.workspaceRoot),
          runtime: "codex",
          memberId: input.memberId,
          handle: input.conversationId,
        });
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : String(cause);
        if (message.includes("member already belongs")) {
          throw new SwarmRecoveryClaimConflictError(
            "Codex Swarm member already belongs to another native Thread.",
            "member",
            { cause },
          );
        }
        if (message.includes("runtime handle already belongs")) {
          throw new SwarmRecoveryClaimConflictError(
            "Codex Swarm native Thread already belongs to another member.",
            "handle",
            { cause },
          );
        }
        throw cause;
      }
    },
    settleCodexMemberArchive(input) {
      if (closed) throw new Error("Swarm recovery owner is closed.");
      return journal.retireBoundMemberForArchive(input.teamId, {
        workspaceKey: journal.workspaceKey(input.workspaceRoot),
        runtime: "codex",
        memberId: input.memberId,
        handle: input.conversationId,
      });
    },
    settleCodexMemberCreationFailure(input) {
      if (closed) throw new Error("Swarm recovery owner is closed.");
      const team = journal.get(input.teamId);
      if (team?.workspaceKey !== journal.workspaceKey(input.workspaceRoot)) return false;
      return journal.settleProvisioningMemberWithoutBinding(input.teamId, input.memberId);
    },
    async dispose() {
      if (closed) return;
      closed = true;
      try {
        recover(journal);
      } finally {
        journal.close();
      }
    },
  };
}

function recover(journal: SwarmJournal): void {
  const now = Date.now();
  journal.recoverUncertainIntents(now);
  journal.recoverInterruptedTasks(now);
  journal.recoverProvisioningMembers();
}

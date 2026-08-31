import type { ProvisioningMemberBindingClaim, SwarmJournal } from "@swarmx/dsh-swarm";
import { z } from "zod";

const WorkspaceKeySchema = z.string().regex(/^swarmx--[0-9a-f]{64}$/u);
const CodexMemberBindingSchema = z.strictObject({
  id: z.string().uuid(),
  conversationId: z.string().min(1).max(2_048),
});

export type CodexMemberBinding = z.infer<typeof CodexMemberBindingSchema>;

export class CodexMemberBindingConflictError extends Error {
  constructor(
    message: string,
    readonly kind: "handle" | "member",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "CodexMemberBindingConflictError";
  }
}

/** Exact transactional claims over the shared Swarm journal's runtime-handle table. */
export class CodexMemberBindingStore {
  private readonly workspaceKey: string;

  constructor(
    private readonly journal: SwarmJournal,
    workspaceKey: string,
  ) {
    this.workspaceKey = WorkspaceKeySchema.parse(workspaceKey);
    this.list();
  }

  list(): CodexMemberBinding[] {
    try {
      return this.journal.listMemberBindings(this.workspaceKey, "codex").map((binding) =>
        CodexMemberBindingSchema.parse({
          id: binding.memberId,
          conversationId: binding.handle,
        }),
      );
    } catch (cause) {
      throw new Error("Codex Swarm member binding store is invalid.", { cause });
    }
  }

  get(id: string): CodexMemberBinding | undefined {
    const memberId = z.string().uuid().parse(id);
    return this.list().find((binding) => binding.id === memberId);
  }

  findByConversation(conversationId: string): CodexMemberBinding | undefined {
    const nativeThread = z.string().min(1).max(2_048).parse(conversationId);
    return this.list().find((binding) => binding.conversationId === nativeThread);
  }

  claim(input: CodexMemberBinding): "created" | "existing" {
    const binding = CodexMemberBindingSchema.parse(input);
    try {
      return this.journal.claimMemberBinding({
        workspaceKey: this.workspaceKey,
        runtime: "codex",
        memberId: binding.id,
        handle: binding.conversationId,
      });
    } catch (cause) {
      throw bindingConflict(cause);
    }
  }

  claimProvisioning(teamId: string, input: CodexMemberBinding): ProvisioningMemberBindingClaim {
    const binding = CodexMemberBindingSchema.parse(input);
    try {
      return this.journal.claimProvisioningMemberBinding(teamId, {
        workspaceKey: this.workspaceKey,
        runtime: "codex",
        memberId: binding.id,
        handle: binding.conversationId,
      });
    } catch (cause) {
      throw bindingConflict(cause);
    }
  }

  retireForArchive(teamId: string, input: CodexMemberBinding): boolean {
    const binding = CodexMemberBindingSchema.parse(input);
    return this.journal.retireBoundMemberForArchive(teamId, {
      workspaceKey: this.workspaceKey,
      runtime: "codex",
      memberId: binding.id,
      handle: binding.conversationId,
    });
  }

  release(input: CodexMemberBinding): boolean {
    const binding = CodexMemberBindingSchema.parse(input);
    return this.journal.releaseMemberBinding({
      workspaceKey: this.workspaceKey,
      runtime: "codex",
      memberId: binding.id,
      handle: binding.conversationId,
    });
  }
}

function bindingConflict(cause: unknown): unknown {
  const message = cause instanceof Error ? cause.message : String(cause);
  if (message.includes("member already belongs")) {
    return new CodexMemberBindingConflictError(
      "Codex Swarm member already belongs to another native Thread.",
      "member",
      { cause },
    );
  }
  if (message.includes("runtime handle already belongs")) {
    return new CodexMemberBindingConflictError(
      "Codex Swarm native Thread already belongs to another member.",
      "handle",
      { cause },
    );
  }
  return cause;
}

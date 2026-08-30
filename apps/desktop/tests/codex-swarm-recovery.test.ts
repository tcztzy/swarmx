import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CodexRpcError } from "../src/runtime/codex/connection.js";
import { CodexMemberBindingStore } from "../src/runtime/codex/member-bindings.js";
import {
  CODEX_PROVISIONING_INTERRUPTED_ERROR,
  reconcileCodexSwarmBindings,
} from "../src/runtime/codex/swarm-recovery.js";
import type { ConversationSnapshot, WorkspaceScope } from "../src/runtime/contracts.js";
import { WorkspaceAuthority } from "../src/runtime/workspace.js";

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function appendMember(
  journal: SwarmJournal,
  teamId: string,
  input: {
    id: string;
    name: string;
    phase: "active" | "failed" | "provisioning";
    error?: string;
  },
) {
  journal.append(teamId, {
    type: "member/updated",
    data: {
      createdAt: 2,
      description: input.name,
      id: input.id,
      name: input.name,
      phase: input.phase,
      role: "legacy",
      modelPolicy: { source: "legacy-default" },
      ...(input.error === undefined ? {} : { error: input.error }),
    },
  });
}

function createCodexTeam(journal: SwarmJournal, workspace: WorkspaceScope): string {
  const teamId = `codex-mcp-thread:${"c".repeat(64)}`;
  journal.append(teamId, {
    type: "team/created",
    data: {
      createdAt: 1,
      lead: {
        createdAt: 1,
        description: "Codex lead",
        id: teamId,
        name: "lead",
        phase: "active",
        role: "lead",
        modelPolicy: { source: "observed" },
      },
      name: "Recovered Codex team",
      workspaceKey: journal.workspaceKey(workspace.root),
    },
  });
  return teamId;
}

function snapshot(
  workspace: WorkspaceScope,
  conversationId: string,
  options: { archived?: boolean; materialized?: boolean } = {},
): ConversationSnapshot {
  return {
    runtime: "codex",
    conversationId,
    workspace: { id: workspace.id, label: workspace.label },
    title: conversationId,
    archived: options.archived ?? false,
    updatedAt: 1,
    turns:
      options.materialized === false
        ? []
        : [
            {
              id: `${conversationId}:turn`,
              status: "completed",
              items: [
                {
                  id: `${conversationId}:prompt`,
                  turnId: `${conversationId}:turn`,
                  type: "user_message",
                  text: "Initial member prompt",
                  createdAt: 1,
                },
              ],
            },
          ],
  };
}

describe("root Codex Swarm binding recovery", () => {
  it("resumes an interrupted bound member and fails only a proven missing Codex binding", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-root-recovery-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = createCodexTeam(journal, workspace);
    const resumableId = randomUUID();
    const missingId = randomUUID();
    appendMember(journal, teamId, {
      id: resumableId,
      name: "resumable",
      phase: "failed",
      error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
    });
    appendMember(journal, teamId, { id: missingId, name: "missing", phase: "active" });
    const bindings = new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root));
    bindings.claim({ id: resumableId, conversationId: "codex:resumable-thread" });
    journal.close();
    const runtime = {
      read: vi.fn(async (conversationId: string) => snapshot(workspace, conversationId)),
      archive: vi.fn(async () => undefined),
    };

    await expect(reconcileCodexSwarmBindings({ journalRoot, runtime, workspace })).resolves.toEqual(
      { archived: 0, failed: 1, resumed: 1 },
    );

    const recovered = new SwarmJournal(journalRoot);
    expect(recovered.get(teamId)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: resumableId, phase: "active" }),
        expect.objectContaining({ id: missingId, phase: "failed" }),
      ]),
    );
    expect(
      new CodexMemberBindingStore(recovered, recovered.workspaceKey(workspace.root)).get(
        resumableId,
      ),
    ).toEqual({ id: resumableId, conversationId: "codex:resumable-thread" });
    recovered.close();
    expect(runtime.read).toHaveBeenCalledOnce();
    expect(runtime.archive).not.toHaveBeenCalled();
  });

  it("archives and conditionally releases stale native bindings once at the root boundary", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-stale-binding-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    createCodexTeam(journal, workspace);
    const staleId = randomUUID();
    const store = new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root));
    store.claim({ id: staleId, conversationId: "codex:orphan-thread" });
    journal.close();
    const runtime = {
      read: vi.fn(async (conversationId: string) => snapshot(workspace, conversationId)),
      archive: vi.fn(async () => undefined),
    };

    await expect(reconcileCodexSwarmBindings({ journalRoot, runtime, workspace })).resolves.toEqual(
      { archived: 1, failed: 0, resumed: 0 },
    );
    expect(runtime.archive).toHaveBeenCalledWith("codex:orphan-thread", expect.anything());
    const reopened = new SwarmJournal(journalRoot);
    expect(
      new CodexMemberBindingStore(reopened, reopened.workspaceKey(workspace.root)).list(),
    ).toEqual([]);
    reopened.close();
  });

  it.each([
    {
      label: "the RPC method is unavailable",
      failure: new CodexRpcError("thread/read method not found", -32601),
    },
    {
      label: "the read backend is unavailable",
      failure: new CodexRpcError("Thread read backend does not exist", -32000),
    },
    {
      label: "an untyped error merely claims that the Thread is missing",
      failure: new Error("Codex Thread not found"),
    },
  ])("retains an interrupted claim when $label", async ({ failure }) => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-recovery-rpc-error-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = createCodexTeam(journal, workspace);
    const memberId = randomUUID();
    appendMember(journal, teamId, {
      id: memberId,
      name: "interrupted",
      phase: "failed",
      error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
    });
    const bindings = new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root));
    bindings.claim({ id: memberId, conversationId: "codex:unread-thread" });
    journal.close();
    const runtime = {
      read: vi.fn(async () => {
        throw failure;
      }),
      archive: vi.fn(async () => undefined),
    };

    await expect(reconcileCodexSwarmBindings({ journalRoot, runtime, workspace })).rejects.toThrow(
      failure.message,
    );

    const retained = new SwarmJournal(journalRoot, { mode: "client" });
    expect(retained.get(teamId)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: memberId,
          phase: "failed",
          error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
        }),
      ]),
    );
    expect(
      new CodexMemberBindingStore(retained, retained.workspaceKey(workspace.root)).get(memberId),
    ).toEqual({ id: memberId, conversationId: "codex:unread-thread" });
    retained.close();
    expect(runtime.archive).not.toHaveBeenCalled();
  });

  it("never turns cancellation text into proof that a native Thread is missing", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-recovery-abort-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = createCodexTeam(journal, workspace);
    const memberId = randomUUID();
    appendMember(journal, teamId, {
      id: memberId,
      name: "interrupted",
      phase: "failed",
      error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
    });
    const bindings = new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root));
    bindings.claim({ id: memberId, conversationId: "codex:cancelled-read" });
    journal.close();
    const runtime = {
      read: vi.fn(async () => snapshot(workspace, "codex:cancelled-read")),
      archive: vi.fn(async () => undefined),
    };
    const controller = new AbortController();
    controller.abort(new Error("Codex Thread not found"));

    await expect(
      reconcileCodexSwarmBindings({
        journalRoot,
        runtime,
        workspace,
        signal: controller.signal,
      }),
    ).rejects.toThrow("Codex Thread not found");

    const retained = new SwarmJournal(journalRoot, { mode: "client" });
    expect(retained.get(teamId)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: memberId,
          phase: "failed",
          error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
        }),
      ]),
    );
    expect(
      new CodexMemberBindingStore(retained, retained.workspaceKey(workspace.root)).get(memberId),
    ).toEqual({ id: memberId, conversationId: "codex:cancelled-read" });
    retained.close();
    expect(runtime.read).not.toHaveBeenCalled();
    expect(runtime.archive).not.toHaveBeenCalled();
  });

  it("finishes an interrupted archive after cold restart removes an unmaterialized Thread", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-archive-recovery-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = createCodexTeam(journal, workspace);
    const memberId = randomUUID();
    appendMember(journal, teamId, { id: memberId, name: "unbound", phase: "provisioning" });
    journal.beginArchive(teamId, 10);
    journal.close();
    const runtime = {
      read: vi.fn(async (conversationId: string) => snapshot(workspace, conversationId)),
      archive: vi.fn(async () => undefined),
    };

    await expect(reconcileCodexSwarmBindings({ journalRoot, runtime, workspace })).resolves.toEqual(
      { archived: 0, failed: 0, resumed: 0 },
    );
    const recovered = new SwarmJournal(journalRoot, { mode: "client" });
    expect(recovered.get(teamId)).toMatchObject({
      phase: "archived",
      members: expect.arrayContaining([
        expect.objectContaining({ id: memberId, phase: "retired" }),
      ]),
    });
    recovered.close();
    expect(runtime.read).not.toHaveBeenCalled();
    expect(runtime.archive).not.toHaveBeenCalled();
  });

  it("fails and cleans archived or empty claimed Threads instead of fabricating activation", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-codex-invalid-recovery-"));
    roots.push(root);
    const workspace = new WorkspaceAuthority().mint(root);
    const journalRoot = join(root, "swarm");
    const journal = new SwarmJournal(journalRoot);
    const teamId = createCodexTeam(journal, workspace);
    const archivedId = randomUUID();
    const emptyId = randomUUID();
    const missingId = randomUUID();
    appendMember(journal, teamId, { id: archivedId, name: "archived", phase: "active" });
    appendMember(journal, teamId, {
      id: emptyId,
      name: "empty",
      phase: "failed",
      error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
    });
    appendMember(journal, teamId, {
      id: missingId,
      name: "missing-native",
      phase: "failed",
      error: CODEX_PROVISIONING_INTERRUPTED_ERROR,
    });
    const bindings = new CodexMemberBindingStore(journal, journal.workspaceKey(workspace.root));
    bindings.claim({ id: archivedId, conversationId: "codex:archived-thread" });
    bindings.claim({ id: emptyId, conversationId: "codex:empty-thread" });
    bindings.claim({ id: missingId, conversationId: "codex:missing-thread" });
    journal.close();
    const runtime = {
      read: vi.fn(async (conversationId: string) => {
        if (conversationId === "codex:missing-thread") {
          throw new CodexRpcError("thread not loaded: missing-thread", -32600);
        }
        return snapshot(workspace, conversationId, {
          archived: conversationId === "codex:archived-thread",
          materialized: conversationId !== "codex:empty-thread",
        });
      }),
      archive: vi.fn(async () => undefined),
    };

    await expect(reconcileCodexSwarmBindings({ journalRoot, runtime, workspace })).resolves.toEqual(
      { archived: 3, failed: 3, resumed: 0 },
    );
    expect(runtime.archive).toHaveBeenCalledTimes(1);
    expect(runtime.archive).toHaveBeenCalledWith("codex:empty-thread", expect.anything());
    const recovered = new SwarmJournal(journalRoot);
    expect(recovered.get(teamId)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: archivedId, phase: "failed" }),
        expect.objectContaining({ id: emptyId, phase: "failed" }),
        expect.objectContaining({ id: missingId, phase: "failed" }),
      ]),
    );
    expect(
      new CodexMemberBindingStore(recovered, recovered.workspaceKey(workspace.root)).list(),
    ).toEqual([]);
    recovered.close();
  });
});

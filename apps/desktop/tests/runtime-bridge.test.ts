import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SwarmJournal } from "@swarmx/dsh-swarm";
import { describe, expect, it, vi } from "vitest";
import { RuntimeBridgeClient, startRuntimeBridge } from "../src/runtime/bridge.js";
import type { ConversationRuntime, WorkspaceScope } from "../src/runtime/index.js";
import { startSwarmRecoveryOwner } from "../src/runtime/swarm-recovery-owner.js";

const workspace: WorkspaceScope = {
  id: "workspace-1",
  label: "swarmx",
  root: "/workspace/swarmx",
  token: "scope-token",
};

function runtime(): ConversationRuntime {
  const create = vi.fn(
    async ({ workspace: current }: Parameters<ConversationRuntime["create"]>[0]) => ({
      runtime: "codex" as const,
      conversationId: "codex:thread-1",
      workspace: current,
      title: "New conversation",
      archived: false,
      updatedAt: 1,
    }),
  );
  return {
    kind: "codex",
    list: vi.fn(async () => []),
    create,
    createProvisionedMember: vi.fn(async (request) => create(request)),
    retireProvisionedMember: vi.fn(async () => undefined),
    read: vi.fn(async () => ({
      runtime: "codex",
      conversationId: "codex:thread-1",
      workspace,
      title: "Thread",
      archived: false,
      turns: [],
    })),
    start: vi.fn(async () => ({ turnId: "codex:turn-1" })),
    steer: vi.fn(async () => undefined),
    interrupt: vi.fn(async () => undefined),
    revise: vi.fn(async () => {
      throw new Error("not used");
    }),
    fork: vi.fn(async () => {
      throw new Error("not used");
    }),
    archive: vi.fn(async () => undefined),
    subscribe: vi.fn(() => () => undefined),
    respondToApproval: vi.fn(async () => undefined),
    dispose: vi.fn(async () => undefined),
  };
}

describe("runtime bridge", () => {
  it("requires its bearer token and supplies only the Host-owned workspace scope", async () => {
    const bridge = await startRuntimeBridge(workspace);
    const attached = runtime();
    bridge.attach(attached);
    try {
      const denied = await fetch(bridge.url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "list" }),
      });
      expect(denied.status).toBe(403);

      const client = new RuntimeBridgeClient(bridge.url, bridge.token);
      await expect(client.request({ action: "create", model: "gpt-test" })).resolves.toMatchObject({
        conversationId: "codex:thread-1",
      });
      expect(attached.create).toHaveBeenCalledWith(
        { workspace, model: "gpt-test" },
        expect.any(AbortSignal),
      );
    } finally {
      await bridge.dispose();
    }
  });

  it("aborts ordinary native operations when the response channel disconnects", async () => {
    const bridge = await startRuntimeBridge(workspace);
    const attached = runtime();
    let observedSignal: AbortSignal | undefined;
    vi.mocked(attached.read).mockImplementation(async (_conversationId, signal) => {
      observedSignal = signal;
      await new Promise<void>((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
      throw new Error("unreachable");
    });
    bridge.attach(attached);
    try {
      const controller = new AbortController();
      const client = new RuntimeBridgeClient(bridge.url, bridge.token);
      const request = client.request(
        { action: "read", conversationId: "codex:thread-1" },
        controller.signal,
      );
      void request.catch(() => undefined);
      await vi.waitFor(() => expect(observedSignal).toBeDefined());
      controller.abort(new Error("response disconnected"));
      await expect(request).rejects.toThrow();
      await vi.waitFor(() => expect(observedSignal?.aborted).toBe(true));
    } finally {
      await bridge.dispose();
    }
  });

  it("single-flights concurrent member retirement across bridge callers", async () => {
    const bridge = await startRuntimeBridge(workspace);
    const attached = runtime();
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    if (attached.retireProvisionedMember === undefined) {
      throw new Error("missing test capability");
    }
    vi.mocked(attached.retireProvisionedMember).mockImplementation(async () => gate);
    bridge.attach(attached);
    const memberId = "80000000-0000-4000-8000-000000000004";
    try {
      const first = new RuntimeBridgeClient(bridge.url, bridge.token).request({
        action: "archive",
        conversationId: "codex:thread-1",
        memberId,
      });
      const second = new RuntimeBridgeClient(bridge.url, bridge.token).request({
        action: "archive",
        conversationId: "codex:thread-1",
        memberId,
      });
      await vi.waitFor(() => expect(attached.retireProvisionedMember).toHaveBeenCalledOnce());
      release();
      await expect(Promise.all([first, second])).resolves.toEqual([{}, {}]);
      expect(attached.retireProvisionedMember).toHaveBeenCalledWith(
        "codex:thread-1",
        memberId,
        expect.any(AbortSignal),
      );
    } finally {
      release?.();
      await bridge.dispose();
    }
  });

  it("bounds shutdown when a disconnected root member creation never settles", async () => {
    const bridge = await startRuntimeBridge(workspace, undefined, { shutdownTimeoutMs: 25 });
    const attached = runtime();
    let entered!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    if (attached.createProvisionedMember === undefined) {
      throw new Error("missing test capability");
    }
    vi.mocked(attached.createProvisionedMember).mockImplementation(
      async ({ workspace: current }) => {
        entered();
        await gate;
        return {
          runtime: "codex",
          conversationId: "codex:thread-never-settled",
          workspace: current,
          title: "Delayed member",
          archived: false,
          updatedAt: 1,
        };
      },
    );
    bridge.attach(attached);
    const controller = new AbortController();
    const request = new RuntimeBridgeClient(bridge.url, bridge.token).request(
      {
        action: "create_member",
        teamId: "codex-mcp-thread:lead",
        memberId: "80000000-0000-4000-8000-000000000005",
      },
      controller.signal,
    );
    void request.catch(() => undefined);
    await started;
    controller.abort(new Error("carrier disconnected"));
    try {
      await expect(bridge.dispose()).resolves.toBeUndefined();
    } finally {
      release();
      await request.catch(() => undefined);
    }
  });

  it("V224: durably claims a created member Thread even when its MCP response is lost", async () => {
    const root = mkdtempSync(join(tmpdir(), "swarmx-runtime-bridge-"));
    const workspaceRoot = join(root, "workspace");
    const journalRoot = join(root, "swarm");
    const scopedWorkspace: WorkspaceScope = {
      id: "workspace-response-loss",
      label: "workspace",
      root: workspaceRoot,
      token: "scope-token",
    };
    const teamId = "codex-mcp-thread:lead";
    const memberId = randomUUID();
    const conversationId = "codex:thread-response-loss";
    let releaseCreate!: () => void;
    let nativeCreated!: () => void;
    const created = new Promise<void>((resolve) => {
      nativeCreated = resolve;
    });
    const release = new Promise<void>((resolve) => {
      releaseCreate = resolve;
    });
    let bridge: Awaited<ReturnType<typeof startRuntimeBridge>> | undefined;
    let owner: ReturnType<typeof startSwarmRecoveryOwner> | undefined;
    try {
      mkdirSync(workspaceRoot);
      owner = startSwarmRecoveryOwner(journalRoot);
      const journal = new SwarmJournal(journalRoot, { mode: "client" });
      const workspaceKey = journal.workspaceKey(workspaceRoot);
      journal.append(teamId, {
        type: "team/created",
        data: {
          createdAt: 1,
          lead: {
            createdAt: 1,
            description: "Lead",
            id: teamId,
            name: "lead",
            phase: "active",
            role: "lead",
            modelPolicy: { source: "observed" },
          },
          name: "Response loss",
          workspaceKey,
        },
      });
      journal.append(teamId, {
        type: "member/updated",
        data: {
          createdAt: 2,
          description: "Member",
          id: memberId,
          name: "member",
          phase: "provisioning",
          role: "legacy",
          modelPolicy: { source: "legacy-default" },
        },
      });
      journal.close();

      const attached = runtime();
      if (attached.createProvisionedMember === undefined)
        throw new Error("missing test capability");
      vi.mocked(attached.createProvisionedMember).mockImplementation(
        async ({ workspace: current }) => {
          nativeCreated();
          await release;
          return {
            runtime: "codex",
            conversationId,
            workspace: current,
            title: "New member",
            archived: false,
            updatedAt: 1,
          };
        },
      );
      bridge = await startRuntimeBridge(scopedWorkspace, owner);
      bridge.attach(attached);
      const client = new RuntimeBridgeClient(bridge.url, bridge.token);
      const controller = new AbortController();
      const request = client.request(
        { action: "create_member", teamId, memberId },
        controller.signal,
      );
      void request.catch(() => undefined);
      await created;
      controller.abort(new Error("MCP response channel was lost"));
      let disposed = false;
      const disposal = bridge.dispose().then(() => {
        disposed = true;
      });
      await new Promise<void>((resolve) => setTimeout(resolve, 25));
      expect(disposed).toBe(false);
      releaseCreate();
      await expect(request).rejects.toThrow();
      await expect(disposal).resolves.toBeUndefined();
      expect(attached.createProvisionedMember).toHaveBeenCalledWith(
        { workspace: scopedWorkspace },
        memberId,
      );

      await vi.waitFor(() => {
        const inspected = new SwarmJournal(journalRoot, { mode: "client" });
        try {
          expect(inspected.listMemberBindings(workspaceKey, "codex")).toContainEqual({
            workspaceKey,
            runtime: "codex",
            memberId,
            handle: conversationId,
          });
        } finally {
          inspected.close();
        }
      });
    } finally {
      releaseCreate?.();
      await bridge?.dispose();
      await owner?.dispose();
      rmSync(root, { recursive: true, force: true });
    }
  });
});

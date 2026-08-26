import { mkdtempSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Agent } from "@deepseek-ai/dsh-agent";
import { SessionId } from "@deepseek-ai/dsh-session";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SwarmCoordinator, type SwarmRuntimeAdapter } from "../src/coordinator.js";
import { SwarmJournal } from "../src/journal.js";

const roots: string[] = [];

function agent(id: string, origin?: "subagent"): Agent {
  const value = {
    id: SessionId(id),
    status: "idle" as const,
    session: {
      header: { cwd: "/opaque/project", id: SessionId(id), origin },
    },
    whenIdle: vi.fn(() => Promise.resolve()),
  };
  return value as unknown as Agent;
}

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "swarmx-swarm-coordinator-"));
  roots.push(root);
  const lead = agent("session-lead");
  const agents = new Map<string, Agent>([[lead.id, lead]]);
  const followup = vi.fn(() => Promise.resolve());
  const inject = vi.fn();
  const interrupt = vi.fn();
  const stopContinuable = vi.fn(async (_parent: Agent, targetId: string) => {
    agents.delete(targetId);
  });
  const runtime: SwarmRuntimeAdapter = {
    exact: (candidate) => agents.get(candidate.id) === candidate,
    getAgent: (id) => agents.get(id),
    workspaceKey: () => `swarmx--${"a".repeat(64)}`,
    inject,
    interrupt,
    stopContinuable,
    followup,
    followupRoot: inject,
    async startContinuable(_parent, request) {
      const child = agent(request.childId, "subagent");
      agents.set(child.id, child);
      return child.id;
    },
  };
  const journal = new SwarmJournal(root);
  const coordinator = new SwarmCoordinator(journal, runtime, {
    maxMembers: 8,
    maxMessageBytes: 1_024,
    maxPendingMessagesPerMember: 2,
    maxTasks: 32,
    quiescenceTimeoutMs: 1_000,
  });
  return {
    agents,
    coordinator,
    followup,
    inject,
    interrupt,
    journal,
    lead,
    runtime,
    stopContinuable,
  };
}

async function teamWithMembers() {
  const value = fixture();
  await value.coordinator.create(value.lead, { name: "Research team" });
  await value.coordinator.addMember(value.lead, {
    description: "Owns implementation",
    name: "alpha",
    prompt: "Join the team and wait for a task.",
  });
  await value.coordinator.addMember(value.lead, {
    description: "Owns verification",
    name: "beta",
    prompt: "Join the team and wait for a task.",
  });
  return value;
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("Swarm coordinator", () => {
  it("V162/V164: uses exact Agent authority and immutable continuable Session identities", async () => {
    const { coordinator, journal, lead } = fixture();
    const created = await coordinator.create(lead, { name: "Research team" });
    const member = await coordinator.addMember(lead, {
      description: "Analyzes evidence",
      name: "analyst",
      prompt: "Wait for evidence tasks.",
    });
    expect(created.id).toBe(lead.id);
    expect(member).toMatchObject({ name: "analyst", phase: "active", role: "member" });
    expect(member.id).not.toBe("analyst");

    await expect(coordinator.snapshot(agent(lead.id))).rejects.toMatchObject({
      code: "SWARM_UNAUTHORIZED",
    });
    await expect(
      coordinator.create(agent("nested", "subagent"), { name: "Nested team" }),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });
    journal.close();
  });

  it("V164/V168: drains the actual child when provisioning changes the reserved identity", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    vi.spyOn(value.runtime, "startContinuable").mockResolvedValue("unexpected-child");

    await expect(
      value.coordinator.addMember(value.lead, {
        description: "Analyzes evidence",
        name: "analyst",
        prompt: "Wait for evidence tasks.",
      }),
    ).rejects.toMatchObject({ code: "SWARM_INVALID_REQUEST" });
    expect(value.stopContinuable).toHaveBeenCalledWith(value.lead, "unexpected-child");
    value.journal.close();
  });

  it("V165/V166: fences revisions and attempts while allowing reads beside one writer", async () => {
    const value = await teamWithMembers();
    const alpha = value.coordinator.memberByName(value.lead, "alpha");
    const beta = value.coordinator.memberByName(value.lead, "beta");
    const writeOne = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Change source",
      kind: "write",
      subject: "Write one",
      writeScopes: ["src/one"],
    });
    const writeTwo = await value.coordinator.createTask(value.lead, {
      assignedTo: "beta",
      blockedBy: [],
      description: "Change another source",
      kind: "write",
      subject: "Write two",
      writeScopes: ["src/two"],
    });
    const read = await value.coordinator.createTask(value.lead, {
      assignedTo: "beta",
      blockedBy: [],
      description: "Review documentation",
      kind: "read",
      subject: "Read",
      writeScopes: [],
    });

    expect(value.coordinator.task(value.lead, writeOne.id)).toMatchObject({
      ownerId: alpha.id,
      status: "in_progress",
    });
    expect(value.coordinator.task(value.lead, writeTwo.id).status).toBe("pending");
    expect(value.coordinator.task(value.lead, read.id)).toMatchObject({
      ownerId: beta.id,
      status: "in_progress",
    });

    const activeWrite = value.coordinator.task(value.lead, writeOne.id);
    await expect(
      value.coordinator.updateTask(agent(alpha.id), {
        action: "complete",
        attemptId: activeWrite.attemptId as string,
        expectedRevision: activeWrite.revision,
        taskId: activeWrite.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });
    await expect(
      value.coordinator.updateTask(alpha, {
        action: "complete",
        attemptId: "stale-attempt",
        expectedRevision: activeWrite.revision,
        taskId: activeWrite.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_STALE_ATTEMPT" });

    await value.coordinator.updateTask(alpha, {
      action: "complete",
      attemptId: activeWrite.attemptId as string,
      expectedRevision: activeWrite.revision,
      taskId: activeWrite.id,
    });
    expect(value.coordinator.task(value.lead, writeOne.id).status).toBe("completed");
    value.journal.close();
  });

  it("V165: rejects missing dependencies and stale task revisions", async () => {
    const { coordinator, journal, lead } = await teamWithMembers();
    await expect(
      coordinator.createTask(lead, {
        blockedBy: ["task-404"],
        description: "Cannot start",
        kind: "read",
        subject: "Blocked",
        writeScopes: [],
      }),
    ).rejects.toMatchObject({ code: "SWARM_TASK_DEPENDENCY" });
    const task = await coordinator.createTask(lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Read",
      kind: "read",
      subject: "Read",
      writeScopes: [],
    });
    const current = coordinator.task(lead, task.id);
    const owner = coordinator.memberByName(lead, "alpha");
    await expect(
      coordinator.updateTask(owner, {
        action: "complete",
        attemptId: current.attemptId as string,
        expectedRevision: current.revision - 1,
        taskId: current.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_STALE_REVISION" });
    journal.close();
  });

  it("V166: assigns one unowned task to exactly one idle member", async () => {
    const value = await teamWithMembers();
    const task = await value.coordinator.createTask(value.lead, {
      blockedBy: [],
      description: "Inspect shared evidence",
      kind: "read",
      subject: "Inspect",
      writeScopes: [],
    });
    expect(value.coordinator.task(value.lead, task.id).status).toBe("in_progress");
    expect(value.followup).toHaveBeenCalledTimes(1);
    value.journal.close();
  });

  it("V167/V169: queues quiet mail, delivers once on recovery, and hides bodies from UI", async () => {
    const value = await teamWithMembers();
    const beta = value.coordinator.memberByName(value.lead, "beta");
    value.agents.delete(beta.id);

    const sent = await value.coordinator.sendMessage(value.lead, {
      content: "private coordination detail",
      delivery: "quiet",
      target: "beta",
    });
    expect(sent.status).toBe("queued");
    expect(value.inject).not.toHaveBeenCalled();

    const resumed = agent(beta.id, "subagent");
    value.agents.set(resumed.id, resumed);
    await value.coordinator.recoverMember(resumed);
    await value.coordinator.recoverMember(resumed);
    expect(value.inject).toHaveBeenCalledTimes(1);

    const snapshot = await value.coordinator.snapshot(resumed);
    expect(snapshot.pendingMessages).toBe(0);
    expect(JSON.stringify(snapshot)).not.toContain("private coordination detail");
    expect(JSON.stringify(snapshot)).not.toContain(beta.id);
    value.journal.close();
  });

  it("V166/V173: waits for old-owner quiescence before reassignment and archival", async () => {
    const value = await teamWithMembers();
    const alpha = value.coordinator.memberByName(value.lead, "alpha");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Mutate",
      kind: "write",
      subject: "Mutate",
      writeScopes: ["src"],
    });
    const active = value.coordinator.task(value.lead, task.id);
    await value.coordinator.reassignTask(value.lead, {
      expectedRevision: active.revision,
      target: "beta",
      taskId: active.id,
    });
    expect(value.interrupt).toHaveBeenCalledWith(value.lead, alpha.id);
    expect(alpha.whenIdle).toHaveBeenCalledOnce();

    await value.coordinator.archive(value.lead);
    expect(value.stopContinuable).toHaveBeenCalledTimes(2);
    expect((await value.coordinator.snapshot(value.lead)).kind).toBe("archived");
    await expect(
      value.coordinator.createTask(value.lead, {
        blockedBy: [],
        description: "Too late",
        kind: "read",
        subject: "Too late",
        writeScopes: [],
      }),
    ).rejects.toMatchObject({ code: "SWARM_ARCHIVED" });
    value.journal.close();
  });
});

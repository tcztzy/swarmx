import { mkdtempSync } from "node:fs";
import { rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Agent } from "@deepseek-ai/dsh-agent";
import { SessionId } from "@deepseek-ai/dsh-session";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type KnowledgeCommitter,
  SwarmCoordinator,
  SwarmMemberStartupError,
  type SwarmRuntimeAdapter,
} from "../src/coordinator.js";
import { SwarmJournal } from "../src/journal.js";

const roots: string[] = [];

function agent(id: string, origin?: "subagent"): Agent {
  const value = {
    id: SessionId(id),
    status: "idle" as const,
    session: {
      header: { cwd: "/opaque/project", id: SessionId(id), origin },
    },
    cancel: vi.fn(),
    whenIdle: vi.fn(() => Promise.resolve()),
  };
  return value as unknown as Agent;
}

function fixture(knowledge?: KnowledgeCommitter) {
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
    getActor: (id) => agents.get(id),
    isSubagent: (candidate) => agents.get(candidate.id)?.session.header.origin === "subagent",
    modelOptions: (candidate) => agents.get(candidate.id)?.options ?? {},
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
  const coordinator = new SwarmCoordinator(
    journal,
    runtime,
    {
      maxMembers: 8,
      maxMessageBytes: 1_024,
      maxPendingMessagesPerMember: 2,
      maxTasks: 32,
      quiescenceTimeoutMs: 1_000,
    },
    knowledge,
  );
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

async function teamWithMembers(knowledge?: KnowledgeCommitter) {
  const value = fixture(knowledge);
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
  it("V187: routes distinct immutable role/model/budget profiles into continuable creation", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    const start = vi.spyOn(value.runtime, "startContinuable");
    await value.coordinator.addMember(value.lead, {
      description: "Implements bounded work",
      name: "cheap-impl",
      prompt: "Wait for implementation work.",
      role: "implementer",
      agentOptions: { provider: "ollama", model: "qwen3:32b", maxTokens: 4_096 },
      budget: { maxWallMs: 30_000, maxOutputTokens: 8_000, warningFraction: 0.75 },
    });
    await value.coordinator.addMember(value.lead, {
      description: "Independently verifies submissions",
      name: "strong-verifier",
      prompt: "Wait for verification work.",
      role: "verifier",
      agentOptions: { provider: "openai", model: "gpt-5.6", maxTokens: 16_384 },
    });

    expect(start.mock.calls[0]?.[1]).toMatchObject({
      role: "implementer",
      agentOptions: { provider: "ollama", model: "qwen3:32b", maxTokens: 4_096 },
    });
    expect(start.mock.calls[1]?.[1]).toMatchObject({
      role: "verifier",
      agentOptions: { provider: "openai", model: "gpt-5.6", maxTokens: 16_384 },
    });
    expect(value.journal.get(value.lead.id)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "cheap-impl",
          role: "implementer",
          modelPolicy: expect.objectContaining({ source: "requested", model: "qwen3:32b" }),
        }),
        expect.objectContaining({
          name: "strong-verifier",
          role: "verifier",
          modelPolicy: expect.objectContaining({ source: "requested", model: "gpt-5.6" }),
        }),
      ]),
    );
    const cheap = value.coordinator.memberByName(value.lead, "cheap-impl");
    value.agents.delete(cheap.id);
    expect(value.coordinator.memberProfileByActorId(cheap.id)).toMatchObject({
      role: "implementer",
      modelPolicy: { source: "requested", provider: "ollama", model: "qwen3:32b" },
    });
    expect(() => value.coordinator.memberProfile(cheap)).toThrowError(
      expect.objectContaining({ code: "SWARM_UNAUTHORIZED" }),
    );
    expect(JSON.stringify(value.journal.get(value.lead.id))).not.toMatch(/credential|apiKey/iu);
    value.journal.close();
  });

  it("V188/V190: requires owner submission and exact independent verifier verdict", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Implements work",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
    });
    await value.coordinator.addMember(value.lead, {
      description: "Verifies work",
      name: "verify",
      prompt: "Wait.",
      role: "verifier",
    });
    const impl = value.coordinator.memberByName(value.lead, "impl");
    const verifier = value.coordinator.memberByName(value.lead, "verify");
    const created = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      verifier: "verify",
      acceptance: {
        summary: "Implementation passes focused tests.",
        requiredChecks: ["unit"],
        expectedArtifacts: ["report"],
      },
      blockedBy: [],
      description: "Implement routing",
      kind: "write",
      subject: "Route models",
      writeScopes: ["packages/core/swarm"],
    });
    const active = value.coordinator.task(value.lead, created.id);

    await expect(
      value.coordinator.updateTask(impl, {
        action: "complete",
        attemptId: active.attemptId as string,
        expectedRevision: active.revision,
        taskId: active.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_VERIFICATION_REQUIRED" });

    const submitted = await value.coordinator.submitTask(impl, {
      taskId: active.id,
      expectedRevision: active.revision,
      attemptId: active.attemptId as string,
      summary: "Routing implemented and focused tests passed.",
      artifactLocators: [{ kind: "reference", label: "report", resource: "artifacts/test-report" }],
      evidenceDigests: [`sha256:${"b".repeat(64)}`],
    });
    expect(submitted.status).toBe("submitted");
    expect(value.coordinator.hasActiveWriteAttempt(impl)).toBe(false);
    const submissionId = submitted.submission?.id as string;

    await expect(
      value.coordinator.startVerification(impl, {
        taskId: submitted.id,
        expectedRevision: submitted.revision,
        attemptId: active.attemptId as string,
        submissionId,
      }),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });
    const verifying = await value.coordinator.startVerification(verifier, {
      taskId: submitted.id,
      expectedRevision: submitted.revision,
      attemptId: active.attemptId as string,
      submissionId,
    });
    expect(verifying.status).toBe("verifying");

    await expect(
      value.coordinator.recordVerdict(verifier, {
        taskId: verifying.id,
        expectedRevision: verifying.revision,
        attemptId: active.attemptId as string,
        submissionId: "20000000-0000-4000-8000-000000000002",
        verdict: "pass",
        checkResults: [{ name: "unit", status: "pass" }],
        rationale: "Stale submission must not pass.",
      }),
    ).rejects.toMatchObject({ code: "SWARM_STALE_SUBMISSION" });
    const completed = await value.coordinator.recordVerdict(verifier, {
      taskId: verifying.id,
      expectedRevision: verifying.revision,
      attemptId: active.attemptId as string,
      submissionId,
      verdict: "pass",
      checkResults: [{ name: "unit", status: "pass" }],
      rationale: "Focused tests and artifact evidence passed.",
    });
    expect(completed).toMatchObject({
      status: "completed",
      verification: { mode: "independent", verdict: "pass", verifierName: "verify" },
    });
    await expect(
      value.coordinator.recordVerdict(verifier, {
        taskId: completed.id,
        expectedRevision: verifying.revision,
        attemptId: active.attemptId as string,
        submissionId,
        verdict: "fail",
        checkResults: [{ name: "unit", status: "fail" }],
        rationale: "A stale verdict cannot replace completion.",
      }),
    ).rejects.toMatchObject({ code: "SWARM_STALE_REVISION" });
    value.journal.close();
  });

  it("V162/V164: uses exact Agent authority and immutable continuable Session identities", async () => {
    const { coordinator, journal, lead } = fixture();
    const created = await coordinator.create(lead, { name: "Research team" });
    const member = await coordinator.addMember(lead, {
      description: "Analyzes evidence",
      name: "analyst",
      prompt: "Wait for evidence tasks.",
    });
    expect(created.id).toBe(lead.id);
    expect(member).toMatchObject({ name: "analyst", phase: "active", role: "legacy" });
    expect(member.id).not.toBe("analyst");

    await expect(coordinator.snapshot(agent(lead.id))).rejects.toMatchObject({
      code: "SWARM_UNAUTHORIZED",
    });
    await expect(
      coordinator.create(agent("nested", "subagent"), { name: "Nested team" }),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });
    journal.close();
  });

  it("V162/V224: rejects an exact actor outside the Team workspace", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Workspace A team" });
    vi.spyOn(value.runtime, "workspaceKey").mockReturnValue(`swarmx--${"b".repeat(64)}`);

    await expect(value.coordinator.snapshot(value.lead)).resolves.toEqual({
      kind: "inactive",
      revision: 0,
    });
    await expect(value.coordinator.archive(value.lead)).rejects.toMatchObject({
      code: "SWARM_NOT_FOUND",
    });
    value.journal.close();
  });

  it("V230: does not revoke a member after native interrupt rejection", async () => {
    const value = await teamWithMembers();
    const failure = new Error("native interrupt rejected");
    value.interrupt.mockRejectedValueOnce(failure);

    await expect(value.coordinator.interruptMember(value.lead, { target: "alpha" })).rejects.toBe(
      failure,
    );
    expect(value.interrupt).toHaveBeenCalledOnce();
    const member = value.coordinator.memberByName(value.lead, "alpha");
    expect(value.coordinator.memberProfileByActorId(member.id).phase).toBe("active");
    value.journal.close();
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

    await expect(
      value.coordinator.updateTask(alpha, {
        action: "complete",
        attemptId: activeWrite.attemptId as string,
        expectedRevision: activeWrite.revision,
        taskId: activeWrite.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_VERIFICATION_REQUIRED" });
    const submitted = await value.coordinator.submitTask(alpha, {
      attemptId: activeWrite.attemptId as string,
      expectedRevision: activeWrite.revision,
      taskId: activeWrite.id,
      summary: "Write completed for independent review.",
      artifactLocators: [],
      evidenceDigests: [],
    });
    const verifying = await value.coordinator.startVerification(value.lead, {
      attemptId: activeWrite.attemptId as string,
      expectedRevision: submitted.revision,
      taskId: submitted.id,
      submissionId: submitted.submission?.id as string,
    });
    await value.coordinator.recordVerdict(value.lead, {
      attemptId: activeWrite.attemptId as string,
      expectedRevision: verifying.revision,
      taskId: verifying.id,
      submissionId: submitted.submission?.id as string,
      verdict: "pass",
      checkResults: [{ name: "review", status: "pass" }],
      rationale: "Lead independently reviewed the member submission.",
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

  it("V166/V175: requires an explicitly assigned write attempt for lead effects", async () => {
    const value = await teamWithMembers();
    expect(() => value.coordinator.beginToolEffect(value.lead, "call-early", "write")).toThrowError(
      expect.objectContaining({ code: "SWARM_UNAUTHORIZED" }),
    );
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "lead",
      blockedBy: [],
      description: "Integrate reviewed changes",
      kind: "write",
      subject: "Integrate",
      writeScopes: ["src"],
    });
    expect(value.coordinator.task(value.lead, task.id)).toMatchObject({
      ownerId: value.lead.id,
      status: "in_progress",
    });
    const effect = value.coordinator.beginToolEffect(value.lead, "call-integrate", "write");
    expect(effect.ownerId).toBe(value.lead.id);
    value.coordinator.settleToolEffect(value.lead, effect.id, {
      status: "succeeded",
      resultDigest: `sha256:${"c".repeat(64)}`,
    });
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

  it("V169: never records a rejected asynchronous runtime delivery as delivered", async () => {
    const value = await teamWithMembers();
    value.inject.mockRejectedValueOnce(new Error("native delivery failed"));

    await expect(
      value.coordinator.sendMessage(value.lead, {
        content: "must remain uncertain",
        delivery: "quiet",
        target: "alpha",
      }),
    ).resolves.toMatchObject({ status: "uncertain" });
    const message = value.journal.get(value.lead.id)?.messages.at(-1);
    expect(message?.deliveryStartedAt).toEqual(expect.any(Number));
    expect(message?.deliveredAt).toBeUndefined();
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

  it("V164/V173: fences an in-flight member admission until its native creation settles", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Concurrent archive" });
    let entered!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.spyOn(value.runtime, "startContinuable").mockImplementation(async (_parent, request) => {
      const child = agent(request.childId, "subagent");
      value.agents.set(child.id, child);
      entered();
      await gate;
      return child.id;
    });

    const adding = value.coordinator.addMember(value.lead, {
      description: "Races archive",
      name: "racing",
      prompt: "Wait.",
    });
    await started;
    const archiving = value.coordinator.archive(value.lead);
    await vi.waitFor(() =>
      expect(value.journal.get(value.lead.id)?.archiveStartedAt).toBeDefined(),
    );
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
    expect(value.journal.get(value.lead.id)).toMatchObject({
      phase: "active",
      members: expect.arrayContaining([
        expect.objectContaining({ name: "racing", phase: "provisioning" }),
      ]),
    });
    release();

    await expect(adding).rejects.toMatchObject({ code: "SWARM_ARCHIVED" });
    await expect(archiving).resolves.toMatchObject({ phase: "archived" });
    expect(value.journal.get(value.lead.id)?.members).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: "racing", phase: "retired" })]),
    );
    value.journal.close();
  });

  it("V164/V173: drains a provisioning member whose native creation fails during archive", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Failed concurrent archive" });
    let entered!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.spyOn(value.runtime, "startContinuable").mockImplementation(async () => {
      entered();
      await gate;
      throw new SwarmMemberStartupError("Native creation failed", "absent");
    });

    const adding = value.coordinator.addMember(value.lead, {
      description: "Fails while archive starts",
      name: "failing",
      prompt: "Fail.",
    });
    await started;
    const archiving = value.coordinator.archive(value.lead);
    await vi.waitFor(() =>
      expect(value.journal.get(value.lead.id)?.archiveStartedAt).toBeDefined(),
    );
    await new Promise<void>((resolve) => setTimeout(resolve, 25));
    expect(value.journal.get(value.lead.id)).toMatchObject({
      phase: "active",
      members: expect.arrayContaining([
        expect.objectContaining({ name: "failing", phase: "provisioning" }),
      ]),
    });
    release();

    await expect(adding).rejects.toThrow("Native creation failed");
    await expect(archiving).resolves.toMatchObject({ phase: "archived" });
    expect(value.journal.get(value.lead.id)?.members).toEqual(
      expect.arrayContaining([expect.objectContaining({ name: "failing", phase: "retired" })]),
    );
    value.journal.close();
  });

  it("V164/V173: does not infer no native handle from an ambiguous startup rejection", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Ambiguous concurrent archive" });
    let entered!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.spyOn(value.runtime, "startContinuable").mockImplementation(async () => {
      entered();
      await gate;
      throw new Error("Carrier disconnected after native creation may have started");
    });

    const adding = value.coordinator.addMember(value.lead, {
      description: "May retain a native handle",
      name: "ambiguous",
      prompt: "Wait.",
    });
    await started;
    const archiving = value.coordinator.archive(value.lead);
    await vi.waitFor(() =>
      expect(value.journal.get(value.lead.id)?.archiveStartedAt).toBeDefined(),
    );
    release();

    await expect(adding).rejects.toThrow("Carrier disconnected");
    await expect(archiving).rejects.toMatchObject({ code: "SWARM_CLOSED" });
    expect(value.journal.get(value.lead.id)).toMatchObject({
      archiveStartedAt: expect.any(Number),
      phase: "active",
      members: expect.arrayContaining([
        expect.objectContaining({ name: "ambiguous", phase: "provisioning" }),
      ]),
    });
    value.journal.close();
  });

  it("V164/V173: keeps an ambiguous failed admission fenced before a later archive", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Later archive" });
    vi.spyOn(value.runtime, "startContinuable").mockRejectedValue(
      new Error("Carrier timed out while root creation may still be running"),
    );

    await expect(
      value.coordinator.addMember(value.lead, {
        description: "May still be created by the root carrier",
        name: "still-creating",
        prompt: "Wait.",
      }),
    ).rejects.toThrow("Carrier timed out");
    expect(value.journal.get(value.lead.id)?.members).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "still-creating", phase: "provisioning" }),
      ]),
    );

    await expect(value.coordinator.archive(value.lead)).rejects.toMatchObject({
      code: "SWARM_CLOSED",
    });
    expect(value.journal.get(value.lead.id)).toMatchObject({
      archiveStartedAt: expect.any(Number),
      phase: "active",
    });
    value.journal.close();
  });

  it("V164/V173: does not retire a provisioned native child when archival stop fails", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Failed stop archive" });
    let entered!: () => void;
    let release!: () => void;
    const started = new Promise<void>((resolve) => {
      entered = resolve;
    });
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.spyOn(value.runtime, "startContinuable").mockImplementation(async (_parent, request) => {
      const child = agent(request.childId, "subagent");
      value.agents.set(child.id, child);
      entered();
      await gate;
      return child.id;
    });
    value.stopContinuable.mockRejectedValue(new Error("Native archive failed"));

    const adding = value.coordinator.addMember(value.lead, {
      description: "Cannot be retired without a native stop",
      name: "unstopped",
      prompt: "Wait.",
    });
    await started;
    const archiving = value.coordinator.archive(value.lead);
    await vi.waitFor(() =>
      expect(value.journal.get(value.lead.id)?.archiveStartedAt).toBeDefined(),
    );
    release();
    await expect(adding).rejects.toThrow();
    await expect(archiving).rejects.toThrow("Native archive failed");

    expect(value.journal.get(value.lead.id)).toMatchObject({
      archiveStartedAt: expect.any(Number),
      phase: "active",
      members: expect.arrayContaining([
        expect.objectContaining({
          name: "unstopped",
          phase: "provisioning",
          runtimeReadyAt: expect.any(Number),
        }),
      ]),
    });
    value.journal.close();
  });

  it("V168/V173: revokes a submitted attempt before the archived edge", async () => {
    const value = await teamWithMembers();
    const alpha = value.coordinator.memberByName(value.lead, "alpha");
    const created = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Await verification",
      kind: "read",
      subject: "Submitted archive",
      writeScopes: [],
    });
    const active = value.coordinator.task(value.lead, created.id);
    await value.coordinator.submitTask(alpha, {
      attemptId: active.attemptId as string,
      expectedRevision: active.revision,
      taskId: active.id,
      summary: "Ready for review.",
      artifactLocators: [],
      evidenceDigests: [],
    });

    await expect(value.coordinator.archive(value.lead)).resolves.toMatchObject({
      phase: "archived",
    });
    expect(value.journal.get(value.lead.id)?.attempts.at(-1)).toMatchObject({
      status: "interrupted",
    });
    expect(value.journal.get(value.lead.id)?.tasks.at(-1)).toMatchObject({
      status: "needs_attention",
    });
    value.journal.close();
  });

  it("V174/V177: keeps K cognitive work parallel but completes only through owner admission", async () => {
    const commit = vi
      .fn()
      .mockResolvedValueOnce({
        kind: "science_evidence",
        entityId: "30000000-0000-4000-8000-000000000001",
        journalSequence: 42,
      })
      .mockResolvedValueOnce({
        kind: "pkb_concept",
        conceptId: "workspaces/project--abcdef123456/concepts/wrong.md",
        revision: `sha256:${"c".repeat(64)}`,
      });
    const value = fixture({ commit });
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Reviews evidence",
      name: "alpha",
      prompt: "Wait for evidence tasks.",
    });
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Review one evidence candidate",
      kind: "knowledge",
      subject: "Review evidence",
      writeScopes: [],
    });
    const active = value.coordinator.task(value.lead, task.id);
    const owner = value.coordinator.memberByName(value.lead, "alpha");
    await expect(
      value.coordinator.updateTask(owner, {
        action: "complete",
        attemptId: active.attemptId as string,
        expectedRevision: active.revision,
        taskId: active.id,
      }),
    ).rejects.toMatchObject({ code: "SWARM_UNAUTHORIZED" });

    const admission = {
      admissionId: "10000000-0000-4000-8000-000000000001",
      sources: [
        {
          kind: "science_entity" as const,
          entityId: "20000000-0000-4000-8000-000000000001",
        },
      ],
      verification: {
        status: "verified" as const,
        method: "reproduced" as const,
        verifiedAt: 1_000,
      },
      target: {
        kind: "science_evidence" as const,
        projectId: "40000000-0000-4000-8000-000000000001",
        claimId: "50000000-0000-4000-8000-000000000001",
        relation: "supports" as const,
        title: "Reproduced result",
        summary: "The registered result supports the claim.",
        tags: ["verified"],
      },
    };
    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...admission,
          taskId: active.id,
          expectedRevision: active.revision,
          attemptId: active.attemptId as string,
        },
        { callId: "call-admit", signal: new AbortController().signal },
      ),
    ).resolves.toMatchObject({ kind: "science_evidence", journalSequence: 42 });
    expect(value.coordinator.task(value.lead, task.id).status).toBe("completed");
    expect(commit).toHaveBeenCalledTimes(1);

    const nextTask = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Review another evidence candidate",
      kind: "knowledge",
      subject: "Review more evidence",
      writeScopes: [],
    });
    const next = value.coordinator.task(value.lead, nextTask.id);
    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...admission,
          taskId: next.id,
          expectedRevision: next.revision,
          attemptId: next.attemptId as string,
        },
        { callId: "call-cross-task", signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ code: "SWARM_ADMISSION_CONFLICT" });
    expect(commit).toHaveBeenCalledTimes(1);

    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...admission,
          admissionId: "10000000-0000-4000-8000-000000000002",
          taskId: next.id,
          expectedRevision: next.revision,
          attemptId: next.attemptId as string,
        },
        { callId: "call-wrong-receipt", signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ code: "SWARM_ADMISSION_CONFLICT" });
    const uncertainAdmission = value.journal.get(value.lead.id)?.admissions.at(-1);
    expect(uncertainAdmission).toMatchObject({ status: "uncertain" });
    expect(uncertainAdmission).not.toHaveProperty("receipt");
    value.journal.close();
  });

  it("V175/V176: suppresses duplicate W dispatch and requires verification after uncertainty", async () => {
    const value = await teamWithMembers();
    const alpha = value.coordinator.memberByName(value.lead, "alpha");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Change source",
      kind: "write",
      subject: "Mutate",
      writeScopes: ["src"],
    });
    const active = value.coordinator.task(value.lead, task.id);
    const effect = value.coordinator.beginToolEffect(alpha, "call-write-1", "write");
    value.coordinator.settleToolEffect(alpha, effect.id, { status: "uncertain" });

    expect(() => value.coordinator.beginToolEffect(alpha, "call-write-1", "write")).toThrowError(
      expect.objectContaining({ code: "SWARM_DUPLICATE_EFFECT" }),
    );
    expect(() => value.coordinator.beginToolEffect(alpha, "call-write-2", "write")).toThrowError(
      expect.objectContaining({ code: "SWARM_EFFECT_UNCERTAIN" }),
    );
    await expect(
      value.coordinator.reassignTask(value.lead, {
        taskId: active.id,
        expectedRevision: active.revision,
        target: "beta",
      }),
    ).rejects.toMatchObject({ code: "SWARM_EFFECT_UNCERTAIN" });
    value.coordinator.resolveEffect(value.lead, {
      effectId: effect.id,
      taskId: active.id,
      expectedRevision: active.revision,
      attemptId: active.attemptId as string,
      resolution: "absent",
      verification: {
        kind: "tool_postcondition",
        reference: "sha256 of expected output is absent",
        verifiedAt: 2_000,
      },
    });
    expect(value.coordinator.beginToolEffect(alpha, "call-write-2", "write").status).toBe(
      "started",
    );
    value.journal.close();
  });

  it("V177/V178: retries an uncertain owner commit only under a new current K attempt", async () => {
    const commit = vi
      .fn()
      .mockRejectedValueOnce(new Error("receipt lost after owner commit"))
      .mockResolvedValueOnce({
        kind: "pkb_concept",
        conceptId: "workspaces/project--abcdef123456/concepts/result.md",
        revision: `sha256:${"b".repeat(64)}`,
      });
    const value = await teamWithMembers({ commit });
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Review a durable synthesis",
      kind: "knowledge",
      subject: "Review synthesis",
      writeScopes: [],
    });
    const first = value.coordinator.task(value.lead, task.id);
    const baseRequest = {
      admissionId: "70000000-0000-4000-8000-000000000001",
      sources: [{ kind: "reference" as const, resource: "https://example.test/evidence" }],
      verification: {
        status: "verified" as const,
        method: "source_reviewed" as const,
        verifiedAt: 3_000,
      },
      target: {
        kind: "pkb_concept" as const,
        scope: "workspace" as const,
        title: "Verified synthesis",
        description: "A reviewed synthesis.",
        type: "Finding",
        body: "# Verified synthesis\n\nReviewed evidence.",
      },
    };
    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...baseRequest,
          taskId: first.id,
          expectedRevision: first.revision,
          attemptId: first.attemptId as string,
        },
        { callId: "call-admit-first", signal: new AbortController().signal },
      ),
    ).rejects.toThrow(/receipt lost/iu);
    expect(value.journal.get(value.lead.id)?.admissions[0]?.status).toBe("uncertain");

    await value.coordinator.reassignTask(value.lead, {
      taskId: first.id,
      expectedRevision: first.revision,
      target: "beta",
    });
    const second = value.coordinator.task(value.lead, task.id);
    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...baseRequest,
          taskId: first.id,
          expectedRevision: first.revision,
          attemptId: first.attemptId as string,
        },
        { callId: "call-admit-stale", signal: new AbortController().signal },
      ),
    ).rejects.toMatchObject({ code: "SWARM_STALE_REVISION" });
    await expect(
      value.coordinator.admitKnowledge(
        value.lead,
        {
          ...baseRequest,
          taskId: second.id,
          expectedRevision: second.revision,
          attemptId: second.attemptId as string,
        },
        { callId: "call-admit-retry", signal: new AbortController().signal },
      ),
    ).resolves.toMatchObject({ kind: "pkb_concept" });
    expect(commit).toHaveBeenCalledTimes(2);
    expect(value.coordinator.task(value.lead, task.id).status).toBe("completed");
    value.journal.close();
  });

  it("V180: deduplicates repeated quiet and wakeup submissions by caller key", async () => {
    const value = await teamWithMembers();
    value.followup.mockClear();
    value.inject.mockClear();
    const quiet = {
      content: "same message",
      delivery: "quiet" as const,
      idempotencyKey: "60000000-0000-4000-8000-000000000001",
      target: "alpha",
    };
    const wakeup = {
      content: "same wakeup",
      delivery: "wakeup" as const,
      idempotencyKey: "60000000-0000-4000-8000-000000000002",
      target: "beta",
    };
    await value.coordinator.sendMessage(value.lead, quiet);
    await value.coordinator.sendMessage(value.lead, quiet);
    await value.coordinator.sendMessage(value.lead, wakeup);
    await value.coordinator.sendMessage(value.lead, wakeup);
    expect(value.inject).toHaveBeenCalledTimes(1);
    expect(value.followup).toHaveBeenCalledTimes(1);
    expect(value.journal.get(value.lead.id)?.messages).toHaveLength(2);
    await expect(
      value.coordinator.sendMessage(value.lead, { ...quiet, content: "different" }),
    ).rejects.toMatchObject({ code: "SWARM_MESSAGE_CONFLICT" });
    value.journal.close();
  });

  it("V180/V230: keeps wakeup queued until its runtime parent is available", async () => {
    const value = await teamWithMembers();
    const alpha = value.coordinator.memberByName(value.lead, "alpha");
    const beta = value.coordinator.memberByName(value.lead, "beta");
    value.agents.delete(value.lead.id);

    await expect(
      value.coordinator.sendMessage(alpha, {
        content: "deliver after parent resumes",
        delivery: "wakeup",
        target: "beta",
      }),
    ).resolves.toMatchObject({ status: "queued" });
    expect(value.journal.get(value.lead.id)?.messages[0]?.deliveryStartedAt).toBeUndefined();

    value.agents.set(value.lead.id, value.lead);
    await expect(value.coordinator.recoverMember(beta)).resolves.toBe(1);
    expect(value.journal.get(value.lead.id)?.messages[0]?.deliveredAt).toEqual(expect.any(Number));
    value.journal.close();
  });

  it("V224/V230: leaves a member-created task pending when the lead carrier is absent", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Split carrier team" });
    await value.coordinator.addMember(value.lead, {
      description: "Creates work from its own carrier",
      name: "worker",
      prompt: "Wait.",
    });
    const worker = value.coordinator.memberByName(value.lead, "worker");
    value.agents.delete(value.lead.id);

    const task = await value.coordinator.createTask(worker, {
      assignedTo: "worker",
      blockedBy: [],
      description: "Wait for the lead carrier before delivery",
      kind: "read",
      subject: "Deferred delivery",
      writeScopes: [],
    });

    expect(task.status).toBe("pending");
    expect(value.journal.get(value.lead.id)?.attempts).toEqual([]);
    value.journal.close();
  });

  it("V189/V191: cancels an exhausted wall-clock attempt once and revokes its authority", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Bounded implementer",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
      budget: { maxWallMs: 100, warningFraction: 0.8 },
    });
    const owner = value.coordinator.memberByName(value.lead, "impl");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      blockedBy: [],
      description: "Perform bounded work",
      kind: "write",
      subject: "Bounded work",
      writeScopes: ["src"],
    });
    const attempt = value.journal.get(value.lead.id)?.attempts[0];
    expect(attempt).toBeDefined();

    await expect(
      value.coordinator.runMonitor((attempt?.startedAt ?? 0) + 101, 10_000),
    ).resolves.toEqual([value.lead.id]);
    expect(owner.cancel).toHaveBeenCalledWith({
      kind: "hook",
      reason: "Swarm budget exhausted: attempt_wall_exhausted",
    });
    expect(value.coordinator.task(value.lead, task.id)).toMatchObject({
      status: "needs_attention",
      escalationReason: "Attempt exceeded its hard wall-clock deadline.",
    });
    expect(value.coordinator.hasActiveWriteAttempt(owner)).toBe(false);
    await expect(
      value.coordinator.runMonitor((attempt?.startedAt ?? 0) + 102, 10_000),
    ).resolves.toEqual([]);
    value.journal.close();
  });

  it("V230: does not journal budget termination before asynchronous cancel acknowledgement", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Bounded implementer",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
      budget: { maxWallMs: 100, warningFraction: 0.8 },
    });
    const owner = value.coordinator.memberByName(value.lead, "impl");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      blockedBy: [],
      description: "Perform bounded work",
      kind: "write",
      subject: "Bounded work",
      writeScopes: ["src"],
    });
    const attempt = value.journal.get(value.lead.id)?.attempts[0];
    vi.mocked(owner.cancel).mockRejectedValueOnce(new Error("native interrupt rejected"));

    await expect(
      value.coordinator.runMonitor((attempt?.startedAt ?? 0) + 101, 10_000),
    ).resolves.toEqual([value.lead.id]);
    expect(value.coordinator.task(value.lead, task.id)).toMatchObject({ status: "in_progress" });
    expect(value.coordinator.hasActiveWriteAttempt(owner)).toBe(true);
    expect(value.journal.get(value.lead.id)?.findings.at(-1)).toMatchObject({
      action: "lead_review",
      code: "attempt_wall_exhausted",
      summary: expect.stringContaining("could not acknowledge interruption"),
    });
    value.journal.close();
  });

  it("V230/V232: ignores foreign Teams with no runtime-owned actor", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Foreign carrier team" });
    await value.coordinator.addMember(value.lead, {
      description: "Runs on another carrier",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
      budget: { maxWallMs: 100, warningFraction: 0.8 },
    });
    await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      blockedBy: [],
      description: "Must remain owned by the other carrier",
      kind: "write",
      subject: "Foreign work",
      writeScopes: ["src"],
    });
    const before = value.journal.get(value.lead.id);
    const attempt = before?.attempts[0];
    value.agents.clear();

    await expect(
      value.coordinator.runMonitor((attempt?.startedAt ?? 0) + 101, 10_000),
    ).resolves.toEqual([]);
    expect(
      value.coordinator.nextMonitorAt((attempt?.startedAt ?? 0) + 101, 10_000),
    ).toBeUndefined();
    expect(value.journal.get(value.lead.id)).toEqual(before);
    value.journal.close();
  });

  it("V230: retains an exhausted attempt when its runtime actor is unavailable", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Unavailable actor team" });
    await value.coordinator.addMember(value.lead, {
      description: "Bounded implementer",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
      budget: { maxWallMs: 100, warningFraction: 0.8 },
    });
    const owner = value.coordinator.memberByName(value.lead, "impl");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      blockedBy: [],
      description: "Perform bounded work",
      kind: "write",
      subject: "Bounded work",
      writeScopes: ["src"],
    });
    const attempt = value.journal.get(value.lead.id)?.attempts[0];
    value.agents.delete(owner.id);

    await expect(
      value.coordinator.runMonitor((attempt?.startedAt ?? 0) + 101, 10_000),
    ).resolves.toEqual([value.lead.id]);
    expect(value.coordinator.task(value.lead, task.id).status).toBe("in_progress");
    expect(value.journal.get(value.lead.id)?.findings.at(-1)).toMatchObject({
      action: "lead_review",
      summary: expect.stringContaining("could not acknowledge interruption"),
    });
    value.journal.close();
  });

  it("V190/V194: interrupting an active verifier closes its ledger actor without accepting work", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Implements work",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
    });
    await value.coordinator.addMember(value.lead, {
      description: "Verifies work",
      name: "verify",
      prompt: "Wait.",
      role: "verifier",
    });
    const impl = value.coordinator.memberByName(value.lead, "impl");
    const verifier = value.coordinator.memberByName(value.lead, "verify");
    const created = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      verifier: "verify",
      blockedBy: [],
      description: "Review interrupt behavior",
      kind: "read",
      subject: "Interrupt verifier",
      writeScopes: [],
    });
    const active = value.coordinator.task(value.lead, created.id);
    const submitted = await value.coordinator.submitTask(impl, {
      taskId: active.id,
      expectedRevision: active.revision,
      attemptId: active.attemptId as string,
      summary: "Ready for verification.",
      artifactLocators: [],
      evidenceDigests: [],
    });
    await value.coordinator.startVerification(verifier, {
      taskId: submitted.id,
      expectedRevision: submitted.revision,
      attemptId: active.attemptId as string,
      submissionId: submitted.submission?.id as string,
    });

    await value.coordinator.interruptMember(value.lead, { target: "verify" });
    expect(value.interrupt).toHaveBeenCalledWith(value.lead, verifier.id);
    expect(value.coordinator.task(value.lead, created.id)).toMatchObject({
      status: "needs_attention",
      submission: { summary: "Ready for verification." },
    });
    const attempt = value.journal.get(value.lead.id)?.attempts[0];
    expect(attempt).toMatchObject({ status: "interrupted" });
    expect(attempt?.actors.every((actor) => actor.endedAt !== undefined)).toBe(true);
    value.journal.close();
  });

  it("V191: an unexpected member exit revokes its non-idempotent active attempt", async () => {
    const value = await teamWithMembers();
    const owner = value.coordinator.memberByName(value.lead, "alpha");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Do not replay after exit",
      kind: "write",
      subject: "Exit recovery",
      writeScopes: ["src"],
    });

    await value.coordinator.recordMemberLifecycleFailure(owner.id);
    expect(value.coordinator.memberProfileByActorId(owner.id)).toMatchObject({ phase: "failed" });
    expect(value.coordinator.task(value.lead, task.id).status).toBe("needs_attention");
    expect(value.journal.get(value.lead.id)?.attempts[0]).toMatchObject({
      status: "interrupted",
      terminalReason: "Host revoked the active attempt",
    });
    value.journal.close();
  });

  it("V191/V230: retries attempt revocation after a partial lifecycle failure", async () => {
    const value = await teamWithMembers();
    const owner = value.coordinator.memberByName(value.lead, "alpha");
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "alpha",
      blockedBy: [],
      description: "Must converge after a transient journal failure",
      kind: "write",
      subject: "Retry lifecycle cleanup",
      writeScopes: ["src"],
    });
    const append = value.journal.append.bind(value.journal);
    let rejectAttemptEnd = true;
    const appending = vi.spyOn(value.journal, "append").mockImplementation((teamId, event) => {
      if (rejectAttemptEnd && event.type === "attempt/ended") {
        rejectAttemptEnd = false;
        throw new Error("transient revoke failure");
      }
      return append(teamId, event);
    });

    await expect(value.coordinator.recordMemberLifecycleFailure(owner.id)).rejects.toThrow(
      "transient revoke failure",
    );
    expect(value.coordinator.memberProfileByActorId(owner.id)).toMatchObject({ phase: "active" });
    expect(value.coordinator.task(value.lead, task.id).status).toBe("in_progress");

    appending.mockRestore();
    await expect(value.coordinator.recordMemberLifecycleFailure(owner.id)).resolves.toBeUndefined();
    expect(value.coordinator.memberProfileByActorId(owner.id)).toMatchObject({
      phase: "failed",
      error: "Continuable member exited unexpectedly",
    });
    expect(value.coordinator.task(value.lead, task.id).status).toBe("needs_attention");
    expect(value.journal.get(value.lead.id)?.attempts[0]).toMatchObject({
      status: "interrupted",
      terminalReason: "Host revoked the active attempt",
    });
    value.journal.close();
  });

  it("V192/V195: wakes an optional semantic monitor with bounded data and accepts only its strict finding", async () => {
    const value = fixture();
    await value.coordinator.create(value.lead, { name: "Research team" });
    await value.coordinator.addMember(value.lead, {
      description: "Implements work",
      name: "impl",
      prompt: "Wait.",
      role: "implementer",
    });
    await value.coordinator.addMember(value.lead, {
      description: "Reviews event summaries",
      name: "monitor",
      prompt: "Wait.",
      role: "monitor",
    });
    value.followup.mockClear();
    const task = await value.coordinator.createTask(value.lead, {
      assignedTo: "impl",
      blockedBy: [],
      description: "Private task body is not relayed",
      kind: "read",
      subject: "Review routing",
      writeScopes: [],
    });
    expect(value.coordinator.nextMonitorAt(Date.now(), 100)).toBeUndefined();
    value.followup.mockClear();
    await expect(
      value.coordinator.triggerSemanticMonitor(value.lead.id, "submission", task.id),
    ).resolves.toBe(true);
    const payload = value.followup.mock.calls[0]?.[2] as string;
    expect(payload).toContain("swarm-semantic-monitor-event");
    expect(payload).toContain("record_monitor_finding");
    expect(payload).not.toContain("Private task body");
    expect(payload).not.toContain("/opaque/project");
    const triggerId = (
      JSON.parse(payload.slice(payload.indexOf("\n") + 1, payload.lastIndexOf("\n"))) as {
        triggerId: string;
      }
    ).triggerId;
    const monitor = value.coordinator.memberByName(value.lead, "monitor");
    const request = {
      triggerId,
      severity: "warning" as const,
      code: "semantic_submission_concern" as const,
      subject: { kind: "task" as const, id: task.id },
      summary: "Submission evidence needs lead review.",
      action: "lead_review" as const,
    };
    expect(() => value.coordinator.recordSemanticFinding(value.lead, request)).toThrowError(
      expect.objectContaining({ code: "SWARM_UNAUTHORIZED" }),
    );
    value.coordinator.recordSemanticFinding(monitor, request);
    value.coordinator.recordSemanticFinding(monitor, request);
    expect(
      value.journal
        .get(value.lead.id)
        ?.findings.filter((finding) => finding.code === "semantic_submission_concern"),
    ).toHaveLength(1);
    value.journal.close();
  });
});

import { describe, expect, it } from "vitest";
import { swarmTeamStateSchema } from "../src/contracts.js";
import { evaluateSwarmMonitor } from "../src/monitor.js";

function team() {
  return swarmTeamStateSchema.parse({
    id: "session-lead",
    revision: 3,
    name: "Research team",
    workspaceKey: `swarmx--${"a".repeat(64)}`,
    phase: "active",
    createdAt: 0,
    updatedAt: 0,
    members: [
      {
        id: "session-lead",
        name: "lead",
        role: "lead",
        phase: "active",
        description: "Team lead",
        createdAt: 0,
        modelPolicy: { source: "observed", provider: "openai", model: "gpt-5.6" },
      },
    ],
    tasks: [
      {
        id: "task-1",
        sequence: 1,
        revision: 2,
        subject: "Implement router",
        description: "Implement the router.",
        kind: "write",
        status: "in_progress",
        ownerId: "session-lead",
        attemptId: "10000000-0000-4000-8000-000000000001",
        blockedBy: [],
        writeScopes: ["packages/core/swarm"],
        createdAt: 0,
        updatedAt: 0,
      },
    ],
    messages: [],
    effects: [],
    admissions: [],
    attempts: [
      {
        id: "10000000-0000-4000-8000-000000000001",
        taskId: "task-1",
        taskRevision: 2,
        ownerId: "session-lead",
        memberName: "lead",
        role: "lead",
        modelPolicy: { source: "observed", provider: "openai", model: "gpt-5.6" },
        budget: { maxWallMs: 1_000, maxInputTokens: 500, warningFraction: 0.8 },
        budgetState: "unknown",
        status: "active",
        usage: {
          availability: "unknown",
          inputTokens: 0,
          outputTokens: 0,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          turns: 1,
          toolCalls: 0,
        },
        startedAt: 0,
        lastProgressAt: 0,
        warningCodes: [],
      },
    ],
    findings: [],
  });
}

describe("V189/V191 deterministic Swarm monitor", () => {
  it("emits one wall warning then one exhaustion without a polling/model dependency", () => {
    const warning = evaluateSwarmMonitor(team(), {
      now: 850,
      stallMs: 10_000,
      maxPendingMessagesPerMember: 32,
    });
    expect(warning).toEqual([
      expect.objectContaining({
        code: "attempt_wall_warning",
        severity: "warning",
        action: "notify",
      }),
    ]);

    const exhausted = evaluateSwarmMonitor(team(), {
      now: 1_001,
      stallMs: 10_000,
      maxPendingMessagesPerMember: 32,
    });
    expect(exhausted).toEqual([
      expect.objectContaining({
        code: "attempt_wall_exhausted",
        severity: "block",
        action: "needs_attention",
      }),
    ]);
    const withRecorded = team();
    const exhaustedFinding = exhausted[0];
    expect(exhaustedFinding).toBeDefined();
    if (!exhaustedFinding) throw new Error("Expected exhausted finding");
    withRecorded.findings.push({
      ...exhaustedFinding,
      id: "30000000-0000-4000-8000-000000000001",
    });
    expect(
      evaluateSwarmMonitor(withRecorded, {
        now: 1_100,
        stallMs: 10_000,
        maxPendingMessagesPerMember: 32,
      }),
    ).toEqual([]);
  });

  it("detects stalls, mailbox pressure, missing artifacts, and unknown usage with bounded safe text", () => {
    const state = team();
    const task = state.tasks[0];
    const attempt = state.attempts[0];
    if (!task || !attempt) throw new Error("Expected monitor fixture task and attempt");
    state.tasks[0] = {
      ...task,
      status: "submitted",
      acceptance: {
        summary: "Provide test report.",
        requiredChecks: ["unit"],
        expectedArtifacts: ["report"],
      },
      submission: {
        id: "20000000-0000-4000-8000-000000000001",
        attemptId: attempt.id,
        summary: "Submitted without report.",
        artifactLocators: [],
        evidenceDigests: [],
        submittedAt: 500,
      },
    };
    state.attempts[0] = {
      ...attempt,
      status: "submitted",
      submittedAt: 500,
      submission: state.tasks[0].submission,
    };
    state.messages.push(
      ...Array.from({ length: 29 }, (_, index) => ({
        id: `message-${index}`,
        sequence: index + 1,
        senderId: "session-lead",
        senderName: "lead" as const,
        targetId: "session-lead",
        delivery: "quiet" as const,
        content: `private-${index}`,
        createdAt: 100 + index,
      })),
    );

    const findings = evaluateSwarmMonitor(state, {
      now: 700,
      stallMs: 600,
      maxPendingMessagesPerMember: 32,
    });
    expect(findings.map((finding) => finding.code)).toEqual(
      expect.arrayContaining([
        "attempt_stalled",
        "mailbox_near_limit",
        "submission_missing_artifact",
        "usage_unknown",
      ]),
    );
    expect(JSON.stringify(findings)).not.toMatch(/private-|session-lead|packages\/core/iu);
    expect(findings.every((finding) => finding.summary.length <= 500)).toBe(true);
  });
});

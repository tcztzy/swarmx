import { describe, expect, it } from "vitest";
import {
  addSwarmMemberRequestSchema,
  createSwarmTaskRequestSchema,
  recordSwarmVerdictRequestSchema,
  submitSwarmTaskRequestSchema,
  swarmMemberSchema,
} from "../src/contracts.js";
import { redactSwarmText } from "../src/privacy.js";

describe("V187/V189/V190 Swarm contracts", () => {
  it("redacts host paths and credential-shaped values from client/semantic summaries", () => {
    expect(
      redactSwarmText(
        "saved /Users/private/result.json api_key=secret file:///tmp/result C:\\private\\x",
        500,
      ),
    ).toBe("saved [redacted-path] [redacted-secret] [redacted-path] [redacted-path]");
  });

  it("accepts bounded role/model/budget policy and rejects unknown or unsafe options", () => {
    expect(
      addSwarmMemberRequestSchema.parse({
        name: "implementer",
        description: "Implements bounded tasks",
        prompt: "Wait for work.",
        role: "implementer",
        agentOptions: { provider: "local-ollama", model: "qwen3:32b", maxTokens: 8_192 },
        budget: {
          maxWallMs: 60_000,
          maxTurns: 8,
          maxInputTokens: 40_000,
          maxOutputTokens: 8_000,
          warningFraction: 0.8,
        },
      }),
    ).toMatchObject({ role: "implementer", agentOptions: { model: "qwen3:32b" } });
    expect(
      addSwarmMemberRequestSchema.safeParse({
        name: "bad",
        description: "Bad route",
        prompt: "Wait.",
        role: "implementer",
        agentOptions: { provider: "provider with spaces" },
      }).success,
    ).toBe(false);
    expect(
      addSwarmMemberRequestSchema.safeParse({
        name: "bad",
        description: "Unknown field",
        prompt: "Wait.",
        role: "implementer",
        credential: "secret",
      }).success,
    ).toBe(false);
  });

  it("maps persisted pre-profile members to explicit legacy defaults", () => {
    const legacy = swarmMemberSchema.parse({
      id: "session-member",
      name: "member",
      role: "member",
      phase: "active",
      description: "Created by the old contract",
      createdAt: 100,
    });
    expect(legacy).toMatchObject({
      role: "legacy",
      modelPolicy: { source: "legacy-default" },
    });
  });

  it("keeps acceptance, submission, and verdict structures strict and path-safe", () => {
    expect(
      createSwarmTaskRequestSchema.parse({
        subject: "Implement router",
        description: "Add role-aware routing.",
        kind: "write",
        assignedTo: "implementer",
        verifier: "verifier",
        blockedBy: [],
        writeScopes: ["packages/core/swarm"],
        acceptance: {
          summary: "Routing is independently verified.",
          requiredChecks: ["unit"],
          expectedArtifacts: ["test-report"],
          rubric: "No provider credential is persisted.",
        },
      }),
    ).toMatchObject({ verifier: "verifier", acceptance: { requiredChecks: ["unit"] } });

    expect(
      submitSwarmTaskRequestSchema.safeParse({
        taskId: "task-1",
        expectedRevision: 2,
        attemptId: "attempt-1",
        summary: "Implemented and tested.",
        artifactLocators: [
          { kind: "reference", label: "test-report", resource: "/Users/private/report.txt" },
        ],
        evidenceDigests: [],
      }).success,
    ).toBe(false);

    expect(
      recordSwarmVerdictRequestSchema.parse({
        taskId: "task-1",
        expectedRevision: 4,
        attemptId: "attempt-1",
        submissionId: "10000000-0000-4000-8000-000000000001",
        verdict: "pass",
        checkResults: [{ name: "unit", status: "pass", digest: `sha256:${"a".repeat(64)}` }],
        rationale: "All required checks passed.",
      }),
    ).toMatchObject({ verdict: "pass", checkResults: [{ status: "pass" }] });
    expect(
      recordSwarmVerdictRequestSchema.safeParse({
        taskId: "task-1",
        expectedRevision: 4,
        attemptId: "attempt-1",
        submissionId: "10000000-0000-4000-8000-000000000001",
        verdict: "fail",
        checkResults: [],
        rationale: "",
      }).success,
    ).toBe(false);
  });
});

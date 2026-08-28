import { describe, expect, it } from "vitest";
import { assessTeamPlan } from "../src/team-policy.js";

describe("V181 Team preflight contract", () => {
  it("recommends cognitive width while reporting W conflict and Tool serialization pressure", () => {
    const assessment = assessTeamPlan([
      {
        id: "read-a",
        kind: "read",
        blockedBy: [],
        writeScopes: [],
        toolCalls: 2,
        effectfulToolCalls: 0,
      },
      {
        id: "read-b",
        kind: "knowledge",
        blockedBy: [],
        writeScopes: [],
        toolCalls: 1,
        effectfulToolCalls: 0,
      },
      {
        id: "write-a",
        kind: "write",
        blockedBy: ["read-a"],
        writeScopes: ["src"],
        toolCalls: 3,
        effectfulToolCalls: 3,
      },
      {
        id: "write-b",
        kind: "write",
        blockedBy: ["read-b"],
        writeScopes: ["src/model"],
        toolCalls: 2,
        effectfulToolCalls: 2,
      },
    ]);

    expect(assessment).toEqual({
      cognitiveParallelWidth: 2,
      writeConflictRate: 1,
      effectfulToolDensity: 5 / 8,
      recommendedMembers: 2,
      recommendation: "create_team",
      serializationPressure: "high",
    });
  });

  it("keeps a serial plan solo and rejects a cyclic DAG", () => {
    expect(
      assessTeamPlan([
        {
          id: "only",
          kind: "write",
          blockedBy: [],
          writeScopes: ["src"],
          toolCalls: 1,
          effectfulToolCalls: 1,
        },
      ]),
    ).toMatchObject({ recommendation: "stay_solo", recommendedMembers: 1 });
    expect(() =>
      assessTeamPlan([
        {
          id: "a",
          kind: "read",
          blockedBy: ["b"],
          writeScopes: [],
          toolCalls: 0,
          effectfulToolCalls: 0,
        },
        {
          id: "b",
          kind: "read",
          blockedBy: ["a"],
          writeScopes: [],
          toolCalls: 0,
          effectfulToolCalls: 0,
        },
      ]),
    ).toThrow(/acyclic/iu);
  });
});

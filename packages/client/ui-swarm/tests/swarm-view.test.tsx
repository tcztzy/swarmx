import type { SwarmUiSnapshot } from "@swarmx/dsh-swarm/contracts";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { SwarmActivity, swarmSideViewEntry } from "../src/client/swarm-view.js";

const snapshot: SwarmUiSnapshot = {
  kind: "active",
  memberName: "lead",
  members: [
    {
      budgetState: "within",
      description: "Team lead",
      modelLabel: "strong/reasoner",
      name: "lead",
      role: "lead",
      status: "idle",
    },
    {
      budgetState: "warning",
      description: "Implementation",
      modelLabel: "local/coder-small",
      name: "alpha",
      role: "implementer",
      status: "running",
    },
  ],
  name: "Research team",
  pendingMessages: 0,
  revision: 7,
  role: "lead",
  tasks: [
    {
      blockedBy: [],
      id: "task-1",
      kind: "write",
      ownerName: "alpha",
      verifierName: "lead",
      ready: true,
      revision: 2,
      status: "verifying",
      subject: "Implement scheduler",
      budgetState: "warning",
      usage: {
        availability: "known",
        cacheReadTokens: 2,
        cacheWriteTokens: 0,
        inputTokens: 120,
        outputTokens: 40,
        toolCalls: 2,
        turns: 1,
        wallMs: 900,
      },
      submission: {
        summary: "Scheduler implemented with regression coverage.",
        artifactCount: 1,
        evidenceCount: 1,
        submittedAt: 90,
      },
      verification: {
        verifierName: "lead",
        verdict: "pass",
        mode: "independent",
        checkResults: [{ name: "tests", status: "pass" }],
        rationale: "Focused tests passed.",
        recordedAt: 99,
      },
    },
  ],
  findings: [
    {
      action: "notify",
      code: "attempt_wall_warning",
      recordedAt: 95,
      severity: "warning",
      summary: "Attempt is approaching its hard wall-clock deadline.",
    },
  ],
  updatedAt: 100,
};

describe("V171 Swarm Side View", () => {
  it("uses a serializable per-Session entry and renders only safe projection fields", () => {
    const entry = swarmSideViewEntry(snapshot);
    expect(entry).toMatchObject({ id: "swarm-activity", kind: "swarm-activity", mode: "inspect" });
    expect(entry.payload).toEqual(snapshot);

    const markup = renderToStaticMarkup(<SwarmActivity snapshot={snapshot} />);
    expect(markup).toContain("Research team");
    expect(markup).toContain("alpha");
    expect(markup).toContain("local/coder-small");
    expect(markup).toContain("Implement scheduler");
    expect(markup).toContain("Scheduler implemented with regression coverage.");
    expect(markup).toContain("Focused tests passed.");
    expect(markup).toContain("120 input");
    expect(markup).not.toContain("session-");
    expect(markup).not.toContain("/Users/");
    expect(markup).not.toContain("private coordination detail");
  });
});

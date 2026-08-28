import { describe, expect, it } from "vitest";
import { runFaultBenchmark } from "../src/verification-model.js";

describe("V179/V182 executable Swarm state model", () => {
  it("exhaustively preserves the four properties while the prompt-only control violates them", () => {
    const report = runFaultBenchmark(7);

    expect(report.enforced.statesExplored).toBeGreaterThan(100);
    expect(report.enforced).toMatchObject({
      safetyViolationRate: 0,
      duplicateEffects: 0,
      recoveryReplays: 0,
      knowledgePollution: 0,
    });
    expect(report.promptOnly.statesExplored).toBeGreaterThan(100);
    expect(report.promptOnly.safetyViolationRate).toBeGreaterThan(0);
    expect(report.promptOnly.duplicateEffects).toBeGreaterThan(0);
    expect(report.promptOnly.recoveryReplays).toBeGreaterThan(0);
    expect(report.promptOnly.knowledgePollution).toBeGreaterThan(0);
    expect(report.enforced.coordinationWrites).toBeGreaterThan(
      report.promptOnly.coordinationWrites,
    );
  });
});

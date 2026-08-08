import { describe, expect, it } from "vitest";
import {
  canonicalSkillOptimizerConfig,
  emptySkillEvolutionState,
  evaluateSkillCandidateVerdict,
  replaySkillEvolutionRecords,
  SKILL_CANDIDATE_TRANSITIONS,
  skillCandidateRevisionId,
  skillEvaluationGateDigest,
} from "../src/skill-evolution.js";
import { SkillEvaluationGateSchema, SkillPromotionReceiptSchema } from "../src/skill-variants.js";

const BASELINE_DIGEST = `sha256:${"a".repeat(64)}`;

function metrics(overrides: Partial<Record<string, number>> = {}) {
  return {
    quality: 0.5,
    safety: 1,
    failureRate: 0,
    latencyMs: 100,
    contextTokens: 400,
    ...overrides,
  };
}

describe("skillCandidateRevisionId", () => {
  it("derives a stable revision id from the content digest", () => {
    expect(skillCandidateRevisionId(BASELINE_DIGEST)).toBe(`r_${"a".repeat(64)}`);
    expect(skillCandidateRevisionId(`sha256:${"b".repeat(64)}`)).toBe(`r_${"b".repeat(64)}`);
    expect(() => skillCandidateRevisionId("not-a-digest")).toThrow();
  });
});

describe("candidate status transitions", () => {
  it("allows only proposed -> evaluating -> staged and terminal states", () => {
    expect(SKILL_CANDIDATE_TRANSITIONS.proposed).toEqual(["evaluating", "rejected", "quarantined"]);
    expect(SKILL_CANDIDATE_TRANSITIONS.evaluating).toEqual(["staged", "rejected", "quarantined"]);
    expect(SKILL_CANDIDATE_TRANSITIONS.staged).toEqual(["rejected", "quarantined"]);
    expect(SKILL_CANDIDATE_TRANSITIONS.rejected).toEqual([]);
  });
});

describe("evaluateSkillCandidateVerdict", () => {
  const gate = SkillEvaluationGateSchema.parse({
    minSampleCount: 4,
    minQualityImprovement: 0.05,
    minImprovedRatio: 0.5,
  });
  const gateDigest = skillEvaluationGateDigest(gate);
  const samples = (overrides: Array<Record<string, unknown>> = []) =>
    overrides.map((entry, index) => ({
      caseId: `c${index}`,
      baseline: {
        passed: false,
        safetyPassed: true,
        contextTokens: 100,
        latencyMs: 10,
        failed: false,
      },
      candidate: {
        passed: true,
        safetyPassed: true,
        contextTokens: 100,
        latencyMs: 10,
        failed: false,
      },
      candidateRanFirst: index % 2 === 0,
      ...entry,
    }));

  it("declares eligible only on strict quality gain with sample-level improvement", () => {
    const verdict = evaluateSkillCandidateVerdict({
      baseline: metrics({ quality: 0.25 }),
      candidate: metrics({ quality: 0.75 }),
      samples: samples([
        {},
        {},
        {},
        {
          candidate: {
            passed: false,
            safetyPassed: true,
            contextTokens: 100,
            latencyMs: 10,
            failed: false,
          },
        },
      ]),
      gate,
      gateDigest,
    });
    expect(verdict.verdict).toBe("eligible");
  });

  it("rejects a regression in safety even when quality rises", () => {
    const verdict = evaluateSkillCandidateVerdict({
      baseline: metrics({ quality: 0.25 }),
      candidate: metrics({ quality: 0.9, safety: 0.5 }),
      samples: samples([{}, {}, {}, {}]),
      gate,
      gateDigest,
    });
    expect(verdict.verdict).toBe("rejected");
    expect(verdict.reasons.join(" ")).toContain("safety");
  });

  it("rejects failure-rate and context-token regressions", () => {
    const failure = evaluateSkillCandidateVerdict({
      baseline: metrics({ quality: 0.25 }),
      candidate: metrics({ quality: 0.9, failureRate: 0.5 }),
      samples: samples([{}, {}, {}, {}]),
      gate,
      gateDigest,
    });
    expect(failure.verdict).toBe("rejected");
    expect(failure.reasons.join(" ")).toContain("failure rate");

    const context = evaluateSkillCandidateVerdict({
      baseline: metrics({ quality: 0.25 }),
      candidate: metrics({ quality: 0.9, contextTokens: 800 }),
      samples: samples([{}, {}, {}, {}]),
      gate,
      gateDigest,
    });
    expect(context.verdict).toBe("rejected");
    expect(context.reasons.join(" ")).toContain("context tokens");
  });

  it("rejects when the sample count is below the minimum", () => {
    const verdict = evaluateSkillCandidateVerdict({
      baseline: metrics(),
      candidate: metrics({ quality: 0.9 }),
      samples: samples([{}, {}]),
      gate,
      gateDigest,
    });
    expect(verdict.verdict).toBe("rejected");
    expect(verdict.reasons.join(" ")).toContain("sample count");
  });

  it("rejects a mean improvement with no sample-level gain", () => {
    const verdict = evaluateSkillCandidateVerdict({
      baseline: metrics({ quality: 0 }),
      candidate: metrics({ quality: 0.25 }),
      samples: [
        {
          caseId: "c0",
          baseline: {
            passed: false,
            safetyPassed: true,
            contextTokens: 100,
            latencyMs: 10,
            failed: false,
          },
          candidate: {
            passed: true,
            safetyPassed: true,
            contextTokens: 100,
            latencyMs: 10,
            failed: false,
          },
          candidateRanFirst: false,
        },
        ...samples([{}, {}, {}]).map((sample) => ({
          ...sample,
          baseline: { ...sample.baseline, passed: false },
          candidate: { ...sample.candidate, passed: false },
        })),
      ],
      gate,
      gateDigest,
    });
    expect(verdict.verdict).toBe("rejected");
    expect(verdict.reasons.join(" ")).toContain("improvement ratio");
  });
});

describe("canonicalSkillOptimizerConfig", () => {
  it("produces stable sha256 digests", () => {
    const input = {
      optimizerId: "deterministic.v1",
      seed: 7,
      proposer: "none",
      budget: { maxModelCalls: 10, maxTokens: 1000, maxWallTimeMs: 60000 },
    };
    const digest = canonicalSkillOptimizerConfig(input);
    expect(digest).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(canonicalSkillOptimizerConfig(input)).toBe(digest);
    expect(canonicalSkillOptimizerConfig({ ...input, seed: 8 })).not.toBe(digest);
  });
});

describe("promotion receipts", () => {
  it("rejects a promote receipt without its parent revision", () => {
    expect(() =>
      SkillPromotionReceiptSchema.parse({
        schemaVersion: 1,
        receiptId: "skp_test1",
        skillId: "skill",
        decision: "promote",
        gate: "human",
        candidateId: "skc_candidate1",
        candidateRevisionId: `r_${"c".repeat(64)}`,
        casExpectedRevisionId: `r_${"a".repeat(64)}`,
        previousRevisionId: null,
        newRevisionId: `r_${"c".repeat(64)}`,
        actor: "alice",
        reason: "approved",
        idempotencyKey: "k",
        decidedAt: new Date().toISOString(),
      }),
    ).toThrow();
  });
});

describe("replay of promotion records enforces compare-and-swap", () => {
  const now = new Date().toISOString();
  const OPTIMIZER = {
    optimizerId: "deterministic.v1",
    optimizerVersion: "1",
    environmentDigest: `sha256:${"4".repeat(64)}`,
    configDigest: `sha256:${"5".repeat(64)}`,
    seed: 0,
  };
  const REQUEST = {
    schemaVersion: 1,
    skillId: "skill",
    variantId: "skill:default",
    parentRevisionId: `r_${"a".repeat(64)}`,
    parentRevisionDigest: BASELINE_DIGEST,
    baselineContentRef: BASELINE_DIGEST,
    baselineContentDigest: BASELINE_DIGEST,
    targetAgentId: "agent",
    targetModelFingerprint: "model",
    trainDataset: {
      role: "train",
      contentRef: `sha256:${"1".repeat(64)}`,
      contentDigest: `sha256:${"1".repeat(64)}`,
      caseCount: 2,
      format: "swarmx.eval.jsonl",
    },
    devDataset: {
      role: "dev",
      contentRef: `sha256:${"2".repeat(64)}`,
      contentDigest: `sha256:${"2".repeat(64)}`,
      caseCount: 1,
      format: "swarmx.eval.jsonl",
    },
    optimizer: OPTIMIZER,
    budget: { maxWallTimeMs: 60000, maxModelCalls: 10, maxTokens: 1000, maxArtifactBytes: 262144 },
    proposer: "none",
    requestedBy: "test",
  };
  const CHECK = {
    contentDigestVerified: true,
    parentRevisionDigestMatches: true,
    lineageMatchesRequest: true,
    instructionDeltaPresent: true,
    sizeWithinBudget: true,
    deliverySupported: true,
    secretScan: { passed: true, findings: [] },
  };

  function record(id: string, kind: string, key: string, payload: Record<string, unknown>) {
    return { schemaVersion: 1, recordId: id, kind, timestamp: now, idempotencyKey: key, payload };
  }

  function candidateRecord(
    id: string,
    key: string,
    status: string,
    revision: string,
    candidateId = "skc_candidate1",
  ) {
    return record(id, "candidate_created", key, {
      manifest: {
        schemaVersion: 1,
        candidateId,
        skillId: "skill",
        variantId: "skill:default",
        revisionId: revision,
        parentRevisionId: `r_${"a".repeat(64)}`,
        parentRevisionDigest: BASELINE_DIGEST,
        contentRef: `sha256:${revision.slice(2)}`,
        contentDigest: `sha256:${revision.slice(2)}`,
        contentSizeBytes: 10,
        mediaType: "text/markdown",
        targetAgentId: "agent",
        targetModelFingerprint: "model",
        optimizer: OPTIMIZER,
        trainDatasetDigest: `sha256:${"1".repeat(64)}`,
        devDatasetDigest: `sha256:${"2".repeat(64)}`,
        staticChecks: CHECK,
        createdAt: now,
        status,
      },
    });
  }

  function evaluationRecord(
    id: string,
    key: string,
    verdict: string,
    candidateId: string,
    evaluationId = "ske_eval1",
    revision = `r_${"c".repeat(64)}`,
  ) {
    return record(id, "evaluation_recorded", key, {
      evaluationId,
      manifest: {
        schemaVersion: 1,
        evaluationId,
        candidateId,
        candidateRevisionId: revision,
        baselineRevisionId: `r_${"a".repeat(64)}`,
        holdoutContentRef: `sha256:${"3".repeat(64)}`,
        holdoutContentDigest: `sha256:${"3".repeat(64)}`,
        holdoutCaseCount: 4,
        evaluatorId: "test",
        scorerFingerprint: "test.v1",
        runtimeFingerprint: "test.runtime",
        seed: 1,
        sampleCount: 4,
        samplesRef: `sha256:${"6".repeat(64)}`,
        baseline: { quality: 0.25, safety: 1, failureRate: 0, latencyMs: 10, contextTokens: 100 },
        candidate: { quality: 0.75, safety: 1, failureRate: 0, latencyMs: 10, contextTokens: 100 },
        verdict,
        reasons: ["gate:g"],
        gate: { minSampleCount: 4, minQualityImprovement: 0.05, minImprovedRatio: 0.5 },
        completedAt: now,
      },
    });
  }

  function stagedCandidateChain(candidateId: string, revision: string, evaluationId = "ske_eval1") {
    return [
      candidateRecord("evl_c0", `k-cand-${candidateId}`, "proposed", revision, candidateId),
      record("evl_c1", "candidate_status_changed", `k-st1-${candidateId}`, {
        candidateId,
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord(
        "evl_c2",
        `k-eval-${candidateId}`,
        "eligible",
        candidateId,
        evaluationId,
        revision,
      ),
      record("evl_c3", "candidate_status_changed", `k-st2-${candidateId}`, {
        candidateId,
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: evaluationId,
      }),
    ];
  }

  function promotionRecord(
    id: string,
    key: string,
    decision: "promote" | "rollback",
    revision: string,
    expected: string | null,
    previous: string | null,
    opts: Partial<Record<string, string>> = {},
  ) {
    return record(id, "promotion_recorded", key, {
      receipt: {
        schemaVersion: 1,
        receiptId: opts.receiptId ?? `skp_${id.replace("evl_", "")}`,
        skillId: "skill",
        decision,
        gate: "human",
        candidateId: opts.candidateId ?? "skc_candidate1",
        candidateRevisionId: revision,
        parentRevisionId: opts.parentRevisionId ?? `r_${"a".repeat(64)}`,
        evaluationRunId: opts.evaluationRunId ?? "ske_eval1",
        casExpectedRevisionId: expected,
        previousRevisionId: previous,
        newRevisionId: revision,
        actor: "alice",
        reason: "ok",
        idempotencyKey: key,
        decidedAt: now,
      },
      ...(decision === "promote"
        ? {
            promotedRevision: {
              revisionId: revision,
              contentRef: `sha256:${revision.slice(2)}`,
              contentDigest: `sha256:${revision.slice(2)}`,
            },
          }
        : {}),
    });
  }

  it("first promotion is anchored to the recorded optimization request, not the receipt", () => {
    const state = replaySkillEvolutionRecords([
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      candidateRecord("evl_1", "k-cand", "proposed", `r_${"c".repeat(64)}`),
      record("evl_2", "candidate_status_changed", "k-st1", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord("evl_3", "k-eval", "eligible", "skc_candidate1"),
      record("evl_4", "candidate_status_changed", "k-st2", {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      }),
      promotionRecord(
        "evl_5",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
      ),
    ]);
    expect(state.activePointers.skill?.revisionId).toBe(`r_${"c".repeat(64)}`);
  });

  it("rejects a promotion receipt whose self-declared parent is not anchored", () => {
    const records = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      ...stagedCandidateChain("skc_candidate1", `r_${"c".repeat(64)}`),
      promotionRecord(
        "evl_3",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"f".repeat(64)}`,
        null,
        { parentRevisionId: `r_${"f".repeat(64)}` },
      ),
    ];
    expect(() => replaySkillEvolutionRecords(records)).toThrow(
      /optimization anchor|No optimization request anchors/i,
    );
  });

  it("rejects a promotion without a staged candidate or an eligible evaluation", () => {
    const withoutCandidate = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      evaluationRecord("evl_2", "k-eval", "eligible", "skc_candidate1"),
      promotionRecord(
        "evl_3",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
      ),
    ];
    expect(() => replaySkillEvolutionRecords(withoutCandidate)).toThrow(
      /staged candidate|requires candidate|EVALUATION_REQUIRES_EVALUATING/i,
    );

    const rejectedEvaluation = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      candidateRecord("evl_1", "k-cand", "proposed", `r_${"c".repeat(64)}`),
      record("evl_1b", "candidate_status_changed", "k-st1b", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord("evl_2", "k-eval", "rejected", "skc_candidate1"),
      promotionRecord(
        "evl_3",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
      ),
    ];
    expect(() => replaySkillEvolutionRecords(rejectedEvaluation)).toThrow(
      /eligible evaluation|contradicts its own metrics/i,
    );
  });

  it("rejects duplicate candidate ids instead of overwriting", () => {
    const first = candidateRecord("evl_1", "k-cand-1", "proposed", `r_${"c".repeat(64)}`);
    const duplicate = candidateRecord("evl_2", "k-cand-2", "proposed", `r_${"5".repeat(64)}`);
    expect(() => replaySkillEvolutionRecords([first, duplicate])).toThrow(/cannot be replaced/i);
  });

  it("rejects a stale-parent promotion after another revision went active", () => {
    const chain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      candidateRecord("evl_1", "k-cand-1", "proposed", `r_${"c".repeat(64)}`),
      record("evl_2", "candidate_status_changed", "k-st1", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord("evl_3", "k-eval", "eligible", "skc_candidate1"),
      record("evl_4", "candidate_status_changed", "k-st2", {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      }),
      promotionRecord(
        "evl_5",
        "k-promote-1",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
        {
          receiptId: "skp_promo1",
        },
      ),
    ];
    const staleSecondChain = stagedCandidateChain(
      "skc_candidate2",
      `r_${"d".repeat(64)}`,
      "ske_eval2",
    );
    const stalePromotion = promotionRecord(
      "evl_8",
      "k-promote-2",
      "promote",
      `r_${"d".repeat(64)}`,
      `r_${"a".repeat(64)}`,
      `r_${"c".repeat(64)}`,
      { receiptId: "skp_promo2", candidateId: "skc_candidate2", evaluationRunId: "ske_eval2" },
    );
    expect(() =>
      replaySkillEvolutionRecords([...chain, ...staleSecondChain, stalePromotion]),
    ).toThrow(/expected r_/);
  });

  it("rollback restores a retained revision", () => {
    const first = promotionRecord(
      "evl_5",
      "k-promote-1",
      "promote",
      `r_${"c".repeat(64)}`,
      `r_${"a".repeat(64)}`,
      null,
      { receiptId: "skp_promo1" },
    );
    const rollback = promotionRecord(
      "evl_6",
      "k-rollback",
      "rollback",
      `r_${"a".repeat(64)}`,
      `r_${"c".repeat(64)}`,
      `r_${"c".repeat(64)}`,
      { receiptId: "skp_roll1" },
    );
    const state = replaySkillEvolutionRecords([
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      ...stagedCandidateChain("skc_candidate1", `r_${"c".repeat(64)}`),
      first,
      rollback,
    ]);
    expect(state.activePointers.skill?.revisionId).toBe(`r_${"a".repeat(64)}`);
  });

  it("refuses reject/quarantine receipts so they can never create an active pointer", () => {
    const chain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      ...stagedCandidateChain("skc_candidate1", `r_${"c".repeat(64)}`),
    ];
    for (const decision of ["reject", "quarantine"] as const) {
      const rejectReceipt = promotionRecord(
        "evl_3",
        `k-${decision}`,
        decision,
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
      );
      expect(() => replaySkillEvolutionRecords([...chain, rejectReceipt])).toThrow(
        /INVALID_PROMOTION_DECISION|may only carry/i,
      );
      expect(() => replaySkillEvolutionRecords([...chain, rejectReceipt])).not.toThrow(
        /active pointer/i,
      );
    }
  });

  it("rejects a rollback to a quarantined revision", () => {
    const chain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      candidateRecord("evl_1", "k-cand", "proposed", `r_${"c".repeat(64)}`),
      record("evl_2", "candidate_status_changed", "k-st1", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord("evl_3", "k-eval", "eligible", "skc_candidate1"),
      record("evl_4", "candidate_status_changed", "k-st2", {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      }),
      promotionRecord(
        "evl_5",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
        {
          receiptId: "skp_promo1",
        },
      ),
      record("evl_6", "candidate_status_changed", "k-quar", {
        candidateId: "skc_candidate1",
        from: "staged",
        to: "quarantined",
        reason: "review",
      }),
    ];
    const rollback = promotionRecord(
      "evl_7",
      "k-rollback",
      "rollback",
      `r_${"c".repeat(64)}`,
      `r_${"c".repeat(64)}`,
      `r_${"c".repeat(64)}`,
      { receiptId: "skp_roll1" },
    );
    expect(() => replaySkillEvolutionRecords([...chain, rollback])).toThrow(/quarantined/i);
  });

  it("rejects a promotion whose content coordinates differ from the candidate manifest", () => {
    const chain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      ...stagedCandidateChain("skc_candidate1", `r_${"c".repeat(64)}`),
    ];
    const forged = promotionRecord(
      "evl_3",
      "k-promote",
      "promote",
      `r_${"c".repeat(64)}`,
      `r_${"a".repeat(64)}`,
      null,
    );
    const payload = forged.payload as { promotedRevision?: Record<string, unknown> };
    payload.promotedRevision = {
      revisionId: `r_${"c".repeat(64)}`,
      contentRef: `sha256:${"9".repeat(64)}`,
      contentDigest: `sha256:${"9".repeat(64)}`,
    };
    expect(() => replaySkillEvolutionRecords([...chain, forged])).toThrow(
      /PROMOTED_CONTENT_MISMATCH|coordinates do not match/i,
    );
    const notDerivedChain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      ...stagedCandidateChain("skc_candidate2", `r_${"d".repeat(64)}`, "ske_eval2"),
    ];
    // A hand-crafted candidate whose manifest revision is not derived from its
    // own content digest cannot be promoted either.
    const inconsistentCandidate = notDerivedChain[1] as {
      payload: { manifest: { revisionId: string; contentRef: string; contentDigest: string } };
    };
    inconsistentCandidate.payload.manifest.contentRef = `sha256:${"c".repeat(64)}`;
    inconsistentCandidate.payload.manifest.contentDigest = `sha256:${"c".repeat(64)}`;
    const notDerived = promotionRecord(
      "evl_3",
      "k-promote-2",
      "promote",
      `r_${"d".repeat(64)}`,
      `r_${"a".repeat(64)}`,
      null,
      { candidateId: "skc_candidate2", evaluationRunId: "ske_eval2" },
    );
    const notDerivedPayload = notDerived.payload as { promotedRevision?: Record<string, unknown> };
    notDerivedPayload.promotedRevision = {
      revisionId: `r_${"d".repeat(64)}`,
      contentRef: `sha256:${"c".repeat(64)}`,
      contentDigest: `sha256:${"c".repeat(64)}`,
    };
    expect(() => replaySkillEvolutionRecords([...notDerivedChain, notDerived])).toThrow(
      /REVISION_NOT_DERIVED|derived from the promoted content digest/i,
    );
  });

  it("refuses a directly staged candidate and an evaluation without the evaluating state", () => {
    const base = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
    ];
    const stagedCandidate = candidateRecord("evl_1", "k-cand", "staged", `r_${"c".repeat(64)}`);
    expect(() => replaySkillEvolutionRecords([...base, stagedCandidate])).toThrow(
      /must be created as "proposed"/i,
    );
    const proposed = candidateRecord("evl_1", "k-cand", "proposed", `r_${"c".repeat(64)}`);
    const evaluation = evaluationRecord("evl_2", "k-eval", "eligible", "skc_candidate1");
    expect(() => replaySkillEvolutionRecords([...base, proposed, evaluation])).toThrow(
      /requires candidate.*to be evaluating/i,
    );
  });

  it("rejects re-anchoring a retained baseline with different content", () => {
    const conflictingRequest = {
      ...REQUEST,
      baselineContentRef: `sha256:${"9".repeat(64)}`,
      baselineContentDigest: `sha256:${"9".repeat(64)}`,
    };
    const first = record("evl_0", "optimization_requested", "k-req-1", {
      requestId: "svr_1",
      request: REQUEST,
    });
    const second = record("evl_1", "optimization_requested", "k-req-2", {
      requestId: "svr_2",
      request: conflictingRequest,
    });
    expect(() => replaySkillEvolutionRecords([first, second])).toThrow(/re-anchor/i);
  });

  it("is idempotent for exact duplicate records", () => {
    const chain = [
      record("evl_0", "optimization_requested", "k-req", { requestId: "svr_1", request: REQUEST }),
      candidateRecord("evl_1", "k-cand", "proposed", `r_${"c".repeat(64)}`),
      record("evl_2", "candidate_status_changed", "k-st1", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
      evaluationRecord("evl_3", "k-eval", "eligible", "skc_candidate1"),
      record("evl_4", "candidate_status_changed", "k-st2", {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      }),
      promotionRecord(
        "evl_5",
        "k-promote",
        "promote",
        `r_${"c".repeat(64)}`,
        `r_${"a".repeat(64)}`,
        null,
      ),
    ];
    const state = replaySkillEvolutionRecords([...chain, ...chain]);
    expect(state.records.length).toBe(chain.length);
    const empty = emptySkillEvolutionState();
    expect(empty.candidates).toEqual({});
  });
});

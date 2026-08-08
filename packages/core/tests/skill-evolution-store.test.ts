import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  SkillEvolutionCasError,
  SkillEvolutionIdempotencyCollisionError,
} from "../src/skill-evolution.js";
import { SkillEvolutionStore } from "../src/skill-evolution-store.js";

const temporaryRoots: string[] = [];
afterEach(async () => {
  while (temporaryRoots.length > 0) {
    const root = temporaryRoots.pop();
    if (root) await rmTree(root);
  }
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(path.join(tmpdir(), "swarmx-evolution-store-"));
  temporaryRoots.push(root);
  return root;
}

function record(id: string, kind: string, key: string, payload: Record<string, unknown>) {
  return {
    schemaVersion: 1,
    recordId: id,
    kind,
    timestamp: new Date().toISOString(),
    idempotencyKey: key,
    payload,
  };
}

function optimizationRequestedRecord(id: string, key: string, requestId: string) {
  return record(id, "optimization_requested", key, {
    requestId,
    requestedBy: "test",
    request: {
      schemaVersion: 1,
      skillId: "skill",
      variantId: "skill:default",
      parentRevisionId: `r_${"a".repeat(64)}`,
      parentRevisionDigest: `sha256:${"a".repeat(64)}`,
      baselineContentRef: `sha256:${"a".repeat(64)}`,
      baselineContentDigest: `sha256:${"a".repeat(64)}`,
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
      optimizer: {
        optimizerId: "deterministic.v1",
        optimizerVersion: "1",
        environmentDigest: `sha256:${"4".repeat(64)}`,
        configDigest: `sha256:${"5".repeat(64)}`,
        seed: 0,
      },
      budget: {
        maxWallTimeMs: 60000,
        maxModelCalls: 10,
        maxTokens: 1000,
        maxArtifactBytes: 262144,
      },
      proposer: "none",
      requestedBy: "test",
    },
  });
}

describe("SkillEvolutionStore", () => {
  it("replays records into an immutable candidate registry", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    store.append(
      record("evl_1", "candidate_created", "k1", {
        manifest: {
          schemaVersion: 1,
          candidateId: "skc_candidate1",
          skillId: "skill",
          variantId: "skill:default",
          revisionId: `r_${"b".repeat(64)}`,
          parentRevisionId: `r_${"a".repeat(64)}`,
          parentRevisionDigest: `sha256:${"a".repeat(64)}`,
          contentRef: `sha256:${"b".repeat(64)}`,
          contentDigest: `sha256:${"b".repeat(64)}`,
          contentSizeBytes: 10,
          mediaType: "text/markdown",
          targetAgentId: "agent",
          targetModelFingerprint: "model",
          optimizer: {
            optimizerId: "deterministic.v1",
            optimizerVersion: "1",
            environmentDigest: `sha256:${"c".repeat(64)}`,
            configDigest: `sha256:${"d".repeat(64)}`,
            seed: 0,
          },
          trainDatasetDigest: `sha256:${"e".repeat(64)}`,
          devDatasetDigest: `sha256:${"f".repeat(64)}`,
          staticChecks: {
            contentDigestVerified: true,
            parentRevisionDigestMatches: true,
            lineageMatchesRequest: true,
            instructionDeltaPresent: true,
            sizeWithinBudget: true,
            deliverySupported: true,
            secretScan: { passed: true, findings: [] },
          },
          createdAt: new Date().toISOString(),
          status: "proposed",
        },
      }),
    );
    const fresh = new SkillEvolutionStore({ rootDir: store.rootDir });
    expect(fresh.state().candidates.skc_candidate1?.status).toBe("proposed");
  });

  it("is idempotent for exact duplicates and rejects key collisions", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    const first = optimizationRequestedRecord("evl_1", "k1", "svr_1");
    const appended = store.append([first, first]);
    expect(appended.state.records.length).toBe(1);
    expect(() => store.append(optimizationRequestedRecord("evl_2", "k1", "svr_2"))).toThrow(
      SkillEvolutionIdempotencyCollisionError,
    );
  });

  it("recovers a torn tail", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    store.append(optimizationRequestedRecord("evl_1", "k1", "svr_1"));
    const { appendFileSync } = await import("node:fs");
    const logPath = store.eventLogPath;
    appendFileSync(
      logPath,
      `{"schemaVersion":1,"recordId":"evl_2","kind":"optimization_requested","timestamp":"${new Date().toISOString()}","idempotencyKey":"k2","payload":{"requestId":"svr_2"}`,
    );
    const inspection = store.inspect();
    expect(inspection.tornTail).toBe(true);
    const recovery = store.recoverTornTail();
    expect(recovery.recovered).toBe(true);
    expect(recovery.state.records.length).toBe(1);
  });

  it("fails closed on a complete corrupt record", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    store.append(optimizationRequestedRecord("evl_1", "k1", "svr_1"));
    const { appendFileSync } = await import("node:fs");
    appendFileSync(store.eventLogPath, "{not-json}\n");
    expect(() => store.state()).toThrow(/corrupt/);
  });

  it("enforces compare-and-swap at append time for promotions", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    store.append(optimizationRequestedRecord("evl_0", "k0", "svr_1"));
    const candidate = record("evl_1", "candidate_created", "kc", {
      manifest: {
        schemaVersion: 1,
        candidateId: "skc_candidate1",
        skillId: "skill",
        variantId: "skill:default",
        revisionId: `r_${"b".repeat(64)}`,
        parentRevisionId: `r_${"a".repeat(64)}`,
        parentRevisionDigest: `sha256:${"a".repeat(64)}`,
        contentRef: `sha256:${"b".repeat(64)}`,
        contentDigest: `sha256:${"b".repeat(64)}`,
        contentSizeBytes: 10,
        mediaType: "text/markdown",
        targetAgentId: "agent",
        targetModelFingerprint: "model",
        optimizer: {
          optimizerId: "deterministic.v1",
          optimizerVersion: "1",
          environmentDigest: `sha256:${"4".repeat(64)}`,
          configDigest: `sha256:${"5".repeat(64)}`,
          seed: 0,
        },
        trainDatasetDigest: `sha256:${"1".repeat(64)}`,
        devDatasetDigest: `sha256:${"2".repeat(64)}`,
        staticChecks: {
          contentDigestVerified: true,
          parentRevisionDigestMatches: true,
          lineageMatchesRequest: true,
          instructionDeltaPresent: true,
          sizeWithinBudget: true,
          deliverySupported: true,
          secretScan: { passed: true, findings: [] },
        },
        createdAt: new Date().toISOString(),
        status: "staged",
      },
    });
    store.append({
      ...candidate,
      payload: {
        ...candidate.payload,
        manifest: { ...candidate.payload.manifest, status: "proposed" },
      },
    });
    store.append(
      record("evl_1b", "candidate_status_changed", "kst1", {
        candidateId: "skc_candidate1",
        from: "proposed",
        to: "evaluating",
        reason: "eval",
      }),
    );
    const evaluation = record("evl_2", "evaluation_recorded", "ke", {
      evaluationId: "ske_eval1",
      manifest: {
        schemaVersion: 1,
        evaluationId: "ske_eval1",
        candidateId: "skc_candidate1",
        candidateRevisionId: `r_${"b".repeat(64)}`,
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
        verdict: "eligible",
        reasons: ["gate:g"],
        gate: { minSampleCount: 4, minQualityImprovement: 0.05, minImprovedRatio: 0.5 },
        completedAt: new Date().toISOString(),
      },
    });
    store.append(evaluation);
    store.append(
      record("evl_2b", "candidate_status_changed", "kst2", {
        candidateId: "skc_candidate1",
        from: "evaluating",
        to: "staged",
        reason: "eligible",
        evaluationRunId: "ske_eval1",
      }),
    );
    const promotion = (id: string, key: string, expected: string | null, previous: string | null) =>
      record(id, "promotion_recorded", key, {
        receipt: {
          schemaVersion: 1,
          receiptId: id.replace("evl_", "skp_"),
          skillId: "skill",
          decision: "promote",
          gate: "human",
          candidateId: "skc_candidate1",
          candidateRevisionId: `r_${"b".repeat(64)}`,
          parentRevisionId: `r_${"a".repeat(64)}`,
          evaluationRunId: "ske_eval1",
          casExpectedRevisionId: expected,
          previousRevisionId: previous,
          newRevisionId: `r_${"b".repeat(64)}`,
          actor: "alice",
          reason: "ok",
          idempotencyKey: key,
          decidedAt: new Date().toISOString(),
        },
        promotedRevision: {
          revisionId: `r_${"b".repeat(64)}`,
          contentRef: `sha256:${"b".repeat(64)}`,
          contentDigest: `sha256:${"b".repeat(64)}`,
        },
      });
    store.append(promotion("evl_3", "kp1", `r_${"a".repeat(64)}`, null));
    expect(store.state().activePointers.skill?.revisionId).toBe(`r_${"b".repeat(64)}`);
    expect(() =>
      store.append(promotion("evl_4", "kp2", `r_${"a".repeat(64)}`, `r_${"b".repeat(64)}`)),
    ).toThrow(SkillEvolutionCasError);
  });

  it("stores and reads content-addressed blobs", async () => {
    const store = new SkillEvolutionStore({ rootDir: await temporaryRoot() });
    const ref = store.putBytes(Buffer.from("skill content"));
    expect(ref.ref).toMatch(/^sha256:[a-f0-9]{64}$/);
    expect(store.readBytes(ref.ref).toString("utf8")).toBe("skill content");
    const jsonRef = store.putJson({ caseId: "h1" });
    expect((store.readJson(jsonRef.ref) as { caseId: string }).caseId).toBe("h1");
  });
});

async function rmTree(root: string): Promise<void> {
  const { rm } = await import("node:fs/promises");
  await rm(root, { recursive: true, force: true });
}

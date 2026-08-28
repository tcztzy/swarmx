import type { Agent } from "@deepseek-ai/dsh-agent";
import { SessionId } from "@deepseek-ai/dsh-session";
import { describe, expect, it, vi } from "vitest";
import { type AdmitKnowledgeRequest, evidenceSourceSchema } from "../src/contracts.js";
import { OwnerKnowledgeCommitter } from "../src/knowledge.js";

const lead = {
  id: SessionId("session-lead"),
  session: { header: { cwd: "/opaque/project" } },
} as Agent;

const base = {
  admissionId: "10000000-0000-4000-8000-000000000001",
  taskId: "task-1",
  expectedRevision: 2,
  attemptId: "attempt-1",
  verification: {
    status: "verified",
    method: "source_reviewed",
    verifiedAt: 1_000,
  },
} as const;

describe("V177/V178 knowledge owner boundary", () => {
  it("rejects absolute host paths as evidence locators", () => {
    for (const resource of [
      "/Users/private/result.json",
      "C:\\private\\result.json",
      "file:///tmp/result.json",
    ]) {
      expect(() => evidenceSourceSchema.parse({ kind: "reference", resource })).toThrow();
    }
  });

  it("commits Science evidence only through linkEvidence with the admission id", async () => {
    const linkEvidence = vi.fn(() => ({
      evidence: { id: "20000000-0000-4000-8000-000000000001" },
      provenance: { journalSeq: 9 },
    }));
    const committer = new OwnerKnowledgeCommitter({
      approval: { request: vi.fn() },
      pkb: { vault: { createConcept: vi.fn() } },
      science: { linkEvidence },
    });
    const request: AdmitKnowledgeRequest = {
      ...base,
      sources: [{ kind: "science_entity", entityId: "30000000-0000-4000-8000-000000000001" }],
      target: {
        kind: "science_evidence",
        projectId: "40000000-0000-4000-8000-000000000001",
        claimId: "50000000-0000-4000-8000-000000000001",
        relation: "supports",
        title: "Verified evidence",
        summary: "The registered entity supports the claim.",
        tags: [],
      },
    };

    await expect(
      committer.commit(lead, request, {
        callId: "call-admit",
        signal: new AbortController().signal,
      }),
    ).resolves.toEqual({
      kind: "science_evidence",
      entityId: "20000000-0000-4000-8000-000000000001",
      journalSequence: 9,
    });
    expect(linkEvidence).toHaveBeenCalledWith(
      lead.id,
      expect.objectContaining({
        requestId: request.admissionId,
        sourceEntityIds: ["30000000-0000-4000-8000-000000000001"],
      }),
      expect.any(AbortSignal),
    );
  });

  it("requires allowed-once before the PKB Vault receives an idempotent create", async () => {
    const approval = {
      request: vi.fn().mockResolvedValueOnce("denied").mockResolvedValueOnce("allowed-once"),
    };
    const createConcept = vi.fn().mockResolvedValue({
      id: "workspaces/project--abcdef123456/concepts/finding.md",
      revision: `sha256:${"a".repeat(64)}`,
    });
    const committer = new OwnerKnowledgeCommitter({
      approval,
      pkb: { vault: { createConcept } },
      science: { linkEvidence: vi.fn() },
    });
    const request: AdmitKnowledgeRequest = {
      ...base,
      sources: [{ kind: "reference", resource: "https://example.test/source" }],
      target: {
        kind: "pkb_concept",
        scope: "workspace",
        title: "Reviewed finding",
        description: "A reviewed personal synthesis.",
        type: "Finding",
        body: "# Reviewed finding\n\nSynthesis.",
      },
    };
    const context = { callId: "call-pkb", signal: new AbortController().signal };

    await expect(committer.commit(lead, request, context)).rejects.toMatchObject({
      code: "SWARM_UNAUTHORIZED",
    });
    expect(createConcept).not.toHaveBeenCalled();
    await expect(committer.commit(lead, request, context)).resolves.toMatchObject({
      kind: "pkb_concept",
    });
    expect(createConcept).toHaveBeenCalledWith(
      "/opaque/project",
      expect.objectContaining({ requestId: request.admissionId }),
    );
  });
});

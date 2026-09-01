import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  type WikiSkillEvaluation,
  type WikiSkillStageRequest,
  WikiSkillStore,
} from "../src/index.js";

const SHA_A = `sha256:${"a".repeat(64)}` as const;
const TASK_SET = `sha256:${"b".repeat(64)}` as const;
const target = { model: "deepseek-chat", preset: "dsh-science" } as const;
const roots: string[] = [];

function stageRequest(overrides: Partial<WikiSkillStageRequest> = {}): WikiSkillStageRequest {
  return {
    operation: "create",
    proposalId: "10000000-0000-4000-8000-000000000001",
    name: "verified-spreadsheet",
    purposeMarkdown: "# Purpose\n\nApply the reviewed spreadsheet pattern.",
    skillMarkdown:
      "---\nname: verified-spreadsheet\ndescription: Apply the verified spreadsheet workflow.\n---\n\n# Workflow\n\nFollow the validated steps.",
    sources: {
      patterns: [{ id: "workspaces/demo/concepts/pattern.md", revision: SHA_A }],
      traces: [{ endSeq: 9, sessionId: "session-1", startSeq: 2 }],
    },
    target,
    ...overrides,
  };
}

function evaluation(
  candidateRevision: `sha256:${string}`,
  baselineScore: number,
  candidateScore: number,
  baselineRevision?: `sha256:${string}`,
): WikiSkillEvaluation {
  const shared = {
    benchmarkId: "sheetbench",
    runs: 3,
    target,
    taskSetRevision: TASK_SET,
  } as const;
  return {
    baseline: {
      ...shared,
      score: baselineScore,
      ...(baselineRevision === undefined ? {} : { skillRevision: baselineRevision }),
    },
    candidate: { ...shared, score: candidateScore, skillRevision: candidateRevision },
  };
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "swarmx-wikiskill-"));
  roots.push(root);
  const store = new WikiSkillStore(root);
  await store.initialize();
  return { root, store };
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { force: true, recursive: true })));
});

describe("V235/V238 WikiSkill staging", () => {
  it("stages one idempotent invisible SKILL.md + PURPOSE.md with exact provenance", async () => {
    const { root, store } = await fixture();
    const request = stageRequest();
    const first = await store.stage(request);
    const second = await store.stage(request);

    expect(second).toEqual(first);
    expect(first).toMatchObject({
      name: request.name,
      operation: "create",
      proposalId: request.proposalId,
      target,
    });
    const proposalRoot = join(root, "staging", request.proposalId, request.name);
    expect(await readFile(join(proposalRoot, "SKILL.md"), "utf8")).toBe(request.skillMarkdown);
    const purpose = await readFile(join(proposalRoot, "PURPOSE.md"), "utf8");
    expect(purpose).toContain("workspaces/demo/concepts/pattern.md");
    expect(purpose).toContain(SHA_A);
    expect(purpose).toContain("session-1#2-9");
    expect((await stat(proposalRoot)).mode & 0o777).toBe(0o700);
    expect((await stat(join(proposalRoot, "SKILL.md"))).mode & 0o777).toBe(0o600);
    expect(store.activeRoot(target)).not.toContain(`${join("staging", request.proposalId)}`);

    await expect(store.stage(stageRequest({ purposeMarkdown: "# Changed" }))).rejects.toMatchObject(
      { code: "WIKISKILL_REVISION_CONFLICT" },
    );
  });

  it("rejects malformed skills and patch proposals without an exact active revision", async () => {
    const { store } = await fixture();
    await expect(
      store.stage(
        stageRequest({
          skillMarkdown: "---\nname: another-name\ndescription: Wrong identity.\n---\n\nBody",
        }),
      ),
    ).rejects.toMatchObject({ code: "WIKISKILL_INVALID_SKILL" });
    await expect(
      store.stage(
        stageRequest({
          operation: "patch",
          proposalId: "10000000-0000-4000-8000-000000000002",
        }),
      ),
    ).rejects.toMatchObject({ code: "WIKISKILL_INVALID_REQUEST" });
  });
});

describe("V234/V236/V238 WikiSkill resolution", () => {
  it("rejects equal scores without changing active skills and returns a PKB-ready impact draft", async () => {
    const { store } = await fixture();
    const proposal = await store.stage(stageRequest());
    const outcome = await store.resolve(
      proposal.proposalId,
      evaluation(proposal.candidateRevision, 60, 60),
    );

    expect(outcome.verdict).toBe("rejected");
    await expect(store.readActive(target, proposal.name)).resolves.toBeUndefined();
    expect(outcome.impact).toMatchObject({
      description: expect.stringContaining("rejected"),
      status: "draft",
      type: "SkillImpact",
    });
    expect(outcome.impact.body).toContain("60");
    expect(outcome.impact.sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ resource: "dsh-session:session-1#seq=2-9" }),
        expect.objectContaining({ resource: expect.stringContaining("workspaces/demo/concepts") }),
      ]),
    );
  });

  it("promotes only a strict improvement and publishes the complete skill instruction last", async () => {
    const { root, store } = await fixture();
    const proposal = await store.stage(stageRequest());
    const outcome = await store.resolve(
      proposal.proposalId,
      evaluation(proposal.candidateRevision, 60, 61),
    );
    const active = await store.readActive(target, proposal.name);

    expect(outcome).toMatchObject({
      activeRevision: proposal.candidateRevision,
      candidateRevision: proposal.candidateRevision,
      verdict: "accepted",
    });
    expect(active).toMatchObject({
      name: proposal.name,
      revision: proposal.candidateRevision,
      skillMarkdown: stageRequest().skillMarkdown,
      target,
    });
    const activeDirectory = join(store.activeRoot(target), proposal.name);
    expect(await readFile(join(activeDirectory, "SKILL.md"), "utf8")).toBe(
      stageRequest().skillMarkdown,
    );
    expect(await readFile(join(activeDirectory, "PURPOSE.md"), "utf8")).toContain(
      "## SwarmX provenance",
    );
    expect(
      await readFile(join(root, "staging", proposal.proposalId, "outcome.json"), "utf8"),
    ).toContain('"verdict": "accepted"');
  });

  it("fails a stale patch without overwriting an externally changed active instruction", async () => {
    const { store } = await fixture();
    const created = await store.stage(stageRequest());
    await store.resolve(created.proposalId, evaluation(created.candidateRevision, 10, 11));
    const patch = await store.stage(
      stageRequest({
        expectedActiveRevision: created.candidateRevision,
        operation: "patch",
        proposalId: "10000000-0000-4000-8000-000000000003",
        skillMarkdown:
          "---\nname: verified-spreadsheet\ndescription: Apply the improved spreadsheet workflow.\n---\n\n# Better workflow",
      }),
    );
    const activePath = join(store.activeRoot(target), patch.name, "SKILL.md");
    await writeFile(
      activePath,
      "---\nname: verified-spreadsheet\ndescription: External edit.\n---\n\n# External",
      "utf8",
    );

    await expect(store.readActive(target, patch.name)).rejects.toMatchObject({
      code: "WIKISKILL_REVISION_CONFLICT",
    });
    await expect(
      store.resolve(
        patch.proposalId,
        evaluation(patch.candidateRevision, 11, 12, created.candidateRevision),
      ),
    ).rejects.toMatchObject({ code: "WIKISKILL_REVISION_CONFLICT" });
    expect(await readFile(activePath, "utf8")).toContain("# External");
  });

  it("fails closed on malformed active metadata", async () => {
    const { store } = await fixture();
    const proposal = await store.stage(stageRequest());
    await store.resolve(proposal.proposalId, evaluation(proposal.candidateRevision, 10, 11));
    await writeFile(
      join(store.activeRoot(target), proposal.name, ".wikiskill.json"),
      '{"schemaVersion":1}',
      "utf8",
    );

    await expect(store.readActive(target, proposal.name)).rejects.toMatchObject({
      code: "WIKISKILL_IO_ERROR",
    });
  });
});

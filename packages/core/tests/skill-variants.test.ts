import { describe, expect, it } from "vitest";
import {
  HarnessRecipeSchema,
  HarnessSkillBindingSchema,
  LogicalSkillSchema,
  SkillEvolutionCandidateSchema,
  SkillPromotionDecisionSchema,
  evaluateSkillCandidate,
  normalizeLogicalSkill,
  resolveHarnessSkillBinding,
} from "../src/skill-variants.js";

const context = {
  agentProfileId: "researcher",
  harnessId: "claude-code-research",
  softwareId: "claude-code",
  modelId: "deepseek-v4-pro",
  modelFamily: "deepseek-v4",
  modelCapabilities: ["tool-calling", "reasoning"],
  platform: "darwin-arm64",
};

function skill() {
  return LogicalSkillSchema.parse({
    id: "research.paper-review",
    name: "Paper review",
    defaultVariantId: "paper-review:default",
    variants: [
      {
        id: "paper-review:default",
        skillId: "research.paper-review",
        version: "1.0.0",
        target: {},
        delivery: { mode: "prompt_fragment", contentRef: "skills/paper-review/SKILL.md" },
        tokenEstimate: 2200,
        lineage: { source: "upstream", revisionId: "rev-default", contentDigest: "sha256:a" },
      },
      {
        id: "paper-review:deepseek",
        skillId: "research.paper-review",
        version: "1.1.0",
        target: { modelIds: ["deepseek-v4-pro"] },
        delivery: { mode: "prompt_fragment", contentRef: "skills/paper-review/deepseek.md" },
        tokenEstimate: 3100,
        lineage: { source: "evolved", revisionId: "rev-deepseek", contentDigest: "sha256:b" },
      },
      {
        id: "paper-review:researcher",
        skillId: "research.paper-review",
        version: "1.2.0",
        target: { agentProfileIds: ["researcher"] },
        delivery: { mode: "prompt_fragment", contentRef: "skills/paper-review/researcher.md" },
        tokenEstimate: 2800,
        lineage: { source: "local", revisionId: "rev-researcher", contentDigest: "sha256:c" },
      },
    ],
  });
}

describe("Skill variants", () => {
  it("models Harness as a versioned Software + Skills + MCP recipe", () => {
    const recipe = HarnessRecipeSchema.parse({
      id: "research-harness",
      revisionId: "research-harness@2",
      softwareId: "claude-code",
      softwareVersion: "1.0.62",
      skillBindings: [
        { skillId: "research.paper-review", mode: "required" },
        { skillId: "memory", mode: "off" },
      ],
      mcpServerIds: ["zotero", "project-fs"],
      projectContext: { instructionFiles: ["AGENTS.md"] },
      permissions: { mode: "plan", deniedTools: ["shell:destructive"] },
      delivery: { unsupportedSkill: "block" },
    });

    expect(recipe).toMatchObject({
      id: "research-harness",
      softwareId: "claude-code",
      mcpServerIds: ["zotero", "project-fs"],
      projectContext: { includeWorkspaceRules: true },
    });
    expect(recipe.skillBindings[0]).toMatchObject({
      skillId: "research.paper-review",
      mode: "required",
    });
    expect(() =>
      HarnessRecipeSchema.parse({
        ...recipe,
        revisionId: "research-harness@3",
        skillBindings: [recipe.skillBindings[0], recipe.skillBindings[0]],
      }),
    ).toThrow(/duplicate Skill/i);
  });

  it("migrates a legacy single-path Skill into one default variant", () => {
    const normalized = normalizeLogicalSkill({
      id: "legacy.memory",
      name: "Memory",
      path: "skills/memory/SKILL.md",
      tokenEstimate: 900,
    });

    expect(normalized.defaultVariantId).toBe("legacy.memory:default");
    expect(normalized.variants[0]).toMatchObject({
      skillId: "legacy.memory",
      delivery: { mode: "prompt_fragment", contentRef: "skills/memory/SKILL.md" },
      lineage: { source: "legacy" },
      tokenEstimate: 900,
    });
  });

  it("selects exact Agent before exact Model and default", () => {
    expect(
      resolveHarnessSkillBinding(
        skill(),
        { skillId: "research.paper-review", mode: "auto" },
        context,
      ),
    ).toMatchObject({
      status: "resolved",
      variantId: "paper-review:researcher",
      revisionId: "rev-researcher",
      tokenEstimate: 2800,
      selectionReason: 'Exact Agent profile match for "researcher".',
    });

    expect(
      resolveHarnessSkillBinding(
        skill(),
        { skillId: "research.paper-review", mode: "auto" },
        { ...context, agentProfileId: "other" },
      ),
    ).toMatchObject({
      status: "resolved",
      variantId: "paper-review:deepseek",
      selectionReason: 'Exact Model match for "deepseek-v4-pro".',
    });
  });

  it("supports off and pinned bindings and blocks equal-rank ambiguity", () => {
    expect(
      resolveHarnessSkillBinding(
        skill(),
        { skillId: "research.paper-review", mode: "off" },
        context,
      ),
    ).toMatchObject({ status: "off" });
    expect(
      resolveHarnessSkillBinding(
        skill(),
        {
          skillId: "research.paper-review",
          mode: "required",
          variantId: "paper-review:default",
        },
        context,
      ),
    ).toMatchObject({ status: "resolved", variantId: "paper-review:default" });

    const ambiguous = skill();
    ambiguous.variants.push({
      ...ambiguous.variants[1],
      id: "paper-review:deepseek-two",
      lineage: { ...ambiguous.variants[1].lineage, revisionId: "rev-deepseek-two" },
    });
    expect(
      resolveHarnessSkillBinding(
        ambiguous,
        { skillId: "research.paper-review", mode: "auto" },
        { ...context, agentProfileId: "other" },
      ),
    ).toMatchObject({ status: "blocked" });
  });

  it("requires real delivery and rejects inline secret-shaped metadata", () => {
    const unsupported = normalizeLogicalSkill({ id: "unsupported.skill" });
    expect(
      resolveHarnessSkillBinding(
        unsupported,
        { skillId: "unsupported.skill", mode: "required" },
        context,
      ),
    ).toMatchObject({ status: "blocked" });
    expect(() =>
      HarnessSkillBindingSchema.parse({
        skillId: "research.paper-review",
        mode: "auto",
        apiKey: "do-not-store",
      }),
    ).toThrow(/secret/i);
  });

  it("keeps evolution candidates immutable and promotion gated", () => {
    const candidate = SkillEvolutionCandidateSchema.parse({
      id: "candidate-1",
      skillId: "research.paper-review",
      variantId: "paper-review:deepseek",
      revisionId: "rev-candidate",
      parentRevisionId: "rev-deepseek",
      targetAgentId: "researcher",
      targetModelFingerprint: "deepseek-v4-pro@2026-07",
      optimizerId: "skillopt",
      optimizerVersion: "0.1.0",
      optimizerConfigDigest: "sha256:config",
      createdAt: "2026-07-14T08:00:00.000Z",
      status: "staged",
    });
    const decision = SkillPromotionDecisionSchema.parse({
      id: "decision-1",
      candidateRevisionId: candidate.revisionId,
      evaluationRunId: "eval-1",
      decision: "promote",
      gate: "human",
      reason: "Held-out quality improved without safety or context regression.",
      decidedAt: "2026-07-14T09:00:00.000Z",
    });

    expect(decision.gate).toBe("human");
    expect(
      evaluateSkillCandidate(
        { quality: 80, safety: 100, failureRate: 0.1, latencyMs: 1000, contextTokens: 3000 },
        { quality: 84, safety: 100, failureRate: 0.08, latencyMs: 1100, contextTokens: 2900 },
      ),
    ).toBe("eligible");
    expect(
      evaluateSkillCandidate(
        { quality: 80, safety: 100, failureRate: 0.1, latencyMs: 1000, contextTokens: 3000 },
        { quality: 85, safety: 99, failureRate: 0.08, latencyMs: 900, contextTokens: 2800 },
      ),
    ).toBe("rejected");
  });
});

import { describe, expect, it } from "vitest";
import { parseWorkflowJson } from "./workflow-workspace.js";

describe("Workflow parsing", () => {
  it("builds the editor graph from a valid workflow", () => {
    const result = parseWorkflowJson(
      JSON.stringify({
        name: "Review",
        root: "review_agent",
        nodes: {
          review_agent: {
            kind: "agent",
            agent: {
              name: "Reviewer",
              backend: { type: "swarmx" },
              model: "gpt-test",
            },
          },
        },
        edges: [],
      }),
    );

    expect(result.error).toBeNull();
    expect(result.config?.name).toBe("Review");
    expect(result.nodes).toEqual([
      expect.objectContaining({
        id: "review_agent",
        title: "Reviewer",
        harnessId: "swarmx",
        harnessLabel: "SwarmX",
        model: "gpt-test",
        isRoot: true,
      }),
    ]);
  });

  it("rejects a root that is absent from the node map", () => {
    const result = parseWorkflowJson(
      JSON.stringify({
        name: "Broken",
        root: "missing",
        nodes: { present: { kind: "tool", tool: {} } },
        edges: [],
      }),
    );

    expect(result.config).toBeNull();
    expect(result.error).toBe('Workflow JSON root "missing" is not in nodes.');
  });
});

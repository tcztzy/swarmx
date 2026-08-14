import { describe, expect, it } from "vitest";
import {
  isLocalToolResult,
  type LocalMcpTool,
  type LocalTextTool,
  type LocalToolProgress,
  localToolResult,
} from "../src/local-tool-contracts.js";

describe("local tool contracts", () => {
  it("brands model-facing results without leaking the brand", () => {
    const result = localToolResult("model text", { value: 1 }, { isError: true });

    expect(isLocalToolResult(result)).toBe(true);
    expect(result).toMatchObject({
      content: "model text",
      structuredContent: { value: 1 },
      isError: true,
    });
    expect(JSON.parse(JSON.stringify(result))).toEqual({
      content: "model text",
      structuredContent: { value: 1 },
      isError: true,
    });
  });

  it("omits undefined optional fields and rejects lookalike values", () => {
    const result = localToolResult("plain text");

    expect(Object.keys(result)).toEqual(["content"]);
    expect(isLocalToolResult({ content: "plain text" })).toBe(false);
    expect(isLocalToolResult(null)).toBe(false);
    expect(isLocalToolResult([])).toBe(false);
    expect(isLocalToolResult("plain text")).toBe(false);
  });

  it("keeps function and text tool call contracts independent from adapters", async () => {
    const progress: LocalToolProgress[] = [];
    const functionTool = {
      name: "lookup",
      description: "Look up a value.",
      inputSchema: { type: "object" },
      async call(input, context) {
        context?.onProgress?.({ content: "working", structuredContent: input });
        return input;
      },
    } satisfies LocalMcpTool;
    const textTool = {
      kind: "text",
      name: "apply_patch",
      format: { type: "grammar", syntax: "lark", definition: "start: /.+/" },
      async call(input) {
        return input.toUpperCase();
      },
    } satisfies LocalTextTool;

    await expect(
      functionTool.call(
        { id: 1 },
        { invocationId: "inv_1", onProgress: (event) => progress.push(event) },
      ),
    ).resolves.toEqual({ id: 1 });
    await expect(textTool.call("patch")).resolves.toBe("PATCH");
    expect(progress).toEqual([{ content: "working", structuredContent: { id: 1 } }]);
  });
});

import { describe, expect, it } from "vitest";
import { Edge } from "../src/edge.js";
import { EdgeConfigSchema } from "../src/types.js";

describe("Edge", () => {
  it("evaluates conditionless edge as always true", () => {
    const edge = new Edge({ source: "a", target: "b" });
    expect(edge.evaluate({})).toBe(true);
    expect(edge.evaluate({ x: 1 })).toBe(true);
  });

  it("rejects empty source or target", () => {
    expect(() => new Edge({ source: "", target: "b" })).toThrow();
    expect(() => new Edge({ source: "a", target: "" })).toThrow();
  });

  it("schema parses valid edge", () => {
    const result = EdgeConfigSchema.safeParse({ source: "a", target: "b" });
    expect(result.success).toBe(true);
  });

  it("schema rejects missing source", () => {
    const result = EdgeConfigSchema.safeParse({ target: "b" });
    expect(result.success).toBe(false);
  });

  it("evaluates CEL condition correctly", () => {
    const edge = new Edge({ source: "a", target: "b", condition: "x > 5" });
    expect(edge.evaluate({ x: 10 })).toBe(true);
    expect(edge.evaluate({ x: 3 })).toBe(false);
  });

  it("evaluates complex CEL condition", () => {
    const edge = new Edge({
      source: "a",
      target: "b",
      condition: 'status == "done" && count >= 3',
    });
    expect(edge.evaluate({ status: "done", count: 5 })).toBe(true);
    expect(edge.evaluate({ status: "pending", count: 5 })).toBe(false);
    expect(edge.evaluate({ status: "done", count: 1 })).toBe(false);
  });

  it("resolves static target", () => {
    const edge = new Edge({ source: "a", target: "b" });
    expect(edge.resolveTargets({})).toEqual(["b"]);
  });

  it("resolves CEL expression target to string", () => {
    const edge = new Edge({
      source: "a",
      target: '"node_" + name',
    });
    expect(edge.resolveTargets({ name: "foo" })).toEqual(["node_foo"]);
  });

  it("returns to static target on CEL parse failure", () => {
    const edge = new Edge({ source: "a", target: "b" });
    expect(edge.resolveTargets({})).toEqual(["b"]);
  });
});

import { describe, expect, it } from "vitest";
import {
  formatScienceResourceId,
  parseScienceResourceId,
  type ScienceResourceKind,
} from "../src/resource-id.js";

const CASES: readonly [ScienceResourceKind, string][] = [
  ["project", "p"],
  ["notebook", "n"],
  ["artifact", "a"],
  ["document", "d"],
  ["figure", "f"],
  ["record", "r"],
  ["experiment", "e"],
  ["run", "x"],
];

describe("Science resource IDs", () => {
  it.each(CASES)("round-trips %s logical and exact IDs", (kind, prefix) => {
    const logical = formatScienceResourceId(kind, "550e8400-e29b-41d4-a716-446655440000");
    const exact = formatScienceResourceId(kind, "550e8400-e29b-41d4-a716-446655440000", 3);

    expect(logical).toBe(`sx:${prefix}/550e8400-e29b-41d4-a716-446655440000`);
    expect(parseScienceResourceId(logical)).toEqual({
      kind,
      entityId: "550e8400-e29b-41d4-a716-446655440000",
      revision: null,
    });
    expect(exact).toBe(`${logical}@3`);
    expect(parseScienceResourceId(exact)).toEqual({
      kind,
      entityId: "550e8400-e29b-41d4-a716-446655440000",
      revision: 3,
    });
  });

  it("round-trips canonical URI-component encoding", () => {
    const id = "data set/@alpha?β#1";
    const formatted = formatScienceResourceId("artifact", id, 12);

    expect(formatted).toBe("sx:a/data%20set%2F%40alpha%3F%CE%B2%231@12");
    expect(parseScienceResourceId(formatted)).toEqual({
      kind: "artifact",
      entityId: id,
      revision: 12,
    });
  });

  it.each([
    "550e8400-e29b-41d4-a716-446655440000",
    "sx:q/id",
    "sx:a/",
    "sx:a/%",
    "sx:a/%2f",
    "sx:a/%61",
    "sx:a/id@0",
    "sx:a/id@-1",
    "sx:a/id@1.5",
    "sx:a/id@01",
    "sx:a/id@9007199254740992",
    "sx:a/id@1trailing",
    "sx:a/id@1/more",
    "prefix-sx:a/id",
  ])("rejects malformed or non-canonical input %s", (value) => {
    expect(() => parseScienceResourceId(value)).toThrowError(
      expect.objectContaining({ code: "INVALID_RESOURCE_ID" }),
    );
  });

  it("rejects invalid formatter inputs", () => {
    expect(() => formatScienceResourceId("artifact", "")).toThrowError(
      expect.objectContaining({ code: "INVALID_RESOURCE_ID" }),
    );
    expect(() => formatScienceResourceId("artifact", "id", -1)).toThrowError(
      expect.objectContaining({ code: "INVALID_RESOURCE_ID" }),
    );
    expect(() =>
      formatScienceResourceId("artifact", "id", Number.MAX_SAFE_INTEGER + 1),
    ).toThrowError(expect.objectContaining({ code: "INVALID_RESOURCE_ID" }));
    expect(() => formatScienceResourceId("artifact", "😀".repeat(100))).toThrowError(
      expect.objectContaining({ code: "INVALID_RESOURCE_ID" }),
    );
  });
});

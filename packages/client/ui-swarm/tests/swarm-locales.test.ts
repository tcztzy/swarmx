import { describe, expect, it } from "vitest";
import { en, zh } from "../src/client/swarm-locales.js";

describe("V194 Swarm locale contract", () => {
  it("keeps English and Chinese dictionaries structurally complete", () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(zh).sort());
    expect(Object.values(en).every((value) => value.trim().length > 0)).toBe(true);
    expect(Object.values(zh).every((value) => value.trim().length > 0)).toBe(true);
  });
});

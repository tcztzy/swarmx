import { describe, expect, it } from "vitest";
import { en, zh } from "../src/client/annotation-locales.js";

describe("V129 annotation locales", () => {
  it("ships matching complete Chinese and English key sets", () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(zh).sort());
    expect(en["tray.countOne"]).toContain("annotation");
    expect(en["tray.countMany"]).toContain("annotations");
    expect(zh["tray.countMany"]).toContain("条批注");
    expect(en["selection.add"]).not.toBe(zh["selection.add"]);
  });
});

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { APP_ICON_URL } from "./app-icon-data.js";

describe("SwarmX application icon asset", () => {
  it("uses the canonical SVG copied by the renderer build", () => {
    const source = readFileSync(
      new URL("../public/app-icon.svg", import.meta.url),
      "utf8",
    ).trimEnd();
    expect(APP_ICON_URL).toBe("./app-icon.svg");
    expect(source).toMatch(/^<svg[\s>]/);
  });
});

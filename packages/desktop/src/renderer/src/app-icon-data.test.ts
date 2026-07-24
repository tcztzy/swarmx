import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { APP_ICON_URL } from "./app-icon-data.js";

describe("embedded SwarmX application icon", () => {
  it("matches the canonical renderer SVG", () => {
    const source = readFileSync(
      new URL("../public/app-icon.svg", import.meta.url),
      "utf8",
    ).trimEnd();
    const embedded = decodeURIComponent(APP_ICON_URL.slice(APP_ICON_URL.indexOf(",") + 1));

    expect(embedded).toBe(source);
  });
});

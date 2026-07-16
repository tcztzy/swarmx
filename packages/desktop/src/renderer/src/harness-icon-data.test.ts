import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { PACKAGED_HARNESS_ICON_URLS } from "./harness-icon-data.js";

const ICON_IDS = ["claude_code", "codex", "hermes", "opencode"] as const;

describe("packaged harness icons", () => {
  it.each(ICON_IDS)("embeds the revisioned %s SVG source", (harnessId) => {
    const iconUrl = PACKAGED_HARNESS_ICON_URLS[harnessId];
    expect(iconUrl).toMatch(/^data:image\/svg\+xml/);

    const source = readFileSync(
      new URL(`../public/harness-icons/${harnessId}.svg`, import.meta.url),
      "utf8",
    ).trimEnd();
    const embedded = decodeURIComponent(iconUrl.slice(iconUrl.indexOf(",") + 1));
    expect(embedded).toBe(source);
  });

  it("does not claim an unverified OpenClaw product icon", () => {
    expect(PACKAGED_HARNESS_ICON_URLS.openclaw).toBeUndefined();
  });
});

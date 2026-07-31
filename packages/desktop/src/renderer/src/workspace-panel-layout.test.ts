/** @vitest-environment node */

import { describe, expect, it } from "vitest";
import { readStylesheet } from "./stylesheet-test-utils.js";

const styles = readStylesheet(new URL("./assets/styles.css", import.meta.url));

function ruleBody(selector: string, occurrence = 0): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matches = [...styles.matchAll(new RegExp(`^${escaped}\\s*\\{`, "gm"))];
  const match = matches.at(occurrence);
  if (!match || match.index === undefined) throw new Error(`Missing CSS rule: ${selector}`);
  const start = styles.indexOf("{", match.index) + 1;
  return styles.slice(start, styles.indexOf("}", start));
}

describe("workspace panel layout contracts", () => {
  it("defaults to an equal split while allowing the right panel to resize", () => {
    const body = ruleBody(".runtime__body--right-panel");
    const panel = ruleBody(".panel-transition--right");

    expect(body).toMatch(/padding-right:\s*var\(--right-panel-width,\s*50%\)/);
    expect(panel).toMatch(/width:\s*var\(--right-panel-width,\s*50%\)/);
    expect(panel).toMatch(/position:\s*absolute/);
    expect(styles).toMatch(
      /\.runtime--right-panel > \.composer-dock,[\s\S]*?width:\s*calc\(100% - var\(--right-panel-width,\s*50%\)\)/,
    );
    expect(ruleBody(".right-panel-resize")).toMatch(/cursor:\s*col-resize/);
  });

  it("does not fall back to a narrow fixed-width drawer", () => {
    expect(styles).not.toMatch(/\.panel-transition--right\s*\{[^}]*width:\s*min\(310px/s);
    expect(styles).not.toMatch(/\.runtime__body--right-panel\s*\{[^}]*minmax\(260px,\s*310px\)/s);
  });

  it("uses an overlay instead of squeezing the conversation at narrow widths", () => {
    const narrow = styles.slice(styles.lastIndexOf("@media (max-width: 680px)"));

    expect(narrow).toMatch(/\.runtime__body--right-panel\s*\{[^}]*padding-right:\s*0[^}]*\}/s);
    expect(narrow).toMatch(
      /\.panel-transition--right\s*\{[^}]*width:\s*min\(100%,\s*var\(--right-panel-width,\s*100%\)\)/s,
    );
    expect(narrow).toMatch(/\.runtime--right-panel > \.composer-dock,[\s\S]*?width:\s*100%/);
  });

  it("centers the compact prompt set in two wider columns", () => {
    const suggestions = ruleBody(".empty-run__suggestions--right-panel");

    expect(suggestions).toMatch(/width:\s*min\(100%,\s*520px\)/);
    expect(suggestions).toMatch(/grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  });
});

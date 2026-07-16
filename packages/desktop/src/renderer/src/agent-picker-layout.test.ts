/** @vitest-environment node */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const styles = readFileSync(new URL("./assets/styles.css", import.meta.url), "utf8");

function ruleBody(selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matches = [...styles.matchAll(new RegExp(`^${escapedSelector}\\s*\\{`, "gm"))];
  const match = matches.at(-1);
  if (!match || match.index === undefined) throw new Error(`Missing CSS rule: ${selector}`);
  const start = match.index;
  const bodyStart = styles.indexOf("{", start) + 1;
  const bodyEnd = styles.indexOf("}", bodyStart);
  return styles.slice(bodyStart, bodyEnd);
}

describe("agent picker layout contracts", () => {
  it("V204 keeps secondary options out of primary layout flow", () => {
    const menu = ruleBody(".agent-picker__menu");
    const primary = ruleBody(".agent-picker__primary");
    const secondary = ruleBody(".agent-picker__secondary");

    expect(menu).toMatch(/width:\s*var\(--agent-picker-primary-width\)/);
    expect(menu).toMatch(/overflow:\s*visible/);
    expect(menu).toMatch(/left:\s*var\(--agent-picker-inline-offset, 0px\)/);
    expect(menu).not.toMatch(/display:\s*(grid|flex)/);
    expect(primary).toMatch(/height:\s*fit-content/);
    expect(secondary).toMatch(/position:\s*absolute/);
    expect(secondary).toMatch(/left:\s*calc\(100% \+ var\(--agent-picker-panel-gap\)\)/);
    expect(secondary).toMatch(/bottom:\s*0/);
    expect(secondary).toMatch(/max-height:\s*min\(360px, 56vh\)/);
    expect(secondary).toMatch(/overflow-y:\s*auto/);
  });

  it("keeps edge flipping on the secondary panel instead of moving primary layout", () => {
    const flippedSecondary = ruleBody(
      '.agent-picker__menu[data-secondary-side="left"] .agent-picker__secondary',
    );

    expect(flippedSecondary).toMatch(/right:\s*calc\(100% \+ var\(--agent-picker-panel-gap\)\)/);
    expect(flippedSecondary).toMatch(/left:\s*auto/);
  });

  it("keeps keyboard focus visible after removing the global button outline", () => {
    expect(styles).toMatch(
      /\.agent-picker__trigger:focus-visible\s*\{[^}]*box-shadow:\s*0 0 0 3px var\(--ring\)/s,
    );
    expect(styles).toMatch(
      /\.agent-picker__row:focus-visible,\s*\.agent-picker__option:focus-visible\s*\{[^}]*box-shadow:\s*inset 0 0 0 2px var\(--ring\)/s,
    );
  });
});

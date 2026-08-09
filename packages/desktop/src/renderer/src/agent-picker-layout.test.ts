/** @vitest-environment node */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readStylesheet } from "./stylesheet-test-utils.js";

const styles = readStylesheet(new URL("./assets/styles.css", import.meta.url));
const source = readFileSync(new URL("./agent-picker.tsx", import.meta.url), "utf8");

function staticClasses(marker: string): string {
  const escapedMarker = marker.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match =
    source.match(new RegExp(`className="([^"]*\\b${escapedMarker}\\b[^"]*)"`)) ??
    source.match(
      new RegExp(`className=\\{String\\.raw\`([^\`]*\\b${escapedMarker}\\b[^\`]*)\`\\}`),
    );
  if (!match) throw new Error(`Missing static class list: ${marker}`);
  return match[1];
}

describe("agent picker layout contracts", () => {
  it("V204 keeps secondary options out of primary layout flow", () => {
    const menu = staticClasses("agent-picker__menu");
    const primary = staticClasses("agent-picker__primary");
    const secondary = staticClasses("agent-picker__secondary");

    expect(menu).toContain("[width:var(--agent-picker-primary-width)]");
    expect(menu).toContain("[overflow:visible]");
    expect(menu).toContain("[left:var(--agent-picker-inline-offset,_0px)]");
    expect(menu).not.toMatch(/\[(?:display):(grid|flex)\]/);
    expect(primary).toContain("[height:fit-content]");
    expect(secondary).toContain("[position:absolute]");
    expect(secondary).toContain("[left:calc(100%_+_var(--agent-picker-panel-gap))]");
    expect(secondary).toContain("[bottom:0]");
    expect(secondary).toContain("[max-height:min(360px,_56vh)]");
    expect(secondary).toContain("[overflow-y:auto]");
  });

  it("keeps edge flipping on the secondary panel instead of moving primary layout", () => {
    const menu = staticClasses("agent-picker__menu");

    expect(menu).toContain(
      "[&[data-secondary-side='left']_.agent-picker\\_\\_secondary]:[right:calc(100%_+_var(--agent-picker-panel-gap))]",
    );
    expect(menu).toContain(
      "[&[data-secondary-side='left']_.agent-picker\\_\\_secondary]:[left:auto]",
    );
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

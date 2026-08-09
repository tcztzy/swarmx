/** @vitest-environment node */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readStylesheet } from "./stylesheet-test-utils.js";

const styles = readStylesheet(new URL("./assets/styles.css", import.meta.url));
const appSource = readFileSync(new URL("./App.tsx", import.meta.url), "utf8");

describe("workspace panel layout contracts", () => {
  it("defaults to an equal split while allowing the right panel to resize", () => {
    expect(appSource).toContain("[padding-right:var(--right-panel-width,_50%)]");
    expect(appSource).toContain("[width:var(--right-panel-width,_50%)]");
    expect(appSource).toContain("[position:absolute]");
    expect(appSource).toContain(
      "[&_>_.composer-dock]:[width:calc(100%_-_var(--right-panel-width,_50%))]",
    );
    expect(appSource).toMatch(/className="[^"]*right-panel-resize[^"]*\[cursor:col-resize\][^"]*"/);
  });

  it("does not fall back to a narrow fixed-width drawer", () => {
    expect(`${styles}\n${appSource}`).not.toContain("[width:min(310px");
    expect(`${styles}\n${appSource}`).not.toContain("minmax(260px,_310px)");
  });

  it("uses an overlay instead of squeezing the conversation at narrow widths", () => {
    expect(appSource).toContain("max-680:[padding-right:0]");
    expect(appSource).toContain("max-680:[width:min(100%,_var(--right-panel-width,_100%))]");
    expect(appSource).toContain("max-680:[&_>_.composer-dock]:[width:100%]");
  });

  it("centers the compact prompt set in two wider columns", () => {
    expect(appSource).toContain("[width:min(100%,_520px)]");
    expect(appSource).toContain("[grid-template-columns:repeat(2,_minmax(0,_1fr))]");
  });
});

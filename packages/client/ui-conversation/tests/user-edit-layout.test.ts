import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const stylesheet = readFileSync(
  new URL("../src/client/actions.module.css", import.meta.url),
  "utf8",
).replace(/\s+/g, " ");

function declarationsFor(selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const declarations = stylesheet.match(new RegExp(`${escaped}\\s*\\{([^}]*)}`, "s"))?.[1];
  if (declarations === undefined) throw new Error(`Missing CSS rule for ${selector}`);
  return declarations;
}

describe("V12 user-message action layout", () => {
  it("reserves the trailing Edit lane after the native clock and Copy row", () => {
    const nativeActions = declarationsFor(
      '[data-chat-flow-kind="user"]:has(+ [data-chat-flow-kind="swarmx-user-edit"]) [data-time-hover-root] > div:last-child',
    );
    const editRow = declarationsFor(".userEditRow");

    expect(nativeActions).toMatch(/margin-right:\s*38px/);
    expect(editRow).toMatch(/padding-right:\s*0(?:px)?\s*;/);
  });
});

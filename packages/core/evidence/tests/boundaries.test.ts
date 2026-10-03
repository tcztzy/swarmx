import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

it("owns portable contracts without depending on Science or the desktop", () => {
  const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
  expect(Object.keys(manifest.dependencies)).toEqual(["zod"]);
  const source = readFileSync(new URL("../src/index.ts", import.meta.url), "utf8");
  expect([...source.matchAll(/from "([^"]+)"/gu)].map((match) => match[1])).toEqual(["zod"]);
});

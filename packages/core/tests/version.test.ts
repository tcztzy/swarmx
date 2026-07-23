import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { SWARMX_VERSION } from "../src/version.js";

const manifestUrls = [
  new URL("../../../package.json", import.meta.url),
  new URL("../package.json", import.meta.url),
  new URL("../../runtime/package.json", import.meta.url),
  new URL("../../acp-server/package.json", import.meta.url),
  new URL("../../cli/package.json", import.meta.url),
  new URL("../../desktop/package.json", import.meta.url),
  new URL("../../swarmx/package.json", import.meta.url),
];

describe("release version", () => {
  it("V507 keeps the runtime and every workspace manifest on 3.1.4", async () => {
    const versions = await Promise.all(
      manifestUrls.map(async (url) => JSON.parse(await readFile(url, "utf8")).version as unknown),
    );

    expect(SWARMX_VERSION).toBe("3.1.4");
    expect(versions).toEqual(manifestUrls.map(() => SWARMX_VERSION));
  });
});

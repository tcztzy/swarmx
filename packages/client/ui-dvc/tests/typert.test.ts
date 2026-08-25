import { describe, expect, it } from "vitest";
import { dvcUiSnapshotSchema } from "../src/contracts.js";
import { DVC_UI_REMOTE } from "../src/remote.js";
import { DVC_UI_INVOCATIONS } from "../src/remote-contract.js";
import { TYPERT } from "../src/typert.js";

describe("DVC UI Typert contract", () => {
  it("keeps Host and Client descriptors identical and read-only", () => {
    expect(TYPERT.invocations).toBe(DVC_UI_INVOCATIONS);
    expect(DVC_UI_REMOTE.descriptors).toBe(DVC_UI_INVOCATIONS);
    expect(DVC_UI_INVOCATIONS.map(({ method }) => method)).toEqual(["snapshot"]);
  });

  it("rejects Host paths and undeclared fields at the Remote boundary", () => {
    expect(
      dvcUiSnapshotSchema.safeParse({
        kind: "project",
        inspection: {
          version: "3.67.1",
          root: "/private/workspace",
          dvcYamlDigest: null,
          dvcLockDigest: null,
          data: { categories: [], digest: `sha256:${"a".repeat(64)}`, entries: 0 },
          pipeline: { categories: [], digest: `sha256:${"b".repeat(64)}`, entries: 0 },
          git: {
            version: "2.50.1",
            objectFormat: "sha1",
            head: "c".repeat(40),
            branch: "main",
            upstream: null,
            ahead: null,
            behind: null,
            clean: true,
            staged: 0,
            unstaged: 0,
            untracked: 0,
            conflicted: 0,
          },
          path: "/private/workspace",
        },
      }).success,
    ).toBe(false);
  });
});

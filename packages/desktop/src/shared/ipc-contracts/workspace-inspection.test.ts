import { describe, expect, it } from "vitest";
import { DesktopEventContractRegistry, DesktopInvokeContractRegistry } from "./index.js";
import { WorkspaceInspectionInvokeContracts } from "./workspace-inspection.js";

const review = {
  root: "/workspace/swarmx",
  branch: "main",
  isRepository: true,
  files: [
    {
      path: "src/App.tsx",
      previousPath: "src/OldApp.tsx",
      status: "R ",
      patch: "@@ -1 +1 @@",
      binary: false,
      additions: 1,
      deletions: 1,
      truncated: false,
      error: "partial diff",
    },
  ],
  truncated: false,
};

describe("Workspace inspection IPC contracts", () => {
  it("owns exactly four read-only inspection contracts", () => {
    expect(Object.keys(WorkspaceInspectionInvokeContracts)).toEqual([
      "workspace:root",
      "workspace:review",
      "workspace:listDirectory",
      "workspace:readFile",
    ]);
    expect(
      Object.values(WorkspaceInspectionInvokeContracts).every(
        (contract) => contract.audit === "intent_outcome",
      ),
    ).toBe(true);
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) =>
        channel.startsWith("workspace:"),
      ),
    ).toEqual([]);
    expect(
      Object.keys(DesktopInvokeContractRegistry).filter((channel) =>
        channel.startsWith("workspace:"),
      ),
    ).toEqual(Object.keys(WorkspaceInspectionInvokeContracts));
  });

  it("accepts canonical optional inputs and rejects extra arguments or fields", () => {
    expect(WorkspaceInspectionInvokeContracts["workspace:root"].args.parse([])).toEqual([]);
    expect(WorkspaceInspectionInvokeContracts["workspace:root"].result.parse("/workspace")).toBe(
      "/workspace",
    );
    expect(WorkspaceInspectionInvokeContracts["workspace:root"].result.safeParse(42).success).toBe(
      false,
    );
    expect(WorkspaceInspectionInvokeContracts["workspace:review"].args.parse([{}])).toEqual([{}]);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].args.parse([
        { cwd: "/workspace/swarmx" },
      ]),
    ).toEqual([{ cwd: "/workspace/swarmx" }]);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:listDirectory"].args.parse([
        { path: "src", cwd: "/workspace/swarmx" },
      ]),
    ).toEqual([{ path: "src", cwd: "/workspace/swarmx" }]);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].args.safeParse([
        { path: "src/App.tsx", rawCredential: "secret" },
      ]).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:root"].args.safeParse(["unexpected"]).success,
    ).toBe(false);
  });

  it("validates strict review, listing, and preview projections", () => {
    expect(WorkspaceInspectionInvokeContracts["workspace:review"].result.parse(review)).toEqual(
      review,
    );
    expect(
      WorkspaceInspectionInvokeContracts["workspace:listDirectory"].result.parse({
        root: "/workspace/swarmx",
        path: "src",
        entries: [
          { name: "nested", path: "src/nested", kind: "directory" },
          { name: "App.tsx", path: "src/App.tsx", kind: "file", size: 24 },
        ],
        truncated: false,
      }),
    ).toMatchObject({ path: "src", entries: [{ kind: "directory" }, { size: 24 }] });
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].result.parse({
        root: "/workspace/swarmx",
        path: "src/App.tsx",
        content: "export function App() {}",
        size: 24,
        binary: false,
        truncated: false,
      }),
    ).toMatchObject({ path: "src/App.tsx", size: 24 });
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].result.safeParse({
        root: "/workspace/swarmx",
        path: "src/App.tsx",
        content: "export function App() {}",
        size: 24,
        binary: false,
        truncated: false,
        sha256: "a".repeat(64),
      }).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].result.safeParse({
        ...review,
        files: [{ ...review.files[0], rawCredential: "secret" }],
      }).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].result.safeParse({
        ...review,
        branch: undefined,
      }).success,
    ).toBe(false);
  });

  it("keeps paths, collections, and previews within Desktop transport bounds", () => {
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].args.safeParse([
        { path: "bad\0path" },
      ]).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].args.safeParse([{ cwd: "bad\0root" }])
        .success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].args.safeParse([{ path: "" }])
        .success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:listDirectory"].result.safeParse({
        root: "/workspace/swarmx",
        path: "",
        entries: Array.from({ length: 501 }, (_, index) => ({
          name: `file-${index}`,
          path: `file-${index}`,
          kind: "file",
        })),
        truncated: true,
      }).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].result.safeParse({
        ...review,
        files: Array.from({ length: 201 }, () => review.files[0]),
      }).success,
    ).toBe(false);
    const maximumPatch = "x".repeat(256 * 1024);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].result.safeParse({
        ...review,
        files: Array.from({ length: 8 }, () => ({
          ...review.files[0],
          patch: maximumPatch,
        })),
      }).success,
    ).toBe(true);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:review"].result.safeParse({
        ...review,
        files: Array.from({ length: 9 }, () => ({
          ...review.files[0],
          patch: maximumPatch,
        })),
      }).success,
    ).toBe(false);
    expect(
      WorkspaceInspectionInvokeContracts["workspace:readFile"].result.safeParse({
        root: "/workspace/swarmx",
        path: "large.txt",
        content: "x".repeat(1024 * 1024 + 1),
        size: 1024 * 1024 + 1,
        binary: false,
        truncated: false,
      }).success,
    ).toBe(false);
  });
});

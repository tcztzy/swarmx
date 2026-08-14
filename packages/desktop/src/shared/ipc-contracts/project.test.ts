import { describe, expect, it } from "vitest";
import { DesktopEventContractRegistry } from "./index.js";
import { ProjectInvokeContracts } from "./project.js";

const project = {
  id: "project-1",
  name: "Project One",
  cwd: "/workspace/project-one",
  pinned: false,
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
};

describe("Project IPC contracts", () => {
  it("owns all eight argument, result, and audit contracts", () => {
    expect(Object.keys(ProjectInvokeContracts)).toEqual([
      "project:list",
      "project:addExisting",
      "project:createScratch",
      "project:setPinned",
      "project:rename",
      "project:reveal",
      "project:archiveTasks",
      "project:remove",
    ]);
    expect(
      Object.values(ProjectInvokeContracts).every(
        (contract) => contract.audit === "intent_outcome",
      ),
    ).toBe(true);
    expect(
      Object.keys(DesktopEventContractRegistry).filter((channel) => channel.startsWith("project:")),
    ).toEqual([]);
  });

  it("rejects extra arguments, malformed identifiers, and widened results", () => {
    expect(ProjectInvokeContracts["project:list"].args.safeParse([]).success).toBe(true);
    expect(ProjectInvokeContracts["project:list"].args.safeParse(["unexpected"]).success).toBe(
      false,
    );
    expect(
      ProjectInvokeContracts["project:setPinned"].args.safeParse([
        { id: "project-1", pinned: true },
      ]).success,
    ).toBe(true);
    expect(
      ProjectInvokeContracts["project:rename"].args.safeParse([{ id: "", name: "Renamed" }])
        .success,
    ).toBe(false);
    expect(
      ProjectInvokeContracts["project:list"].result.safeParse([
        { ...project, rawCredential: "secret" },
      ]).success,
    ).toBe(false);
    expect(
      ProjectInvokeContracts["project:list"].result.safeParse([
        Object.fromEntries(Object.entries(project).filter(([key]) => key !== "pinned")),
      ]).success,
    ).toBe(false);
  });

  it("preserves nullable selection and scalar action results", () => {
    expect(ProjectInvokeContracts["project:addExisting"].result.parse(null)).toBeNull();
    expect(ProjectInvokeContracts["project:createScratch"].result.parse(project)).toEqual(project);
    expect(ProjectInvokeContracts["project:reveal"].result.parse(true)).toBe(true);
    expect(ProjectInvokeContracts["project:archiveTasks"].result.parse(0)).toBe(0);
  });
});

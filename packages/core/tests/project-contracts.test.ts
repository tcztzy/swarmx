import { describe, expect, it } from "vitest";
import { ProjectDataSchema } from "../src/project-contracts.js";

const project = {
  id: "project-1",
  name: "Project One",
  cwd: "/workspace/project-one",
  pinned: false,
  createdAt: "2026-08-13T00:00:00.000Z",
  updatedAt: "2026-08-13T00:00:00.000Z",
};

describe("Project contracts", () => {
  it("accepts canonical Project records and optional removal metadata", () => {
    expect(ProjectDataSchema.parse(project)).toEqual(project);
    expect(
      ProjectDataSchema.parse({ ...project, removedAt: "2026-08-13T01:00:00.000Z" }),
    ).toMatchObject({ removedAt: "2026-08-13T01:00:00.000Z" });
  });

  it("preserves the persisted contract while rejecting invalid field shapes", () => {
    expect(ProjectDataSchema.safeParse({ ...project, id: "" }).success).toBe(false);
    expect(ProjectDataSchema.safeParse({ ...project, pinned: "yes" }).success).toBe(false);
    expect(
      ProjectDataSchema.parse(
        Object.fromEntries(Object.entries(project).filter(([key]) => key !== "pinned")),
      ),
    ).toMatchObject({ pinned: false });
    expect(ProjectDataSchema.parse({ ...project, legacyMetadata: true })).toEqual(project);
  });
});

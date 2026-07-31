import { describe, expect, it } from "vitest";
import {
  type DiscoveredSession,
  type ProjectData,
  filterSessionGroups,
  mergeProjectsIntoSessionGroups,
  sortProjectSessionGroups,
} from "./session-navigation.js";

const PROJECT: ProjectData = {
  id: "project-1",
  name: "SwarmX",
  cwd: "/work/swarmx",
  pinned: true,
  createdAt: "2026-07-30T00:00:00.000Z",
  updatedAt: "2026-07-31T00:00:00.000Z",
};

const SESSION: DiscoveredSession = {
  id: "session-1",
  title: "Renderer refactor",
  projectId: PROJECT.id,
  cwd: PROJECT.cwd,
  harnessId: "codex",
  harnessLabel: "Codex",
  source: "local",
  updatedAt: "2026-07-31T01:00:00.000Z",
};

describe("Session and Project navigation", () => {
  it("merges sessions into persisted Projects and keeps pinned Projects first", () => {
    const otherProject = { ...PROJECT, id: "project-2", name: "Other", pinned: false };
    const groups = mergeProjectsIntoSessionGroups(
      [otherProject, PROJECT],
      [{ id: PROJECT.cwd, label: PROJECT.cwd, sessions: [SESSION] }],
    );

    expect(sortProjectSessionGroups(groups, "priority").map((group) => group.id)).toEqual([
      PROJECT.id,
      otherProject.id,
    ]);
    expect(groups.find((group) => group.id === PROJECT.id)?.sessions).toEqual([SESSION]);
  });

  it("filters by session title, harness, or path while retaining the Project group", () => {
    const groups = mergeProjectsIntoSessionGroups(
      [PROJECT],
      [{ id: PROJECT.cwd, label: PROJECT.cwd, sessions: [SESSION] }],
    );

    expect(filterSessionGroups(groups, "codex")).toEqual(groups);
    expect(filterSessionGroups(groups, "missing")).toEqual([]);
  });
});

import { describe, expect, it } from "vitest";
import {
  type DiscoveredSession,
  filterSessionGroups,
  mergeProjectsIntoSessionGroups,
  type ProjectData,
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

  it("labels sessions without a Project as Recents and keeps them after every Project group", () => {
    const discoveredProjectSession: DiscoveredSession = {
      ...SESSION,
      id: "session-2",
      projectId: undefined,
      cwd: "/work/discovered",
      updatedAt: "2026-08-02T01:00:00.000Z",
    };
    const recentSession: DiscoveredSession = {
      ...SESSION,
      id: "session-3",
      projectId: undefined,
      cwd: "",
      updatedAt: "2026-08-03T01:00:00.000Z",
    };
    const groups = mergeProjectsIntoSessionGroups(
      [PROJECT],
      [
        { id: PROJECT.cwd, label: PROJECT.cwd, sessions: [SESSION] },
        {
          id: "__no_project__",
          label: "No project",
          sessions: [recentSession],
        },
        {
          id: discoveredProjectSession.cwd,
          label: discoveredProjectSession.cwd,
          sessions: [discoveredProjectSession],
        },
      ],
    );

    expect(groups.find((group) => group.id === "__no_project__")?.label).toBe("Recents");
    for (const mode of ["priority", "last-updated", "manual"] as const) {
      expect(sortProjectSessionGroups(groups, mode).at(-1)?.id).toBe("__no_project__");
    }
  });

  it("keeps sessions reachable when their persisted Project no longer exists", () => {
    const orphanedSession: DiscoveredSession = {
      ...SESSION,
      id: "session-orphaned",
      title: "Orphaned project session",
      projectId: "removed-project",
      cwd: "/work/removed-project",
    };
    const groups = mergeProjectsIntoSessionGroups(
      [PROJECT],
      [
        { id: PROJECT.cwd, label: PROJECT.cwd, sessions: [SESSION] },
        {
          id: orphanedSession.cwd,
          label: orphanedSession.cwd,
          sessions: [orphanedSession],
        },
      ],
    );

    expect(groups.find((group) => group.id === orphanedSession.cwd)?.sessions).toEqual([
      orphanedSession,
    ]);
    expect(filterSessionGroups(groups, "orphaned").flatMap((group) => group.sessions)).toEqual([
      orphanedSession,
    ]);
  });
});

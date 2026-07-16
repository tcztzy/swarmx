import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  dismissProject,
  listProjects,
  registerDefaultProject,
  registerProject,
  removeProject,
  renameProject,
  setProjectPinned,
} from "../src/project.js";

const tempRoots: string[] = [];
const projectIds: string[] = [];

afterEach(async () => {
  for (const id of projectIds.splice(0)) removeProject(id);
  await Promise.all(tempRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("Project registry", () => {
  it("V322 registers one canonical realpath and persists its display name", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-project-"));
    tempRoots.push(root);

    const first = registerProject(root, "Research project");
    projectIds.push(first.id);
    const second = registerProject(root, "Ignored duplicate name");

    expect(second).toEqual(first);
    expect(listProjects()).toContainEqual(
      expect.objectContaining({
        id: first.id,
        name: "Research project",
        cwd: await realpath(root),
      }),
    );
  });

  it("rejects paths that are not directories", () => {
    expect(() => registerProject(path.join(tmpdir(), "missing-swarmx-project"))).toThrow(
      /must be a directory|ENOENT/,
    );
  });

  it("pins and renames projects while keeping pinned projects first", async () => {
    const firstRoot = await mkdtemp(path.join(tmpdir(), "swarmx-project-first-"));
    const secondRoot = await mkdtemp(path.join(tmpdir(), "swarmx-project-second-"));
    tempRoots.push(firstRoot, secondRoot);
    const first = registerProject(firstRoot, "Alpha");
    const second = registerProject(secondRoot, "Zulu");
    projectIds.push(first.id, second.id);

    expect(renameProject(second.id, "Beta")).toMatchObject({ name: "Beta" });
    expect(setProjectPinned(second.id, true)).toMatchObject({ pinned: true });
    const fixtureIds = new Set([first.id, second.id]);
    expect(
      listProjects()
        .map((project) => project.id)
        .filter((id) => fixtureIds.has(id)),
    ).toEqual([second.id, first.id]);
  });

  it("keeps a dismissed default project hidden until the user explicitly restores it", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-project-default-"));
    tempRoots.push(root);
    const project = registerDefaultProject(root);
    if (!project) throw new Error("default project was not registered");
    projectIds.push(project.id);

    expect(dismissProject(project.id)).toBe(true);
    expect(registerDefaultProject(root)).toBeNull();
    expect(listProjects()).not.toContainEqual(expect.objectContaining({ id: project.id }));

    expect(registerProject(root)).toMatchObject({ id: project.id });
    expect(listProjects()).toContainEqual(expect.objectContaining({ id: project.id }));
  });
});

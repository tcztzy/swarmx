import assert from "node:assert/strict";
import {
  chmod,
  mkdir,
  mkdtemp,
  readdir,
  readFile,
  rm,
  stat,
  symlink,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DefaultResourceLoader } from "@earendil-works/pi-coding-agent";
import { afterEach, expect, it, vi } from "vitest";
import { LearningResources, ResourceUpdateSchema } from "../src/host/learning-resources.js";

vi.mock("node:fs/promises", async (importOriginal) => {
  const original = await importOriginal<typeof import("node:fs/promises")>();
  return { ...original, chmod: vi.fn(original.chmod) };
});

const roots: string[] = [];
const signal = () => new AbortController().signal;
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function fixture(
  validator = `
import { readFile, appendFile } from 'node:fs/promises';
const content = await readFile(process.argv[2], 'utf8');
if (!content.includes('VALID')) process.exit(1);
await appendFile('validated.txt', content + '\\n');
`,
) {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-resources-"));
  roots.push(root);
  await mkdir(join(root, ".swarmx"));
  await writeFile(join(root, "AGENTS.md"), "Original instructions\n");
  await writeFile(join(root, ".swarmx", "validate.mjs"), validator);
  const resource = {
    id: "instructions",
    kind: "agent",
    path: "AGENTS.md",
    validate: [process.execPath, ".swarmx/validate.mjs"],
  };
  const registration = { resources: [resource] };
  const configure = () =>
    writeFile(join(root, ".swarmx", "learning.json"), JSON.stringify(registration));
  await configure();
  const resources = new LearningResources(root);
  const [snapshot] = await resources.snapshot(signal());
  assert.ok(snapshot);
  const request = {
    id: snapshot.id,
    expectedRevision: snapshot.expectedRevision,
    content: "VALID new instructions\n",
  };
  return { root, resources, snapshot, request, resource, registration, configure };
}

it("has no resources without explicit registration", async () => {
  const root = await mkdtemp(join(tmpdir(), "swarmx-learning-resources-"));
  roots.push(root);
  expect(await new LearningResources(root).snapshot(signal())).toEqual([]);
});

it("validates the candidate before replacing the registered file and safely repeats", async () => {
  const { root, resources, snapshot, request } = await fixture();
  expect(snapshot).toMatchObject({
    id: "instructions",
    kind: "agent",
    path: "AGENTS.md",
    content: "Original instructions\n",
    expectedRevision: expect.stringMatching(/^sha256:[a-f0-9]{64}$/u),
    configurationRevision: expect.stringMatching(/^sha256:[a-f0-9]{64}$/u),
  });
  expect(ResourceUpdateSchema.parse({ action: "update_resource", request })).toEqual({
    action: "update_resource",
    request,
  });
  const result = await resources.apply(request, snapshot, signal());
  expect(result).toMatchObject({ id: "instructions", changed: true, validation: "passed" });
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(request.content);
  expect(await readFile(join(root, "validated.txt"), "utf8")).toBe(`${request.content}\n`);
  const before = await stat(join(root, "AGENTS.md"));
  expect(await resources.apply(request, snapshot, signal())).toMatchObject({ changed: false });
  const after = await stat(join(root, "AGENTS.md"));
  expect(after.ino).toBe(before.ino);
  expect(after.mtimeMs).toBe(before.mtimeMs);
  expect(await readFile(join(root, "validated.txt"), "utf8")).toBe(
    `${request.content}\n${request.content}\n`,
  );
  expect((await readdir(root)).sort()).toEqual([".swarmx", "AGENTS.md", "validated.txt"]);
});

it("retains the original when the configured validator rejects the candidate", async () => {
  const { root, resources, snapshot, request } = await fixture();
  await expect(
    resources.apply({ ...request, content: "Rejected" }, snapshot, signal()),
  ).rejects.toThrow("validation");
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(snapshot.content);
  expect((await readdir(root)).sort()).toEqual([".swarmx", "AGENTS.md"]);
});

it("rejects changed resources, mismatched targets and changed registration", async () => {
  const { root, resources, snapshot, request, resource, configure } = await fixture();
  await expect(
    resources.apply({ ...request, id: "different" }, snapshot, signal()),
  ).rejects.toThrow();
  await writeFile(join(root, "AGENTS.md"), "Owner edit\n");
  await expect(resources.apply(request, snapshot, signal())).rejects.toThrow("revision");
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe("Owner edit\n");
  await writeFile(join(root, "AGENTS.md"), snapshot.content);
  resource.validate.push("changed");
  await configure();
  await expect(resources.apply(request, snapshot, signal())).rejects.toThrow("registration");
});

it.each(["../outside.md", "/tmp/outside.md", ".swarmx/learning.json"])(
  "rejects unauthorized resource path %s",
  async (path) => {
    const { resources, resource, configure } = await fixture();
    resource.path = path;
    await configure();
    await expect(resources.snapshot(signal())).rejects.toThrow();
  },
);

it("rejects symlinked resources and parent directories", async () => {
  const { root, resources, resource, configure } = await fixture();
  await symlink(join(root, "AGENTS.md"), join(root, "linked.md"));
  resource.path = "linked.md";
  await configure();
  await expect(resources.snapshot(signal())).rejects.toThrow("symbolic");
  await symlink(root, join(root, "linked-directory"));
  resource.path = "linked-directory/AGENTS.md";
  await configure();
  await expect(resources.snapshot(signal())).rejects.toThrow("symbolic");
});

it("rechecks the resource and registration after validation", async () => {
  const { root, resources, snapshot, request } = await fixture(`
import { writeFile } from 'node:fs/promises';
await writeFile('AGENTS.md', 'Concurrent edit');
`);
  await expect(resources.apply(request, snapshot, signal())).rejects.toThrow("revision");
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe("Concurrent edit");
  const second = await fixture(`
import { writeFile } from 'node:fs/promises';
await writeFile('.swarmx/learning.json', '{"resources":[]}');
`);
  await expect(second.resources.apply(second.request, second.snapshot, signal())).rejects.toThrow(
    "registration",
  );
  expect(await readFile(join(second.root, "AGENTS.md"), "utf8")).toBe(second.snapshot.content);
});

it("cancels an active validator without replacing the original", async () => {
  const { root, resources, snapshot, request } = await fixture(`
import { writeFile } from 'node:fs/promises';
await writeFile('started', 'yes');
setInterval(() => {}, 1000);
`);
  const controller = new AbortController();
  const work = resources.apply(request, snapshot, controller.signal);
  const rejected = expect(work).rejects.toThrow();
  await expect
    .poll(async () => readFile(join(root, "started"), "utf8").catch(() => ""))
    .toBe("yes");
  controller.abort(new Error("Stop validation"));
  await rejected;
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(snapshot.content);
  expect((await readdir(root)).sort()).toEqual([".swarmx", "AGENTS.md", "started"]);
});

it("rejects concurrent applications before launching a second validator", async () => {
  const { root, resources, snapshot, request } = await fixture(`
import { appendFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { setTimeout } from 'node:timers/promises';
await appendFile('started', 'validator\\n');
while (!existsSync('release')) await setTimeout(10);
`);
  const first = resources.apply(request, snapshot, signal());
  await expect
    .poll(async () => readFile(join(root, "started"), "utf8").catch(() => ""))
    .toBe("validator\n");
  let rejected: unknown;
  const second = resources.apply(request, snapshot, signal()).catch((error: unknown) => {
    rejected = error;
  });
  try {
    await Promise.resolve();
    expect(rejected).toBeInstanceOf(Error);
    expect(String(rejected)).toContain("already running");
  } finally {
    await writeFile(join(root, "release"), "release");
    await Promise.allSettled([first, second]);
  }
  expect(await readFile(join(root, "started"), "utf8")).toBe("validator\n");
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(request.content);
});

it("does not dispatch a validator when cancelled before application", async () => {
  const { root, resources, snapshot, request } = await fixture();
  await expect(resources.apply(request, snapshot, AbortSignal.abort())).rejects.toThrow();
  expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(snapshot.content);
  expect((await readdir(root)).sort()).toEqual([".swarmx", "AGENTS.md"]);
});

it.each(["abort", "edit"])(
  "respects %s after validation and before the final check",
  async (change) => {
    const { root, resources, snapshot, request } = await fixture();
    const native = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
    const reached = Promise.withResolvers<void>();
    const release = Promise.withResolvers<void>();
    vi.mocked(chmod).mockImplementationOnce(async (path, mode) => {
      await native.chmod(path, mode);
      reached.resolve();
      await release.promise;
    });
    const controller = new AbortController();
    const work = resources.apply(request, snapshot, controller.signal);
    const rejected = expect(work).rejects.toThrow();
    await reached.promise;
    if (change === "abort") controller.abort(new Error("Stopped before commit"));
    else await writeFile(join(root, "AGENTS.md"), "Owner changed the instructions");
    release.resolve();
    await rejected;
    expect(await readFile(join(root, "validated.txt"), "utf8")).toBe(`${request.content}\n`);
    expect(await readFile(join(root, "AGENTS.md"), "utf8")).toBe(
      change === "abort" ? snapshot.content : "Owner changed the instructions",
    );
  },
);

it("rejects oversized snapshots without truncating registered content", async () => {
  const { root, resources } = await fixture();
  await writeFile(join(root, "AGENTS.md"), "a".repeat(32_001));
  await expect(resources.snapshot(signal())).rejects.toThrow("32,000");
});

it("rejects a validator that mutates or redirects the candidate", async () => {
  const first = await fixture(`
import { writeFile } from 'node:fs/promises';
await writeFile(process.argv[2], 'Different content');
`);
  await expect(first.resources.apply(first.request, first.snapshot, signal())).rejects.toThrow(
    "candidate",
  );
  expect(await readFile(join(first.root, "AGENTS.md"), "utf8")).toBe(first.snapshot.content);
  const second = await fixture(`
import { unlink, symlink } from 'node:fs/promises';
await unlink(process.argv[2]);
await symlink('AGENTS.md', process.argv[2]);
`);
  await expect(second.resources.apply(second.request, second.snapshot, signal())).rejects.toThrow(
    "symbolic",
  );
  expect(await readFile(join(second.root, "AGENTS.md"), "utf8")).toBe(second.snapshot.content);
});

it("the native Pi loader sees updated project agent instructions and skills on reload", async () => {
  const { root, resources, resource, registration, configure } = await fixture();
  const skillPath = ".pi/skills/research/SKILL.md";
  await mkdir(join(root, ".pi/skills/research"), { recursive: true });
  await writeFile(
    join(root, skillPath),
    "---\nname: research\ndescription: Original method\n---\nRead evidence.\n",
  );
  registration.resources.push({ ...resource, id: "research", kind: "skill", path: skillPath });
  await configure();
  const loader = new DefaultResourceLoader({
    cwd: root,
    agentDir: join(root, "native-user"),
    noExtensions: true,
    noPromptTemplates: true,
    noThemes: true,
  });
  await loader.reload();
  expect(loader.getSkills().skills.find(({ name }) => name === "research")?.description).toBe(
    "Original method",
  );
  for (const snapshot of await resources.snapshot(signal())) {
    const content =
      snapshot.kind === "skill"
        ? "---\nname: research\ndescription: VALID updated method\n---\nCheck the original evidence.\n"
        : "VALID updated agent instructions\n";
    await resources.apply(
      { id: snapshot.id, expectedRevision: snapshot.expectedRevision, content },
      snapshot,
      signal(),
    );
  }
  await loader.reload();
  expect(loader.getSkills().diagnostics).toEqual([]);
  expect(loader.getSkills().skills.find(({ name }) => name === "research")?.description).toBe(
    "VALID updated method",
  );
  expect(loader.getAgentsFiles().agentsFiles).toContainEqual({
    path: join(root, "AGENTS.md"),
    content: "VALID updated agent instructions\n",
  });
});

import { chmod, mkdir, mkdtemp, open, readdir, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { writePrivateJsonFile } from "./private-json-file.js";

const temporaryRoots: string[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true })));
});

describe("writePrivateJsonFile", () => {
  it("writes exact JSON through private parent and file modes", async () => {
    const root = await temporaryRoot();
    const filePath = join(root, "private", "document.json");

    await writePrivateJsonFile(filePath, { name: "SwarmX", enabled: true });

    expect(await readFile(filePath, "utf8")).toBe('{\n  "name": "SwarmX",\n  "enabled": true\n}\n');
    if (process.platform !== "win32") {
      expect((await stat(dirname(filePath))).mode & 0o777).toBe(0o700);
      expect((await stat(filePath)).mode & 0o777).toBe(0o600);
    }
  });

  it("uses collision-resistant temporary files for concurrent replacement", async () => {
    const root = await temporaryRoot();
    const filePath = join(root, "document.json");
    vi.spyOn(Date, "now").mockReturnValue(1);

    await Promise.all([
      writePrivateJsonFile(filePath, { writer: "first", values: [1, 2, 3] }),
      writePrivateJsonFile(filePath, { writer: "second", values: [4, 5, 6] }),
    ]);

    expect([
      { writer: "first", values: [1, 2, 3] },
      { writer: "second", values: [4, 5, 6] },
    ]).toContainEqual(JSON.parse(await readFile(filePath, "utf8")));
    expect((await readdir(root)).filter((name) => name.includes(".tmp-"))).toEqual([]);
  });

  it("replaces an existing permissive target with a private file", async () => {
    const root = await temporaryRoot();
    const filePath = join(root, "document.json");
    await writePrivateJsonFile(filePath, { version: 1 });
    if (process.platform !== "win32") await chmod(filePath, 0o644);

    await writePrivateJsonFile(filePath, { version: 2 });

    expect(JSON.parse(await readFile(filePath, "utf8"))).toEqual({ version: 2 });
    if (process.platform !== "win32") {
      expect((await stat(filePath)).mode & 0o777).toBe(0o600);
    }
  });

  it("removes only its own temporary file when replacement fails", async () => {
    const root = await temporaryRoot();
    const targetDirectory = join(root, "document.json");
    await mkdir(targetDirectory);

    await expect(writePrivateJsonFile(targetDirectory, { version: 1 })).rejects.toThrow();

    expect((await stat(targetDirectory)).isDirectory()).toBe(true);
    expect((await readdir(root)).filter((name) => name.includes(".tmp-"))).toEqual([]);
  });

  it("tolerates only known unsupported directory fsync errors", async () => {
    const root = await temporaryRoot();
    const sync = await mockDirectorySyncFailure(root, "EINVAL");

    await expect(writePrivateJsonFile(join(root, "document.json"), { version: 1 })).resolves.toBe(
      undefined,
    );
    expect(sync).toHaveBeenCalledTimes(2);
  });

  it("reports unexpected directory fsync errors after committing a complete file", async () => {
    const root = await temporaryRoot();
    const filePath = join(root, "document.json");
    const sync = await mockDirectorySyncFailure(root, "EIO");

    await expect(writePrivateJsonFile(filePath, { version: 1 })).rejects.toThrow(
      "directory sync failed",
    );

    expect(JSON.parse(await readFile(filePath, "utf8"))).toEqual({ version: 1 });
    expect(sync).toHaveBeenCalledTimes(2);
  });
});

async function temporaryRoot(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "swarmx-private-json-"));
  temporaryRoots.push(root);
  return root;
}

async function mockDirectorySyncFailure(root: string, code: string) {
  const probe = await open(join(root, "probe"), "w");
  const prototype = Object.getPrototypeOf(probe) as Awaited<ReturnType<typeof open>>;
  await probe.close();
  await rm(join(root, "probe"));
  let calls = 0;
  return vi.spyOn(prototype, "sync").mockImplementation(async () => {
    calls += 1;
    if (calls === 2) {
      throw Object.assign(new Error("directory sync failed"), { code });
    }
  });
}

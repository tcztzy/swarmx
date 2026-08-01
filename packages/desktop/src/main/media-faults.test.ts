import { writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DesktopMediaImport } from "./media.js";

const fsMocks = vi.hoisted(() => ({
  copyFile: vi.fn(),
  stat: vi.fn(),
  writeFile: vi.fn(),
}));

vi.mock("node:fs/promises", async () => {
  const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
  return {
    ...actual,
    copyFile: fsMocks.copyFile,
    stat: fsMocks.stat,
    writeFile: fsMocks.writeFile,
  };
});

import type { MediaAttachment } from "@swarmx/core";
import { DesktopMediaService, mediaProtocolUrl } from "./media.js";

const temporaryDirectories = new Set<string>();
let realFs: typeof import("node:fs/promises");

beforeEach(async () => {
  realFs = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
  resetFileMocks();
});

afterEach(async () => {
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("DesktopMediaService filesystem faults", () => {
  it("rejects a source that disappears after content inspection", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "notes.md");
    writeFileSync(source, "notes");
    const service = new DesktopMediaService(path.join(root, "media"));

    fsMocks.stat.mockImplementationOnce((...args) => realFs.stat(...args));
    fsMocks.stat.mockRejectedValueOnce(new Error("source disappeared"));

    await expect(service.importPaths([source])).rejects.toThrow(/Attachment changed/i);
  });

  it("accepts copy and write races when another importer wins", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "notes.md");
    writeFileSync(source, "notes");
    const service = new DesktopMediaService(path.join(root, "media"));
    const copy = realFs.copyFile;
    fsMocks.copyFile.mockImplementationOnce(async (...args) => {
      await copy(...args);
      throw Object.assign(new Error("already exists"), { code: "EEXIST" });
    });

    await expect(service.importPaths([source])).resolves.toHaveLength(1);

    resetFileMocks();
    const bytes: DesktopMediaImport = {
      name: "notes.txt",
      bytes: new TextEncoder().encode("notes"),
    };
    const write = realFs.writeFile;
    fsMocks.writeFile.mockImplementationOnce(async (...args) => {
      await write(...args);
      throw Object.assign(new Error("already exists"), { code: "EEXIST" });
    });

    await expect(service.importBytes([bytes])).resolves.toHaveLength(1);
  });

  it("surfaces a race when the competing file is not present anymore", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "notes.md");
    writeFileSync(source, "notes");
    const service = new DesktopMediaService(path.join(root, "media"));
    fsMocks.copyFile.mockRejectedValueOnce(
      Object.assign(new Error("already exists"), { code: "EEXIST" }),
    );

    await expect(service.importPaths([source])).rejects.toThrow(/already exists/i);

    resetFileMocks();
    fsMocks.writeFile.mockRejectedValueOnce(
      Object.assign(new Error("already exists"), { code: "EEXIST" }),
    );
    await expect(
      service.importBytes([{ name: "notes.txt", bytes: new TextEncoder().encode("notes") }]),
    ).rejects.toThrow(/already exists/i);
  });

  it("cleans up a newly copied file when validation fails", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "notes.md");
    writeFileSync(source, "notes");
    const service = new DesktopMediaService(path.join(root, "media"));
    for (let index = 0; index < 3; index += 1) {
      fsMocks.stat.mockImplementationOnce((...args) => realFs.stat(...args));
    }
    fsMocks.stat.mockRejectedValueOnce(new Error("stored file disappeared"));

    await expect(service.importPaths([source])).rejects.toThrow(/Media preview is unavailable/i);
  });

  it("handles missing files during stored-path and receipt validation", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));
    const [attachment] = await service.importBytes([
      { name: "notes.txt", bytes: new TextEncoder().encode("notes") },
    ]);
    const stored = attachment as MediaAttachment;

    fsMocks.stat.mockClear();
    fsMocks.stat.mockRejectedValueOnce(new Error("file disappeared"));
    await expect(service.validatedStoredPath(stored)).rejects.toThrow(
      /Media preview is unavailable/i,
    );

    resetFileMocks();
    fsMocks.stat.mockImplementationOnce((...args) => realFs.stat(...args));
    fsMocks.stat.mockRejectedValueOnce(new Error("file disappeared after inspection"));
    await expect(service.validatedStoredPath(stored)).rejects.toThrow(
      /Managed media changed after it was imported/i,
    );

    resetFileMocks();
    await service.validatedStoredPath(stored, true);
    fsMocks.stat.mockClear();
    fsMocks.stat.mockRejectedValueOnce(new Error("receipt file disappeared"));
    await expect(service.resolveProtocolUrl(mediaProtocolUrl(stored))).resolves.toMatch(
      /notes\.txt$/,
    );
  });
});

function resetFileMocks(): void {
  fsMocks.copyFile.mockReset();
  fsMocks.stat.mockReset();
  fsMocks.writeFile.mockReset();
  fsMocks.copyFile.mockImplementation((...args) => realFs.copyFile(...args));
  fsMocks.stat.mockImplementation((...args) => realFs.stat(...args));
  fsMocks.writeFile.mockImplementation((...args) => realFs.writeFile(...args));
}

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-media-fault-test-"));
  temporaryDirectories.add(directory);
  return directory;
}

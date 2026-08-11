import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { MediaAttachment } from "@swarmx/core";
import { afterEach, describe, expect, it, vi } from "vitest";

const streamSpies = vi.hoisted(() => ({
  createReadStream: vi.fn(),
}));

vi.mock("node:fs", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs")>();
  return {
    ...actual,
    createReadStream: (...args: Parameters<typeof actual.createReadStream>) => {
      streamSpies.createReadStream(...args);
      return actual.createReadStream(...args);
    },
  };
});

import { DesktopMediaService } from "./media.js";

const temporaryDirectories = new Set<string>();

afterEach(async () => {
  streamSpies.createReadStream.mockClear();
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("DesktopMediaService preview validation", () => {
  it("reuses one content validation for the immediate media protocol request", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-media-preview-hash-"));
    temporaryDirectories.add(root);
    const service = new DesktopMediaService(path.join(root, "media"));
    const [attachment] = await service.importBytes([
      {
        name: "diagram.png",
        bytes: Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1]),
      },
    ]);

    const preview = await service.preview(attachment as MediaAttachment);
    await service.resolveProtocolUrl(preview.previewUrl as string);

    expect(streamSpies.createReadStream).toHaveBeenCalledTimes(1);
  });

  it("consumes no stale receipt after the managed file identity changes", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "swarmx-media-preview-hash-"));
    temporaryDirectories.add(root);
    const service = new DesktopMediaService(path.join(root, "media"));
    const [attachment] = await service.importBytes([
      {
        name: "diagram.png",
        bytes: Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1]),
      },
    ]);
    const preview = await service.preview(attachment as MediaAttachment);
    streamSpies.createReadStream.mockClear();

    await writeFile(
      fileURLToPath((attachment as MediaAttachment).uri),
      Uint8Array.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 2]),
    );

    await expect(service.resolveProtocolUrl(preview.previewUrl as string)).rejects.toThrow(
      /changed after it was imported/i,
    );
    expect(streamSpies.createReadStream).toHaveBeenCalledTimes(1);
  });
});

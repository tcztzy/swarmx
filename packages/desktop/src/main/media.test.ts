import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  MAX_MEDIA_ATTACHMENT_BYTES,
  MAX_MEDIA_ATTACHMENTS,
  MAX_MEDIA_TURN_BYTES,
  type MediaAttachment,
} from "@swarmx/core";
import { afterEach, describe, expect, it } from "vitest";
import { DesktopMediaService } from "./media.js";

const temporaryDirectories = new Set<string>();

afterEach(async () => {
  await Promise.all(
    [...temporaryDirectories].map((directory) => rm(directory, { recursive: true, force: true })),
  );
  temporaryDirectories.clear();
});

describe("DesktopMediaService", () => {
  it("imports browser bytes into a content-addressed store and previews text", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));

    const [attachment] = await service.importBytes([
      {
        name: "notes.md",
        mimeType: "text/markdown",
        bytes: new TextEncoder().encode("# Notes\n"),
      },
    ]);

    expect(attachment).toMatchObject({
      name: "notes.md",
      kind: "text",
      mimeType: "text/markdown",
      sizeBytes: 8,
      source: "user",
    });
    expect(attachment?.uri).toMatch(/^file:/);
    await expect(service.preview(attachment as MediaAttachment)).resolves.toMatchObject({
      status: "available",
      text: "# Notes\n",
    });
  });

  it("copies selected files so later source changes do not mutate a sent attachment", async () => {
    const root = await temporaryDirectory();
    const source = path.join(root, "diagram.png");
    const firstBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1]);
    await writeFile(source, firstBytes);
    const service = new DesktopMediaService(path.join(root, "media"));

    const [attachment] = await service.importPaths([source]);
    await writeFile(source, Buffer.from("changed"));

    expect(attachment).toMatchObject({ kind: "image", mimeType: "image/png" });
    await expect(service.validatedStoredPath(attachment as MediaAttachment)).resolves.toMatch(
      /diagram\.png$/,
    );
    await expect(service.preview(attachment as MediaAttachment)).resolves.toMatchObject({
      status: "available",
      previewUrl: expect.stringMatching(/^swarmx-media:\/\/asset\//),
    });
  });

  it("rejects same-size changes inside the content-addressed store", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));
    const [attachment] = await service.importBytes([
      {
        name: "notes.md",
        bytes: new TextEncoder().encode("# Notes\n"),
      },
    ]);

    await writeFile(fileURLToPath((attachment as MediaAttachment).uri), "# Evil!\n");

    await expect(service.validatedStoredPath(attachment as MediaAttachment)).rejects.toThrow(
      /changed after it was imported/i,
    );
    await expect(service.preview(attachment as MediaAttachment)).resolves.toMatchObject({
      status: "unavailable",
      error: expect.stringMatching(/changed after it was imported/i),
    });
  });

  it("does not trust a renderer-supplied MIME type without a signature or extension", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));

    const [attachment] = await service.importBytes([
      {
        name: "payload.bin",
        mimeType: "image/png",
        bytes: new TextEncoder().encode("not an image"),
      },
    ]);

    expect(attachment).toMatchObject({
      kind: "file",
      mimeType: "application/octet-stream",
    });
  });

  it("rejects untrusted paths and protocol traversal outside the managed store", async () => {
    const root = await temporaryDirectory();
    const outside = path.join(root, "outside.txt");
    await writeFile(outside, "private");
    const service = new DesktopMediaService(path.join(root, "media"));
    const forged: MediaAttachment = {
      id: "forged",
      name: "outside.txt",
      kind: "text",
      mimeType: "text/plain",
      sizeBytes: 7,
      uri: pathToFileURL(outside).href,
      source: "user",
    };

    await expect(service.validatedStoredPath(forged)).rejects.toThrow(/outside the managed/i);
    await expect(
      service.resolveProtocolUrl("swarmx-media://asset/../../outside.txt"),
    ).rejects.toThrow(/invalid media preview/i);
  });

  it("enforces per-turn attachment count before importing bytes", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));

    await expect(
      service.importBytes(
        Array.from({ length: 21 }, (_, index) => ({
          name: `${index}.txt`,
          bytes: new Uint8Array(),
        })),
      ),
    ).rejects.toThrow(/at most 20 files/i);
  });

  it("V558 applies count and byte limits to existing plus newly imported attachments", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));
    const existing = Array.from({ length: MAX_MEDIA_ATTACHMENTS }, (_, index) => ({
      id: `existing-${index}`,
      name: `existing-${index}.txt`,
      kind: "text" as const,
      mimeType: "text/plain",
      sizeBytes: 1,
      uri: pathToFileURL(path.join(root, `existing-${index}.txt`)).href,
      source: "user" as const,
    }));
    await expect(
      service.importBytes([{ name: "one-too-many.txt", bytes: new Uint8Array() }], existing),
    ).rejects.toThrow(/at most 20 files/i);
    await expect(
      service.importBytes(
        [{ name: "over-total.bin", bytes: new Uint8Array([1]) }],
        Array.from({ length: 5 }, (_, index) => ({
          ...existing[index],
          sizeBytes: MAX_MEDIA_ATTACHMENT_BYTES,
        })) as MediaAttachment[],
      ),
    ).rejects.toThrow(/500 MiB or less/i);
  });

  it("enforces per-file and aggregate byte limits before copying browser data", async () => {
    const root = await temporaryDirectory();
    const service = new DesktopMediaService(path.join(root, "media"));

    await expect(
      service.importBytes([
        {
          name: "oversized.bin",
          bytes: { byteLength: MAX_MEDIA_ATTACHMENT_BYTES + 1 } as Uint8Array,
        },
      ]),
    ).rejects.toThrow(/100 MiB or smaller/i);

    const chunkSize = Math.floor(MAX_MEDIA_TURN_BYTES / 6) + 1;
    await expect(
      service.importBytes(
        Array.from({ length: 6 }, (_, index) => ({
          name: `chunk-${index}.bin`,
          bytes: { byteLength: chunkSize } as Uint8Array,
        })),
      ),
    ).rejects.toThrow(/500 MiB or less/i);
  });
});

async function temporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(path.join(tmpdir(), "swarmx-desktop-media-test-"));
  temporaryDirectories.add(directory);
  return directory;
}

import { createHash, randomUUID } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { copyFile, mkdir, open, realpath, rm, stat, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { MediaAttachment } from "@swarmx/core";
import {
  detectMediaMimeType,
  MAX_MEDIA_ATTACHMENT_BYTES,
  MAX_MEDIA_ATTACHMENTS,
  MAX_MEDIA_TURN_BYTES,
  MEDIA_SNIFF_BYTES,
  MediaAttachmentSchema,
  mediaKindFromMimeType,
  validateMediaAttachments,
} from "@swarmx/core";

// Duplicated in shared/desktop-api.ts: tsconfig.main-lib rootDir forbids importing ../shared.
export interface DesktopMediaImport {
  name: string;
  mimeType?: string;
  bytes: Uint8Array;
}

export interface DesktopMediaPreview {
  status: "available" | "unavailable";
  attachment: MediaAttachment;
  previewUrl?: string;
  text?: string;
  error?: string;
}

const MAX_TEXT_PREVIEW_BYTES = 512 * 1024;
const HASH_PATTERN = /^[a-f0-9]{64}$/;
const MAX_PREVIEW_RECEIPTS = 64;
const PREVIEW_RECEIPT_TTL_MS = 30_000;

interface StoredFileFingerprint {
  dev: number;
  ino: number;
  size: number;
  mtimeMs: number;
  ctimeMs: number;
}

interface PreviewReceipt {
  expiresAt: number;
  fingerprint: StoredFileFingerprint;
}

export class DesktopMediaService {
  readonly root: string;
  readonly #previewReceipts = new Map<string, PreviewReceipt>();

  constructor(root = path.join(homedir(), ".swarmx", "media")) {
    this.root = path.resolve(root);
  }

  async importPaths(
    paths: readonly string[],
    existingAttachments: readonly MediaAttachment[] = [],
  ): Promise<MediaAttachment[]> {
    const existing = validateMediaAttachments(existingAttachments);
    validateImportCount(existing.length + paths.length);
    const infos = await Promise.all(
      paths.map(async (filePath) => {
        const absolute = path.resolve(filePath);
        const info = await stat(absolute).catch(() => null);
        if (!info?.isFile()) throw new Error(`Attachment is not a readable file: ${filePath}`);
        return { absolute, info };
      }),
    );
    validateImportBytes([
      ...existing.map((attachment) => attachment.sizeBytes),
      ...infos.map(({ info }) => info.size),
    ]);
    return Promise.all(
      infos.map(({ absolute, info }) =>
        this.importFile(absolute, info.size, Math.floor(info.mtimeMs)),
      ),
    );
  }

  async importBytes(
    files: readonly DesktopMediaImport[],
    existingAttachments: readonly MediaAttachment[] = [],
  ): Promise<MediaAttachment[]> {
    const existing = validateMediaAttachments(existingAttachments);
    validateImportCount(existing.length + files.length);
    validateImportBytes([
      ...existing.map((attachment) => attachment.sizeBytes),
      ...files.map((file) => file.bytes.byteLength),
    ]);
    return Promise.all(
      files.map(async (file) => {
        if (!file.name.trim()) throw new Error("Attachment name cannot be empty.");
        const bytes = Buffer.from(file.bytes);
        const mimeType =
          detectMediaMimeType(bytes.subarray(0, MEDIA_SNIFF_BYTES), file.name) ??
          "application/octet-stream";
        const digest = createHash("sha256").update(bytes).digest("hex");
        const target = await this.writeStoredFile(digest, file.name, bytes);
        return attachmentRecord({
          digest,
          name: file.name,
          mimeType,
          sizeBytes: bytes.byteLength,
          target,
          lastModifiedMs: Date.now(),
        });
      }),
    );
  }

  async preview(input: MediaAttachment): Promise<DesktopMediaPreview> {
    const attachment = MediaAttachmentSchema.parse(input);
    try {
      const hasProtocolPreview = attachment.kind !== "text" && attachment.kind !== "file";
      const filePath = await this.validatedStoredPath(attachment, hasProtocolPreview);
      if (attachment.kind === "text") {
        const previewSize = Math.min(attachment.sizeBytes, MAX_TEXT_PREVIEW_BYTES);
        const bytes = Buffer.alloc(previewSize);
        const handle = await open(filePath, "r");
        const bytesRead = await handle
          .read(bytes, 0, previewSize, 0)
          .then((result) => result.bytesRead)
          .finally(() => handle.close());
        const truncated = attachment.sizeBytes > MAX_TEXT_PREVIEW_BYTES;
        const text = bytes.subarray(0, bytesRead).toString("utf8");
        return {
          status: "available",
          attachment,
          text: truncated ? `${text}\n\n[Preview truncated]` : text,
        };
      }
      return {
        status: "available",
        attachment,
        ...(attachment.kind === "file" ? {} : { previewUrl: mediaProtocolUrl(attachment) }),
      };
    } catch (error) {
      return {
        status: "unavailable",
        attachment,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  }

  async validatedStoredPath(input: MediaAttachment, rememberForProtocol = false): Promise<string> {
    const attachment = MediaAttachmentSchema.parse(input);
    const filePath = fileURLToPath(attachment.uri);
    const resolved = path.resolve(filePath);
    if (!isInside(this.root, resolved)) {
      throw new Error("Attachment is outside the managed media store.");
    }
    const digest = path.basename(path.dirname(resolved));
    if (!HASH_PATTERN.test(digest)) {
      throw new Error("Attachment media id is invalid.");
    }
    const inspected = await this.inspectStoredPath(resolved, digest, attachment.sizeBytes).catch(
      (error) => {
        const detail = error instanceof Error ? error.message : String(error);
        throw new Error(`Attachment "${attachment.name}" is unavailable: ${detail}`);
      },
    );
    const detectedMime =
      detectMediaMimeType(inspected.head, attachment.name) ?? "application/octet-stream";
    if (detectedMime !== attachment.mimeType) {
      throw new Error(`Attachment "${attachment.name}" no longer matches its media type.`);
    }
    if (rememberForProtocol) {
      this.rememberPreviewReceipt(inspected.filePath, inspected.fingerprint);
    }
    return inspected.filePath;
  }

  async resolveProtocolUrl(url: string): Promise<string> {
    const parsed = new URL(url);
    if (parsed.protocol !== "swarmx-media:" || parsed.hostname !== "asset") {
      throw new Error("Invalid media preview URL.");
    }
    const segments = parsed.pathname
      .split("/")
      .filter(Boolean)
      .map((segment) => decodeURIComponent(segment));
    const digest = segments[0];
    const name = segments[1];
    if (
      segments.length !== 2 ||
      !digest ||
      !HASH_PATTERN.test(digest) ||
      !name ||
      name !== safeFilename(name)
    ) {
      throw new Error("Invalid media preview path.");
    }
    const candidate = path.join(this.root, digest, name);
    if (!isInside(this.root, candidate)) throw new Error("Invalid media preview path.");
    const receiptPath = await this.consumePreviewReceipt(candidate);
    if (receiptPath) return receiptPath;
    return (await this.inspectStoredPath(candidate, digest)).filePath;
  }

  private async importFile(
    filePath: string,
    sizeBytes: number,
    lastModifiedMs: number,
  ): Promise<MediaAttachment> {
    const inspected = await inspectFileContent(filePath);
    const afterRead = await stat(filePath).catch(() => null);
    if (
      !afterRead?.isFile() ||
      afterRead.size !== sizeBytes ||
      Math.floor(afterRead.mtimeMs) !== lastModifiedMs
    ) {
      throw new Error(`Attachment changed: ${filePath}`);
    }
    const name = path.basename(filePath);
    const mimeType = detectMediaMimeType(inspected.head, name) ?? "application/octet-stream";
    const { target, created } = await this.copyStoredFile(inspected.digest, name, filePath);
    try {
      await this.inspectStoredPath(target, inspected.digest, sizeBytes);
    } catch (error) {
      if (created) await rm(target, { force: true });
      throw error;
    }
    return attachmentRecord({
      digest: inspected.digest,
      name,
      mimeType,
      sizeBytes,
      target,
      lastModifiedMs,
    });
  }

  private async copyStoredFile(
    digest: string,
    name: string,
    source: string,
  ): Promise<{ target: string; created: boolean }> {
    const target = await this.storedTarget(digest, name);
    const existing = await stat(target).catch(() => null);
    let created = false;
    if (!existing?.isFile()) {
      await copyFile(source, target, constants.COPYFILE_EXCL)
        .then(() => {
          created = true;
        })
        .catch(async (error: NodeJS.ErrnoException) => {
          if (error.code !== "EEXIST" || !(await stat(target).catch(() => null))?.isFile()) {
            throw error;
          }
        });
    }
    return { target, created };
  }

  private async writeStoredFile(digest: string, name: string, bytes: Buffer): Promise<string> {
    const target = await this.storedTarget(digest, name);
    const existing = await stat(target).catch(() => null);
    if (!existing?.isFile()) {
      await writeFile(target, bytes, { flag: "wx" }).catch(async (error: NodeJS.ErrnoException) => {
        if (error.code !== "EEXIST" || !(await stat(target).catch(() => null))?.isFile()) {
          throw error;
        }
      });
    }
    return target;
  }

  private async storedTarget(digest: string, name: string): Promise<string> {
    const directory = path.join(this.root, digest);
    await mkdir(directory, { recursive: true });
    return path.join(directory, safeFilename(name));
  }

  private async inspectStoredPath(
    candidate: string,
    expectedDigest: string,
    expectedSize?: number,
  ): Promise<{ filePath: string; fingerprint: StoredFileFingerprint; head: Buffer }> {
    const canonicalRoot = await realpath(this.root);
    const canonicalFile = await realpath(candidate).catch(() => null);
    if (!canonicalFile || !isInside(canonicalRoot, canonicalFile)) {
      throw new Error("Attachment resolved outside the managed media store.");
    }
    const before = await stat(canonicalFile).catch(() => null);
    if (
      !before?.isFile() ||
      before.size > MAX_MEDIA_ATTACHMENT_BYTES ||
      (expectedSize !== undefined && before.size !== expectedSize)
    ) {
      throw new Error("Media preview is unavailable.");
    }
    const inspected = await inspectFileContent(canonicalFile);
    const after = await stat(canonicalFile).catch(() => null);
    if (
      !after?.isFile() ||
      !sameFingerprint(fileFingerprint(after), fileFingerprint(before)) ||
      inspected.digest !== expectedDigest
    ) {
      throw new Error("Managed media changed after it was imported.");
    }
    return {
      filePath: canonicalFile,
      fingerprint: fileFingerprint(after),
      head: inspected.head,
    };
  }

  private rememberPreviewReceipt(filePath: string, fingerprint: StoredFileFingerprint): void {
    const now = Date.now();
    for (const [key, receipt] of this.#previewReceipts) {
      if (receipt.expiresAt <= now) this.#previewReceipts.delete(key);
    }
    this.#previewReceipts.delete(filePath);
    while (this.#previewReceipts.size >= MAX_PREVIEW_RECEIPTS) {
      const oldest = this.#previewReceipts.keys().next().value;
      if (typeof oldest !== "string") break;
      this.#previewReceipts.delete(oldest);
    }
    this.#previewReceipts.set(filePath, {
      expiresAt: now + PREVIEW_RECEIPT_TTL_MS,
      fingerprint,
    });
  }

  private async consumePreviewReceipt(candidate: string): Promise<string | null> {
    const canonicalRoot = await realpath(this.root);
    const canonicalFile = await realpath(candidate).catch(() => null);
    if (!canonicalFile || !isInside(canonicalRoot, canonicalFile)) return null;
    const receipt = this.#previewReceipts.get(canonicalFile);
    if (!receipt) return null;
    this.#previewReceipts.delete(canonicalFile);
    if (receipt.expiresAt <= Date.now()) return null;
    const info = await stat(canonicalFile).catch(() => null);
    if (
      !info?.isFile() ||
      info.size > MAX_MEDIA_ATTACHMENT_BYTES ||
      !sameFingerprint(fileFingerprint(info), receipt.fingerprint)
    ) {
      return null;
    }
    return canonicalFile;
  }
}

export function mediaProtocolUrl(attachment: MediaAttachment): string {
  const filePath = fileURLToPath(attachment.uri);
  const digest = path.basename(path.dirname(filePath));
  const name = path.basename(filePath);
  if (!HASH_PATTERN.test(digest)) throw new Error("Attachment media id is invalid.");
  return `swarmx-media://asset/${digest}/${encodeURIComponent(name)}`;
}

function attachmentRecord(input: {
  digest: string;
  name: string;
  mimeType: string;
  sizeBytes: number;
  target: string;
  lastModifiedMs: number;
}): MediaAttachment {
  const storedName = path.basename(input.target);
  return MediaAttachmentSchema.parse({
    id: `${input.digest}:${createHash("sha256").update(storedName).digest("hex").slice(0, 12)}`,
    name: input.name.slice(0, 512),
    kind: mediaKindFromMimeType(input.mimeType),
    mimeType: input.mimeType,
    sizeBytes: input.sizeBytes,
    uri: pathToFileURL(input.target).href,
    source: "user",
    lastModifiedMs: input.lastModifiedMs,
  });
}

function validateImportCount(count: number): void {
  if (count > MAX_MEDIA_ATTACHMENTS) {
    throw new Error(`You can attach at most ${MAX_MEDIA_ATTACHMENTS} files at once.`);
  }
}

function validateImportBytes(sizes: readonly number[]): void {
  for (const size of sizes) {
    if (size > MAX_MEDIA_ATTACHMENT_BYTES) {
      throw new Error("Each attachment must be 100 MiB or smaller.");
    }
  }
  if (sizes.reduce((sum, size) => sum + size, 0) > MAX_MEDIA_TURN_BYTES) {
    throw new Error("Attachments must total 500 MiB or less.");
  }
}

function safeFilename(value: string): string {
  const basename = path
    .basename(value)
    .replace(/[^\p{L}\p{N}._ -]+/gu, "_")
    .trim();
  return (basename || `attachment-${randomUUID()}`).slice(0, 180);
}

function isInside(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative.length > 0 && !relative.startsWith("..") && !path.isAbsolute(relative);
}

async function inspectFileContent(filePath: string): Promise<{ digest: string; head: Buffer }> {
  const hash = createHash("sha256");
  let head = Buffer.alloc(0);
  for await (const chunk of createReadStream(filePath)) {
    const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    hash.update(bytes);
    if (head.byteLength < MEDIA_SNIFF_BYTES) {
      head = Buffer.concat([head, bytes.subarray(0, MEDIA_SNIFF_BYTES - head.byteLength)]);
    }
  }
  return { digest: hash.digest("hex"), head };
}

function fileFingerprint(info: {
  dev: number;
  ino: number;
  size: number;
  mtimeMs: number;
  ctimeMs: number;
}): StoredFileFingerprint {
  return {
    dev: info.dev,
    ino: info.ino,
    size: info.size,
    mtimeMs: info.mtimeMs,
    ctimeMs: info.ctimeMs,
  };
}

function sameFingerprint(left: StoredFileFingerprint, right: StoredFileFingerprint): boolean {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeMs === right.mtimeMs &&
    left.ctimeMs === right.ctimeMs
  );
}

import { createHash, randomUUID } from "node:crypto";
import {
  chmodSync,
  closeSync,
  constants,
  fstatSync,
  fsyncSync,
  linkSync,
  mkdirSync,
  mkdtempSync,
  openSync,
  readdirSync,
  readSync,
  realpathSync,
  rmSync,
  unlinkSync,
  writeFileSync,
  writeSync,
} from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import {
  ARTIFACT_METADATA_KEYWORD,
  type ArtifactMetadataMime,
  createPngMetadataChunk,
  injectArtifactMetadata,
  isPngMetadataKeywordPrefix,
  MAX_ARTIFACT_METADATA_BYTES,
  PNG_SIGNATURE,
  validatePngMetadataData,
} from "./artifact-metadata.js";
import type { FigureReproducibilityMetadata } from "./contracts.js";
import { ScienceError } from "./errors.js";

const ARTIFACT_VERSION = "v1";
const HASH_ALGORITHM = "sha256";
const COPY_BUFFER_BYTES = 64 * 1024;

export interface CapturedArtifactObject {
  readonly digest: `sha256:${string}`;
  readonly size: number;
}

export interface MaterializedArtifactInput {
  readonly digest: string;
  readonly size: number;
  readonly title: string;
}

export interface MaterializedArtifactInputs {
  readonly paths: readonly string[];
  dispose(): void;
}

function artifactError(error: unknown, message: string): never {
  if (error instanceof ScienceError) throw error;
  throw new ScienceError(message, "ARTIFACT_IO_FAILED", { cause: error });
}

function syncDirectory(path: string): void {
  if (process.platform === "win32") return;
  const descriptor = openSync(path, constants.O_RDONLY);
  try {
    fsyncSync(descriptor);
  } finally {
    closeSync(descriptor);
  }
}

function writeAll(descriptor: number, buffer: Buffer, length: number): void {
  let offset = 0;
  while (offset < length) {
    offset += writeSync(descriptor, buffer, offset, length - offset);
  }
}

function readExactly(descriptor: number, buffer: Buffer, length: number): void {
  let offset = 0;
  while (offset < length) {
    const read = readSync(descriptor, buffer, offset, length - offset, null);
    if (read === 0) {
      throw new ScienceError("PNG chunk ended unexpectedly", "ARTIFACT_IO_FAILED");
    }
    offset += read;
  }
}

function updatePngCrc(crc: number, content: Uint8Array): number {
  let updated = crc;
  for (const byte of content) {
    updated ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      updated = (updated >>> 1) ^ (updated & 1 ? 0xedb88320 : 0);
    }
  }
  return updated;
}

function streamPngWithMetadata(
  sourceDescriptor: number,
  sourceSize: number,
  metadata: FigureReproducibilityMetadata,
  write: (buffer: Buffer, length: number) => void,
  signal?: AbortSignal,
): void {
  const metadataChunk = createPngMetadataChunk(metadata);
  const signature = Buffer.alloc(PNG_SIGNATURE.length);
  readExactly(sourceDescriptor, signature, signature.length);
  if (!signature.equals(PNG_SIGNATURE)) {
    throw new ScienceError(
      "Artifact metadata injection requires a valid PNG signature",
      "ARTIFACT_IO_FAILED",
    );
  }
  write(signature, signature.length);
  let consumed = signature.length;
  let chunkIndex = 0;
  let ownedCount = 0;
  let sawIdat = false;
  let sawIend = false;
  const copyBuffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
  const copy = (bytes: number, include: boolean, consume: (content: Uint8Array) => void): void => {
    let remaining = bytes;
    while (remaining > 0) {
      signal?.throwIfAborted();
      const length = Math.min(remaining, copyBuffer.length);
      readExactly(sourceDescriptor, copyBuffer, length);
      consumed += length;
      consume(copyBuffer.subarray(0, length));
      if (include) write(copyBuffer, length);
      remaining -= length;
    }
  };

  while (consumed < sourceSize) {
    signal?.throwIfAborted();
    if (sourceSize - consumed < 12) {
      throw new ScienceError("PNG chunk ended unexpectedly", "ARTIFACT_IO_FAILED");
    }
    const header = Buffer.allocUnsafe(8);
    readExactly(sourceDescriptor, header, header.length);
    consumed += header.length;
    const length = header.readUInt32BE(0);
    const type = header.subarray(4, 8).toString("ascii");
    if (length > 0x7fffffff || length + 4 > sourceSize - consumed || !/^[A-Za-z]{4}$/u.test(type)) {
      throw new ScienceError("PNG chunk header is invalid", "ARTIFACT_IO_FAILED");
    }
    if (chunkIndex === 0 && (type !== "IHDR" || length !== 13)) {
      throw new ScienceError("PNG must begin with one IHDR chunk", "ARTIFACT_IO_FAILED");
    }
    if (chunkIndex > 0 && type === "IHDR") {
      throw new ScienceError("PNG contains a duplicate IHDR chunk", "ARTIFACT_IO_FAILED");
    }
    const prefixLength =
      type === "iTXt" ? Math.min(length, Buffer.byteLength(ARTIFACT_METADATA_KEYWORD) + 1) : 0;
    const prefix = Buffer.allocUnsafe(prefixLength);
    if (prefixLength > 0) {
      readExactly(sourceDescriptor, prefix, prefixLength);
      consumed += prefixLength;
    }
    const replaceOwned = type === "iTXt" && isPngMetadataKeywordPrefix(prefix, length);
    const include = !replaceOwned && type !== "IEND";
    if (include) {
      write(header, header.length);
      if (prefixLength > 0) write(prefix, prefixLength);
    }
    let crc = updatePngCrc(0xffffffff, header.subarray(4, 8));
    crc = updatePngCrc(crc, prefix);
    if (replaceOwned && length > MAX_ARTIFACT_METADATA_BYTES + 256) {
      throw new ScienceError("PNG SwarmX metadata is too large", "ARTIFACT_IO_FAILED");
    }
    const ownedData = replaceOwned ? Buffer.allocUnsafe(length) : undefined;
    if (ownedData) prefix.copy(ownedData, 0);
    let ownedOffset = prefix.length;
    copy(length - prefixLength, include, (content) => {
      crc = updatePngCrc(crc, content);
      if (ownedData) {
        Buffer.from(content).copy(ownedData, ownedOffset);
        ownedOffset += content.length;
      }
    });
    const crcBytes = Buffer.allocUnsafe(4);
    readExactly(sourceDescriptor, crcBytes, crcBytes.length);
    consumed += crcBytes.length;
    if (crcBytes.readUInt32BE(0) !== (crc ^ 0xffffffff) >>> 0) {
      throw new ScienceError("PNG chunk CRC is invalid", "ARTIFACT_IO_FAILED");
    }
    if (include) write(crcBytes, crcBytes.length);
    if (ownedData) {
      validatePngMetadataData(ownedData);
      ownedCount += 1;
      if (ownedCount > 1) {
        throw new ScienceError(
          "PNG contains duplicate SwarmX metadata chunks",
          "ARTIFACT_IO_FAILED",
        );
      }
    }
    if (type === "IDAT") sawIdat = true;
    if (type === "IEND") {
      if (length !== 0 || !sawIdat) {
        throw new ScienceError("PNG IEND chunk is invalid", "ARTIFACT_IO_FAILED");
      }
      write(metadataChunk, metadataChunk.length);
      write(header, header.length);
      write(crcBytes, crcBytes.length);
      if (consumed !== sourceSize) {
        throw new ScienceError("PNG contains bytes after IEND", "ARTIFACT_IO_FAILED");
      }
      sawIend = true;
      break;
    }
    chunkIndex += 1;
  }
  if (!sawIend) throw new ScienceError("PNG is missing IEND", "ARTIFACT_IO_FAILED");
}

/** Owner-only generic object store. Journal metadata is committed by the caller after capture. */
export class ArtifactStore {
  readonly root: string;

  private readonly objectsRoot: string;
  private readonly stagingRoot: string;

  constructor(
    scienceRoot: string,
    private readonly maxArtifactBytes: number,
  ) {
    this.root = join(scienceRoot, "artifacts", ARTIFACT_VERSION);
    this.objectsRoot = join(this.root, "objects");
    this.stagingRoot = join(this.root, "staging");
    this.ensureDirectory(this.root);
    this.ensureDirectory(this.objectsRoot);
    this.ensureDirectory(this.stagingRoot);
    this.clearStaging();
  }

  async capture(
    workspaceRoot: string,
    relativePath: string,
    signal?: AbortSignal,
    artifactMetadata?: {
      readonly metadata: FigureReproducibilityMetadata;
      readonly mime: ArtifactMetadataMime;
    },
  ): Promise<CapturedArtifactObject> {
    signal?.throwIfAborted();
    const sourcePath = this.resolveSource(workspaceRoot, relativePath);
    const stagingPath = join(this.stagingRoot, randomUUID());
    let sourceDescriptor: number | undefined;
    let stagingDescriptor: number | undefined;
    try {
      sourceDescriptor = openSync(sourcePath, constants.O_RDONLY);
      const before = fstatSync(sourceDescriptor);
      if (!before.isFile()) {
        throw new ScienceError("Artifact source must be a regular file", "ARTIFACT_PATH_INVALID");
      }
      if (before.size > this.maxArtifactBytes) {
        throw new ScienceError(
          `Artifact exceeds the configured ${this.maxArtifactBytes} byte limit`,
          "ARTIFACT_TOO_LARGE",
        );
      }

      stagingDescriptor = openSync(
        stagingPath,
        constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY,
        0o600,
      );
      const outputDescriptor = stagingDescriptor;
      const hash = createHash(HASH_ALGORITHM);
      const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
      let size = 0;
      const writeCaptured = (content: Buffer, length: number): void => {
        size += length;
        if (size > this.maxArtifactBytes) {
          throw new ScienceError(
            `Artifact exceeds the configured ${this.maxArtifactBytes} byte limit`,
            "ARTIFACT_TOO_LARGE",
          );
        }
        hash.update(content.subarray(0, length));
        writeAll(outputDescriptor, content, length);
      };
      if (artifactMetadata?.mime === "image/png") {
        streamPngWithMetadata(
          sourceDescriptor,
          before.size,
          artifactMetadata.metadata,
          writeCaptured,
          signal,
        );
      } else if (artifactMetadata) {
        const source = Buffer.allocUnsafe(before.size);
        readExactly(sourceDescriptor, source, source.length);
        const transformed = await injectArtifactMetadata(
          source,
          artifactMetadata.mime,
          artifactMetadata.metadata,
        );
        signal?.throwIfAborted();
        writeCaptured(transformed, transformed.length);
      } else {
        while (true) {
          signal?.throwIfAborted();
          const read = readSync(sourceDescriptor, buffer, 0, buffer.length, null);
          if (read === 0) break;
          writeCaptured(buffer, read);
        }
      }
      const after = fstatSync(sourceDescriptor);
      if (
        before.dev !== after.dev ||
        before.ino !== after.ino ||
        before.size !== after.size ||
        before.mtimeMs !== after.mtimeMs
      ) {
        throw new ScienceError(
          "Artifact source changed while it was being captured",
          "ARTIFACT_SOURCE_CHANGED",
        );
      }

      fsyncSync(stagingDescriptor);
      closeSync(stagingDescriptor);
      stagingDescriptor = undefined;
      const hex = hash.digest("hex");
      const targetDirectory = join(this.objectsRoot, hex.slice(0, 2));
      this.ensureDirectory(targetDirectory);
      const targetPath = join(targetDirectory, hex);
      try {
        linkSync(stagingPath, targetPath);
        chmodSync(targetPath, 0o600);
        syncDirectory(targetDirectory);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        this.verifyObject(targetPath, hex, size, signal);
      }
      signal?.throwIfAborted();
      return { digest: `sha256:${hex}`, size };
    } catch (error) {
      if (signal?.aborted && error === signal.reason) throw error;
      return artifactError(error, "Artifact capture failed");
    } finally {
      if (sourceDescriptor !== undefined) closeSync(sourceDescriptor);
      if (stagingDescriptor !== undefined) closeSync(stagingDescriptor);
      rmSync(stagingPath, { force: true });
    }
  }

  fingerprint(
    workspaceRoot: string,
    relativePath: string,
    signal?: AbortSignal,
  ): CapturedArtifactObject {
    signal?.throwIfAborted();
    const sourcePath = this.resolveSource(workspaceRoot, relativePath);
    let descriptor: number | undefined;
    try {
      descriptor = openSync(sourcePath, constants.O_RDONLY);
      const before = fstatSync(descriptor);
      if (!before.isFile()) {
        throw new ScienceError("Artifact source must be a regular file", "ARTIFACT_PATH_INVALID");
      }
      if (before.size > this.maxArtifactBytes) {
        throw new ScienceError(
          `Artifact exceeds the configured ${this.maxArtifactBytes} byte limit`,
          "ARTIFACT_TOO_LARGE",
        );
      }
      const hash = createHash(HASH_ALGORITHM);
      const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
      let size = 0;
      while (true) {
        signal?.throwIfAborted();
        const read = readSync(descriptor, buffer, 0, buffer.length, null);
        if (read === 0) break;
        size += read;
        hash.update(buffer.subarray(0, read));
      }
      const after = fstatSync(descriptor);
      if (
        before.dev !== after.dev ||
        before.ino !== after.ino ||
        before.size !== after.size ||
        before.mtimeMs !== after.mtimeMs
      ) {
        throw new ScienceError(
          "Artifact source changed while it was being fingerprinted",
          "ARTIFACT_SOURCE_CHANGED",
        );
      }
      return { digest: `sha256:${hash.digest("hex")}`, size };
    } catch (error) {
      if (signal?.aborted && error === signal.reason) throw error;
      return artifactError(error, "Artifact fingerprint failed");
    } finally {
      if (descriptor !== undefined) closeSync(descriptor);
    }
  }

  publishText(content: string, maxBytes: number, signal?: AbortSignal): CapturedArtifactObject {
    return this.publishBytes(Buffer.from(content, "utf8"), maxBytes, signal);
  }

  publishBytes(
    content: Uint8Array,
    maxBytes: number,
    signal?: AbortSignal,
  ): CapturedArtifactObject {
    signal?.throwIfAborted();
    const buffer = Buffer.from(content);
    if (buffer.byteLength > maxBytes || buffer.byteLength > this.maxArtifactBytes) {
      throw new ScienceError(
        `Generated artifact exceeds the configured ${Math.min(maxBytes, this.maxArtifactBytes)} byte limit`,
        "ARTIFACT_TOO_LARGE",
      );
    }
    const hex = createHash(HASH_ALGORITHM).update(buffer).digest("hex");
    const stagingPath = join(this.stagingRoot, randomUUID());
    let descriptor: number | undefined;
    try {
      descriptor = openSync(
        stagingPath,
        constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY,
        0o600,
      );
      signal?.throwIfAborted();
      writeAll(descriptor, buffer, buffer.byteLength);
      fsyncSync(descriptor);
      closeSync(descriptor);
      descriptor = undefined;
      const targetDirectory = join(this.objectsRoot, hex.slice(0, 2));
      this.ensureDirectory(targetDirectory);
      const targetPath = join(targetDirectory, hex);
      try {
        linkSync(stagingPath, targetPath);
        chmodSync(targetPath, 0o600);
        syncDirectory(targetDirectory);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
        this.verifyObject(targetPath, hex, buffer.byteLength, signal);
      }
      signal?.throwIfAborted();
      return { digest: `sha256:${hex}`, size: buffer.byteLength };
    } catch (error) {
      if (signal?.aborted && error === signal.reason) throw error;
      return artifactError(error, "Generated artifact publish failed");
    } finally {
      if (descriptor !== undefined) closeSync(descriptor);
      rmSync(stagingPath, { force: true });
    }
  }

  materializeInputs(
    inputs: readonly MaterializedArtifactInput[],
    maxBytes: number,
    signal?: AbortSignal,
  ): MaterializedArtifactInputs {
    signal?.throwIfAborted();
    const declaredSize = inputs.reduce((total, input) => total + input.size, 0);
    if (declaredSize > maxBytes) {
      throw new ScienceError(
        `Notebook inputs exceed the configured ${maxBytes} byte limit`,
        "ARTIFACT_TOO_LARGE",
      );
    }
    const directory = mkdtempSync(join(this.stagingRoot, "inputs-"));
    chmodSync(directory, 0o700);
    try {
      const paths = inputs.map((input, index) => {
        signal?.throwIfAborted();
        const bytes = this.readBytes(input.digest, maxBytes, signal);
        if (bytes.byteLength !== input.size) {
          throw new ScienceError(
            "Stored artifact size does not match its Journal metadata",
            "ARTIFACT_IO_FAILED",
          );
        }
        const safeTitle = input.title
          .replace(/[^A-Za-z0-9._-]+/gu, "_")
          .replace(/^\.+/u, "")
          .slice(0, 120);
        const path = join(directory, `${index}-${safeTitle || "artifact"}`);
        writeFileSync(path, bytes, { flag: "wx", mode: 0o400 });
        return path;
      });
      let disposed = false;
      return {
        paths,
        dispose() {
          if (disposed) return;
          disposed = true;
          rmSync(directory, { recursive: true, force: true });
        },
      };
    } catch (error) {
      rmSync(directory, { recursive: true, force: true });
      throw error;
    }
  }

  readBytes(digest: string, maxBytes: number, signal?: AbortSignal): Uint8Array {
    signal?.throwIfAborted();
    const hex = digest.slice("sha256:".length);
    if (!/^[0-9a-f]{64}$/u.test(hex)) {
      throw new ScienceError("Artifact digest is invalid", "ARTIFACT_IO_FAILED");
    }
    const path = join(this.objectsRoot, hex.slice(0, 2), hex);
    const descriptor = openSync(path, constants.O_RDONLY);
    try {
      const stat = fstatSync(descriptor);
      if (stat.size > maxBytes || stat.size > this.maxArtifactBytes) {
        throw new ScienceError("Stored artifact exceeds the read limit", "ARTIFACT_TOO_LARGE");
      }
      const buffer = Buffer.alloc(stat.size);
      let offset = 0;
      while (offset < buffer.length) {
        signal?.throwIfAborted();
        const read = readSync(descriptor, buffer, offset, buffer.length - offset, offset);
        if (read === 0) break;
        offset += read;
      }
      if (offset !== buffer.length) {
        throw new ScienceError("Stored artifact ended unexpectedly", "ARTIFACT_IO_FAILED");
      }
      if (createHash(HASH_ALGORITHM).update(buffer).digest("hex") !== hex) {
        throw new ScienceError("Stored artifact failed verification", "ARTIFACT_IO_FAILED");
      }
      return buffer;
    } catch (error) {
      if (error instanceof ScienceError) throw error;
      return artifactError(error, "Stored artifact read failed");
    } finally {
      closeSync(descriptor);
    }
  }

  readText(digest: string, maxBytes: number, signal?: AbortSignal): string {
    return Buffer.from(this.readBytes(digest, maxBytes, signal)).toString("utf8");
  }

  private resolveSource(workspaceRoot: string, requestedPath: string): string {
    if (
      isAbsolute(requestedPath) ||
      requestedPath.split(/[\\/]/u).some((segment) => segment === "..")
    ) {
      throw new ScienceError(
        "Artifact source must be a traversal-free relative workspace path",
        "ARTIFACT_PATH_INVALID",
      );
    }
    try {
      const canonicalRoot = realpathSync.native(workspaceRoot);
      const source = realpathSync.native(resolve(canonicalRoot, requestedPath));
      const contained = relative(canonicalRoot, source);
      if (
        contained === "" ||
        contained === ".." ||
        contained.startsWith(`..${sep}`) ||
        isAbsolute(contained)
      ) {
        throw new ScienceError(
          "Artifact source resolves outside the live workspace",
          "ARTIFACT_PATH_INVALID",
        );
      }
      return source;
    } catch (error) {
      if (error instanceof ScienceError) throw error;
      throw new ScienceError("Artifact source cannot be resolved", "ARTIFACT_PATH_INVALID", {
        cause: error,
      });
    }
  }

  private verifyObject(
    path: string,
    expectedHash: string,
    expectedSize: number,
    signal?: AbortSignal,
  ): void {
    const descriptor = openSync(path, constants.O_RDONLY);
    try {
      const hash = createHash(HASH_ALGORITHM);
      const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
      let size = 0;
      while (true) {
        signal?.throwIfAborted();
        const read = readSync(descriptor, buffer, 0, buffer.length, null);
        if (read === 0) break;
        size += read;
        hash.update(buffer.subarray(0, read));
      }
      if (size !== expectedSize || hash.digest("hex") !== expectedHash) {
        throw new ScienceError(
          "Existing artifact object failed verification",
          "ARTIFACT_IO_FAILED",
        );
      }
    } finally {
      closeSync(descriptor);
    }
  }

  private ensureDirectory(path: string): void {
    mkdirSync(path, { recursive: true, mode: 0o700 });
    chmodSync(path, 0o700);
  }

  private clearStaging(): void {
    for (const entry of readdirSync(this.stagingRoot, { withFileTypes: true })) {
      const path = join(this.stagingRoot, entry.name);
      if (entry.isFile()) unlinkSync(path);
      else rmSync(path, { recursive: true, force: true });
    }
    syncDirectory(this.stagingRoot);
  }
}

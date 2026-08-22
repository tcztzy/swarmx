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

  capture(
    workspaceRoot: string,
    relativePath: string,
    signal?: AbortSignal,
  ): CapturedArtifactObject {
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
      const hash = createHash(HASH_ALGORITHM);
      const buffer = Buffer.allocUnsafe(COPY_BUFFER_BYTES);
      let size = 0;
      while (true) {
        signal?.throwIfAborted();
        const read = readSync(sourceDescriptor, buffer, 0, buffer.length, null);
        if (read === 0) break;
        size += read;
        if (size > this.maxArtifactBytes) {
          throw new ScienceError(
            `Artifact exceeds the configured ${this.maxArtifactBytes} byte limit`,
            "ARTIFACT_TOO_LARGE",
          );
        }
        hash.update(buffer.subarray(0, read));
        writeAll(stagingDescriptor, buffer, read);
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
      const source = realpathSync.native(resolve(workspaceRoot, requestedPath));
      const contained = relative(workspaceRoot, source);
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

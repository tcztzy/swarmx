import { randomUUID } from "node:crypto";
import { chmod, mkdir, open, rename, rm } from "node:fs/promises";
import { dirname } from "node:path";

const PRIVATE_DIRECTORY_MODE = 0o700;
const PRIVATE_FILE_MODE = 0o600;
const UNSUPPORTED_DIRECTORY_SYNC_CODES = new Set(["EBADF", "EINVAL", "EISDIR", "ENOTSUP"]);

/** Atomically replaces one private JSON file without owning domain-level concurrency. */
export async function writePrivateJsonFile(filePath: string, value: unknown): Promise<void> {
  const contents = `${JSON.stringify(value, null, 2)}\n`;
  const directoryPath = dirname(filePath);
  await mkdir(directoryPath, { recursive: true, mode: PRIVATE_DIRECTORY_MODE });
  const temporaryPath = `${filePath}.tmp-${process.pid}-${randomUUID()}`;
  let temporaryCreated = false;

  try {
    const file = await open(temporaryPath, "wx", PRIVATE_FILE_MODE);
    temporaryCreated = true;
    try {
      await file.writeFile(contents, "utf8");
      await file.sync();
    } finally {
      await file.close();
    }
    await rename(temporaryPath, filePath);
    temporaryCreated = false;
    await chmod(filePath, PRIVATE_FILE_MODE);
    await syncDirectory(directoryPath);
  } finally {
    if (temporaryCreated) await rm(temporaryPath, { force: true });
  }
}

async function syncDirectory(directoryPath: string): Promise<void> {
  let directory: Awaited<ReturnType<typeof open>> | undefined;
  try {
    directory = await open(directoryPath, "r");
    await directory.sync();
  } catch (error) {
    if (!isUnsupportedDirectorySyncError(error)) throw error;
  } finally {
    await directory?.close();
  }
}

function isUnsupportedDirectorySyncError(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    typeof error.code === "string" &&
    UNSUPPORTED_DIRECTORY_SYNC_CODES.has(error.code)
  );
}

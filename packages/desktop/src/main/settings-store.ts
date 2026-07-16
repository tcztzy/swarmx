import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import {
  type DesktopSettingsDocument,
  createDefaultDesktopSettings,
  parseDesktopSettingsDocument,
} from "@swarmx/core";

export interface DesktopSettingsStoreLike {
  read(): Promise<DesktopSettingsDocument>;
  update(
    mutation: (
      current: DesktopSettingsDocument,
    ) => DesktopSettingsDocument | Promise<DesktopSettingsDocument>,
  ): Promise<DesktopSettingsDocument>;
}

export interface DesktopSettingsStoreOptions {
  path?: string;
}

export class DesktopSettingsStore implements DesktopSettingsStoreLike {
  readonly path: string;
  #tail: Promise<void> = Promise.resolve();

  constructor(options: DesktopSettingsStoreOptions = {}) {
    this.path = options.path ?? join(homedir(), ".swarmx", "settings.json");
  }

  async read(): Promise<DesktopSettingsDocument> {
    await this.#tail;
    return this.#readCurrent();
  }

  async update(
    mutation: (
      current: DesktopSettingsDocument,
    ) => DesktopSettingsDocument | Promise<DesktopSettingsDocument>,
  ): Promise<DesktopSettingsDocument> {
    let resolveResult!: (value: DesktopSettingsDocument) => void;
    let rejectResult!: (reason: unknown) => void;
    const result = new Promise<DesktopSettingsDocument>((resolve, reject) => {
      resolveResult = resolve;
      rejectResult = reject;
    });
    const operation = this.#tail.then(async () => {
      try {
        const current = await this.#readCurrent();
        const next = parseDesktopSettingsDocument(await mutation(current));
        await writeJsonAtomic(this.path, next);
        resolveResult(next);
      } catch (error) {
        rejectResult(error);
      }
    });
    this.#tail = operation.catch(() => undefined);
    return result;
  }

  async #readCurrent(): Promise<DesktopSettingsDocument> {
    try {
      return parseDesktopSettingsDocument(JSON.parse(await readFile(this.path, "utf8")));
    } catch (error) {
      if (isMissingFileError(error)) return createDefaultDesktopSettings();
      throw error;
    }
  }
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporaryPath, path);
}

function isMissingFileError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "code" in error &&
    (error as { code?: unknown }).code === "ENOENT"
  );
}

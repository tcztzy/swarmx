import { createHash, randomUUID } from "node:crypto";
import {
  chmodSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";

const IMPORT_MARKER = ".legacy-product-import.json";
const PRODUCT_DIRECTORIES = ["pkb", "science", "swarm"] as const;
const MAX_ENTRIES = 100_000;
const MAX_BYTES = 1024 * 1024 * 1024;

export interface LegacyProductImportLimits {
  maxEntries: number;
  maxBytes: number;
}

interface LegacyProductImportBudget extends LegacyProductImportLimits {
  entries: number;
  bytes: number;
}

export class ProductHomeImportError extends Error {}

export type ProductHomeImportResult =
  | "imported"
  | "already_imported"
  | "already_initialized"
  | "no_legacy_data";

export function resolveProductHome(
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  const configured = environment.SWARMX_HOME?.trim();
  return configured ? configured : join(homedir(), ".swarmx");
}

export function resolveLegacyProductHome(
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  const configured = environment.DSH_HOME?.trim();
  return configured ? configured : join(homedir(), ".dsh");
}

export function importLegacyProductState(options: {
  legacyHome: string;
  productHome: string;
  limits?: LegacyProductImportLimits;
}): ProductHomeImportResult {
  const { legacyHome, productHome } = options;
  const limits = options.limits ?? { maxEntries: MAX_ENTRIES, maxBytes: MAX_BYTES };
  assertImportLimits(limits);
  const marker = join(productHome, IMPORT_MARKER);
  if (existsSync(marker)) return "already_imported";

  mkdirSync(productHome, { recursive: true, mode: 0o700 });
  chmodSync(productHome, 0o700);
  if (readdirSync(productHome).length > 0) {
    return "already_initialized";
  }

  const available = PRODUCT_DIRECTORIES.filter((name) => existsSync(join(legacyHome, name)));
  if (available.length === 0) {
    writeMarker(productHome, legacyHome, []);
    return "no_legacy_data";
  }

  const stage = join(dirname(productHome), `.${basename(productHome)}.import-${randomUUID()}`);
  mkdirSync(stage, { mode: 0o700 });
  try {
    const budget: LegacyProductImportBudget = { ...limits, entries: 0, bytes: 0 };
    for (const name of available) {
      const source = join(legacyHome, name);
      const destination = join(stage, name);
      copyVerifiedTree(source, destination, budget);
      if (treeDigest(source) !== treeDigest(destination)) {
        throw new ProductHomeImportError(`Legacy product state verification failed for "${name}".`);
      }
    }
    writeMarker(stage, legacyHome, available);
    rmdirSync(productHome);
    renameSync(stage, productHome);
    return "imported";
  } catch (error) {
    rmSync(stage, { recursive: true, force: true });
    if (error instanceof ProductHomeImportError) throw error;
    throw new ProductHomeImportError(
      `Failed to import legacy product state: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

function copyVerifiedTree(
  source: string,
  destination: string,
  budget: LegacyProductImportBudget,
): void {
  const sourceStat = lstatSync(source);
  if (!sourceStat.isDirectory() || sourceStat.isSymbolicLink()) {
    throw new ProductHomeImportError(`Legacy product entry "${source}" is not a directory.`);
  }
  mkdirSync(destination, { mode: 0o700 });
  for (const entry of readdirSync(source, { withFileTypes: true })) {
    const sourcePath = join(source, entry.name);
    const destinationPath = join(destination, entry.name);
    const stat = lstatSync(sourcePath);
    consumeImportBudget(budget, 1, stat.isFile() ? stat.size : 0);
    if (stat.isSymbolicLink()) {
      throw new ProductHomeImportError(
        `Legacy product state contains symbolic link "${sourcePath}".`,
      );
    }
    if (stat.isDirectory()) {
      copyVerifiedTree(sourcePath, destinationPath, budget);
      continue;
    }
    if (!stat.isFile()) {
      throw new ProductHomeImportError(
        `Legacy product state contains unsupported entry "${sourcePath}".`,
      );
    }
    copyFileSync(sourcePath, destinationPath);
    chmodSync(destinationPath, 0o600);
  }
}

function consumeImportBudget(
  budget: LegacyProductImportBudget,
  entries: number,
  bytes: number,
): void {
  budget.entries += entries;
  budget.bytes += bytes;
  if (budget.entries > budget.maxEntries || budget.bytes > budget.maxBytes) {
    throw new ProductHomeImportError("Legacy product state exceeds the bounded import limit.");
  }
}

function assertImportLimits(limits: LegacyProductImportLimits): void {
  if (
    !Number.isSafeInteger(limits.maxEntries) ||
    limits.maxEntries < 1 ||
    !Number.isSafeInteger(limits.maxBytes) ||
    limits.maxBytes < 1
  ) {
    throw new ProductHomeImportError(
      "Legacy product import limits must be positive safe integers.",
    );
  }
}

function treeDigest(root: string): string {
  const hash = createHash("sha256");
  const visit = (directory: string, prefix: string): void => {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) =>
      a.name.localeCompare(b.name),
    )) {
      const path = join(directory, entry.name);
      const name = prefix ? `${prefix}/${entry.name}` : entry.name;
      hash.update(`${entry.isDirectory() ? "d" : "f"}:${name}\0`);
      if (entry.isDirectory()) visit(path, name);
      else hash.update(readFileSync(path));
    }
  };
  visit(root, "");
  return hash.digest("hex");
}

function writeMarker(productHome: string, legacyHome: string, imported: readonly string[]): void {
  const marker = join(productHome, IMPORT_MARKER);
  writeFileSync(marker, `${JSON.stringify({ version: 1, legacyHome, imported }, undefined, 2)}\n`, {
    mode: 0o600,
  });
}

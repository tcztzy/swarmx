import { readFileSync } from "node:fs";

const IMPORT_PATTERN = /^@import\s+"(\.[^"]+)";\s*$/gm;

export function readStylesheet(entry: URL): string {
  const source = readFileSync(entry, "utf8");
  return source.replace(IMPORT_PATTERN, (_rule, relativePath: string) =>
    readStylesheet(new URL(relativePath, entry)),
  );
}

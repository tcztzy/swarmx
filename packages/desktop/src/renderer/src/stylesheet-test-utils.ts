import { readFileSync } from "node:fs";

const LOCAL_IMPORT_PATTERN = /^@import\s+"(\.[^"]+)"(?:\s+layer\([^)]+\))?;\s*$/gm;

export function readStylesheet(entry: URL): string {
  const source = readFileSync(entry, "utf8");
  return source.replace(LOCAL_IMPORT_PATTERN, (_rule, relativePath: string) =>
    readStylesheet(new URL(relativePath, entry)),
  );
}

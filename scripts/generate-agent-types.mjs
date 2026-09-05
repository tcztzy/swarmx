import { execFileSync } from "node:child_process";
import { readdirSync, renameSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const manifest = new URL("../apps/desktop/package.json", import.meta.url);
const root = dirname(fileURLToPath(manifest));
const output = resolve(root, "src/agents/generated");
execFileSync("codex", ["app-server", "generate-ts", "--out", output], {
  stdio: "inherit",
});
// All generated files are declarations; do not ship empty JS modules for protocol types.
for (const entry of readdirSync(output, { recursive: true, withFileTypes: true })) {
  if (entry.isFile() && entry.name.endsWith(".ts") && !entry.name.endsWith(".d.ts")) {
    const path = join(entry.parentPath, entry.name);
    renameSync(path, path.replace(/\.ts$/u, ".d.ts"));
  }
}

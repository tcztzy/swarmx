import { readdirSync, readFileSync } from "node:fs";
import { extname, join, relative } from "node:path";

const root = process.cwd();
const extensions = new Set([".ts", ".tsx", ".js", ".mjs", ".cjs"]);

function walk(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    if (
      entry.isDirectory() &&
      ["coverage", "dist", "lib", "node_modules", "generated"].includes(entry.name)
    ) {
      return [];
    }
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return walk(path);
    return extensions.has(extname(entry.name)) ? [relative(root, path)] : [];
  });
}

const sourceFiles = [join(root, "apps"), join(root, "packages"), join(root, "scripts")]
  .flatMap((directory) => walk(directory))
  .sort();
const map = readFileSync(join(root, "CODEBASE.md"), "utf8");
const documented = [...map.matchAll(/`([^`]+)`/gu)].map((match) => match[1]);
const missing = sourceFiles.filter(
  (file) =>
    !documented.some((path) => (path?.endsWith("/") ? file.startsWith(path) : path === file)),
);

if (missing.length > 0) {
  console.error("CODEBASE.md is missing authored source paths:");
  for (const file of missing) console.error(`- ${file}`);
  process.exitCode = 1;
} else {
  console.log(`CODEBASE.md covers ${sourceFiles.length} authored source/test files.`);
}

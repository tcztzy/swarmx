import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const ROOT = process.cwd();
const SOURCE_EXTENSIONS = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".css",
  ".html",
  ".py",
]);
const EXTRA_ROOTS = [
  "packages/desktop/scripts",
  "packages/runtime/python",
  "packages/swarmx/bin",
  "scripts",
  "evals/inspect",
];

function statSafe(filePath) {
  try {
    return statSync(filePath);
  } catch {
    return null;
  }
}

function walk(relativeRoot, extensions = SOURCE_EXTENSIONS) {
  const absoluteRoot = path.join(ROOT, relativeRoot);
  if (!statSafe(absoluteRoot)?.isDirectory()) return [];

  const files = [];
  for (const entry of readdirSync(absoluteRoot, { withFileTypes: true })) {
    const relativePath = path.join(relativeRoot, entry.name);
    if (entry.isDirectory()) {
      files.push(...walk(relativePath, extensions));
    } else if (extensions.has(path.extname(entry.name))) {
      files.push(relativePath.split(path.sep).join("/"));
    }
  }
  return files;
}

const packageRoots = readdirSync(path.join(ROOT, "packages"), { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .flatMap((entry) => [`packages/${entry.name}/src`, `packages/${entry.name}/tests`]);
const sourceFiles = [
  ...new Set([...packageRoots, ...EXTRA_ROOTS].flatMap((root) => walk(root))),
].sort();
const docs = walk("docs/codebase", new Set([".md"]))
  .filter((file) => file.endsWith(".md"))
  .map((file) => readFileSync(path.join(ROOT, file), "utf8"))
  .join("\n");
const missing = sourceFiles.filter((file) => !docs.includes(file));

if (missing.length > 0) {
  console.error("Codebase documentation is missing these authored source paths:");
  for (const file of missing) console.error(`- ${file}`);
  process.exitCode = 1;
} else {
  console.log(`Codebase documentation covers ${sourceFiles.length} authored source/test files.`);
}

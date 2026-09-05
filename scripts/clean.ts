import { rmSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const outputs = [
  resolve(root, "apps/desktop/dist"),
  resolve(root, "packages/core/annotation/lib"),
  resolve(root, "packages/core/dvc/lib"),
  resolve(root, "packages/core/knowledge-base/lib"),
  resolve(root, "packages/core/swarm/lib"),
  resolve(root, "packages/science/core/lib"),
  resolve(root, "packages/science/core/bin"),
];

for (const output of outputs) rmSync(output, { force: true, recursive: true });

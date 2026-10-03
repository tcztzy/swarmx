import { rmSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const outputs = [
  resolve(root, "apps/desktop/release/app"),
  resolve(root, "apps/desktop/dist"),
  resolve(root, "packages/core/annotation/lib"),
  resolve(root, "packages/core/evidence/lib"),
  resolve(root, "packages/core/dvc/lib"),
  resolve(root, "packages/core/memory/lib"),
  resolve(root, "packages/core/swarm/lib"),
];

for (const output of outputs) rmSync(output, { force: true, recursive: true });

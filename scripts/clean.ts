import { rmSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const outputs = [
  resolve(root, "apps/desktop/dist"),
  resolve(root, "packages/core/annotation/lib"),
  resolve(root, "packages/client/ui-conversation/lib"),
  resolve(root, "packages/client/ui-science/lib"),
  resolve(root, "packages/science/core/lib"),
  resolve(root, "packages/science/core/bin"),
];

for (const output of outputs) rmSync(output, { force: true, recursive: true });

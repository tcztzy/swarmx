import { rmSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const outputs = [
  resolve(root, "apps/desktop/dist"),
  resolve(root, "packages/client/ui-conversation/lib"),
];

for (const output of outputs) rmSync(output, { force: true, recursive: true });

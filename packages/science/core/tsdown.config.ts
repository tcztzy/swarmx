import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "lib/types/index.js",
    core: "lib/types/core.js",
    contracts: "lib/types/contracts.js",
    typert: "lib/types/typert.js",
    remote: "lib/types/remote.js",
    tools: "lib/types/tools.js",
    preset: "lib/types/preset.js",
    "resource-id": "lib/types/resource-id.js",
    "resource-resolver": "lib/types/resource-resolver.js",
    "resource-view": "lib/types/resource-view.js",
  },
  outDir: "lib",
  format: ["esm"],
  platform: "node",
  target: "es2024",
  fixedExtension: false,
  dts: false,
  clean: false,
  outputOptions: {
    chunkFileNames: "[name].js",
  },
});

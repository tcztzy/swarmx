import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "lib/types/index.js",
    contracts: "lib/types/contracts.js",
    tools: "lib/types/tools.js",
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

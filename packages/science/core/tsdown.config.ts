import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "lib/types/index.js",
    contracts: "lib/types/contracts.js",
    typert: "lib/types/typert.js",
    remote: "lib/types/remote.js",
    tools: "lib/types/tools.js",
    preset: "lib/types/preset.js",
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

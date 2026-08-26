import { defineConfig } from "tsdown";

export default defineConfig({
  entry: {
    index: "lib/types/index.js",
    contracts: "lib/types/contracts.js",
    remote: "lib/types/remote.js",
    "remote-contract": "lib/types/remote-contract.js",
    tools: "lib/types/tools.js",
    typert: "lib/types/typert.js",
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

import { defineConfig } from "tsdown";

function isBuildFaceClient(value: unknown): boolean {
  if (value === undefined || value === "host") return false;
  if (value === "client") return true;
  throw new Error(`tsdown: --env.DSH_BUILD_FACE must be host or client, received ${String(value)}`);
}

export default defineConfig(({ env }) => ({
  workspace: ["packages/*/*"],
  entry: isBuildFaceClient(env?.DSH_BUILD_FACE) ? "" : ["lib/types/index.js"],
  outDir: "lib",
  format: ["esm"],
  platform: "node",
  target: "es2024",
  fixedExtension: false,
  dts: false,
  clean: false,
}));

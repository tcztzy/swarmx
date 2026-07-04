import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      {
        find: /^@swarmx\/core\/rendering$/,
        replacement: fileURLToPath(
          new URL("../../packages/core/src/rendering.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core$/,
        replacement: fileURLToPath(new URL("../../packages/core/src/index.ts", import.meta.url)),
      },
    ],
  },
});

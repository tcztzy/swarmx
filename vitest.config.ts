import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const desktopNodeModules = "./packages/desktop/node_modules";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      {
        find: /^@swarmx\/core\/rendering$/,
        replacement: fileURLToPath(new URL("./packages/core/src/rendering.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core$/,
        replacement: fileURLToPath(new URL("./packages/core/src/index.ts", import.meta.url)),
      },
      {
        find: /^react$/,
        replacement: fileURLToPath(
          new URL(`${desktopNodeModules}/react/index.js`, import.meta.url),
        ),
      },
      {
        find: /^react\/jsx-runtime$/,
        replacement: fileURLToPath(
          new URL(`${desktopNodeModules}/react/jsx-runtime.js`, import.meta.url),
        ),
      },
      {
        find: /^react\/jsx-dev-runtime$/,
        replacement: fileURLToPath(
          new URL(`${desktopNodeModules}/react/jsx-dev-runtime.js`, import.meta.url),
        ),
      },
      {
        find: /^react-dom$/,
        replacement: fileURLToPath(
          new URL(`${desktopNodeModules}/react-dom/index.js`, import.meta.url),
        ),
      },
      {
        find: /^react-dom\/server$/,
        replacement: fileURLToPath(
          new URL(`${desktopNodeModules}/react-dom/server.node.js`, import.meta.url),
        ),
      },
    ],
  },
});

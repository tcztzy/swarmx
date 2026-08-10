import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const desktopNodeModules = "./packages/desktop/node_modules";

export default defineConfig({
  plugins: [react()],
  test: {
    maxWorkers: 4,
    coverage: {
      provider: "v8",
      reporter: ["text", "json-summary"],
      include: [
        "packages/acp-server/src/server.ts",
        "packages/core/src/media.ts",
        "packages/desktop/src/main/media.ts",
        "packages/desktop/src/main/window-security.ts",
      ],
      thresholds: {
        statements: 85,
        branches: 70,
        functions: 90,
        lines: 85,
        "packages/acp-server/src/server.ts": {
          statements: 76,
          branches: 60,
          functions: 85,
          lines: 76,
        },
        "packages/core/src/media.ts": {
          statements: 86,
          branches: 76,
          functions: 85,
          lines: 86,
        },
        "packages/desktop/src/main/media.ts": {
          statements: 92,
          branches: 71,
          functions: 100,
          lines: 92,
        },
        "packages/desktop/src/main/window-security.ts": {
          statements: 93,
          branches: 90,
          functions: 100,
          lines: 93,
        },
      },
    },
  },
  resolve: {
    alias: [
      {
        find: /^@swarmx\/core\/rendering$/,
        replacement: fileURLToPath(new URL("./packages/core/src/rendering.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core\/model-capabilities$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/model-capabilities.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/harness$/,
        replacement: fileURLToPath(new URL("./packages/core/src/harness.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core\/memory-runtime-protocol$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/memory-runtime-protocol.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/memory-links$/,
        replacement: fileURLToPath(new URL("./packages/core/src/memory-links.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core\/memory$/,
        replacement: fileURLToPath(new URL("./packages/core/src/memory.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/runtime$/,
        replacement: fileURLToPath(new URL("./packages/runtime/src/index.ts", import.meta.url)),
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

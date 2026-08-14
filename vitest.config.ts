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
        "packages/cli/src/send-config.ts",
        "packages/core/src/audit.ts",
        "packages/core/src/context-engine.ts",
        "packages/core/src/local-tool-contracts.ts",
        "packages/core/src/media.ts",
        "packages/core/src/project-contracts.ts",
        "packages/core/src/request-scope.ts",
        "packages/core/src/task-runtime.ts",
        "packages/core/src/task-supervisor.ts",
        "packages/desktop/src/main/app-update-ipc.ts",
        "packages/desktop/src/main/browser-host.ts",
        "packages/desktop/src/main/browser-ipc.ts",
        "packages/desktop/src/main/global-memory-service.ts",
        "packages/desktop/src/main/global-memory-ipc.ts",
        "packages/desktop/src/main/ipc-router.ts",
        "packages/desktop/src/main/media.ts",
        "packages/desktop/src/main/private-json-file.ts",
        "packages/desktop/src/main/project-ipc.ts",
        "packages/desktop/src/main/project-service.ts",
        "packages/desktop/src/main/session-messages.ts",
        "packages/desktop/src/main/task-supervisor.ts",
        "packages/desktop/src/main/task-runtime-ipc.ts",
        "packages/desktop/src/main/terminal-host.ts",
        "packages/desktop/src/main/terminal-ipc.ts",
        "packages/desktop/src/main/window-security.ts",
        "packages/desktop/src/main/workspace-inspection-ipc.ts",
        "packages/desktop/src/main/workspace-tool-permissions.ts",
        "packages/desktop/src/renderer/src/doctor-controller.ts",
        "packages/desktop/src/renderer/src/terminal-controller.ts",
        "packages/desktop/src/shared/ipc-contracts/app-update.ts",
        "packages/desktop/src/shared/ipc-contracts/base.ts",
        "packages/desktop/src/shared/ipc-contracts/browser.ts",
        "packages/desktop/src/shared/ipc-contracts/global-memory.ts",
        "packages/desktop/src/shared/ipc-contracts/index.ts",
        "packages/desktop/src/shared/ipc-contracts/project.ts",
        "packages/desktop/src/shared/ipc-contracts/task-runtime.ts",
        "packages/desktop/src/shared/ipc-contracts/terminal.ts",
        "packages/desktop/src/shared/ipc-contracts/workspace-inspection.ts",
      ],
      thresholds: {
        statements: 84,
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
        "packages/cli/src/send-config.ts": {
          statements: 100,
          branches: 76,
          functions: 100,
          lines: 100,
        },
        "packages/core/src/audit.ts": {
          statements: 87,
          branches: 76,
          functions: 95,
          lines: 90,
        },
        "packages/core/src/context-engine.ts": {
          statements: 84,
          branches: 67,
          functions: 93,
          lines: 89,
        },
        "packages/core/src/local-tool-contracts.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/core/src/request-scope.ts": {
          statements: 100,
          branches: 92,
          functions: 100,
          lines: 100,
        },
        "packages/core/src/project-contracts.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/core/src/task-runtime.ts": {
          statements: 79,
          branches: 70,
          functions: 92,
          lines: 81,
        },
        "packages/core/src/task-supervisor.ts": {
          statements: 77,
          branches: 55,
          functions: 77,
          lines: 83,
        },
        "packages/desktop/src/main/media.ts": {
          statements: 92,
          branches: 71,
          functions: 100,
          lines: 92,
        },
        "packages/desktop/src/main/app-update-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/browser-host.ts": {
          statements: 94,
          branches: 84,
          functions: 96,
          lines: 97,
        },
        "packages/desktop/src/main/browser-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/global-memory-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/global-memory-service.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/ipc-router.ts": {
          statements: 100,
          branches: 92,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/private-json-file.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/project-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/project-service.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/session-messages.ts": {
          statements: 86,
          branches: 84,
          functions: 92,
          lines: 94,
        },
        "packages/desktop/src/main/task-supervisor.ts": {
          statements: 85,
          branches: 68,
          functions: 100,
          lines: 87,
        },
        "packages/desktop/src/main/task-runtime-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/terminal-host.ts": {
          statements: 88,
          branches: 74,
          functions: 100,
          lines: 90,
        },
        "packages/desktop/src/main/terminal-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/window-security.ts": {
          statements: 93,
          branches: 90,
          functions: 100,
          lines: 93,
        },
        "packages/desktop/src/main/workspace-inspection-ipc.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/main/workspace-tool-permissions.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/renderer/src/doctor-controller.ts": {
          statements: 100,
          branches: 95,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/renderer/src/terminal-controller.ts": {
          statements: 95,
          branches: 83,
          functions: 88,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/app-update.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/base.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/browser.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/global-memory.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/index.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/project.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/task-runtime.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/terminal.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
        "packages/desktop/src/shared/ipc-contracts/workspace-inspection.ts": {
          statements: 100,
          branches: 100,
          functions: 100,
          lines: 100,
        },
      },
    },
  },
  resolve: {
    alias: [
      {
        find: /^@swarmx\/core\/personal-memory$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/personal-memory.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/task-runtime$/,
        replacement: fileURLToPath(new URL("./packages/core/src/task-runtime.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core\/task-worker-protocol$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/task-worker-protocol.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/rendering$/,
        replacement: fileURLToPath(new URL("./packages/core/src/rendering.ts", import.meta.url)),
      },
      {
        find: /^@swarmx\/core\/local-tool-contracts$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/local-tool-contracts.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/request-scope$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/request-scope.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/project-contracts$/,
        replacement: fileURLToPath(
          new URL("./packages/core/src/project-contracts.ts", import.meta.url),
        ),
      },
      {
        find: /^@swarmx\/core\/project$/,
        replacement: fileURLToPath(new URL("./packages/core/src/project.ts", import.meta.url)),
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

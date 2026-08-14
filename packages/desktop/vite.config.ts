import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { notBundle } from "vite-plugin-electron/plugin";
import electron from "vite-plugin-electron/simple";
import { forwardElectronStderr } from "./scripts/electron-stderr.mjs";

const desktopRoot = fileURLToPath(new URL(".", import.meta.url));

function electronProcessEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  return env;
}

export default defineConfig(async () => ({
  root: `${desktopRoot}/src/renderer`,
  plugins: [
    tailwindcss(),
    react(),
    ...(await electron({
      main: {
        entry: {
          index: `${desktopRoot}/src/main/index.ts`,
          library: `${desktopRoot}/src/main/library.ts`,
          "task-supervisor-entry": `${desktopRoot}/src/main/task-supervisor-entry.ts`,
        },
        onstart: async ({ startup }) => {
          const started = await startup(["."], {
            cwd: desktopRoot,
            env: electronProcessEnv(),
            stdio: ["inherit", "inherit", "pipe", "ipc"],
          });
          if (started) forwardElectronStderr(process.electronApp?.stderr);
        },
        vite: {
          plugins: [notBundle()],
          build: {
            outDir: `${desktopRoot}/out/main`,
            emptyOutDir: true,
          },
        },
      },
      preload: {
        input: `${desktopRoot}/src/preload/index.ts`,
        vite: {
          plugins: [notBundle({ filter: ["electron"] })],
          build: {
            outDir: `${desktopRoot}/out/preload`,
            emptyOutDir: true,
          },
        },
      },
    })),
  ],
  build: {
    outDir: `${desktopRoot}/out/renderer`,
    emptyOutDir: true,
    cssCodeSplit: false,
    rolldownOptions: {
      output: {
        assetFileNames: (assetInfo) =>
          assetInfo.names.some((name) => name.endsWith(".css"))
            ? "assets/swarmx.css"
            : "assets/[name]-[hash][extname]",
      },
    },
  },
  resolve: {
    alias: {
      "@": `${desktopRoot}/src/renderer/src`,
      "@swarmx/core/harness": `${desktopRoot}/../core/src/harness.ts`,
      "@swarmx/core/local-tool-contracts": `${desktopRoot}/../core/src/local-tool-contracts.ts`,
      "@swarmx/core/personal-memory": `${desktopRoot}/../core/src/personal-memory.ts`,
      "@swarmx/core/project-contracts": `${desktopRoot}/../core/src/project-contracts.ts`,
      "@swarmx/core/project": `${desktopRoot}/../core/src/project.ts`,
      "@swarmx/core/rendering": `${desktopRoot}/../core/src/rendering.ts`,
      "@swarmx/core/request-scope": `${desktopRoot}/../core/src/request-scope.ts`,
      "@swarmx/core/task-runtime": `${desktopRoot}/../core/src/task-runtime.ts`,
      "@swarmx/core/task-worker-protocol": `${desktopRoot}/../core/src/task-worker-protocol.ts`,
      "@swarmx/core/memory-runtime-protocol": `${desktopRoot}/../core/src/memory-runtime-protocol.ts`,
    },
  },
}));

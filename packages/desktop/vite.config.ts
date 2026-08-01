import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { notBundle } from "vite-plugin-electron/plugin";
import electron from "vite-plugin-electron/simple";

const desktopRoot = fileURLToPath(new URL(".", import.meta.url));

function electronProcessEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  return env;
}

export default defineConfig(async () => ({
  root: `${desktopRoot}/src/renderer`,
  plugins: [
    react(),
    ...(await electron({
      main: {
        entry: {
          index: `${desktopRoot}/src/main/index.ts`,
          library: `${desktopRoot}/src/main/library.ts`,
        },
        onstart: ({ startup }) => startup(["."], { cwd: desktopRoot, env: electronProcessEnv() }),
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
          plugins: [notBundle()],
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
  },
  resolve: {
    alias: {
      "@": `${desktopRoot}/src/renderer/src`,
      "@swarmx/core/rendering": `${desktopRoot}/../core/src/rendering.ts`,
    },
  },
}));

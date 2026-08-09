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
      "@swarmx/core/rendering": `${desktopRoot}/../core/src/rendering.ts`,
    },
  },
}));

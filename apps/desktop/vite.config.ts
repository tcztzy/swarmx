import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const desktopRoot = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root: desktopRoot,
  base: "./",
  plugins: [tailwindcss(), react()],
  resolve: {
    alias: [
      {
        find: "@/components/ui/radix",
        replacement: `${desktopRoot}src/renderer/components/ui/radix`,
      },
      { find: "@/components/ui", replacement: `${desktopRoot}src/renderer/components/ui/radix` },
      { find: "@", replacement: `${desktopRoot}src/renderer` },
    ],
  },
  build: {
    outDir: fileURLToPath(new URL("./dist/renderer", import.meta.url)),
    emptyOutDir: true,
  },
});

import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "@/components/ui/radix",
        replacement: fileURLToPath(
          new URL("./apps/desktop/src/renderer/components/ui/radix", import.meta.url),
        ),
      },
      {
        find: "@/components/ui",
        replacement: fileURLToPath(
          new URL("./apps/desktop/src/renderer/components/ui/radix", import.meta.url),
        ),
      },
      {
        find: "@",
        replacement: fileURLToPath(new URL("./apps/desktop/src/renderer", import.meta.url)),
      },
    ],
  },
  test: {
    include: ["apps/*/tests/**/*.test.{ts,tsx}", "packages/*/*/tests/**/*.test.{ts,tsx}"],
  },
});

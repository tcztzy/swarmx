import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

export default defineConfig({
  resolve: {
    alias: {
      "@swarmx/dsh-science/remote": fileURLToPath(
        new URL("./packages/science/core/src/remote.ts", import.meta.url),
      ),
      "@swarmx/dsh-science/types": fileURLToPath(
        new URL("./packages/science/core/src/contracts.ts", import.meta.url),
      ),
    },
  },
  test: {
    include: [
      "apps/*/tests/**/*.test.ts",
      "packages/*/*/tests/**/*.test.{ts,tsx}",
      "scripts/**/*.test.ts",
    ],
  },
});

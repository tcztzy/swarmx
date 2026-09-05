import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: [
      "apps/*/tests/**/*.test.ts",
      "packages/*/*/tests/**/*.test.{ts,tsx}",
      "scripts/**/*.test.ts",
    ],
  },
});

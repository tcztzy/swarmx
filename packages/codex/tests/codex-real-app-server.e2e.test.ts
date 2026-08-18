import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { CodexServerClient } from "../src/codex-server-client.js";
import { CODEX_MODULE_COMMAND, resolveCodexAcpLaunch } from "../src/index.js";

const CODEX_E2E_BIN = process.env.CODEX_E2E_BIN?.trim();

describe("real Codex app-server transport", () => {
  const runReal = CODEX_E2E_BIN ? it : it.skip;

  runReal(
    "resolves codexCommand and PATH before initializing app-server without ACP",
    async () => {
      const binary = CODEX_E2E_BIN as string;
      expect(
        resolveCodexAcpLaunch(
          { command: CODEX_MODULE_COMMAND, args: [] },
          { codexCommand: binary, envPath: dirname(binary) },
        ),
      ).toEqual({ command: binary, args: ["app-server"], env: {} });
      const pathLaunch = resolveCodexAcpLaunch(
        { command: CODEX_MODULE_COMMAND, args: [] },
        { envPath: dirname(binary) },
      );
      expect(pathLaunch).toEqual({ command: binary, args: ["app-server"], env: {} });

      const home = mkdtempSync(join(tmpdir(), "swarmx-codex-e2e-"));
      const client = new CodexServerClient(pathLaunch);
      try {
        const sessions = await withTimeout(
          client.listSessions(
            {
              command: binary,
              args: [],
              cwd: process.cwd(),
              env: { CODEX_HOME: home },
            },
            process.cwd(),
          ),
          20_000,
        );
        expect(Array.isArray(sessions)).toBe(true);
      } finally {
        client.kill();
        rmSync(home, { recursive: true, force: true });
      }
    },
    25_000,
  );
});

async function withTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`Codex app-server test timed out after ${timeoutMs}ms`)),
          timeoutMs,
        );
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Context } from "@deepseek-ai/cordis";
import SessionStore, { type SessionId } from "@deepseek-ai/dsh-session";
import LocalSubprocessRuntime from "@deepseek-ai/dsh-subprocess-local";
import ScienceService from "../src/index.js";

export interface ScienceFixture {
  readonly context: Context;
  readonly databasePath: string;
  readonly root: string;
  readonly scratch: string;
  readonly sessionA: SessionId;
  readonly sessionB: SessionId;
  scienceFiber: Awaited<ReturnType<Context["plugin"]>>;
  readonly workspaceA: string;
  readonly workspaceB: string;
  dispose(): Promise<void>;
  remount(): Promise<void>;
}

export interface ScienceFixtureConfig {
  readonly maxArtifactBytes?: number;
  readonly maxCellOutputBytes?: number;
  readonly maxExportBytes?: number;
  readonly notebookRuntime?: "isolated" | "jupymcp";
  readonly jupymcpArgs?: readonly string[];
  readonly jupymcpCommand?: string;
  readonly jupymcpRequestTimeoutMs?: number;
  readonly pythonCommand?: string;
}

export async function createScienceFixture(
  config: ScienceFixtureConfig | number = {},
): Promise<ScienceFixture> {
  const scratch = mkdtempSync(join(tmpdir(), "swarmx-science-t14-"));
  const root = join(scratch, "science");
  const workspaceA = join(scratch, "workspace-a");
  const workspaceB = join(scratch, "workspace-b");
  mkdirSync(workspaceA);
  mkdirSync(workspaceB);
  const context = new Context();
  await context.plugin(SessionStore);
  await context.plugin(LocalSubprocessRuntime);
  const sessionA = "science-t14-session-a" as SessionId;
  const sessionB = "science-t14-session-b" as SessionId;
  context.sessions.create(sessionA, { meta: { cwd: workspaceA } });
  context.sessions.create(sessionB, { meta: { cwd: workspaceB } });
  const options = typeof config === "number" ? { maxArtifactBytes: config } : config;
  const serviceConfig = { root, notebookRuntime: "isolated" as const, ...options };
  const fixture: ScienceFixture = {
    context,
    databasePath: join(root, "science.sqlite"),
    root,
    scratch,
    scienceFiber: await context.plugin(ScienceService, serviceConfig as never),
    sessionA,
    sessionB,
    workspaceA,
    workspaceB,
    async dispose() {
      await context.fiber.dispose();
      rmSync(scratch, { recursive: true, force: true });
    },
    async remount() {
      fixture.scienceFiber = await context.plugin(ScienceService, serviceConfig as never);
    },
  };
  return fixture;
}

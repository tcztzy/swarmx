import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { NodeScienceProcessRuntime } from "../../../../apps/desktop/src/host/process-runner.js";
import ScienceCore, { ScienceError } from "../src/index.js";

interface ScienceTestContext {
  science: ScienceCore;
}

interface ScienceTestFiber {
  dispose(): Promise<void>;
}

export interface ScienceFixture {
  readonly context: ScienceTestContext;
  readonly databasePath: string;
  readonly root: string;
  readonly scratch: string;
  readonly sessionA: string;
  readonly sessionB: string;
  scienceFiber: ScienceTestFiber;
  readonly workspaceA: string;
  readonly workspaceB: string;
  dispose(): Promise<void>;
  remount(): Promise<void>;
}

export interface ScienceFixtureConfig {
  readonly embedArtifactMetadata?: boolean;
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
  const sessionA = "science-t14-session-a";
  const sessionB = "science-t14-session-b";
  const workspaces = new Map([
    [sessionA, { key: "workspace-a", root: workspaceA }],
    [sessionB, { key: "workspace-b", root: workspaceB }],
  ]);
  const options = typeof config === "number" ? { maxArtifactBytes: config } : config;
  const serviceConfig = { root, notebookRuntime: "isolated" as const, ...options };
  const context = {} as ScienceTestContext;
  const mount = (): ScienceTestFiber => {
    const disposers: Array<() => Promise<void>> = [];
    context.science = new ScienceCore(
      {
        subprocess: new NodeScienceProcessRuntime(),
        onDispose: (dispose) => disposers.push(dispose),
      },
      serviceConfig,
      (sessionId) => {
        const workspace = workspaces.get(sessionId);
        if (workspace === undefined) {
          throw new ScienceError(`Session "${sessionId}" was not found.`, "SESSION_NOT_FOUND");
        }
        return workspace;
      },
    );
    let disposed = false;
    return {
      async dispose() {
        if (disposed) return;
        disposed = true;
        for (const dispose of disposers.reverse()) await dispose();
      },
    };
  };
  const fixture: ScienceFixture = {
    context,
    databasePath: join(root, "science.sqlite"),
    root,
    scratch,
    scienceFiber: mount(),
    sessionA,
    sessionB,
    workspaceA,
    workspaceB,
    async dispose() {
      await fixture.scienceFiber.dispose();
      rmSync(scratch, { recursive: true, force: true });
    },
    async remount() {
      fixture.scienceFiber = mount();
    },
  };
  return fixture;
}

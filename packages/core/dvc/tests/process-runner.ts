import {
  NodeScienceProcessRuntime,
  spawnProcess,
} from "../../../../apps/desktop/src/host/process-runner.js";
import type { ProcessRunner } from "../src/process.js";

const runtime = new NodeScienceProcessRuntime();

export const processRunner: ProcessRunner = {
  resolveExecutable: (command, _options, signal) => runtime.resolveExecutable(command, {}, signal),
  spawn(options) {
    const spawned = spawnProcess({
      ...options,
      ...options.stdio,
      env: { ...process.env, ...options.env },
      graceMs: options.graceMs ?? 1_000,
    });
    return {
      collected: {
        ...(spawned.stdout === undefined ? {} : { stdout: spawned.stdout }),
        ...(spawned.stderr === undefined ? {} : { stderr: spawned.stderr }),
      },
      done: spawned.done,
      terminate: spawned.terminate,
      waitForExit: () => spawned.done,
    };
  },
};

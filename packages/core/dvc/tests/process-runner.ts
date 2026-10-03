import { access } from "node:fs/promises";
import { delimiter, dirname, isAbsolute, join, resolve } from "node:path";
import { spawnProcess } from "../../../../scripts/test-process.js";
import type { ProcessRunner } from "../src/process.js";

export const processRunner: ProcessRunner = {
  async resolveExecutable(command, _options, signal) {
    signal?.throwIfAborted();
    if (command.includes("/") || command.includes("\\")) {
      const candidate = isAbsolute(command) ? command : resolve(command);
      await access(candidate);
      return candidate;
    }
    const extensions =
      process.platform === "win32" ? (process.env.PATHEXT ?? ".EXE;.CMD;.BAT").split(";") : [""];
    for (const directory of (process.env.PATH ?? "").split(delimiter)) {
      for (const extension of extensions) {
        signal?.throwIfAborted();
        const candidate = join(directory || dirname(process.execPath), `${command}${extension}`);
        try {
          await access(candidate);
          return candidate;
        } catch {
          // Continue through PATH; absence is expected.
        }
      }
    }
    throw new Error(`Executable "${command}" was not found on PATH.`);
  },
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

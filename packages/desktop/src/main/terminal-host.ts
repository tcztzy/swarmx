import { randomUUID } from "node:crypto";
import { constants, chmodSync, statSync } from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import type { IDisposable, IPty } from "node-pty";
import * as pty from "node-pty";

const DEFAULT_COLUMNS = 80;
const DEFAULT_ROWS = 24;
const MAX_COLUMNS = 1_000;
const MAX_ROWS = 500;
const MAX_INPUT_LENGTH = 1024 * 1024;

export interface TerminalOwner {
  id: number;
  isDestroyed?(): boolean;
  send(channel: string, value: unknown): void;
}

export interface CreateTerminalRequest {
  id?: string;
  cwd: string;
  cols?: number;
  rows?: number;
}

export interface TerminalProcessFactory {
  spawn(
    file: string,
    args: string[],
    options: {
      name: string;
      cols: number;
      rows: number;
      cwd: string;
      env: Record<string, string | undefined>;
    },
  ): IPty;
}

interface TerminalSession {
  owner: TerminalOwner;
  process: IPty;
  dataSubscription: IDisposable;
  exitSubscription: IDisposable;
}

export class TerminalHost {
  readonly #sessions = new Map<string, TerminalSession>();

  constructor(
    private readonly factory: TerminalProcessFactory = pty,
    private readonly platform: NodeJS.Platform = process.platform,
    private readonly env: NodeJS.ProcessEnv = process.env,
  ) {}

  create(owner: TerminalOwner, request: CreateTerminalRequest): { id: string; pid: number } {
    const cwd = request.cwd.trim();
    if (!cwd) throw new Error("Terminal working directory is required.");

    const id = request.id?.trim() || randomUUID();
    if (this.#sessions.has(id)) throw new Error("Terminal id is already active.");
    const { file, args } = terminalShell(this.platform, this.env);
    if (this.factory === pty) ensurePtySpawnHelperExecutable(this.platform);
    const process = this.factory.spawn(file, args, {
      name: "xterm-256color",
      cols: terminalDimension(request.cols, DEFAULT_COLUMNS, MAX_COLUMNS),
      rows: terminalDimension(request.rows, DEFAULT_ROWS, MAX_ROWS),
      cwd,
      env: {
        ...this.env,
        TERM: "xterm-256color",
        COLORTERM: "truecolor",
        TERM_PROGRAM: "SwarmX",
      },
    });

    const dataSubscription = process.onData((data) => {
      const session = this.#sessions.get(id);
      if (!session || session.owner.isDestroyed?.()) return;
      session.owner.send("terminal:data", { id, data });
    });
    const exitSubscription = process.onExit(({ exitCode, signal }) => {
      const session = this.#sessions.get(id);
      if (!session) return;
      this.#sessions.delete(id);
      session.dataSubscription.dispose();
      session.exitSubscription.dispose();
      if (!session.owner.isDestroyed?.()) {
        session.owner.send("terminal:exit", { id, exitCode, signal });
      }
    });

    this.#sessions.set(id, { owner, process, dataSubscription, exitSubscription });
    return { id, pid: process.pid };
  }

  write(ownerId: number, id: string, data: string): boolean {
    const session = this.#ownedSession(ownerId, id);
    if (!session) return false;
    if (data.length > MAX_INPUT_LENGTH) throw new Error("Terminal input is too large.");
    session.process.write(data);
    return true;
  }

  resize(ownerId: number, id: string, cols: number, rows: number): boolean {
    const session = this.#ownedSession(ownerId, id);
    if (!session) return false;
    session.process.resize(
      terminalDimension(cols, DEFAULT_COLUMNS, MAX_COLUMNS),
      terminalDimension(rows, DEFAULT_ROWS, MAX_ROWS),
    );
    return true;
  }

  kill(ownerId: number, id: string): boolean {
    const session = this.#ownedSession(ownerId, id);
    if (!session) return false;
    this.#close(id, session);
    return true;
  }

  cleanupOwner(ownerId: number): void {
    for (const [id, session] of this.#sessions) {
      if (session.owner.id === ownerId) this.#close(id, session);
    }
  }

  dispose(): void {
    for (const [id, session] of this.#sessions) this.#close(id, session);
  }

  #ownedSession(ownerId: number, id: string): TerminalSession | undefined {
    const session = this.#sessions.get(id);
    return session?.owner.id === ownerId ? session : undefined;
  }

  #close(id: string, session: TerminalSession): void {
    this.#sessions.delete(id);
    session.dataSubscription.dispose();
    session.exitSubscription.dispose();
    session.process.kill();
  }
}

function ensurePtySpawnHelperExecutable(platform: NodeJS.Platform): void {
  if (platform === "win32") return;
  const require = createRequire(import.meta.url);
  const packageRoot = path.resolve(path.dirname(require.resolve("node-pty")), "..");
  const candidates = [
    path.join(packageRoot, "prebuilds", `${platform}-${process.arch}`, "spawn-helper"),
    path.join(packageRoot, "build", "Release", "spawn-helper"),
  ];
  for (const candidate of candidates) {
    try {
      const mode = statSync(candidate).mode;
      if ((mode & constants.S_IXUSR) === 0) chmodSync(candidate, mode | constants.S_IXUSR);
      return;
    } catch {
      // The active node-pty distribution may use the other known helper location.
    }
  }
}

function terminalShell(
  platform: NodeJS.Platform,
  env: NodeJS.ProcessEnv,
): { file: string; args: string[] } {
  if (platform === "win32") {
    return { file: env.COMSPEC || "powershell.exe", args: [] };
  }
  return { file: env.SHELL || os.userInfo().shell || "/bin/zsh", args: ["-l"] };
}

function terminalDimension(value: number | undefined, fallback: number, maximum: number): number {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(maximum, Math.max(1, Math.floor(value ?? fallback)));
}

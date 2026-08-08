import { randomUUID } from "node:crypto";
import os from "node:os";
import type { IDisposable, IPty } from "node-pty";
import * as pty from "node-pty";
import { ensurePtySpawnHelperExecutable } from "./pty-runtime.js";

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

export type TerminalAuditOperation = "create" | "write" | "resize" | "close" | "exit";

export type TerminalAuditCloseReason = "user_kill" | "owner_cleanup" | "app_dispose";

export type TerminalAuditOutcome = "succeeded" | "failed" | "rejected";

export type TerminalAuditFailureReason =
  | "invalid_cwd"
  | "duplicate_terminal"
  | "not_owned_or_missing"
  | "input_too_large"
  | "create_failed"
  | "operation_failed";

export interface TerminalAuditEvent {
  operation: TerminalAuditOperation;
  phase: "attempt" | "outcome";
  terminalId: string;
  ownerId: number;
  outcome?: TerminalAuditOutcome;
  reason?: TerminalAuditFailureReason;
  byteCount?: number;
  cols?: number;
  rows?: number;
  pid?: number;
  exitCode?: number;
  signal?: number | null;
  closeReason?: TerminalAuditCloseReason;
}

export type TerminalAuditCallback = (event: Readonly<TerminalAuditEvent>) => void;

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
    private readonly audit?: TerminalAuditCallback,
  ) {}

  create(owner: TerminalOwner, request: CreateTerminalRequest): { id: string; pid: number } {
    const cwd = request.cwd.trim();
    const id = request.id?.trim() || randomUUID();
    const cols = terminalDimension(request.cols, DEFAULT_COLUMNS, MAX_COLUMNS);
    const rows = terminalDimension(request.rows, DEFAULT_ROWS, MAX_ROWS);
    const auditContext = { terminalId: id, ownerId: owner.id, cols, rows };

    this.#recordAudit({ operation: "create", phase: "attempt", ...auditContext });
    if (!cwd) {
      this.#recordAudit({
        operation: "create",
        phase: "outcome",
        outcome: "rejected",
        reason: "invalid_cwd",
        ...auditContext,
      });
      throw new Error("Terminal working directory is required.");
    }
    if (this.#sessions.has(id)) {
      this.#recordAudit({
        operation: "create",
        phase: "outcome",
        outcome: "rejected",
        reason: "duplicate_terminal",
        ...auditContext,
      });
      throw new Error("Terminal id is already active.");
    }

    let terminalProcess: IPty;
    let dataSubscription: IDisposable;
    let exitSubscription: IDisposable;
    try {
      const { file, args } = terminalShell(this.platform, this.env);
      if (this.factory === pty) ensurePtySpawnHelperExecutable(this.platform);
      terminalProcess = this.factory.spawn(file, args, {
        name: "xterm-256color",
        cols,
        rows,
        cwd,
        env: {
          ...this.env,
          TERM: "xterm-256color",
          COLORTERM: "truecolor",
          TERM_PROGRAM: "SwarmX",
        },
      });
      dataSubscription = terminalProcess.onData((data) => {
        const session = this.#sessions.get(id);
        if (!session || session.owner.isDestroyed?.()) return;
        session.owner.send("terminal:data", { id, data });
      });
      exitSubscription = terminalProcess.onExit(({ exitCode, signal }) => {
        const session = this.#sessions.get(id);
        if (!session) return;
        this.#recordAudit({
          operation: "exit",
          phase: "outcome",
          terminalId: id,
          ownerId: session.owner.id,
          outcome: exitCode === 0 && !signal ? "succeeded" : "failed",
          exitCode,
          signal: signal ?? null,
        });
        this.#sessions.delete(id);
        session.dataSubscription.dispose();
        session.exitSubscription.dispose();
        if (!session.owner.isDestroyed?.()) {
          session.owner.send("terminal:exit", { id, exitCode, signal });
        }
      });
      this.#sessions.set(id, {
        owner,
        process: terminalProcess,
        dataSubscription,
        exitSubscription,
      });
    } catch (error) {
      this.#recordAudit({
        operation: "create",
        phase: "outcome",
        outcome: "failed",
        reason: "create_failed",
        ...auditContext,
      });
      throw error;
    }
    this.#recordAudit({
      operation: "create",
      phase: "outcome",
      outcome: "succeeded",
      pid: terminalProcess.pid,
      ...auditContext,
    });
    return { id, pid: terminalProcess.pid };
  }

  write(ownerId: number, id: string, data: string): boolean {
    const byteCount = Buffer.byteLength(data, "utf8");
    const auditContext = { terminalId: id, ownerId, byteCount };
    this.#recordAudit({ operation: "write", phase: "attempt", ...auditContext });

    const session = this.#ownedSession(ownerId, id);
    if (!session) {
      this.#recordAudit({
        operation: "write",
        phase: "outcome",
        outcome: "rejected",
        reason: "not_owned_or_missing",
        ...auditContext,
      });
      return false;
    }
    if (data.length > MAX_INPUT_LENGTH) {
      this.#recordAudit({
        operation: "write",
        phase: "outcome",
        outcome: "rejected",
        reason: "input_too_large",
        ...auditContext,
      });
      throw new Error("Terminal input is too large.");
    }
    try {
      session.process.write(data);
    } catch (error) {
      this.#recordAudit({
        operation: "write",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
        ...auditContext,
      });
      throw error;
    }
    this.#recordAudit({
      operation: "write",
      phase: "outcome",
      outcome: "succeeded",
      ...auditContext,
    });
    return true;
  }

  resize(ownerId: number, id: string, cols: number, rows: number): boolean {
    const resizedCols = terminalDimension(cols, DEFAULT_COLUMNS, MAX_COLUMNS);
    const resizedRows = terminalDimension(rows, DEFAULT_ROWS, MAX_ROWS);
    const auditContext = {
      terminalId: id,
      ownerId,
      cols: resizedCols,
      rows: resizedRows,
    };
    this.#recordAudit({ operation: "resize", phase: "attempt", ...auditContext });

    const session = this.#ownedSession(ownerId, id);
    if (!session) {
      this.#recordAudit({
        operation: "resize",
        phase: "outcome",
        outcome: "rejected",
        reason: "not_owned_or_missing",
        ...auditContext,
      });
      return false;
    }
    try {
      session.process.resize(resizedCols, resizedRows);
    } catch (error) {
      this.#recordAudit({
        operation: "resize",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
        ...auditContext,
      });
      throw error;
    }
    this.#recordAudit({
      operation: "resize",
      phase: "outcome",
      outcome: "succeeded",
      ...auditContext,
    });
    return true;
  }

  kill(ownerId: number, id: string): boolean {
    const session = this.#ownedSession(ownerId, id);
    if (!session) {
      const auditContext = { terminalId: id, ownerId, closeReason: "user_kill" as const };
      this.#recordAudit({ operation: "close", phase: "attempt", ...auditContext });
      this.#recordAudit({
        operation: "close",
        phase: "outcome",
        outcome: "rejected",
        reason: "not_owned_or_missing",
        ...auditContext,
      });
      return false;
    }
    this.#close("user_kill", id, session);
    return true;
  }

  cleanupOwner(ownerId: number): void {
    for (const [id, session] of this.#sessions) {
      if (session.owner.id === ownerId) this.#close("owner_cleanup", id, session);
    }
  }

  dispose(): void {
    for (const [id, session] of this.#sessions) this.#close("app_dispose", id, session);
  }

  #ownedSession(ownerId: number, id: string): TerminalSession | undefined {
    const session = this.#sessions.get(id);
    return session?.owner.id === ownerId ? session : undefined;
  }

  #close(closeReason: TerminalAuditCloseReason, id: string, session: TerminalSession): void {
    const auditContext = { terminalId: id, ownerId: session.owner.id, closeReason };
    this.#recordAudit({ operation: "close", phase: "attempt", ...auditContext });
    try {
      this.#sessions.delete(id);
      session.dataSubscription.dispose();
      session.exitSubscription.dispose();
      session.process.kill();
    } catch (error) {
      this.#recordAudit({
        operation: "close",
        phase: "outcome",
        outcome: "failed",
        reason: "operation_failed",
        ...auditContext,
      });
      throw error;
    }
    this.#recordAudit({
      operation: "close",
      phase: "outcome",
      outcome: "succeeded",
      ...auditContext,
    });
  }

  #recordAudit(event: TerminalAuditEvent): void {
    this.audit?.(event);
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

import type { Context } from "@deepseek-ai/cordis";
import { SessionId } from "@deepseek-ai/dsh-session";
import { TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";
import { DvcError } from "@swarmx/dsh-dvc";
import { type DvcUiSnapshot, dvcUiSnapshotSchema } from "./contracts.js";

export class DvcUiError extends Error {
  readonly code: "SESSION_NOT_FOUND" | "WORKSPACE_UNAVAILABLE";

  constructor(
    message: string,
    code: "SESSION_NOT_FOUND" | "WORKSPACE_UNAVAILABLE",
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "DvcUiError";
    this.code = code;
  }
}

declare module "@deepseek-ai/cordis" {
  interface Context {
    dvcUi: DvcUiService;
  }
}

export class DvcUiService extends TypertRemoteService {
  static inject = ["dvc", "sessions"];

  constructor(ctx: Context) {
    super(ctx, "dvcUi");
  }

  async snapshot(sessionId: SessionId, signal?: AbortSignal): Promise<DvcUiSnapshot> {
    signal?.throwIfAborted();
    const cwd = this.workspace(sessionId);
    try {
      const inspection = await this.ctx.dvc.inspect(cwd, signal);
      return dvcUiSnapshotSchema.parse({ kind: "project", inspection });
    } catch (error) {
      if (signal?.aborted) signal.throwIfAborted();
      if (error instanceof DvcError) {
        if (error.code === "NOT_A_DVC_REPOSITORY" || error.code === "DVC_GIT_INVALID") {
          return { kind: "not-project", message: "Workspace is not a DVC project" };
        }
        if (error.code === "DVC_UNAVAILABLE" || error.code === "DVC_GIT_UNAVAILABLE") {
          return { kind: "unavailable", message: "DVC executable is unavailable" };
        }
      }
      throw error;
    }
  }

  private workspace(sessionId: SessionId): string {
    const session = this.ctx.sessions.get(SessionId(sessionId));
    if (!session) throw new DvcUiError("Live session not found", "SESSION_NOT_FOUND");
    const cwd = session.header.cwd;
    if (!cwd) throw new DvcUiError("Session has no workspace directory", "WORKSPACE_UNAVAILABLE");
    return cwd;
  }
}

export * from "./contracts.js";
export { DVC_UI_REMOTE } from "./remote.js";
export default DvcUiService;

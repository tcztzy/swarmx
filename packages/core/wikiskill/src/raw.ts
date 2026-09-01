import { type SessionEvent, SessionId } from "@deepseek-ai/dsh-session";
import type { SessionQueryEngine } from "@deepseek-ai/dsh-session-query";
import { type DshRawTraceLocator, dshRawTraceLocatorSchema } from "./contracts.js";
import { WikiSkillError } from "./errors.js";

export interface DshRawTrace {
  readonly events: readonly SessionEvent[];
  readonly locator: DshRawTraceLocator;
  readonly trust: "untrusted-execution-trace";
}

export interface DshRawTraceReaderConfig {
  readonly maxEvents?: number;
}

function invalidRequest(message: string, cause?: unknown): WikiSkillError {
  return new WikiSkillError(
    message,
    "WIKISKILL_INVALID_REQUEST",
    cause === undefined ? undefined : { cause },
  );
}

export class DshRawTraceReader {
  private readonly maxEvents: number;

  constructor(
    private readonly sessionQuery: Pick<SessionQueryEngine, "readSession">,
    config: DshRawTraceReaderConfig = {},
  ) {
    this.maxEvents = config.maxEvents ?? 256;
    if (!Number.isSafeInteger(this.maxEvents) || this.maxEvents < 1) {
      throw invalidRequest("WikiSkill Raw maxEvents must be a positive safe integer");
    }
  }

  async read(rawLocator: DshRawTraceLocator, signal?: AbortSignal): Promise<DshRawTrace> {
    let locator: DshRawTraceLocator;
    try {
      locator = dshRawTraceLocatorSchema.parse(rawLocator);
    } catch (cause) {
      throw invalidRequest("Invalid WikiSkill Raw locator", cause);
    }
    const count = locator.endSeq - locator.startSeq + 1;
    if (count > this.maxEvents) {
      throw invalidRequest(`WikiSkill Raw window exceeds ${String(this.maxEvents)} events`);
    }
    signal?.throwIfAborted();
    const snapshot = await this.sessionQuery.readSession(SessionId(locator.sessionId));
    signal?.throwIfAborted();
    if (String(snapshot.session.id) !== locator.sessionId) {
      throw new WikiSkillError(
        "DSH Raw Session identity changed during read",
        "WIKISKILL_IO_ERROR",
      );
    }
    const events = snapshot.events.filter(
      ({ seq }) => seq >= locator.startSeq && seq <= locator.endSeq,
    );
    if (
      events.length !== count ||
      events.some(({ seq }, index) => seq !== locator.startSeq + index)
    ) {
      throw new WikiSkillError(
        "DSH Raw event window is missing or non-contiguous",
        "WIKISKILL_RAW_GAP",
      );
    }
    return {
      events: structuredClone(events),
      locator,
      trust: "untrusted-execution-trace",
    };
  }
}

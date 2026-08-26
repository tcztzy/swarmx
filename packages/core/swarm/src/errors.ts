export type SwarmErrorCode =
  | "SWARM_ARCHIVED"
  | "SWARM_CLOSED"
  | "SWARM_INVALID_REQUEST"
  | "SWARM_LIMIT"
  | "SWARM_MEMBER_NOT_FOUND"
  | "SWARM_NOT_FOUND"
  | "SWARM_STALE_ATTEMPT"
  | "SWARM_STALE_REVISION"
  | "SWARM_TASK_DEPENDENCY"
  | "SWARM_TASK_NOT_FOUND"
  | "SWARM_UNAUTHORIZED";

export class SwarmError extends Error {
  readonly code: SwarmErrorCode;

  constructor(message: string, code: SwarmErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "SwarmError";
    this.code = code;
  }
}

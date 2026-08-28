export type SwarmErrorCode =
  | "SWARM_ARCHIVED"
  | "SWARM_CLOSED"
  | "SWARM_ADMISSION_CONFLICT"
  | "SWARM_DUPLICATE_EFFECT"
  | "SWARM_EFFECT_UNCERTAIN"
  | "SWARM_INVALID_REQUEST"
  | "SWARM_LIMIT"
  | "SWARM_MEMBER_NOT_FOUND"
  | "SWARM_MESSAGE_CONFLICT"
  | "SWARM_NOT_FOUND"
  | "SWARM_STALE_ATTEMPT"
  | "SWARM_STALE_REVISION"
  | "SWARM_STALE_SUBMISSION"
  | "SWARM_TASK_DEPENDENCY"
  | "SWARM_TASK_NOT_FOUND"
  | "SWARM_UNAUTHORIZED"
  | "SWARM_VERIFICATION_REQUIRED";

export class SwarmError extends Error {
  readonly code: SwarmErrorCode;

  constructor(message: string, code: SwarmErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "SwarmError";
    this.code = code;
  }
}

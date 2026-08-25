export type PkbErrorCode =
  | "AUTHORIZATION_REQUIRED"
  | "INVALID_REQUEST"
  | "INVALID_CONCEPT"
  | "CONCEPT_NOT_FOUND"
  | "REVISION_CONFLICT"
  | "UNSAFE_PATH"
  | "WORKSPACE_UNAVAILABLE"
  | "IO_ERROR";

export class PkbError extends Error {
  readonly code: PkbErrorCode;

  constructor(message: string, code: PkbErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "PkbError";
    this.code = code;
  }
}

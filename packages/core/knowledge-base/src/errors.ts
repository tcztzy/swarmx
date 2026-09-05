export type KnowledgeBaseErrorCode =
  | "AUTHORIZATION_REQUIRED"
  | "INVALID_REQUEST"
  | "INVALID_CONCEPT"
  | "CONCEPT_NOT_FOUND"
  | "REVISION_CONFLICT"
  | "UNSAFE_PATH"
  | "WORKSPACE_UNAVAILABLE"
  | "IO_ERROR";

export class KnowledgeBaseError extends Error {
  readonly code: KnowledgeBaseErrorCode;

  constructor(message: string, code: KnowledgeBaseErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "KnowledgeBaseError";
    this.code = code;
  }
}

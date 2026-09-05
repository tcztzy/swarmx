export type MemoryErrorCode =
  | "AUTHORIZATION_REQUIRED"
  | "INVALID_REQUEST"
  | "INVALID_CONCEPT"
  | "CONCEPT_NOT_FOUND"
  | "REVISION_CONFLICT"
  | "UNSAFE_PATH"
  | "WORKSPACE_UNAVAILABLE"
  | "IO_ERROR";

export interface MemoryIssue {
  readonly ruleId: string;
  readonly severity: "error" | "warning";
  readonly line: number;
  readonly column: number;
  readonly message: string;
}

export class MemoryError extends Error {
  readonly code: MemoryErrorCode;
  readonly issues: readonly MemoryIssue[];

  constructor(
    message: string,
    code: MemoryErrorCode,
    options?: ErrorOptions & { readonly issues?: readonly MemoryIssue[] },
  ) {
    const issues = options?.issues ?? [];
    super(
      [
        message,
        ...issues.map(
          (issue) =>
            `${issue.ruleId} (${String(issue.line)}:${String(issue.column)}): ${issue.message}`,
        ),
      ].join("\n"),
      options,
    );
    this.name = "MemoryError";
    this.code = code;
    this.issues = issues;
  }
}

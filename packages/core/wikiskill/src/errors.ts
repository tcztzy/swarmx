export type WikiSkillErrorCode =
  | "WIKISKILL_INVALID_REQUEST"
  | "WIKISKILL_INVALID_SKILL"
  | "WIKISKILL_IO_ERROR"
  | "WIKISKILL_PROPOSAL_NOT_FOUND"
  | "WIKISKILL_RAW_GAP"
  | "WIKISKILL_REVISION_CONFLICT";

export class WikiSkillError extends Error {
  readonly code: WikiSkillErrorCode;

  constructor(message: string, code: WikiSkillErrorCode, options?: ErrorOptions) {
    super(message, options);
    this.name = "WikiSkillError";
    this.code = code;
  }
}

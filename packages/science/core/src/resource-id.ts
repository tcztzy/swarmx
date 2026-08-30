import { ScienceError } from "./errors.js";

export const SCIENCE_RESOURCE_KINDS = [
  "project",
  "notebook",
  "artifact",
  "document",
  "figure",
  "record",
  "experiment",
  "run",
] as const;

export type ScienceResourceKind = (typeof SCIENCE_RESOURCE_KINDS)[number];

export interface ParsedScienceResourceId {
  readonly kind: ScienceResourceKind;
  readonly entityId: string;
  readonly revision: number | null;
}

const KIND_PREFIX = {
  project: "p",
  notebook: "n",
  artifact: "a",
  document: "d",
  figure: "f",
  record: "r",
  experiment: "e",
  run: "x",
} as const satisfies Record<ScienceResourceKind, string>;

const PREFIX_KIND: ReadonlyMap<string, ScienceResourceKind> = new Map(
  Object.entries(KIND_PREFIX).map(([kind, prefix]) => [prefix, kind as ScienceResourceKind]),
);
const RESOURCE_ID_PATTERN = /^sx:([a-z])\/([^@]+)(?:@([1-9]\d*))?$/u;
const MAX_ENTITY_ID_LENGTH = 200;

function invalidResourceId(cause?: unknown): ScienceError {
  return new ScienceError(
    "Science resource ID is invalid or non-canonical",
    "INVALID_RESOURCE_ID",
    {
      ...(cause === undefined ? {} : { cause }),
    },
  );
}

function validateRevision(revision: number | null): void {
  if (revision !== null && (!Number.isSafeInteger(revision) || revision <= 0)) {
    throw invalidResourceId();
  }
}

/** Format one canonical typed local Science resource ID. */
export function formatScienceResourceId(
  kind: ScienceResourceKind,
  entityId: string,
  revision: number | null = null,
): string {
  const prefix = KIND_PREFIX[kind];
  if (
    prefix === undefined ||
    entityId.length === 0 ||
    entityId.length > MAX_ENTITY_ID_LENGTH ||
    entityId.includes("\0")
  ) {
    throw invalidResourceId();
  }
  validateRevision(revision);
  let encodedEntityId: string;
  try {
    encodedEntityId = encodeURIComponent(entityId);
  } catch (error) {
    throw invalidResourceId(error);
  }
  const logical = `sx:${prefix}/${encodedEntityId}`;
  const formatted = revision === null ? logical : `${logical}@${revision}`;
  if (formatted.length > 1_024) throw invalidResourceId();
  return formatted;
}

/** Parse only the canonical local `sx:` namespace; bare or ambiguous IDs fail closed. */
export function parseScienceResourceId(value: string): ParsedScienceResourceId {
  if (typeof value !== "string" || value.length > 1_024) throw invalidResourceId();
  const match = RESOURCE_ID_PATTERN.exec(value);
  if (!match) throw invalidResourceId();
  const [, prefix = "", encodedEntityId = "", revisionText] = match;
  const kind = PREFIX_KIND.get(prefix);
  if (!kind) throw invalidResourceId();

  let entityId: string;
  try {
    entityId = decodeURIComponent(encodedEntityId);
  } catch (error) {
    throw invalidResourceId(error);
  }
  const revision = revisionText === undefined ? null : Number(revisionText);
  validateRevision(revision);
  if (
    entityId.length === 0 ||
    entityId.length > MAX_ENTITY_ID_LENGTH ||
    entityId.includes("\0") ||
    formatScienceResourceId(kind, entityId, revision) !== value
  ) {
    throw invalidResourceId();
  }
  return { kind, entityId, revision };
}

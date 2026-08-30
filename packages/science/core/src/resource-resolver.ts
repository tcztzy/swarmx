import type {
  ScienceArtifact,
  ScienceDocument,
  ScienceExperiment,
  ScienceFigure,
  ScienceNotebook,
  ScienceProject,
  ScienceResearchRecord,
  ScienceRun,
  ScienceWorkspaceSnapshot,
} from "./contracts.js";
import { ScienceError } from "./errors.js";
import {
  formatScienceResourceId,
  parseScienceResourceId,
  type ScienceResourceKind,
} from "./resource-id.js";

export interface ScienceResourceRef {
  readonly id: string;
  readonly exactId: string;
  readonly kind: ScienceResourceKind;
  readonly title: string;
  readonly revision: number;
  readonly digest: `sha256:${string}` | null;
}

export type ScienceResourceEntity =
  | ScienceProject
  | ScienceNotebook
  | ScienceArtifact
  | ScienceDocument
  | ScienceFigure
  | ScienceResearchRecord
  | ScienceExperiment
  | ScienceRun;

export interface ResolvedScienceResource {
  readonly entityId: string;
  readonly entity: ScienceResourceEntity;
  readonly kind: ScienceResourceKind;
  readonly ref: ScienceResourceRef;
}

function resourceTitle(kind: ScienceResourceKind, entity: ScienceResourceEntity): string {
  if (kind === "document") return (entity as ScienceDocument).name;
  if (kind === "run") return "Experiment run";
  return (entity as Exclude<ScienceResourceEntity, ScienceDocument | ScienceRun>).title;
}

function resourceDigest(
  kind: ScienceResourceKind,
  entity: ScienceResourceEntity,
): `sha256:${string}` | null {
  if (kind === "artifact") return (entity as ScienceArtifact).digest as `sha256:${string}`;
  if (kind === "document") {
    return (
      ((entity as ScienceDocument).revisions.at(-1)?.sourceHash as
        | `sha256:${string}`
        | undefined) ?? null
    );
  }
  if (kind === "figure") {
    return (
      ((entity as ScienceFigure).revisions.at(-1)?.codeHash as `sha256:${string}` | undefined) ??
      null
    );
  }
  return null;
}

/** Read-only typed index over one already-authorized Science workspace snapshot. */
export class ScienceResourceResolver {
  private readonly byKind = new Map<ScienceResourceKind, Map<string, ScienceResourceEntity>>();
  private readonly kindsByEntityId = new Map<string, Set<ScienceResourceKind>>();

  constructor(readonly snapshot: ScienceWorkspaceSnapshot) {
    this.add("project", snapshot.projects);
    this.add("notebook", snapshot.notebooks);
    this.add("artifact", snapshot.artifacts);
    this.add("document", snapshot.documents);
    this.add("figure", snapshot.figures);
    this.add("record", snapshot.records);
    this.add("experiment", snapshot.experiments);
    this.add("run", snapshot.runs);
  }

  resolve(id: string): ResolvedScienceResource {
    const parsed = parseScienceResourceId(id);
    const entity = this.byKind.get(parsed.kind)?.get(parsed.entityId);
    if (!entity) {
      const presentKinds = this.kindsByEntityId.get(parsed.entityId);
      if (presentKinds && presentKinds.size > 0) {
        throw new ScienceError(
          "Science resource ID kind does not match the workspace entity",
          "RESOURCE_KIND_MISMATCH",
        );
      }
      throw new ScienceError(
        "Science resource was not found in this workspace",
        "RESOURCE_NOT_FOUND",
      );
    }
    if (parsed.revision !== null && parsed.revision !== entity.revision) {
      throw new ScienceError(
        `Science resource revision ${parsed.revision} does not match current revision ${entity.revision}`,
        "RESOURCE_REVISION_MISMATCH",
      );
    }
    const logicalId = formatScienceResourceId(parsed.kind, parsed.entityId);
    return {
      entityId: parsed.entityId,
      entity,
      kind: parsed.kind,
      ref: {
        id: logicalId,
        exactId: formatScienceResourceId(parsed.kind, parsed.entityId, entity.revision),
        kind: parsed.kind,
        title: resourceTitle(parsed.kind, entity),
        revision: entity.revision,
        digest: resourceDigest(parsed.kind, entity),
      },
    };
  }

  resolveUntypedEntityId(entityId: string): ResolvedScienceResource | null {
    const kinds = this.kindsByEntityId.get(entityId);
    if (!kinds || kinds.size === 0) return null;
    if (kinds.size !== 1) {
      throw new ScienceError(
        "Untyped Science relation target is ambiguous across resource kinds",
        "RESOURCE_INDEX_CONFLICT",
      );
    }
    const [kind] = kinds;
    if (!kind) throw new Error("Science resource kind index disappeared");
    return this.resolve(formatScienceResourceId(kind, entityId));
  }

  private add(kind: ScienceResourceKind, entities: readonly ScienceResourceEntity[]): void {
    const index = new Map<string, ScienceResourceEntity>();
    for (const entity of entities) {
      if (index.has(entity.id)) {
        throw new ScienceError(
          "Science workspace contains duplicate resources for one typed ID",
          "RESOURCE_INDEX_CONFLICT",
        );
      }
      index.set(entity.id, entity);
      const kinds = this.kindsByEntityId.get(entity.id) ?? new Set<ScienceResourceKind>();
      kinds.add(kind);
      this.kindsByEntityId.set(entity.id, kinds);
    }
    this.byKind.set(kind, index);
  }
}

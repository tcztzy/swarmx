import type { AdmitKnowledgeRequest, EvidenceSource, KnowledgeCommitReceipt } from "./contracts.js";
import type { KnowledgeCommitContext, KnowledgeCommitter, SwarmActor } from "./coordinator.js";
import { SwarmError } from "./errors.js";

interface ScienceOwner {
  linkEvidence(
    actorId: string,
    request: {
      requestId: string;
      projectId: string;
      claimId: string;
      relation: "supports" | "refutes";
      title: string;
      summary: string;
      tags: string[];
      sourceEntityIds: string[];
    },
    signal?: AbortSignal,
  ): {
    evidence: { id: string };
    provenance: { journalSeq: number };
  };
}

interface PkbOwner {
  vault: {
    createConcept(
      cwd: string,
      request: {
        requestId: string;
        scope: "global" | "workspace";
        title: string;
        description: string;
        type: string;
        body: string;
        tags?: string[];
        aliases?: string[];
        status?: "draft" | "stable";
        sources: Array<Record<string, unknown>>;
      },
    ): Promise<{ id: string; revision: string }>;
  };
}

interface ApprovalOwner {
  request(input: {
    actorId: string;
    callId: string;
    reason: string;
    signal: AbortSignal;
    toolName: string;
  }): Promise<string>;
}

export interface KnowledgeOwners {
  readonly approval: ApprovalOwner;
  readonly pkb: PkbOwner;
  readonly science: ScienceOwner;
  workspaceRoot(actorId: string): string | undefined;
}

function pkbSource(source: EvidenceSource): Record<string, unknown> {
  if (source.kind === "science_entity") {
    return { resource: `urn:uuid:${source.entityId}`, title: "Science Journal entity" };
  }
  return {
    resource: source.resource,
    ...(source.title ? { title: source.title } : {}),
    ...(source.digest ? { digest: source.digest } : {}),
  };
}

export class OwnerKnowledgeCommitter implements KnowledgeCommitter {
  constructor(private readonly owners: KnowledgeOwners) {}

  async commit(
    lead: SwarmActor,
    request: AdmitKnowledgeRequest,
    context: KnowledgeCommitContext,
  ): Promise<KnowledgeCommitReceipt> {
    context.signal.throwIfAborted();
    if (request.target.kind === "science_evidence") {
      const sourceEntityIds = request.sources.map((source) => {
        if (source.kind !== "science_entity") {
          throw new SwarmError(
            "Science evidence requires Science entity sources",
            "SWARM_INVALID_REQUEST",
          );
        }
        return source.entityId;
      });
      const linked = this.owners.science.linkEvidence(
        lead.id,
        {
          requestId: request.admissionId,
          projectId: request.target.projectId,
          claimId: request.target.claimId,
          relation: request.target.relation,
          title: request.target.title,
          summary: request.target.summary,
          tags: request.target.tags,
          sourceEntityIds,
        },
        context.signal,
      );
      return {
        kind: "science_evidence",
        entityId: linked.evidence.id,
        journalSequence: linked.provenance.journalSeq,
      };
    }

    const cwd = this.owners.workspaceRoot(lead.id);
    if (!cwd) {
      throw new SwarmError("Knowledge admission requires a workspace", "SWARM_UNAUTHORIZED");
    }
    const approval = await this.owners.approval.request({
      actorId: lead.id,
      callId: context.callId,
      reason: "Admit one verified Team candidate into the private PKB.",
      signal: context.signal,
      toolName: "swarm",
    });
    if (approval !== "allowed-once") {
      throw new SwarmError(
        `PKB evidence admission was not approved (${approval})`,
        "SWARM_UNAUTHORIZED",
      );
    }
    const concept = await this.owners.pkb.vault.createConcept(cwd, {
      requestId: request.admissionId,
      scope: request.target.scope,
      title: request.target.title,
      description: request.target.description,
      type: request.target.type,
      body: request.target.body,
      sources: request.sources.map(pkbSource),
      ...(request.target.tags ? { tags: request.target.tags } : {}),
      ...(request.target.aliases ? { aliases: request.target.aliases } : {}),
      ...(request.target.status ? { status: request.target.status } : {}),
    });
    return {
      kind: "pkb_concept",
      conceptId: concept.id,
      revision: concept.revision as `sha256:${string}`,
    };
  }
}

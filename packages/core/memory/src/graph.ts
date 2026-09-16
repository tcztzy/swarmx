import { MemoryError } from "./errors.js";
import type { MemoryConcept } from "./vault.js";

export function dependencyOrder(concepts: readonly MemoryConcept[], roots: readonly string[]) {
  const byId = new Map(concepts.map((concept) => [concept.id, concept]));
  const ordered: MemoryConcept[] = [];
  const active = new Set<string>();
  const visited = new Set<string>();
  const visit = (id: string) => {
    if (active.has(id)) throw new MemoryError("Memory dependency cycle.", "INVALID_CONCEPT");
    if (visited.has(id)) return;
    const concept = byId.get(id);
    if (!concept) throw new MemoryError("Memory dependency is missing.", "INVALID_CONCEPT");
    active.add(id);
    for (const dependency of concept.metadata.swarmx_dependencies ?? []) {
      visit(dependency.id);
    }
    active.delete(id);
    visited.add(id);
    ordered.push(concept);
  };
  for (const id of roots) visit(id);
  return ordered;
}

export function memoryGraph(concepts: readonly MemoryConcept[], now = Date.now()) {
  const ordered = dependencyOrder(
    concepts,
    concepts.map(({ id }) => id),
  );
  const byId = new Map(concepts.map((concept) => [concept.id, concept]));
  const stale = new Set<string>();
  const edges = ordered.flatMap((concept) => {
    if (
      concept.metadata.status === "deprecated" ||
      (concept.metadata.stale_after && Date.parse(concept.metadata.stale_after) <= now)
    )
      stale.add(concept.id);
    return (concept.metadata.swarmx_dependencies ?? []).map((dependency) => {
      const outdated =
        byId.get(dependency.id)?.revision !== dependency.revision || stale.has(dependency.id);
      if (outdated) stale.add(concept.id);
      return {
        source: concept.id,
        target: dependency.id,
        revision: dependency.revision,
        stale: outdated,
      };
    });
  });
  return {
    nodes: ordered.map(({ id, revision, metadata }) => ({
      id,
      revision,
      title: metadata.title,
      description: metadata.description,
      type: metadata.type,
      status: metadata.status,
      stale: stale.has(id),
    })),
    edges,
  };
}

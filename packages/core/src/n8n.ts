import type { EdgeConfig, SwarmConfig } from "./types.js";
import { SwarmConfigSchema } from "./types.js";

const DEFAULT_IMPORTED_MODEL = "gpt-4o";
const CYCLE_EDGE_CONDITION = "false";

export class N8nImportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "N8nImportError";
  }
}

export interface N8nImportResult {
  config: SwarmConfig;
  warnings: string[];
  nodeMap: Record<string, string>;
}

interface N8nNode {
  raw: Record<string, unknown>;
  id?: string;
  name: string;
  type: string;
}

interface NodeMapping {
  n8n: N8nNode;
  swarmId: string;
}

export function importN8nWorkflow(source: unknown): N8nImportResult {
  const warnings: string[] = [
    "Imported n8n workflow as SwarmX agent nodes. Native n8n node execution is not imported.",
    "Credential references are preserved, but credential secrets are not imported.",
  ];
  const parsed = parseImportSource(source);
  const workflow = normalizeWorkflow(parsed);
  const n8nNodes = readN8nNodes(workflow);
  const mappings = buildNodeMappings(n8nNodes);
  const nodeMap = Object.fromEntries(
    mappings.map((mapping) => [mapping.n8n.name, mapping.swarmId]),
  );
  const edges = readN8nEdges(workflow.connections, mappings, warnings);
  const root = deriveRoot(mappings, edges);
  const config = SwarmConfigSchema.parse({
    name: readString(workflow.record, "name") || "imported_n8n_workflow",
    description: "Imported from n8n workflow JSON.",
    root,
    nodes: Object.fromEntries(
      mappings.map((mapping) => [
        mapping.swarmId,
        toSwarmAgentNode(mapping.n8n, mapping.swarmId, workflow.record),
      ]),
    ),
    edges,
    parameters: {
      n8n: compactRecord({
        id: readString(workflow.record, "id") || undefined,
        name: readString(workflow.record, "name") || undefined,
        active: readBoolean(workflow.record, "active"),
        sourceN8nVersion: readString(workflow.record, "sourceN8nVersion") || undefined,
        settings: readRecord(workflow.record.settings),
        tags: Array.isArray(workflow.record.tags) ? workflow.record.tags : undefined,
      }),
    },
  });

  if (edges.length === 0) {
    warnings.push(
      "No n8n connections were imported; the SwarmX workflow will run only the root node.",
    );
  }

  return { config, warnings, nodeMap };
}

function parseImportSource(source: unknown): unknown {
  if (typeof source !== "string") return source;
  const trimmed = source.trim();
  if (!trimmed) {
    throw new N8nImportError("n8n workflow JSON is empty.");
  }

  try {
    return JSON.parse(trimmed);
  } catch (error) {
    throw new N8nImportError(`n8n workflow JSON parse error: ${errorMessage(error)}`);
  }
}

function normalizeWorkflow(source: unknown): {
  record: Record<string, unknown>;
  nodes: unknown;
  connections: unknown;
} {
  if (Array.isArray(source)) {
    return {
      record: { name: "imported_n8n_selection" },
      nodes: source,
      connections: undefined,
    };
  }

  if (!isRecord(source)) {
    throw new N8nImportError("n8n workflow JSON must be an object with a nodes array.");
  }

  if (!Array.isArray(source.nodes)) {
    throw new N8nImportError("n8n workflow JSON needs a nodes array.");
  }

  return {
    record: source,
    nodes: source.nodes,
    connections: source.connections,
  };
}

function readN8nNodes(workflow: { nodes: unknown }): N8nNode[] {
  if (!Array.isArray(workflow.nodes) || workflow.nodes.length === 0) {
    throw new N8nImportError("n8n workflow JSON needs at least one node.");
  }

  return workflow.nodes.map((node, index) => {
    if (!isRecord(node)) {
      throw new N8nImportError(`n8n node ${index + 1} must be an object.`);
    }

    const name = readString(node, "name");
    if (!name) {
      throw new N8nImportError(`n8n node ${index + 1} needs a name.`);
    }

    return {
      raw: node,
      id: readString(node, "id") || undefined,
      name,
      type: readString(node, "type") || "n8n-node",
    };
  });
}

function buildNodeMappings(nodes: N8nNode[]): NodeMapping[] {
  const used = new Set<string>();
  return nodes.map((node) => {
    const swarmId = uniqueIdentifier(sanitizeIdentifier(node.name), used);
    return { n8n: node, swarmId };
  });
}

function readN8nEdges(
  connections: unknown,
  mappings: NodeMapping[],
  warnings: string[],
): EdgeConfig[] {
  if (connections === undefined || connections === null) return [];
  if (!isRecord(connections)) {
    warnings.push("Ignored n8n connections because they are not an object.");
    return [];
  }

  const nameToId = new Map(mappings.map((mapping) => [mapping.n8n.name, mapping.swarmId]));
  const importedEdges: EdgeConfig[] = [];
  const executableEdges: Array<{ source: string; target: string }> = [];
  const seen = new Set<string>();

  for (const [sourceName, outputs] of Object.entries(connections)) {
    const source = nameToId.get(sourceName);
    if (!source) {
      warnings.push(`Ignored n8n connections from missing source node "${sourceName}".`);
      continue;
    }
    if (!isRecord(outputs)) continue;

    for (const [outputType, outputGroups] of Object.entries(outputs)) {
      if (!Array.isArray(outputGroups)) continue;

      outputGroups.forEach((group, outputIndex) => {
        if (!Array.isArray(group)) return;

        group.forEach((connection, connectionIndex) => {
          if (!isRecord(connection)) return;
          const targetName = readString(connection, "node");
          if (!targetName) return;

          const target = nameToId.get(targetName);
          if (!target) {
            warnings.push(
              `Ignored n8n connection from "${sourceName}" to missing target node "${targetName}".`,
            );
            return;
          }

          const edgeKey = `${source}\u001f${target}`;
          if (seen.has(edgeKey)) return;
          seen.add(edgeKey);

          const edge: EdgeConfig = { source, target };
          if (createsCycle(executableEdges, { source, target })) {
            edge.condition = CYCLE_EDGE_CONDITION;
            warnings.push(
              `Imported cycle-forming n8n connection "${sourceName}" -> "${targetName}" as a disabled SwarmX edge.`,
            );
          } else {
            executableEdges.push({ source, target });
          }

          if (outputType !== "main" || outputIndex > 0 || connectionIndex > 0) {
            warnings.push(
              `Flattened n8n ${outputType} connection "${sourceName}" output ${outputIndex} to "${targetName}".`,
            );
          }

          importedEdges.push(edge);
        });
      });
    }
  }

  return importedEdges;
}

function deriveRoot(mappings: NodeMapping[], edges: EdgeConfig[]): string {
  const incoming = new Set(edges.filter((edge) => !edge.condition).map((edge) => edge.target));
  const roots = mappings.filter((mapping) => !incoming.has(mapping.swarmId));
  return (
    roots.find((mapping) => isTriggerNode(mapping.n8n))?.swarmId ??
    roots[0]?.swarmId ??
    mappings.find((mapping) => isTriggerNode(mapping.n8n))?.swarmId ??
    mappings[0].swarmId
  );
}

function toSwarmAgentNode(
  node: N8nNode,
  swarmId: string,
  workflow: Record<string, unknown>,
): SwarmConfig["nodes"][string] {
  return {
    kind: "agent",
    agent: {
      name: swarmId,
      description: `Imported n8n node "${node.name}" (${node.type}).`,
      model: inferModel(node.raw) ?? DEFAULT_IMPORTED_MODEL,
      backend: { type: "swarmx" },
      parameters: {
        harness: n8nHarnessDescriptor(workflow),
        n8n: n8nNodeMetadata(node),
      },
      instructions:
        "This node was imported from n8n as a structural SwarmX step. " +
        "Use parameters.n8n to inspect the original node configuration. " +
        "Native n8n execution semantics and credentials are not available unless explicitly reimplemented.",
    },
  };
}

function n8nHarnessDescriptor(workflow: Record<string, unknown>): Record<string, unknown> {
  return {
    software: compactRecord({
      name: "n8n-import",
      version: readString(workflow, "sourceN8nVersion") || undefined,
    }),
    mcps: [],
    skills: [],
    projectFiles: [],
  };
}

function n8nNodeMetadata(node: N8nNode): Record<string, unknown> {
  const raw = node.raw;
  return compactRecord({
    id: node.id,
    name: node.name,
    type: node.type,
    typeVersion: raw.typeVersion,
    position: readPosition(raw.position),
    parameters: readRecord(raw.parameters) ?? {},
    credentials: credentialReferences(raw.credentials),
    disabled: readBoolean(raw, "disabled"),
    notes: readString(raw, "notes") || undefined,
    notesInFlow: readBoolean(raw, "notesInFlow"),
    retryOnFail: readBoolean(raw, "retryOnFail"),
    continueOnFail: readBoolean(raw, "continueOnFail"),
  });
}

function credentialReferences(value: unknown): Record<string, unknown> | undefined {
  if (!isRecord(value)) return undefined;

  const refs: Record<string, unknown> = {};
  for (const [credentialType, credential] of Object.entries(value)) {
    if (isRecord(credential)) {
      refs[credentialType] = compactRecord({
        type: readString(credential, "type") || credentialType,
        id: readString(credential, "id") || undefined,
        name: readString(credential, "name") || undefined,
      });
    } else if (typeof credential === "string" && credential.trim() !== "") {
      refs[credentialType] = {
        type: credentialType,
        id: credential,
      };
    }
  }

  return Object.keys(refs).length > 0 ? refs : undefined;
}

function inferModel(node: Record<string, unknown>): string | undefined {
  const parameters = readRecord(node.parameters);
  if (!parameters) return undefined;

  for (const key of ["model", "modelName", "chatModel", "textModel"]) {
    const value = readString(parameters, key);
    if (value) return value;
  }

  return findStringByKey(parameters, "model");
}

function findStringByKey(value: unknown, needle: string): string | undefined {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findStringByKey(item, needle);
      if (found) return found;
    }
    return undefined;
  }
  if (!isRecord(value)) return undefined;

  for (const [key, item] of Object.entries(value)) {
    if (key.toLowerCase().includes(needle) && typeof item === "string" && item.trim() !== "") {
      return item;
    }
    const found = findStringByKey(item, needle);
    if (found) return found;
  }

  return undefined;
}

function createsCycle(
  edges: Array<{ source: string; target: string }>,
  candidate: { source: string; target: string },
): boolean {
  const adjacency = new Map<string, string[]>();
  for (const edge of [...edges, candidate]) {
    const targets = adjacency.get(edge.source) ?? [];
    targets.push(edge.target);
    adjacency.set(edge.source, targets);
  }

  const seen = new Set<string>();
  const stack = [candidate.target];
  while (stack.length > 0) {
    const node = stack.pop();
    if (!node || seen.has(node)) continue;
    if (node === candidate.source) return true;
    seen.add(node);
    stack.push(...(adjacency.get(node) ?? []));
  }

  return false;
}

function sanitizeIdentifier(value: string): string {
  const sanitized = value
    .trim()
    .replace(/[^A-Za-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .replace(/_+/g, "_");
  const base = sanitized || "node";
  return /^[A-Za-z]/.test(base) ? base : `n8n_${base}`;
}

function uniqueIdentifier(base: string, used: Set<string>): string {
  let candidate = base;
  let suffix = 2;
  while (used.has(candidate)) {
    candidate = `${base}_${suffix}`;
    suffix++;
  }
  used.add(candidate);
  return candidate;
}

function isTriggerNode(node: N8nNode): boolean {
  const source = `${node.name} ${node.type}`.toLowerCase();
  return source.includes("trigger") || source.includes("webhook");
}

function readPosition(value: unknown): unknown {
  if (!Array.isArray(value)) return undefined;
  if (value.length < 2) return undefined;
  const [x, y] = value;
  if (typeof x !== "number" || typeof y !== "number") return undefined;
  return [x, y];
}

function readRecord(value: unknown): Record<string, unknown> | undefined {
  return isRecord(value) ? value : undefined;
}

function readString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

function readBoolean(record: Record<string, unknown>, key: string): boolean | undefined {
  const value = record[key];
  return typeof value === "boolean" ? value : undefined;
}

function compactRecord(record: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(record).filter(([, value]) => value !== undefined));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

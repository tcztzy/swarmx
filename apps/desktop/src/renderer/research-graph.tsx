import type { RoCrateMetadataDocument } from "@swarmx/science/types";
import {
  Background,
  Controls,
  type Edge,
  MarkerType,
  type Node,
  Position,
  ReactFlow,
} from "@xyflow/react";
import { t, useTranslation } from "./i18n.js";
import "@xyflow/react/dist/style.css";

export function crateGraph(
  document: RoCrateMetadataDocument,
  query: string,
  selected: string,
  neighborhood: boolean,
) {
  const rootId = document["@graph"].find((entity) => entity["@id"] === "ro-crate-metadata.json")
    ?.about?.["@id"];
  const entities = document["@graph"].filter(
    (entity) => entity["@id"] !== "ro-crate-metadata.json",
  );
  const ids = new Set(entities.map((entity) => entity["@id"]));
  const edges: Edge[] = [];
  for (const entity of entities)
    for (const [relation, values] of Object.entries(entity)) {
      for (const value of Array.isArray(values) ? values : [values]) {
        if (
          typeof value !== "object" ||
          value === null ||
          !("@id" in value) ||
          typeof value["@id"] !== "string" ||
          !ids.has(value["@id"])
        )
          continue;
        edges.push({
          id: JSON.stringify([entity["@id"], relation, value["@id"]]),
          source: entity["@id"],
          target: value["@id"],
          label: relation,
          markerEnd: { type: MarkerType.ArrowClosed },
          type: "smoothstep",
        });
      }
    }
  const neighbors = new Set([
    selected,
    ...edges
      .filter((edge) => edge.source === selected || edge.target === selected)
      .flatMap((edge) => [edge.source, edge.target]),
  ]);
  const matches = entities.filter(
    (entity) =>
      (!neighborhood || !selected || neighbors.has(entity["@id"])) &&
      `${entity.name ?? ""} ${entity["@type"]} ${entity["@id"]}`
        .toLocaleLowerCase()
        .includes(query.trim().toLocaleLowerCase()),
  );
  const rows = [0, 0, 0, 0];
  const nodes: Node[] = matches.slice(0, 200).map((entity) => {
    const types = Array.isArray(entity["@type"]) ? entity["@type"].join(" · ") : entity["@type"];
    const column =
      entity["@id"] === rootId
        ? 0
        : /Action/u.test(types)
          ? 2
          : /ImageObject|MediaObject/u.test(types)
            ? 3
            : 1;
    const row = rows[column] ?? 0;
    rows[column] = row + 1;
    return {
      id: entity["@id"],
      data: {
        label: (
          <div className="text-left">
            <span className="mb-1 block text-[10px] text-neutral-400">{types}</span>
            <span className="line-clamp-2 text-xs font-medium">
              {String(entity.name ?? entity["@id"])}
            </span>
          </div>
        ),
      },
      position: { x: column * 280, y: row * 120 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      selected: entity["@id"] === selected,
      deletable: false,
    };
  });
  const visible = new Set(nodes.map((node) => node.id));
  return {
    nodes,
    edges: edges.filter((edge) => visible.has(edge.source) && visible.has(edge.target)),
    count: matches.length,
  };
}

export function ResearchGraph({
  document,
  query,
  selected,
  neighborhood,
  onSelect,
}: {
  document: RoCrateMetadataDocument;
  query: string;
  selected: string;
  neighborhood: boolean;
  onSelect(id: string): void;
}) {
  useTranslation();
  const graph = crateGraph(document, query, selected, neighborhood);
  return <GraphView graph={graph} onSelect={onSelect} label={t("研究关系图谱")} />;
}

export function GraphView({
  graph,
  onSelect,
  label,
}: {
  graph: { nodes: Node[]; edges: Edge[]; count: number };
  onSelect(id: string): void;
  label: string;
}) {
  useTranslation();
  return (
    <div className="relative h-full min-h-[360px] bg-neutral-50" aria-label={label}>
      <ReactFlow
        key={graph.nodes.map(({ id }) => id).join(":")}
        nodes={graph.nodes}
        edges={graph.edges}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesReconnectable={false}
        deleteKeyCode={null}
        fitView
        minZoom={0.2}
        maxZoom={1.8}
        onNodeClick={(_event, node) => onSelect(node.id)}
        ariaLabelConfig={{
          "controls.zoomIn.ariaLabel": t("放大"),
          "controls.zoomOut.ariaLabel": t("缩小"),
          "controls.fitView.ariaLabel": t("适应视图"),
          "controls.ariaLabel": t("图谱控制"),
        }}
      >
        <Background gap={22} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
      <p className="pointer-events-none absolute top-3 right-3 rounded-md border border-neutral-200 bg-white px-3 py-2 text-[11px] text-neutral-500">
        {t("{{count}} 个实体 · {{edges}} 条关系", {
          count: graph.nodes.length,
          edges: graph.edges.length,
        })}
        {graph.count > 200 ? t(" · 仅显示前 200 个，请搜索缩小范围") : t(" · 点击节点查看详情")}
      </p>
    </div>
  );
}

import type { ScienceArtifactPreview } from "@swarmx/dsh-science/types";
import { AllCommunityModule, type ColDef, ModuleRegistry, themeQuartz } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";

ModuleRegistry.registerModules([AllCommunityModule]);

type TablePreview = Extract<ScienceArtifactPreview, { kind: "table" }>;
type TableRow = Record<string, boolean | number | string | null>;

export function scienceColumnTypeToGrid(
  type: TablePreview["columns"][number]["type"],
): "boolean" | "number" | "text" {
  return type === "string" ? "text" : type;
}

export function ScienceTableGrid({ preview }: { readonly preview: TablePreview }) {
  const rowData: TableRow[] = preview.rows.map((row) =>
    Object.fromEntries(preview.columns.map((column, index) => [column.id, row[index] ?? null])),
  );
  const columnDefs: ColDef<TableRow>[] = preview.columns.map((column, index) => ({
    cellDataType: scienceColumnTypeToGrid(column.type),
    field: column.id,
    headerName: column.name || `Column ${index + 1}`,
  }));
  const height = Math.min(480, Math.max(220, 52 + preview.rows.length * 32));

  return (
    <div style={{ height }}>
      <AgGridReact<TableRow>
        theme={themeQuartz}
        rowData={rowData}
        columnDefs={columnDefs}
        defaultColDef={{ filter: true, flex: 1, minWidth: 96, resizable: true, sortable: true }}
        pagination={preview.rows.length > 100}
        paginationPageSize={100}
        paginationPageSizeSelector={false}
        suppressCellFocus={false}
      />
    </div>
  );
}

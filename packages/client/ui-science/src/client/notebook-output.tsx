import type * as nbformat from "@jupyterlab/nbformat";
import type { NotebookOutputBlock } from "@swarmx/dsh-science/types";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import css from "./science-workspace.module.css";

const useBrowserLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

function mimeBundle(
  data: Extract<NotebookOutputBlock, { type: "display_data" | "execute_result" }>["data"],
): nbformat.IMimeBundle {
  return Object.fromEntries(data.map((item) => [item.mime, item.data]));
}

/** Adapt the client-safe Science contract into the standard model consumed by JupyterLab. */
export function notebookOutputsToNbformat(
  outputs: readonly NotebookOutputBlock[],
): nbformat.IOutput[] {
  return outputs.map((output) => {
    if (output.type === "stream") {
      return { output_type: "stream", name: output.name, text: output.text };
    }
    if (output.type === "error") {
      return {
        output_type: "error",
        ename: output.name,
        evalue: output.message,
        traceback: [],
      };
    }
    if (output.type === "execute_result") {
      return {
        output_type: "execute_result",
        data: mimeBundle(output.data),
        metadata: {},
        execution_count: null,
      };
    }
    return {
      output_type: "display_data",
      data: mimeBundle(output.data),
      metadata: {},
    };
  });
}

function plainText(outputs: readonly NotebookOutputBlock[]): string {
  return outputs
    .flatMap((output) => {
      if (output.type === "stream") return [output.text];
      if (output.type === "error") return [`${output.name}: ${output.message}`];
      const text = output.data.find((item) => item.mime === "text/plain");
      return text ? [text.data] : [];
    })
    .join("");
}

export function NotebookOutputArea({
  outputs,
}: {
  readonly outputs: readonly NotebookOutputBlock[];
}) {
  const host = useRef<HTMLDivElement>(null);
  const root = useRef<HTMLDivElement>(null);
  const [rendererFailed, setRendererFailed] = useState(false);

  useBrowserLayoutEffect(() => {
    const hostNode = host.current;
    const rootNode = root.current;
    if (!hostNode || !rootNode) return;
    let disposed = false;
    let model: import("@jupyterlab/outputarea").OutputAreaModel | undefined;
    let area: import("@jupyterlab/outputarea").SimplifiedOutputArea | undefined;
    let widget: typeof import("@lumino/widgets").Widget | undefined;
    void (async () => {
      try {
        const [outputarea, rendermime, widgets] = await Promise.all([
          import("@jupyterlab/outputarea"),
          import("@jupyterlab/rendermime"),
          import("@lumino/widgets"),
        ]);
        if (disposed) return;
        model = new outputarea.OutputAreaModel({
          trusted: false,
          values: notebookOutputsToNbformat(outputs),
        });
        const registry = new rendermime.RenderMimeRegistry({
          initialFactories: rendermime.standardRendererFactories,
        });
        area = new outputarea.SimplifiedOutputArea({ model, rendermime: registry });
        widget = widgets.Widget;
        widget.attach(area, hostNode);
        rootNode.dataset.enhanced = "true";
        setRendererFailed(false);
      } catch (error) {
        if (disposed) return;
        console.error("Science Notebook rich output renderer failed", error);
        rootNode.dataset.enhanced = "false";
        setRendererFailed(true);
      }
    })();
    return () => {
      disposed = true;
      delete rootNode.dataset.enhanced;
      if (area?.isAttached) widget?.detach(area);
      area?.dispose();
      model?.dispose();
    };
  }, [outputs]);

  const fallback = plainText(outputs);
  return (
    <div
      ref={root}
      className={css.notebookOutput}
      data-jupyter-output-area="true"
      aria-label="Notebook output"
    >
      <div ref={host} className={css.notebookOutputHost} />
      <pre className={css.notebookOutputFallback}>{fallback || "Notebook output"}</pre>
      {rendererFailed && (
        <small className={css.notebookOutputNotice} role="status">
          Rich output renderer unavailable; showing plain text.
        </small>
      )}
    </div>
  );
}

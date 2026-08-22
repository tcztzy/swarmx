import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { NotebookOutputBlock } from "../../../science/core/src/contracts.js";
import { NotebookOutputArea, notebookOutputsToNbformat } from "../src/client/notebook-output.js";

describe("T30 JupyterLab Notebook output renderer", () => {
  it("V78 preserves ordered stream, MIME, and error outputs in nbformat", () => {
    const outputs: NotebookOutputBlock[] = [
      { type: "stream", name: "stdout", text: "starting\n", truncated: false },
      {
        type: "display_data",
        data: [
          {
            mime: "text/html",
            data: "<strong>42</strong>",
            encoding: "utf8",
            truncated: false,
          },
          {
            mime: "image/png",
            data: "aGVsbG8=",
            encoding: "base64",
            truncated: false,
          },
        ],
      },
      { type: "error", name: "ValueError", message: "invalid value", truncated: false },
    ];

    expect(notebookOutputsToNbformat(outputs)).toEqual([
      { output_type: "stream", name: "stdout", text: "starting\n" },
      {
        output_type: "display_data",
        data: { "text/html": "<strong>42</strong>", "image/png": "aGVsbG8=" },
        metadata: {},
      },
      {
        output_type: "error",
        ename: "ValueError",
        evalue: "invalid value",
        traceback: [],
      },
    ]);
  });

  it("V78 exposes an accessible untrusted Jupyter output host with plain-text fallback", () => {
    const markup = renderToStaticMarkup(
      <NotebookOutputArea
        outputs={[
          {
            type: "execute_result",
            data: [
              { mime: "text/html", data: "<em>rich</em>", encoding: "utf8", truncated: false },
              { mime: "text/plain", data: "rich", encoding: "utf8", truncated: false },
            ],
          },
        ]}
      />,
    );

    expect(markup).toContain('data-jupyter-output-area="true"');
    expect(markup).toContain('aria-label="Notebook output"');
    expect(markup).toContain("rich");
    expect(markup).not.toContain("<em>rich</em>");
  });
});

/** Stable Cordis plugin name. */
export const name = "swarmx-ui-science";

/** The Host half owns only the generic file-link contract used by its browser UI. */
export const inject = ["systemPrompt"];

export function apply(ctx: {
  systemPrompt: {
    section(input: { readonly name: string; readonly order: number; readonly text: string }): void;
  };
}): void {
  ctx.systemPrompt.section({
    name: "ui:deliverable-file-references",
    order: 190,
    text: "When you successfully create or modify files, mention the primary outputs in your final response using standard Markdown links such as `[filename](./workspace-relative/path)`. Prefer this Markdown link over a bare path or Markdown inline code, use the exact workspace-relative file-tool path, and never reveal an absolute host path. A Typst paper link opens the live Science paper workbench.",
  });
}

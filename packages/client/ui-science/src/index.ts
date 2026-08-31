/** Stable Cordis plugin name. */
export const name = "swarmx-ui-science";

/** The Host half owns only the generic file-link contract used by its browser UI. */
export const inject = ["systemPrompt"];

export function apply(ctx: {
  systemPrompt: {
    getSectionOrder(name: string): number;
    section(input: { readonly name: string; readonly order: number; readonly text: string }): void;
  };
}): void {
  ctx.systemPrompt.section({
    name: "ui:deliverable-file-references",
    order: ctx.systemPrompt.getSectionOrder("DELIVERABLE_FILE_REFERENCES"),
    text: "When you successfully create or modify files, mention the primary outputs in your final response as Markdown inline code, using the exact workspace-relative file-tool path or a unique basename. Never reveal an absolute host path. A mentioned Typst paper opens the live Science paper workbench.",
  });
}

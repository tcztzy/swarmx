/** Stable Cordis plugin name for the dsh-science agent-plane contract. */
export const name = "swarmx-science-preset-contract";

/** Science guidance and policy are resolved only through the selected preset. */
export const inject = ["systemPrompt", "tools"];

function unquote(value: string): string {
  return value.length >= 2 && value.at(0) === value.at(-1) && /^["']/u.test(value)
    ? value.slice(1, -1)
    : value;
}

function isAssignment(value: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*=/u.test(value);
}

function isManagedTypstSegment(segment: string): boolean {
  const tokens = segment.trim().split(/\s+/u).filter(Boolean);
  while (tokens[0] !== undefined && isAssignment(tokens[0])) tokens.shift();
  if (tokens[0] === "command" || tokens[0] === "exec") tokens.shift();
  if (tokens[0] === "env") {
    tokens.shift();
    while (tokens[0] !== undefined && (tokens[0].startsWith("-") || isAssignment(tokens[0]))) {
      tokens.shift();
    }
  }
  const executable = unquote(tokens[0] ?? "")
    .replaceAll("\\", "/")
    .split("/")
    .at(-1);
  return executable === "typst" && (tokens[1] === "compile" || tokens[1] === "watch");
}

function isManagedTypstCommand(command: string): boolean {
  return command.split(/(?:&&|\|\||[;\n|&])/u).some(isManagedTypstSegment);
}

/** Install the dsh-science-only prompt sections and managed Typst boundary. */
export function apply(ctx: {
  systemPrompt: {
    getSectionOrder(name: "TOOLS_SDK"): number;
    section(input: { readonly name: string; readonly order: number; readonly text: string }): void;
  };
  tools: {
    guard(
      guard: (execution: {
        readonly name: string;
        readonly arguments: unknown;
      }) => string | undefined,
    ): () => void;
  };
}): () => void {
  const toolsSdkOrder = ctx.systemPrompt.getSectionOrder("TOOLS_SDK");
  ctx.systemPrompt.section({
    name: "science:typst-workflow",
    order: toolsSdkOrder + 100,
    text: "For `.typ` or `.typst`, finish after editing the source and mention its exact workspace-relative path as Markdown inline code. Do not call `typst compile` or `typst watch` through Bash or another tool; dsh-science owns compilation through its managed Host watcher.",
  });
  ctx.systemPrompt.section({
    name: "science:annotations",
    order: toolsSdkOrder + 101,
    text: "Treat each <dsh-annotation>{...}</dsh-annotation> object in user input as structured context. Its type follows the SwarmX annotation union, which contains OpenAI Responses citation/path objects unchanged plus type=comment and type=message_quote targets. For comment.target.type=image_point, call science_query with action=inspect_annotation and the complete comment object before discussing the protected image. For document_text or document_region, use the relative Typst source and rendered revision context to make the requested change; never infer a host path. A message_quote carries exact selected text plus its source Session/message locator and optional user note.",
  });
  ctx.systemPrompt.section({
    name: "science:literature-search",
    order: toolsSdkOrder + 102,
    text: "For scientific publication discovery, use `literature_search` to search the user's running local Zotero library first. It returns citation-ready BibTeX and is distinct from `science_query`, which only reads Science workspace state. Do not substitute a web or cloud literature search unless the user explicitly requests online search.",
  });
  ctx.systemPrompt.section({
    name: "science:resource-addressing",
    order: toolsSdkOrder + 103,
    text: "For local Science state, call `science_query` with `head` first, then use only the needed `get`, `select`, or `neighbors` view and reuse the returned `exactId`. On `RESOURCE_REVISION_MISMATCH`, call `head` again and do not remove the revision guard. Never request a complete workspace or full entity.",
  });
  return ctx.tools.guard((execution) => {
    if (execution.name !== "bash") return undefined;
    if (
      typeof execution.arguments !== "object" ||
      execution.arguments === null ||
      !("command" in execution.arguments) ||
      typeof execution.arguments.command !== "string" ||
      !isManagedTypstCommand(execution.arguments.command)
    ) {
      return undefined;
    }
    return "Typst compilation is managed automatically by dsh-science. Edit the .typ/.typst source and mention its exact workspace-relative path as Markdown inline code; do not run typst compile or typst watch.";
  });
}

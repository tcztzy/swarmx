import type { RuntimeKind } from "./contracts.js";

export function resolveRuntimeSelection(
  args: readonly string[],
  env: Readonly<Record<string, string | undefined>> = process.env,
): RuntimeKind {
  const values: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--runtime") {
      const value = args[index + 1];
      if (value === undefined || value.startsWith("--")) {
        throw new Error("--runtime requires dsh or codex.");
      }
      values.push(value);
      index += 1;
    } else if (argument?.startsWith("--runtime=")) {
      values.push(argument.slice("--runtime=".length));
    }
  }
  if (values.length > 1) throw new Error("--runtime was specified more than once.");
  const selected = values[0] ?? env.SWARMX_RUNTIME ?? "dsh";
  if (selected !== "dsh" && selected !== "codex") {
    throw new Error(`Unknown runtime "${selected}"; expected dsh or codex.`);
  }
  return selected;
}

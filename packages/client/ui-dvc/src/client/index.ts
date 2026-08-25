import type { Context } from "@deepseek-ai/cordis";
import type {} from "@deepseek-ai/dsh-api-gateway/client";
import { DVC_UI_REMOTE } from "../remote.js";

export const inject = ["remote"];

/** Mount the read-only DVC Remote consumed by the shared Version Control presentation. */
export async function apply(ctx: Context): Promise<() => Promise<void>> {
  const disposeRemote = await ctx.remote.$mount(DVC_UI_REMOTE);
  return async () => disposeRemote();
}

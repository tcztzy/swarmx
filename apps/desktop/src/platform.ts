import { createHash } from "node:crypto";
import { mkdir, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, join, resolve } from "node:path";
import { type AgentId, selectedAgent } from "./agent.js";
import type { NativeAgent } from "./agents/types.js";
import { ProductServices, type Workspace } from "./host/product-services.js";
import { type SwarmXHost, startHost } from "./host/server.js";

export interface DesktopPlatform {
  readonly url: string;
  readonly agent: NativeAgent;
  readonly workspaceRoot: string;
  readonly a2aUrl: string;
  issueLaunchUrl(): string;
  dispose(): Promise<void>;
}

export async function startDesktopPlatform(options: {
  readonly workspaceRoot: string;
  readonly rendererRoot: string;
  readonly productHome?: string;
  readonly agentId?: AgentId;
}): Promise<DesktopPlatform> {
  const workspace = await resolveWorkspace(options.workspaceRoot);
  const productHome = resolve(
    options.productHome ?? process.env.SWARMX_HOME ?? join(homedir(), ".swarmx"),
  );
  await mkdir(productHome, { recursive: true });
  const products = await ProductServices.create({ productHome, workspace });
  let host: SwarmXHost | undefined;
  try {
    host = await startHost({ products, rendererRoot: options.rendererRoot, workspace });
    await products.attachAgents(
      host.internalUrl,
      host.internalToken,
      undefined,
      options.agentId ?? selectedAgent(),
    );
    return {
      url: host.issueLaunchUrl(),
      agent: products.rootAgent,
      workspaceRoot: workspace.root,
      a2aUrl: `${host.internalUrl}/a2a/swarm`,
      issueLaunchUrl: () => host?.issueLaunchUrl() ?? failClosed(),
      dispose: () => dispose(host, products),
    };
  } catch (error) {
    await dispose(host, products);
    throw error;
  }
}

async function resolveWorkspace(path: string): Promise<Workspace> {
  const root = await realpath(resolve(path));
  if (!(await stat(root)).isDirectory()) throw new Error("Workspace is not a directory.");
  return {
    root,
    label: basename(root),
    id: createHash("sha256").update(root).digest("hex").slice(0, 12),
  };
}

async function dispose(host: SwarmXHost | undefined, products: ProductServices): Promise<void> {
  const results = await Promise.allSettled([host?.dispose(), products.dispose()]);
  const failure = results.find((result) => result.status === "rejected");
  if (failure?.status === "rejected") throw failure.reason;
}

function failClosed(): never {
  throw new Error("SwarmX Host is unavailable.");
}

import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { type AgentId, selectedAgent } from "./agent.js";
import type { NativeAgent } from "./agents/types.js";
import { ProductServices } from "./host/product-services.js";
import { type SwarmXHost, startHost } from "./host/server.js";
import { ProjectStore, resolveWorkspace } from "./host/workspace-settings.js";

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
  readonly development?: boolean;
}): Promise<DesktopPlatform> {
  const productHome = resolve(
    options.productHome ?? process.env.SWARMX_HOME ?? join(homedir(), ".swarmx"),
  );
  await mkdir(productHome, { recursive: true });
  const catalog = new ProjectStore(productHome);
  const saved = catalog.read();
  const workspace =
    saved.activeId === null
      ? catalog.register(await resolveWorkspace(options.workspaceRoot))
      : catalog.get(saved.activeId);
  if ((await resolveWorkspace(workspace.root)).root !== workspace.root)
    throw new Error("Project directory changed. Add its current directory as a project.");
  const products = await ProductServices.create({ productHome, workspace });
  let host: SwarmXHost | undefined;
  try {
    host = await startHost({
      products,
      rendererRoot: options.rendererRoot,
      workspace,
      development: options.development,
    });
    await products.attachAgents(
      host.internalUrl,
      host.internalToken,
      undefined,
      options.agentId ?? selectedAgent(),
    );
    return {
      url: host.issueLaunchUrl(),
      get agent() {
        return host?.products.rootAgent ?? failClosed();
      },
      get workspaceRoot() {
        return host?.products.options.workspace.root ?? failClosed();
      },
      a2aUrl: `${host.internalUrl}/projects/${workspace.id}/a2a/swarm`,
      issueLaunchUrl: () => host?.issueLaunchUrl() ?? failClosed(),
      dispose: () => dispose(host, products),
    };
  } catch (error) {
    await dispose(host, products);
    throw error;
  }
}

async function dispose(host: SwarmXHost | undefined, products: ProductServices): Promise<void> {
  if (host) await host.dispose();
  else await products.dispose();
}

function failClosed(): never {
  throw new Error("SwarmX Host is unavailable.");
}

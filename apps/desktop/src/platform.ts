import { mkdir } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { type AgentId, selectedAgent } from "./agent.js";
import { acpAgent } from "./host/acp.js";
import { HostOperations } from "./host/operations.js";
import { ProductServices } from "./host/product-services.js";
import { startHost } from "./host/server.js";
import type { ReferenceProvider } from "./reference-provider.js";

export interface DesktopPlatform {
  readonly a2aUrl: string;
  readonly cwd: string;
  readonly operations: HostOperations;
  readonly acp: ReturnType<typeof acpAgent>;
  dispose(): Promise<void>;
}

export async function startDesktopPlatform(options: {
  readonly cwd: string;
  readonly productHome?: string;
  readonly agentId?: AgentId;
  readonly referenceProvider?: ReferenceProvider;
}): Promise<DesktopPlatform> {
  const productHome = resolve(
    options.productHome ?? process.env.SWARMX_HOME ?? join(homedir(), ".swarmx"),
  );
  await mkdir(productHome, { recursive: true });
  const products = await ProductServices.create({
    productHome,
    cwd: options.cwd,
    ...(options.referenceProvider ? { referenceProvider: options.referenceProvider } : {}),
  });
  try {
    const host = await startHost({ products, agentId: options.agentId ?? selectedAgent() });
    return {
      a2aUrl: `${host.origin}/a2a/swarm`,
      cwd: products.options.cwd,
      operations: new HostOperations(host),
      acp: acpAgent(products.rootAgent, products.options.cwd),
      dispose: () => host.dispose(),
    };
  } catch (error) {
    await products.dispose();
    throw error;
  }
}

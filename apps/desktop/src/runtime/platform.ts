import { join } from "node:path";
import { type Harness, startHarness } from "../harness.js";
import { type RuntimeBridge, startRuntimeBridge } from "./bridge.js";
import { type StartCodexRuntimeOptions, startCodexRuntime } from "./codex/index.js";
import type { ConversationRuntime, RuntimeKind } from "./contracts.js";
import { DshConversationRuntime, type DshRuntimeHost } from "./dsh/runtime.js";
import {
  importLegacyProductState,
  resolveLegacyProductHome,
  resolveProductHome,
} from "./product-home.js";
import { ConversationRuntimeRegistry } from "./registry.js";
import { type SwarmRecoveryOwner, startSwarmRecoveryOwner } from "./swarm-recovery-owner.js";
import ConversationRuntimeService, {
  type ConversationRuntimeService as RuntimeService,
} from "./web-plugin.js";
import { WorkspaceAuthority } from "./workspace.js";

export interface DesktopPlatform {
  runtimes: ConversationRuntimeRegistry;
  defaultRuntimeKind: RuntimeKind;
  url: string;
  dispose(): Promise<void>;
}

export interface StartDesktopPlatformOptions {
  runtime: RuntimeKind;
  workspaceRoot: string;
  productHome?: string;
  legacyProductHome?: string;
  codex?: StartCodexRuntimeOptions;
}

export async function startDesktopPlatform(
  options: StartDesktopPlatformOptions,
): Promise<DesktopPlatform> {
  const authority = new WorkspaceAuthority();
  const workspace = authority.mint(options.workspaceRoot);
  const productHome = options.productHome ?? resolveProductHome();
  importLegacyProductState({
    legacyHome: options.legacyProductHome ?? resolveLegacyProductHome(),
    productHome,
  });
  let harness: Harness | undefined;
  let bridge: RuntimeBridge | undefined;
  const adapters: ConversationRuntime[] = [];
  let registry: ConversationRuntimeRegistry | undefined;
  let runtimePlugin: { dispose(): Promise<void> } | undefined;
  let runtimeService: RuntimeService | undefined;
  let swarmRecoveryOwner: SwarmRecoveryOwner | undefined;
  try {
    swarmRecoveryOwner = startSwarmRecoveryOwner(join(productHome, "swarm"));
    harness = await startHarness({ productHome });
    adapters.push(new DshConversationRuntime(harness.ctx as DshRuntimeHost));
    if (options.runtime === "codex") {
      bridge = await startRuntimeBridge(workspace, swarmRecoveryOwner);
      const codex = await startCodexRuntime({
        ...options.codex,
        bridgeToken: bridge.token,
        bridgeUrl: bridge.url,
        productHome,
        scienceConfig: harness.scienceConfig,
        workspace,
      });
      bridge.attach(codex);
      adapters.push(codex);
    }
    registry = new ConversationRuntimeRegistry(adapters, options.runtime);
    runtimePlugin = await harness.ctx.plugin(ConversationRuntimeService, {
      registry,
      workspace,
    });
    runtimeService = harness.ctx.conversationRuntimes;
    return {
      runtimes: registry,
      defaultRuntimeKind: options.runtime,
      url: harness.url,
      dispose: once(async () => {
        await disposeInOrder([
          () => runtimeService?.shutdown(),
          () => runtimePlugin?.dispose(),
          () => bridge?.dispose(),
          () => registry?.dispose(),
          () => harness?.ctx.fiber.dispose(),
          () => swarmRecoveryOwner?.dispose(),
        ]);
      }),
    };
  } catch (startupError) {
    try {
      await disposeInOrder([
        () => runtimeService?.shutdown(),
        () => runtimePlugin?.dispose(),
        () => bridge?.dispose(),
        () =>
          registry === undefined
            ? Promise.allSettled(adapters.reverse().map((runtime) => runtime.dispose())).then(
                (results) => {
                  const failures = results.flatMap((result) =>
                    result.status === "rejected" ? [result.reason] : [],
                  );
                  if (failures.length > 0) {
                    throw new AggregateError(failures, "Conversation runtime cleanup failed.");
                  }
                },
              )
            : registry.dispose(),
        () => harness?.ctx.fiber.dispose(),
        () => swarmRecoveryOwner?.dispose(),
      ]);
    } catch (cleanupError) {
      throw new AggregateError(
        [startupError, cleanupError],
        "Desktop platform startup and cleanup both failed.",
        { cause: startupError },
      );
    }
    throw startupError;
  }
}

function once(operation: () => Promise<void>): () => Promise<void> {
  let result: Promise<void> | undefined;
  return () => (result ??= operation());
}

async function disposeInOrder(
  operations: Array<() => Promise<unknown> | undefined>,
): Promise<void> {
  const failures: unknown[] = [];
  for (const operation of operations) {
    try {
      await operation();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(failures, "Desktop platform cleanup failed.");
  }
}

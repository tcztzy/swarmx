export interface Session {
  readonly sessionId: string;
  readonly title?: string;
  readonly updatedAt?: string;
}

export interface RunOptions {
  readonly model?: string | undefined;
  readonly effort?: string | undefined;
  readonly mode?: string | undefined;
}

export interface ModelCatalog {
  readonly modes?:
    | { readonly id: string; readonly name: string; readonly description?: string | undefined }[]
    | undefined;
  readonly models: {
    readonly id: string;
    readonly name: string;
    readonly description?: string | undefined;
    readonly efforts: { readonly id: string; readonly name: string }[];
    readonly defaultEffort?: string | undefined;
  }[];
  readonly current: RunOptions;
}

/** Native SDK and UI boundary; recursive composition itself uses ACP connections. */
export interface Agent<Observer> {
  readonly name: string;
  readonly capabilities: AgentCapabilities;
  models(sessionId?: string): Promise<ModelCatalog>;
  list(): Promise<Session[]>;
  create(): Promise<string>;
  read(sessionId: string, observer: Observer): Promise<void>;
  start(
    sessionId: string,
    text: string,
    observer: Observer,
    options?: RunOptions,
  ): Promise<PromptResponse>;
  steer(sessionId: string, text: string): Promise<void>;
  interrupt(sessionId: string): Promise<void>;
  dispose(): Promise<void>;
}

/** One ACP relay per upstream connection. The connector owns downstream policy and transport. */
export function createSwarm(
  name: string,
  connectLead: (client: acp.ClientApp) => acp.ClientConnection,
): acp.AgentApp {
  let downstream: acp.ClientConnection;
  const app = acp.agent({ name }).onConnect((upstream) => {
    if (downstream) throw new Error("Create one ACP Swarm app per connection.");
    const client = acp.client({ name });
    client.onNotification(acp.methods.client.session.update, ({ params }) =>
      upstream.client.notify(acp.methods.client.session.update, params),
    );
    client.onNotification(acp.methods.client.elicitation.complete, ({ params }) =>
      upstream.client.notify(acp.methods.client.elicitation.complete, params),
    );
    for (const method of [
      acp.methods.client.session.requestPermission,
      ...Object.values(acp.methods.client.fs),
      ...Object.values(acp.methods.client.terminal),
      acp.methods.client.elicitation.create,
    ])
      client.onRequest(
        method,
        (ctx: acp.ClientRequestContext<acp.ClientRequestParamsByMethod[typeof method]>) =>
          upstream.client.request(method, ctx.params, { cancellationSignal: ctx.signal }),
      );
    downstream = connectLead(client);
    upstream.signal.addEventListener("abort", () => downstream.close(), { once: true });
    downstream.signal.addEventListener("abort", () => upstream.close(), { once: true });
  });
  app.onRequest(acp.methods.agent.initialize, async ({ params, signal }) => {
    const result = await downstream.agent.request(acp.methods.agent.initialize, params, {
      cancellationSignal: signal,
    });
    return {
      ...result,
      agentInfo: { ...result.agentInfo, name, version: result.agentInfo?.version ?? "0.1.0" },
    };
  });
  for (const method of [
    acp.methods.agent.session.new,
    acp.methods.agent.session.list,
    acp.methods.agent.session.load,
    acp.methods.agent.session.resume,
    acp.methods.agent.session.setConfigOption,
    acp.methods.agent.session.setMode,
    acp.methods.agent.session.prompt,
  ] as const)
    app.onRequest(
      method,
      (ctx: acp.AgentRequestContext<acp.AgentRequestParamsByMethod[typeof method]>) =>
        downstream.agent.request(method, ctx.params, { cancellationSignal: ctx.signal }),
    );
  app.onNotification(acp.methods.agent.session.cancel, ({ params }) =>
    downstream.agent.notify(acp.methods.agent.session.cancel, params),
  );
  // Extension payloads are validated by their owning leaf, and are never interpreted by the relay.
  for (const method of ["_swarmx/models", "_swarmx/session/permissions", "_swarmx/session/steer"])
    app.onRequest(method, z.unknown(), ({ params, signal }) =>
      downstream.agent.request(method, params, { cancellationSignal: signal }),
    );
  return app;
}

import type { AgentCapabilities, PromptResponse } from "@agentclientprotocol/sdk";
import * as acp from "@agentclientprotocol/sdk";
import { z } from "zod";

export type { AgentCapabilities, PromptResponse } from "@agentclientprotocol/sdk";

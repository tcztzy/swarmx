import type { InvocationDescriptor } from "@deepseek-ai/dsh-typert-protocol";
import { z } from "zod";
import { swarmUiSnapshotSchema, waitForSwarmChangeRequestSchema } from "./contracts.js";

const sessionIdSchema = z.string().min(1).max(500);

function parameter(
  name: string,
  typeSymbol: string,
  schema: z.ZodType,
): InvocationDescriptor["parameters"][number] {
  return {
    name,
    wire: name,
    source: "json",
    codec: { mode: "strict", typeSymbol, schema },
  };
}

function descriptor(
  method: "uiSnapshot" | "waitUi",
  parameters: InvocationDescriptor["parameters"],
): InvocationDescriptor {
  return {
    id: `@swarmx/dsh-swarm#swarm/${method}`,
    service: "swarm",
    namespace: "swarm",
    method,
    invocation: { kind: "direct" },
    parameters,
    cancellation: { parameter: "signal" },
    result: {
      mode: "strict",
      typeSymbol: "@swarmx/dsh-swarm/contracts#SwarmUiSnapshot",
      schema: swarmUiSnapshotSchema,
    },
  };
}

export const SWARM_INVOCATIONS = Object.freeze([
  descriptor("uiSnapshot", [
    parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
  ]),
  descriptor("waitUi", [
    parameter("sessionId", "@deepseek-ai/dsh-session/types#SessionId", sessionIdSchema),
    parameter(
      "request",
      "@swarmx/dsh-swarm/contracts#WaitForSwarmChangeRequest",
      waitForSwarmChangeRequestSchema,
    ),
  ]),
]);

import type { InvocationDescriptor } from "@deepseek-ai/dsh-typert-protocol";
import { z } from "zod";
import { dvcUiSnapshotSchema } from "./contracts.js";

const sessionIdSchema = z.string().min(1).max(200);

export const DVC_UI_INVOCATIONS = Object.freeze([
  {
    id: "@swarmx/dsh-ui-dvc#dvcUi/snapshot",
    service: "dvcUi",
    namespace: "dvcUi",
    method: "snapshot",
    invocation: { kind: "direct" },
    parameters: [
      {
        name: "sessionId",
        wire: "sessionId",
        source: "json",
        codec: {
          mode: "strict",
          typeSymbol: "@deepseek-ai/dsh-session/types#SessionId",
          schema: sessionIdSchema,
        },
      },
    ],
    cancellation: { parameter: "signal" },
    result: {
      mode: "strict",
      typeSymbol: "@swarmx/dsh-ui-dvc/contracts#DvcUiSnapshot",
      schema: dvcUiSnapshotSchema,
    },
  },
] as const satisfies readonly InvocationDescriptor[]);

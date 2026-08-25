import type { InvocationDescriptor } from "@deepseek-ai/dsh-typert-protocol";
import { z } from "zod";
import { gitUiSnapshotSchema } from "./contracts.js";

const sessionIdSchema = z.string().min(1).max(200);

export const GIT_UI_INVOCATIONS = Object.freeze([
  {
    id: "@swarmx/dsh-ui-git#gitUi/snapshot",
    service: "gitUi",
    namespace: "gitUi",
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
      typeSymbol: "@swarmx/dsh-ui-git/contracts#GitUiSnapshot",
      schema: gitUiSnapshotSchema,
    },
  },
] as const satisfies readonly InvocationDescriptor[]);

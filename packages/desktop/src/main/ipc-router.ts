import type { IpcMainInvokeEvent } from "electron";
import type { z } from "zod";
import type { DesktopInvokeContract, DesktopIpcAuditPolicy } from "../shared/ipc-contracts/base.js";

export interface SemanticAuditReceipt {
  readonly semanticAuditRecorded: boolean;
  readonly recordSemanticAudit: () => void;
}

export class DesktopIpcBoundaryError extends Error {}

export type DesktopAuthorizedIpcHandler = (
  event: IpcMainInvokeEvent,
  receipt: SemanticAuditReceipt,
  ...args: unknown[]
) => unknown;

export interface DesktopIpcRegistrar {
  register<Contract extends DesktopInvokeContract>(
    channel: string,
    contract: Contract,
    handler: (
      event: IpcMainInvokeEvent,
      args: z.output<Contract["args"]>,
      receipt: SemanticAuditReceipt,
    ) => z.input<Contract["result"]> | Promise<z.input<Contract["result"]>>,
  ): void;
}

export function createDesktopIpcRegistrar(options: {
  registerAuthorized: (channel: string, listener: DesktopAuthorizedIpcHandler) => void;
  auditPolicy: (channel: string) => DesktopIpcAuditPolicy;
}): DesktopIpcRegistrar {
  function register<Contract extends DesktopInvokeContract>(
    channel: string,
    contract: Contract,
    handler: (
      event: IpcMainInvokeEvent,
      args: z.output<Contract["args"]>,
      receipt: SemanticAuditReceipt,
    ) => z.input<Contract["result"]> | Promise<z.input<Contract["result"]>>,
  ): void {
    if (options.auditPolicy(channel) !== contract.audit) {
      throw new Error(`Desktop IPC channel ${channel} has inconsistent audit policy.`);
    }
    options.registerAuthorized(channel, (event, receipt, ...rawArgs) => {
      const args = parseBoundary(contract.args, rawArgs, channel, "arguments") as z.output<
        Contract["args"]
      >;
      const result = handler(event, args, receipt);
      return isPromiseLike(result)
        ? Promise.resolve(result).then((value) =>
            parseBoundary(contract.result, value, channel, "result"),
          )
        : parseBoundary(contract.result, result, channel, "result");
    });
  }

  return { register };
}

export function createSemanticAuditReceipt(): SemanticAuditReceipt {
  const receipt = {
    semanticAuditRecorded: false,
    recordSemanticAudit: () => {
      receipt.semanticAuditRecorded = true;
    },
  };
  return receipt;
}

function parseBoundary<Schema extends z.ZodType>(
  schema: Schema,
  value: unknown,
  channel: string,
  boundary: string,
): z.output<Schema> {
  const parsed = schema.safeParse(value);
  if (parsed.success) return parsed.data;
  const path = parsed.error.issues[0]?.path.join(".");
  throw new DesktopIpcBoundaryError(
    `Desktop IPC ${channel} ${boundary} failed validation${path ? ` at ${path}` : ""}.`,
  );
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    (typeof value === "object" || typeof value === "function") &&
    value !== null &&
    "then" in value &&
    typeof value.then === "function"
  );
}

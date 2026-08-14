import type { z } from "zod";

export type DesktopIpcAuditPolicy = "intent_outcome" | "failure_only" | "semantic_only";

export interface DesktopInvokeContract<
  Arguments extends z.ZodTuple = z.ZodTuple,
  Result extends z.ZodType = z.ZodType,
> {
  kind: "invoke";
  args: Arguments;
  result: Result;
  audit: DesktopIpcAuditPolicy;
}

export interface DesktopEventContract<Payload extends z.ZodType = z.ZodType> {
  kind: "event";
  payload: Payload;
}

export type DesktopInvokeContracts = Readonly<Record<string, DesktopInvokeContract>>;
export type DesktopEventContracts = Readonly<Record<string, DesktopEventContract>>;

export function composeContractMaps<Contract>(
  ...features: readonly Readonly<Record<string, Contract>>[]
): Readonly<Record<string, Contract>> {
  const contracts: Record<string, Contract> = {};
  for (const feature of features) {
    for (const [channel, contract] of Object.entries(feature)) {
      if (channel in contracts) throw new Error(`Desktop IPC channel ${channel} is duplicated.`);
      contracts[channel] = contract;
    }
  }
  return Object.freeze(contracts);
}

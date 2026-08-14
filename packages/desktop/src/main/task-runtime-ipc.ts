import type { TaskSupervisorCommand, TaskSupervisorResponse } from "@swarmx/core";
import { TaskRuntimeInvokeContracts } from "../shared/ipc-contracts/task-runtime.js";
import type { DesktopIpcRegistrar } from "./ipc-router.js";

type RendererTaskCommand = Extract<
  TaskSupervisorCommand,
  { operation: "list" | "cancel" | "decide" }
>;
type TaskSupervisorSuccessResponse = Exclude<TaskSupervisorResponse, { ok: false }>;

export interface TaskRuntimeIpcSupervisor {
  request(command: RendererTaskCommand): Promise<TaskSupervisorSuccessResponse>;
}

export function registerTaskRuntimeIpc(
  registrar: DesktopIpcRegistrar,
  supervisor: TaskRuntimeIpcSupervisor,
): void {
  registrar.register(
    "taskRuntime:list",
    TaskRuntimeInvokeContracts["taskRuntime:list"],
    async () => {
      const response = await supervisor.request({ operation: "list" });
      if (response.operation !== "list") throw unexpectedResponse("list", response.operation);
      return response;
    },
  );
  registrar.register(
    "taskRuntime:cancel",
    TaskRuntimeInvokeContracts["taskRuntime:cancel"],
    async (_event, [input]) => {
      const response = await supervisor.request({
        operation: "cancel",
        workItemId: input.workItemId,
        ...(input.reason === undefined ? {} : { reason: input.reason }),
      });
      if (response.operation !== "cancel") throw unexpectedResponse("cancel", response.operation);
      return response;
    },
  );
  registrar.register(
    "taskRuntime:decide",
    TaskRuntimeInvokeContracts["taskRuntime:decide"],
    async (_event, [input]) => {
      const response = await supervisor.request({
        operation: "decide",
        approvalId: input.approvalId,
        status: input.status,
        decidedBy: input.decidedBy,
        ...(input.reason === undefined ? {} : { reason: input.reason }),
        ...(input.response === undefined ? {} : { response: input.response }),
      });
      if (response.operation !== "decide") throw unexpectedResponse("decide", response.operation);
      return response;
    },
  );
}

function unexpectedResponse(expected: string, actual: string): Error {
  return new Error(`Task Runtime received an unexpected ${actual} response for ${expected}.`);
}
